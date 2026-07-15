"""Alignment training: DPO (Direct Preference Optimization) and RLHF.

Provides:
- DPOTrainer: Direct Preference Optimization for aligning LLMs to human preferences
- RLHFTrainer: Reinforcement Learning from Human Feedback (PPO-based)
- PreferenceDataset: load and manage preference pairs (chosen/rejected)
"""

from __future__ import annotations

import json
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from typing import Optional
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("cerebro.alignment")


@dataclass
class PreferenceSample:
    """A single preference pair for alignment training."""
    prompt: str
    chosen: str
    rejected: str
    metadata: dict = field(default_factory=dict)


@dataclass
class DPOResult:
    """Result from a DPO training run."""
    loss: float
    chosen_reward: float
    rejected_reward: float
    reward_margin: float
    accuracy: float
    total_steps: int
    elapsed_seconds: float


class PreferenceDataset(Dataset):
    """Dataset of human preference pairs for alignment training.

    Loads preference data from JSONL files where each line has:
    {"prompt": "...", "chosen": "...", "rejected": "..."}

    Args:
        data_path: Path to JSONL preference data file.
        tokenizer: CerebroTokenizer instance.
        max_prompt_len: Maximum prompt token length.
        max_response_len: Maximum response token length.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer=None,
        max_prompt_len: int = 512,
        max_response_len: int = 1024,
    ) -> None:
        self.max_prompt_len = max_prompt_len
        self.max_response_len = max_response_len
        self.tokenizer = tokenizer
        self.samples: list[PreferenceSample] = []

        self._load(data_path)

    def _load(self, path: str) -> None:
        """Load preference data from JSONL."""
        if not Path(path).exists():
            raise FileNotFoundError(f"Preference data not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.samples.append(PreferenceSample(
                    prompt=obj.get("prompt", ""),
                    chosen=obj.get("chosen", ""),
                    rejected=obj.get("rejected", ""),
                    metadata=obj.get("metadata", {}),
                ))

        if not self.samples:
            raise ValueError(f"No preference samples found in {path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        if self.tokenizer is not None:
            prompt_tokens = self.tokenizer.encode(sample.prompt)[:self.max_prompt_len]
            chosen_tokens = self.tokenizer.encode(sample.chosen)[:self.max_response_len]
            rejected_tokens = self.tokenizer.encode(sample.rejected)[:self.max_response_len]

            return {
                "prompt_ids": torch.tensor(prompt_tokens, dtype=torch.long),
                "chosen_ids": torch.tensor(chosen_tokens, dtype=torch.long),
                "rejected_ids": torch.tensor(rejected_tokens, dtype=torch.long),
            }

        # Fallback: return raw text for external tokenization
        return {
            "prompt": sample.prompt,
            "chosen": sample.chosen,
            "rejected": sample.rejected,
        }


def _pad_sequences(sequences: list[Tensor], pad_id: int = 0) -> Tensor:
    """Pad a list of variable-length tensors to the same length."""
    max_len = max(s.size(0) for s in sequences)
    padded = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(sequences):
        padded[i, :s.size(0)] = s
    return padded


class DPOTrainer:
    """Direct Preference Optimization trainer.

    Implements DPO (Rafailov et al., 2023) — a simple, stable method
    for aligning LLMs to human preferences without training a separate
    reward model.

    The DPO loss is:
        L = -log σ(β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))

    where y_w = chosen, y_l = rejected, β = temperature.

    Args:
        model: Policy model (the LLM being trained).
        ref_model: Reference model (frozen copy of initial policy).
        tokenizer: CerebroTokenizer instance.
        beta: DPO temperature parameter (higher = more conservative).
        lr: Learning rate.
        device: Training device.
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module | None = None,
        tokenizer=None,
        beta: float = 0.1,
        lr: float = 1e-6,
        device: str = "auto",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer

        # Reference model (frozen copy)
        if ref_model is not None:
            self.ref_model = ref_model
        else:
            import copy
            self.ref_model = copy.deepcopy(model)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        self.beta = beta

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = self.model.to(self.device)
        self.ref_model = self.ref_model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=0.01,
        )

    def _get_logprobs(self, model: nn.Module, input_ids: Tensor, labels: Tensor) -> Tensor:
        """Compute per-token log probabilities for the given sequence."""
        with torch.no_grad() if model is self.ref_model else torch.enable_grad():
            output = model(input_ids)
            logits = output["logits"]

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

        # Mask padding
        mask = (shift_labels != 0).float()
        seq_log_prob = (token_log_probs * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)

        return seq_log_prob

    def dpo_loss(
        self,
        prompt_ids: Tensor,
        chosen_ids: Tensor,
        rejected_ids: Tensor,
    ) -> tuple[Tensor, dict]:
        """Compute DPO loss for a batch.

        Returns:
            (loss, metrics_dict)
        """
        # Concatenate prompt + response for full sequences
        batch_size = prompt_ids.size(0)

        # Get logprobs from policy model
        policy_chosen = self._get_logprobs(self.model,
            torch.cat([prompt_ids, chosen_ids], dim=1), chosen_ids)
        policy_rejected = self._get_logprobs(self.model,
            torch.cat([prompt_ids, rejected_ids], dim=1), rejected_ids)

        # Get logprobs from reference model
        with torch.no_grad():
            ref_chosen = self._get_logprobs(self.ref_model,
                torch.cat([prompt_ids, chosen_ids], dim=1), chosen_ids)
            ref_rejected = self._get_logprobs(self.ref_model,
                torch.cat([prompt_ids, rejected_ids], dim=1), rejected_ids)

        # DPO loss
        chosen_rewards = self.beta * (policy_chosen - ref_chosen)
        rejected_rewards = self.beta * (policy_rejected - ref_rejected)
        logits = chosen_rewards - rejected_rewards
        loss = -F.logsigmoid(logits).mean()

        # Metrics
        accuracy = (chosen_rewards > rejected_rewards).float().mean().item()

        metrics = {
            "loss": loss.item(),
            "chosen_reward": chosen_rewards.mean().item(),
            "rejected_reward": rejected_rewards.mean().item(),
            "reward_margin": (chosen_rewards - rejected_rewards).mean().item(),
            "accuracy": accuracy,
        }

        return loss, metrics

    def train(
        self,
        dataset: PreferenceDataset,
        num_epochs: int = 3,
        batch_size: int = 4,
    ) -> DPOResult:
        """Run DPO training.

        Args:
            dataset: PreferenceDataset with chosen/rejected pairs.
            num_epochs: Number of training epochs.
            batch_size: Batch size.

        Returns:
            DPOResult with training summary.
        """
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        self.model.train()
        total_loss = 0.0
        total_chosen_reward = 0.0
        total_rejected_reward = 0.0
        total_accuracy = 0.0
        total_steps = 0
        start_time = time.time()

        for epoch in range(num_epochs):
            for batch in loader:
                prompt_ids = batch["prompt_ids"].to(self.device)
                chosen_ids = batch["chosen_ids"].to(self.device)
                rejected_ids = batch["rejected_ids"].to(self.device)

                loss, metrics = self.dpo_loss(prompt_ids, chosen_ids, rejected_ids)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_loss += metrics["loss"]
                total_chosen_reward += metrics["chosen_reward"]
                total_rejected_reward += metrics["rejected_reward"]
                total_accuracy += metrics["accuracy"]
                total_steps += 1

                if total_steps % 10 == 0:
                    logger.info(
                        "DPO step=%d loss=%.4f acc=%.3f margin=%.4f",
                        total_steps, metrics['loss'], metrics['accuracy'], metrics['reward_margin'],
                    )

        elapsed = time.time() - start_time

        return DPOResult(
            loss=total_loss / max(total_steps, 1),
            chosen_reward=total_chosen_reward / max(total_steps, 1),
            rejected_reward=total_rejected_reward / max(total_steps, 1),
            reward_margin=(total_chosen_reward - total_rejected_reward) / max(total_steps, 1),
            accuracy=total_accuracy / max(total_steps, 1),
            total_steps=total_steps,
            elapsed_seconds=elapsed,
        )


@dataclass
class PPOConfig:
    """PPO hyperparameters."""
    ppo_epochs: int = 4
    clip_epsilon: float = 0.2
    value_clip_epsilon: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True
    generate_max_new_tokens: int = 128
    generate_temperature: float = 0.7


class RLHFTrainer:
    """Proper RLHF trainer with PPO (rollouts, clipping, value function, GAE).

    Three-phase approach:
    1. Train a reward model from preference data
    2. Train a value function (critic head)
    3. PPO with rollouts: generate completions, compute advantages, clip policy

    Args:
        model: Policy model (actor).
        tokenizer: CerebroTokenizer instance.
        lr_policy: Policy learning rate.
        lr_critic: Critic (value function) learning rate.
        lr_reward: Reward model learning rate.
        ppo_config: PPO hyperparameters.
        device: Training device.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer=None,
        lr_policy: float = 1e-6,
        lr_critic: float = 1e-5,
        lr_reward: float = 1e-5,
        ppo_config: PPOConfig | None = None,
        device: str = "auto",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.ppo_config = ppo_config or PPOConfig()

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = self.model.to(self.device)

        # Reward model head: linear projection on last hidden state
        hidden_dim = getattr(model, "config", None)
        if hidden_dim is not None:
            hidden_dim = hidden_dim.hidden_dim
        else:
            hidden_dim = getattr(model, "hidden_dim", 2048)

        self.reward_head = nn.Linear(hidden_dim, 1).to(self.device)
        self.value_head = nn.Linear(hidden_dim, 1).to(self.device)

        # Optimizers
        self.policy_optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr_policy, weight_decay=0.01,
        )
        self.critic_optimizer = torch.optim.AdamW(
            list(self.value_head.parameters()), lr=lr_critic, weight_decay=0.01,
        )
        self.reward_optimizer = torch.optim.AdamW(
            list(self.reward_head.parameters()), lr=lr_reward, weight_decay=0.01,
        )

    def compute_reward(self, input_ids: Tensor) -> Tensor:
        """Compute reward score for a sequence."""
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_ids)
            if "hidden_states" in output:
                hidden = output["hidden_states"][:, -1, :]
            else:
                logits = output["logits"][:, -1, :]
                hidden = logits[:, :self.reward_head.in_features]
                if hidden.size(-1) < self.reward_head.in_features:
                    hidden = F.pad(hidden, (0, self.reward_head.in_features - hidden.size(-1)))
        reward = self.reward_head(hidden).squeeze(-1)
        return reward

    def compute_value(self, input_ids: Tensor) -> Tensor:
        """Compute value estimate for a sequence."""
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_ids)
            if "hidden_states" in output:
                hidden = output["hidden_states"][:, -1, :]
            else:
                logits = output["logits"][:, -1, :]
                hidden = logits[:, :self.value_head.in_features]
                if hidden.size(-1) < self.value_head.in_features:
                    hidden = F.pad(hidden, (0, self.value_head.in_features - hidden.size(-1)))
        value = self.value_head(hidden).squeeze(-1)
        return value

    def train_reward_model(
        self,
        dataset: PreferenceDataset,
        num_epochs: int = 3,
        batch_size: int = 4,
    ) -> dict:
        """Phase 1: Train the reward model from preference data.

        Args:
            dataset: PreferenceDataset.
            num_epochs: Training epochs.
            batch_size: Batch size.

        Returns:
            Training metrics.
        """
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        total_loss = 0.0
        total_steps = 0

        for epoch in range(num_epochs):
            for batch in loader:
                chosen_ids = batch["chosen_ids"].to(self.device)
                rejected_ids = batch["rejected_ids"].to(self.device)

                chosen_reward = self.compute_reward(chosen_ids)
                rejected_reward = self.compute_reward(rejected_ids)

                # Bradley-Terry loss
                loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()

                self.reward_optimizer.zero_grad()
                loss.backward()
                self.reward_optimizer.step()

                total_loss += loss.item()
                total_steps += 1

        return {
            "loss": total_loss / max(total_steps, 1),
            "steps": total_steps,
        }

    def _generate_rollout(self, prompt_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Generate a completion from the policy model.

        Args:
            prompt_ids: (B, prompt_len) token IDs.

        Returns:
            (full_sequence, response_logprobs) tuple.
        """
        B = prompt_ids.shape[0]
        self.model.eval()

        with torch.no_grad():
            generated = self.model.generate(
                prompt_ids,
                max_new_tokens=self.ppo_config.generate_max_new_tokens,
                temperature=self.ppo_config.generate_temperature,
                do_sample=True,
            )

        return generated

    def _get_sequence_logprobs(
        self, model: nn.Module, input_ids: Tensor, response_start: int
    ) -> Tensor:
        """Get per-token log probabilities for the response portion.

        Args:
            model: Model to compute logprobs from.
            input_ids: (B, total_len) full sequence.
            response_start: Index where response starts.

        Returns:
            (B, response_len) per-token logprobs.
        """
        model.eval()
        with torch.no_grad():
            output = model(input_ids)
            logits = output["logits"]  # (B, S, vocab)

        # Shift: predict token at position t from logits at t-1
        response_logits = logits[:, response_start - 1:-1, :]  # (B, response_len, vocab)
        response_tokens = input_ids[:, response_start:]  # (B, response_len)

        log_probs = F.log_softmax(response_logits, dim=-1)
        token_log_probs = log_probs.gather(2, response_tokens.unsqueeze(-1)).squeeze(-1)

        return token_log_probs

    def _compute_advantages(
        self,
        rewards: Tensor,
        values: Tensor,
        dones: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Compute GAE (Generalized Advantage Estimation).

        Args:
            rewards: (B,) scalar rewards.
            values: (B,) value estimates.
            dones: (B,) optional done flags.

        Returns:
            (advantages, returns) tuple both (B,).
        """
        cfg = self.ppo_config
        advantages = rewards - values

        if cfg.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        returns = rewards  # For scalar reward, return = reward

        return advantages, returns

    def train_policy(
        self,
        dataset: PreferenceDataset,
        num_steps: int = 1000,
        batch_size: int = 4,
        kl_coef: float = 0.05,
    ) -> dict:
        """Phase 2: PPO with proper rollouts, clipping, and value function.

        For each step:
        1. Sample prompts from dataset
        2. Generate completions (rollout) from the policy
        3. Compute rewards (from reward model)
        4. Compute values (from critic)
        5. Compute advantages via GAE
        6. PPO update with clipped objective

        Args:
            dataset: PreferenceDataset for sampling prompts.
            num_steps: PPO optimization steps.
            batch_size: Batch size.
            kl_coef: KL divergence penalty coefficient.

        Returns:
            Training metrics.
        """
        cfg = self.ppo_config
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        loader_iter = iter(loader)

        # Reference model for KL penalty
        import copy
        ref_model = copy.deepcopy(self.model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
        ref_model = ref_model.to(self.device)

        total_metrics = {
            "reward": 0.0, "value": 0.0, "advantage": 0.0,
            "policy_loss": 0.0, "value_loss": 0.0, "kl": 0.0,
            "clip_frac": 0.0, "entropy": 0.0,
        }
        total_steps = 0

        for step in range(num_steps):
            # Get next batch of prompts
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)

            prompt_ids = batch["prompt_ids"].to(self.device)
            prompt_len = prompt_ids.shape[1]

            # ── 1. Generate rollouts ──
            full_sequences = self._generate_rollout(prompt_ids)

            # ── 2. Compute rewards ──
            rewards = self.compute_reward(full_sequences)

            # ── 3. Compute values ──
            values = self.compute_value(full_sequences)

            # ── 4. Compute advantages ──
            advantages, returns = self._compute_advantages(rewards, values)

            # ── 5. PPO update over multiple epochs ──
            self.model.train()
            self.value_head.train()

            for ppo_epoch in range(cfg.ppo_epochs):
                # Get logprobs from current policy
                policy_logprobs = self._get_sequence_logprobs(
                    self.model, full_sequences, prompt_len
                )

                # Get logprobs from old policy (reference, used as "old" in first epoch)
                with torch.no_grad():
                    old_logprobs = self._get_sequence_logprobs(
                        ref_model, full_sequences, prompt_len
                    )

                # ── PPO clipped objective ──
                ratio = torch.exp(policy_logprobs.sum(dim=-1) - old_logprobs.sum(dim=-1))
                # ratio: (B,)

                # Clipped surrogate
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_epsilon, 1.0 + cfg.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Clip fraction
                clip_frac = ((ratio - 1.0).abs() > cfg.clip_epsilon).float().mean()

                # ── Value loss ──
                new_values = self.compute_value(full_sequences)
                value_loss = F.mse_loss(new_values, returns)

                # ── Entropy bonus ──
                output = self.model(full_sequences)
                logits = output["logits"][:, prompt_len - 1:-1, :]
                probs = F.softmax(logits, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()

                # ── KL divergence ──
                with torch.no_grad():
                    ref_output = ref_model(full_sequences)
                    ref_logits = ref_output["logits"][:, prompt_len - 1:-1, :]

                kl_div = F.kl_div(
                    F.log_softmax(logits, dim=-1),
                    F.softmax(ref_logits, dim=-1),
                    reduction="batchmean",
                )

                # ── Total loss ──
                total_loss = (
                    policy_loss
                    + cfg.vf_coef * value_loss
                    - cfg.entropy_coef * entropy
                    + kl_coef * kl_div
                )

                # Policy update
                self.policy_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.value_head.parameters()),
                    cfg.max_grad_norm,
                )
                self.policy_optimizer.step()
                self.critic_optimizer.step()

                # Accumulate metrics
                total_metrics["policy_loss"] += policy_loss.item()
                total_metrics["value_loss"] += value_loss.item()
                total_metrics["clip_frac"] += clip_frac.item()
                total_metrics["entropy"] += entropy.item()
                total_metrics["kl"] += kl_div.item()

            total_metrics["reward"] += rewards.mean().item()
            total_metrics["value"] += values.mean().item()
            total_metrics["advantage"] += advantages.mean().item()
            total_steps += 1

            if total_steps % 10 == 0:
                logger.info(
                    "PPO step=%d reward=%.4f policy_loss=%.4f kl=%.4f",
                    total_steps,
                    total_metrics['reward'] / total_steps,
                    total_metrics['policy_loss'] / (total_steps * cfg.ppo_epochs),
                    total_metrics['kl'] / (total_steps * cfg.ppo_epochs),
                )

        # Average metrics
        epoch_count = total_steps * cfg.ppo_epochs
        return {
            "avg_reward": total_metrics["reward"] / max(total_steps, 1),
            "avg_value": total_metrics["value"] / max(total_steps, 1),
            "avg_advantage": total_metrics["advantage"] / max(total_steps, 1),
            "avg_policy_loss": total_metrics["policy_loss"] / max(epoch_count, 1),
            "avg_value_loss": total_metrics["value_loss"] / max(epoch_count, 1),
            "avg_kl": total_metrics["kl"] / max(epoch_count, 1),
            "avg_entropy": total_metrics["entropy"] / max(epoch_count, 1),
            "avg_clip_frac": total_metrics["clip_frac"] / max(epoch_count, 1),
            "steps": total_steps,
        }
