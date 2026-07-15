"""Cerebro Inference Engine with KV-cache.

Efficient autoregressive generation using:
- Key-Value cache to avoid recomputing past tokens
- Prefill + decode phases
- Batch inference support
"""

from __future__ import annotations

import time
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from cerebro.config import CerebroConfig
from cerebro.model.cerebro_model import Cerebro
from cerebro.model.attention import KVCache
from cerebro.model.recursion import SelfVerificationModule
from cerebro.inference.sampler import Sampler


class CerebroInferenceEngine:
    """High-performance inference engine for Cerebro models.

    Features:
    - KV-cache for O(1) per-step attention (delete_decode)
    - Prefill phase for prompt processing
    - Decode phase for autoregressive generation
    - Batch inference support
    - Recursion controller integration

    Args:
        model: Cerebro model instance.
        config: Model configuration.
        device: Target device (cuda/cpu).
    """

    def __init__(
        self,
        model: Cerebro,
        config: CerebroConfig,
        device: str = "auto",
    ) -> None:
        self.config = config
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.model.eval()

        # KV-cache: one per layer
        self.kv_caches: list[KVCache] = [KVCache() for _ in range(config.num_layers)]

        # Sampler
        self.sampler = Sampler()

    def reset_cache(self) -> None:
        """Clear all KV-caches."""
        for cache in self.kv_caches:
            cache.reset()

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model weights from a checkpoint.

        Supports both safetensors and PyTorch formats.

        Args:
            checkpoint_path: Path to checkpoint file or directory.
        """
        import os
        from pathlib import Path

        path = Path(checkpoint_path)

        if path.is_dir():
            safetensors_path = path / "model.safetensors"
            pt_path = path / "model.pt"
        else:
            safetensors_path = path if path.suffix == ".safetensors" else None
            pt_path = path if path.suffix == ".pt" else None

        if safetensors_path and safetensors_path.exists():
            from safetensors.torch import load_file
            state_dict = load_file(str(safetensors_path))
            self.model.load_state_dict(state_dict)
        elif pt_path and pt_path.exists():
            state_dict = torch.load(str(pt_path), map_location="cpu", weights_only=True)
            self.model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def prefill(self, input_ids: Tensor) -> Tensor:
        """Process the prompt (prefill phase).

        Runs a full forward pass on the prompt to populate the KV-cache.

        Args:
            input_ids: (B, prompt_len) prompt token IDs.

        Returns:
            (B, vocab_size) logits for the next token.
        """
        input_ids = input_ids.to(self.device)
        output = self.model(input_ids, kv_caches=self.kv_caches)
        next_logits = output["logits"][:, -1, :]  # (B, vocab_size)
        return next_logits

    @torch.no_grad()
    def decode_step(self, token_ids: Tensor) -> Tensor:
        """Single decode step: process one token using KV-cache.

        Only the new token is passed through the model. KV-cache
        stores all previous K,V tensors, making this O(1) per step.

        Args:
            token_ids: (B, 1) single new token.

        Returns:
            (B, vocab_size) logits for the next token.
        """
        token_ids = token_ids.to(self.device)
        output = self.model(token_ids, kv_caches=self.kv_caches)
        return output["logits"][:, -1, :]

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        eos_token_id: int | None = None,
    ) -> Tensor:
        """Generate text autoregressively with KV-cache.

        Args:
            input_ids: (B, prompt_len) prompt token IDs.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling top-p.
            top_k: Top-k sampling cutoff.
            repetition_penalty: Repetition penalty factor.
            do_sample: If False, use greedy decoding.
            eos_token_id: End-of-sequence token ID.

        Returns:
            (B, prompt_len + generated_len) complete sequence.
        """
        self.sampler.temperature = temperature
        self.sampler.top_p = top_p
        self.sampler.top_k = top_k
        self.sampler.repetition_penalty = repetition_penalty

        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id

        self.reset_cache()
        generated = input_ids.clone().to(self.device)
        B = input_ids.shape[0]

        all_tokens: list[list[int]] = [
            generated[i].tolist() for i in range(B)
        ]

        self.model.recursion_controller.reset()
        start_time = time.time()

        # Prefill: process the full prompt through all layers
        next_logits = self.prefill(input_ids)

        for step in range(max_new_tokens):
            next_tokens = self.sampler.sample_batch(
                next_logits.clone(),
                generated_tokens=all_tokens if step > 0 else None,
                do_sample=do_sample,
            )  # (B, 1)

            generated = torch.cat([generated, next_tokens], dim=1)

            for i in range(B):
                all_tokens[i].append(next_tokens[i, 0].item())

            if (next_tokens.squeeze(-1) == eos_token_id).all():
                break

            # Decode step: only process the new token with KV-cache
            next_logits = self.decode_step(next_tokens)

        elapsed = time.time() - start_time
        tokens_generated = generated.shape[1] - input_ids.shape[1]
        self._last_gen_speed = tokens_generated / elapsed if tokens_generated > 0 else 0.0

        return generated

    @torch.no_grad()
    def generate_with_recursion(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
    ) -> Tensor:
        """Generate with bounded recursion controller.

        When the model's confidence is low, the recursion controller
        may trigger additional refinement passes.

        Args:
            Same as generate().

        Returns:
            (B, prompt_len + generated_len) complete sequence.
        """
        self.model.recursion_controller.reset()
        generated = input_ids.clone().to(self.device)
        B = input_ids.shape[0]
        eos_token_id = self.config.eos_token_id

        self.sampler.temperature = temperature
        self.sampler.top_p = top_p
        self.sampler.top_k = top_k
        self.sampler.repetition_penalty = repetition_penalty

        self.reset_cache()
        all_tokens: list[list[int]] = [generated[i].tolist() for i in range(B)]

        next_logits = self.prefill(input_ids)

        for step in range(max_new_tokens):
            confidence = SelfVerificationModule.compute_confidence(next_logits[0])
            step_entropy = -torch.log(torch.tensor(max(confidence, 1e-8))).item()
            can_recurse = self.model.recursion_controller.can_recurse(step_entropy)

            if can_recurse and self.model.recursion_controller.should_refine(confidence):
                refine_logits = self.decode_step(generated[:, -1:])
                refined_confidence = SelfVerificationModule.compute_confidence(refine_logits[0])
                if refined_confidence > confidence:
                    next_logits = refine_logits

            next_tokens = self.sampler.sample_batch(
                next_logits.clone(),
                generated_tokens=all_tokens if step > 0 else None,
                do_sample=do_sample,
            )

            generated = torch.cat([generated, next_tokens], dim=1)
            for i in range(B):
                all_tokens[i].append(next_tokens[i, 0].item())

            if (next_tokens.squeeze(-1) == eos_token_id).all():
                break

            next_logits = self.decode_step(next_tokens)

        return generated

    def benchmark(
        self,
        seq_len: int = 512,
        num_warmup: int = 5,
        num_runs: int = 20,
    ) -> dict[str, float]:
        """Benchmark inference speed.

        Args:
            seq_len: Input sequence length for benchmarking.
            num_warmup: Warmup iterations.
            num_runs: Timed iterations.

        Returns:
            dict with benchmark metrics.
        """
        import torch

        # Create dummy input
        dummy = torch.randint(3, self.config.vocab_size, (1, seq_len)).to(self.device)

        # Warmup
        for _ in range(num_warmup):
            self.model(dummy)
            if self.device.type == "cuda":
                torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            self.model(dummy)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        times_tensor = torch.tensor(times)
        return {
            "mean_latency_ms": times_tensor.mean().item() * 1000,
            "median_latency_ms": times_tensor.median().item() * 1000,
            "p95_latency_ms": torch.quantile(times_tensor, 0.95).item() * 1000,
            "throughput_tok_s": seq_len / times_tensor.mean().item(),
            "seq_len": seq_len,
            "num_params": self.model.num_parameters(),
        }

    # ─── Speculative Decoding ────────────────────────────────────────

    @torch.no_grad()
    def generate_with_speculation(
        self,
        input_ids: Tensor,
        draft_model: nn.Module | None = None,
        max_new_tokens: int = 256,
        num_speculative_tokens: int = 5,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        eos_token_id: int | None = None,
    ) -> Tensor:
        """Generate text with speculative decoding for 2-3x speedup.

        Speculative decoding uses a smaller draft model to propose K tokens
        quickly, then the main model verifies them in a single forward pass.
        Accepted tokens are kept; rejected ones are resampled.

        Algorithm:
        1. Draft model generates K tokens (fast, auto-regressive)
        2. Main model runs one forward pass on prompt + K draft tokens
        3. Compare draft tokens to main model predictions (rejection sampling)
        4. Accept matching tokens, resample from main model at first mismatch
        5. Repeat from step 1

        Args:
            input_ids: (B, prompt_len) prompt token IDs.
            draft_model: Smaller model for fast drafting (if None, uses self as draft).
            max_new_tokens: Maximum tokens to generate.
            num_speculative_tokens: K — number of tokens to draft per step.
            temperature: Sampling temperature.
            top_p: Nucleus sampling top-p.
            top_k: Top-k sampling cutoff.
            repetition_penalty: Repetition penalty factor.
            do_sample: If False, use greedy decoding.
            eos_token_id: End-of-sequence token ID.

        Returns:
            (B, prompt_len + generated_len) complete sequence.
        """
        self.sampler.temperature = temperature
        self.sampler.top_p = top_p
        self.sampler.top_k = top_k
        self.sampler.repetition_penalty = repetition_penalty

        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id

        # Use main model as draft if no separate draft model
        if draft_model is None:
            draft_model = self.model

        self.reset_cache()
        generated = input_ids.clone().to(self.device)
        B = input_ids.shape[0]

        all_tokens: list[list[int]] = [
            generated[i].tolist() for i in range(B)
        ]

        self.model.recursion_controller.reset()
        start_time = time.time()

        total_accepted = 0
        total_drafted = 0

        # Prefill
        next_logits = self.prefill(input_ids)

        while generated.shape[1] - input_ids.shape[1] < max_new_tokens:
            # ── Step 1: Draft K tokens with draft model ──
            draft_tokens = []
            draft_logits = next_logits  # start from main model's logits

            for k in range(num_speculative_tokens):
                next_tok = self.sampler.sample_batch(
                    draft_logits.clone(),
                    generated_tokens=all_tokens,
                    do_sample=do_sample,
                )  # (B, 1)
                draft_tokens.append(next_tok)

                # Run draft model forward for next logits
                if draft_model is not self.model:
                    draft_output = draft_model(
                        torch.cat([generated] + [t for t in draft_tokens], dim=1)
                    )
                    draft_logits = draft_output["logits"][:, -1, :]
                else:
                    # When using same model, just shift logits
                    draft_logits = self.model.embed_tokens(next_tok)

            # ── Step 2: Verify all K draft tokens with main model ──
            draft_sequence = torch.cat([generated] + draft_tokens, dim=1)
            output = self.model(draft_sequence)
            target_logits = output["logits"]  # (B, prompt_len + K, vocab)

            # ── Step 3: Rejection sampling ──
            accepted = 0
            for k in range(num_speculative_tokens):
                pos = generated.shape[1] + k
                target_logit = target_logits[:, pos - 1, :]  # (B, vocab)
                draft_token = draft_tokens[k]  # (B, 1)

                # Get probabilities
                target_probs = F.softmax(target_logit / max(temperature, 1e-8), dim=-1)
                draft_probs = F.softmax(
                    draft_logits[:, :target_logit.shape[-1]] / max(temperature, 1e-8)
                    if k == 0 else target_logit / max(temperature, 1e-8),
                    dim=-1,
                )

                # Acceptance probability
                draft_token_idx = draft_token.squeeze(-1)
                accept_prob = torch.min(
                    torch.ones(B, device=self.device),
                    target_probs.gather(1, draft_token.unsqueeze(-1)).squeeze(-1)
                    / draft_probs.gather(1, draft_token.unsqueeze(-1)).squeeze(-1).clamp(min=1e-8),
                )

                # Accept or reject
                random_vals = torch.rand(B, device=self.device)
                accept_mask = random_vals < accept_prob

                if accept_mask.all():
                    # All accepted
                    accepted += 1
                    generated = torch.cat([generated, draft_token], dim=1)
                    for i in range(B):
                        all_tokens[i].append(draft_token[i, 0].item())
                else:
                    # At least one rejection: resample from target
                    if accepted > 0:
                        generated = torch.cat([generated]
                            + [draft_tokens[a] for a in range(accepted)], dim=1)
                        for i in range(B):
                            for a in range(accepted):
                                all_tokens[i].append(draft_tokens[a][i, 0].item())

                    # Resample at mismatch position
                    resampled = self.sampler.sample_batch(
                        target_logit.clone(),
                        generated_tokens=all_tokens,
                        do_sample=do_sample,
                    )
                    generated = torch.cat([generated, resampled], dim=1)
                    for i in range(B):
                        all_tokens[i].append(resampled[i, 0].item())
                    break

                if (draft_token.squeeze(-1) == eos_token_id).all():
                    break

            total_accepted += accepted
            total_drafted += num_speculative_tokens

            if generated.shape[1] - input_ids.shape[1] >= max_new_tokens:
                break

            if (generated[:, -1] == eos_token_id).all():
                break

            # Prep for next iteration
            next_logits = target_logits[:, -1, :]

        elapsed = time.time() - start_time
        tokens_generated = generated.shape[1] - input_ids.shape[1]
        self._last_gen_speed = tokens_generated / elapsed if tokens_generated > 0 else 0.0
        self._spec_acceptance_rate = total_accepted / max(total_drafted, 1)

        return generated

    @property
    def spec_acceptance_rate(self) -> float:
        """Acceptance rate from last speculative decode run."""
        return getattr(self, "_spec_acceptance_rate", 0.0)
