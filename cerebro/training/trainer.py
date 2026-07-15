"""Cerebro training loop.

Features:
- Mixed precision (BF16)
- Gradient accumulation
- Gradient clipping
- Cosine LR schedule with warmup
- Checkpoint save/load with safetensors
"""

from __future__ import annotations

import os
import time
import json
import math
import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm
from pathlib import Path

from cerebro.config import CerebroConfig
from cerebro.model.cerebro_model import Cerebro
from cerebro.training.scheduler import CosineSchedule
from cerebro.training.data import create_dataloader
from cerebro.training.monitoring import MonitorCallback, WandBLogger, TensorBoardLogger


class CerebroTrainer:
    """Training orchestrator for Cerebro models."""

    def __init__(
        self,
        config: CerebroConfig,
        device: str = "auto",
        output_dir: str = "checkpoints",
        monitor: MonitorCallback | None = None,
    ) -> None:
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor

        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Model
        self.model = Cerebro(config).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
        )

        # LR schedule
        self.scheduler = CosineSchedule(
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
            min_lr=config.learning_rate * 0.1,
            max_lr=config.learning_rate,
        )

        # Mixed precision
        self.scaler = None
        self.use_amp = config.bf16 and self.device.type == "cuda"
        if self.use_amp:
            self.autocast_dtype = torch.bfloat16

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float("inf")
        self.log_interval = 10
        self._grad_norm = 0.0

    def save_checkpoint(self, tag: str = "latest") -> None:
        """Save model checkpoint using safetensors."""
        ckpt_dir = self.output_dir / tag
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights
        try:
            from safetensors.torch import save_file
            # Clone tensors to avoid shared memory issues
            state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            save_file(
                state_dict,
                str(ckpt_dir / "model.safetensors"),
            )
        except (ImportError, RuntimeError):
            # Fall back to torch.save if safetensors fails or not available
            torch.save(self.model.state_dict(), ckpt_dir / "model.pt")

        # Save optimizer and training state
        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "optimizer": self.optimizer.state_dict(),
            "config": {
                k: v for k, v in self.config.__dict__.items()
                if isinstance(v, (int, float, str, bool, type(None)))
            },
        }
        torch.save(state, ckpt_dir / "training_state.pt")

        # Save config
        with open(ckpt_dir / "config.json", "w") as f:
            json.dump(state["config"], f, indent=2)

    def load_checkpoint(self, tag: str = "latest") -> None:
        """Load model checkpoint."""
        ckpt_dir = self.output_dir / tag

        # Load model weights
        safetensors_path = ckpt_dir / "model.safetensors"
        pt_path = ckpt_dir / "model.pt"

        if safetensors_path.exists():
            from safetensors.torch import load_file
            state_dict = load_file(str(safetensors_path))
            self.model.load_state_dict(state_dict)
        elif pt_path.exists():
            self.model.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))

        # Load training state
        state_path = ckpt_dir / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location="cpu", weights_only=True)
            self.global_step = state.get("global_step", 0)
            self.epoch = state.get("epoch", 0)
            self.best_loss = state.get("best_loss", float("inf"))
            self.optimizer.load_state_dict(state["optimizer"])

        self.model.to(self.device)

    def train(
        self,
        data_dir: str | None = None,
        num_epochs: int = 1,
    ) -> dict[str, float]:
        """Run the training loop.

        Args:
            data_dir: Directory with tokenized training data.
            num_epochs: Number of training epochs.

        Returns:
            dict with final training metrics.
        """
        config = self.config

        # Initialize monitoring
        if self.monitor is not None:
            self.monitor.init()
            if self.monitor.wandb and self.monitor.wandb.is_enabled:
                self.monitor.wandb.watch(self.model, log_freq=100)

        # Create dataloader
        loader = create_dataloader(
            data_dir=data_dir,
            seq_len=config.max_seq_len,
            batch_size=config.batch_size,
            vocab_size=config.vocab_size,
        )

        self.model.train()
        total_loss = 0.0
        num_batches = 0
        start_time = time.time()

        for epoch in range(num_epochs):
            self.epoch = epoch
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}")

            for batch_idx, batch in enumerate(pbar):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Update learning rate
                lr = self.scheduler.get_lr(self.global_step)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

                # Forward pass
                if self.use_amp:
                    with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                        output = self.model(input_ids, labels=labels)
                        loss = output["loss"] / config.grad_accum_steps
                else:
                    output = self.model(input_ids, labels=labels)
                    loss = output["loss"] / config.grad_accum_steps

                # Backward pass
                loss.backward()

                # Gradient accumulation step
                if (batch_idx + 1) % config.grad_accum_steps == 0:
                    # Clip gradients and capture norm
                    self._grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), config.grad_clip
                    ).item()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    # Log
                    total_loss += loss.item() * config.grad_accum_steps
                    num_batches += 1

                    if self.global_step % self.log_interval == 0:
                        avg_loss = total_loss / num_batches
                        elapsed = time.time() - start_time
                        tokens_per_sec = (
                            config.batch_size * config.max_seq_len * self.global_step
                        ) / max(elapsed, 1.0)
                        pbar.set_postfix({
                            "loss": f"{avg_loss:.4f}",
                            "lr": f"{lr:.2e}",
                            "tok/s": f"{tokens_per_sec:.0f}",
                            "step": self.global_step,
                        })

                        # Monitor callback
                        if self.monitor is not None:
                            self.monitor.on_step(
                                step=self.global_step,
                                loss=avg_loss,
                                lr=lr,
                                tok_per_s=tokens_per_sec,
                                grad_norm=self._grad_norm,
                                model=self.model,
                            )

                    # Save periodic checkpoint
                    if self.global_step > 0 and self.global_step % 1000 == 0:
                        self.save_checkpoint("latest")

                    # Check if done
                    if self.global_step >= config.max_steps:
                        break

            # Epoch-end monitoring
            if self.monitor is not None and num_batches > 0:
                self.monitor.on_epoch(
                    epoch=epoch,
                    avg_loss=total_loss / num_batches,
                    step=self.global_step,
                )

            if self.global_step >= config.max_steps:
                break

        # Final save
        self.save_checkpoint("final")

        # Finish monitoring
        if self.monitor is not None:
            self.monitor.finish()

        elapsed = time.time() - start_time
        avg_loss = total_loss / max(num_batches, 1)

        return {
            "final_loss": avg_loss,
            "total_steps": self.global_step,
            "elapsed_seconds": elapsed,
            "tokens_per_second": (
                config.batch_size * config.max_seq_len * self.global_step
            ) / max(elapsed, 1.0),
        }
