"""Cosine learning rate schedule with linear warmup."""

from __future__ import annotations

import math


class CosineSchedule:
    """Cosine annealing with linear warmup.

    Args:
        warmup_steps: Number of linear warmup steps.
        max_steps: Total training steps.
        min_lr: Minimum learning rate after decay.
        max_lr: Peak learning rate after warmup.
    """

    def __init__(
        self,
        warmup_steps: int,
        max_steps: int,
        min_lr: float = 1e-5,
        max_lr: float = 3e-4,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.max_lr = max_lr

    def get_lr(self, step: int) -> float:
        """Get learning rate for the given step."""
        if step < self.warmup_steps:
            # Linear warmup
            return self.max_lr * step / max(1, self.warmup_steps)

        if step >= self.max_steps:
            return self.min_lr

        # Cosine decay
        progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))
