"""Cerebro optimizer — AdamW wrapper with configurable options."""

from __future__ import annotations

import torch


def create_optimizer(model: torch.nn.Module, lr: float = 3e-4, weight_decay: float = 0.1) -> torch.optim.Optimizer:
    """Create AdamW optimizer with weight decay applied selectively.

    Bias and LayerNorm parameters are excluded from weight decay.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name or "log_temperature" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
