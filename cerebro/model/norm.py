"""Root Mean Square Layer Normalization."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class RMSNorm(nn.Module):
    """RMSNorm with learned scale parameter (LLaMA-style).

    Normalizes by dividing by the root-mean-square of the last dimension,
    then applies a learned per-dimension scale. A small epsilon prevents
    division by zero.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., dim)
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight
