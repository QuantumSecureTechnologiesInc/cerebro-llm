"""SwiGLU Feed-Forward Network.

Uses the SwiGLU activation: gate(x) * up(x) where gate uses SiLU.
This is the standard FFN for modern LLMs (Llama, PaLM, etc.).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward block.

    Architecture:
        output = down_proj(SiLU(gate_proj(x)) * up_proj(x))

    With hidden_dim=2048 and ffn_dim=8192 (4x expansion):
        gate_proj: 2048 -> 8192
        up_proj:   2048 -> 8192
        down_proj: 8192 -> 2048
    """

    def __init__(self, hidden_dim: int, ffn_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, hidden_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: (B, S, hidden_dim)

        Returns:
            (B, S, hidden_dim)
        """
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
