"""Entropic Gating Layer — thermodynamic attention modulation.

Computes per-token information entropy from attention weights, bounds it
with learned temperature, and modulates the attention output to prevent
overconfident predictions.

Inspired by the Landauer Sink's chaos vacuum from the Vortex Engine design.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class EntropicGatingLayer(nn.Module):
    """Modulates attention output based on information entropy bounds.

    When attention is too peaked (low entropy), the gate dampens to prevent
    overconfidence. When attention is too uniform (high entropy), the gate
    allows full signal through.

    H_bounded = clamp(H, entropy_min, entropy_max)
    gate = exp(-H_bounded / temperature)
    output = attention_output * gate
    """

    def __init__(
        self,
        hidden_dim: int,
        entropy_min: float = 0.1,
        entropy_max: float = 8.0,
        init_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.entropy_min = entropy_min
        self.entropy_max = entropy_max
        # Learnable temperature parameter
        self.log_temperature = nn.Parameter(
            torch.tensor(float(init_temperature)).log()
        )
        # Per-dimension gating bias
        self.gate_bias = nn.Parameter(torch.zeros(hidden_dim))

    @property
    def temperature(self) -> Tensor:
        return self.log_temperature.exp()

    def compute_entropy(self, attn_weights: Tensor) -> Tensor:
        """Compute per-token Shannon entropy from attention weights.

        Args:
            attn_weights: (B, H, S, S) softmax attention weights

        Returns:
            (B, S) average entropy per token across heads
        """
        # H = -sum(p * log(p)) for each position
        eps = 1e-10
        entropy = -(attn_weights * (attn_weights + eps).log()).sum(dim=-1)
        # Average over heads: (B, H, S) -> (B, S)
        return entropy.mean(dim=1)

    def forward(
        self,
        x: Tensor,
        attn_weights: Tensor | None = None,
    ) -> Tensor:
        """Apply entropic gating to the attention output.

        Args:
            x: (B, S, hidden_dim) attention output
            attn_weights: (B, H, S, S) attention weights (detached)

        Returns:
            (B, S, hidden_dim) gated output
        """
        if attn_weights is None:
            # No attention weights available, pass through with bias
            return x + self.gate_bias.unsqueeze(0).unsqueeze(0)

        # Compute entropy: (B, S)
        H = self.compute_entropy(attn_weights.detach())

        # Bound entropy
        H_bounded = torch.clamp(H, min=self.entropy_min, max=self.entropy_max)

        # Compute gate: (B, S, 1)
        T = self.temperature
        gate = torch.exp(-H_bounded / T).unsqueeze(-1)

        # Apply gate with bias
        return x * gate + self.gate_bias.unsqueeze(0).unsqueeze(0)
