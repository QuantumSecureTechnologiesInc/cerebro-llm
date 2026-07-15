"""Reasoning Core — lightweight self-verification stack.

A small set of additional transformer layers applied after the main
encoder stack that implement chain-of-thought attention with
self-verification cross-checking.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from cerebro.model.attention import QuaternionMultiHeadAttention
from cerebro.model.entropic import EntropicGatingLayer
from cerebro.model.ffn import SwiGLUFFN
from cerebro.model.norm import RMSNorm


class ReasoningBlock(nn.Module):
    """A single reasoning layer — same structure as CerebroBlock
    but used in the reasoning core stack."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        ffn_dim: int,
        max_seq_len: int,
        rope_theta: float,
        entropy_min: float,
        entropy_max: float,
        init_temperature: float,
    ) -> None:
        super().__init__()
        self.attn = QuaternionMultiHeadAttention(
            hidden_dim, num_heads, num_kv_heads, head_dim, max_seq_len, rope_theta
        )
        self.entropic_gate = EntropicGatingLayer(
            hidden_dim, entropy_min, entropy_max, init_temperature
        )
        self.ffn = SwiGLUFFN(hidden_dim, ffn_dim)
        self.norm1 = RMSNorm(hidden_dim)
        self.norm2 = RMSNorm(hidden_dim)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        # Attention + residual
        residual = x
        x = self.norm1(x)
        x = self.attn(x, mask)
        x = self.entropic_gate(x, self.attn.get_last_attention_weights())
        x = residual + x

        # FFN + residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x


class ReasoningCore(nn.Module):
    """Stack of reasoning blocks for self-verification.

    Applied after the main encoder stack during forward pass.
    At inference, the BoundedRecursionController can trigger
    additional passes through this core.
    """

    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        ffn_dim: int,
        max_seq_len: int,
        rope_theta: float,
        entropy_min: float,
        entropy_max: float,
        init_temperature: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            ReasoningBlock(
                hidden_dim, num_heads, num_kv_heads, head_dim,
                ffn_dim, max_seq_len, rope_theta,
                entropy_min, entropy_max, init_temperature,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x
