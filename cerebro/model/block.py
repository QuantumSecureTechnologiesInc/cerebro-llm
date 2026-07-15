"""CerebroBlock — single transformer layer combining all components.

Architecture:
    x = x + EntropicGate(QMHA(RMSNorm(x)))
    x = x + SwiGLU(RMSNorm(x))
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from cerebro.model.attention import QuaternionMultiHeadAttention, KVCache
from cerebro.model.entropic import EntropicGatingLayer
from cerebro.model.ffn import SwiGLUFFN
from cerebro.model.norm import RMSNorm


class CerebroBlock(nn.Module):
    """Single Cerebro transformer block.

    Combines:
    - Quaternion Multi-Head Attention with GQA
    - Entropic Gating Layer
    - SwiGLU Feed-Forward Network
    - RMSNorm (pre-norm)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        ffn_dim: int,
        max_seq_len: int,
        rope_theta: float,
        entropy_min: float = 0.1,
        entropy_max: float = 8.0,
        init_temperature: float = 1.0,
    ) -> None:
        super().__init__()

        self.attn = QuaternionMultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
        )
        self.entropic_gate = EntropicGatingLayer(
            hidden_dim=hidden_dim,
            entropy_min=entropy_min,
            entropy_max=entropy_max,
            init_temperature=init_temperature,
        )
        self.ffn = SwiGLUFFN(hidden_dim=hidden_dim, ffn_dim=ffn_dim)
        self.norm1 = RMSNorm(hidden_dim)
        self.norm2 = RMSNorm(hidden_dim)

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        kv_cache: KVCache | None = None,
    ) -> Tensor:
        """Forward pass through one Cerebro block.

        Args:
            x: (B, S, hidden_dim)
            mask: optional attention mask
            position_ids: optional position indices
            kv_cache: optional KV-cache for autoregressive inference

        Returns:
            (B, S, hidden_dim)
        """
        # ── Attention + Entropic Gating + Residual ──
        residual = x
        x = self.norm1(x)
        x = self.attn(x, mask=mask, position_ids=position_ids, kv_cache=kv_cache)
        x = self.entropic_gate(x, self.attn.get_last_attention_weights())
        x = residual + x

        # ── FFN + Residual ──
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x
