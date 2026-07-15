"""Quaternion Multi-Head Attention (QMHA) with Grouped Query Attention.

Dual-path architecture: standard vector attention runs in parallel with
quaternion attention. A learned fusion gate blends the two paths.

Supports KV-cache for efficient autoregressive inference.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

from cerebro.model.quaternion import QLinear, qmul, qconj, real_part
from cerebro.model.rope import RotaryPositionEmbedding


class KVCache:
    """Key-Value cache for a single attention layer.

    Stores past K and V tensors to avoid recomputation during
    autoregressive decoding. Managed per-layer for efficient O(1)
    per-step attention.
    """

    def __init__(self) -> None:
        self.k_cache: Tensor | None = None
        self.v_cache: Tensor | None = None
        self.past_len: int = 0

    def update(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Append new K,V to cache and return full K,V."""
        if self.k_cache is None:
            self.k_cache = k
            self.v_cache = v
        else:
            self.k_cache = torch.cat([self.k_cache, k], dim=2)
            self.v_cache = torch.cat([self.v_cache, v], dim=2)
        self.past_len = self.k_cache.shape[2]
        return self.k_cache, self.v_cache

    def reset(self) -> None:
        """Clear the cache."""
        self.k_cache = None
        self.v_cache = None
        self.past_len = 0


class QuaternionMultiHeadAttention(nn.Module):
    """Dual-path attention: vector (standard) + quaternion (QMHA).

    Uses Grouped Query Attention (GQA) to reduce KV memory:
    - num_heads query heads
    - num_kv_heads key/value heads (typically num_heads / 4)

    The quaternion path uses QLinear projections and computes attention
    via quaternion conjugate products (non-commutative scoring).
    A learned fusion gate alpha blends both paths.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int | None = None,
        max_seq_len: int = 8192,
        rope_theta: float = 10_000.0,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or (hidden_dim // num_heads)
        self.num_groups = num_heads // num_kv_heads

        # ── Vector path (standard attention) ──
        self.q_proj = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_dim, bias=False)

        # ── Quaternion path ──
        quat_dim = hidden_dim // 4  # quaternion_dim
        num_q_heads = num_heads
        num_kv_q_heads = num_kv_heads
        self.q_qproj = QLinear(quat_dim, num_q_heads * (self.head_dim // 4), bias=False)
        self.k_qproj = QLinear(quat_dim, num_kv_q_heads * (self.head_dim // 4), bias=False)
        self.v_qproj = QLinear(quat_dim, num_kv_q_heads * (self.head_dim // 4), bias=False)
        self.o_qproj = nn.Linear(num_q_heads * (self.head_dim // 4) * 4, hidden_dim, bias=False)

        # ── Fusion gate ──
        self.fusion_gate = nn.Parameter(torch.tensor(0.5))  # learned alpha in [0,1]

        # ── RoPE ──
        self.rope = RotaryPositionEmbedding(self.head_dim, max_seq_len, rope_theta)

        # ── Scale ──
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Store attention weights for entropic gating
        self._last_attn_weights: Tensor | None = None

    def _expand_kv(self, kv: Tensor) -> Tensor:
        """Expand KV heads to match query heads for GQA.

        Args:
            kv: (B, num_kv_heads, S, D)

        Returns:
            (B, num_heads, S, D)
        """
        if self.num_groups == 1:
            return kv
        B, H, S, D = kv.shape
        kv = kv.unsqueeze(2)  # (B, H, 1, S, D)
        kv = kv.expand(B, H, self.num_groups, S, D)
        return kv.reshape(B, H * self.num_groups, S, D)

    def _vector_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Standard scaled dot-product attention via PyTorch SDPA.

        Args:
            q: (B, H, S, D)
            k: (B, H_kv, S, D)
            v: (B, H_kv, S, D)
            mask: optional causal mask

        Returns:
            (output (B, H, S, D), attn_weights (B, H, S, S))
        """
        k_exp = self._expand_kv(k)
        v_exp = self._expand_kv(v)

        # Use PyTorch's efficient SDPA (Flash Attention when available)
        is_causal = mask is None
        out = F.scaled_dot_product_attention(
            q, k_exp, v_exp,
            attn_mask=mask,
            is_causal=is_causal,
            scale=self.scale,
        )

        # Compute attention weights for entropic gating (detached)
        with torch.no_grad():
            scores = torch.matmul(q, k_exp.transpose(-2, -1)) * self.scale
            if mask is not None:
                scores = scores + mask
            attn_weights = F.softmax(scores, dim=-1)

        return out, attn_weights

    def _quaternion_attention(
        self,
        x: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Quaternion path attention.

        Args:
            x: (B, S, hidden_dim) — input hidden states

        Returns:
            (B, S, hidden_dim) quaternion path output
        """
        B, S, _ = x.shape
        quat_dim = self.hidden_dim // 4
        q_head_dim = self.head_dim // 4

        # Reshape to quaternion: (B, S, quat_dim, 4)
        x_q = x.view(B, S, quat_dim, 4)

        # Q, K, V quaternion projections
        q = self.q_qproj(x_q)  # (B, S, num_heads * q_head_dim, 4)
        k = self.k_qproj(x_q)  # (B, S, num_kv_heads * q_head_dim, 4)
        v = self.v_qproj(x_q)  # (B, S, num_kv_heads * q_head_dim, 4)

        # Reshape to heads: (B, H, S, D_q, 4)
        q = q.view(B, S, self.num_heads, q_head_dim, 4).transpose(1, 2)
        k = k.view(B, S, self.num_kv_heads, q_head_dim, 4).transpose(1, 2)
        v = v.view(B, S, self.num_kv_heads, q_head_dim, 4).transpose(1, 2)

        # Flatten quaternion dim for standard attention computation
        # (B, H, S, D_q*4)
        q_flat = q.reshape(B, self.num_heads, S, q_head_dim * 4)
        k_flat = k.reshape(B, self.num_kv_heads, S, q_head_dim * 4)
        v_flat = v.reshape(B, self.num_kv_heads, S, q_head_dim * 4)

        # Standard attention on flattened quaternion features
        k_exp = self._expand_kv(k_flat)
        v_exp = self._expand_kv(v_flat)

        is_causal = mask is None
        out = F.scaled_dot_product_attention(
            q_flat, k_exp, v_exp,
            attn_mask=mask,
            is_causal=is_causal,
            scale=1.0 / math.sqrt(q_head_dim * 4),
        )

        # (B, H, S, D_q*4) -> (B, S, H*D_q*4)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)

        # Output projection back to hidden_dim
        out = self.o_qproj(out)

        return out

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        kv_cache: KVCache | None = None,
    ) -> Tensor:
        """Dual-path attention forward pass with optional KV-cache.

        Args:
            x: (B, S, hidden_dim)
            mask: optional attention mask
            position_ids: optional position indices for RoPE
            kv_cache: optional KV-cache for efficient autoregressive decoding

        Returns:
            (B, S, hidden_dim) fused attention output
        """
        B, S, _ = x.shape

        # ── Vector path ──
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = self.rope(q, k, position_ids)

        # Use KV-cache if provided (append cache, then use full cached K,V for attention)
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        vec_out, attn_weights = self._vector_attention(q, k, v, mask)
        self._last_attn_weights = attn_weights

        # (B, H, S, D) -> (B, S, H*D)
        vec_out = vec_out.transpose(1, 2).contiguous().view(B, S, -1)
        vec_out = self.o_proj(vec_out)

        # ── Quaternion path ──
        quat_out = self._quaternion_attention(x, mask)

        # ── Fusion ──
        alpha = torch.sigmoid(self.fusion_gate)
        out = alpha * vec_out + (1.0 - alpha) * quat_out

        return out

    def get_last_attention_weights(self) -> Tensor | None:
        """Return attention weights from the last forward pass (for entropic gating)."""
        return self._last_attn_weights
