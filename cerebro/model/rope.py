"""Rotary Position Embedding (RoPE) for Cerebro.

Applied to both vector and quaternion attention paths. Supports
configurable theta and sequence length. Frequencies are cached
as an nn.Module buffer for efficiency.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor


class RotaryPositionEmbedding(nn.Module):
    """Rotary position embedding with cached frequency tables.

    Precomputes and caches cos/sin tables up to max_seq_len.
    Registered as non-persistent buffers (not saved in checkpoints).
    """

    def __init__(self, dim: int, max_seq_len: int = 8192, theta: float = 10_000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Compute inverse frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute and cache cos/sin tables
        self._cached_seq_len = 0
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)
        self._update_cache(max_seq_len)

    def _update_cache(self, seq_len: int) -> None:
        """Rebuild cos/sin cache for the given sequence length."""
        if seq_len <= self._cached_seq_len:
            return
        device = self.inv_freq.device
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        self.cos_cached = emb.cos()
        self.sin_cached = emb.sin()
        self._cached_seq_len = seq_len

    def apply_rotary(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Apply rotary embedding to a (B, H, S, D) tensor.

        cos, sin: (S, D) — broadcast over batch and head dims.
        """
        rot_dim = cos.shape[-1]
        x_rot = x[..., :rot_dim]
        x_pass = x[..., rot_dim:]

        x1_h = x_rot[..., : rot_dim // 2]
        x2_h = x_rot[..., rot_dim // 2 :]

        rotated = torch.cat([-x2_h, x1_h], dim=-1)
        out_rot = x_rot * cos[:rot_dim] + rotated * sin[:rot_dim]

        if rot_dim < x.shape[-1]:
            return torch.cat([out_rot, x_pass], dim=-1)
        return out_rot

    def forward(self, q: Tensor, k: Tensor, position_ids: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Apply RoPE to query and key tensors.

        Args:
            q: (B, H, S, D)
            k: (B, H_kv, S, D)
            position_ids: optional (B, S) position indices

        Returns:
            (q_rotated, k_rotated) with same shapes.
        """
        seq_len = q.shape[2]
        dtype = q.dtype

        # Extend cache if needed
        if seq_len > self._cached_seq_len:
            self._update_cache(seq_len)

        cos = self.cos_cached[:seq_len].to(dtype)
        sin = self.sin_cached[:seq_len].to(dtype)

        if position_ids is not None:
            cos = cos[position_ids]  # (B, S, D)
            sin = sin[position_ids]
            cos = cos.unsqueeze(1)   # (B, 1, S, D)
            sin = sin.unsqueeze(1)
        else:
            cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, S, D)
            sin = sin.unsqueeze(0).unsqueeze(0)

        q_out = self.apply_rotary(q, cos, sin)
        k_out = self.apply_rotary(k, cos, sin)
        return q_out, k_out
