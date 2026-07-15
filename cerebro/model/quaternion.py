"""Quaternion algebra for Cerebro's Hybrid Transformer-Quaternion Architecture.

Implements quaternion tensor operations and a parameter-efficient QLinear layer
where each quaternion weight encodes 4 real values, giving 4x parameter
efficiency compared to standard real-valued linear layers.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ── Pure functions ────────────────────────────────────────────────────────


def qmul(a: Tensor, b: Tensor) -> Tensor:
    """Hamilton product of two quaternion tensors.

    Args:
        a: (..., 4) — (w, x, y, z) components
        b: (..., 4) — (w, x, y, z) components

    Returns:
        (..., 4) quaternion product a ⊗ b (non-commutative).
    """
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)

    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,  # real
            aw * bx + ax * bw + ay * bz - az * by,  # i
            aw * by - ax * bz + ay * bw + az * bx,  # j
            aw * bz + ax * by - ay * bx + az * bw,  # k
        ],
        dim=-1,
    )


def qconj(q: Tensor) -> Tensor:
    """Quaternion conjugate: (w, x, y, z) -> (w, -x, -y, -z)."""
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def qnorm(q: Tensor) -> Tensor:
    """Quaternion norm (magnitude): sqrt(w^2 + x^2 + y^2 + z^2)."""
    return torch.sqrt(torch.sum(q * q, dim=-1, keepdim=True).clamp_min(1e-12))


def qnormalize(q: Tensor) -> Tensor:
    """Normalize quaternion to unit norm."""
    return q / qnorm(q)


def real_part(q: Tensor) -> Tensor:
    """Extract the real (scalar) component: (..., 4) -> (...,)."""
    return q[..., 0]


# ── QuaternionTensor wrapper ─────────────────────────────────────────────


class QuaternionTensor:
    """Convenience wrapper providing quaternion ops on a (..., 4) tensor."""

    def __init__(self, data: Tensor) -> None:
        assert data.shape[-1] == 4, "Last dim must be 4 (w, x, y, z)"
        self.data = data

    @property
    def shape(self):
        return self.data.shape

    def __mul__(self, other: QuaternionTensor) -> QuaternionTensor:
        return QuaternionTensor(qmul(self.data, other.data))

    def conjugate(self) -> QuaternionTensor:
        return QuaternionTensor(qconj(self.data))

    def norm(self) -> Tensor:
        return qnorm(self.data)

    def normalize(self) -> QuaternionTensor:
        return QuaternionTensor(qnormalize(self.data))

    def real(self) -> Tensor:
        return real_part(self.data)

    def to_tensor(self) -> Tensor:
        return self.data


# ── QLinear ───────────────────────────────────────────────────────────────


class QLinear(nn.Module):
    """Quaternion-valued linear layer.

    Uses a (out_features, in_features, 4) weight tensor where each quaternion
    parameter encodes 4 real values, giving ~4x parameter efficiency vs a
    standard nn.Linear of the same feature dimensions.

    Input:  (B, S, in_features, 4)  quaternion tensor
    Output: (B, S, out_features, 4) quaternion tensor
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, 4))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, 4))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Quaternion-aware Xavier initialization
        fan_in = self.in_features * 4  # 4 real components per quaternion
        fan_out = self.out_features * 4
        std = math.sqrt(2.0 / (fan_in + fan_out))
        nn.init.normal_(self.weight, mean=0.0, std=std)
        # Bias the real part slightly positive for numerical stability
        with torch.no_grad():
            self.weight[..., 0] += 0.01

    def forward(self, x: Tensor) -> Tensor:
        """Quaternion linear transform.

        Args:
            x: (B, S, in_features, 4)

        Returns:
            (B, S, out_features, 4)
        """
        # x: (B, S, in_features, 4), w: (out_features, in_features, 4)
        # We compute: for each output feature o, sum_i qmul(x[..., i, :], w[o, i, :])
        # This is a quaternion matrix-vector product.

        B, S, D, _ = x.shape
        out_D = self.out_features

        # Reshape for batched quaternion multiply
        # x: (B, S, 1, D, 4) broadcast with w: (1, 1, out_D, D, 4)
        x_exp = x.unsqueeze(2)          # (B, S, 1, D, 4)
        w_exp = self.weight.unsqueeze(0).unsqueeze(0)  # (1, 1, out_D, D, 4)

        # Element-wise quaternion multiply: (B, S, out_D, D, 4)
        products = qmul(x_exp, w_exp)

        # Sum over input dimension: (B, S, out_D, 4)
        out = products.sum(dim=3)

        if self.bias is not None:
            out = out + self.bias

        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )
