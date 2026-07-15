"""Model quantization for efficient serving.

Provides:
- QuantizedModel: Wrapper that quantizes model weights to int4/int8
- GPTQ-style quantization: optimal brain quantizer with Hessian calibration
- AWQ-style quantization: activation-aware weight quantization
- Dynamic quantization: PyTorch native dynamic quantization
- QuantizedLinear: drop-in replacement for nn.Linear with int4/int8 weights
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from dataclasses import dataclass


# ─── Quantized Linear ────────────────────────────────────────────────


class QuantizedLinear(nn.Module):
    """Int4/Int8 quantized linear layer with per-channel scaling.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        bits: Quantization bits (4 or 8).
        group_size: Group size for per-group quantization.
        sym: Symmetric quantization.
        bias: Whether to include bias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 4,
        group_size: int = 128,
        sym: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = min(group_size, in_features)
        self.sym = sym

        self.register_buffer(
            "qweight",
            torch.zeros(out_features, in_features // (8 // bits), dtype=torch.int32),
        )
        self.register_buffer(
            "scales",
            torch.zeros(out_features, (in_features + self.group_size - 1) // self.group_size),
        )
        self.register_buffer(
            "zeros",
            torch.zeros(out_features, (in_features + self.group_size - 1) // self.group_size, dtype=torch.int32),
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_buffer("bias", None)

    def _unpack(self, packed: Tensor) -> Tensor:
        """Unpack packed int32 weights to int values with sign extension."""
        num_groups = packed.shape[1]
        unpacked = torch.zeros(
            packed.shape[0], num_groups * (32 // self.bits),
            dtype=torch.int32, device=packed.device,
        )
        for i in range(32 // self.bits):
            shift = i * self.bits
            mask = (1 << self.bits) - 1
            unpacked[:, i::32 // self.bits] = (packed >> shift) & mask
        result = unpacked[:, :self.in_features]

        # Sign extension for symmetric quantization
        if self.sym:
            half = 1 << (self.bits - 1)
            result = ((result.int() ^ half) - half).to(torch.int32)

        return result

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with quantized weights.

        Args:
            x: (..., in_features) input tensor.

        Returns:
            (..., out_features) output tensor.
        """
        # Unpack weights
        weight = self._unpack(self.qweight).float()  # (out_features, in_features)

        # Dequantize with per-group scales
        if self.group_size < self.in_features:
            num_groups = self.scales.shape[1]
            for g in range(num_groups):
                start = g * self.group_size
                end = min(start + self.group_size, self.in_features)
                weight[:, start:end] = (weight[:, start:end] - self.zeros[:, g].unsqueeze(1)) * self.scales[:, g].unsqueeze(1)

        return F.linear(x, weight, self.bias)


# ─── GPTQ-style Quantization ─────────────────────────────────────────


@dataclass
class GPTQConfig:
    """GPTQ quantization configuration.

    Args:
        bits: Quantization bits (2, 3, 4, 8).
        group_size: Group size for per-group quantization.
        damp_percent: Damping factor for Hessian.
        desc_act: Whether to quantize in descending activation order.
        sym: Symmetric quantization.
        true_sequential: Whether to quantize layers sequentially.
    """

    bits: int = 4
    group_size: int = 128
    damp_percent: float = 0.01
    desc_act: bool = True
    sym: bool = True
    true_sequential: bool = True


class GPTQQuantizer:
    """GPTQ (Optimal Brain Quantizer) post-training quantization.

    Quantizes model weights while minimizing output error using
    approximate second-order information (Hessian).

    Args:
        config: GPTQ configuration.
    """

    def __init__(self, config: GPTQConfig | None = None) -> None:
        self.config = config or GPTQConfig()

    def quantize(self, model: nn.Module, calibration_data: list[Tensor] | None = None) -> nn.Module:
        """Quantize a model using GPTQ.

        Args:
            model: Model to quantize.
            calibration_data: List of calibration input tensors.

        Returns:
            Quantized model.
        """
        cfg = self.config

        if calibration_data is None:
            # Generate synthetic calibration data
            dummy_input = torch.randint(0, 32000, (8, 128))
            calibration_data = [dummy_input]

        # Collect Hessian info per layer
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and module.weight.shape[0] > 128:
                self._quantize_linear(model, module, calibration_data, cfg)

        return model

    def _quantize_linear(
        self,
        root: nn.Module,
        linear: nn.Linear,
        calibration_data: list[Tensor],
        cfg: GPTQConfig,
    ) -> None:
        """Quantize a single linear layer.

        Uses the OBQ algorithm: iteratively quantize one weight at a time
        and adjust remaining weights to compensate.
        """
        W = linear.weight.data.clone()  # (out_features, in_features)
        out_features, in_features = W.shape

        # Compute Hessian approximation from calibration data
        H = torch.zeros(in_features, in_features, device=W.device, dtype=W.dtype)
        n_samples = 0
        for inp in calibration_data:
            if inp.dim() == 2:
                inp = inp.to(W.device, dtype=W.dtype)
            else:
                continue
            H += inp.T @ inp / inp.shape[0]
            n_samples += 1
        H /= max(n_samples, 1)

        # Add damping
        damp = cfg.damp_percent * torch.mean(torch.diag(H))
        H += damp * torch.eye(in_features, device=H.device, dtype=H.dtype)

        # Cholesky decomposition
        try:
            L = torch.linalg.cholesky(H)
        except RuntimeError:
            # Fallback: use diagonal approximation
            H = torch.diag(torch.diag(H)) + damp * torch.eye(in_features, device=H.device, dtype=H.dtype)
            L = torch.linalg.cholesky(H)

        Linv = torch.linalg.inv(L)

        # Quantize column by column
        Q = torch.zeros_like(W)
        max_val = 2 ** (cfg.bits - 1) - 1 if cfg.sym else 2 ** cfg.bits - 1
        min_val = -max_val if cfg.sym else 0

        for col in range(in_features):
            w_col = W[:, col]
            q_col = torch.clamp(torch.round(w_col), min_val, max_val)
            Q[:, col] = q_col

            # Update remaining columns
            if col < in_features - 1:
                err = (w_col - q_col) / Linv[col, col]
                W[:, col+1:] -= err.unsqueeze(1) * Linv[col, col+1:].unsqueeze(0)

        # Pack into QuantizedLinear
        qlinear = QuantizedLinear(
            in_features,
            out_features,
            bits=cfg.bits,
            group_size=cfg.group_size,
            sym=cfg.sym,
            bias=linear.bias is not None,
        )

        # Pack weights
        pack_factor = 32 // cfg.bits
        packed = torch.zeros(out_features, (in_features + pack_factor - 1) // pack_factor, dtype=torch.int32)
        for i in range(pack_factor):
            packed |= (Q.long() & ((1 << cfg.bits) - 1)) << (i * cfg.bits)
            Q = Q >> cfg.bits
        qlinear.qweight = packed

        if linear.bias is not None:
            qlinear.bias = linear.bias.data.clone()

        # Replace the linear layer
        parent = self._find_parent(root, linear)
        if parent is not None:
            for child_name, child in list(parent.named_children()):
                if child is linear:
                    setattr(parent, child_name, qlinear)
                    break

    @staticmethod
    def _find_parent(root: nn.Module, target: nn.Module) -> nn.Module | None:
        """Find the parent module of a given module by walking from root."""
        for parent in root.modules():
            for child_name, child in parent.named_children():
                if child is target:
                    return parent
        return None


# ─── AWQ-style Quantization ──────────────────────────────────────────


@dataclass
class AWQConfig:
    """AWQ quantization configuration.

    Args:
        bits: Quantization bits (4 or 8).
        group_size: Group size for per-group quantization.
        zero_point: Whether to include zero point.
        version: "gemm" or "gemv" kernel.
    """

    bits: int = 4
    group_size: int = 128
    zero_point: bool = True
    version: str = "gemm"


class AWQQuantizer:
    """Activation-Aware Weight Quantization (AWQ).

    Finds optimal per-channel scaling factors by analyzing
    activation distributions, then quantizes weights.

    Args:
        config: AWQ configuration.
    """

    def __init__(self, config: AWQConfig | None = None) -> None:
        self.config = config or AWQConfig()

    def quantize(self, model: nn.Module, calibration_data: list[Tensor] | None = None) -> nn.Module:
        """Quantize a model using AWQ.

        Args:
            model: Model to quantize.
            calibration_data: List of calibration input tensors.

        Returns:
            Quantized model.
        """
        cfg = self.config

        if calibration_data is None:
            calibration_data = [torch.randint(0, 32000, (8, 128))]

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and module.weight.shape[0] > 128:
                self._quantize_linear(model, module, calibration_data, cfg)

        return model

    def _quantize_linear(
        self,
        root: nn.Module,
        linear: nn.Linear,
        calibration_data: list[Tensor],
        cfg: AWQConfig,
    ) -> None:
        """Quantize a single linear layer with AWQ."""
        W = linear.weight.data.float()
        out_features, in_features = W.shape

        # Collect activation statistics
        act_max = torch.zeros(in_features, device=W.device)
        for inp in calibration_data:
            if inp.dim() == 2:
                inp = inp.to(W.device, dtype=torch.float)
            else:
                continue
            act_max = torch.maximum(act_max, inp.abs().max(dim=0)[0])

        # Find optimal scaling factors
        group_size = cfg.group_size
        num_groups = (in_features + group_size - 1) // group_size

        scales = torch.ones(out_features, num_groups, device=W.device)

        for g in range(num_groups):
            start = g * group_size
            end = min(start + group_size, in_features)

            w_group = W[:, start:end]
            a_group = act_max[start:end]

            # Scale by max activation magnitude
            s = a_group.max() / (a_group + 1e-8)
            s = s / s.max()  # normalize

            # Apply scaling
            scaled_w = w_group * s.unsqueeze(0)
            scaled_w = scaled_w / scaled_w.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-8)

            max_val = 2 ** (cfg.bits - 1) - 1
            q_w = torch.clamp(torch.round(scaled_w * max_val), -max_val, max_val)
            scales[:, g] = s.mean()

            # Pack back
            W[:, start:end] = q_w / max_val * scales[:, g].unsqueeze(1)

        # Replace layer
        qlinear = QuantizedLinear(
            in_features, out_features,
            bits=cfg.bits, group_size=cfg.group_size,
            bias=linear.bias is not None,
        )

        # Pack weights
        pack_factor = 32 // cfg.bits
        Q = torch.clamp(torch.round(W * (2 ** (cfg.bits - 1) - 1)), -2 ** (cfg.bits - 1) + 1, 2 ** (cfg.bits - 1) - 1).long()
        packed = torch.zeros(out_features, (in_features + pack_factor - 1) // pack_factor, dtype=torch.int32)
        for i in range(pack_factor):
            packed |= (Q & ((1 << cfg.bits) - 1)) << (i * cfg.bits)
            Q = Q >> cfg.bits
        qlinear.qweight = packed

        if linear.bias is not None:
            qlinear.bias = linear.bias.data.clone()

        # Replace
        parent = self._find_parent(root, linear)
        if parent is not None:
            for child_name, child in list(parent.named_children()):
                if child is linear:
                    setattr(parent, child_name, qlinear)
                    break

    @staticmethod
    def _find_parent(root: nn.Module, target: nn.Module) -> nn.Module | None:
        for parent in root.modules():
            for child_name, child in parent.named_children():
                if child is target:
                    return parent
        return None


# ─── Dynamic Quantization (PyTorch native) ────────────────────────────


def quantize_dynamic(
    model: nn.Module,
    dtype: torch.dtype = torch.qint8,
    layers_to_quantize: tuple = (nn.Linear,),
) -> nn.Module:
    """Apply PyTorch dynamic quantization to a model.

    This is the simplest quantization method: quantizes weights to int8
    at load time, with activations quantized dynamically per forward pass.

    Args:
        model: Model to quantize.
        dtype: Quantization dtype (qint8 or quint8).
        layers_to_quantize: Module types to quantize.

    Returns:
        Quantized model (in-place).
    """
    return torch.quantization.quantize_dynamic(
        model,
        {layers_to_quantize},
        dtype=dtype,
    )


# ─── Quantization Utilities ──────────────────────────────────────────


def estimate_model_size(model: nn.Module, bits: int = 4) -> dict[str, float]:
    """Estimate memory footprint after quantization.

    Args:
        model: Model to estimate.
        bits: Target quantization bits.

    Returns:
        Dict with size estimates in GB.
    """
    total_params = sum(p.numel() for p in model.parameters())
    fp32_size = total_params * 4 / 1e9  # GB
    fp16_size = total_params * 2 / 1e9
    quant_size = total_params * bits / 8 / 1e9

    return {
        "total_params": total_params,
        "fp32_gb": fp32_size,
        "fp16_gb": fp16_size,
        f"int{bits}_gb": quant_size,
        f"compression_ratio": fp16_size / max(quant_size, 1e-9),
    }


def get_quantization_config(
    model_size_gb: float,
    gpu_memory_gb: float | None = None,
) -> dict:
    """Recommend quantization config based on model and GPU size.

    Args:
        model_size_gb: Model size in GB (FP16).
        gpu_memory_gb: Available GPU memory in GB.

    Returns:
        Dict with recommended bits, group_size, and backend.
    """
    if gpu_memory_gb is None:
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
        else:
            gpu_memory_gb = 16.0  # assume CPU with 16GB

    if model_size_gb <= gpu_memory_gb * 0.8:
        return {"bits": 16, "backend": "fp16", "fits": True}
    elif model_size_gb <= gpu_memory_gb * 0.8 * 2:
        return {"bits": 8, "backend": "int8", "group_size": 128, "fits": True}
    elif model_size_gb <= gpu_memory_gb * 0.8 * 4:
        return {"bits": 4, "backend": "int4", "group_size": 128, "fits": True}
    else:
        return {"bits": 4, "backend": "int4", "group_size": 64, "fits": False, "cpu_offload": True}