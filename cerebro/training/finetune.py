"""LoRA and QLoRA fine-tuning for Cerebro.

Provides:
- LoRALinear: drop-in replacement for nn.Linear with low-rank adapters
- LoRAConfig: hyperparameters for LoRA adaptation
- apply_lora / remove_lora: attach/detach adapters to a model
- LoRATrainer: fine-tuning orchestrator with frozen base weights
"""

from __future__ import annotations

import os
import math
import json
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from typing import Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning."""
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    bias: str = "none"  # "none", "all", "lora_only"
    quantize: bool = False  # QLoRA: 4-bit quantization of base weights

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": self.target_modules,
            "bias": self.bias,
            "quantize": self.quantize,
        }

    @classmethod
    def from_dict(cls, d: dict) -> LoRAConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class LoRALinear(nn.Module):
    """Linear layer with LoRA low-rank adaptation.

    Replaces W with W + (α/r) * B @ A, where:
    - A: (in_features, rank) — down-projection
    - B: (rank, out_features) — up-projection
    - W: frozen base weights

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        rank: LoRA rank (low = fewer trainable params).
        alpha: LoRA scaling factor.
        dropout: Dropout on LoRA path.
        quantize: Quantize base weights to 4-bit (QLoRA).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.05,
        quantize: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Base weight (frozen)
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Quantized base weight (QLoRA)
        self.quantize = quantize
        if quantize:
            self._quantized_weight: Tensor | None = None
            self._quant_scale: Tensor | None = None

        # LoRA adapters (trainable)
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Initialize A with Kaiming, B with zeros (so LoRA starts as identity)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def quantize_base_weight(self) -> None:
        """Quantize base weight to 4-bit for QLoRA."""
        if not self.quantize:
            return

        w = self.weight.data
        # Simple per-channel 4-bit quantization
        absmax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        scale = absmax / 7.0  # 4-bit signed: -8 to 7
        quantized = torch.clamp(torch.round(w / scale), -8, 7).to(torch.int8)

        self._quantized_weight = quantized
        self._quant_scale = scale.squeeze(1)
        # Free the float weight
        self.weight = nn.Parameter(torch.empty(0), requires_grad=False)

    def _dequantize_weight(self) -> Tensor:
        """Dequantize base weight back to float."""
        if self._quantized_weight is None:
            return self.weight

        scale = self._quant_scale.unsqueeze(1)
        return (self._quantized_weight.float() * scale)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: base_weight @ x + scaling * (B @ A @ dropout(x))."""
        # Base path (frozen)
        if self.quantize and self._quantized_weight is not None:
            base_out = F.linear(x, self._dequantize_weight())
        else:
            base_out = F.linear(x, self.weight)

        # LoRA path (trainable)
        lora_x = self.lora_dropout(x)
        lora_out = lora_x @ self.lora_A @ self.lora_B * self.scaling

        return base_out + lora_out

    @property
    def trainable_params(self) -> int:
        """Number of trainable LoRA parameters."""
        return self.lora_A.numel() + self.lora_B.numel()

    def merge_weights(self) -> None:
        """Merge LoRA weights into base weight (for inference)."""
        if self.quantize:
            base = self._dequantize_weight()
        else:
            base = self.weight.data

        merged = base + (self.lora_B.T @ self.lora_A.T) * self.scaling

        self.weight = nn.Parameter(merged, requires_grad=False)
        self.lora_A = nn.Parameter(torch.zeros_like(self.lora_A), requires_grad=False)
        self.lora_B = nn.Parameter(torch.zeros_like(self.lora_B), requires_grad=False)
        self._quantized_weight = None
        self._quant_scale = None

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, "
            f"quantize={self.quantize}, "
            f"trainable={self.trainable_params:,}"
        )


def _find_linear_modules(model: nn.Module, target_names: list[str]) -> dict[str, nn.Linear]:
    """Find all linear layers matching target module names."""
    modules = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            short_name = name.split(".")[-1]
            if short_name in target_names or not target_names:
                modules[name] = module
    return modules


def apply_lora(
    model: nn.Module,
    config: LoRAConfig | None = None,
) -> dict[str, LoRALinear]:
    """Apply LoRA adapters to target linear layers in a model.

    Replaces target nn.Linear modules with LoRALinear, freezing
    original weights and adding trainable low-rank adapters.

    Args:
        model: PyTorch model to adapt.
        config: LoRA configuration.

    Returns:
        Dict of module_name -> LoRALinear replacements.
    """
    if config is None:
        config = LoRAConfig()

    targets = _find_linear_modules(model, config.target_modules)
    lora_modules = {}

    for name, linear in targets.items():
        lora_linear = LoRALinear(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout,
            quantize=config.quantize,
        )
        # Copy base weights
        lora_linear.weight.data.copy_(linear.weight.data)

        if config.quantize:
            lora_linear.quantize_base_weight()

        # Replace in model
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], lora_linear)

        lora_modules[name] = lora_linear

    return lora_modules


def remove_lora(model: nn.Module, merge: bool = True) -> None:
    """Remove LoRA adapters from a model.

    Args:
        model: Model with LoRA adapters.
        merge: If True, merge LoRA weights into base before removing.
    """
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            if merge:
                module.merge_weights()

            # Replace with regular Linear
            linear = nn.Linear(module.in_features, module.out_features, bias=False)
            if merge:
                linear.weight.data.copy_(module.weight.data)
            else:
                if module.quantize and module._quantized_weight is not None:
                    linear.weight.data.copy_(module._dequantize_weight())
                else:
                    linear.weight.data.copy_(module.weight.data)

            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], linear)


def get_lora_params(model: nn.Module) -> list[nn.Parameter]:
    """Get all trainable LoRA parameters."""
    params = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            params.append(module.lora_A)
            params.append(module.lora_B)
    return params


def count_lora_params(model: nn.Module) -> dict[str, int]:
    """Count LoRA and base model parameters.

    Returns:
        Dict with 'total', 'trainable', 'lora', 'base' counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora = sum(m.trainable_params for m in model.modules() if isinstance(m, LoRALinear))

    return {
        "total": total,
        "trainable": trainable,
        "lora": lora,
        "base": total - lora,
        "lora_percent": (lora / max(total, 1)) * 100,
    }


def save_lora_weights(model: nn.Module, path: str) -> None:
    """Save only the LoRA adapter weights (small file)."""
    lora_state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state[f"{name}.lora_A"] = module.lora_A.data.cpu()
            lora_state[f"{name}.lora_B"] = module.lora_B.data.cpu()

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(lora_state, path)


def load_lora_weights(model: nn.Module, path: str) -> None:
    """Load LoRA adapter weights into a model with LoRA applied."""
    lora_state = torch.load(path, map_location="cpu", weights_only=True)

    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            key_a = f"{name}.lora_A"
            key_b = f"{name}.lora_B"
            if key_a in lora_state:
                module.lora_A.data.copy_(lora_state[key_a])
            if key_b in lora_state:
                module.lora_B.data.copy_(lora_state[key_b])


class LoRATrainer:
    """Fine-tuning orchestrator using LoRA adapters.

    Freezes base model weights and only trains low-rank adapters,
    enabling efficient domain adaptation on consumer hardware.

    Args:
        model: Base Cerebro model.
        config: LoRA configuration.
        lr: Learning rate for adapters.
        device: Training device.
    """

    def __init__(
        self,
        model: nn.Module,
        config: LoRAConfig | None = None,
        lr: float = 2e-4,
        device: str = "auto",
    ) -> None:
        self.config = config or LoRAConfig()
        self.model = model

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = self.model.to(self.device)

        # Apply LoRA and freeze base
        self.lora_modules = apply_lora(self.model, self.config)

        # Freeze all non-LoRA parameters
        for name, param in self.model.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False

        # Optimizer only for LoRA params
        lora_params = get_lora_params(self.model)
        self.optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=0.01)

    def param_summary(self) -> str:
        counts = count_lora_params(self.model)
        return (
            f"LoRA Configuration: rank={self.config.rank}, alpha={self.config.alpha}\n"
            f"Total params:     {counts['total']:,}\n"
            f"Trainable (LoRA): {counts['lora']:,} ({counts['lora_percent']:.2f}%)\n"
            f"Frozen (base):    {counts['base']:,}"
        )

    def train(
        self,
        dataset: Dataset,
        num_epochs: int = 3,
        batch_size: int = 4,
    ) -> dict:
        """Run LoRA fine-tuning.

        Args:
            dataset: Training dataset yielding {'input_ids', 'labels'}.
            num_epochs: Number of training epochs.
            batch_size: Batch size.

        Returns:
            Training metrics dict.
        """
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        self.model.train()
        total_loss = 0.0
        total_steps = 0
        start_time = time.time()

        for epoch in range(num_epochs):
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                output = self.model(input_ids, labels=labels)
                loss = output["loss"]

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], 1.0,
                )
                self.optimizer.step()

                total_loss += loss.item()
                total_steps += 1

                if total_steps % 50 == 0:
                    avg = total_loss / total_steps
                    print(f"  Epoch {epoch+1} Step {total_steps}: loss={loss.item():.4f} avg={avg:.4f}")

        elapsed = time.time() - start_time

        return {
            "final_loss": total_loss / max(total_steps, 1),
            "total_steps": total_steps,
            "elapsed_seconds": elapsed,
            **count_lora_params(self.model),
        }

    def save(self, path: str) -> None:
        """Save LoRA weights and config."""
        save_lora_weights(self.model, path)
        config_path = str(Path(path).with_suffix(".json"))
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

    def merge_and_save(self, path: str) -> None:
        """Merge LoRA into base weights and save full model."""
        remove_lora(self.model, merge=True)
        torch.save(self.model.state_dict(), path)
