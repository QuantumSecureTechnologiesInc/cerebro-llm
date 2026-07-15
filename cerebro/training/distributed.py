"""Distributed training support for Cerebro: FSDP and DeepSpeed.

Provides:
- FSDPWrapper: Fully Sharded Data Parallel (PyTorch native)
- DeepSpeedConfig: DeepSpeed ZeRO configuration builder
- DistributedTrainer: Mixin that adds FSDP/DeepSpeed to the base trainer
- get_distributed_config: auto-detect best strategy for hardware
"""

from __future__ import annotations

import os
import json
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, Callable, Any
from dataclasses import dataclass, field
from pathlib import Path


# ─── FSDP ────────────────────────────────────────────────────────────


def _fsdp_available() -> bool:
    """Check if FSDP is available (PyTorch >= 2.0)."""
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel
        return True
    except ImportError:
        return False


def wrap_fsdp(
    model: nn.Module,
    mixed_precision: bool = True,
    sharding_strategy: str = "full",
    cpu_offload: bool = False,
    auto_wrap_policy: str | None = "transformer_layer",
    backward_prefetch: str = "backward_pre",
    activation_checkpointing: bool = True,
) -> nn.Module:
    """Wrap a model with FSDP for distributed training.

    Args:
        model: Cerebro model instance.
        mixed_precision: Use BF16 mixed precision.
        sharding_strategy: "full" (ZeRO-3), "hybrid", or "grad" (ZeRO-2).
        cpu_offload: Offload parameters to CPU.
        auto_wrap_policy: Module class to auto-wrap ("transformer_layer" or None).
        backward_prefetch: Prefetch strategy for backward pass.
        activation_checkpointing: Enable gradient checkpointing.

    Returns:
        FSDP-wrapped model.
    """
    if not _fsdp_available():
        raise ImportError("FSDP requires PyTorch >= 2.0. Install: pip install torch>=2.0")

    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
        BackwardPrefetch,
        CPUOffload,
    )
    from torch.distributed.fsdp.wrap import (
        size_based_auto_wrap_policy,
        transformer_auto_wrap_policy,
        enable_wrap,
        wrap,
    )

    # Mixed precision
    mp_policy = None
    if mixed_precision:
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

    # Sharding strategy
    strategy_map = {
        "full": ShardingStrategy.FULL_SHARD,       # ZeRO-3
        "hybrid": ShardingStrategy.HYBRID_SHARD,    # ZeRO-3 across nodes
        "grad": ShardingStrategy.SHARD_GRAD_OP,     # ZeRO-2
        "none": ShardingStrategy.NO_SHARD,           # DDP only
    }
    strategy = strategy_map.get(sharding_strategy, ShardingStrategy.FULL_SHARD)

    # Auto-wrap policy
    auto_wrap = None
    if auto_wrap_policy == "transformer_layer":
        try:
            from cerebro.model.block import CerebroBlock
            auto_wrap = {CerebroBlock}
        except ImportError:
            pass

    # Backward prefetch
    prefetch_map = {
        "backward_pre": BackwardPrefetch.BACKWARD_PRE,
        "backward_post": BackwardPrefetch.BACKWARD_POST,
    }
    prefetch = prefetch_map.get(backward_prefetch, BackwardPrefetch.BACKWARD_PRE)

    wrapped = FSDP(
        model,
        sharding_strategy=strategy,
        mixed_precision=mp_policy,
        auto_wrap_policy=auto_wrap,
        backward_prefetch=prefetch,
        cpu_offload=CPUOffload(offload_params=cpu_offload),
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
    )

    # Activation checkpointing
    if activation_checkpointing:
        from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy
        try:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                checkpoint_wrapper,
                CheckpointImpl,
                apply_activation_checkpointing,
            )
            non_reentrant = CheckpointImpl.NO_REENTRANT
            apply_activation_checkpointing(
                wrapped,
                checkpoint_wrapper_fn=lambda m: checkpoint_wrapper(m, checkpoint_impl=non_reentrant),
                auto_wrap_policy=auto_wrap,
            )
        except ImportError:
            pass

    return wrapped


# ─── DeepSpeed ────────────────────────────────────────────────────────


def _deepspeed_available() -> bool:
    """Check if DeepSpeed is available."""
    try:
        import deepspeed
        return True
    except ImportError:
        return False


@dataclass
class DeepSpeedConfig:
    """Build DeepSpeed ZeRO configuration.

    Args:
        stage: ZeRO stage (1, 2, or 3).
        offload_optimizer: Offload optimizer states to CPU.
        offload_param: Offload parameters to CPU.
        bf16: Use BF16 mixed precision.
        fp16: Use FP16 mixed precision.
        wall_clock_breakdown: Enable timing breakdown.
        contiguous_gradients: Reduce memory fragmentation.
        overlap_comm: Overlap communication with computation.
        reduce_bucket_size: Gradient reduction bucket size.
        allgather_bucket_size: All-gather bucket size.
        stage3_param_persistence_threshold: Param persistence threshold for ZeRO-3.
        stage3_prefetch_bucket_size: Prefetch bucket size for ZeRO-3.
        stage3_max_live_parameters: Max live parameters for ZeRO-3.
    """

    stage: int = 2
    offload_optimizer: bool = False
    offload_param: bool = False
    bf16: bool = True
    fp16: bool = False
    wall_clock_breakdown: bool = False
    contiguous_gradients: bool = True
    overlap_comm: bool = True
    reduce_bucket_size: int = 5e8
    allgather_bucket_size: int = 5e8
    stage3_param_persistence_threshold: int = 1e6
    stage3_prefetch_bucket_size: int = 5e8
    stage3_max_live_parameters: int = 1e9

    def to_dict(self) -> dict:
        cfg = {
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
            "gradient_accumulation_steps": "auto",
            "zero_optimization": {
                "stage": self.stage,
                "contiguous_gradients": self.contiguous_gradients,
                "overlap_comm": self.overlap_comm,
                "reduce_bucket_size": self.reduce_bucket_size,
                "allgather_bucket_size": self.allgather_bucket_size,
            },
            "wall_clock_breakdown": self.wall_clock_breakdown,
        }

        if self.stage == 3:
            cfg["zero_optimization"].update({
                "stage3_param_persistence_threshold": self.stage3_param_persistence_threshold,
                "stage3_prefetch_bucket_size": self.stage3_prefetch_bucket_size,
                "stage3_max_live_parameters": self.stage3_max_live_parameters,
            })

        if self.offload_optimizer:
            cfg["zero_optimization"]["offload_optimizer"] = {
                "device": "cpu",
                "pin_memory": True,
            }
        if self.offload_param:
            cfg["zero_optimization"]["offload_param"] = {
                "device": "cpu",
                "pin_memory": True,
            }

        if self.bf16:
            cfg["bf16"] = {"enabled": True}
        if self.fp16:
            cfg["fp16"] = {"enabled": True}

        return cfg

    def to_json(self, path: str | None = None) -> str:
        """Serialize to JSON string or file.

        Args:
            path: Optional file path to write to.

        Returns:
            JSON string.
        """
        json_str = json.dumps(self.to_dict(), indent=2)
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                f.write(json_str)
        return json_str

    @classmethod
    def from_preset(cls, preset: str = "7b") -> "DeepSpeedConfig":
        """Create config from size preset.

        Args:
            preset: "7b", "13b", "70b", or "cpu_offload".

        Returns:
            DeepSpeedConfig instance.
        """
        presets = {
            "7b": cls(stage=2, offload_optimizer=False),
            "13b": cls(stage=2, offload_optimizer=True),
            "70b": cls(
                stage=3,
                offload_optimizer=True,
                offload_param=True,
                stage3_prefetch_bucket_size=5e7,
                stage3_max_live_parameters=1e8,
            ),
            "cpu_offload": cls(stage=3, offload_optimizer=True, offload_param=True),
        }
        return presets.get(preset, cls())


def wrap_deepspeed(
    model: nn.Module,
    config: DeepSpeedConfig | None = None,
    config_path: str | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[nn.Module, Any, Any]:
    """Wrap a model with DeepSpeed.

    Args:
        model: Cerebro model instance.
        config: DeepSpeedConfig object.
        config_path: Path to DeepSpeed JSON config file.
        optimizer: PyTorch optimizer (optional, DeepSpeed can create one).

    Returns:
        (model_engine, optimizer, dataloader) tuple.
    """
    if not _deepspeed_available():
        raise ImportError("DeepSpeed is not installed. Install: pip install deepspeed")

    import deepspeed

    if config is None and config_path is None:
        config = DeepSpeedConfig()

    ds_config = config.to_dict() if config else config_path

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config_params=ds_config if isinstance(ds_config, dict) else config_path,
    )

    return model_engine, optimizer


# ─── Distributed Trainer Mixin ───────────────────────────────────────


class DistributedTrainerMixin:
    """Mixin that adds FSDP/DeepSpeed support to the Cerebro trainer.

    Usage:
        class MyTrainer(DistributedTrainerMixin, BaseTrainer):
            ...
    """

    def __init__(self, *args, **kwargs) -> None:
        self._distributed_backend: str | None = None
        self._distributed_model: nn.Module | None = None

    @property
    def is_distributed(self) -> bool:
        """Whether the model is currently distributed."""
        return self._distributed_model is not None

    @property
    def rank(self) -> int:
        """Global rank of this process."""
        if dist.is_initialized():
            return dist.get_rank()
        return 0

    @property
    def world_size(self) -> int:
        """Total number of processes."""
        if dist.is_initialized():
            return dist.get_world_size()
        return 1

    @property
    def is_main_process(self) -> bool:
        """Whether this is the main (rank 0) process."""
        return self.rank == 0

    @property
    def device(self) -> torch.device:
        """Current device."""
        if torch.cuda.is_available():
            return torch.device(f"cuda:{self.rank}")
        return torch.device("cpu")

    def reduce_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Average loss across all processes."""
        if not dist.is_initialized() or self.world_size == 1:
            return loss
        loss_tensor = loss.clone().detach()
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        return loss_tensor / self.world_size

    def init_distributed(
        self,
        model: nn.Module,
        backend: str = "auto",
        deepspeed_config: DeepSpeedConfig | None = None,
        deepspeed_config_path: str | None = None,
    ) -> nn.Module:
        """Initialize distributed training.

        Args:
            model: Model to distribute.
            backend: "auto", "fsdp", "deepspeed", or "ddp".
            deepspeed_config: DeepSpeed configuration.
            deepspeed_config_path: Path to DeepSpeed config JSON.

        Returns:
            Distributed model.
        """
        if backend == "auto":
            backend = get_distributed_config().get("backend", "fsdp")

        self._distributed_backend = backend

        if backend == "fsdp":
            if not dist.is_initialized():
                raise RuntimeError("torch.distributed must be initialized before FSDP wrapping")
            self._distributed_model = wrap_fsdp(model)
        elif backend == "deepspeed":
            self._distributed_model, _ = wrap_deepspeed(
                model, config=deepspeed_config, config_path=deepspeed_config_path,
            )
        elif backend == "ddp":
            self._distributed_model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[torch.cuda.current_device()],
            )
        else:
            raise ValueError(f"Unknown distributed backend: {backend}")

        return self._distributed_model

    @property
    def distributed_model(self) -> nn.Module | None:
        return self._distributed_model

    def save_distributed_checkpoint(self, path: str) -> None:
        """Save a checkpoint from distributed training.

        Handles FSDP state dict consolidation.
        """
        if self._distributed_backend == "fsdp":
            from torch.distributed.fsdp import (
                FullyShardedDataParallel as FSDP,
                FullStateDictConfig,
                StateDictType,
            )
            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self._distributed_model, StateDictType.FULL_STATE_DICT, cfg):
                state_dict = self._distributed_model.state_dict()
            if dist.get_rank() == 0:
                torch.save(state_dict, path)
        elif self._distributed_backend == "deepspeed":
            self._distributed_model.save_checkpoint(path)
        else:
            if dist.get_rank() == 0:
                torch.save(self._distributed_model.state_dict(), path)


# ─── Auto-detection ──────────────────────────────────────────────────


def get_distributed_config() -> dict:
    """Auto-detect the best distributed strategy for the hardware.

    Returns:
        Dict with recommended backend, stage, and settings.
    """
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        return {"backend": "none", "gpu_count": 0, "reason": "No GPUs detected"}

    # Check GPU memory
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_mem / 1e9

    if gpu_count == 1:
        if gpu_mem_gb >= 40:  # A100-80GB, H100
            return {"backend": "fsdp", "stage": 1, "gpu_count": 1, "gpu_mem_gb": gpu_mem_gb}
        elif gpu_mem_gb >= 24:  # RTX 4090, A10
            return {"backend": "deepspeed", "stage": 2, "gpu_count": 1, "gpu_mem_gb": gpu_mem_gb}
        else:
            return {"backend": "deepspeed", "stage": 2, "cpu_offload": True, "gpu_count": 1, "gpu_mem_gb": gpu_mem_gb}

    if gpu_mem_gb >= 40 and gpu_count >= 8:
        return {"backend": "deepspeed", "stage": 3, "gpu_count": gpu_count, "gpu_mem_gb": gpu_mem_gb}
    elif gpu_mem_gb >= 24 and gpu_count >= 4:
        return {"backend": "deepspeed", "stage": 2, "gpu_count": gpu_count, "gpu_mem_gb": gpu_mem_gb}
    else:
        return {"backend": "fsdp", "stage": 2, "gpu_count": gpu_count, "gpu_mem_gb": gpu_mem_gb}


def init_process_group(
    backend: str = "nccl",
    timeout_seconds: int = 600,
) -> None:
    """Initialize the PyTorch distributed process group.

    Args:
        backend: Communication backend (nccl, gloo, mpi).
        timeout_seconds: Timeout for collective operations.
    """
    if dist.is_initialized():
        return

    dist.init_process_group(
        backend=backend,
        timeout=torch.distributed.timedelta(seconds=timeout_seconds),
    )

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)


# Alias for backward compatibility
DistributedTrainer = DistributedTrainerMixin


def get_launch_command(
    num_gpus: int,
    num_nodes: int = 1,
    script: str = "train.py",
    extra_args: str = "",
) -> str:
    """Generate torchrun launch command for distributed training.

    Args:
        num_gpus: GPUs per node.
        num_nodes: Number of nodes.
        script: Training script path.
        extra_args: Additional script arguments.

    Returns:
        torchrun command string (or 'python script' for single GPU).
    """
    if num_gpus == 1 and num_nodes == 1:
        return f"python {script} {extra_args}".strip()
    return (
        f"torchrun --nproc_per_node={num_gpus} "
        f"--nnodes={num_nodes} "
        f"--rdzv_backend=c10d "
        f"--rdzv_endpoint=localhost:29500 "
        f"{script} {extra_args}"
    ).strip()