"""Experiment monitoring: WandB and TensorBoard integration.

Provides:
- WandBLogger: Weights & Biases experiment tracking
- TensorBoardLogger: TensorBoard scalar/histogram logging
- MonitorCallback: Unified logging to both backends
- Optional: no crash if wandb/tensorboard not installed

Follows production LLM monitoring patterns (LLaMA, Mistral, Gemma).
"""

from __future__ import annotations

import os
import time
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("cerebro.monitoring")


class WandBLogger:
    """Weights & Biases experiment tracking.

    Logs: loss, learning rate, throughput, gradient norms, parameter histograms.

    Args:
        project: WandB project name.
        name: Run name (auto-generated if None).
        config: Model/training config dict to log.
        log_dir: Directory for local WandB logs.
        entity: WandB team/entity name.
        tags: List of tags for the run.
        resume: Whether to resume a previous run.
    """

    def __init__(
        self,
        project: str = "cerebro",
        name: str | None = None,
        config: dict | None = None,
        log_dir: str = "logs/wandb",
        entity: str | None = None,
        tags: list[str] | None = None,
        resume: bool = False,
    ) -> None:
        self.project = project
        self.name = name
        self.config = config or {}
        self.log_dir = log_dir
        self.entity = entity
        self.tags = tags or []
        self.resume = resume
        self._run = None
        self._enabled = False

    def init(self) -> None:
        """Initialize WandB run."""
        try:
            import wandb
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
            self._run = wandb.init(
                project=self.project,
                name=self.name,
                config=self.config,
                dir=self.log_dir,
                entity=self.entity,
                tags=self.tags,
                resume=self.resume,
            )
            self._enabled = True
        except ImportError:
            self._enabled = False
        except Exception:
            logger.debug("WandB init failed", exc_info=True)
            self._enabled = False

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log metrics to WandB.

        Args:
            metrics: Dict of metric_name -> value.
            step: Global step number.
        """
        if not self._enabled or self._run is None:
            return
        try:
            import wandb
            wandb.log(metrics, step=step)
        except Exception:
            logger.debug("WandB log failed", exc_info=True)

    def log_histogram(self, name: str, values: Any, step: int) -> None:
        """Log a histogram of values."""
        if not self._enabled or self._run is None:
            return
        try:
            import wandb
            wandb.log({name: wandb.Histogram(values)}, step=step)
        except Exception:
            logger.debug("WandB histogram log failed", exc_info=True)

    def watch(self, model: Any, log_freq: int = 100) -> None:
        """Watch model gradients and parameters.

        Args:
            model: PyTorch model.
            log_freq: How often to log gradients (in batches).
        """
        if not self._enabled or self._run is None:
            return
        try:
            import wandb
            wandb.watch(model, log="gradients", log_freq=log_freq)
        except Exception:
            logger.debug("WandB watch failed", exc_info=True)

    def finish(self) -> None:
        """Close the WandB run."""
        if self._enabled and self._run is not None:
            try:
                import wandb
                wandb.finish()
            except Exception:
                logger.debug("WandB finish failed", exc_info=True)
            self._enabled = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled


class TensorBoardLogger:
    """TensorBoard scalar and histogram logging.

    Args:
        log_dir: Directory for TensorBoard event files.
        flush_secs: How often to flush events to disk.
    """

    def __init__(
        self,
        log_dir: str = "logs/tensorboard",
        flush_secs: int = 30,
    ) -> None:
        self.log_dir = log_dir
        self.flush_secs = flush_secs
        self._writer = None
        self._enabled = False

    def init(self) -> None:
        """Initialize TensorBoard writer."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(
                log_dir=self.log_dir,
                flush_secs=self.flush_secs,
            )
            self._enabled = True
        except ImportError:
            self._enabled = False
        except Exception:
            logger.debug("TensorBoard init failed", exc_info=True)
            self._enabled = False

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value.

        Args:
            tag: Metric name (e.g., 'train/loss').
            value: Scalar value.
            step: Global step.
        """
        if not self._enabled or self._writer is None:
            return
        try:
            self._writer.add_scalar(tag, value, step)
        except Exception:
            logger.debug("TensorBoard add_scalar failed", exc_info=True)

    def log_scalars(self, main_tag: str, tag_scalar_dict: dict[str, float], step: int) -> None:
        """Log multiple scalars under a main tag."""
        if not self._enabled or self._writer is None:
            return
        try:
            self._writer.add_scalars(main_tag, tag_scalar_dict, step)
        except Exception:
            logger.debug("TensorBoard add_scalars failed", exc_info=True)

    def log_histogram(self, tag: str, values: Any, step: int) -> None:
        """Log a histogram."""
        if not self._enabled or self._writer is None:
            return
        try:
            self._writer.add_histogram(tag, values, step)
        except Exception:
            logger.debug("TensorBoard add_histogram failed", exc_info=True)

    def log_text(self, tag: str, text: str, step: int) -> None:
        """Log text."""
        if not self._enabled or self._writer is None:
            return
        try:
            self._writer.add_text(tag, text, step)
        except Exception:
            logger.debug("TensorBoard add_text failed", exc_info=True)

    def close(self) -> None:
        """Close the TensorBoard writer."""
        if self._enabled and self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                logger.debug("TensorBoard close failed", exc_info=True)
            self._enabled = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled


class MonitorCallback:
    """Unified monitoring callback for WandB + TensorBoard.

    Production LLMs typically log to both backends simultaneously.
    Gracefully handles missing dependencies.

    Args:
        wandb_logger: Optional WandBLogger instance.
        tb_logger: Optional TensorBoardLogger instance.
        log_interval: Steps between log events.
        log_grads: Whether to log gradient norms.
        log_histograms: Whether to log parameter histograms periodically.
        histogram_interval: Steps between histogram logging.
    """

    def __init__(
        self,
        wandb_logger: WandBLogger | None = None,
        tb_logger: TensorBoardLogger | None = None,
        log_interval: int = 10,
        log_grads: bool = True,
        log_histograms: bool = False,
        histogram_interval: int = 500,
    ) -> None:
        self.wandb = wandb_logger
        self.tb = tb_logger
        self.log_interval = log_interval
        self.log_grads = log_grads
        self.log_histograms = log_histograms
        self.histogram_interval = histogram_interval

    def init(self) -> None:
        """Initialize all loggers."""
        if self.wandb:
            self.wandb.init()
        if self.tb:
            self.tb.init()

    def on_step(
        self,
        step: int,
        loss: float,
        lr: float,
        tok_per_s: float,
        grad_norm: float | None = None,
        model: Any = None,
    ) -> None:
        """Log training step metrics.

        Args:
            step: Global step number.
            loss: Current loss value.
            lr: Current learning rate.
            tok_per_s: Tokens per second throughput.
            grad_norm: Gradient norm (if computed).
            model: Model reference (for histogram logging).
        """
        if step % self.log_interval != 0:
            return

        metrics = {
            "train/loss": loss,
            "train/lr": lr,
            "train/tok_per_s": tok_per_s,
        }
        if grad_norm is not None:
            metrics["train/grad_norm"] = grad_norm

        # WandB
        if self.wandb and self.wandb.is_enabled:
            self.wandb.log(metrics, step=step)

        # TensorBoard
        if self.tb and self.tb.is_enabled:
            for key, value in metrics.items():
                self.tb.log_scalar(key, value, step)

        # Periodic histograms
        if self.log_histograms and step % self.histogram_interval == 0 and model is not None:
            self._log_histograms(step, model)

    def on_epoch(self, epoch: int, avg_loss: float, step: int) -> None:
        """Log epoch-level metrics.

        Args:
            epoch: Epoch number.
            avg_loss: Average loss for the epoch.
            step: Global step at epoch end.
        """
        if self.wandb and self.wandb.is_enabled:
            self.wandb.log({"epoch": epoch, "train/epoch_loss": avg_loss}, step=step)
        if self.tb and self.tb.is_enabled:
            self.tb.log_scalar("train/epoch_loss", avg_loss, step)

    def _log_histograms(self, step: int, model: Any) -> None:
        """Log parameter histograms."""
        try:
            import torch
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    values = param.data.detach().cpu().flatten()
                    if self.wandb and self.wandb.is_enabled:
                        self.wandb.log_histogram(f"params/{name}", values, step)
                    if self.tb and self.tb.is_enabled:
                        self.tb.log_histogram(f"params/{name}", values, step)
        except Exception:
            logger.debug("Histogram logging failed", exc_info=True)

    def finish(self) -> None:
        """Close all loggers."""
        if self.wandb:
            self.wandb.finish()
        if self.tb:
            self.tb.close()

    @property
    def is_enabled(self) -> bool:
        return bool(
            (self.wandb and self.wandb.is_enabled) or
            (self.tb and self.tb.is_enabled)
        )