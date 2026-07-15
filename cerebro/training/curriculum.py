"""Curriculum learning with progressive sequence length training.

Starts training with short sequences and gradually increases context
length. This approach:
- Speeds up early training (shorter sequences = faster steps)
- Stabilizes learning of positional embeddings
- Enables efficient long-context training (2K → 8K → 32K → 128K)

Usage::

    curriculum = CurriculumScheduler(
        stages=[(2048, 50_000), (8192, 30_000), (32768, 20_000)],
    )
    seq_len = curriculum.get_seq_len(step=0)  # 2048
    seq_len = curriculum.get_seq_len(step=50_000)  # 8192
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CurriculumStage:
    """A single stage in the curriculum."""
    seq_len: int
    num_steps: int
    batch_size: int | None = None  # Override batch size for this stage
    learning_rate: float | None = None  # Override LR for this stage

    def __repr__(self) -> str:
        extras = ""
        if self.batch_size:
            extras += f", bs={self.batch_size}"
        if self.learning_rate:
            extras += f", lr={self.learning_rate:.2e}"
        return f"Stage(seq={self.seq_len}, steps={self.num_steps}{extras})"


class CurriculumScheduler:
    """Progressive sequence length training scheduler.

    Manages training stages where sequence length increases over time.
    Each stage specifies a sequence length and the number of steps to
    train at that length.

    Args:
        stages: List of (seq_len, num_steps) tuples or CurriculumStage objects.
                Must be ordered from shortest to longest.
        warmup_ratio: Fraction of each stage used for LR warmup.
    """

    def __init__(
        self,
        stages: list[tuple[int, int] | CurriculumStage],
        warmup_ratio: float = 0.02,
    ) -> None:
        self.stages: list[CurriculumStage] = []
        for s in stages:
            if isinstance(s, tuple):
                self.stages.append(CurriculumStage(seq_len=s[0], num_steps=s[1]))
            else:
                self.stages.append(s)

        if not self.stages:
            raise ValueError("At least one curriculum stage is required")

        # Validate ordering
        for i in range(1, len(self.stages)):
            if self.stages[i].seq_len < self.stages[i - 1].seq_len:
                raise ValueError(
                    f"Stages must be ordered by increasing seq_len. "
                    f"Stage {i} ({self.stages[i].seq_len}) < "
                    f"Stage {i-1} ({self.stages[i-1].seq_len})"
                )

        self.warmup_ratio = warmup_ratio

        # Pre-compute step boundaries
        self._boundaries: list[int] = []
        cumulative = 0
        for stage in self.stages:
            cumulative += stage.num_steps
            self._boundaries.append(cumulative)

    @property
    def total_steps(self) -> int:
        """Total training steps across all stages."""
        return sum(s.num_steps for s in self.stages)

    @property
    def max_seq_len(self) -> int:
        """Maximum sequence length in the curriculum."""
        return self.stages[-1].seq_len

    @property
    def num_stages(self) -> int:
        return len(self.stages)

    def get_stage(self, step: int) -> int:
        """Get the current stage index for a given step.

        Args:
            step: Current global training step.

        Returns:
            Stage index (0-based).
        """
        for i, boundary in enumerate(self._boundaries):
            if step < boundary:
                return i
        return len(self.stages) - 1

    def get_seq_len(self, step: int) -> int:
        """Get the sequence length for a given training step.

        Args:
            step: Current global training step.

        Returns:
            Sequence length for the current stage.
        """
        stage_idx = self.get_stage(step)
        return self.stages[stage_idx].seq_len

    def get_batch_size(self, step: int, default: int = 4) -> int:
        """Get the batch size for a given step.

        Larger sequence lengths often need smaller batch sizes
        to fit in GPU memory.

        Args:
            step: Current step.
            default: Default batch size if stage doesn't override.

        Returns:
            Batch size for the current stage.
        """
        stage_idx = self.get_stage(step)
        return self.stages[stage_idx].batch_size or default

    def get_learning_rate(
        self,
        step: int,
        base_lr: float = 3e-4,
    ) -> float:
        """Get learning rate with per-stage warmup.

        Each stage gets a small warmup when transitioning to a new
        sequence length, then decays within the stage.

        Args:
            step: Current step.
            base_lr: Base (maximum) learning rate.

        Returns:
            Learning rate for the current step.
        """
        import math

        stage_idx = self.get_stage(step)
        stage = self.stages[stage_idx]

        if stage.learning_rate is not None:
            base_lr = stage.learning_rate

        # Steps within current stage
        stage_start = self._boundaries[stage_idx - 1] if stage_idx > 0 else 0
        steps_in_stage = step - stage_start
        warmup_steps = int(stage.num_steps * self.warmup_ratio)

        # Warmup at start of each stage
        if steps_in_stage < warmup_steps:
            return base_lr * (steps_in_stage + 1) / max(warmup_steps, 1)

        # Cosine decay within stage
        progress = (steps_in_stage - warmup_steps) / max(stage.num_steps - warmup_steps, 1)
        min_lr = base_lr * 0.1
        return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

    def should_transition(self, step: int) -> bool:
        """Check if this step is a stage boundary (transition point).

        Useful for triggering data loader recreation when seq_len changes.

        Args:
            step: Current step.

        Returns:
            True if this step is the first step of a new stage.
        """
        if step == 0:
            return True
        for boundary in self._boundaries[:-1]:
            if step == boundary:
                return True
        return False

    def progress(self, step: int) -> float:
        """Get overall training progress as a fraction (0.0 to 1.0).

        Args:
            step: Current step.

        Returns:
            Progress fraction.
        """
        return min(step / max(self.total_steps, 1), 1.0)

    def summary(self) -> str:
        """Get a human-readable curriculum summary."""
        lines = ["Curriculum Schedule:"]
        lines.append(f"  Total steps: {self.total_steps:,}")
        lines.append(f"  Max seq len: {self.max_seq_len:,}")
        lines.append(f"  Stages: {self.num_stages}")
        lines.append("")

        cumulative = 0
        for i, stage in enumerate(self.stages):
            end = cumulative + stage.num_steps
            extras = ""
            if stage.batch_size:
                extras += f" bs={stage.batch_size}"
            if stage.learning_rate:
                extras += f" lr={stage.learning_rate:.2e}"
            lines.append(
                f"  Stage {i+1}: seq_len={stage.seq_len:>7,}  "
                f"steps={cumulative:>8,}→{end:>8,} "
                f"({stage.num_steps:,} steps){extras}"
            )
            cumulative = end

        return "\n".join(lines)

    @classmethod
    def from_preset(
        cls,
        model_tier: str = "nano",
        custom_stages: list[tuple[int, int]] | None = None,
    ) -> CurriculumScheduler:
        """Create a curriculum from model tier presets.

        Args:
            model_tier: Model size ('nano', 'core', 'pro', 'ultra', 'sovereign').
            custom_stages: Override with custom stages.

        Returns:
            CurriculumScheduler.
        """
        if custom_stages:
            return cls(custom_stages)

        presets = {
            "nano": [
                (2048, 60_000),
                (4096, 30_000),
                (8192, 10_000),
            ],
            "core": [
                (2048, 80_000),
                (4096, 40_000),
                (8192, 30_000),
                (16384, 20_000),
                (32768, 10_000),
            ],
            "pro": [
                (2048, 100_000),
                (4096, 50_000),
                (8192, 40_000),
                (16384, 30_000),
                (32768, 20_000),
                (65536, 10_000),
            ],
            "ultra": [
                (2048, 120_000),
                (4096, 60_000),
                (8192, 50_000),
                (16384, 40_000),
                (32768, 30_000),
                (65536, 20_000),
                (131072, 10_000),
            ],
            "sovereign": [
                (2048, 200_000),
                (4096, 100_000),
                (8192, 80_000),
                (16384, 60_000),
                (32768, 50_000),
                (65536, 40_000),
                (131072, 30_000),
                (262144, 10_000),
            ],
        }

        tier = model_tier.lower()
        if tier not in presets:
            raise ValueError(f"Unknown tier '{tier}'. Choose from: {list(presets)}")

        return cls(presets[tier])
