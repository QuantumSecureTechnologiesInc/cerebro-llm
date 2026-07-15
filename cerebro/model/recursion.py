"""Bounded Recursion Controller — inference-time self-correction.

Tracks cumulative entropy budget across generation steps and decides
whether to accept the current output or refine further. This is a
generation-time feature, NOT part of the training forward pass.

The model learns to reason via standard next-token prediction during
training. At inference, the recursion controller can trigger additional
"thinking" passes when confidence is low.
"""

from __future__ import annotations

import torch
from torch import Tensor


class BoundedRecursionController:
    """Controls recursive refinement during generation.

    Attributes:
        max_depth: Maximum number of recursion steps.
        entropy_budget: Total entropy budget for a generation sequence.
        verification_threshold: Minimum confidence to accept output.
    """

    def __init__(
        self,
        max_depth: int = 5,
        entropy_budget: float = 100.0,
        verification_threshold: float = 0.85,
    ) -> None:
        self.max_depth = max_depth
        self.entropy_budget = entropy_budget
        self.verification_threshold = verification_threshold
        self.reset()

    def reset(self) -> None:
        """Reset controller state for a new generation sequence."""
        self.current_depth = 0
        self.used_entropy = 0.0
        self.step_count = 0

    def step(self, step_entropy: float) -> None:
        """Record one generation step."""
        self.used_entropy += step_entropy
        self.step_count += 1

    def can_recurse(self, step_entropy: float) -> bool:
        """Check if another recursion step is allowed.

        Args:
            step_entropy: Entropy of the current generation step.

        Returns:
            True if recursion is permitted (within budget and depth).
        """
        self.step(step_entropy)

        if self.current_depth >= self.max_depth:
            return False
        if self.used_entropy >= self.entropy_budget:
            return False
        return True

    def should_refine(self, confidence: float) -> bool:
        """Decide whether to refine the current output.

        Args:
            confidence: Model's self-assessed confidence [0, 1].

        Returns:
            True if the output should be refined (confidence below threshold).
        """
        return confidence < self.verification_threshold

    def get_remaining_budget(self) -> float:
        """Return remaining entropy budget."""
        return max(0.0, self.entropy_budget - self.used_entropy)

    def get_budget_fraction(self) -> float:
        """Return fraction of budget used [0, 1]."""
        return min(1.0, self.used_entropy / self.entropy_budget)


class SelfVerificationModule:
    """Lightweight self-verification for generated tokens.

    Computes a confidence score from the model's output distribution
    that the recursion controller uses to decide whether to refine.
    """

    @staticmethod
    def compute_confidence(logits: Tensor) -> float:
        """Compute confidence score from output logits.

        Uses the entropy of the softmax distribution as an inverse
        confidence measure: low entropy = high confidence.

        Args:
            logits: (vocab_size,) raw logits for the next token.

        Returns:
            Confidence score in [0, 1].
        """
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * (probs + 1e-10).log()).sum().item()
        max_entropy = torch.tensor(logits.shape[-1]).log().item()
        # Normalize to [0, 1]: 1.0 = very confident, 0.0 = uniform
        confidence = 1.0 - (entropy / max_entropy)
        return max(0.0, min(1.0, confidence))

    @staticmethod
    def compute_top_k_confidence(logits: Tensor, k: int = 5) -> float:
        """Confidence from top-k probability mass."""
        probs = torch.softmax(logits, dim=-1)
        top_k_probs = torch.topk(probs, k).values
        return top_k_probs.sum().item()
