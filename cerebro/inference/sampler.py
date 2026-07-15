"""Sampling strategies for text generation.

Supports greedy, temperature, top-p (nucleus), top-k, and
repetition penalty.
"""

from __future__ import annotations

import torch
from torch import Tensor


class Sampler:
    """Token sampler with multiple decoding strategies.

    Args:
        temperature: Sampling temperature (>0). Lower = more deterministic.
        top_p: Nucleus sampling probability mass cutoff [0, 1].
        top_k: Top-k sampling cutoff (0 = disabled).
        repetition_penalty: Penalty factor for previously generated tokens.
    """

    def __init__(
        self,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
    ) -> None:
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty

    @torch.no_grad()
    def sample(
        self,
        logits: Tensor,
        generated_tokens: list[int] | None = None,
        do_sample: bool = True,
    ) -> int:
        """Sample the next token from logits.

        Args:
            logits: (vocab_size,) raw logits for the next token.
            generated_tokens: List of previously generated token IDs.
            do_sample: If False, use greedy decoding.

        Returns:
            Sampled token ID.
        """
        if not do_sample:
            return logits.argmax(dim=-1).item()

        # Apply repetition penalty
        if generated_tokens and self.repetition_penalty != 1.0:
            for token_id in set(generated_tokens):
                if logits[token_id] > 0:
                    logits[token_id] /= self.repetition_penalty
                else:
                    logits[token_id] *= self.repetition_penalty

        # Temperature scaling
        logits = logits / max(self.temperature, 1e-8)
        probs = torch.softmax(logits, dim=-1)

        # Top-k filtering
        if self.top_k > 0:
            top_k_val = torch.topk(probs, min(self.top_k, probs.shape[-1])).values[-1]
            probs = probs.masked_fill(probs < top_k_val, 0.0)

        # Top-p (nucleus) filtering
        if self.top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative = sorted_probs.cumsum(dim=-1)
            sorted_mask = cumulative > self.top_p
            sorted_mask[1:] = sorted_mask[:-1].clone()
            sorted_mask[0] = False
            probs[sorted_indices[sorted_mask]] = 0.0

        # Renormalize
        probs = probs / (probs.sum() + 1e-8)
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token.item()

    @torch.no_grad()
    def sample_batch(
        self,
        logits: Tensor,
        generated_tokens: list[list[int]] | None = None,
        do_sample: bool = True,
    ) -> Tensor:
        """Sample next tokens for a batch.

        Args:
            logits: (B, vocab_size) raw logits.
            generated_tokens: List of lists of previously generated IDs per batch.
            do_sample: If False, use greedy decoding.

        Returns:
            (B, 1) sampled token IDs.
        """
        B = logits.shape[0]
        if not do_sample:
            return logits.argmax(dim=-1, keepdim=True)

        # Apply repetition penalty per batch element
        if generated_tokens and self.repetition_penalty != 1.0:
            for i in range(B):
                for token_id in set(generated_tokens[i]):
                    if logits[i, token_id] > 0:
                        logits[i, token_id] /= self.repetition_penalty
                    else:
                        logits[i, token_id] *= self.repetition_penalty

        logits = logits / max(self.temperature, 1e-8)
        probs = torch.softmax(logits, dim=-1)

        # Top-k
        if self.top_k > 0:
            top_k_val = torch.topk(probs, min(self.top_k, probs.shape[-1])).values[:, -1:]
            probs = probs.masked_fill(probs < top_k_val, 0.0)

        # Top-p
        if self.top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative = sorted_probs.cumsum(dim=-1)
            sorted_mask = cumulative > self.top_p
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False
            for i in range(B):
                probs[i, sorted_indices[i][sorted_mask[i]]] = 0.0

        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
        return torch.multinomial(probs, num_samples=1)
