"""Cerebro — full model assembly.

The complete Cerebro language model with:
- Token embedding
- Stack of CerebroBlock transformer layers
- Reasoning core (self-verification stack)
- Output head with logit soft-capping
- forward() for training, generate() for inference
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from cerebro.config import CerebroConfig
from cerebro.model.block import CerebroBlock
from cerebro.model.norm import RMSNorm
from cerebro.model.reasoning import ReasoningCore
from cerebro.model.recursion import BoundedRecursionController, SelfVerificationModule
from cerebro.model.attention import KVCache

# Lazy imports for vision (avoids circular deps and heavy torchvision import)
_VISION_AVAILABLE = False


def _ensure_vision() -> None:
    """Lazy-import vision module on first use."""
    global _VISION_AVAILABLE
    if not _VISION_AVAILABLE:
        try:
            from cerebro.vision import VisionEncoder, VisionTextFusion
            _VISION_AVAILABLE = True
        except ImportError:
            pass


class Cerebro(nn.Module):
    """Cerebro: Cognitive Entropic Reasoning Engine with Bounded Recursive Optimization.

    A next-generation Large Language Model with Hybrid Transformer-Quaternion Architecture,
    entropic gating, bounded recursion, and a reasoning core for self-verification.
    """

    def __init__(self, config: CerebroConfig) -> None:
        super().__init__()
        self.config = config

        # ── Token Embedding ──
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_dim,
            padding_idx=config.pad_token_id,
        )

        # ── Encoder Stack ──
        self.layers = nn.ModuleList([
            CerebroBlock(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads,
                head_dim=config.head_dim,
                ffn_dim=config.ffn_dim,
                max_seq_len=config.max_seq_len,
                rope_theta=config.rope_theta,
                entropy_min=config.entropy_min,
                entropy_max=config.entropy_max,
                init_temperature=config.init_temperature,
            )
            for _ in range(config.num_layers)
        ])

        # ── Final Norm ──
        self.norm = RMSNorm(config.hidden_dim)

        # ── Reasoning Core ──
        self.reasoning_core = ReasoningCore(
            num_layers=config.reasoning_layers,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            ffn_dim=config.ffn_dim,
            max_seq_len=config.max_seq_len,
            rope_theta=config.rope_theta,
            entropy_min=config.entropy_min,
            entropy_max=config.entropy_max,
            init_temperature=config.init_temperature,
        )

        # ── Output Head ──
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # ── Logit soft-cap scale ──
        self.logit_soft_cap = config.logit_soft_cap

        # ── Recursion controller (inference only) ──
        self.recursion_controller = BoundedRecursionController(
            max_depth=config.max_recursion_depth,
            entropy_budget=config.entropy_budget,
            verification_threshold=config.verification_threshold,
        )

        # ── Vision encoder (multimodal support) ──
        self.vision_encoder = None
        self.vision_fusion = None
        self._vision_initialized = False

        # Initialize weights
        self.apply(self._init_weights)

        # Tie embeddings (input embedding = output projection weight)
        self.lm_head.weight = self.embed_tokens.weight

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with careful attention to residual scaling."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _init_vision(self) -> None:
        """Lazy-initialize vision encoder and fusion module."""
        if self._vision_initialized:
            return
        _ensure_vision()
        if not _VISION_AVAILABLE:
            return
        from cerebro.vision import VisionEncoder, VisionTextFusion
        self.vision_encoder = VisionEncoder(
            embed_dim=self.config.hidden_dim,
            num_heads=self.config.num_heads,
        )
        self.vision_fusion = VisionTextFusion(
            hidden_dim=self.config.hidden_dim,
        )
        self._vision_initialized = True

    def _build_causal_mask(self, seq_len: int, device: torch.device) -> Tensor:
        """Build causal attention mask."""
        mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )
        return mask

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor | None = None,
        kv_caches: list[KVCache] | None = None,
        images: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Training or inference forward pass.

        Args:
            input_ids: (B, S) token indices
            attention_mask: (B, S) optional padding mask (1=keep, 0=pad)
            position_ids: (B, S) optional position indices
            labels: (B, S) optional target token indices for loss computation
            kv_caches: optional list of KVCache per layer for autoregressive decode
            images: (B, C, H, W) optional image tensor for multimodal input

        Returns:
            dict with 'logits' (B, S, vocab_size) and optionally 'loss'
        """
        B, S = input_ids.shape

        # Embed tokens
        x = self.embed_tokens(input_ids)  # (B, S, hidden_dim)

        # ── Vision input: encode and fuse with text embeddings ──
        if images is not None:
            self._init_vision()
            if self.vision_encoder is not None:
                vision_embeddings = self.vision_encoder(images)
                x = self.vision_fusion(x, vision_embeddings)
                S = x.shape[1]  # sequence length increased by vision tokens

        # Causal mask
        causal_mask = self._build_causal_mask(S, x.device)

        # Encoder stack
        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches else None
            x = layer(x, mask=causal_mask, position_ids=position_ids, kv_cache=cache)

        # Final norm
        x = self.norm(x)

        # Reasoning core
        x = self.reasoning_core(x, mask=causal_mask)

        # Logits with soft-capping (Gemma-style)
        logits = self.lm_head(x)
        logits = self.logit_soft_cap * torch.tanh(logits / self.logit_soft_cap)

        result = {"logits": logits}

        # Compute loss if labels provided
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id,
            )
            result["loss"] = loss

        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        images: Tensor | None = None,
    ) -> Tensor:
        """Autoregressive text generation.

        Args:
            input_ids: (B, S) prompt token indices
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature
            top_p: nucleus sampling threshold
            top_k: top-k sampling cutoff
            repetition_penalty: penalty factor for repeated tokens
            do_sample: if False, use greedy decoding
            images: (B, C, H, W) optional image tensor for multimodal generation

        Returns:
            (B, S + max_new_tokens) generated token indices
        """
        self.eval()
        self.recursion_controller.reset()
        generated = input_ids.clone()
        B = input_ids.shape[0]

        for step in range(max_new_tokens):
            # Forward pass on the full sequence
            # For efficiency, we could use KV-cache (see inference/engine.py)
            # Here we use the simple full-sequence approach
            seq_len = generated.shape[1]
            if seq_len > self.config.max_seq_len:
                # Truncate from the left
                input_chunk = generated[:, -self.config.max_seq_len:]
            else:
                input_chunk = generated

            output = self.forward(input_chunk, images=images if step == 0 else None)
            next_logits = output["logits"][:, -1, :]  # (B, vocab_size)

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(B):
                    for token_id in generated[i].unique():
                        if next_logits[i, token_id] > 0:
                            next_logits[i, token_id] /= repetition_penalty
                        else:
                            next_logits[i, token_id] *= repetition_penalty

            # Sample or greedy
            if do_sample:
                next_logits = next_logits / max(temperature, 1e-8)
                probs = F.softmax(next_logits, dim=-1)

                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = probs < torch.topk(probs, top_k).values[:, -1:]
                    probs[indices_to_remove] = 0.0

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative = sorted_probs.cumsum(dim=-1)
                    # Remove tokens above top_p
                    sorted_mask = cumulative > top_p
                    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                    sorted_mask[..., 0] = False
                    for i in range(B):
                        probs[i, sorted_indices[i][sorted_mask[i]]] = 0.0

                # Renormalize
                probs = probs / probs.sum(dim=-1, keepdim=True)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS
            if (next_token == self.config.eos_token_id).all():
                break

        return generated

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def estimate_params(self) -> dict[str, int]:
        """Detailed parameter count breakdown."""
        counts = {}
        counts["embedding"] = sum(p.numel() for p in self.embed_tokens.parameters())
        counts["encoder_layers"] = sum(p.numel() for p in self.layers.parameters())
        counts["reasoning_core"] = sum(p.numel() for p in self.reasoning_core.parameters())
        counts["output_head"] = sum(p.numel() for p in self.lm_head.parameters())
        counts["total"] = self.num_parameters()
        return counts
