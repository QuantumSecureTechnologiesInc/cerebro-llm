"""BPE Tokenizer for Cerebro.

Wraps tiktoken for fast BPE tokenization with Cerebro-specific
special tokens for reasoning, verification, and control.
"""

from __future__ import annotations

import os
import json
from typing import Optional
from pathlib import Path


# Special tokens for Cerebro
SPECIAL_TOKENS = [
    "<|endoftext|>",      # EOS / padding
    "<|beginoftext|>",    # BOS
    "<|reasoning|>",      # Start reasoning chain
    "<|verified|>",       # Self-verification marker
    "<|think|>",          # Thinking tag open
    "<|/think|>",         # Thinking tag close
    "<|answer|>",         # Answer start
    "<|code|>",           # Code block start
    "<|/code|>",          # Code block end
    "<|system|>",         # System prompt marker
    "<|user|>",           # User message marker
    "<|assistant|>",      # Assistant message marker
]


class CerebroTokenizer:
    """BPE tokenizer with Cerebro special tokens.

    Uses tiktoken's cl100k_base encoding as a foundation, extended with
    Cerebro-specific special tokens for reasoning and verification.

    Attributes:
        vocab_size: Total vocabulary size including special tokens.
        special_tokens: Dict mapping token strings to IDs.
    """

    def __init__(self, vocab_size: int = 128_000, model_path: Optional[str] = None) -> None:
        self._vocab_size = vocab_size
        self._model_path = model_path
        self._encoder = None
        self._special_tokens = {}
        self._setup()

    def _setup(self) -> None:
        """Initialize tokenizer with tiktoken base + special tokens."""
        try:
            import tiktoken
            # Use cl100k_base as foundation (100K tokens)
            self._encoder = tiktoken.get_encoding("cl100k_base")
            base_vocab_size = self._encoder.max_token_value + 1

            # Map special tokens to IDs above base vocab
            for i, token in enumerate(SPECIAL_TOKENS):
                self._special_tokens[token] = base_vocab_size + i

            self._base_vocab_size = base_vocab_size
            self._total_vocab_size = base_vocab_size + len(SPECIAL_TOKENS)

        except ImportError:
            # Fallback: simple character-level tokenizer
            self._encoder = None
            self._base_vocab_size = 256
            self._total_vocab_size = 256 + len(SPECIAL_TOKENS)
            for i, token in enumerate(SPECIAL_TOKENS):
                self._special_tokens[token] = 256 + i

    @property
    def vocab_size(self) -> int:
        return self._total_vocab_size

    @property
    def special_tokens(self) -> dict[str, int]:
        return dict(self._special_tokens)

    @property
    def pad_token_id(self) -> int:
        return self._special_tokens["<|endoftext|>"]

    @property
    def bos_token_id(self) -> int:
        return self._special_tokens["<|beginoftext|>"]

    @property
    def eos_token_id(self) -> int:
        return self._special_tokens["<|endoftext|>"]

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Input text string
            add_bos: Prepend begin-of-text token
            add_eos: Append end-of-text token

        Returns:
            List of token IDs
        """
        tokens = []
        if add_bos:
            tokens.append(self.bos_token_id)

        if self._encoder is not None:
            # Check for special tokens in text and split around them
            remaining = text
            while remaining:
                # Find next special token
                found = False
                for st in SPECIAL_TOKENS:
                    idx = remaining.find(st)
                    if idx == 0:
                        tokens.append(self._special_tokens[st])
                        remaining = remaining[len(st):]
                        found = True
                        break
                    elif idx > 0:
                        # Encode text before special token
                        tokens.extend(self._encoder.encode(remaining[:idx]))
                        tokens.append(self._special_tokens[st])
                        remaining = remaining[idx + len(st):]
                        found = True
                        break
                if not found:
                    tokens.extend(self._encoder.encode(remaining))
                    break
        else:
            # Fallback: byte-level encoding
            tokens.extend(list(text.encode("utf-8")))

        if add_eos:
            tokens.append(self.eos_token_id)

        return tokens

    def decode(self, token_ids: list[int], skip_special: bool = True) -> str:
        """Decode token IDs back to text.

        Args:
            token_ids: List of token IDs
            skip_special: If True, omit special tokens from output

        Returns:
            Decoded text string
        """
        # Reverse special token mapping
        id_to_special = {v: k for k, v in self._special_tokens.items()}

        text_parts = []
        regular_ids = []

        for tid in token_ids:
            if tid in id_to_special:
                # Flush accumulated regular tokens
                if regular_ids:
                    if self._encoder is not None:
                        text_parts.append(self._encoder.decode(regular_ids))
                    else:
                        text_parts.append(bytes(regular_ids).decode("utf-8", errors="replace"))
                    regular_ids = []
                if not skip_special:
                    text_parts.append(id_to_special[tid])
            else:
                regular_ids.append(tid)

        # Flush remaining regular tokens
        if regular_ids:
            if self._encoder is not None:
                text_parts.append(self._encoder.decode(regular_ids))
            else:
                text_parts.append(bytes(regular_ids).decode("utf-8", errors="replace"))

        return "".join(text_parts)

    def save(self, path: str) -> None:
        """Save tokenizer config (special token mappings)."""
        config = {
            "vocab_size": self._total_vocab_size,
            "base_vocab_size": self._base_vocab_size,
            "special_tokens": self._special_tokens,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, path: str) -> CerebroTokenizer:
        """Load tokenizer from saved config."""
        with open(path) as f:
            config = json.load(f)
        tok = cls.__new__(cls)
        tok._special_tokens = config["special_tokens"]
        tok._base_vocab_size = config["base_vocab_size"]
        tok._total_vocab_size = config["vocab_size"]
        tok._vocab_size = config["vocab_size"]
        tok._model_path = None
        tok._encoder = None
        # Re-initialize tiktoken
        try:
            import tiktoken
            tok._encoder = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            pass
        return tok
