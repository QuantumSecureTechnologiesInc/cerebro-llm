"""Tests for the Cerebro tokenizer.

Verifies:
- Encode/decode round-trip
- Special token handling
- BOS/EOS insertion
- Vocabulary size
"""

import pytest
from cerebro.tokenizer.tokenizer import CerebroTokenizer, SPECIAL_TOKENS


class TestTokenizerBasic:
    """Basic tokenizer operations."""

    def test_instantiation(self):
        """Tokenizer can be created."""
        tok = CerebroTokenizer()
        assert tok.vocab_size > 0

    def test_encode_produces_list(self):
        """Encode returns a list of integers."""
        tok = CerebroTokenizer()
        tokens = tok.encode("Hello, world!")
        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)
        assert len(tokens) > 0

    def test_decode_produces_string(self):
        """Decode returns a string."""
        tok = CerebroTokenizer()
        tokens = tok.encode("Hello, world!")
        text = tok.decode(tokens)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_round_trip(self):
        """Encode then decode should approximately recover the original text."""
        tok = CerebroTokenizer()
        original = "The quick brown fox jumps over the lazy dog."
        tokens = tok.encode(original)
        recovered = tok.decode(tokens)
        assert recovered == original

    def test_round_trip_unicode(self):
        """Unicode text round-trips correctly."""
        tok = CerebroTokenizer()
        original = "Hello 世界! Ça va?"
        tokens = tok.encode(original)
        recovered = tok.decode(tokens)
        assert recovered == original


class TestSpecialTokens:
    """Special token handling."""

    def test_special_tokens_exist(self):
        """All expected special tokens are present."""
        tok = CerebroTokenizer()
        for st in SPECIAL_TOKENS:
            assert st in tok.special_tokens

    def test_special_token_ids_unique(self):
        """All special token IDs are unique."""
        tok = CerebroTokenizer()
        ids = list(tok.special_tokens.values())
        assert len(ids) == len(set(ids))

    def test_encode_special_token(self):
        """Encoding text with a special token preserves it."""
        tok = CerebroTokenizer()
        tokens = tok.encode("<|reasoning|>Hello")
        assert tok.special_tokens["<|reasoning|>"] in tokens

    def test_decode_skip_special(self):
        """Decoding with skip_special=True omits special tokens."""
        tok = CerebroTokenizer()
        tokens = tok.encode("<|reasoning|>Hello world", add_bos=True)
        text_skip = tok.decode(tokens, skip_special=True)
        text_keep = tok.decode(tokens, skip_special=False)

        assert "<|reasoning|>" not in text_skip
        assert "<|reasoning|>" in text_keep

    def test_pad_token_id(self):
        """Pad token ID is defined."""
        tok = CerebroTokenizer()
        assert tok.pad_token_id == tok.special_tokens["<|endoftext|>"]

    def test_bos_token_id(self):
        """BOS token ID is defined."""
        tok = CerebroTokenizer()
        assert tok.bos_token_id == tok.special_tokens["<|beginoftext|>"]

    def test_eos_token_id(self):
        """EOS token ID is defined."""
        tok = CerebroTokenizer()
        assert tok.eos_token_id == tok.special_tokens["<|endoftext|>"]


class TestBosEos:
    """BOS and EOS token insertion."""

    def test_add_bos(self):
        """add_bos prepends the BOS token."""
        tok = CerebroTokenizer()
        tokens = tok.encode("Hello", add_bos=True)
        assert tokens[0] == tok.bos_token_id

    def test_add_eos(self):
        """add_eos appends the EOS token."""
        tok = CerebroTokenizer()
        tokens = tok.encode("Hello", add_eos=True)
        assert tokens[-1] == tok.eos_token_id

    def test_add_both(self):
        """Both BOS and EOS can be added."""
        tok = CerebroTokenizer()
        tokens = tok.encode("Hello", add_bos=True, add_eos=True)
        assert tokens[0] == tok.bos_token_id
        assert tokens[-1] == tok.eos_token_id

    def test_no_bos_by_default(self):
        """BOS is not added by default."""
        tok = CerebroTokenizer()
        tokens_with = tok.encode("Hello", add_bos=True)
        tokens_without = tok.encode("Hello")
        assert len(tokens_with) == len(tokens_without) + 1


class TestSaveLoad:
    """Save and load tokenizer config."""

    def test_save_load_round_trip(self, tmp_path):
        """Saving and loading preserves vocabulary size."""
        tok = CerebroTokenizer()
        path = str(tmp_path / "tokenizer.json")
        tok.save(path)

        loaded = CerebroTokenizer.load(path)
        assert loaded.vocab_size == tok.vocab_size

    def test_loaded_special_tokens(self, tmp_path):
        """Loaded tokenizer preserves special token mappings."""
        tok = CerebroTokenizer()
        path = str(tmp_path / "tokenizer.json")
        tok.save(path)

        loaded = CerebroTokenizer.load(path)
        assert loaded.special_tokens == tok.special_tokens


class TestEdgeCases:
    """Edge cases."""

    def test_empty_string(self):
        """Encoding empty string returns empty list."""
        tok = CerebroTokenizer()
        tokens = tok.encode("")
        assert tokens == []

    def test_long_text(self):
        """Encoding long text works."""
        tok = CerebroTokenizer()
        text = "word " * 1000
        tokens = tok.encode(text)
        assert len(tokens) > 0

    def test_decode_empty(self):
        """Decoding empty list returns empty string."""
        tok = CerebroTokenizer()
        text = tok.decode([])
        assert text == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
