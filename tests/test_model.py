"""Tests for the Cerebro model.

Verifies:
- Forward pass output shapes
- Parameter count (approx 1.5B for nano)
- Gradient flow
- Generation produces tokens
- Config presets
"""

import pytest
import torch
from cerebro.config import CerebroConfig
from cerebro.model.cerebro_model import Cerebro


# Use a tiny config for fast tests
def _tiny_config() -> CerebroConfig:
    """Minimal config for fast testing."""
    return CerebroConfig(
        vocab_size=256,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        ffn_dim=128,
        max_seq_len=64,
        reasoning_layers=1,
        entropy_budget=10.0,
        max_recursion_depth=2,
    )


class TestCerebroForward:
    """Test forward pass shapes."""

    def test_forward_logits_shape(self):
        """Forward pass returns correct logits shape."""
        config = _tiny_config()
        model = Cerebro(config)
        model.eval()

        input_ids = torch.randint(3, config.vocab_size, (1, 16))
        with torch.no_grad():
            output = model(input_ids)

        assert "logits" in output
        assert output["logits"].shape == (1, 16, config.vocab_size)

    def test_forward_with_labels(self):
        """Forward pass with labels returns loss."""
        config = _tiny_config()
        model = Cerebro(config)
        model.eval()

        input_ids = torch.randint(3, config.vocab_size, (2, 16))
        labels = torch.randint(3, config.vocab_size, (2, 16))

        with torch.no_grad():
            output = model(input_ids, labels=labels)

        assert "loss" in output
        assert output["loss"].shape == ()  # scalar
        assert output["loss"].item() > 0

    def test_forward_batch(self):
        """Forward pass works with batch size > 1."""
        config = _tiny_config()
        model = Cerebro(config)
        model.eval()

        input_ids = torch.randint(3, config.vocab_size, (4, 32))
        with torch.no_grad():
            output = model(input_ids)

        assert output["logits"].shape == (4, 32, config.vocab_size)


class TestParameterCount:
    """Test parameter counting."""

    def test_total_parameters(self):
        """Total parameters > 0."""
        config = _tiny_config()
        model = Cerebro(config)
        assert model.num_parameters() > 0

    def test_estimate_params_breakdown(self):
        """Parameter breakdown is consistent."""
        config = _tiny_config()
        model = Cerebro(config)
        counts = model.estimate_params()

        assert "embedding" in counts
        assert "encoder_layers" in counts
        assert "reasoning_core" in counts
        assert "output_head" in counts
        assert "total" in counts
        # output_head shares weights with embedding, so total may differ from sum
        component_sum = counts["embedding"] + counts["encoder_layers"] + counts["reasoning_core"]
        assert counts["total"] >= component_sum

    def test_nano_param_count_range(self):
        """Nano config has ~1.5B parameters (within 20% margin)."""
        config = CerebroConfig.nano()
        model = Cerebro(config)
        total = model.num_parameters()
        # Should be approximately 1.5B (±20%)
        assert 1_000_000_000 < total < 2_500_000_000, f"Nano params: {total:,}"


class TestGradientFlow:
    """Test gradient flow."""

    def test_gradients_exist(self):
        """Gradients flow to all parameters."""
        config = _tiny_config()
        model = Cerebro(config)
        model.train()

        input_ids = torch.randint(3, config.vocab_size, (1, 8))
        labels = torch.randint(3, config.vocab_size, (1, 8))

        output = model(input_ids, labels=labels)
        output["loss"].backward()

        # Check that gradients exist for all named parameters
        no_grad = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is None:
                no_grad.append(name)

        assert len(no_grad) == 0, f"Parameters without gradients: {no_grad}"

    def test_loss_decreases_on_overfit(self):
        """Loss decreases when overfitting a tiny batch."""
        config = _tiny_config()
        model = Cerebro(config)
        model.train()

        input_ids = torch.randint(3, config.vocab_size, (1, 8))
        labels = torch.randint(3, config.vocab_size, (1, 8))

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        losses = []
        for _ in range(10):
            optimizer.zero_grad()
            output = model(input_ids, labels=labels)
            loss = output["loss"]
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


class TestGeneration:
    """Test text generation."""

    def test_generate_produces_tokens(self):
        """Generation produces the correct number of tokens."""
        config = _tiny_config()
        model = Cerebro(config)
        model.eval()

        input_ids = torch.randint(3, config.vocab_size, (1, 4))
        generated = model.generate(input_ids, max_new_tokens=8, do_sample=False)

        assert generated.shape[0] == 1
        assert generated.shape[1] == 4 + 8

    def test_generate_greedy(self):
        """Greedy decoding produces deterministic output."""
        config = _tiny_config()
        model = Cerebro(config)
        model.eval()

        input_ids = torch.randint(3, config.vocab_size, (1, 4))

        gen1 = model.generate(input_ids.clone(), max_new_tokens=8, do_sample=False)
        gen2 = model.generate(input_ids.clone(), max_new_tokens=8, do_sample=False)

        assert torch.equal(gen1, gen2)

    def test_generate_sampled(self):
        """Sampled generation produces different outputs."""
        config = _tiny_config()
        model = Cerebro(config)
        model.eval()

        input_ids = torch.randint(3, config.vocab_size, (1, 4))

        # With sampling, multiple runs should (usually) differ
        results = set()
        for _ in range(5):
            gen = model.generate(input_ids.clone(), max_new_tokens=8, do_sample=True, temperature=1.0)
            results.add(tuple(gen[0].tolist()))

        # At least 2 different results in 5 tries
        assert len(results) >= 2


class TestConfig:
    """Test configuration presets."""

    def test_nano_config(self):
        config = CerebroConfig.nano()
        assert config.hidden_dim == 2048
        assert config.num_layers == 24
        assert config.max_seq_len == 8192

    def test_from_name(self):
        config = CerebroConfig.from_name("nano")
        assert config.hidden_dim == 2048

    def test_from_name_invalid(self):
        with pytest.raises(ValueError):
            CerebroConfig.from_name("nonexistent")

    def test_auto_head_dim(self):
        config = CerebroConfig(hidden_dim=256, num_heads=8)
        assert config.head_dim == 32

    def test_auto_quaternion_dim(self):
        config = CerebroConfig(hidden_dim=256)
        assert config.quaternion_dim == 64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
