"""Tests for quaternion algebra (quaternion.py).

Verifies:
- Quaternion multiplication properties
- Conjugate properties
- Norm properties
- QLinear forward pass shapes
"""

import pytest
import torch
from cerebro.model.quaternion import qmul, qconj, qnorm, qnormalize, real_part, QLinear


class TestQuaternionMultiplication:
    """Test quaternion multiplication (Hamilton product)."""

    def test_identity_multiplication(self):
        """q * 1 = q for any quaternion q."""
        q = torch.randn(2, 4, 4)  # (B, S, 4)
        identity = torch.zeros_like(q)
        identity[..., 0] = 1.0  # (1, 0, 0, 0)

        result = qmul(q, identity)
        assert torch.allclose(result, q, atol=1e-6)

    def test_conjugate_product_is_norm_squared(self):
        """q * conj(q) = |q|^2 for any quaternion q."""
        q = torch.randn(2, 4, 4)
        conj_q = qconj(q)
        product = qmul(q, conj_q)

        # Real part should be |q|^2
        expected_norm_sq = qnorm(q).squeeze(-1) ** 2
        assert torch.allclose(product[..., 0], expected_norm_sq, atol=1e-5)

        # Imaginary parts should be zero
        assert torch.allclose(product[..., 1:], torch.zeros_like(product[..., 1:]), atol=1e-5)

    def test_non_commutativity(self):
        """q1 * q2 != q2 * q1 in general."""
        q1 = torch.randn(2, 4, 4)
        q2 = torch.randn(2, 4, 4)

        prod1 = qmul(q1, q2)
        prod2 = qmul(q2, q1)

        # They should NOT be equal
        assert not torch.allclose(prod1, prod2, atol=1e-6)

    def test_associativity(self):
        """(q1 * q2) * q3 = q1 * (q2 * q3)."""
        q1 = torch.randn(2, 4, 4)
        q2 = torch.randn(2, 4, 4)
        q3 = torch.randn(2, 4, 4)

        left = qmul(qmul(q1, q2), q3)
        right = qmul(q1, qmul(q2, q3))

        assert torch.allclose(left, right, atol=1e-5)

    def test_batch_dimensions(self):
        """Multiplication preserves batch dimensions."""
        q1 = torch.randn(3, 8, 4)
        q2 = torch.randn(3, 8, 4)
        result = qmul(q1, q2)
        assert result.shape == (3, 8, 4)


class TestQuaternionConjugate:
    """Test quaternion conjugate."""

    def test_double_conjugate(self):
        """conj(conj(q)) = q."""
        q = torch.randn(2, 4, 4)
        assert torch.allclose(qconj(qconj(q)), q)

    def test_conjugate_flips_imaginary(self):
        """Conjugate negates i, j, k components."""
        q = torch.randn(2, 4, 4)
        conj_q = qconj(q)
        assert torch.allclose(conj_q[..., 0], q[..., 0])
        assert torch.allclose(conj_q[..., 1:], -q[..., 1:])


class TestQuaternionNorm:
    """Test quaternion norm."""

    def test_norm_non_negative(self):
        """Norm is always non-negative."""
        q = torch.randn(10, 4)
        norms = qnorm(q).squeeze(-1)
        assert (norms >= 0).all()

    def test_norm_zero_for_zero_quaternion(self):
        """Zero quaternion has near-zero norm (epsilon floor)."""
        q = torch.zeros(5, 4)
        norms = qnorm(q).squeeze(-1)
        assert torch.allclose(norms, torch.zeros(5), atol=1e-5)

    def test_normalize_unit_norm(self):
        """Normalized quaternion has unit norm."""
        q = torch.randn(10, 4)
        qn = qnormalize(q)
        norms = qnorm(qn).squeeze(-1)
        assert torch.allclose(norms, torch.ones(10), atol=1e-6)


class TestRealPart:
    """Test real-part extraction."""

    def test_real_part_shape(self):
        """Real part drops the last dimension."""
        q = torch.randn(2, 4, 4)
        r = real_part(q)
        assert r.shape == (2, 4)

    def test_real_part_values(self):
        """Real part is the first component."""
        q = torch.randn(3, 5, 4)
        r = real_part(q)
        assert torch.allclose(r, q[..., 0])


class TestQLinear:
    """Test QLinear (quaternion-valued linear layer)."""

    def test_forward_shape(self):
        """QLinear produces correct output shape."""
        layer = QLinear(in_features=32, out_features=64, bias=False)
        # Input: (B, S, in_features, 4) — quaternion-valued
        x = torch.randn(2, 8, 32, 4)
        out = layer(x)
        assert out.shape == (2, 8, 64, 4)

    def test_forward_with_bias(self):
        """QLinear with bias produces correct shape."""
        layer = QLinear(in_features=16, out_features=32, bias=True)
        x = torch.randn(1, 4, 16, 4)
        out = layer(x)
        assert out.shape == (1, 4, 32, 4)

    def test_parameter_count(self):
        """QLinear has 4x fewer parameters than equivalent real linear."""
        ql = QLinear(in_features=64, out_features=128, bias=False)
        # Should have: 64 * 128 * 4 params (4 quaternion components)
        total = sum(p.numel() for p in ql.parameters())
        assert total == 64 * 128 * 4

    def test_gradient_flow(self):
        """Gradients flow through QLinear."""
        layer = QLinear(in_features=8, out_features=16, bias=True)
        x = torch.randn(1, 4, 8, 4, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
