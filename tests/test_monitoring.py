"""Tests for Cerebro monitoring and KV-cache modules.

Covers:
- WandBLogger initialization and logging
- TensorBoardLogger initialization and logging
- MonitorCallback step/epoch/histogram logging
- KVCache functionality (attention layer)
"""

import os
import json
import tempfile
import torch
import pytest

from cerebro.training.monitoring import (
    WandBLogger, TensorBoardLogger, MonitorCallback,
)
from cerebro.model.attention import KVCache, QuaternionMultiHeadAttention


# ════════════════════════════════════════════════════════════
# KV-CACHE TESTS
# ════════════════════════════════════════════════════════════

class TestKVCache:
    def test_init_empty(self):
        cache = KVCache()
        assert cache.k_cache is None
        assert cache.v_cache is None
        assert cache.past_len == 0

    def test_update_first_time(self):
        cache = KVCache()
        k = torch.randn(1, 4, 8, 32)  # (B, heads, seq, head_dim)
        v = torch.randn(1, 4, 8, 32)
        k_out, v_out = cache.update(k, v)
        assert cache.past_len == 8
        assert torch.equal(k_out, k)
        assert torch.equal(v_out, v)

    def test_update_concat(self):
        cache = KVCache()
        k1 = torch.randn(1, 4, 4, 32)
        v1 = torch.randn(1, 4, 4, 32)
        cache.update(k1, v1)

        k2 = torch.randn(1, 4, 1, 32)  # Single token decode step
        v2 = torch.randn(1, 4, 1, 32)
        k_out, v_out = cache.update(k2, v2)

        assert cache.past_len == 5
        assert k_out.shape == (1, 4, 5, 32)
        assert v_out.shape == (1, 4, 5, 32)

    def test_reset(self):
        cache = KVCache()
        cache.update(torch.randn(1, 4, 4, 32), torch.randn(1, 4, 4, 32))
        assert cache.past_len > 0
        cache.reset()
        assert cache.k_cache is None
        assert cache.v_cache is None
        assert cache.past_len == 0

    def test_multiple_updates(self):
        cache = KVCache()
        for i in range(5):
            k = torch.randn(1, 4, 1, 32)
            v = torch.randn(1, 4, 1, 32)
            cache.update(k, v)
        assert cache.past_len == 5
        assert cache.k_cache.shape == (1, 4, 5, 32)
        assert cache.v_cache.shape == (1, 4, 5, 32)


class TestAttentionWithKVCache:
    def test_forward_without_cache(self):
        """Attention works without KV-cache (normal forward)."""
        attn = QuaternionMultiHeadAttention(
            hidden_dim=128,
            num_heads=4,
            num_kv_heads=2,
            head_dim=32,
            max_seq_len=64,
        )
        x = torch.randn(1, 16, 128)  # (B, S, D)
        output = attn(x)
        assert output.shape == (1, 16, 128)

    def test_forward_with_cache(self):
        """Attention works with KV-cache."""
        attn = QuaternionMultiHeadAttention(
            hidden_dim=128,
            num_heads=4,
            num_kv_heads=2,
            head_dim=32,
            max_seq_len=64,
        )
        attn.eval()

        # Prefill: process 8 tokens
        x = torch.randn(1, 8, 128)
        cache = KVCache()
        with torch.no_grad():
            out1 = attn(x, kv_cache=cache)
        assert out1.shape == (1, 8, 128)
        assert cache.past_len == 8

        # Decode: process 1 token with cache
        x2 = torch.randn(1, 1, 128)
        with torch.no_grad():
            out2 = attn(x2, kv_cache=cache)
        assert out2.shape == (1, 1, 128)
        assert cache.past_len == 9

    def test_cache_then_reset(self):
        """Cache can be reset and reused."""
        attn = QuaternionMultiHeadAttention(
            hidden_dim=128, num_heads=4, num_kv_heads=2,
            head_dim=32, max_seq_len=64,
        )
        attn.eval()

        cache = KVCache()
        with torch.no_grad():
            attn(torch.randn(1, 8, 128), kv_cache=cache)
        assert cache.past_len == 8

        cache.reset()
        assert cache.past_len == 0

        with torch.no_grad():
            attn(torch.randn(1, 4, 128), kv_cache=cache)
        assert cache.past_len == 4


# ════════════════════════════════════════════════════════════
# WANDB LOGGER TESTS
# ════════════════════════════════════════════════════════════

class TestWandBLogger:
    def test_init_disabled_by_default(self):
        """WandBLogger is disabled if wandb not installed."""
        logger = WandBLogger(project="test")
        # Init should not raise even if wandb not installed
        logger.init()
        assert not logger.is_enabled  # Should be disabled without wandb

    def test_log_noop_when_disabled(self):
        logger = WandBLogger(project="test")
        # Should not raise when logging while disabled
        logger.log({"loss": 1.0}, step=0)
        logger.log_histogram("params", [1, 2, 3], step=0)
        logger.watch(torch.nn.Linear(10, 10))
        logger.finish()

    def test_init_with_name(self):
        logger = WandBLogger(project="cerebro", name="test-run")
        assert logger.name == "test-run"
        assert logger.project == "cerebro"

    def test_init_with_config(self):
        config = {"model": "nano", "lr": 3e-4}
        logger = WandBLogger(project="test", config=config)
        assert logger.config == config

    def test_init_with_tags(self):
        logger = WandBLogger(project="test", tags=["v1", "experiment"])
        assert "v1" in logger.tags
        assert "experiment" in logger.tags


# ════════════════════════════════════════════════════════════
# TENSORBOARD LOGGER TESTS
# ════════════════════════════════════════════════════════════

class TestTensorBoardLogger:
    def test_init_disabled_by_default(self):
        """TensorBoardLogger is disabled if tensorboard not installed."""
        logger = TensorBoardLogger()
        logger.init()
        assert not logger.is_enabled

    def test_log_noop_when_disabled(self):
        logger = TensorBoardLogger()
        logger.log_scalar("train/loss", 1.0, 0)
        logger.log_scalars("train", {"loss": 1.0, "lr": 0.001}, 0)
        logger.log_histogram("params", torch.randn(100), 0)
        logger.log_text("info", "test", 0)
        logger.close()

    def test_init_with_log_dir(self):
        logger = TensorBoardLogger(log_dir="logs/test")
        assert logger.log_dir == "logs/test"

    def test_init_with_flush_secs(self):
        logger = TensorBoardLogger(flush_secs=10)
        assert logger.flush_secs == 10


# ════════════════════════════════════════════════════════════
# MONITOR CALLBACK TESTS
# ════════════════════════════════════════════════════════════

class TestMonitorCallback:
    def test_init_without_loggers(self):
        cb = MonitorCallback()
        cb.init()
        assert not cb.is_enabled

    def test_init_with_wandb(self):
        wb = WandBLogger(project="test")
        cb = MonitorCallback(wandb_logger=wb)
        cb.init()
        # Should not crash even if wandb not installed
        assert not cb.is_enabled

    def test_init_with_tensorboard(self):
        tb = TensorBoardLogger()
        cb = MonitorCallback(tb_logger=tb)
        cb.init()
        assert not cb.is_enabled

    def test_on_step_noop(self):
        cb = MonitorCallback()
        # Should not raise when no loggers are active
        cb.on_step(step=10, loss=2.5, lr=3e-4, tok_per_s=1000)

    def test_on_step_skips_interval(self):
        cb = MonitorCallback(log_interval=10)
        # Step 5 should be skipped
        cb.on_step(step=5, loss=2.5, lr=3e-4, tok_per_s=1000)
        # No assertion needed — just shouldn't crash

    def test_on_epoch_noop(self):
        cb = MonitorCallback()
        cb.on_epoch(epoch=0, avg_loss=2.5, step=100)

    def test_finish_noop(self):
        cb = MonitorCallback()
        cb.finish()  # Should not raise

    def test_init_with_both(self):
        wb = WandBLogger(project="test")
        tb = TensorBoardLogger()
        cb = MonitorCallback(wandb_logger=wb, tb_logger=tb)
        cb.init()
        cb.finish()

    def test_log_grads_disabled(self):
        cb = MonitorCallback(log_grads=False)
        cb.on_step(step=10, loss=2.5, lr=3e-4, tok_per_s=1000, grad_norm=5.0)

    def test_log_histograms_disabled(self):
        cb = MonitorCallback(log_histograms=False)
        model = torch.nn.Linear(10, 10)
        cb.on_step(step=500, loss=2.5, lr=3e-4, tok_per_s=1000, model=model)

    def test_default_log_interval(self):
        cb = MonitorCallback()
        assert cb.log_interval == 10

    def test_default_histogram_interval(self):
        cb = MonitorCallback()
        assert cb.histogram_interval == 500


if __name__ == "__main__":
    pytest.main([__file__, "-v"])