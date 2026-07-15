"""Tests for all new Cerebro training infrastructure modules."""

import os
import json
import tempfile
import numpy as np
import torch
import torch.nn as nn
import pytest

# ════════════════════════════════════════════════════════════
# TOKENIZE MODULE TESTS
# ════════════════════════════════════════════════════════════

class TestTokenizeStats:
    def test_creation(self):
        from cerebro.training.tokenize import TokenizeStats
        stats = TokenizeStats()
        assert stats.files_processed == 0
        assert stats.total_tokens == 0
        assert stats.chars_per_token == 0.0

    def test_summary(self):
        from cerebro.training.tokenize import TokenizeStats
        stats = TokenizeStats(files_processed=5, total_characters=1000, total_tokens=250)
        summary = stats.summary()
        assert "Files processed" in summary
        assert "5" in summary
        assert stats.chars_per_token == 4.0


class TestReadTextFiles:
    def test_read_txt(self):
        from cerebro.training.tokenize import read_text_files
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.txt")
            with open(path, "w") as f:
                f.write("Hello world")
            results = list(read_text_files(input_dir=tmpdir))
            assert len(results) == 1
            assert results[0][1] == "Hello world"

    def test_read_jsonl(self):
        from cerebro.training.tokenize import read_text_files
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.jsonl")
            with open(path, "w") as f:
                f.write('{"text": "Sample text one"}\n')
                f.write('{"text": "Sample text two"}\n')
            results = list(read_text_files(input_dir=tmpdir))
            assert len(results) == 2

    def test_read_json_array(self):
        from cerebro.training.tokenize import read_text_files
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.json")
            with open(path, "w") as f:
                json.dump([{"text": "Item 1"}, {"text": "Item 2"}], f)
            results = list(read_text_files(input_dir=tmpdir))
            assert len(results) == 2

    def test_explicit_files(self):
        from cerebro.training.tokenize import read_text_files
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "explicit.txt")
            with open(path, "w") as f:
                f.write("Explicit file content")
            results = list(read_text_files(files=[path]))
            assert len(results) == 1
            assert "Explicit file content" in results[0][1]

    def test_skips_unsupported(self):
        from cerebro.training.tokenize import read_text_files
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "image.png"), "w") as f:
                f.write("fake png")
            results = list(read_text_files(input_dir=tmpdir))
            assert len(results) == 0


class TestTokenizeSingleFile:
    def test_tokenize_single(self):
        from cerebro.training.tokenize import tokenize_single_file
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.txt")
            output_path = os.path.join(tmpdir, "output.bin")
            with open(input_path, "w") as f:
                f.write("The quick brown fox jumps over the lazy dog.")
            num_tokens = tokenize_single_file(input_path, output_path)
            assert num_tokens > 0
            assert os.path.exists(output_path)
            tokens = np.fromfile(output_path, dtype=np.uint32)
            assert len(tokens) == num_tokens


class TestTokenizeToShards:
    def test_basic_tokenization(self):
        from cerebro.training.tokenize import tokenize_to_shards
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = os.path.join(tmpdir, "raw")
            output_dir = os.path.join(tmpdir, "tokens")
            os.makedirs(input_dir)

            for i in range(3):
                with open(os.path.join(input_dir, f"doc_{i}.txt"), "w") as f:
                    f.write(f"Document {i}: " + "word " * 100)

            stats = tokenize_to_shards(
                input_dir=input_dir,
                output_dir=output_dir,
                shard_size=500,
            )
            assert stats.files_processed == 3
            assert stats.total_tokens > 0
            assert stats.shards_written > 0
            assert os.path.exists(os.path.join(output_dir, "meta.json"))


# ════════════════════════════════════════════════════════════
# MIXING / STREAMING MODULE TESTS
# ════════════════════════════════════════════════════════════

class TestStreamingTokenDataset:
    def test_streaming_basic(self):
        from cerebro.training.mixing import StreamingTokenDataset
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a shard
            tokens = np.arange(1000, dtype=np.uint32)
            tokens.tofile(os.path.join(tmpdir, "shard_0000.bin"))

            ds = StreamingTokenDataset(tmpdir, seq_len=32)
            samples = list(ds)
            assert len(samples) > 0
            assert "input_ids" in samples[0]
            assert "labels" in samples[0]
            assert samples[0]["input_ids"].shape == (32,)

    def test_estimate_length(self):
        from cerebro.training.mixing import StreamingTokenDataset
        with tempfile.TemporaryDirectory() as tmpdir:
            tokens = np.arange(5000, dtype=np.uint32)
            tokens.tofile(os.path.join(tmpdir, "shard_0000.bin"))

            ds = StreamingTokenDataset(tmpdir, seq_len=100)
            est = ds.estimate_length()
            assert est == 50  # 5000 // 100

    def test_empty_dir_raises(self):
        from cerebro.training.mixing import StreamingTokenDataset
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="No .bin"):
                StreamingTokenDataset(tmpdir, seq_len=32)


class TestInterleavedDataset:
    def test_interleaved_basic(self):
        from cerebro.training.mixing import InterleavedDataset
        from cerebro.training.data import RandomTokenDataset

        ds1 = RandomTokenDataset(100, seq_len=16, vocab_size=1000)
        ds2 = RandomTokenDataset(100, seq_len=16, vocab_size=1000)

        mixed = InterleavedDataset([ds1, ds2], weights=[0.7, 0.3], num_samples=50)
        assert len(mixed) == 50

        sample = mixed[0]
        assert "input_ids" in sample
        assert sample["input_ids"].shape == (16,)

    def test_single_dataset(self):
        from cerebro.training.mixing import InterleavedDataset
        from cerebro.training.data import RandomTokenDataset

        ds = RandomTokenDataset(50, seq_len=16, vocab_size=1000)
        mixed = InterleavedDataset([ds], num_samples=50)
        assert len(mixed) == 50


class TestDataMixer:
    def test_add_source(self):
        from cerebro.training.mixing import DataMixer
        mixer = DataMixer()
        mixer.add_source("/tmp/fake", weight=1.0, name="test")
        assert len(mixer.sources) == 1

    def test_summary(self):
        from cerebro.training.mixing import DataMixer
        mixer = DataMixer()
        mixer.add_source("/tmp/a", weight=0.6, name="wikipedia")
        mixer.add_source("/tmp/b", weight=0.4, name="code")
        summary = mixer.summary()
        assert "wikipedia" in summary
        assert "code" in summary

    def test_create_loader(self):
        from cerebro.training.mixing import DataMixer
        mixer = DataMixer()
        # No real data — will use random fallback
        mixer.add_source("/tmp/nonexistent", weight=1.0)
        loader = mixer.create_loader(seq_len=16, batch_size=2, num_samples=20)
        batch = next(iter(loader))
        assert "input_ids" in batch
        assert batch["input_ids"].shape[0] == 2


# ════════════════════════════════════════════════════════════
# DPO / ALIGNMENT MODULE TESTS
# ════════════════════════════════════════════════════════════

class TestPreferenceSample:
    def test_creation(self):
        from cerebro.training.alignment import PreferenceSample
        sample = PreferenceSample(
            prompt="What is AI?",
            chosen="AI is artificial intelligence.",
            rejected="I don't know.",
        )
        assert sample.prompt == "What is AI?"
        assert sample.chosen == "AI is artificial intelligence."


class TestPreferenceDataset:
    def test_load_jsonl(self):
        from cerebro.training.alignment import PreferenceDataset
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({
                "prompt": "Hello",
                "chosen": "Hi there!",
                "rejected": "Go away.",
            }) + "\n")
            f.write(json.dumps({
                "prompt": "What is 2+2?",
                "chosen": "4",
                "rejected": "5",
            }) + "\n")
            f.flush()

        dataset = PreferenceDataset(f.name)
        assert len(dataset) == 2
        assert dataset.samples[0].prompt == "Hello"

        os.unlink(f.name)


class TestDPOResult:
    def test_creation(self):
        from cerebro.training.alignment import DPOResult
        result = DPOResult(
            loss=0.5, chosen_reward=1.2, rejected_reward=-0.3,
            reward_margin=1.5, accuracy=0.85,
            total_steps=100, elapsed_seconds=60.0,
        )
        assert result.accuracy == 0.85


# ════════════════════════════════════════════════════════════
# LORA / FINETUNE MODULE TESTS
# ════════════════════════════════════════════════════════════

class TestLoRAConfig:
    def test_defaults(self):
        from cerebro.training.finetune import LoRAConfig
        config = LoRAConfig()
        assert config.rank == 16
        assert config.alpha == 32.0
        assert config.scaling == 2.0

    def test_serialization(self):
        from cerebro.training.finetune import LoRAConfig
        config = LoRAConfig(rank=8, alpha=16.0)
        d = config.to_dict()
        restored = LoRAConfig.from_dict(d)
        assert restored.rank == 8
        assert restored.alpha == 16.0


class TestLoRALinear:
    def test_forward_shape(self):
        from cerebro.training.finetune import LoRALinear
        lora = LoRALinear(in_features=64, out_features=32, rank=4, alpha=8.0)
        x = torch.randn(2, 64)
        output = lora(x)
        assert output.shape == (2, 32)

    def test_starts_as_identity(self):
        """LoRA should start as identity (B initialized to zeros)."""
        from cerebro.training.finetune import LoRALinear
        lora = LoRALinear(in_features=32, out_features=32, rank=4)
        # Base path only
        base_out = torch.nn.functional.linear(torch.randn(1, 32), lora.weight)
        # Full LoRA output should equal base (since B=0)
        x = torch.randn(1, 32)
        full_out = lora(x)
        expected = torch.nn.functional.linear(x, lora.weight)
        assert torch.allclose(full_out, expected, atol=1e-5)

    def test_trainable_params(self):
        from cerebro.training.finetune import LoRALinear
        lora = LoRALinear(in_features=64, out_features=32, rank=8)
        expected = 64 * 8 + 8 * 32  # A + B
        assert lora.trainable_params == expected

    def test_merge_weights(self):
        from cerebro.training.finetune import LoRALinear
        lora = LoRALinear(in_features=32, out_features=32, rank=4)
        # Set B to something non-zero
        lora.lora_B.data.fill_(0.1)
        lora.merge_weights()
        # After merge, A and B should be zero
        assert lora.lora_A.abs().sum() == 0
        assert lora.lora_B.abs().sum() == 0

    def test_quantize(self):
        from cerebro.training.finetune import LoRALinear
        lora = LoRALinear(in_features=32, out_features=16, rank=4, quantize=True)
        lora.quantize_base_weight()
        assert lora._quantized_weight is not None
        assert lora._quant_scale is not None
        # Should still produce output
        x = torch.randn(2, 32)
        output = lora(x)
        assert output.shape == (2, 16)


class TestApplyRemoveLora:
    def test_apply_lora(self):
        from cerebro.training.finetune import LoRAConfig, LoRALinear, apply_lora

        model = nn.Sequential(
            nn.Linear(32, 32),
            nn.Linear(32, 16),
        )
        # Name the modules so they match target
        model[0]._parameters_names = "q_proj"
        config = LoRAConfig(rank=4, target_modules=[])  # empty = all linear
        modules = apply_lora(model, config)
        assert len(modules) == 2
        assert isinstance(model[0], LoRALinear)
        assert isinstance(model[1], LoRALinear)

    def test_remove_lora(self):
        from cerebro.training.finetune import LoRAConfig, apply_lora, remove_lora

        model = nn.Sequential(nn.Linear(16, 16))
        config = LoRAConfig(rank=4, target_modules=[])
        apply_lora(model, config)

        remove_lora(model, merge=True)
        assert isinstance(model[0], nn.Linear)


class TestLoRAParamCounting:
    def test_count_lora_params(self):
        from cerebro.training.finetune import LoRAConfig, apply_lora, count_lora_params

        model = nn.Sequential(nn.Linear(512, 512), nn.Linear(512, 256))
        config = LoRAConfig(rank=4, target_modules=[])
        apply_lora(model, config)

        counts = count_lora_params(model)
        assert counts["lora"] > 0
        assert counts["lora_percent"] > 0
        assert counts["lora_percent"] < 10  # LoRA should be a small fraction


# ════════════════════════════════════════════════════════════
# CHAT TEMPLATE MODULE TESTS
# ════════════════════════════════════════════════════════════

class TestChatMessage:
    def test_from_dict(self):
        from cerebro.training.chat_template import ChatMessage
        msg = ChatMessage.from_dict({"role": "user", "content": "Hello"})
        assert msg.role == "user"
        assert msg.content == "Hello"


class TestChatFormatter:
    def test_format_message(self):
        from cerebro.training.chat_template import ChatFormatter, ChatMessage
        fmt = ChatFormatter()
        msg = ChatMessage(role="user", content="Hello")
        result = fmt.format_message(msg)
        assert "<|user|>" in result
        assert "Hello" in result

    def test_format_conversation(self):
        from cerebro.training.chat_template import (
            ChatFormatter, ChatMessage, ChatConversation,
        )
        conv = ChatConversation(messages=[
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assistant", content="Hello!"),
        ])
        fmt = ChatFormatter()
        text = fmt.format_conversation(conv)
        assert "<|system|>" in text
        assert "<|user|>" in text
        assert "<|assistant|>" in text
        assert "Hello!" in text

    def test_format_with_loss_masking(self):
        from cerebro.training.chat_template import (
            ChatFormatter, ChatMessage, ChatConversation,
        )
        conv = ChatConversation(messages=[
            ChatMessage(role="user", content="Question"),
            ChatMessage(role="assistant", content="Answer"),
        ])
        fmt = ChatFormatter()
        result = fmt.format_for_loss_masking(conv, tokenizer=None)
        assert "text" in result
        assert result["input_ids"] is None  # No tokenizer provided


class TestLoadFormats:
    def test_load_openai_format(self):
        from cerebro.training.chat_template import load_openai_format
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello!"},
                ]
            }) + "\n")
            f.flush()

        convs = load_openai_format(f.name)
        assert len(convs) == 1
        assert len(convs[0].messages) == 2
        os.unlink(f.name)

    def test_load_alpaca_format(self):
        from cerebro.training.chat_template import load_alpaca_format
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([{
                "instruction": "Explain AI",
                "input": "",
                "output": "AI is artificial intelligence.",
            }], f)
            f.flush()

        convs = load_alpaca_format(f.name)
        assert len(convs) == 1
        assert convs[0].messages[0].role == "user"
        assert convs[0].messages[1].role == "assistant"
        os.unlink(f.name)

    def test_load_sharegpt_format(self):
        from cerebro.training.chat_template import load_sharegpt_format
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([{
                "conversations": [
                    {"from": "human", "value": "Hello"},
                    {"from": "gpt", "value": "Hi there!"},
                ]
            }], f)
            f.flush()

        convs = load_sharegpt_format(f.name)
        assert len(convs) == 1
        assert convs[0].messages[0].role == "user"
        assert convs[0].messages[1].role == "assistant"
        os.unlink(f.name)


class TestSFTDataset:
    def test_statistics(self):
        from cerebro.training.chat_template import (
            SFTDataset, ChatConversation, ChatMessage,
        )
        convs = [
            ChatConversation(messages=[
                ChatMessage(role="user", content="Q1"),
                ChatMessage(role="assistant", content="A1"),
            ]),
            ChatConversation(messages=[
                ChatMessage(role="user", content="Q2"),
                ChatMessage(role="assistant", content="A2"),
                ChatMessage(role="user", content="Q3"),
                ChatMessage(role="assistant", content="A3"),
            ]),
        ]
        ds = SFTDataset(convs)
        stats = ds.statistics()
        assert stats["conversations"] == 2
        assert stats["total_messages"] == 6
        assert stats["avg_turns"] == 3.0


# ════════════════════════════════════════════════════════════
# CURRICULUM LEARNING TESTS
# ════════════════════════════════════════════════════════════

class TestCurriculumScheduler:
    def test_basic_stages(self):
        from cerebro.training.curriculum import CurriculumScheduler
        sched = CurriculumScheduler(stages=[(2048, 1000), (8192, 500)])
        assert sched.num_stages == 2
        assert sched.total_steps == 1500
        assert sched.max_seq_len == 8192

    def test_get_seq_len(self):
        from cerebro.training.curriculum import CurriculumScheduler
        sched = CurriculumScheduler(stages=[(2048, 1000), (8192, 500)])
        assert sched.get_seq_len(0) == 2048
        assert sched.get_seq_len(500) == 2048
        assert sched.get_seq_len(999) == 2048
        assert sched.get_seq_len(1000) == 8192
        assert sched.get_seq_len(1499) == 8192

    def test_get_stage(self):
        from cerebro.training.curriculum import CurriculumScheduler
        sched = CurriculumScheduler(stages=[(1024, 100), (2048, 200), (4096, 300)])
        assert sched.get_stage(0) == 0
        assert sched.get_stage(99) == 0
        assert sched.get_stage(100) == 1
        assert sched.get_stage(299) == 1
        assert sched.get_stage(300) == 2

    def test_should_transition(self):
        from cerebro.training.curriculum import CurriculumScheduler
        sched = CurriculumScheduler(stages=[(1024, 100), (2048, 200)])
        assert sched.should_transition(0) == True
        assert sched.should_transition(1) == False
        assert sched.should_transition(100) == True  # boundary

    def test_progress(self):
        from cerebro.training.curriculum import CurriculumScheduler
        sched = CurriculumScheduler(stages=[(1024, 100), (2048, 100)])
        assert sched.progress(0) == 0.0
        assert sched.progress(100) == 0.5
        assert sched.progress(200) == 1.0

    def test_summary(self):
        from cerebro.training.curriculum import CurriculumScheduler
        sched = CurriculumScheduler(stages=[(2048, 1000), (8192, 500)])
        summary = sched.summary()
        assert "Curriculum" in summary
        assert "2,048" in summary or "2048" in summary

    def test_from_preset(self):
        from cerebro.training.curriculum import CurriculumScheduler
        sched = CurriculumScheduler.from_preset("nano")
        assert sched.num_stages >= 2
        assert sched.max_seq_len >= 8192

    def test_invalid_ordering(self):
        from cerebro.training.curriculum import CurriculumScheduler
        with pytest.raises(ValueError, match="increasing"):
            CurriculumScheduler(stages=[(8192, 100), (2048, 100)])

    def test_learning_rate(self):
        from cerebro.training.curriculum import CurriculumScheduler
        sched = CurriculumScheduler(stages=[(2048, 1000), (8192, 1000)])
        lr_start = sched.get_learning_rate(0, base_lr=3e-4)
        lr_mid = sched.get_learning_rate(500, base_lr=3e-4)
        # Warmup: early LR should be lower
        assert lr_start < lr_mid

    def test_batch_size_override(self):
        from cerebro.training.curriculum import CurriculumStage, CurriculumScheduler
        stages = [
            CurriculumStage(seq_len=2048, num_steps=100, batch_size=8),
            CurriculumStage(seq_len=8192, num_steps=100, batch_size=2),
        ]
        sched = CurriculumScheduler(stages)
        assert sched.get_batch_size(0) == 8
        assert sched.get_batch_size(100) == 2

    def test_custom_stages(self):
        from cerebro.training.curriculum import CurriculumScheduler
        sched = CurriculumScheduler.from_preset("core", custom_stages=[(512, 50), (1024, 50)])
        assert sched.num_stages == 2
        assert sched.get_seq_len(0) == 512
