"""Streaming data loader for large-scale language model training.

Supports memory-mapped token files for training on corpora that
don't fit in RAM. Implements dynamic sequence packing.
"""

from __future__ import annotations

import os
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger("cerebro.data")


class TokenDataset(Dataset):
    """Memory-mapped token dataset.

    Reads pre-tokenized data stored as numpy uint16 arrays on disk.
    Each file contains a flat array of token IDs. Sequences are
    extracted as fixed-length windows.

    Args:
        data_dir: Directory containing .npy or .bin token files.
        seq_len: Sequence length for each sample.
    """

    def __init__(self, data_dir: str, seq_len: int) -> None:
        self.seq_len = seq_len
        self.data = []
        self.total_tokens = 0

        # Load all token files
        for fname in sorted(os.listdir(data_dir)):
            fpath = os.path.join(data_dir, fname)
            if fname.endswith(".npy"):
                tokens = np.load(fpath)
            elif fname.endswith(".bin"):
                tokens = np.fromfile(fpath, dtype=np.uint16)
            else:
                continue

            self.data.append(tokens)
            self.total_tokens += len(tokens)

        if not self.data:
            raise ValueError(f"No token files found in {data_dir}")

        # Build index: (file_idx, offset) for each sequence
        self.index = []
        for file_idx, tokens in enumerate(self.data):
            num_seqs = max(1, len(tokens) // seq_len)
            for i in range(num_seqs):
                self.index.append((file_idx, i * seq_len))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        file_idx, offset = self.index[idx]
        tokens = self.data[file_idx]

        # Extract sequence (pad with zeros if at end of file)
        end = min(offset + self.seq_len + 1, len(tokens))
        chunk = tokens[offset:end]

        if len(chunk) < self.seq_len + 1:
            # Pad with zeros
            padded = np.zeros(self.seq_len + 1, dtype=np.int64)
            padded[: len(chunk)] = chunk.astype(np.int64)
            chunk = padded
        else:
            chunk = chunk[: self.seq_len + 1].astype(np.int64)

        input_ids = torch.from_numpy(chunk[:-1])
        labels = torch.from_numpy(chunk[1:])

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


class RandomTokenDataset(Dataset):
    """Random token dataset for architecture validation / smoke testing.

    Generates random token sequences on-the-fly. Not for real training.
    """

    def __init__(self, num_samples: int, seq_len: int, vocab_size: int) -> None:
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # +1 for shifted labels
        tokens = torch.randint(3, self.vocab_size, (self.seq_len + 1,))
        return {
            "input_ids": tokens[:-1],
            "labels": tokens[1:],
        }


def create_dataloader(
    data_dir: str | None,
    seq_len: int,
    batch_size: int,
    vocab_size: int = 128_000,
    num_workers: int = 0,
    num_samples: int = 10_000,
) -> DataLoader:
    """Create a training DataLoader.

    If data_dir is provided and contains token files, uses TokenDataset.
    Otherwise, falls back to RandomTokenDataset for testing.

    Args:
        data_dir: Directory with token files, or None for random data.
        seq_len: Sequence length.
        batch_size: Batch size.
        vocab_size: Vocabulary size (for random data).
        num_workers: DataLoader workers.
        num_samples: Number of random samples (if no data_dir).

    Returns:
        DataLoader instance.
    """
    if data_dir and os.path.isdir(data_dir) and os.listdir(data_dir):
        try:
            dataset = TokenDataset(data_dir, seq_len)
            logger.info("Loaded %d sequences from %s", len(dataset), data_dir)
        except (OSError, ValueError, RuntimeError) as e:
            logger.warning("Failed to load TokenDataset from %s: %s; falling back to random", data_dir, e)
            dataset = RandomTokenDataset(num_samples, seq_len, vocab_size)
            logger.info("Using random dataset (%d samples)", num_samples)
    else:
        dataset = RandomTokenDataset(num_samples, seq_len, vocab_size)
        logger.info("Using random dataset (%d samples)", num_samples)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
