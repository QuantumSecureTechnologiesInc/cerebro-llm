"""Streaming and mixing data loaders for large-scale training.

Provides:
- StreamingTokenDataset: memory-mapped on-disk access without loading all into RAM
- InterleavedDataset: weighted sampling across multiple data sources
- DataMixer: configure and manage multi-source training data
"""

from __future__ import annotations

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import Iterator


class StreamingTokenDataset(IterableDataset):
    """Memory-efficient streaming token dataset.

    Reads .bin token shards on-demand via memory mapping instead of
    loading everything into RAM. Supports datasets larger than available
    memory.

    Args:
        data_dir: Directory with .bin token shards and meta.json.
        seq_len: Sequence length per sample.
        shuffle: Shuffle shard order each epoch.
        max_shards_in_memory: Max shards to load at once (None = all).
    """

    def __init__(
        self,
        data_dir: str,
        seq_len: int,
        shuffle: bool = True,
        max_shards_in_memory: int | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.max_shards = max_shards_in_memory

        # Discover shards
        self.shard_paths = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".bin")
        ])

        if not self.shard_paths:
            raise ValueError(f"No .bin token shards found in {data_dir}")

        # Load metadata
        meta_path = os.path.join(data_dir, "meta.json")
        self.total_tokens = 0
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            self.total_tokens = meta.get("total_tokens", 0)

        # Estimate sequences per shard
        self._shard_seq_counts: list[int] = []
        for path in self.shard_paths:
            file_size = os.path.getsize(path)
            num_tokens = file_size // 4  # uint32 = 4 bytes
            self._shard_seq_counts.append(max(1, num_tokens // seq_len))

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """Iterate over all token sequences, shard by shard."""
        shard_indices = list(range(len(self.shard_paths)))
        if self.shuffle:
            np.random.shuffle(shard_indices)

        for shard_idx in shard_indices:
            shard_path = self.shard_paths[shard_idx]
            tokens = np.memmap(shard_path, dtype=np.uint32, mode="r")

            num_seqs = max(1, len(tokens) // self.seq_len)
            seq_indices = list(range(num_seqs))
            if self.shuffle:
                np.random.shuffle(seq_indices)

            for seq_idx in seq_indices:
                start = seq_idx * self.seq_len
                end = min(start + self.seq_len + 1, len(tokens))

                if end - start < 2:
                    continue

                chunk = tokens[start:end].astype(np.int64)

                # Pad if needed
                if len(chunk) < self.seq_len + 1:
                    padded = np.zeros(self.seq_len + 1, dtype=np.int64)
                    padded[:len(chunk)] = chunk
                    chunk = padded
                else:
                    chunk = chunk[:self.seq_len + 1]

                yield {
                    "input_ids": torch.from_numpy(chunk[:-1]),
                    "labels": torch.from_numpy(chunk[1:]),
                }

            # Release mmap
            del tokens

    def estimate_length(self) -> int:
        """Estimate total number of sequences."""
        total = 0
        for path in self.shard_paths:
            file_size = os.path.getsize(path)
            num_tokens = file_size // 4
            total += max(1, num_tokens // self.seq_len)
        return total


class InterleavedDataset(Dataset):
    """Weighted interleaved sampling across multiple data sources.

    Draws samples from multiple datasets with configurable weights,
    allowing mixed training on diverse data (Wikipedia + code + books).

    Args:
        datasets: List of Dataset objects.
        weights: Sampling weights per dataset (must sum to 1.0 or will be normalized).
        num_samples: Total samples per epoch.
    """

    def __init__(
        self,
        datasets: list[Dataset],
        weights: list[float] | None = None,
        num_samples: int = 100_000,
    ) -> None:
        self.datasets = datasets
        self.num_samples = num_samples

        if not datasets:
            raise ValueError("At least one dataset is required")

        # Normalize weights
        if weights is None:
            weights = [1.0 / len(datasets)] * len(datasets)
        total = sum(weights)
        self.weights = [w / total for w in weights]

        # Pre-compute dataset sizes
        self._sizes = []
        for ds in datasets:
            try:
                self._sizes.append(len(ds))
            except TypeError:
                self._sizes.append(num_samples)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Select dataset based on weights (deterministic per idx)
        rng = np.random.RandomState(idx)
        dataset_idx = rng.choice(len(self.datasets), p=self.weights)
        dataset = self.datasets[dataset_idx]

        # Select sample within chosen dataset
        sample_idx = rng.randint(0, self._sizes[dataset_idx])
        return dataset[sample_idx]


class DataMixer:
    """Configure and manage multi-source training data.

    Provides a fluent API for adding data sources with weights
    and creating a unified DataLoader.

    Example::

        mixer = DataMixer()
        mixer.add_source("data/wikipedia/", weight=0.4, name="wikipedia")
        mixer.add_source("data/code/", weight=0.3, name="code")
        mixer.add_source("data/books/", weight=0.3, name="books")
        loader = mixer.create_loader(seq_len=4096, batch_size=4)
    """

    def __init__(self) -> None:
        self.sources: list[dict] = []

    def add_source(
        self,
        data_dir: str,
        weight: float = 1.0,
        name: str | None = None,
        streaming: bool = False,
    ) -> DataMixer:
        """Add a data source with a sampling weight.

        Args:
            data_dir: Directory with token shards (.bin files).
            weight: Sampling weight (higher = more samples from this source).
            name: Human-readable name for logging.
            streaming: Use streaming (memory-mapped) loading.

        Returns:
            self for chaining.
        """
        self.sources.append({
            "data_dir": data_dir,
            "weight": weight,
            "name": name or os.path.basename(data_dir),
            "streaming": streaming,
        })
        return self

    def add_directory(
        self,
        base_dir: str,
        weight: float = 1.0,
        streaming: bool = False,
    ) -> DataMixer:
        """Auto-discover subdirectories as data sources.

        Each subdirectory with .bin files becomes a source.

        Args:
            base_dir: Parent directory containing source subdirectories.
            weight: Default weight for each discovered source.
            streaming: Use streaming loading.

        Returns:
            self for chaining.
        """
        if not os.path.isdir(base_dir):
            raise ValueError(f"Directory not found: {base_dir}")

        for entry in sorted(os.listdir(base_dir)):
            subdir = os.path.join(base_dir, entry)
            if os.path.isdir(subdir):
                bins = [f for f in os.listdir(subdir) if f.endswith(".bin")]
                if bins:
                    self.add_source(subdir, weight=weight, name=entry, streaming=streaming)

        return self

    def summary(self) -> str:
        """Get a summary of configured sources."""
        lines = [f"DataMixer — {len(self.sources)} sources:"]
        total_weight = sum(s["weight"] for s in self.sources)
        for s in self.sources:
            pct = s["weight"] / total_weight * 100 if total_weight > 0 else 0
            lines.append(f"  [{s['name']}] weight={s['weight']:.2f} ({pct:.1f}%) — {s['data_dir']}")
        return "\n".join(lines)

    def create_datasets(
        self,
        seq_len: int,
        vocab_size: int = 128_000,
        num_samples: int = 100_000,
    ) -> list[Dataset]:
        """Create Dataset objects for each source.

        Args:
            seq_len: Sequence length.
            vocab_size: Vocab size for random fallback.
            num_samples: Samples per epoch.

        Returns:
            List of Dataset objects.
        """
        from cerebro.training.data import TokenDataset, RandomTokenDataset

        datasets = []
        for source in self.sources:
            data_dir = source["data_dir"]
            if os.path.isdir(data_dir) and any(
                f.endswith(".bin") or f.endswith(".npy")
                for f in os.listdir(data_dir)
            ):
                if source.get("streaming"):
                    ds = StreamingTokenDataset(data_dir, seq_len)
                else:
                    ds = TokenDataset(data_dir, seq_len)
            else:
                ds = RandomTokenDataset(num_samples, seq_len, vocab_size)
            datasets.append(ds)

        return datasets

    def create_loader(
        self,
        seq_len: int,
        batch_size: int,
        vocab_size: int = 128_000,
        num_samples: int = 100_000,
        num_workers: int = 0,
    ) -> DataLoader:
        """Create a mixed DataLoader from all sources.

        Args:
            seq_len: Sequence length.
            batch_size: Batch size.
            vocab_size: Vocab size for random fallback.
            num_samples: Samples per epoch.
            num_workers: DataLoader workers.

        Returns:
            DataLoader with interleaved sampling.
        """
        if not self.sources:
            from cerebro.training.data import RandomTokenDataset
            dataset = RandomTokenDataset(num_samples, seq_len, vocab_size)
        elif len(self.sources) == 1:
            datasets = self.create_datasets(seq_len, vocab_size, num_samples)
            dataset = datasets[0]
        else:
            datasets = self.create_datasets(seq_len, vocab_size, num_samples)
            weights = [s["weight"] for s in self.sources]
            dataset = InterleavedDataset(datasets, weights, num_samples)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=not isinstance(dataset, IterableDataset),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
