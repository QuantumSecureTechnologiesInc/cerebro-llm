"""Data auto-detection for the training pipeline.

Scans the datasets/ directory and classifies contents:
- Pre-tokenized: .bin or .npy files in datasets/tokens/
- Raw text: .txt, .jsonl, .json, .md, .csv files in datasets/raw/
- Mixed: both types present
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass


@dataclass
class DataInventory:
    """Summary of available training data."""
    has_pretokenized: bool = False
    has_raw_text: bool = False
    pretokenized_dir: str = ""
    raw_text_dir: str = ""
    pretokenized_files: list[str] | None = None
    raw_text_files: list[str] | None = None
    total_raw_files: int = 0
    total_pretokenized_files: int = 0
    status: str = "empty"  # "empty", "raw", "pretokenized", "mixed"

    def summary(self) -> str:
        """Human-readable summary."""
        if self.status == "empty":
            return "No training data found in datasets/"
        
        lines = [f"Data Status: {self.status.upper()}"]
        if self.has_pretokenized:
            lines.append(f"  Pre-tokenized: {self.total_pretokenized_files} shards in {self.pretokenized_dir}")
        if self.has_raw_text:
            lines.append(f"  Raw text: {self.total_raw_files} files in {self.raw_text_dir}")
        return "\n".join(lines)


def scan_datasets(base_dir: str = "datasets") -> DataInventory:
    """Scan the datasets directory and classify contents.
    
    Args:
        base_dir: Root datasets directory (relative to cerebro/).
        
    Returns:
        DataInventory with classification results.
    """
    base_path = Path(base_dir)
    inventory = DataInventory()
    
    # Check for pre-tokenized data in datasets/tokens/
    tokens_dir = base_path / "tokens"
    if tokens_dir.exists() and tokens_dir.is_dir():
        token_files = [
            f for f in tokens_dir.iterdir()
            if f.suffix in {".bin", ".npy"} and f.is_file()
        ]
        if token_files:
            inventory.has_pretokenized = True
            inventory.pretokenized_dir = str(tokens_dir)
            inventory.pretokenized_files = [str(f) for f in sorted(token_files)]
            inventory.total_pretokenized_files = len(token_files)
    
    # Check for raw text in datasets/raw/
    raw_dir = base_path / "raw"
    text_extensions = {".txt", ".jsonl", ".json", ".md", ".csv", ".tsv", ".rst"}
    
    if raw_dir.exists() and raw_dir.is_dir():
        text_files = []
        # Recursively find all text files
        for root, dirs, files in os.walk(raw_dir):
            for fname in files:
                if Path(fname).suffix.lower() in text_extensions:
                    text_files.append(os.path.join(root, fname))
        
        if text_files:
            inventory.has_raw_text = True
            inventory.raw_text_dir = str(raw_dir)
            inventory.raw_text_files = sorted(text_files)
            inventory.total_raw_files = len(text_files)
    
    # Also check for loose text files directly in datasets/
    if base_path.exists() and base_path.is_dir():
        loose_files = [
            f for f in base_path.iterdir()
            if f.is_file() and f.suffix.lower() in text_extensions
        ]
        if loose_files and not inventory.has_raw_text:
            inventory.has_raw_text = True
            inventory.raw_text_dir = str(base_path)
            inventory.raw_text_files = [str(f) for f in sorted(loose_files)]
            inventory.total_raw_files = len(loose_files)
    
    # Classify status
    if inventory.has_pretokenized and inventory.has_raw_text:
        inventory.status = "mixed"
    elif inventory.has_pretokenized:
        inventory.status = "pretokenized"
    elif inventory.has_raw_text:
        inventory.status = "raw"
    else:
        inventory.status = "empty"
    
    return inventory


def get_data_path(inventory: DataInventory, prefer_pretokenized: bool = True) -> str | None:
    """Get the best data path for training.
    
    Args:
        inventory: Data inventory from scan_datasets().
        prefer_pretokenized: Prefer pre-tokenized data if both exist.
        
    Returns:
        Path to data directory, or None if empty.
    """
    if inventory.status == "empty":
        return None
    
    if prefer_pretokenized and inventory.has_pretokenized:
        return inventory.pretokenized_dir
    elif inventory.has_raw_text:
        return inventory.raw_text_dir
    elif inventory.has_pretokenized:
        return inventory.pretokenized_dir
    
    return None
