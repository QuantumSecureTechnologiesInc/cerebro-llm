"""Preprocessing script to tokenize raw text data into binary token files.

Reads text files (.txt, .jsonl, .json, .md) from an input directory,
tokenizes them with the Cerebro BPE tokenizer, and writes packed
uint32 binary files suitable for memory-mapped training.

Usage:
    cerebro tokenize --input data/raw/ --output data/tokens/
    cerebro tokenize --input data/raw/ --output data/tokens/ --shard-size 100M
    cerebro tokenize --files "file1.txt,file2.txt" --output data/tokens/
"""

from __future__ import annotations

import os
import sys
import json
import glob
import time
import numpy as np
from pathlib import Path
from typing import Iterator
from dataclasses import dataclass, field


@dataclass
class TokenizeStats:
    """Statistics from a tokenization run."""
    files_processed: int = 0
    total_characters: int = 0
    total_tokens: int = 0
    total_bytes_written: int = 0
    shards_written: int = 0
    skipped_files: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def chars_per_token(self) -> float:
        if self.total_tokens == 0:
            return 0.0
        return self.total_characters / self.total_tokens

    def summary(self) -> str:
        lines = [
            "Tokenization Complete",
            f"  Files processed:  {self.files_processed}",
            f"  Characters:       {self.total_characters:,}",
            f"  Tokens:           {self.total_tokens:,}",
            f"  Chars/token:      {self.chars_per_token:.2f}",
            f"  Shards written:   {self.shards_written}",
            f"  Bytes written:    {self.total_bytes_written:,}",
            f"  Skipped:          {len(self.skipped_files)}",
            f"  Elapsed:          {self.elapsed_seconds:.1f}s",
        ]
        return "\n".join(lines)


SUPPORTED_EXTENSIONS = {".txt", ".jsonl", ".json", ".md", ".rst", ".csv", ".tsv"}


def read_text_files(
    input_dir: str | None = None,
    files: list[str] | None = None,
    recursive: bool = True,
) -> Iterator[tuple[str, str]]:
    """Yield (filepath, text_content) from input sources.

    Args:
        input_dir: Directory to scan for text files.
        files: Explicit list of file paths.
        recursive: Search subdirectories.

    Yields:
        (filepath, text) tuples.
    """
    paths = []

    if files:
        for f in files:
            if os.path.isfile(f):
                paths.append(f)

    if input_dir and os.path.isdir(input_dir):
        if recursive:
            for root, dirs, fnames in os.walk(input_dir):
                for fname in sorted(fnames):
                    fpath = os.path.join(root, fname)
                    ext = os.path.splitext(fname)[1].lower()
                    if ext in SUPPORTED_EXTENSIONS:
                        paths.append(fpath)
        else:
            for fname in sorted(os.listdir(input_dir)):
                fpath = os.path.join(input_dir, fname)
                if os.path.isfile(fpath):
                    ext = os.path.splitext(fname)[1].lower()
                    if ext in SUPPORTED_EXTENSIONS:
                        paths.append(fpath)

    for filepath in paths:
        try:
            ext = os.path.splitext(filepath)[1].lower()
            if ext == ".jsonl":
                yield from _read_jsonl(filepath)
            elif ext == ".json":
                yield from _read_json(filepath)
            else:
                with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read()
                if text.strip():
                    yield (filepath, text)
        except Exception as e:
            print(f"  Warning: Skipping {filepath}: {e}", file=sys.stderr)


def _read_jsonl(filepath: str) -> Iterator[tuple[str, str]]:
    """Read text fields from a JSONL file."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # Try common text fields
                for key in ("text", "content", "body", "input", "prompt", "message"):
                    if key in obj and isinstance(obj[key], str):
                        yield (filepath, obj[key])
                        break
                # Also try conversations for chat data
                if "conversations" in obj:
                    parts = []
                    for msg in obj["conversations"]:
                        if isinstance(msg, dict) and "value" in msg:
                            parts.append(msg["value"])
                        elif isinstance(msg, str):
                            parts.append(msg)
                    if parts:
                        yield (filepath, "\n".join(parts))
            except (json.JSONDecodeError, KeyError):
                continue


def _read_json(filepath: str) -> Iterator[tuple[str, str]]:
    """Read text fields from a JSON file (array or object)."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return

    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                yield (filepath, item)
            elif isinstance(item, dict):
                for key in ("text", "content", "body", "input"):
                    if key in item and isinstance(item[key], str):
                        yield (filepath, item[key])
                        break
    elif isinstance(data, dict):
        for key in ("text", "content", "body"):
            if key in data and isinstance(data[key], str):
                yield (filepath, data[key])
                break


def tokenize_to_shards(
    input_dir: str | None = None,
    files: list[str] | None = None,
    output_dir: str = "data/tokens",
    shard_size: int = 100_000_000,
    recursive: bool = True,
    add_bos: bool = True,
    add_eos: bool = True,
    separator: str = "\n",
) -> TokenizeStats:
    """Tokenize text files into binary token shards.

    Args:
        input_dir: Directory with raw text files.
        files: Explicit file list.
        output_dir: Output directory for .bin shards.
        shard_size: Max tokens per shard (default 100M).
        recursive: Search subdirectories.
        add_bos: Add begin-of-sequence token.
        add_eos: Add end-of-sequence token.
        separator: Text between documents.

    Returns:
        TokenizeStats with processing summary.
    """
    from cerebro.tokenizer.tokenizer import CerebroTokenizer

    start_time = time.time()
    stats = TokenizeStats()

    os.makedirs(output_dir, exist_ok=True)
    tokenizer = CerebroTokenizer()

    token_buffer: list[int] = []
    shard_idx = 0

    def write_shard(tokens: list[int]) -> None:
        nonlocal shard_idx
        arr = np.array(tokens, dtype=np.uint32)
        shard_path = os.path.join(output_dir, f"shard_{shard_idx:04d}.bin")
        arr.tofile(shard_path)
        stats.shards_written += 1
        stats.total_bytes_written += arr.nbytes
        shard_idx += 1

    for filepath, text in read_text_files(input_dir, files, recursive):
        try:
            tokens = tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)

            if add_eos:
                sep_tokens = tokenizer.encode(separator, add_bos=False, add_eos=False)
                tokens.extend(sep_tokens)

            token_buffer.extend(tokens)
            stats.files_processed += 1
            stats.total_characters += len(text)
            stats.total_tokens += len(tokens)

            # Write shard when buffer is full
            while len(token_buffer) >= shard_size:
                write_shard(token_buffer[:shard_size])
                token_buffer = token_buffer[shard_size:]

            if stats.files_processed % 100 == 0:
                print(f"  Processed {stats.files_processed} files, "
                      f"{stats.total_tokens:,} tokens, "
                      f"{stats.shards_written} shards")

        except Exception as e:
            stats.skipped_files.append(f"{filepath}: {e}")
            continue

    # Write remaining tokens
    if token_buffer:
        write_shard(token_buffer)

    # Write metadata
    meta = {
        "total_tokens": stats.total_tokens,
        "total_characters": stats.total_characters,
        "total_files": stats.files_processed,
        "num_shards": stats.shards_written,
        "shard_size": shard_size,
        "vocab_size": tokenizer.vocab_size,
        "dtype": "uint32",
        "files_processed": stats.files_processed,
    }
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    stats.elapsed_seconds = time.time() - start_time
    return stats


def tokenize_single_file(
    input_file: str,
    output_file: str,
    add_bos: bool = True,
    add_eos: bool = True,
) -> int:
    """Tokenize a single text file to a single .bin file.

    Args:
        input_file: Path to input text file.
        output_file: Path to output .bin file.
        add_bos: Add BOS token.
        add_eos: Add EOS token.

    Returns:
        Number of tokens written.
    """
    from cerebro.tokenizer.tokenizer import CerebroTokenizer

    tokenizer = CerebroTokenizer()

    with open(input_file, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    tokens = tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)
    arr = np.array(tokens, dtype=np.uint32)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    arr.tofile(output_file)

    return len(tokens)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tokenize raw text for Cerebro training")
    parser.add_argument("--input", type=str, help="Input directory with text files")
    parser.add_argument("--files", type=str, help="Comma-separated list of input files")
    parser.add_argument("--output", type=str, default="data/tokens", help="Output directory")
    parser.add_argument("--shard-size", type=str, default="100M", help="Tokens per shard (e.g., 100M, 50M)")
    parser.add_argument("--no-bos", action="store_true", help="Don't add BOS tokens")
    parser.add_argument("--no-eos", action="store_true", help="Don't add EOS tokens")
    parser.add_argument("--no-recursive", action="store_true", help="Don't search subdirectories")

    args = parser.parse_args()

    # Parse shard size
    shard_str = args.shard_size.upper()
    if shard_str.endswith("M"):
        shard_size = int(shard_str[:-1]) * 1_000_000
    elif shard_str.endswith("K"):
        shard_size = int(shard_str[:-1]) * 1_000
    elif shard_str.endswith("B"):
        shard_size = int(shard_str[:-1]) * 1_000_000_000
    else:
        shard_size = int(shard_str)

    files_list = args.files.split(",") if args.files else None

    stats = tokenize_to_shards(
        input_dir=args.input,
        files=files_list,
        output_dir=args.output,
        shard_size=shard_size,
        recursive=not args.no_recursive,
        add_bos=not args.no_bos,
        add_eos=not args.no_eos,
    )

    print(stats.summary())
