"""HuggingFace dataset downloader for Cerebro training.

Downloads datasets from HuggingFace Hub and saves them as JSONL
files in the datasets/raw/ directory for tokenization.
"""

from __future__ import annotations

import os
import json
import sys
from pathlib import Path
from typing import Iterator


def _resolve_datasets_import():
    """Fix namespace conflict: cerebro/datasets/ shadows pip datasets package.
    
    Temporarily removes the current directory from sys.path so the
    installed 'datasets' library can be imported correctly.
    """
    saved = []
    for entry in ("", "."):
        if entry in sys.path:
            sys.path.remove(entry)
            saved.append(entry)
    return saved


def _restore_sys_path(saved: list[str]) -> None:
    """Restore sys.path entries removed by _resolve_datasets_import."""
    for entry in saved:
        if entry not in sys.path:
            sys.path.insert(0, entry)


def check_datasets_library() -> bool:
    """Check if the datasets library is installed.
    
    Returns:
        True if available, False otherwise.
    """
    saved = _resolve_datasets_import()
    try:
        import datasets
        return True
    except ImportError:
        return False
    finally:
        _restore_sys_path(saved)


def download_hf_dataset(
    dataset_name: str,
    output_dir: str = "datasets/raw",
    subset: str | None = None,
    split: str = "train",
    max_samples: int | None = None,
    streaming: bool = False,
    hf_token: str | None = None,
) -> dict:
    """Download a HuggingFace dataset and save as JSONL.
    
    Args:
        dataset_name: HuggingFace dataset ID or catalog name.
        output_dir: Output directory (will create subdirectory).
        subset: Dataset subset/config (e.g., "en" for c4).
        split: Dataset split (default: "train").
        max_samples: Limit number of samples (None = all).
        streaming: Use streaming mode for large datasets.
        hf_token: HuggingFace API token (or set HF_TOKEN env var).
        
    Returns:
        Dict with download statistics.
    """
    if not check_datasets_library():
        raise ImportError(
            "The 'datasets' library is required for HuggingFace downloads.\n"
            "Install it with: pip install datasets"
        )
    
    saved = _resolve_datasets_import()
    try:
        from datasets import load_dataset
    finally:
        _restore_sys_path(saved)
    from pipeline.hf_catalog import get_dataset
    
    # Check if it's a catalog dataset
    catalog_entry = get_dataset(dataset_name)
    if catalog_entry:
        hf_id = catalog_entry.hf_id
        text_field = catalog_entry.text_field
        # Use catalog subset if not overridden
        if subset is None and catalog_entry.subset:
            subset = catalog_entry.subset
        print(f"Found in catalog: {catalog_entry.name}")
    else:
        # Treat as direct HF ID
        hf_id = dataset_name
        text_field = "text"
        print(f"Using direct HF ID: {hf_id}")
    
    # Create output directory
    dataset_slug = hf_id.replace("/", "_")
    out_path = Path(output_dir) / dataset_slug
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Get token from env if not provided
    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN")
    
    print(f"Downloading {hf_id}...")
    print(f"  Subset: {subset or '(default)'}")
    print(f"  Split: {split}")
    print(f"  Streaming: {streaming}")
    print(f"  Output: {out_path}")
    print("-" * 60)
    
    # Load dataset
    # Use slice notation for large datasets to avoid materializing everything
    effective_split = split
    if max_samples and not streaming:
        effective_split = f"{split}[:{max_samples}]"
    
    try:
        load_kwargs = {
            "path": hf_id,
            "split": effective_split,
            "streaming": streaming,
        }
        if subset:
            load_kwargs["name"] = subset
        if hf_token:
            load_kwargs["token"] = hf_token
        
        dataset = load_dataset(**load_kwargs)
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        raise
    
    # Determine output file
    output_file = out_path / f"{split}.jsonl"
    
    # Write to JSONL
    samples_written = 0
    chars_written = 0
    
    print("Writing samples...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in dataset:
            # Extract text from the sample
            text = None
            
            # Handle instruction+response format (instruction may be JSON with messages)
            if "instruction" in sample and "response" in sample:
                instr = sample["instruction"]
                resp = sample["response"]
                try:
                    instr_data = json.loads(instr)
                    if isinstance(instr_data, dict) and "messages" in instr_data:
                        parts = []
                        for m in instr_data["messages"]:
                            if isinstance(m, dict):
                                role = m.get("role", "")
                                content = m.get("content", "")
                                if content:
                                    parts.append(f"{role}: {content}")
                        parts.append(f"assistant: {resp}")
                        text = "\n".join(parts)
                except (json.JSONDecodeError, TypeError):
                    pass
                if not text:
                    text = f"{instr}\n{resp}"
            
            # Handle chat-format datasets with 'messages' field (list of {role, content})
            # Handle chat-format datasets with 'messages' or 'conversations' field
            if not text and "conversations" in sample and isinstance(sample["conversations"], (list, tuple)):
                parts = []
                for msg in sample["conversations"]:
                    if isinstance(msg, dict):
                        role = msg.get("role", msg.get("from", ""))
                        content = msg.get("content", msg.get("value", ""))
                        if content:
                            parts.append(f"{role}: {content}")
                if parts:
                    text = "\n".join(parts)
            
            if not text and "messages" in sample and isinstance(sample["messages"], (list, tuple)):
                parts = []
                for msg in sample["messages"]:
                    if isinstance(msg, dict):
                        role = msg.get("role", "")
                        content = msg.get("content", "")
                        if content:
                            parts.append(f"{role}: {content}")
                if parts:
                    text = "\n".join(parts)
            
            # Try the configured text field
            if not text and text_field in sample and isinstance(sample[text_field], str):
                text = sample[text_field]
            
            # Try common alternatives
            if not text:
                for field in ["text", "content", "body", "input", "prompt", "message"]:
                    if field in sample and isinstance(sample[field], str):
                        text = sample[field]
                        break
            
            # Skip if no text found
            if not text or not text.strip():
                continue
            
            # Write as JSONL
            record = {"text": text}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
            samples_written += 1
            chars_written += len(text)
            
            # Progress reporting
            if samples_written % 10000 == 0:
                print(f"  {samples_written:,} samples written...")
            
            # Check limit
            if max_samples and samples_written >= max_samples:
                print(f"  Reached max_samples limit: {max_samples}")
                break
    
    print("-" * 60)
    print(f"Download complete!")
    print(f"  Samples: {samples_written:,}")
    print(f"  Characters: {chars_written:,}")
    print(f"  Output file: {output_file}")
    print(f"  File size: {output_file.stat().st_size / (1024*1024):.2f} MB")
    
    return {
        "samples": samples_written,
        "characters": chars_written,
        "output_file": str(output_file),
        "dataset_name": hf_id,
    }


def download_from_catalog(
    names: list[str],
    output_dir: str = "datasets/raw",
    max_samples_per_dataset: int | None = None,
) -> list[dict]:
    """Download multiple datasets from the catalog.
    
    Args:
        names: List of dataset names or HF IDs.
        output_dir: Base output directory.
        max_samples_per_dataset: Limit samples per dataset.
        
    Returns:
        List of download statistics.
    """
    results = []
    
    for i, name in enumerate(names, 1):
        print(f"\n{'='*60}")
        print(f"Dataset {i}/{len(names)}: {name}")
        print('='*60)
        
        try:
            stats = download_hf_dataset(
                dataset_name=name,
                output_dir=output_dir,
                max_samples=max_samples_per_dataset,
            )
            results.append(stats)
        except Exception as e:
            print(f"Failed to download {name}: {e}", file=sys.stderr)
            results.append({
                "dataset_name": name,
                "error": str(e),
            })
    
    return results


def print_download_instructions() -> None:
    """Print instructions for manual dataset download."""
    print("\n" + "=" * 70)
    print("DATASET DOWNLOAD INSTRUCTIONS")
    print("=" * 70)
    print("""
1. Install the datasets library:
   pip install datasets

2. Set your HuggingFace token (if using private datasets):
   export HF_TOKEN=your_token_here
   
   Or on Windows PowerShell:
   $env:HF_TOKEN="your_token_here"

3. Download a dataset:
   python -m pipeline.downloader --dataset alpaca
   
   Or with custom parameters:
   python -m pipeline.downloader --dataset c4 --subset en --max-samples 10000

4. Place the downloaded data in datasets/raw/<dataset_name>/
   The pipeline will automatically detect and tokenize it.

5. For very large datasets, use streaming mode:
   python -m pipeline.downloader --dataset c4 --streaming --max-samples 100000
""")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download HuggingFace datasets for Cerebro training")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name or HF ID")
    parser.add_argument("--output", type=str, default="datasets/raw", help="Output directory")
    parser.add_argument("--subset", type=str, default=None, help="Dataset subset/config")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to download")
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace API token")
    
    args = parser.parse_args()
    
    try:
        stats = download_hf_dataset(
            dataset_name=args.dataset,
            output_dir=args.output,
            subset=args.subset,
            split=args.split,
            max_samples=args.max_samples,
            streaming=args.streaming,
            hf_token=args.token,
        )
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        print_download_instructions()
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
