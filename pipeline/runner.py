"""Main training pipeline orchestrator.

Coordinates data detection, tokenization, and training in a single
end-to-end workflow.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any


@dataclass
class PipelineConfig:
    """Configuration for the training pipeline."""
    # Model config
    model_preset: str = "nano"
    
    # Training hyperparameters
    learning_rate: float | None = None
    batch_size: int | None = None
    max_steps: int | None = None
    epochs: int = 1
    
    # Paths
    datasets_dir: str = "datasets"
    output_dir: str = "checkpoints"
    resume_from: str | None = None
    
    # Device
    device: str = "auto"
    
    # Data handling
    prefer_pretokenized: bool = True
    shard_size: str = "100M"
    
    def summary(self) -> str:
        """Human-readable config summary."""
        lines = [
            "Pipeline Configuration:",
            f"  Model: {self.model_preset}",
            f"  Epochs: {self.epochs}",
            f"  Learning rate: {self.learning_rate or '(preset default)'}",
            f"  Batch size: {self.batch_size or '(preset default)'}",
            f"  Max steps: {self.max_steps or '(preset default)'}",
            f"  Device: {self.device}",
            f"  Datasets dir: {self.datasets_dir}",
            f"  Output dir: {self.output_dir}",
        ]
        if self.resume_from:
            lines.append(f"  Resume from: {self.resume_from}")
        return "\n".join(lines)


@dataclass
class PipelineResult:
    """Results from a pipeline run."""
    success: bool
    data_status: str
    tokenization_stats: dict | None = None
    training_metrics: dict | None = None
    error: str | None = None
    
    def summary(self) -> str:
        """Human-readable results summary."""
        lines = [f"\n{'='*60}", "PIPELINE RESULTS", '='*60]
        lines.append(f"Status: {'SUCCESS' if self.success else 'FAILED'}")
        lines.append(f"Data: {self.data_status}")
        
        if self.tokenization_stats:
            lines.append(f"\nTokenization:")
            lines.append(f"  Files: {self.tokenization_stats.get('files_processed', 0):,}")
            lines.append(f"  Tokens: {self.tokenization_stats.get('total_tokens', 0):,}")
            lines.append(f"  Shards: {self.tokenization_stats.get('shards_written', 0)}")
        
        if self.training_metrics:
            lines.append(f"\nTraining:")
            lines.append(f"  Final loss: {self.training_metrics.get('final_loss', 0):.4f}")
            lines.append(f"  Total steps: {self.training_metrics.get('total_steps', 0):,}")
            elapsed = self.training_metrics.get('elapsed_seconds', 0)
            lines.append(f"  Elapsed: {elapsed:.1f}s")
            tok_s = self.training_metrics.get('tokens_per_second', 0)
            lines.append(f"  Tokens/sec: {tok_s:.0f}")
        
        if self.error:
            lines.append(f"\nError: {self.error}")
        
        lines.append('='*60 + '\n')
        return "\n".join(lines)


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """Run the complete training pipeline.
    
    Steps:
    1. Scan datasets/ directory
    2. Tokenize raw text if needed
    3. Load model config
    4. Train model
    5. Save checkpoint
    
    Args:
        config: Pipeline configuration.
        
    Returns:
        PipelineResult with metrics and status.
    """
    from pipeline.detector import scan_datasets, get_data_path
    
    print("\n" + "="*60)
    print("CEREBRO TRAINING PIPELINE")
    print("="*60 + "\n")
    
    # Ensure datasets directory exists
    datasets_path = Path(config.datasets_dir)
    datasets_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Scan for data
    print("Step 1: Scanning datasets directory...")
    print("-" * 60)
    inventory = scan_datasets(config.datasets_dir)
    print(inventory.summary())
    
    if inventory.status == "empty":
        return PipelineResult(
            success=False,
            data_status="empty",
            error="No training data found. Place data in datasets/raw/ or datasets/tokens/",
        )
    
    # Step 2: Tokenize if needed
    tokenization_stats = None
    if inventory.has_raw_text and (not inventory.has_pretokenized or not config.prefer_pretokenized):
        print("\nStep 2: Tokenizing raw text...")
        print("-" * 60)
        
        try:
            from cerebro.training.tokenize import tokenize_to_shards
            
            # Parse shard size
            shard_str = config.shard_size.upper()
            if shard_str.endswith("M"):
                shard_size = int(shard_str[:-1]) * 1_000_000
            elif shard_str.endswith("K"):
                shard_size = int(shard_str[:-1]) * 1_000
            elif shard_str.endswith("B"):
                shard_size = int(shard_str[:-1]) * 1_000_000_000
            else:
                shard_size = int(shard_str)
            
            tokens_output = str(datasets_path / "tokens")
            
            stats = tokenize_to_shards(
                input_dir=inventory.raw_text_dir,
                output_dir=tokens_output,
                shard_size=shard_size,
                recursive=True,
            )
            
            tokenization_stats = {
                "files_processed": stats.files_processed,
                "total_tokens": stats.total_tokens,
                "total_characters": stats.total_characters,
                "shards_written": stats.shards_written,
            }
            
            print(stats.summary())
            
            # Re-scan after tokenization
            inventory = scan_datasets(config.datasets_dir)
            
        except Exception as e:
            return PipelineResult(
                success=False,
                data_status=inventory.status,
                error=f"Tokenization failed: {e}",
            )
    else:
        print("\nStep 2: Tokenization skipped (using pre-tokenized data)")
        print("-" * 60)
    
    # Step 3: Get data path
    data_path = get_data_path(inventory, config.prefer_pretokenized)
    if not data_path:
        return PipelineResult(
            success=False,
            data_status=inventory.status,
            error="No valid data path found after tokenization",
        )
    
    print(f"\nStep 3: Data path resolved")
    print("-" * 60)
    print(f"  Using: {data_path}")
    
    # Step 4: Load model config
    print(f"\nStep 4: Loading model configuration...")
    print("-" * 60)
    
    try:
        from cerebro.config import CerebroConfig
        
        cerebro_config = CerebroConfig.from_name(config.model_preset)
        
        # Override with pipeline config
        if config.learning_rate:
            cerebro_config.learning_rate = config.learning_rate
        if config.batch_size:
            cerebro_config.batch_size = config.batch_size
        if config.max_steps:
            cerebro_config.max_steps = config.max_steps
        
        print(f"Model: Cerebro-{config.model_preset}")
        print(f"  Hidden dim: {cerebro_config.hidden_dim}")
        print(f"  Layers: {cerebro_config.num_layers}")
        print(f"  Max seq len: {cerebro_config.max_seq_len}")
        print(f"  Learning rate: {cerebro_config.learning_rate}")
        print(f"  Batch size: {cerebro_config.batch_size}")
        print(f"  Max steps: {cerebro_config.max_steps}")
        
    except Exception as e:
        return PipelineResult(
            success=False,
            data_status=inventory.status,
            tokenization_stats=tokenization_stats,
            error=f"Failed to load model config: {e}",
        )
    
    # Step 5: Train
    print(f"\nStep 5: Starting training...")
    print("-" * 60)
    print(config.summary())
    print("-" * 60)
    
    try:
        from cerebro.training.trainer import CerebroTrainer
        
        trainer = CerebroTrainer(
            config=cerebro_config,
            device=config.device,
            output_dir=config.output_dir,
        )
        
        # Load checkpoint if resuming
        if config.resume_from:
            trainer.load_checkpoint(config.resume_from)
            print(f"Resumed from checkpoint: {config.resume_from}")
        
        # Count parameters
        params = trainer.model.estimate_params()
        print(f"\nModel parameters: {params['total']:,}")
        print(f"  Embedding: {params['embedding']:,}")
        print(f"  Encoder:   {params['encoder_layers']:,}")
        print(f"  Reasoning: {params['reasoning_core']:,}")
        print(f"\nTraining on device: {trainer.device}")
        print("="*60 + "\n")
        
        # Run training
        metrics = trainer.train(
            data_dir=data_path,
            num_epochs=config.epochs,
        )
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"  Final loss:     {metrics['final_loss']:.4f}")
        print(f"  Total steps:    {metrics['total_steps']:,}")
        print(f"  Elapsed:        {metrics['elapsed_seconds']:.1f}s")
        print(f"  Tokens/sec:     {metrics['tokens_per_second']:.0f}")
        print(f"  Checkpoints:    {config.output_dir}")
        print("="*60 + "\n")
        
        return PipelineResult(
            success=True,
            data_status=inventory.status,
            tokenization_stats=tokenization_stats,
            training_metrics=metrics,
        )
        
    except Exception as e:
        import traceback
        return PipelineResult(
            success=False,
            data_status=inventory.status,
            tokenization_stats=tokenization_stats,
            error=f"Training failed: {e}\n{traceback.format_exc()}",
        )


def run_with_multi_source_mixing(config: PipelineConfig) -> PipelineResult:
    """Run pipeline with multi-source data mixing.
    
    Automatically discovers subdirectories in datasets/raw/ or datasets/tokens/
    and uses DataMixer for weighted interleaved training.
    
    Args:
        config: Pipeline configuration.
        
    Returns:
        PipelineResult with metrics.
    """
    from pipeline.detector import scan_datasets
    from cerebro.config import CerebroConfig
    from cerebro.training.trainer import CerebroTrainer
    from cerebro.training.mixing import DataMixer
    
    print("\n" + "="*60)
    print("CEREBRO TRAINING PIPELINE (Multi-Source Mixing)")
    print("="*60 + "\n")
    
    # Scan for data
    inventory = scan_datasets(config.datasets_dir)
    
    if inventory.status == "empty":
        return PipelineResult(
            success=False,
            data_status="empty",
            error="No training data found",
        )
    
    # Set up data mixer
    mixer = DataMixer()
    
    # Add pre-tokenized sources
    if inventory.has_pretokenized:
        tokens_path = Path(inventory.pretokenized_dir)
        if tokens_path.parent.name == "tokens":
            # Check if parent has subdirectories
            parent = tokens_path.parent
            for subdir in parent.iterdir():
                if subdir.is_dir() and subdir.name != "tokens":
                    bins = list(subdir.glob("*.bin"))
                    if bins:
                        mixer.add_source(str(subdir), weight=1.0, name=subdir.name)
        
        # If no subdirs, just add the tokens dir
        if not mixer.sources:
            mixer.add_source(inventory.pretokenized_dir, weight=1.0, name="tokens")
    
    # Add raw text sources (after tokenization)
    elif inventory.has_raw_text:
        raw_path = Path(inventory.raw_text_dir)
        # Tokenize each subdirectory separately
        for subdir in raw_path.iterdir():
            if subdir.is_dir():
                output_dir = Path(config.datasets_dir) / "tokens" / subdir.name
                if not output_dir.exists():
                    from cerebro.training.tokenize import tokenize_to_shards
                    tokenize_to_shards(
                        input_dir=str(subdir),
                        output_dir=str(output_dir),
                    )
                mixer.add_source(str(output_dir), weight=1.0, name=subdir.name)
        
        # If no subdirs, tokenize the whole raw dir
        if not mixer.sources:
            output_dir = Path(config.datasets_dir) / "tokens"
            from cerebro.training.tokenize import tokenize_to_shards
            tokenize_to_shards(
                input_dir=inventory.raw_text_dir,
                output_dir=str(output_dir),
            )
            mixer.add_source(str(output_dir), weight=1.0, name="data")
    
    print(mixer.summary())
    
    # Load config
    cerebro_config = CerebroConfig.from_name(config.model_preset)
    if config.learning_rate:
        cerebro_config.learning_rate = config.learning_rate
    if config.batch_size:
        cerebro_config.batch_size = config.batch_size
    if config.max_steps:
        cerebro_config.max_steps = config.max_steps
    
    # Create trainer
    trainer = CerebroTrainer(
        config=cerebro_config,
        device=config.device,
        output_dir=config.output_dir,
    )
    
    # Create mixed loader
    loader = mixer.create_loader(
        seq_len=cerebro_config.max_seq_len,
        batch_size=cerebro_config.batch_size,
    )
    
    print(f"\nTraining with mixed data sources...")
    print(f"Model: Cerebro-{config.model_preset}")
    print(f"Device: {trainer.device}")
    
    # Train (using the loader directly)
    trainer.model.train()
    total_loss = 0.0
    total_steps = 0
    
    import time
    from tqdm import tqdm
    import torch
    
    start_time = time.time()
    
    for epoch in range(config.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(trainer.device)
            labels = batch["labels"].to(trainer.device)
            
            # Update LR
            lr = trainer.scheduler.get_lr(trainer.global_step)
            for param_group in trainer.optimizer.param_groups:
                param_group["lr"] = lr
            
            # Forward
            if trainer.use_amp:
                with torch.autocast(device_type="cuda", dtype=trainer.autocast_dtype):
                    output = trainer.model(input_ids, labels=labels)
                    loss = output["loss"] / cerebro_config.grad_accum_steps
            else:
                output = trainer.model(input_ids, labels=labels)
                loss = output["loss"] / cerebro_config.grad_accum_steps
            
            # Backward
            loss.backward()
            
            # Accumulation step
            if (batch_idx + 1) % cerebro_config.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), cerebro_config.grad_clip)
                trainer.optimizer.step()
                trainer.optimizer.zero_grad()
                trainer.global_step += 1
                
                total_loss += loss.item() * cerebro_config.grad_accum_steps
                total_steps += 1
                
                if trainer.global_step % trainer.log_interval == 0:
                    avg_loss = total_loss / total_steps
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}", "step": trainer.global_step})
                
                if trainer.global_step >= cerebro_config.max_steps:
                    break
        
        if trainer.global_step >= cerebro_config.max_steps:
            break
    
    # Final save
    trainer.save_checkpoint("final")
    
    elapsed = time.time() - start_time
    avg_loss = total_loss / max(total_steps, 1)
    
    metrics = {
        "final_loss": avg_loss,
        "total_steps": trainer.global_step,
        "elapsed_seconds": elapsed,
        "tokens_per_second": (cerebro_config.batch_size * cerebro_config.max_seq_len * trainer.global_step) / max(elapsed, 1.0),
    }
    
    return PipelineResult(
        success=True,
        data_status=inventory.status,
        training_metrics=metrics,
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Cerebro Training Pipeline")
    parser.add_argument("--config", type=str, default="nano", help="Model preset")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--max-steps", type=int, default=None, help="Max training steps")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--datasets-dir", type=str, default="datasets", help="Datasets directory")
    parser.add_argument("--output", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--multi-source", action="store_true", help="Use multi-source mixing")
    
    args = parser.parse_args()
    
    config = PipelineConfig(
        model_preset=args.config,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        device=args.device,
        datasets_dir=args.datasets_dir,
        output_dir=args.output,
        resume_from=args.resume,
    )
    
    if args.multi_source:
        result = run_with_multi_source_mixing(config)
    else:
        result = run_pipeline(config)
    
    print(result.summary())
    
    sys.exit(0 if result.success else 1)
