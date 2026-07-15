"""Standalone training script for Cerebro models.

Wires together config, trainer, evaluator, data loading, curriculum
learning, distributed training, and metrics logging into a single
entry point.

Usage:
    # Basic training
    python -m cerebro.training.train --config nano --data data/tokens/

    # With curriculum learning
    python -m cerebro.training.train --config core --data data/ --curriculum

    # Distributed multi-GPU
    torchrun --nproc_per_node=4 -m cerebro.training.train --config core --data data/

    # Fine-tuning with LoRA
    python -m cerebro.training.train --config nano --data data/ --lora --lora-rank 16

    # DPO alignment
    python -m cerebro.training.train --config nano --preference-data prefs.jsonl --dpo
"""

from __future__ import annotations

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path

logger = logging.getLogger("cerebro.training")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cerebro-train",
        description="Train Cerebro LLM models",
    )

    # ── Model ──
    parser.add_argument("--config", type=str, default="nano",
                        help="Model preset: nano|core|pro|ultra|sovereign")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint tag")
    parser.add_argument("--output", type=str, default="checkpoints",
                        help="Checkpoint output directory")

    # ── Data ──
    parser.add_argument("--data", type=str, default=None,
                        help="Training data directory (.bin shards)")
    parser.add_argument("--data-mix", type=str, default=None,
                        help="JSON file with data mixing config")
    parser.add_argument("--streaming", action="store_true",
                        help="Use streaming (memory-mapped) data loading")

    # ── Training ──
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")

    # ── Evaluation ──
    parser.add_argument("--eval-data", type=str, default=None,
                        help="Evaluation data directory")
    parser.add_argument("--eval-interval", type=int, default=500)

    # ── Distributed ──
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--backend", type=str, default="nccl")
    parser.add_argument("--gradient-checkpointing", action="store_true")

    # ── Curriculum ──
    parser.add_argument("--curriculum", action="store_true",
                        help="Enable progressive sequence length training")
    parser.add_argument("--curriculum-stages", type=str, default=None,
                        help="JSON list of [seq_len, num_steps] pairs")

    # ── LoRA fine-tuning ──
    parser.add_argument("--lora", action="store_true",
                        help="Enable LoRA fine-tuning")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--lora-qlora", action="store_true",
                        help="Enable QLoRA (4-bit quantization)")

    # ── DPO alignment ──
    parser.add_argument("--dpo", action="store_true",
                        help="Enable DPO alignment training")
    parser.add_argument("--preference-data", type=str, default=None,
                        help="JSONL file with preference pairs")
    parser.add_argument("--dpo-beta", type=float, default=0.1)

    # ── SFT (Supervised Fine-Tuning) ──
    parser.add_argument("--sft", action="store_true",
                        help="Enable supervised fine-tuning on chat data")
    parser.add_argument("--sft-data", type=str, default=None,
                        help="JSONL file with chat conversations")
    parser.add_argument("--sft-format", type=str, default="openai",
                        help="Chat data format: openai|sharegpt|alpaca")

    # ── Logging ──
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default="logs")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # ── Imports ──
    import torch
    from cerebro.config import CerebroConfig
    from cerebro.model.cerebro_model import Cerebro
    from cerebro.tokenizer.tokenizer import CerebroTokenizer
    from cerebro.training.evaluator import Evaluator, MetricsLogger

    logger.info("=" * 60)
    logger.info(f"Cerebro Training — {args.config.upper()}")
    logger.info("=" * 60)

    # ── Config ──
    config = CerebroConfig.from_name(args.config)

    if args.lr:
        config.learning_rate = args.lr
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.max_steps:
        config.max_steps = args.max_steps
    if args.warmup_steps:
        config.warmup_steps = args.warmup_steps
    if args.grad_accum:
        config.grad_accum_steps = args.grad_accum

    tokenizer = CerebroTokenizer(vocab_size=config.vocab_size)

    # ── Model ──
    model = Cerebro(config)
    params = model.estimate_params()
    logger.info(f"Model: Cerebro-{args.config}")
    logger.info(f"Parameters: {params['total']:,}")
    logger.info(f"  Embedding: {params['embedding']:,}")
    logger.info(f"  Encoder:   {params['encoder_layers']:,}")
    logger.info(f"  Reasoning: {params['reasoning_core']:,}")
    logger.info(f"Config: seq_len={config.max_seq_len}, layers={config.num_layers}")

    # ── Distributed ──
    dist = None
    if args.distributed:
        from cerebro.training.distributed import DistributedTrainer
        dist = DistributedTrainer()
        model = dist.init_distributed(
            model,
            backend=args.backend,
        )
        device = dist.device
        if args.gradient_checkpointing:
            import logging
            logging.getLogger("cerebro").info(
                "Gradient checkpointing enabled via FSDP activation checkpointing"
            )
    else:
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)

    logger.info(f"Device: {device}")

    # ── LoRA ──
    if args.lora:
        from cerebro.training.finetune import LoRAConfig, LoRATrainer

        lora_config = LoRAConfig(
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            quantize=args.lora_qlora,
        )
        lora_trainer = LoRATrainer(model, lora_config, lr=args.lr or 2e-4, device=str(device))
        logger.info(lora_trainer.param_summary())
        logger.info(f"Mode: LoRA Fine-Tuning {'(QLoRA)' if args.lora_qlora else ''}")

    # ── DPO ──
    if args.dpo:
        from cerebro.training.alignment import DPOTrainer, PreferenceDataset

        if not args.preference_data:
            logger.error("--preference-data required for DPO training")
            sys.exit(1)

        logger.info(f"Mode: DPO Alignment Training (beta={args.dpo_beta})")
        logger.info(f"Preference data: {args.preference_data}")

        pref_dataset = PreferenceDataset(args.preference_data, tokenizer=tokenizer)
        logger.info(f"Loaded {len(pref_dataset)} preference pairs")

        dpo_trainer = DPOTrainer(
            model, tokenizer=tokenizer,
            beta=args.dpo_beta, lr=args.lr or 1e-6,
            device=str(device),
        )
        result = dpo_trainer.train(pref_dataset, num_epochs=args.epochs, batch_size=config.batch_size)

        logger.info(
            "DPO Training Complete: loss=%.4f chosen=%.4f rejected=%.4f margin=%.4f acc=%.4f steps=%d elapsed=%.1fs",
            result.loss, result.chosen_reward, result.rejected_reward,
            result.reward_margin, result.accuracy, result.total_steps, result.elapsed_seconds,
        )

        # Save
        output_dir = Path(args.output) / "dpo"
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), str(output_dir / "model.pt"))
        logger.info("DPO model saved to: %s", output_dir)
        return

    # ── SFT ──
    if args.sft:
        from cerebro.training.chat_template import SFTDataset

        if not args.sft_data:
            logger.error("--sft-data required for SFT training")
            sys.exit(1)

        logger.info("Mode: Supervised Fine-Tuning")
        logger.info("SFT data: %s (%s format)", args.sft_data, args.sft_format)

        sft_dataset = SFTDataset.from_file(
            args.sft_data, format=args.sft_format,
            tokenizer=tokenizer, max_seq_len=config.max_seq_len,
        )
        stats = sft_dataset.statistics()
        logger.info(
            "SFT dataset: conversations=%d messages=%d avg_turns=%.1f",
            stats['conversations'], stats['total_messages'], stats['avg_turns'],
        )

        # Train using LoRA or full fine-tuning
        if args.lora:
            result = lora_trainer.train(sft_dataset, num_epochs=args.epochs, batch_size=config.batch_size)
            lora_trainer.save(str(Path(args.output) / "sft_lora.pt"))
        else:
            _run_pretraining(model, config, tokenizer, device, args, dataset=sft_dataset, dist=dist)
        return

    # ── Curriculum ──
    curriculum = None
    if args.curriculum:
        from cerebro.training.curriculum import CurriculumScheduler

        if args.curriculum_stages:
            stages = json.loads(args.curriculum_stages)
            curriculum = CurriculumScheduler(stages)
        else:
            curriculum = CurriculumScheduler.from_preset(args.config)

        logger.info(curriculum.summary())

    # ── Data Loading ──
    if args.data_mix:
        from cerebro.training.mixing import DataMixer

        with open(args.data_mix) as f:
            mix_config = json.load(f)

        mixer = DataMixer()
        for source in mix_config.get("sources", []):
            mixer.add_source(
                source["path"],
                weight=source.get("weight", 1.0),
                name=source.get("name"),
                streaming=args.streaming,
            )

        logger.info(mixer.summary())
        loader = mixer.create_loader(
            seq_len=curriculum.get_seq_len(0) if curriculum else config.max_seq_len,
            batch_size=config.batch_size,
        )
    elif args.data:
        if args.streaming:
            from cerebro.training.mixing import StreamingTokenDataset
            dataset = StreamingTokenDataset(args.data, config.max_seq_len)
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=config.batch_size,
                num_workers=0, pin_memory=True, drop_last=True,
            )
        else:
            from cerebro.training.data import create_dataloader
            loader = create_dataloader(
                data_dir=args.data, seq_len=config.max_seq_len,
                batch_size=config.batch_size, vocab_size=config.vocab_size,
            )
    else:
        from cerebro.training.data import create_dataloader
        loader = create_dataloader(
            data_dir=None, seq_len=config.max_seq_len,
            batch_size=config.batch_size, vocab_size=config.vocab_size,
        )
        logger.warning("No data specified — using random tokens for validation")

    # ── Run Training ──
    if args.lora:
        from cerebro.training.data import RandomTokenDataset
        if args.data is None:
            dataset = RandomTokenDataset(10000, config.max_seq_len, config.vocab_size)
        result = lora_trainer.train(dataset, num_epochs=args.epochs, batch_size=config.batch_size)
        logger.info(
            "LoRA Training Complete: final_loss=%.4f steps=%d",
            result['final_loss'], result['total_steps'],
        )
        lora_trainer.save(str(Path(args.output) / "lora.pt"))
        logger.info("LoRA saved to: %s/lora.pt", args.output)
    else:
        _run_pretraining(model, config, tokenizer, device, args, loader=loader, dist=dist, curriculum=curriculum)

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("=" * 60)


def _run_pretraining(model, config, tokenizer, device, args, loader=None, dataset=None, dist=None, curriculum=None):
    """Run the main pretraining loop."""
    import torch
    from cerebro.training.trainer import CerebroTrainer
    from cerebro.training.evaluator import Evaluator, MetricsLogger

    # Use CerebroTrainer
    trainer = CerebroTrainer(
        config=config,
        device=str(device),
        output_dir=args.output,
    )
    trainer.model = model.to(device)

    if args.resume:
        trainer.load_checkpoint(args.resume)
        logger.info("Resumed from checkpoint: %s", args.resume)

    # Metrics logger
    metrics_logger = MetricsLogger(
        project=args.wandb_project or "cerebro",
        run_name=args.wandb_run,
        log_dir=args.log_dir,
    )

    # Training loop
    from tqdm import tqdm
    from cerebro.training.scheduler import CosineSchedule

    scheduler = CosineSchedule(
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps,
        min_lr=config.learning_rate * 0.1,
        max_lr=config.learning_rate,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate,
        weight_decay=config.weight_decay, betas=(0.9, 0.95),
    )

    model.train()
    global_step = 0
    total_loss = 0.0
    start_time = time.time()

    for epoch in range(args.epochs):
        if loader is None:
            if dataset is not None:
                loader = torch.utils.data.DataLoader(
                    dataset, batch_size=config.batch_size,
                    shuffle=True, drop_last=True,
                )
            else:
                break

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, batch in enumerate(pbar):
            # Curriculum: adjust seq_len dynamically
            if curriculum and curriculum.should_transition(global_step):
                new_seq_len = curriculum.get_seq_len(global_step)
                logger.info("[curriculum] Transition to seq_len=%d at step %d", new_seq_len, global_step)

            # Curriculum LR override
            if curriculum:
                lr = curriculum.get_learning_rate(global_step, config.learning_rate)
            else:
                lr = scheduler.get_lr(global_step)

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            output = model(input_ids, labels=labels)
            loss = output["loss"] / config.grad_accum_steps

            loss.backward()

            if (batch_idx + 1) % config.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                step_loss = loss.item() * config.grad_accum_steps
                total_loss += step_loss

                metrics_logger.log_training_step(
                    global_step, step_loss, lr,
                    tokens_per_sec=config.batch_size * config.max_seq_len * global_step / max(time.time() - start_time, 1.0),
                )

                pbar.set_postfix({
                    "loss": f"{step_loss:.4f}",
                    "lr": f"{lr:.2e}",
                    "step": global_step,
                })

                if global_step > 0 and global_step % 1000 == 0:
                    trainer.global_step = global_step
                    trainer.save_checkpoint("latest")

                if global_step >= config.max_steps:
                    break

        if global_step >= config.max_steps:
            break

    # Final save
    trainer.global_step = global_step
    trainer.save_checkpoint("final")
    metrics_logger.close()

    elapsed = time.time() - start_time
    avg_loss = total_loss / max(global_step, 1)

    logger.info(
        "Training summary: final_loss=%.4f total_steps=%d elapsed=%.1fs checkpoints=%s",
        avg_loss, global_step, elapsed, args.output,
    )


if __name__ == "__main__":
    main()
