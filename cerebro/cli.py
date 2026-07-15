"""Cerebro CLI — unified command-line interface.

Commands:
    cerebro train       — Train a Cerebro model
    cerebro tokenize    — Tokenize raw text for training
    cerebro finetune    — LoRA/QLoRA fine-tuning
    cerebro align       — DPO/RLHF alignment training
    cerebro generate    — Generate text from a checkpoint
    cerebro benchmark   — Run inference benchmark
"""

from __future__ import annotations

import argparse
import sys
import time


def train_command(args: argparse.Namespace) -> None:
    """Train a Cerebro model."""
    # If advanced options are used, delegate to the advanced trainer
    advanced = getattr(args, 'curriculum', False) or getattr(args, 'streaming', False) or \
               getattr(args, 'distributed', False) or getattr(args, 'lora', False) or \
               getattr(args, 'dpo', False) or getattr(args, 'sft', False) or \
               getattr(args, 'data_mix', None)

    if advanced:
        from cerebro.training.train import main as train_main
        import sys
        # Reconstruct sys.argv for the advanced parser
        train_main()
        return

    from cerebro.config import CerebroConfig
    from cerebro.training.trainer import CerebroTrainer

    config = CerebroConfig.from_name(args.config)

    # Override config with CLI args
    if args.lr:
        config.learning_rate = args.lr
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.max_steps:
        config.max_steps = args.max_steps

    # Setup monitoring
    monitor = None
    if args.wandb or args.tensorboard:
        from cerebro.training.monitoring import MonitorCallback, WandBLogger, TensorBoardLogger
        wb = None
        tb = None
        if args.wandb:
            wb = WandBLogger(
                project=args.wandb_project or "cerebro",
                name=args.wandb_name,
                config={
                    "model": args.config,
                    "lr": config.learning_rate,
                    "batch_size": config.batch_size,
                    "max_steps": config.max_steps,
                },
            )
        if args.tensorboard:
            tb = TensorBoardLogger(
                log_dir=args.tensorboard_dir or "logs/tensorboard",
            )
        monitor = MonitorCallback(wandb_logger=wb, tb_logger=tb)

    trainer = CerebroTrainer(
        config=config,
        device=args.device or "auto",
        output_dir=args.output or "checkpoints",
        monitor=monitor,
    )

    # Load checkpoint if resuming
    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"Resumed from checkpoint: {args.resume}")

    # Count parameters
    params = trainer.model.estimate_params()
    print(f"Model: Cerebro-{args.config}")
    print(f"Parameters: {params['total']:,}")
    print(f"  Embedding: {params['embedding']:,}")
    print(f"  Encoder:   {params['encoder_layers']:,}")
    print(f"  Reasoning: {params['reasoning_core']:,}")
    print(f"Training on device: {trainer.device}")
    print(f"Data: {args.data or '(random tokens for validation)'}")
    print("-" * 60)

    results = trainer.train(
        data_dir=args.data,
        num_epochs=args.epochs or 1,
    )

    print("\n" + "=" * 60)
    print(f"Training complete!")
    print(f"  Final loss:     {results['final_loss']:.4f}")
    print(f"  Total steps:    {results['total_steps']}")
    print(f"  Elapsed:        {results['elapsed_seconds']:.1f}s")
    print(f"  Tokens/sec:     {results['tokens_per_second']:.0f}")
    print(f"  Checkpoints:    {args.output or 'checkpoints'}")


def generate_command(args: argparse.Namespace) -> None:
    """Generate text from a Cerebro checkpoint."""
    import torch
    from cerebro.config import CerebroConfig
    from cerebro.model.cerebro_model import Cerebro
    from cerebro.inference.engine import CerebroInferenceEngine
    from cerebro.tokenizer.tokenizer import CerebroTokenizer

    # Load config
    if args.config:
        config = CerebroConfig.from_name(args.config)
    else:
        config = CerebroConfig.nano()

    # Load tokenizer
    tokenizer = CerebroTokenizer(vocab_size=config.vocab_size)

    # Create model and engine
    model = Cerebro(config)
    engine = CerebroInferenceEngine(model, config, device=args.device or "auto")

    # Load checkpoint
    if args.checkpoint:
        engine.load_checkpoint(args.checkpoint)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("No checkpoint specified — using random weights (for testing only)")

    # Tokenize prompt
    prompt = args.prompt or "Once upon a time"
    tokens = tokenizer.encode(prompt, add_bos=True)
    input_ids = torch.tensor([tokens], dtype=torch.long)

    print(f"\nPrompt: {prompt}")
    print(f"Tokens: {len(tokens)}")
    print("-" * 60)

    # Generate
    start = time.time()
    generated = engine.generate(
        input_ids,
        max_new_tokens=args.max_tokens or 256,
        temperature=args.temperature or 0.7,
        top_p=args.top_p or 0.9,
        top_k=args.top_k or 50,
        repetition_penalty=args.repetition_penalty or 1.1,
        do_sample=not args.greedy,
    )
    elapsed = (time.time() - start) * 1000

    # Decode
    output_tokens = generated[0].tolist()
    text = tokenizer.decode(output_tokens[len(tokens):], skip_special=True)

    print(f"\n{text}")
    print("-" * 60)
    print(f"Generated {len(output_tokens) - len(tokens)} tokens in {elapsed:.0f}ms")
    print(f"Speed: {(len(output_tokens) - len(tokens)) / max(elapsed / 1000, 0.001):.1f} tok/s")


def benchmark_command(args: argparse.Namespace) -> None:
    """Run inference benchmark."""
    from cerebro.config import CerebroConfig
    from cerebro.model.cerebro_model import Cerebro
    from cerebro.inference.engine import CerebroInferenceEngine

    config = CerebroConfig.from_name(args.config)
    model = Cerebro(config)
    engine = CerebroInferenceEngine(model, config, device=args.device or "auto")

    params = model.estimate_params()
    print(f"Cerebro-{args.config}")
    print(f"Parameters: {params['total']:,}")
    print(f"Device: {engine.device}")
    print(f"Sequence length: {args.seq_len or 512}")
    print("-" * 60)

    results = engine.benchmark(
        seq_len=args.seq_len or 512,
        num_warmup=5,
        num_runs=20,
    )

    print(f"\nBenchmark Results:")
    print(f"  Mean latency:   {results['mean_latency_ms']:.2f} ms")
    print(f"  Median latency: {results['median_latency_ms']:.2f} ms")
    print(f"  P95 latency:    {results['p95_latency_ms']:.2f} ms")
    print(f"  Throughput:     {results['throughput_tok_s']:.1f} tok/s")


def serve_command(args: argparse.Namespace) -> None:
    """Start the Cerebro API server."""
    import torch
    from cerebro.config import CerebroConfig
    from cerebro.model.cerebro_model import Cerebro
    from cerebro.inference.engine import CerebroInferenceEngine
    from cerebro.tokenizer.tokenizer import CerebroTokenizer
    from cerebro.api import run_server

    config = CerebroConfig.from_name(args.config or "nano")
    tokenizer = CerebroTokenizer(vocab_size=config.vocab_size)
    model = Cerebro(config)
    engine = CerebroInferenceEngine(model, config, device=args.device or "auto")

    # Load checkpoint if provided
    if args.checkpoint:
        engine.load_checkpoint(args.checkpoint)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("Warning: No checkpoint specified — using random weights")

    print(f"Starting Cerebro API server on {args.host}:{args.port}")
    api_keys = args.api_keys.split(",") if args.api_keys else None
    run_server(
        host=args.host, port=args.port,
        engine=engine, tokenizer=tokenizer, api_keys=api_keys,
    )


def chat_command(args: argparse.Namespace) -> None:
    """Interactive multi-turn chat session."""
    import torch
    from cerebro.config import CerebroConfig
    from cerebro.model.cerebro_model import Cerebro
    from cerebro.inference.engine import CerebroInferenceEngine
    from cerebro.tokenizer.tokenizer import CerebroTokenizer
    from cerebro.chat import ConversationMemory

    config = CerebroConfig.from_name(args.config or "nano")
    tokenizer = CerebroTokenizer(vocab_size=config.vocab_size)
    model = Cerebro(config)
    engine = CerebroInferenceEngine(model, config, device=args.device or "auto")

    if args.checkpoint:
        engine.load_checkpoint(args.checkpoint)
        print(f"Loaded checkpoint: {args.checkpoint}")

    print("Cerebro Chat — Interactive Session")
    print("Type 'quit' or 'exit' to end. Type 'clear' to reset.")
    print("-" * 60)

    memory = ConversationMemory(
        system_prompt=args.system or "You are Cerebro, a helpful AI assistant.",
        max_tokens=7000,
        tokenizer=tokenizer,
    )

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if user_input.lower() == "clear":
            memory.clear()
            print("[Conversation cleared]")
            continue

        memory.add_user_message(user_input)
        prompt = memory.format_for_prompt()

        # Generate actual response
        tokens = tokenizer.encode(prompt, add_bos=True)
        input_ids = torch.tensor([tokens], dtype=torch.long)
        generated = engine.generate(
            input_ids,
            max_new_tokens=args.max_tokens or 256,
            temperature=args.temperature or 0.7,
            top_p=0.9,
            do_sample=True,
        )
        output_tokens = generated[0].tolist()
        response = tokenizer.decode(output_tokens[len(tokens):], skip_special=True)

        print(f"\nCerebro: {response}")
        memory.add_assistant_message(response)


def agent_command(args: argparse.Namespace) -> None:
    """Run the Cerebro autonomous agent."""
    import asyncio
    from cerebro.agents import CerebroAgent

    task = args.task
    print(f"Cerebro Agent — Task: {task}")
    print(f"Max steps: {args.max_steps or 20}")
    print("-" * 60)

    agent = CerebroAgent(max_steps=args.max_steps or 20)
    result = asyncio.run(agent.run(task))

    print(f"\nResult:\n{result}")
    print(f"\nState: {agent.get_state()}")


def tokenize_command(args: argparse.Namespace) -> None:
    """Tokenize raw text data into binary token shards."""
    from cerebro.training.tokenize import tokenize_to_shards

    print("Cerebro Tokenizer — Preprocessing Raw Text")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Shard size: {args.shard_size}")
    print("-" * 60)

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


def finetune_command(args: argparse.Namespace) -> None:
    """LoRA/QLoRA fine-tuning of a Cerebro model."""
    from cerebro.config import CerebroConfig
    from cerebro.model.cerebro_model import Cerebro
    from cerebro.training.finetune import LoRAConfig, LoRATrainer
    from cerebro.training.data import create_dataloader

    config = CerebroConfig.from_name(args.config)
    model = Cerebro(config)

    lora_config = LoRAConfig(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        quantize=args.qlora,
    )

    trainer = LoRATrainer(
        model, lora_config,
        lr=args.lr or 2e-4,
        device=args.device or "auto",
    )

    print(f"Cerebro Fine-Tuning — LoRA (rank={args.lora_rank})")
    print(trainer.param_summary())
    print("-" * 60)

    loader = create_dataloader(
        data_dir=args.data, seq_len=config.max_seq_len,
        batch_size=config.batch_size, vocab_size=config.vocab_size,
    )

    results = trainer.train(loader.dataset, num_epochs=args.epochs or 3, batch_size=config.batch_size)

    print(f"\nFine-tuning complete!")
    print(f"  Final loss: {results['final_loss']:.4f}")
    print(f"  Steps: {results['total_steps']}")
    print(f"  LoRA params: {results['lora']:,}")

    output = args.output or "checkpoints/lora.pt"
    trainer.save(output)
    print(f"  Saved to: {output}")


def align_command(args: argparse.Namespace) -> None:
    """DPO/RLHF alignment training."""
    from cerebro.config import CerebroConfig
    from cerebro.model.cerebro_model import Cerebro
    from cerebro.tokenizer.tokenizer import CerebroTokenizer
    from cerebro.training.alignment import DPOTrainer, PreferenceDataset

    config = CerebroConfig.from_name(args.config)
    model = Cerebro(config)
    tokenizer = CerebroTokenizer(vocab_size=config.vocab_size)

    print(f"Cerebro Alignment — DPO (beta={args.dpo_beta})")
    print(f"Preference data: {args.preference_data}")
    print("-" * 60)

    dataset = PreferenceDataset(args.preference_data, tokenizer=tokenizer)
    print(f"Loaded {len(dataset)} preference pairs")

    trainer = DPOTrainer(
        model, tokenizer=tokenizer,
        beta=args.dpo_beta,
        lr=args.lr or 1e-6,
        device=args.device or "auto",
    )

    result = trainer.train(dataset, num_epochs=args.epochs or 3, batch_size=config.batch_size)

    print(f"\nAlignment complete!")
    print(f"  Loss: {result.loss:.4f}")
    print(f"  Reward margin: {result.reward_margin:.4f}")
    print(f"  Accuracy: {result.accuracy:.4f}")

    import torch
    output = args.output or "checkpoints"
    from pathlib import Path
    out_dir = Path(output) / "aligned"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(out_dir / "model.pt"))
    print(f"  Saved to: {out_dir}")


def search_command(args: argparse.Namespace) -> None:
    """Run a web search with citations."""
    from cerebro.search import WebSearch

    query = args.query
    print(f"Searching: {query}")
    print("-" * 60)

    searcher = WebSearch(backend=args.backend or "duckduckgo")
    response = searcher.search(query)

    print(f"\nFound {len(response.results)} results in {response.search_time:.2f}s\n")
    for i, result in enumerate(response.results, 1):
        print(f"[{i}] {result.title}")
        print(f"    {result.url}")
        print(f"    {result.snippet[:150]}")
        print()


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="cerebro",
        description="Cerebro LLM — Cognitive Entropic Reasoning Engine",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── Train (advanced — delegates to training.train) ──
    train_parser = subparsers.add_parser("train", help="Train a Cerebro model")
    train_parser.add_argument("--config", type=str, default="nano", help="Model preset")
    train_parser.add_argument("--data", type=str, default=None, help="Training data directory")
    train_parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    train_parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    train_parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    train_parser.add_argument("--max-steps", type=int, default=None, help="Max training steps")
    train_parser.add_argument("--output", type=str, default="checkpoints", help="Output directory")
    train_parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    train_parser.add_argument("--device", type=str, default=None, help="Device: cuda|cpu|auto")
    train_parser.add_argument("--curriculum", action="store_true", help="Curriculum learning")
    train_parser.add_argument("--streaming", action="store_true", help="Streaming data loading")
    train_parser.add_argument("--distributed", action="store_true", help="Distributed training")
    train_parser.add_argument("--data-mix", type=str, default=None, help="Data mixing JSON config")
    train_parser.add_argument("--lora", action="store_true", help="LoRA fine-tuning mode")
    train_parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    train_parser.add_argument("--dpo", action="store_true", help="DPO alignment mode")
    train_parser.add_argument("--preference-data", type=str, default=None, help="Preference JSONL")
    train_parser.add_argument("--sft", action="store_true", help="Supervised fine-tuning")
    train_parser.add_argument("--sft-data", type=str, default=None, help="Chat data JSONL")
    train_parser.add_argument("--sft-format", type=str, default="openai", help="Chat format")
    train_parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    train_parser.add_argument("--wandb-project", type=str, default=None, help="WandB project name")
    train_parser.add_argument("--wandb-name", type=str, default=None, help="WandB run name")
    train_parser.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging")
    train_parser.add_argument("--tensorboard-dir", type=str, default=None, help="TensorBoard log directory")

    # ── Tokenize ──
    tok_parser = subparsers.add_parser("tokenize", help="Tokenize raw text data")
    tok_parser.add_argument("--input", type=str, help="Input directory with text files")
    tok_parser.add_argument("--files", type=str, help="Comma-separated input files")
    tok_parser.add_argument("--output", type=str, default="data/tokens", help="Output directory")
    tok_parser.add_argument("--shard-size", type=str, default="100M", help="Tokens per shard")
    tok_parser.add_argument("--no-bos", action="store_true", help="Don't add BOS tokens")
    tok_parser.add_argument("--no-eos", action="store_true", help="Don't add EOS tokens")
    tok_parser.add_argument("--no-recursive", action="store_true", help="No subdirectory search")

    # ── Fine-tune (LoRA) ──
    ft_parser = subparsers.add_parser("finetune", help="LoRA/QLoRA fine-tuning")
    ft_parser.add_argument("--config", type=str, default="nano", help="Model preset")
    ft_parser.add_argument("--data", type=str, default=None, help="Training data directory")
    ft_parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    ft_parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    ft_parser.add_argument("--output", type=str, default="checkpoints/lora.pt", help="Output path")
    ft_parser.add_argument("--device", type=str, default=None, help="Device")
    ft_parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    ft_parser.add_argument("--lora-alpha", type=float, default=32.0, help="LoRA alpha")
    ft_parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    ft_parser.add_argument("--qlora", action="store_true", help="Enable QLoRA (4-bit)")

    # ── Align (DPO/RLHF) ──
    align_parser = subparsers.add_parser("align", help="DPO/RLHF alignment training")
    align_parser.add_argument("--config", type=str, default="nano", help="Model preset")
    align_parser.add_argument("--preference-data", type=str, required=True, help="Preference JSONL")
    align_parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    align_parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    align_parser.add_argument("--dpo-beta", type=float, default=0.1, help="DPO beta")
    align_parser.add_argument("--output", type=str, default="checkpoints", help="Output dir")
    align_parser.add_argument("--device", type=str, default=None, help="Device")

    # ── Generate ──
    gen_parser = subparsers.add_parser("generate", help="Generate text")
    gen_parser.add_argument("--config", type=str, default=None, help="Model preset")
    gen_parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    gen_parser.add_argument("--prompt", type=str, default=None, help="Input prompt")
    gen_parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")
    gen_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    gen_parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling top-p")
    gen_parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    gen_parser.add_argument("--repetition-penalty", type=float, default=1.1, help="Repetition penalty")
    gen_parser.add_argument("--greedy", action="store_true", help="Use greedy decoding")
    gen_parser.add_argument("--device", type=str, default=None, help="Device")

    # ── Benchmark ──
    bench_parser = subparsers.add_parser("benchmark", help="Run inference benchmark")
    bench_parser.add_argument("--config", type=str, default="nano", help="Model preset")
    bench_parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    bench_parser.add_argument("--device", type=str, default=None, help="Device")

    # ── Serve (API Server) ──
    serve_parser = subparsers.add_parser("serve", help="Start the API server")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host")
    serve_parser.add_argument("--port", type=int, default=8000, help="Bind port")
    serve_parser.add_argument("--api-keys", type=str, default=None, help="Comma-separated API keys")
    serve_parser.add_argument("--config", type=str, default=None, help="Model preset")
    serve_parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    serve_parser.add_argument("--device", type=str, default=None, help="Device: cuda|cpu|auto")

    # ── Chat (Interactive) ──
    chat_parser = subparsers.add_parser("chat", help="Interactive multi-turn chat")
    chat_parser.add_argument("--system", type=str, default=None, help="System prompt")
    chat_parser.add_argument("--config", type=str, default=None, help="Model preset")
    chat_parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    chat_parser.add_argument("--device", type=str, default=None, help="Device: cuda|cpu|auto")
    chat_parser.add_argument("--max-tokens", type=int, default=256, help="Max response tokens")
    chat_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")

    # ── Agent ──
    agent_parser = subparsers.add_parser("agent", help="Run autonomous agent")
    agent_parser.add_argument("--task", type=str, required=True, help="Task description")
    agent_parser.add_argument("--max-steps", type=int, default=20, help="Max agent steps")

    # ── Search ──
    search_parser = subparsers.add_parser("search", help="Web search with citations")
    search_parser.add_argument("--query", type=str, required=True, help="Search query")
    search_parser.add_argument("--backend", type=str, default="duckduckgo", help="Search backend")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "train": train_command,
        "tokenize": tokenize_command,
        "finetune": finetune_command,
        "align": align_command,
        "generate": generate_command,
        "benchmark": benchmark_command,
        "serve": serve_command,
        "chat": chat_command,
        "agent": agent_command,
        "search": search_command,
    }

    if args.command not in commands:
        parser.print_help()
        sys.exit(1)

    commands[args.command](args)


if __name__ == "__main__":
    main()
