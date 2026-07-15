"""Cerebro Generation Entry Point.

Usage:
    python generate.py --checkpoint path/to/ckpt --prompt "Hello world"
    python generate.py --prompt "Once upon a time"  # Random weights for testing
"""

from cerebro.cli import generate_command
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with Cerebro")
    parser.add_argument("--config", type=str, default=None, help="Model preset")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus top-p")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--repetition-penalty", type=float, default=1.1, help="Repetition penalty")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding")
    parser.add_argument("--device", type=str, default=None, help="Device")

    args = parser.parse_args()
    generate_command(args)


if __name__ == "__main__":
    main()
