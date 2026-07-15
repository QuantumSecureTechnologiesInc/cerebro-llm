"""Cerebro Training Entry Point.

Usage:
    python train.py --config nano --data path/to/tokens
    python train.py --config nano  # Uses random tokens for validation
"""

from cerebro.cli import train_command
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Cerebro model")
    parser.add_argument("--config", type=str, default="nano", help="Model preset")
    parser.add_argument("--data", type=str, default=None, help="Training data directory")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--max-steps", type=int, default=None, help="Max training steps")
    parser.add_argument("--output", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Device: cuda|cpu|auto")

    args = parser.parse_args()
    train_command(args)


if __name__ == "__main__":
    main()
