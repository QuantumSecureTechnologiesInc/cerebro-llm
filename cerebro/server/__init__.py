"""Cerebro server package — web UI and static assets for the API server."""

from pathlib import Path

STATIC_DIR = Path(__file__).parent / "static"

__all__ = ["STATIC_DIR"]