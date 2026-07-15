"""HuggingFace Hub dataset search for Cerebro training.

Searches the HuggingFace Hub for datasets matching a query,
returning formatted results with metadata.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass


def _safe_str(s: str) -> str:
    """Encode string safely for Windows console (cp1252 fallback)."""
    return s.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(
        sys.stdout.encoding or "utf-8", errors="replace"
    )


@dataclass
class SearchResult:
    """A single search result from HuggingFace Hub."""
    dataset_id: str
    author: str
    description: str
    downloads: int
    likes: int
    tags: list[str]
    last_modified: str
    gated: bool
    private: bool

    def display(self, index: int | None = None) -> str:
        """Human-readable single-line + detail display."""
        prefix = f"[{index}] " if index is not None else ""
        lines = [
            f"{prefix}{_safe_str(self.dataset_id)}",
            f"     Author: {_safe_str(self.author)}",
            f"     Downloads: {self.downloads:,}  Likes: {self.likes}",
        ]
        if self.description:
            desc = _safe_str(self.description[:120].replace("\n", " "))
            lines.append(f"     {desc}")
        if self.tags:
            tags_str = _safe_str(', '.join(self.tags[:8]))
            lines.append(f"     Tags: {tags_str}")
        if self.gated:
            lines.append("     [GATED - requires access request]")
        return "\n".join(lines)

    def compact(self, index: int) -> str:
        """Single-line compact display."""
        desc = _safe_str((self.description or "")[:80].replace("\n", " "))
        gate = " [GATED]" if self.gated else ""
        return f"[{index:2d}] {_safe_str(self.dataset_id):50s}  DL:{self.downloads:>10,}  {desc}{gate}"


def search_datasets(
    query: str,
    limit: int = 20,
    author: str | None = None,
    task_category: str | None = None,
    sort: str = "downloads",
) -> list[SearchResult]:
    """Search HuggingFace Hub for datasets.

    Args:
        query: Search query string.
        limit: Maximum number of results (default 20).
        author: Filter by author/organization.
        task_category: Filter by task category (e.g., "text-generation").
        sort: Sort order - "downloads", "likes", "last_modified" (default "downloads").

    Returns:
        List of SearchResult objects.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print(
            "The 'huggingface_hub' library is required for Hub search.\n"
            "Install it with: pip install huggingface_hub",
            file=sys.stderr,
        )
        return []

    api = HfApi()

    # Map sort string to DatasetSort enum if available
    sort_kwargs = {}
    if sort:
        try:
            from huggingface_hub.hf_api import DatasetSort
            sort_map = {
                "downloads": DatasetSort.DOWNLOADS,
                "likes": DatasetSort.LIKES,
                "last_modified": DatasetSort.LAST_MODIFIED,
                "created": DatasetSort.CREATED_AT,
                "trending": DatasetSort.TRENDING_SCORE,
            }
            sort_kwargs["sort"] = sort_map.get(sort, DatasetSort.DOWNLOADS)
            sort_kwargs["direction"] = -1  # descending
        except ImportError:
            pass

    try:
        results = api.list_datasets(
            search=query,
            author=author,
            filter=task_category,
            limit=limit,
            **sort_kwargs,
        )
    except TypeError:
        # Fallback if sort/direction not supported in this version
        results = api.list_datasets(
            search=query,
            author=author,
            filter=task_category,
            limit=limit,
        )

    parsed = []
    for info in results:
        tags = info.tags if info.tags else []
        if isinstance(tags, list):
            tag_names = [t if isinstance(t, str) else str(t) for t in tags]
        else:
            tag_names = []

        parsed.append(SearchResult(
            dataset_id=info.id,
            author=info.author or "unknown",
            description=info.description or "",
            downloads=info.downloads or 0,
            likes=info.likes or 0,
            tags=tag_names,
            last_modified=str(info.last_modified) if info.last_modified else "",
            gated=info.gated or False,
            private=info.private or False,
        ))

    return parsed


def print_search_results(
    query: str,
    results: list[SearchResult],
    compact: bool = False,
) -> None:
    """Print search results in a formatted table.

    Args:
        query: The search query used.
        results: List of SearchResult objects.
        compact: Use compact single-line format.
    """
    print()
    print("=" * 90)
    print(f"HF HUB SEARCH: '{query}'")
    print(f"Found: {len(results)} results")
    print("=" * 90)

    if not results:
        print("No results found. Try a different query or broader terms.")
        print("=" * 90)
        print()
        return

    if compact:
        for i, r in enumerate(results):
            print(r.compact(i))
    else:
        for i, r in enumerate(results):
            print(r.display(i))
            print()

    print("=" * 90)
    print(f"Showing {len(results)} results. Use --limit to adjust, --compact for less detail.")
    print("=" * 90)
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Search HuggingFace Hub for datasets"
    )
    parser.add_argument(
        "query", type=str, nargs="?", default=None,
        help="Search query (required)",
    )
    parser.add_argument(
        "--query", "-q", type=str, dest="query_opt", default=None,
        help="Search query (alternative form)",
    )
    parser.add_argument(
        "--limit", "-n", type=int, default=20,
        help="Maximum results (default: 20)",
    )
    parser.add_argument(
        "--author", type=str, default=None,
        help="Filter by author/organization",
    )
    parser.add_argument(
        "--task", type=str, default=None,
        help="Filter by task category",
    )
    parser.add_argument(
        "--sort", type=str, default="downloads",
        choices=["downloads", "likes", "last_modified", "created", "trending"],
        help="Sort order (default: downloads)",
    )
    parser.add_argument(
        "--compact", action="store_true",
        help="Use compact single-line format",
    )

    args = parser.parse_args()

    query = args.query or args.query_opt
    if not query:
        parser.error("A search query is required. Usage: python -m pipeline.hf_search 'query text'")

    results = search_datasets(
        query=query,
        limit=args.limit,
        author=args.author,
        task_category=args.task,
        sort=args.sort,
    )

    print_search_results(query, results, compact=args.compact)