"""Web search and citation module for Cerebro.

Perplexity-style grounded search with:
- Web search via multiple backends
- Result extraction and ranking
- Inline citation formatting [1], [2], etc.
- Source attribution in responses
- Cached results for efficiency
"""

from __future__ import annotations

import json
import hashlib
import time
import re
import logging
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger("cerebro.search")


@dataclass
class SearchResult:
    """A single web search result."""
    title: str
    url: str
    snippet: str
    score: float = 0.0
    source: str = "web"

    def to_citation(self, index: int) -> str:
        domain = urlparse(self.url).netloc
        return f"[{index}] {self.title} ({domain})"

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "score": self.score,
        }


@dataclass
class SearchResponse:
    """Complete search response with results and formatted answer."""
    query: str
    results: list[SearchResult]
    answer: str = ""
    citations: list[str] = field(default_factory=list)
    search_time: float = 0.0

    def format_citations(self) -> str:
        """Format all citations as numbered references."""
        lines = ["\nSources:"]
        for i, result in enumerate(self.results, 1):
            lines.append(f"  {result.to_citation(i)}")
            lines.append(f"       {result.url}")
        return "\n".join(lines)

    def format_answer_with_citations(self) -> str:
        """Format the answer with inline citation numbers."""
        if not self.answer:
            return ""
        output = self.answer
        if self.results:
            output += self.format_citations()
        return output

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "answer": self.answer,
            "citations": self.citations,
        }


class WebSearch:
    """Web search engine for grounded generation.

    Supports multiple search backends:
    - DuckDuckGo (free, no API key required)
    - SerpAPI (requires API key)
    - Brave Search (requires API key)
    - Custom search providers

    Args:
        backend: Search backend to use.
        api_key: API key for the search backend.
        max_results: Maximum results per query.
        cache_ttl: Cache time-to-live in seconds.
    """

    def __init__(
        self,
        backend: str = "duckduckgo",
        api_key: str | None = None,
        max_results: int = 10,
        cache_ttl: int = 300,
    ) -> None:
        self.backend = backend
        self.api_key = api_key
        self.max_results = max_results
        self.cache_ttl = cache_ttl
        self._cache: dict[str, tuple[float, list[SearchResult]]] = {}

    def search(self, query: str) -> SearchResponse:
        """Search the web for a query.

        Args:
            query: Search query string.

        Returns:
            SearchResponse with results.
        """
        start = time.time()

        # Check cache
        cache_key = hashlib.md5(query.encode()).hexdigest()
        if cache_key in self._cache:
            cached_time, cached_results = self._cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                return SearchResponse(
                    query=query,
                    results=cached_results,
                    search_time=0.0,
                )

        # Execute search
        if self.backend == "duckduckgo":
            results = self._search_duckduckgo(query)
        elif self.backend == "serpapi":
            results = self._search_serpapi(query)
        elif self.backend == "brave":
            results = self._search_brave(query)
        else:
            results = self._search_duckduckgo(query)

        elapsed = time.time() - start

        # Cache results
        self._cache[cache_key] = (time.time(), results)

        return SearchResponse(
            query=query,
            results=results,
            search_time=elapsed,
        )

    def _search_duckduckgo(self, query: str) -> list[SearchResult]:
        """Search using DuckDuckGo (no API key required)."""
        try:
            import requests
            from html.parser import HTMLParser

            resp = requests.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                headers={"User-Agent": "Cerebro/1.0"},
                timeout=10,
            )

            results = []

            class DDGParser(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.results = []
                    self._current = {}
                    self._in_result = False
                    self._in_snippet = False

                def handle_starttag(self, tag, attrs):
                    attrs_dict = dict(attrs)
                    if tag == "a" and "result__a" in attrs_dict.get("class", ""):
                        self._in_result = True
                        self._current["url"] = attrs_dict.get("href", "")
                        self._current["title"] = ""
                    elif tag == "a" and "result__snippet" in attrs_dict.get("class", ""):
                        self._in_snippet = True
                        self._current["snippet"] = ""

                def handle_data(self, data):
                    if self._in_result:
                        self._current["title"] = self._current.get("title", "") + data
                    elif self._in_snippet:
                        self._current["snippet"] = self._current.get("snippet", "") + data

                def handle_endtag(self, tag):
                    if tag == "a" and self._in_result:
                        self._in_result = False
                    elif tag == "a" and self._in_snippet:
                        self._in_snippet = False
                        if self._current.get("title") and self._current.get("url"):
                            self.results.append(self._current)
                            self._current = {}

            parser = DDGParser()
            parser.feed(resp.text)

            return [
                SearchResult(
                    title=r.get("title", "").strip(),
                    url=r.get("url", ""),
                    snippet=r.get("snippet", "").strip(),
                    score=1.0 / (i + 1),
                )
                for i, r in enumerate(parser.results[:self.max_results])
            ]

        except (OSError, ValueError, RuntimeError) as e:
            logger.warning("DuckDuckGo search failed: %s", e)
            return []

    def _search_serpapi(self, query: str) -> list[SearchResult]:
        """Search using SerpAPI (requires API key)."""
        if not self.api_key:
            return []
        try:
            import requests
            resp = requests.get(
                "https://serpapi.com/search.json",
                params={"q": query, "api_key": self.api_key, "num": self.max_results},
                timeout=10,
            )
            data = resp.json()
            results = []
            for i, item in enumerate(data.get("organic_results", [])[:self.max_results]):
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    score=1.0 / (i + 1),
                ))
            return results
        except (OSError, ValueError, RuntimeError) as e:
            logger.warning("SerpAPI search failed: %s", e)
            return []

    def _search_brave(self, query: str) -> list[SearchResult]:
        """Search using Brave Search API (requires API key)."""
        if not self.api_key:
            return []
        try:
            import requests
            resp = requests.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": self.max_results},
                headers={"X-Subscription-Token": self.api_key},
                timeout=10,
            )
            data = resp.json()
            results = []
            for i, item in enumerate(data.get("web", {}).get("results", [])[:self.max_results]):
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("description", ""),
                    score=1.0 / (i + 1),
                ))
            return results
        except (OSError, ValueError, RuntimeError) as e:
            logger.warning("Brave search failed: %s", e)
            return []

    def format_context_for_prompt(self, response: SearchResponse) -> str:
        """Format search results as context for the model prompt.

        Args:
            response: SearchResponse with results.

        Returns:
            Formatted context string.
        """
        if not response.results:
            return ""

        lines = ["Based on the following search results:"]
        for i, result in enumerate(response.results, 1):
            lines.append(f"\n[{i}] {result.title}")
            lines.append(f"    {result.snippet}")
        return "\n".join(lines)
