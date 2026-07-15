"""Content safety and moderation for Cerebro.

Input/output filtering with:
- Keyword and pattern-based detection
- Configurable categories (harm, hate, self-harm, sexual, violence)
- Prompt injection / jailbreak detection
- Custom blocklists
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ContentSafetyResult:
    """Result of content safety check."""

    safe: bool
    blocked: bool = False
    category: str | None = None
    reason: str | None = None
    score: float = 0.0  # 0.0 = safe, 1.0 = blocked

    @classmethod
    def ok(cls) -> ContentSafetyResult:
        return cls(safe=True, blocked=False)

    @classmethod
    def block(cls, category: str, reason: str, score: float = 1.0) -> ContentSafetyResult:
        return cls(safe=False, blocked=True, category=category, reason=reason, score=score)


# ── Default blocklists ──

JAILBREAK_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)",
    r"(you\s+are|be)\s+(now\s+)?(DAN|GPT|unfiltered|uncensored)",
    r"act\s+as\s+(if\s+you\s+(are|were)\s+)?(DAN|a\s+different\s+AI)",
    r"pretend\s+(you\s+are|to\s+be)\s+(not|someone|a\s+different)",
    r"developer\s+mode|jailbreak|override\s+safety",
    r"system\s*prompt\s*[:=]|reveal\s+your\s+(system\s+)?(prompt|instructions)",
    r"you\s+(must|have\s+to|should)\s+(ignore|disregard|forget)",
    r"new\s+instructions?\s*[:=]|from\s+now\s+on\s+you\s+(are|will)",
    r"do\s+not\s+(follow|obey|respect)\s+(any\s+)?(rules?|guidelines?)",
]

HARM_CONTENT_PATTERNS = [
    r"how\s+to\s+(make|build|create|manufacture)\s+(a\s+)?(bomb|weapon|explosive)",
    r"instructions?\s+(for|on)\s+(making|building|manufacturing)\s+(weapons?|explosives?)",
    r"how\s+to\s+(kill|murder|assassinate)",
    r"ways\s+to\s+(hurt|harm|injure)\s+(yourself|myself|people)",
    r"suicide\s+method|best\s+way\s+to\s+(die|kill\s+myself)",
    r"child\s+(abuse|pornography|exploitation)",
    r"how\s+to\s+(hack|bypass\s+security|crack\s+password)",
    r"distribute\s+(malware|ransomware|virus)",
    r"cp\b|CSAM",
]

HATE_CONTENT_PATTERNS = [
    r"(hate|kill|exterminate)\s+all\s+\w+",
    r"racial\s+slur|ethnic\s+cleansing",
    r"white\s+supremac|neo[\s-]*nazi",
    r"genocide\s+(of|against)\s+\w+",
    r"terroris(ts?|m)\s+(are|is)\s+(good|right|hero)",
]

# ── Content Safety Filter ──


class ContentSafetyFilter:
    """Content safety filter for input and output moderation.

    Uses pattern matching for fast, configurable safety checks.
    Designed to be extended with ML-based classifiers.

    Args:
        blocklists: Optional dict of category -> list of regex patterns.
        enabled_categories: Categories to check (default: all).
        block_threshold: If a score exceeds this, block the content.
    """

    def __init__(
        self,
        blocklists: dict[str, list[str]] | None = None,
        enabled_categories: list[str] | None = None,
        block_threshold: float = 1.0,
    ) -> None:
        self.blocklists = blocklists or {
            "jailbreak": JAILBREAK_PATTERNS,
            "harm": HARM_CONTENT_PATTERNS,
            "hate": HATE_CONTENT_PATTERNS,
        }
        self.enabled_categories = enabled_categories or list(self.blocklists.keys())
        self.block_threshold = block_threshold

        # Compile patterns
        self._compiled: dict[str, list[re.Pattern]] = {}
        for category, patterns in self.blocklists.items():
            self._compiled[category] = [re.compile(p, re.IGNORECASE) for p in patterns]

    def check_input(self, text: str) -> ContentSafetyResult:
        """Check user input for safety violations.

        Args:
            text: User input text to check.

        Returns:
            ContentSafetyResult with check outcome.
        """
        if not text or not text.strip():
            return ContentSafetyResult.ok()

        text_lower = text.lower()

        for category in self.enabled_categories:
            if category not in self._compiled:
                continue
            for pattern in self._compiled[category]:
                match = pattern.search(text_lower)
                if match:
                    return ContentSafetyResult.block(
                        category=category,
                        reason=f"Input matches blocked pattern in '{category}' category",
                    )

        return ContentSafetyResult.ok()

    def check_output(self, text: str) -> ContentSafetyResult:
        """Check model output for safety violations.

        Args:
            text: Model output text to check.

        Returns:
            ContentSafetyResult with check outcome.
        """
        if not text or not text.strip():
            return ContentSafetyResult.ok()

        text_lower = text.lower()

        # Check output-specific patterns (harm content)
        for category in ["harm", "hate"]:
            if category not in self._compiled or category not in self.enabled_categories:
                continue
            for pattern in self._compiled[category]:
                match = pattern.search(text_lower)
                if match:
                    return ContentSafetyResult.block(
                        category=category,
                        reason=f"Output contains blocked content in '{category}' category",
                    )

        return ContentSafetyResult.ok()

    def check(self, text: str, is_input: bool = True) -> ContentSafetyResult:
        """Check content for safety violations.

        Args:
            text: Text to check.
            is_input: True for user input, False for model output.

        Returns:
            ContentSafetyResult with check outcome.
        """
        if is_input:
            return self.check_input(text)
        else:
            return self.check_output(text)

    def add_patterns(self, category: str, patterns: list[str]) -> None:
        """Add custom patterns to a category.

        Args:
            category: Category name (creates new if doesn't exist).
            patterns: List of regex pattern strings.
        """
        if category not in self._compiled:
            self._compiled[category] = []
        for p in patterns:
            self._compiled[category].append(re.compile(p, re.IGNORECASE))

    def load_from_file(self, filepath: str) -> None:
        """Load blocklist patterns from a JSON file.

        Expected format: {"category": ["pattern1", "pattern2", ...]}

        Args:
            filepath: Path to JSON file.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        for category, patterns in data.items():
            self.blocklists.setdefault(category, []).extend(patterns)
            if category not in self._compiled:
                self._compiled[category] = []
            for p in patterns:
                self._compiled[category].append(re.compile(p, re.IGNORECASE))

    @property
    def stats(self) -> dict:
        """Get safety filter statistics."""
        return {
            "categories": list(self.blocklists.keys()),
            "enabled": self.enabled_categories,
            "total_patterns": sum(len(v) for v in self._compiled.values()),
        }