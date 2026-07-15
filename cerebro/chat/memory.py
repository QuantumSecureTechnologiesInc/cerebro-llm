"""Conversation memory and message management.

Multi-turn chat with system prompts, sliding window context,
and token-aware history management.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Optional
from copy import deepcopy


@dataclass
class Message:
    """A single message in a conversation."""
    role: str  # "system", "user", "assistant", "tool"
    content: str
    timestamp: float = field(default_factory=time.time)
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.name:
            d["name"] = self.name
        return d

    @classmethod
    def from_dict(cls, data: dict) -> Message:
        return cls(
            role=data.get("role", "user"),
            content=data.get("content", ""),
            tool_calls=data.get("tool_calls"),
            tool_call_id=data.get("tool_call_id"),
            name=data.get("name"),
        )


class ConversationMemory:
    """Manages multi-turn conversation history.

    Features:
    - System prompt persistence
    - Sliding window context (token-aware)
    - Message history with role tracking
    - Tool call/result tracking
    """

    def __init__(
        self,
        system_prompt: str | None = None,
        max_tokens: int = 7000,
        tokenizer=None,
    ) -> None:
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer
        self.messages: list[Message] = []
        self._token_counts: list[int] = []

    def add_message(
        self,
        role: str,
        content: str,
        tool_calls: list[dict] | None = None,
        tool_call_id: str | None = None,
        name: str | None = None,
    ) -> Message:
        """Add a message to the conversation."""
        msg = Message(
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
            name=name,
        )
        self.messages.append(msg)
        self._token_counts.append(self._count_tokens(content))
        self._trim_if_needed()
        return msg

    def add_user_message(self, content: str) -> Message:
        return self.add_message("user", content)

    def add_assistant_message(
        self,
        content: str,
        tool_calls: list[dict] | None = None,
    ) -> Message:
        return self.add_message("assistant", content, tool_calls=tool_calls)

    def add_tool_result(self, tool_call_id: str, content: str, name: str = "") -> Message:
        return self.add_message(
            "tool", content, tool_call_id=tool_call_id, name=name
        )

    def get_messages(self) -> list[dict]:
        """Get conversation as list of message dicts for the model."""
        result = []
        if self.system_prompt:
            result.append({"role": "system", "content": self.system_prompt})
        for msg in self.messages:
            result.append(msg.to_dict())
        return result

    def format_for_prompt(self) -> str:
        """Format conversation as a single string for tokenization."""
        parts = []
        if self.system_prompt:
            parts.append(f"<|system|>\n{self.system_prompt}\n")

        for msg in self.messages:
            if msg.role == "user":
                parts.append(f"<|user|>\n{msg.content}\n")
            elif msg.role == "assistant":
                parts.append(f"<|assistant|>\n{msg.content}\n")
            elif msg.role == "tool":
                parts.append(f"<|tool|>\n{msg.content}\n")

        parts.append("<|assistant|>\n")
        return "".join(parts)

    def clear(self) -> None:
        """Clear all messages (keeps system prompt)."""
        self.messages.clear()
        self._token_counts.clear()

    def _count_tokens(self, text: str) -> int:
        """Estimate token count for a string."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        # Rough estimate: ~4 chars per token
        return max(1, len(text) // 4)

    def _trim_if_needed(self) -> None:
        """Remove oldest messages if token count exceeds limit."""
        total = sum(self._token_counts)
        while total > self.max_tokens and len(self.messages) > 1:
            removed = self.messages.pop(0)
            removed_count = self._token_counts.pop(0)
            total -= removed_count

    def to_json(self) -> str:
        """Serialize conversation to JSON."""
        data = {
            "system_prompt": self.system_prompt,
            "max_tokens": self.max_tokens,
            "messages": [msg.to_dict() for msg in self.messages],
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> ConversationMemory:
        """Deserialize conversation from JSON."""
        data = json.loads(json_str)
        mem = cls(
            system_prompt=data.get("system_prompt"),
            max_tokens=data.get("max_tokens", 7000),
        )
        for msg_data in data.get("messages", []):
            msg = Message.from_dict(msg_data)
            mem.messages.append(msg)
            mem._token_counts.append(mem._count_tokens(msg.content))
        return mem
