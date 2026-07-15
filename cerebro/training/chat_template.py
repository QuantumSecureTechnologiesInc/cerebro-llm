"""Chat template formatting for supervised fine-tuning (SFT).

Converts multi-turn conversation data into tokenized training sequences
using Cerebro's special token format. Supports:
- OpenAI chat format
- ShareGPT format
- Alpaca instruction format
- Custom formats via templates
"""

from __future__ import annotations

import json
import os
from typing import Iterator
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class ChatMessage:
    """A single message in a conversation."""
    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> ChatMessage:
        return cls(
            role=d.get("role", "user"),
            content=d.get("content", ""),
            tool_calls=d.get("tool_calls"),
            tool_call_id=d.get("tool_call_id"),
        )


@dataclass
class ChatConversation:
    """A multi-turn conversation for training."""
    messages: list[ChatMessage]
    source: str = ""
    metadata: dict = field(default_factory=dict)


# ── Format templates ──

CEREBRO_TEMPLATE = {
    "system": "<|system|>\n{content}\n",
    "user": "<|user|>\n{content}\n",
    "assistant": "<|assistant|>\n{content}\n",
    "tool": "<|tool|>\n{content}\n",
    "bos": "<|beginoftext|>",
    "eos": "<|endoftext|>",
}

CHATML_TEMPLATE = {
    "system": "<|im_start|>system\n{content}\n<|im_end|>\n",
    "user": "<|im_start|>user\n{content}\n<|im_end|>\n",
    "assistant": "<|im_start|>assistant\n{content}\n<|im_end|>\n",
    "tool": "<|im_start|>tool\n{content}\n<|im_end|>\n",
    "bos": "",
    "eos": "",
}

ALPACA_TEMPLATE = {
    "instruction": "### Instruction:\n{content}\n\n",
    "input": "### Input:\n{content}\n\n",
    "response": "### Response:\n{content}\n",
    "bos": "",
    "eos": "\n\n",
}


class ChatFormatter:
    """Format conversations into training text using templates.

    Supports multiple conversation formats and applies masking
    to only compute loss on assistant responses.

    Args:
        template: Format template dict (role -> format string).
        mask_user: If True, mask user tokens from loss computation.
    """

    def __init__(
        self,
        template: dict | None = None,
        mask_user: bool = True,
    ) -> None:
        self.template = template or CEREBRO_TEMPLATE
        self.mask_user = mask_user

    def format_message(self, message: ChatMessage) -> str:
        """Format a single message using the template."""
        role = message.role
        fmt = self.template.get(role, self.template.get("user", "{content}"))
        return fmt.format(content=message.content)

    def format_conversation(self, conversation: ChatConversation) -> str:
        """Format a full conversation into a single training string.

        Args:
            conversation: ChatConversation with messages.

        Returns:
            Formatted string ready for tokenization.
        """
        parts = []
        bos = self.template.get("bos", "")
        if bos:
            parts.append(bos)

        for msg in conversation.messages:
            parts.append(self.format_message(msg))

        eos = self.template.get("eos", "")
        if eos:
            parts.append(eos)

        return "".join(parts)

    def format_for_loss_masking(
        self,
        conversation: ChatConversation,
        tokenizer=None,
    ) -> dict:
        """Format conversation with loss masking info.

        Returns a dict with:
        - 'text': Full formatted text
        - 'input_ids': Token IDs
        - 'labels': Token IDs with -100 for masked (non-assistant) tokens

        Args:
            conversation: ChatConversation.
            tokenizer: Optional tokenizer for encoding.

        Returns:
            Dict with text, input_ids, and labels.
        """
        full_text = self.format_conversation(conversation)

        if tokenizer is None:
            return {"text": full_text, "input_ids": None, "labels": None}

        # Tokenize each section separately to build loss mask
        input_ids = []
        labels = []

        bos = self.template.get("bos", "")
        if bos:
            bos_tokens = tokenizer.encode(bos, add_bos=False, add_eos=False)
            input_ids.extend(bos_tokens)
            labels.extend([-100] * len(bos_tokens))

        for msg in conversation.messages:
            msg_text = self.format_message(msg)
            msg_tokens = tokenizer.encode(msg_text, add_bos=False, add_eos=False)
            input_ids.extend(msg_tokens)

            # Only compute loss on assistant responses
            if msg.role == "assistant":
                labels.extend(msg_tokens)
            else:
                labels.extend([-100] * len(msg_tokens))

        eos = self.template.get("eos", "")
        if eos:
            eos_tokens = tokenizer.encode(eos, add_bos=False, add_eos=False)
            input_ids.extend(eos_tokens)
            labels.extend([-100] * len(eos_tokens))

        return {
            "text": full_text,
            "input_ids": input_ids,
            "labels": labels,
        }


def load_openai_format(filepath: str) -> list[ChatConversation]:
    """Load conversations from OpenAI-style JSONL.

    Expected format per line:
    {"messages": [{"role": "user", "content": "..."}, ...]}

    Args:
        filepath: Path to JSONL file.

    Returns:
        List of ChatConversation objects.
    """
    conversations = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            messages = [ChatMessage.from_dict(m) for m in obj.get("messages", [])]
            if messages:
                conversations.append(ChatConversation(
                    messages=messages,
                    source="openai",
                    metadata=obj.get("metadata", {}),
                ))

    return conversations


def load_sharegpt_format(filepath: str) -> list[ChatConversation]:
    """Load conversations from ShareGPT format.

    Expected format per line:
    {"conversations": [{"from": "human", "value": "..."}, ...]}

    Args:
        filepath: Path to JSON/JSONL file.

    Returns:
        List of ChatConversation objects.
    """
    role_map = {
        "human": "user",
        "gpt": "assistant",
        "system": "system",
        "observation": "tool",
        "function_call": "assistant",
    }

    conversations = []

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Try JSON array first, then JSONL
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            data = [data]
    except json.JSONDecodeError:
        data = [json.loads(line) for line in content.strip().split("\n") if line.strip()]

    for item in data:
        messages = []
        for turn in item.get("conversations", []):
            role = role_map.get(turn.get("from", "user"), "user")
            messages.append(ChatMessage(role=role, content=turn.get("value", "")))
        if messages:
            conversations.append(ChatConversation(
                messages=messages,
                source="sharegpt",
            ))

    return conversations


def load_alpaca_format(filepath: str) -> list[ChatConversation]:
    """Load from Alpaca instruction format.

    Expected format per line:
    {"instruction": "...", "input": "...", "output": "..."}

    Args:
        filepath: Path to JSON/JSONL file.

    Returns:
        List of ChatConversation objects.
    """
    conversations = []

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    try:
        data = json.loads(content)
        if isinstance(data, dict):
            data = [data]
    except json.JSONDecodeError:
        data = [json.loads(line) for line in content.strip().split("\n") if line.strip()]

    for item in data:
        messages = []
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output_text = item.get("output", "")

        if instruction:
            user_content = instruction
            if input_text:
                user_content += f"\n\n{input_text}"
            messages.append(ChatMessage(role="user", content=user_content))

        if output_text:
            messages.append(ChatMessage(role="assistant", content=output_text))

        if messages:
            conversations.append(ChatConversation(
                messages=messages,
                source="alpaca",
            ))

    return conversations


class SFTDataset:
    """Supervised Fine-Tuning dataset.

    Loads conversation data, formats it with templates, and
    produces tokenized training sequences with proper loss masking.

    Args:
        conversations: List of ChatConversation objects.
        formatter: ChatFormatter for text formatting.
        tokenizer: CerebroTokenizer for encoding.
        max_seq_len: Maximum sequence length (truncate longer).
    """

    def __init__(
        self,
        conversations: list[ChatConversation],
        formatter: ChatFormatter | None = None,
        tokenizer=None,
        max_seq_len: int = 4096,
    ) -> None:
        self.conversations = conversations
        self.formatter = formatter or ChatFormatter()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    @classmethod
    def from_file(
        cls,
        filepath: str,
        format: str = "openai",
        tokenizer=None,
        max_seq_len: int = 4096,
    ) -> SFTDataset:
        """Load SFT dataset from a file.

        Args:
            filepath: Path to data file.
            format: File format ('openai', 'sharegpt', 'alpaca').
            tokenizer: CerebroTokenizer.
            max_seq_len: Max sequence length.

        Returns:
            SFTDataset instance.
        """
        loaders = {
            "openai": load_openai_format,
            "sharegpt": load_sharegpt_format,
            "alpaca": load_alpaca_format,
        }

        if format not in loaders:
            raise ValueError(f"Unknown format '{format}'. Choose from: {list(loaders)}")

        conversations = loaders[format](filepath)
        return cls(conversations, tokenizer=tokenizer, max_seq_len=max_seq_len)

    def __len__(self) -> int:
        return len(self.conversations)

    def __getitem__(self, idx: int) -> dict:
        import torch

        conv = self.conversations[idx]
        formatted = self.formatter.format_for_loss_masking(conv, self.tokenizer)

        if formatted["input_ids"] is not None:
            input_ids = formatted["input_ids"][:self.max_seq_len]
            labels = formatted["labels"][:self.max_seq_len]

            # Pad to max_seq_len
            pad_len = self.max_seq_len - len(input_ids)
            if pad_len > 0:
                input_ids = input_ids + [0] * pad_len
                labels = labels + [-100] * pad_len

            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }

        # Fallback: just return formatted text
        return {"text": formatted["text"]}

    def statistics(self) -> dict:
        """Compute dataset statistics."""
        total_messages = sum(len(c.messages) for c in self.conversations)
        total_chars = sum(
            sum(len(m.content) for m in c.messages)
            for c in self.conversations
        )
        roles = {}
        for c in self.conversations:
            for m in c.messages:
                roles[m.role] = roles.get(m.role, 0) + 1

        return {
            "conversations": len(self.conversations),
            "total_messages": total_messages,
            "total_characters": total_chars,
            "avg_turns": total_messages / max(len(self.conversations), 1),
            "roles": roles,
        }
