"""Cerebro: Cognitive Entropic Reasoning Engine with Bounded Recursive Optimization.

A next-generation Large Language Model with:
- Hybrid Transformer-Quaternion Architecture (HTQA)
- Bounded recursion with self-verification
- Tool calling and agent framework
- Code interpreter sandbox
- Web search with citations (Perplexity-style)
- Computer use (browser, files, shell)
- RAG (Retrieval-Augmented Generation)
- Vision (multimodal input)
- Production API server with SSE streaming
- LoRA/QLoRA fine-tuning and DPO alignment
- Curriculum learning and dataset mixing
- Streaming out-of-core data loading
- Post-quantum cryptography security
"""

from cerebro.config import CerebroConfig
from cerebro.chat import ConversationMemory, Message
from cerebro.tools import ToolRegistry, ToolCall, ToolResult, ToolCallParser
from cerebro.search import WebSearch, SearchResponse
from cerebro.rag import RAGPipeline, VectorStore
from cerebro.computer import ComputerUse
from cerebro.agents import CerebroAgent
from cerebro.vision import VisionEncoder, VisionProcessor
from cerebro.training.curriculum import CurriculumScheduler
from cerebro.training.mixing import DataMixer, StreamingTokenDataset
from cerebro.training.finetune import LoRAConfig, LoRALinear, apply_lora
from cerebro.training.alignment import DPOTrainer, PreferenceDataset
from cerebro.training.chat_template import ChatFormatter, SFTDataset
from cerebro.training.tokenize import tokenize_to_shards

__version__ = "1.1.0"

__all__ = [
    # Config
    "CerebroConfig",
    # Chat
    "ConversationMemory",
    "Message",
    # Tools
    "ToolRegistry",
    "ToolCall",
    "ToolResult",
    "ToolCallParser",
    # Search
    "WebSearch",
    "SearchResponse",
    # RAG
    "RAGPipeline",
    "VectorStore",
    # Computer Use
    "ComputerUse",
    # Agent
    "CerebroAgent",
    # Vision
    "VisionEncoder",
    "VisionProcessor",
    # Training
    "CurriculumScheduler",
    "DataMixer",
    "StreamingTokenDataset",
    "LoRAConfig",
    "LoRALinear",
    "apply_lora",
    "DPOTrainer",
    "PreferenceDataset",
    "ChatFormatter",
    "SFTDataset",
    "tokenize_to_shards",
]
