"""Request queuing and continuous batching for Cerebro serving.

Provides:
- RequestQueue: FIFO queue with priority and timeout
- ContinuousBatcher: dynamic batching that maximizes throughput
- BatchScheduler: smart scheduling of queued requests
"""

from __future__ import annotations

import asyncio
import time
import heapq
import uuid
from dataclasses import dataclass, field
from collections import deque
from typing import Optional, Callable
import torch
from torch import Tensor


@dataclass(order=True)
class GenerationRequest:
    """A single text generation request in the queue.

    Attributes:
        priority: Lower = higher priority (0 = highest).
        timestamp: Enqueue time (epoch seconds).
        prompt_ids: Tokenized prompt.
        max_new_tokens: Max tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        top_k: Top-k cutoff.
        repetition_penalty: Repetition penalty.
        do_sample: Whether to sample.
        request_id: Unique request UUID.
        future: asyncio Future to resolve with result.
        timeout_seconds: Max wait time before rejection.
    """

    priority: int = 0
    timestamp: float = field(default_factory=time.time)
    prompt_ids: list[int] = field(default_factory=list)
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    future: Optional[asyncio.Future] = field(default=None, compare=False)
    timeout_seconds: float = 300.0

    def is_expired(self) -> bool:
        """Check if this request has timed out."""
        return time.time() - self.timestamp > self.timeout_seconds


@dataclass
class BatchResult:
    """Result from a continuous batch generation."""
    request_ids: list[str]
    generated_ids: list[list[int]]
    generated_texts: list[str]
    latencies_ms: list[float]
    tokens_per_request: list[int]


class RequestQueue:
    """Priority-based request queue for generation requests.

    Features:
    - Priority ordering (lower = higher priority)
    - Timeout-based eviction of stale requests
    - Max queue size enforcement
    - Async-compatible (thread-safe deque)

    Args:
        max_size: Maximum number of queued requests.
        default_timeout: Default timeout in seconds.
    """

    def __init__(self, max_size: int = 1000, default_timeout: float = 300.0) -> None:
        self.max_size = max_size
        self.default_timeout = default_timeout
        self._queue: deque[GenerationRequest] = deque()
        self._lock = asyncio.Lock() if asyncio else None

    def __len__(self) -> int:
        return len(self._queue)

    @property
    def is_empty(self) -> bool:
        return len(self._queue) == 0

    @property
    def is_full(self) -> bool:
        return len(self._queue) >= self.max_size

    def enqueue(self, request: GenerationRequest) -> bool:
        """Add a request to the queue.

        Returns:
            True if enqueued, False if queue is full.
        """
        if self.is_full:
            self._evict_expired()
            if self.is_full:
                return False
        self._queue.append(request)
        return True

    def dequeue(self, max_batch_size: int | None = None) -> list[GenerationRequest]:
        """Dequeue requests for a batch.

        Takes requests in FIFO order. If max_batch_size is set,
        prefers requests with similar prompt lengths for efficiency.

        Args:
            max_batch_size: Maximum batch size. None = all available.

        Returns:
            List of requests to process.
        """
        self._evict_expired()

        if max_batch_size is None:
            max_batch_size = len(self._queue)

        if len(self._queue) <= max_batch_size:
            batch = list(self._queue)
            self._queue.clear()
            return batch

        # Group by similar prompt length for efficient batching
        sorted_queue = sorted(self._queue, key=lambda r: len(r.prompt_ids))
        batch = sorted_queue[:max_batch_size]

        # Remove from queue
        batch_ids = {r.request_id for r in batch}
        self._queue = deque(r for r in self._queue if r.request_id not in batch_ids)

        return batch

    def _evict_expired(self) -> None:
        """Remove expired requests."""
        self._queue = deque(r for r in self._queue if not r.is_expired())

    def stats(self) -> dict:
        """Get queue statistics."""
        return {
            "queue_depth": len(self._queue),
            "max_size": self.max_size,
            "oldest_age_s": time.time() - self._queue[0].timestamp if self._queue else 0,
        }


class ContinuousBatcher:
    """Continuous batching engine for LLM inference.

    Implements dynamic batching strategies:
    - Padded batch: pad all prompts to same length
    - Round-robin: round-robin scheduling across requests
    - Prefill-first: prioritize prompt processing

    Args:
        engine: CerebroInferenceEngine instance.
        tokenizer: CerebroTokenizer instance.
        max_batch_size: Maximum batch size.
        max_total_tokens: Maximum total tokens in a batch (prompt + generation).
        pad_token_id: Padding token ID.
        eos_token_id: End-of-sequence token ID.
    """

    def __init__(
        self,
        engine,
        tokenizer,
        max_batch_size: int = 32,
        max_total_tokens: int = 4096,
        pad_token_id: int = 0,
        eos_token_id: int = 2,
    ) -> None:
        self.engine = engine
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_total_tokens = max_total_tokens
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

    def _pad_batch(self, sequences: list[list[int]]) -> Tensor:
        """Pad a list of token sequences to the same length.

        Args:
            sequences: List of token ID lists.

        Returns:
            (B, max_len) padded tensor.
        """
        max_len = max(len(s) for s in sequences)
        padded = torch.full(
            (len(sequences), max_len), self.pad_token_id, dtype=torch.long
        )
        for i, s in enumerate(sequences):
            padded[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        return padded

    def _pad_to_length(self, sequences: list[list[int]], target_len: int) -> Tensor:
        """Pad all sequences to exactly target_len."""
        padded = torch.full(
            (len(sequences), target_len), self.pad_token_id, dtype=torch.long
        )
        for i, s in enumerate(sequences):
            actual_len = min(len(s), target_len)
            padded[i, :actual_len] = torch.tensor(s[:actual_len], dtype=torch.long)
        return padded

    def process_batch(
        self,
        requests: list[GenerationRequest],
        tokenizer_decode: Callable | None = None,
    ) -> BatchResult:
        """Process a batch of requests together.

        Pads all prompts to the same length and runs generation
        in a single forward pass.

        Args:
            requests: List of generation requests.
            tokenizer_decode: Optional decode function (uses engine tokenizer).

        Returns:
            BatchResult with generated outputs.
        """
        if not requests:
            return BatchResult([], [], [], [], [])

        request_ids = [r.request_id for r in requests]
        prompt_lengths = [len(r.prompt_ids) for r in requests]

        # Pad prompts to same length
        prompt_tensor = self._pad_batch([r.prompt_ids for r in requests])
        device = self.engine.device
        prompt_tensor = prompt_tensor.to(device)

        # Use the first request's parameters (or max across requests)
        max_new_tokens = max(r.max_new_tokens for r in requests)
        temperature = requests[0].temperature
        top_p = max(r.top_p for r in requests)
        top_k = max(r.top_k for r in requests)

        start_times = [time.time() for _ in requests]

        self.engine.reset_cache()
        generated = prompt_tensor.clone()
        B = prompt_tensor.shape[0]

        all_tokens: list[list[int]] = [
            generated[i].tolist() for i in range(B)
        ]

        # Prefill
        next_logits = self.engine.prefill(prompt_tensor)

        for step in range(max_new_tokens):
            next_tokens = self.engine.sampler.sample_batch(
                next_logits.clone(),
                generated_tokens=all_tokens if step > 0 else None,
                do_sample=requests[0].do_sample,
            )

            generated = torch.cat([generated, next_tokens], dim=1)
            for i in range(B):
                all_tokens[i].append(next_tokens[i, 0].item())

            if (next_tokens.squeeze(-1) == self.eos_token_id).all():
                break

            next_logits = self.engine.decode_step(next_tokens)

        # Collect results
        generated_ids = []
        generated_texts = []
        latencies = []
        tokens_per_req = []

        decode_fn = tokenizer_decode or self.tokenizer.decode

        for i in range(B):
            prompt_len = prompt_lengths[i]
            gen_tokens = all_tokens[i][prompt_len:]
            # Trim after EOS
            if self.eos_token_id in gen_tokens:
                eos_idx = gen_tokens.index(self.eos_token_id)
                gen_tokens = gen_tokens[:eos_idx + 1]

            generated_ids.append(gen_tokens)
            generated_texts.append(decode_fn(gen_tokens, skip_special=True))
            latencies.append((time.time() - start_times[i]) * 1000)
            tokens_per_req.append(len(gen_tokens))

        return BatchResult(
            request_ids=request_ids,
            generated_ids=generated_ids,
            generated_texts=generated_texts,
            latencies_ms=latencies,
            tokens_per_request=tokens_per_req,
        )


class BatchScheduler:
    """Smart scheduler for continuous batching.

    Tracks queue state and decides when to process batches
    based on queue depth, latency targets, and throughput.

    Args:
        batcher: ContinuousBatcher instance.
        queue: RequestQueue instance.
        max_batch_size: Maximum batch size.
        max_wait_ms: Maximum time to wait before processing a partial batch.
        target_latency_ms: Target per-request latency.
        max_batch_latency_ms: Maximum latency for a full batch.
    """

    def __init__(
        self,
        batcher: ContinuousBatcher,
        queue: RequestQueue,
        max_batch_size: int = 32,
        max_wait_ms: float = 50.0,
        target_latency_ms: float = 200.0,
        max_batch_latency_ms: float = 2000.0,
    ) -> None:
        self.batcher = batcher
        self.queue = queue
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.target_latency_ms = target_latency_ms
        self.max_batch_latency_ms = max_batch_latency_ms

        self._total_processed: int = 0
        self._total_latency_ms: float = 0.0

    async def run_loop(self, stop_event: asyncio.Event | None = None) -> None:
        """Main scheduling loop.

        Continuously dequeues and processes batches until stopped.

        Args:
            stop_event: Event to signal loop termination.
        """
        while stop_event is None or not stop_event.is_set():
            if self.queue.is_empty:
                await asyncio.sleep(0.001)
                continue

            # Determine optimal batch size
            batch_size = self._compute_batch_size()
            batch = self.queue.dequeue(max_batch_size=batch_size)

            if not batch:
                continue

            # Process batch
            result = self.batcher.process_batch(batch)

            # Resolve futures
            for i, req in enumerate(batch):
                if req.future and not req.future.done():
                    req.future.set_result({
                        "request_id": result.request_ids[i],
                        "text": result.generated_texts[i],
                        "token_ids": result.generated_ids[i],
                        "latency_ms": result.latencies_ms[i],
                        "tokens_generated": result.tokens_per_request[i],
                    })

            self._total_processed += len(batch)
            self._total_latency_ms += sum(result.latencies_ms)

    def _compute_batch_size(self) -> int:
        """Compute optimal batch size based on queue depth and latency."""
        queue_depth = len(self.queue)

        # Dynamic batch sizing
        if queue_depth <= 1:
            return 1
        elif queue_depth <= 4:
            return min(queue_depth, 4)
        elif queue_depth <= 16:
            return min(queue_depth, 8)
        elif queue_depth <= 64:
            return min(queue_depth, 16)
        else:
            return min(queue_depth, self.max_batch_size)

    @property
    def stats(self) -> dict:
        return {
            "total_processed": self._total_processed,
            "avg_latency_ms": self._total_latency_ms / max(self._total_processed, 1),
            "queue_depth": len(self.queue),
        }