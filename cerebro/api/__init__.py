"""Cerebro API Server — production HTTP API with streaming.

FastAPI-based server providing:
- /v1/chat/completions — OpenAI-compatible chat endpoint
- /v1/completions — Text completion endpoint
- /v1/agent — Autonomous agent endpoint
- /v1/search — Perplexity-style grounded search
- SSE streaming for real-time token generation
- API key authentication
- Rate limiting
- Request logging and monitoring
"""

from __future__ import annotations

import json
import time
import uuid
import asyncio
import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Optional
from collections import defaultdict

logger = logging.getLogger("cerebro.api")

try:
    from fastapi import FastAPI, HTTPException, Depends, Request
    from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


# ============================================================
# Request/Response models
# ============================================================

if HAS_FASTAPI:

    class ChatMessage(BaseModel):
        role: str = "user"
        content: str = ""
        tool_calls: list[dict] | None = None
        tool_call_id: str | None = None

    class ChatCompletionRequest(BaseModel):
        model: str = "cerebro-nano"
        messages: list[ChatMessage]
        temperature: float = 0.7
        top_p: float = 0.9
        max_tokens: int = 1024
        stream: bool = False
        tools: list[dict] | None = None
        system_prompt: str | None = None

    class CompletionRequest(BaseModel):
        model: str = "cerebro-nano"
        prompt: str
        temperature: float = 0.7
        top_p: float = 0.9
        max_tokens: int = 1024
        stream: bool = False

    class AgentRequest(BaseModel):
        task: str
        max_steps: int = 20
        tools: list[str] | None = None

    class SearchRequest(BaseModel):
        query: str
        max_results: int = 10


# ============================================================
# Rate limiter
# ============================================================

class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, requests_per_minute: int = 60, burst: int = 10) -> None:
        self.rpm = requests_per_minute
        self.burst = burst
        self._buckets: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, key: str) -> bool:
        now = time.time()
        window = 60.0
        self._buckets[key] = [
            t for t in self._buckets[key] if now - t < window
        ]
        if len(self._buckets[key]) >= self.rpm:
            return False
        self._buckets[key].append(now)
        return True


# ============================================================
# API Server
# ============================================================

def create_app(
    engine=None,
    tokenizer=None,
    api_keys: list[str] | None = None,
    rate_limit: int = 60,
    auth=None,  # CerebroAuth instance
) -> Any:
    """Create the FastAPI application.

    Args:
        engine: CerebroInferenceEngine instance.
        tokenizer: CerebroTokenizer instance.
        api_keys: List of valid API keys (None = no auth).
        rate_limit: Requests per minute per API key.
        auth: Optional CerebroAuth instance for JWT/OAuth2/API key auth.

    Returns:
        FastAPI app instance.
    """
    if not HAS_FASTAPI:
        raise ImportError(
            "FastAPI is required. Install with: pip install fastapi uvicorn"
        )

    app = FastAPI(
        title="Cerebro LLM API",
        description="Cognitive Entropic Reasoning Engine API",
        version="1.0.0",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ─── Static file serving for web chat UI ───
    import os
    static_dir = os.path.join(os.path.dirname(__file__), "..", "server", "static")
    if os.path.isdir(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

        @app.get("/")
        async def chat_ui():
            return FileResponse(os.path.join(static_dir, "index.html"))

    limiter = RateLimiter(requests_per_minute=rate_limit)
    _api_keys = set(api_keys) if api_keys else None
    _auth = auth  # CerebroAuth instance for JWT/OAuth2/API key auth
    _request_log: list[dict] = []
    _max_request_log = 1000  # Ring buffer limit to prevent memory leak

    def verify_api_key(request: Request) -> str:
        """Verify API key from Authorization header.

        Uses CerebroAuth if available (JWT/OAuth2/API key rotation),
        falls back to simple API key list.
        """
        auth_header = request.headers.get("Authorization", "")

        if _auth is not None:
            ctx = _auth.authenticate(auth_header)
            if ctx is not None:
                return ctx.principal
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")

        # Legacy simple API key mode
        if _api_keys is None:
            return "anonymous"
        if not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing API key")
        key = auth_header[7:]
        if key not in _api_keys:
            raise HTTPException(status_code=403, detail="Invalid API key")
        return key

    def check_rate_limit(key: str) -> None:
        if not limiter.is_allowed(key):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Try again later.",
            )

    def log_request(endpoint: str, key: str, status: int, latency: float) -> None:
        _request_log.append({
            "endpoint": endpoint,
            "key": key[:8] + "...",
            "status": status,
            "latency_ms": latency,
            "timestamp": time.time(),
        })
        # Ring buffer eviction to prevent memory leak
        if len(_request_log) > _max_request_log:
            _request_log[:] = _request_log[-_max_request_log:]

    # ─── Health check ───
    @app.get("/health")
    async def health():
        return {"status": "ok", "model": "cerebro", "version": "1.0.0"}

    # ─── Chat completions ───
    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: ChatCompletionRequest,
        req: Request,
    ):
        start = time.time()
        key = verify_api_key(req)
        check_rate_limit(key)

        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        # Build prompt from messages
        parts = []
        if request.system_prompt:
            parts.append(f"<|system|>\n{request.system_prompt}\n")
        for msg in request.messages:
            parts.append(f"<|{msg.role}|>\n{msg.content}\n")
        parts.append("<|assistant|>\n")
        prompt = "".join(parts)

        if engine is None or tokenizer is None:
            content = f"[Cerebro response to: {prompt[:100]}]"
        else:
            import torch
            tokens = tokenizer.encode(prompt, add_bos=True)
            input_ids = torch.tensor([tokens], dtype=torch.long)
            generated = engine.generate(
                input_ids,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            )
            output_tokens = generated[0].tolist()
            content = tokenizer.decode(output_tokens[len(tokens):], skip_special=True)

        latency = time.time() - start
        log_request("/v1/chat/completions", key, 200, latency * 1000)

        if request.stream:
            async def stream_response():
                # Real token-level streaming: generate tokens one at a time
                if engine is not None and tokenizer is not None:
                    import torch
                    tokens = tokenizer.encode(prompt, add_bos=True)
                    input_ids = torch.tensor([tokens], dtype=torch.long)

                    engine.reset_cache()
                    engine.sampler.temperature = request.temperature
                    engine.sampler.top_p = request.top_p

                    # Prefill: process full prompt through KV-cache
                    next_logits = engine.prefill(input_ids)

                    for _ in range(request.max_tokens):
                        next_token = engine.sampler.sample_batch(
                            next_logits, do_sample=True,
                        )
                        token_id = next_token[0, 0].item()
                        if token_id == engine.config.eos_token_id:
                            break

                        word = tokenizer.decode([token_id], skip_special=True)
                        if word:
                            chunk = {
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "choices": [{
                                    "delta": {"content": word},
                                    "index": 0,
                                }],
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"
                        next_logits = engine.decode_step(next_token)

                yield "data: [DONE]\n\n"

            return StreamingResponse(
                stream_response(),
                media_type="text/event-stream",
            )

        return {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": len(tokenizer.encode(prompt)) if tokenizer else 0,
                "completion_tokens": len(tokenizer.encode(content)) if tokenizer else 0,
            },
        }

    # ─── Text completions ───
    @app.post("/v1/completions")
    async def completions(
        request: CompletionRequest,
        req: Request,
    ):
        start = time.time()
        key = verify_api_key(req)
        check_rate_limit(key)

        if engine is None or tokenizer is None:
            content = f"[Cerebro completion for: {request.prompt[:100]}]"
        else:
            import torch
            tokens = tokenizer.encode(request.prompt, add_bos=True)
            input_ids = torch.tensor([tokens], dtype=torch.long)
            generated = engine.generate(
                input_ids,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
            )
            output_tokens = generated[0].tolist()
            content = tokenizer.decode(output_tokens[len(tokens):], skip_special=True)

        latency = time.time() - start
        log_request("/v1/completions", key, 200, latency * 1000)

        return {
            "id": f"cmpl-{uuid.uuid4().hex[:12]}",
            "object": "text_completion",
            "choices": [{"text": content, "index": 0, "finish_reason": "stop"}],
        }

    # ─── Agent ───
    @app.post("/v1/agent")
    async def agent_endpoint(
        request: AgentRequest,
        req: Request,
    ):
        key = verify_api_key(req)
        check_rate_limit(key)

        from cerebro.agents import CerebroAgent
        agent = CerebroAgent(
            model_engine=engine,
            tokenizer=tokenizer,
            max_steps=request.max_steps,
        )
        result = await agent.run(request.task)

        return {
            "task": request.task,
            "result": result,
            "steps": len(agent.steps),
            "status": agent.status.value,
        }

    # ─── Search ───
    @app.post("/v1/search")
    async def search_endpoint(
        request: SearchRequest,
        req: Request,
    ):
        key = verify_api_key(req)
        check_rate_limit(key)

        from cerebro.search import WebSearch
        searcher = WebSearch(backend="duckduckgo")
        response = searcher.search(request.query)

        return response.to_dict()

    # ─── Metrics ───
    @app.get("/metrics")
    async def metrics():
        total = len(_request_log)
        if total == 0:
            return {"total_requests": 0, "requests_per_endpoint": {}, "avg_latency_ms": 0}

        # Aggregate stats by endpoint
        by_endpoint: dict[str, list[float]] = {}
        for entry in _request_log:
            ep = entry.get("endpoint", "unknown")
            lat = entry.get("latency_ms", 0)
            by_endpoint.setdefault(ep, []).append(lat)

        endpoint_stats = {}
        for ep, latencies in by_endpoint.items():
            endpoint_stats[ep] = {
                "count": len(latencies),
                "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
                "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2),
            }

        all_latencies = [e.get("latency_ms", 0) for e in _request_log]
        return {
            "total_requests": total,
            "requests_per_endpoint": endpoint_stats,
            "avg_latency_ms": round(sum(all_latencies) / len(all_latencies), 2),
        }

    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    engine=None,
    tokenizer=None,
    api_keys: list[str] | None = None,
    auth=None,  # CerebroAuth instance
) -> None:
    """Start the Cerebro API server.

    Args:
        host: Bind host.
        port: Bind port.
        engine: Inference engine.
        tokenizer: Tokenizer.
        api_keys: Valid API keys.
        auth: Optional CerebroAuth instance for JWT/OAuth2/API key auth.
    """
    try:
        import uvicorn
    except ImportError:
        raise ImportError("uvicorn required: pip install uvicorn")

    app = create_app(engine=engine, tokenizer=tokenizer, api_keys=api_keys, auth=auth)
    logger.info("Cerebro API server starting on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")
