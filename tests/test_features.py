"""Tests for all new Cerebro production features."""

import os
import json
import asyncio
import tempfile
import torch
import pytest

from cerebro.config import CerebroConfig
from cerebro.chat.memory import ConversationMemory, Message
from cerebro.tools.registry import (
    ToolRegistry, ToolCall, ToolResult, ToolDefinition,
    ToolParameter, ToolCallParser,
)
from cerebro.tools.code_interpreter import CodeInterpreter, ShellExecutor, CodeResult
from cerebro.computer import ComputerUse
from cerebro.search import WebSearch, SearchResult, SearchResponse
from cerebro.rag import RAGPipeline, VectorStore, DocumentChunker, Document
from cerebro.agents import CerebroAgent, AgentPlan, AgentStatus
from cerebro.vision import PatchEmbedding, VisionEncoder, VisionProcessor
from cerebro.training.distributed import DistributedTrainer, get_launch_command
from cerebro.training.evaluator import Evaluator, MetricsLogger, EvalResult


# ========== TOOL CALLING TESTS ==========

class TestToolRegistry:
    def test_register_and_list(self):
        reg = ToolRegistry()
        reg.register("add", "Add two numbers", lambda a, b: a + b,
                      [ToolParameter("a", "integer", "First"),
                       ToolParameter("b", "integer", "Second")])
        assert len(reg.list_tools()) == 1
        assert reg.get_tool("add").name == "add"

    def test_register_function(self):
        reg = ToolRegistry()
        def multiply(x: int, y: int) -> int:
            """Multiply two numbers."""
            return x * y
        reg.register_function(multiply)
        assert reg.get_tool("multiply") is not None
        schemas = reg.get_json_schemas()
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "multiply"

    def test_unregister(self):
        reg = ToolRegistry()
        reg.register("test", "Test tool", lambda: "ok")
        assert len(reg.list_tools()) == 1
        reg.unregister("test")
        assert len(reg.list_tools()) == 0

    def test_tools_prompt(self):
        reg = ToolRegistry()
        reg.register("calc", "Calculator", lambda x: x,
                      [ToolParameter("x", "string", "Expression")])
        prompt = reg.get_tools_prompt()
        assert "calc" in prompt
        assert "Calculator" in prompt

    def test_execute_sync(self):
        reg = ToolRegistry()
        reg.register("greet", "Greet", lambda name: f"Hello {name}!",
                      [ToolParameter("name", "string", "Name")])
        call = ToolCall(id="t1", name="greet", arguments={"name": "Alice"})
        result = asyncio.run(reg.execute(call))
        assert result.success
        assert "Hello Alice!" in result.content

    def test_execute_missing_tool(self):
        reg = ToolRegistry()
        call = ToolCall(id="t1", name="nonexistent", arguments={})
        result = asyncio.run(reg.execute(call))
        assert not result.success

    def test_execute_with_error(self):
        reg = ToolRegistry()
        def bad_tool():
            raise ValueError("Something went wrong")
        reg.register("bad", "Bad tool", bad_tool)
        call = ToolCall(id="t1", name="bad", arguments={})
        result = asyncio.run(reg.execute(call))
        assert not result.success
        assert "ValueError" in result.content

    def test_json_schema_format(self):
        tool = ToolDefinition(
            name="search", description="Search the web",
            parameters=[
                ToolParameter("query", "string", "Search query"),
                ToolParameter("limit", "integer", "Max results", required=False),
            ],
        )
        schema = tool.to_json_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "search"
        assert "query" in schema["function"]["parameters"]["properties"]

    def test_tool_call_to_dict(self):
        call = ToolCall(id="c1", name="test", arguments={"x": 1})
        d = call.to_dict()
        assert d["id"] == "c1"
        assert d["function"]["name"] == "test"

    def test_tool_result_to_dict(self):
        result = ToolResult(tool_call_id="c1", name="test", content="ok")
        d = result.to_dict()
        assert d["role"] == "tool"
        assert d["content"] == "ok"


class TestToolCallParser:
    def test_parse_json_format(self):
        parser = ToolCallParser()
        text = 'Here: {"tool": "calc", "arguments": {"expr": "2+2"}}'
        calls = parser.parse(text)
        assert len(calls) == 1
        assert calls[0].name == "calc"
        assert calls[0].arguments["expr"] == "2+2"

    def test_has_tool_calls(self):
        parser = ToolCallParser()
        assert parser.has_tool_calls('{"tool": "x", "arguments": {}}')
        assert not parser.has_tool_calls("No tools here")


# ========== CONVERSATION MEMORY TESTS ==========

class TestConversationMemory:
    def test_add_messages(self):
        mem = ConversationMemory(system_prompt="You are helpful.")
        mem.add_user_message("Hello")
        mem.add_assistant_message("Hi there!")
        assert len(mem.messages) == 2
        assert mem.messages[0].role == "user"
        assert mem.messages[1].role == "assistant"

    def test_get_messages(self):
        mem = ConversationMemory(system_prompt="System")
        mem.add_user_message("Hi")
        msgs = mem.get_messages()
        assert len(msgs) == 2  # system + user
        assert msgs[0]["role"] == "system"

    def test_format_for_prompt(self):
        mem = ConversationMemory(system_prompt="You are Cerebro.")
        mem.add_user_message("Hello")
        prompt = mem.format_for_prompt()
        assert "<|system|>" in prompt
        assert "<|user|>" in prompt
        assert "<|assistant|>" in prompt

    def test_clear(self):
        mem = ConversationMemory(system_prompt="System")
        mem.add_user_message("Hi")
        mem.add_assistant_message("Hello")
        mem.clear()
        assert len(mem.messages) == 0

    def test_sliding_window(self):
        mem = ConversationMemory(max_tokens=50)
        for i in range(20):
            mem.add_user_message(f"Message {i} with some content")
        # Should have trimmed old messages
        assert len(mem.messages) < 20

    def test_tool_result(self):
        mem = ConversationMemory()
        mem.add_tool_result("call_1", "Result data", "search")
        assert mem.messages[0].role == "tool"
        assert mem.messages[0].tool_call_id == "call_1"

    def test_serialization(self):
        mem = ConversationMemory(system_prompt="Test")
        mem.add_user_message("Hello")
        mem.add_assistant_message("Hi")
        json_str = mem.to_json()
        restored = ConversationMemory.from_json(json_str)
        assert len(restored.messages) == 2
        assert restored.system_prompt == "Test"


# ========== CODE INTERPRETER TESTS ==========

class TestCodeInterpreter:
    def test_simple_execution(self):
        interp = CodeInterpreter(timeout=10)
        result = interp.execute("print('hello world')")
        assert result.success
        assert "hello world" in result.output

    def test_math_computation(self):
        interp = CodeInterpreter(timeout=10)
        result = interp.execute("print(2 + 3 * 4)")
        assert result.success
        assert "14" in result.output

    def test_error_handling(self):
        interp = CodeInterpreter(timeout=10)
        result = interp.execute("raise ValueError('test error')")
        assert not result.success
        assert "ValueError" in result.error

    def test_timeout(self):
        interp = CodeInterpreter(timeout=2)
        result = interp.execute("import time; time.sleep(10)")
        assert not result.success
        assert "timed out" in result.error.lower()

    def test_execution_time_tracked(self):
        interp = CodeInterpreter(timeout=10)
        result = interp.execute("print('fast')")
        assert result.execution_time > 0

    def test_execute_with_context(self):
        interp = CodeInterpreter(timeout=10)
        result = interp.execute_with_context(
            "print(x + y)",
            variables={"x": 10, "y": 20},
        )
        assert result.success
        assert "30" in result.output


class TestShellExecutor:
    def test_simple_command(self):
        shell = ShellExecutor(timeout=10)
        result = shell.execute("echo hello")
        assert result.success
        assert "hello" in result.output

    def test_command_error(self):
        shell = ShellExecutor(timeout=10)
        result = shell.execute("python -c 'import sys; sys.exit(1)'")
        assert result.exit_code != 0


# ========== COMPUTER USE TESTS ==========

class TestComputerUse:
    def test_file_operations(self):
        comp = ComputerUse()
        try:
            result = comp.write_file("test.txt", "Hello World")
            assert "Successfully" in result

            content = comp.read_file("test.txt")
            assert "Hello World" in content

            listing = comp.list_files(".")
            assert "test.txt" in listing
        finally:
            comp.cleanup()

    def test_run_python(self):
        comp = ComputerUse()
        result = comp.run_python("print('from computer')")
        assert result.success
        assert "from computer" in result.output

    def test_run_shell(self):
        comp = ComputerUse()
        result = comp.run_shell("echo shell_test")
        assert result.success
        assert "shell_test" in result.output

    def test_search_files(self):
        comp = ComputerUse()
        try:
            comp.write_file("sub/a.py", "code")
            comp.write_file("sub/b.py", "code")
            result = comp.search_files("*.py", "sub")
            assert "a.py" in result
            assert "b.py" in result
        finally:
            comp.cleanup()

    def test_state_tracking(self):
        comp = ComputerUse()
        try:
            comp.run_shell("echo test")
            assert len(comp.state.command_history) == 1
            comp.write_file("x.txt", "data")
            assert "x.txt" in comp.state.files_modified
        finally:
            comp.cleanup()


# ========== WEB SEARCH TESTS ==========

class TestWebSearch:
    def test_search_result_creation(self):
        r = SearchResult(title="Test", url="https://example.com", snippet="Desc")
        assert r.to_citation(1) == "[1] Test (example.com)"
        d = r.to_dict()
        assert d["title"] == "Test"

    def test_search_response(self):
        results = [
            SearchResult("A", "https://a.com", "First", score=1.0),
            SearchResult("B", "https://b.com", "Second", score=0.5),
        ]
        resp = SearchResponse(query="test", results=results)
        citations = resp.format_citations()
        assert "[1]" in citations
        assert "[2]" in citations

    def test_search_response_dict(self):
        resp = SearchResponse(query="test", results=[])
        d = resp.to_dict()
        assert d["query"] == "test"
        assert d["results"] == []

    def test_format_context(self):
        searcher = WebSearch()
        results = [SearchResult("Title", "https://url.com", "Content here")]
        resp = SearchResponse(query="test", results=results)
        ctx = searcher.format_context_for_prompt(resp)
        assert "[1]" in ctx
        assert "Title" in ctx

    def test_web_search_init(self):
        s = WebSearch(backend="duckduckgo", max_results=5)
        assert s.backend == "duckduckgo"
        assert s.max_results == 5


# ========== RAG TESTS ==========

class TestDocumentChunker:
    def test_short_text(self):
        chunker = DocumentChunker(chunk_size=1000)
        chunks = chunker.chunk_text("Short text")
        assert len(chunks) == 1
        assert chunks[0] == "Short text"

    def test_long_text_chunking(self):
        chunker = DocumentChunker(chunk_size=100, overlap=20)
        text = ". ".join(f"Sentence {i}" for i in range(50))
        chunks = chunker.chunk_text(text)
        assert len(chunks) > 1

    def test_chunk_document(self):
        doc = Document(id="d1", content="A" * 2000, source="test")
        chunker = DocumentChunker(chunk_size=500)
        result = chunker.chunk_document(doc)
        assert len(result.chunks) > 0


class TestVectorStore:
    def test_add_and_search(self):
        store = VectorStore()
        doc = Document(id="d1", content="Python is a programming language", source="doc1")
        store.add_document(doc)
        assert store.num_documents == 1
        assert store.num_chunks > 0

        results = store.search("Python programming")
        assert len(results) > 0

    def test_empty_search(self):
        store = VectorStore()
        results = store.search("nothing here")
        assert len(results) == 0

    def test_clear(self):
        store = VectorStore()
        store.add_document(Document(id="d1", content="test content"))
        store.clear()
        assert store.num_documents == 0


class TestRAGPipeline:
    def test_add_document(self):
        rag = RAGPipeline()
        doc_id = rag.add_document("The quick brown fox jumps over the lazy dog.", source="test")
        assert doc_id is not None
        assert rag.store.num_documents == 1

    def test_retrieve(self):
        rag = RAGPipeline()
        rag.add_document("Python is great for machine learning and data science.")
        rag.add_document("JavaScript is used for web development.")
        chunks = rag.retrieve("machine learning")
        assert len(chunks) > 0

    def test_build_context_prompt(self):
        rag = RAGPipeline()
        chunks = rag.retrieve("test")
        prompt = rag.build_context_prompt("What is AI?", chunks)
        assert "What is AI?" in prompt

    def test_add_file(self):
        rag = RAGPipeline()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Document content for testing RAG pipeline.")
            f.flush()
            doc_id = rag.add_file(f.name)
        os.unlink(f.name)
        assert doc_id is not None


# ========== AGENT TESTS ==========

class TestAgentPlan:
    def test_plan_creation(self):
        plan = AgentPlan(task="Research AI", steps=["Step 1", "Step 2", "Step 3"])
        assert len(plan.steps) == 3
        assert not plan.completed

    def test_next_step(self):
        plan = AgentPlan(task="Test", steps=["A", "B"])
        assert plan.next_step() == "A"
        assert plan.next_step() == "B"
        assert plan.next_step() is None
        assert plan.completed

    def test_progress(self):
        plan = AgentPlan(task="Test", steps=["A", "B", "C", "D"])
        assert plan.progress() == 0.0
        plan.next_step()
        assert plan.progress() == 0.25
        plan.next_step()
        assert plan.progress() == 0.5


class TestCerebroAgent:
    def test_agent_creation(self):
        agent = CerebroAgent(max_steps=5)
        assert agent.max_steps == 5
        assert agent.status == AgentStatus.PLANNING

    def test_agent_state(self):
        agent = CerebroAgent(max_steps=10)
        state = agent.get_state()
        assert state["status"] == "planning"
        assert state["max_steps"] == 10

    def test_agent_run(self):
        agent = CerebroAgent(max_steps=3)
        result = asyncio.run(agent.run("Simple test task"))
        assert isinstance(result, str)
        assert agent.status == AgentStatus.COMPLETE


# ========== VISION TESTS ==========

class TestPatchEmbedding:
    def test_forward_shape(self):
        pe = PatchEmbedding(image_size=224, patch_size=16, embed_dim=128)
        images = torch.randn(2, 3, 224, 224)
        output = pe(images)
        num_patches = (224 // 16) ** 2
        assert output.shape == (2, num_patches, 128)

    def test_num_patches(self):
        pe = PatchEmbedding(image_size=224, patch_size=16, embed_dim=64)
        assert pe.num_patches == (224 // 16) ** 2  # 196


class TestVisionEncoder:
    def test_forward_shape(self):
        enc = VisionEncoder(
            image_size=64, patch_size=16,
            embed_dim=128, num_layers=2, num_heads=4,
        )
        images = torch.randn(1, 3, 64, 64)
        output = enc(images)
        num_patches = (64 // 16) ** 2 + 1  # +1 for CLS
        assert output.shape == (1, num_patches, 128)


class TestVisionProcessor:
    def test_init(self):
        proc = VisionProcessor(image_size=224, device="cpu")
        assert proc.image_size == 224

    def test_describe_without_engine(self):
        proc = VisionProcessor(device="cpu")
        desc = proc.describe_image("test.png")
        assert "Vision description" in desc


# ========== DISTRIBUTED TRAINING TESTS ==========

class TestDistributedTrainer:
    def test_default_init(self):
        trainer = DistributedTrainer()
        assert not trainer.is_distributed
        assert trainer.rank == 0
        assert trainer.world_size == 1
        assert trainer.is_main_process

    def test_device_cpu(self):
        trainer = DistributedTrainer()
        assert trainer.device == torch.device("cpu") or "cuda" in str(trainer.device)

    def test_reduce_loss_single(self):
        trainer = DistributedTrainer()
        loss = torch.tensor(2.5)
        result = trainer.reduce_loss(loss)
        assert abs(result - 2.5) < 0.01


class TestLaunchCommand:
    def test_single_gpu(self):
        cmd = get_launch_command(1, 1, "train.py")
        assert "python train.py" in cmd

    def test_multi_gpu(self):
        cmd = get_launch_command(4, 1, "train.py", "--config nano")
        assert "torchrun" in cmd
        assert "nproc_per_node=4" in cmd

    def test_multi_node(self):
        cmd = get_launch_command(4, 2, "train.py")
        assert "nnodes=2" in cmd


# ========== EVALUATOR TESTS ==========

class TestEvalResult:
    def test_creation(self):
        result = EvalResult(
            loss=2.5, perplexity=12.2,
            tokens_evaluated=1000, elapsed_seconds=5.0,
            tokens_per_second=200.0,
        )
        assert result.loss == 2.5
        assert result.perplexity == 12.2


class TestMetricsLogger:
    def test_json_logging(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(log_dir=tmpdir)
            logger.log({"loss": 2.5}, step=100)
            logger.log({"loss": 2.3}, step=200)
            logger.flush_json()

            log_path = os.path.join(tmpdir, "metrics.json")
            assert os.path.exists(log_path)
            with open(log_path) as f:
                data = json.load(f)
            assert len(data) == 2

    def test_close(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(log_dir=tmpdir)
            logger.log({"x": 1}, step=0)
            logger.close()  # Should not raise
