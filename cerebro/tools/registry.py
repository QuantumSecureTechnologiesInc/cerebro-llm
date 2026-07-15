"""Tool calling framework for Cerebro.

Implements function calling with:
- Tool registry with JSON schema
- Tool call parsing from model output
- Tool execution with error handling
- Result formatting for conversations
"""

from __future__ import annotations

import json
import re
import inspect
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class ToolParameter:
    """A parameter definition for a tool."""
    name: str
    type: str
    description: str
    required: bool = True
    enum: list[str] | None = None
    default: Any = None


@dataclass
class ToolDefinition:
    """Complete definition of a callable tool."""
    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    handler: Callable | None = None
    category: str = "general"

    def to_json_schema(self) -> dict:
        properties = {}
        required = []
        for param in self.parameters:
            prop = {"type": param.type, "description": param.description}
            if param.enum:
                prop["enum"] = param.enum
            properties[param.name] = prop
            if param.required:
                required.append(param.name)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def to_prompt_format(self) -> str:
        params = ", ".join(
            f"{p.name}: {p.type}" for p in self.parameters if p.required
        )
        return f"- {self.name}({params}): {self.description}"


@dataclass
class ToolCall:
    """A parsed tool call from model output."""
    id: str
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments),
            },
        }


@dataclass
class ToolResult:
    """Result from executing a tool."""
    tool_call_id: str
    name: str
    content: str
    success: bool = True
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "content": self.content,
        }


class ToolRegistry:
    """Registry for all available tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._handlers: dict[str, Callable] = {}

    def register(
        self, name: str, description: str, handler: Callable,
        parameters: list[ToolParameter] | None = None,
        category: str = "general",
    ) -> None:
        tool = ToolDefinition(
            name=name, description=description,
            parameters=parameters or [], handler=handler, category=category,
        )
        self._tools[name] = tool
        self._handlers[name] = handler

    def register_function(self, func: Callable, description: str = "") -> None:
        """Auto-register a Python function as a tool."""
        sig = inspect.signature(func)
        params = []
        type_map = {str: "string", int: "integer", float: "number", bool: "boolean"}
        for pname, p in sig.parameters.items():
            if pname in ("self", "cls"):
                continue
            ptype = "string"
            if p.annotation != inspect.Parameter.empty:
                ptype = type_map.get(p.annotation, "string")
            params.append(ToolParameter(
                name=pname, type=ptype,
                description=f"Parameter {pname}",
                required=(p.default == inspect.Parameter.empty),
            ))
        self.register(
            name=func.__name__,
            description=description or func.__doc__ or "",
            handler=func, parameters=params,
        )

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)
        self._handlers.pop(name, None)

    def get_tool(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def list_tools(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def get_json_schemas(self) -> list[dict]:
        return [t.to_json_schema() for t in self._tools.values()]

    def get_tools_prompt(self) -> str:
        lines = ["You have access to the following tools:"]
        for tool in self._tools.values():
            lines.append(tool.to_prompt_format())
        lines.append("")
        lines.append(
            'To call a tool, output a JSON block: {"tool": "name", "arguments": {...}}'
        )
        return "\n".join(lines)

    async def execute(self, call: ToolCall) -> ToolResult:
        handler = self._handlers.get(call.name)
        if handler is None:
            return ToolResult(
                tool_call_id=call.id, name=call.name,
                content=f"Error: Tool not found: {call.name}", success=False,
            )
        try:
            if inspect.iscoroutinefunction(handler):
                result = await handler(**call.arguments)
            else:
                result = handler(**call.arguments)
            return ToolResult(
                tool_call_id=call.id, name=call.name,
                content=str(result), success=True,
            )
        except Exception as e:
            return ToolResult(
                tool_call_id=call.id, name=call.name,
                content=f"Error: {type(e).__name__}: {e}",
                success=False,
                metadata={"traceback": traceback.format_exc()},
            )


class ToolCallParser:
    """Parses tool calls from model text output.

    Supports JSON blocks and XML-style tags.
    """

    JSON_PATTERN = re.compile(
        r'\{\s*"tool"\s*:\s*"(\w+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})\s*\}'
    )
    XML_PATTERN = re.compile(
        r'<tool_call>\s*<name>(\w+)</name>\s*<arguments>(.*?)</arguments>\s*',
        re.DOTALL,
    )

    def parse(self, text: str) -> list[ToolCall]:
        """Parse all tool calls from text."""
        calls = []
        calls.extend(self._parse_json(text))
        calls.extend(self._parse_xml(text))
        return calls

    def _parse_json(self, text: str) -> list[ToolCall]:
        calls = []
        for match in self.JSON_PATTERN.finditer(text):
            name = match.group(1)
            try:
                args = json.loads(match.group(2))
                calls.append(ToolCall(id=str(uuid.uuid4()), name=name, arguments=args))
            except json.JSONDecodeError:
                continue
        return calls

    def _parse_xml(self, text: str) -> list[ToolCall]:
        calls = []
        for match in self.XML_PATTERN.finditer(text):
            name = match.group(1).strip()
            args_text = match.group(2).strip()
            try:
                args = json.loads(args_text)
                calls.append(ToolCall(id=str(uuid.uuid4()), name=name, arguments=args))
            except json.JSONDecodeError:
                calls.append(ToolCall(id=str(uuid.uuid4()), name=name, arguments={"raw": args_text}))
        return calls

    def has_tool_calls(self, text: str) -> bool:
        return bool(self.JSON_PATTERN.search(text) or self.XML_PATTERN.search(text))
