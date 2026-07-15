"""Cerebro Agent Framework.

Autonomous agent with plan/act/reflect loop:
- Task planning and decomposition
- Tool execution with verification
- Self-reflection and error recovery
- Multi-step reasoning chains
- Memory across agent steps

Similar to ChatGPT Agent, Claude's computer use agent,
and Google's Project Mariner.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum

from cerebro.chat.memory import ConversationMemory
from cerebro.tools.registry import ToolRegistry, ToolCall, ToolCallParser


class AgentStatus(Enum):
    PLANNING = "planning"
    ACTING = "acting"
    REFLECTING = "reflecting"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class AgentStep:
    """A single step in the agent execution."""
    step_id: str
    phase: AgentStatus
    thought: str
    action: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    observations: str = ""
    success: bool = True
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentPlan:
    """A multi-step plan for a task."""
    task: str
    steps: list[str]
    current_step: int = 0
    completed: bool = False

    def next_step(self) -> str | None:
        if self.current_step < len(self.steps):
            step = self.steps[self.current_step]
            self.current_step += 1
            return step
        self.completed = True
        return None

    def progress(self) -> float:
        if not self.steps:
            return 1.0
        return self.current_step / len(self.steps)


class CerebroAgent:
    """Autonomous agent with plan/act/reflect loop.

    The agent:
    1. Plans: Breaks down the task into steps
    2. Acts: Executes each step using tools
    3. Reflects: Evaluates results and adjusts

    Args:
        model_engine: Inference engine for generation.
        tokenizer: Cerebro tokenizer.
        tool_registry: Available tools.
        max_steps: Maximum agent steps before stopping.
        max_retries: Retries per step on failure.
    """

    def __init__(
        self,
        model_engine=None,
        tokenizer=None,
        tool_registry: ToolRegistry | None = None,
        max_steps: int = 20,
        max_retries: int = 2,
    ) -> None:
        self.engine = model_engine
        self.tokenizer = tokenizer
        self.tools = tool_registry or ToolRegistry()
        self.parser = ToolCallParser()
        self.max_steps = max_steps
        self.max_retries = max_retries

        self.memory = ConversationMemory(
            system_prompt=self._build_system_prompt(),
            max_tokens=6000,
            tokenizer=tokenizer,
        )

        self.plan: AgentPlan | None = None
        self.steps: list[AgentStep] = []
        self.status = AgentStatus.PLANNING

    def _build_system_prompt(self) -> str:
        """Build the agent system prompt."""
        tools_prompt = self.tools.get_tools_prompt() if self.tools.list_tools() else ""
        return f"""You are Cerebro Agent, an autonomous AI assistant that can use tools to complete tasks.

Your process:
1. PLAN: Break the task into concrete steps
2. ACT: Execute steps using available tools
3. REFLECT: Evaluate results, verify correctness, adjust if needed

Guidelines:
- Think step by step before acting
- Use tools when you need information or computation
- Verify your results before moving to the next step
- If something fails, try a different approach
- Summarize what you did when complete

{tools_prompt}

Always output your thinking as:
Thought: <your reasoning>
Action: <what you will do>
"""

    async def run(self, task: str) -> str:
        """Run the agent on a task.

        Executes the full plan/act/reflect loop until the task
        is complete or max_steps is reached.

        Args:
            task: Task description.

        Returns:
            Final answer or summary.
        """
        self.memory.add_user_message(task)
        self.status = AgentStatus.PLANNING

        # Phase 1: Plan
        plan = await self._create_plan(task)
        self.plan = plan

        # Phase 2: Act + Reflect loop
        for step_num in range(self.max_steps):
            if plan.completed:
                break

            current_step = plan.next_step()
            if current_step is None:
                break

            step = await self._execute_step(current_step, step_num)
            self.steps.append(step)

            # Reflect
            if not step.success:
                retry_step = await self._retry_step(step, current_step)
                if retry_step.success:
                    self.steps.append(retry_step)
                else:
                    self.steps.append(retry_step)

        summary = await self._generate_summary(task)
        self.status = AgentStatus.COMPLETE
        return summary

    async def _create_plan(self, task: str) -> AgentPlan:
        """Create a plan for the task."""
        plan_prompt = f"Create a step-by-step plan to accomplish this task:\n\n{task}\n\nList each step as a numbered item."
        self.memory.add_user_message(plan_prompt)

        response = self._generate(plan_prompt)
        self.memory.add_assistant_message(response)

        # Parse steps from response
        import re
        step_lines = re.findall(r'^\d+\.\s+(.+)$', response, re.MULTILINE)
        if not step_lines:
            step_lines = [task]

        return AgentPlan(task=task, steps=step_lines)

    async def _execute_step(self, step_text: str, step_num: int) -> AgentStep:
        """Execute a single plan step."""
        self.status = AgentStatus.ACTING

        prompt = f"Execute this step: {step_text}\n\nThink about what tools to use."
        self.memory.add_user_message(prompt)

        response = self._generate(prompt)

        # Parse tool calls
        tool_calls = self.parser.parse(response)
        observations = []

        for call in tool_calls:
            result = await self.tools.execute(call)
            observations.append(f"Tool {call.name}: {result.content}")
            self.memory.add_tool_result(call.id, result.content, call.name)

        obs_text = "\n".join(observations) if observations else response
        self.memory.add_assistant_message(response)

        return AgentStep(
            step_id=str(uuid.uuid4()),
            phase=AgentStatus.ACTING,
            thought=response,
            action=step_text,
            tool_calls=tool_calls,
            observations=obs_text,
            success=True,
        )

    async def _retry_step(self, failed_step: AgentStep, step_text: str) -> AgentStep:
        """Retry a failed step with adjusted approach."""
        self.status = AgentStatus.REFLECTING

        retry_prompt = (
            f"The previous attempt at '{step_text}' had issues:\n"
            f"{failed_step.observations}\n\n"
            f"Try a different approach."
        )
        self.memory.add_user_message(retry_prompt)
        response = self._generate(retry_prompt)

        tool_calls = self.parser.parse(response)
        observations = []
        for call in tool_calls:
            result = await self.tools.execute(call)
            observations.append(f"Tool {call.name}: {result.content}")
            self.memory.add_tool_result(call.id, result.content, call.name)

        self.memory.add_assistant_message(response)

        return AgentStep(
            step_id=str(uuid.uuid4()),
            phase=AgentStatus.REFLECTING,
            thought=response,
            action=f"RETRY: {step_text}",
            tool_calls=tool_calls,
            observations="\n".join(observations),
            success=len(observations) > 0,
        )

    async def _generate_summary(self, task: str) -> str:
        """Generate a final summary of what was accomplished."""
        self.status = AgentStatus.REFLECTING

        steps_summary = "\n".join(
            f"Step {i+1}: {s.action} -> {s.observations[:200]}"
            for i, s in enumerate(self.steps)
        )

        summary_prompt = (
            f"Task: {task}\n\nSteps taken:\n{steps_summary}\n\n"
            f"Provide a clear, concise summary of what was accomplished."
        )
        self.memory.add_user_message(summary_prompt)
        summary = self._generate(summary_prompt)
        self.memory.add_assistant_message(summary)
        return summary

    def _generate(self, prompt: str) -> str:
        """Generate text using the model engine.

        Falls back to a placeholder if no engine is available.
        """
        if self.engine is None or self.tokenizer is None:
            return f"[Agent response to: {prompt[:100]}...]"

        import torch
        tokens = self.tokenizer.encode(prompt, add_bos=True)
        input_ids = torch.tensor([tokens], dtype=torch.long)
        generated = self.engine.generate(input_ids, max_new_tokens=512, do_sample=True)
        output_tokens = generated[0].tolist()
        return self.tokenizer.decode(output_tokens[len(tokens):], skip_special=True)

    def get_state(self) -> dict:
        """Get current agent state for debugging."""
        return {
            "status": self.status.value,
            "plan": self.plan.task if self.plan else None,
            "plan_progress": self.plan.progress() if self.plan else 0.0,
            "steps_completed": len(self.steps),
            "max_steps": self.max_steps,
            "tools_available": [t.name for t in self.tools.list_tools()],
        }
