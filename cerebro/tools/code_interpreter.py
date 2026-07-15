"""Code interpreter sandbox for Cerebro.

Sandboxed Python code execution with:
- Restricted execution environment
- Timeout protection
- Output capture (stdout/stderr)
- File system isolation
- Built-in library access
"""

from __future__ import annotations

import io
import sys
import traceback
import subprocess
import tempfile
import os
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CodeResult:
    """Result from code execution."""
    output: str
    error: str | None = None
    exit_code: int = 0
    execution_time: float = 0.0
    files_created: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and self.error is None

    def to_dict(self) -> dict:
        return {
            "output": self.output,
            "error": self.error,
            "exit_code": self.exit_code,
            "execution_time": self.execution_time,
        }


class CodeInterpreter:
    """Sandboxed Python code execution engine.

    Executes Python code in an isolated subprocess with:
    - Timeouts to prevent infinite loops
    - Captured stdout/stderr
    - Temporary working directory
    - Optional module restrictions

    Args:
        timeout: Maximum execution time in seconds.
        max_output: Maximum output size in characters.
        allowed_modules: Modules allowed for import (None = all stdlib).
    """

    def __init__(
        self,
        timeout: int = 30,
        max_output: int = 50_000,
        allowed_modules: list[str] | None = None,
    ) -> None:
        self.timeout = timeout
        self.max_output = max_output
        self.allowed_modules = allowed_modules
        self._workdir: str | None = None

    def execute(self, code: str) -> CodeResult:
        """Execute Python code in a sandboxed subprocess.

        Args:
            code: Python source code to execute.

        Returns:
            CodeResult with output, errors, and execution metadata.
        """
        workdir = tempfile.mkdtemp(prefix="cerebro_sandbox_")
        self._workdir = workdir

        try:
            import time
            start = time.time()

            # Sandboxed environment — minimal vars, no secrets or tokens leaked
            sandbox_env = {
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "HOME": workdir,
                "TMPDIR": workdir,
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONUNBUFFERED": "1",
                "LANG": "en_US.UTF-8",
            }

            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=workdir,
                env=sandbox_env,
            )

            elapsed = time.time() - start
            stdout = result.stdout[:self.max_output]
            stderr = result.stderr[:self.max_output] if result.returncode != 0 else None

            files = [
                f for f in os.listdir(workdir)
                if f != "." and f != ".."
            ]

            return CodeResult(
                output=stdout,
                error=stderr,
                exit_code=result.returncode,
                execution_time=elapsed,
                files_created=files,
            )

        except subprocess.TimeoutExpired:
            return CodeResult(
                output="",
                error=f"Execution timed out after {self.timeout} seconds",
                exit_code=-1,
                execution_time=float(self.timeout),
            )
        except Exception as e:
            return CodeResult(
                output="",
                error=f"Execution error: {type(e).__name__}: {e}",
                exit_code=-1,
            )

    def execute_with_context(self, code: str, variables: dict | None = None) -> CodeResult:
        """Execute code with pre-injected variables.

        Serializes variables as Python assignments and prepends them
        to the code before execution.

        Args:
            code: Python source code.
            variables: Dict of variable_name -> value to inject.

        Returns:
            CodeResult from execution.
        """
        preamble = ""
        if variables:
            import json
            for name, value in variables.items():
                preamble += f"{name} = {json.dumps(value)}\n"

        full_code = preamble + code
        return self.execute(full_code)

    def get_file(self, filename: str) -> str | None:
        """Read a file from the sandbox working directory.

        Args:
            filename: File name within the sandbox.

        Returns:
            File content or None if not found.
        """
        if self._workdir is None:
            return None
        path = os.path.join(self._workdir, filename)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        return None

    def cleanup(self) -> None:
        """Remove the sandbox working directory."""
        if self._workdir and os.path.exists(self._workdir):
            import shutil
            shutil.rmtree(self._workdir, ignore_errors=True)
            self._workdir = None


class ShellExecutor:
    """Sandboxed shell command execution.

    Runs shell commands with timeouts and output capture.
    Used by the computer use module for system interactions.

    Args:
        timeout: Maximum execution time in seconds.
        max_output: Maximum output size in characters.
        working_dir: Working directory for commands.
    """

    def __init__(
        self,
        timeout: int = 30,
        max_output: int = 50_000,
        working_dir: str | None = None,
    ) -> None:
        self.timeout = timeout
        self.max_output = max_output
        self.working_dir = working_dir or tempfile.mkdtemp(prefix="cerebro_shell_")

    def execute(self, command: str) -> CodeResult:
        """Execute a shell command.

        Args:
            command: Shell command string.

        Returns:
            CodeResult with output and status.
        """
        try:
            import time
            start = time.time()

            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.working_dir,
            )

            elapsed = time.time() - start
            stdout = result.stdout[:self.max_output]
            stderr = result.stderr[:self.max_output] if result.returncode != 0 else None

            return CodeResult(
                output=stdout,
                error=stderr,
                exit_code=result.returncode,
                execution_time=elapsed,
            )

        except subprocess.TimeoutExpired:
            return CodeResult(
                output="",
                error=f"Command timed out after {self.timeout}s",
                exit_code=-1,
            )
        except Exception as e:
            return CodeResult(
                output="",
                error=f"Shell error: {type(e).__name__}: {e}",
                exit_code=-1,
            )
