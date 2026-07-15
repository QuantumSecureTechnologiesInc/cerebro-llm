"""Computer use module for Cerebro.

Provides the model with the ability to interact with the computer:
- Browser automation (navigate, click, type, screenshot)
- File system operations (read, write, list, search)
- Shell command execution
- Screen capture and interaction

This is Cerebro's internal "computer use" capability, similar to
ChatGPT's code interpreter and Claude's computer use.
"""

from __future__ import annotations

import os
import json
import tempfile
import subprocess
from dataclasses import dataclass, field
from typing import Any, Optional

from cerebro.tools.code_interpreter import CodeInterpreter, ShellExecutor, CodeResult


@dataclass
class ComputerState:
    """Current state of the computer use environment."""
    working_dir: str
    open_urls: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    command_history: list[str] = field(default_factory=list)
    screenshots: list[str] = field(default_factory=list)


class ComputerUse:
    """Computer use environment for Cerebro agent.

    Provides tools for the model to:
    - Execute Python code
    - Run shell commands
    - Browse the web
    - Read/write files
    - Take screenshots

    Args:
        working_dir: Base working directory.
        timeout: Default timeout for operations.
        enable_browser: Enable browser automation.
    """

    def __init__(
        self,
        working_dir: str | None = None,
        timeout: int = 30,
        enable_browser: bool = True,
    ) -> None:
        self.working_dir = working_dir or tempfile.mkdtemp(prefix="cerebro_computer_")
        self.timeout = timeout
        self.enable_browser = enable_browser

        self.code_interpreter = CodeInterpreter(timeout=timeout)
        self.shell = ShellExecutor(timeout=timeout, working_dir=self.working_dir)
        self.state = ComputerState(working_dir=self.working_dir)
        self._browser = None

    def run_python(self, code: str) -> CodeResult:
        """Execute Python code in sandbox.

        Args:
            code: Python source code.

        Returns:
            CodeResult with execution output.
        """
        result = self.code_interpreter.execute(code)
        return result

    def run_shell(self, command: str) -> CodeResult:
        """Execute a shell command.

        Args:
            command: Shell command string.

        Returns:
            CodeResult with command output.
        """
        self.state.command_history.append(command)
        result = self.shell.execute(command)
        return result

    def read_file(self, path: str) -> str:
        """Read a file from the working directory.

        Args:
            path: File path (relative to working_dir or absolute).

        Returns:
            File contents as string.
        """
        full_path = path if os.path.isabs(path) else os.path.join(self.working_dir, path)
        if not os.path.exists(full_path):
            return f"Error: File not found: {path}"
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.state.files_modified.append(path)
            return content
        except Exception as e:
            return f"Error reading file: {type(e).__name__}: {e}"

    def write_file(self, path: str, content: str) -> str:
        """Write content to a file.

        Args:
            path: File path (relative to working_dir or absolute).
            content: File content to write.

        Returns:
            Success or error message.
        """
        full_path = path if os.path.isabs(path) else os.path.join(self.working_dir, path)
        try:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            self.state.files_modified.append(path)
            return f"Successfully wrote {len(content)} characters to {path}"
        except Exception as e:
            return f"Error writing file: {type(e).__name__}: {e}"

    def list_files(self, directory: str = ".") -> str:
        """List files in a directory.

        Args:
            directory: Directory to list (relative to working_dir).

        Returns:
            Formatted file listing.
        """
        full_path = directory if os.path.isabs(directory) else os.path.join(self.working_dir, directory)
        if not os.path.exists(full_path):
            return f"Error: Directory not found: {directory}"
        try:
            entries = os.listdir(full_path)
            lines = []
            for entry in sorted(entries):
                entry_path = os.path.join(full_path, entry)
                if os.path.isdir(entry_path):
                    lines.append(f"  [DIR]  {entry}/")
                else:
                    size = os.path.getsize(entry_path)
                    lines.append(f"  [FILE] {entry} ({size:,} bytes)")
            return "\n".join(lines) if lines else "(empty directory)"
        except Exception as e:
            return f"Error listing directory: {type(e).__name__}: {e}"

    def search_files(self, pattern: str, directory: str = ".") -> str:
        """Search for files matching a pattern.

        Args:
            pattern: Glob pattern (e.g., "*.py", "*.txt").
            directory: Base directory to search.

        Returns:
            List of matching file paths.
        """
        import glob
        full_path = directory if os.path.isabs(directory) else os.path.join(self.working_dir, directory)
        search_pattern = os.path.join(full_path, "**", pattern)
        matches = glob.glob(search_pattern, recursive=True)
        if not matches:
            return f"No files matching '{pattern}' found in {directory}"
        lines = [f"Found {len(matches)} files:"]
        for m in sorted(matches)[:50]:
            lines.append(f"  {os.path.relpath(m, self.working_dir)}")
        if len(matches) > 50:
            lines.append(f"  ... and {len(matches) - 50} more")
        return "\n".join(lines)

    def browse_web(self, url: str) -> str:
        """Navigate to a URL and return page content.

        Uses a headless browser or requests fallback.

        Args:
            url: URL to navigate to.

        Returns:
            Page content (text extracted from HTML).
        """
        self.state.open_urls.append(url)

        try:
            import requests
            resp = requests.get(url, timeout=self.timeout, headers={
                "User-Agent": "Cerebro/1.0 (AI Assistant)"
            })
            resp.raise_for_status()

            from html.parser import HTMLParser

            class TextExtractor(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.text = []
                    self._skip = False

                def handle_starttag(self, tag, attrs):
                    if tag in ("script", "style", "nav", "footer"):
                        self._skip = True

                def handle_endtag(self, tag):
                    if tag in ("script", "style", "nav", "footer"):
                        self._skip = False

                def handle_data(self, data):
                    if not self._skip:
                        text = data.strip()
                        if text:
                            self.text.append(text)

            parser = TextExtractor()
            parser.feed(resp.text)
            content = "\n".join(parser.text)[:20_000]
            return f"URL: {url}\nStatus: {resp.status_code}\n\n{content}"

        except Exception as e:
            return f"Error browsing {url}: {type(e).__name__}: {e}"

    def get_tools_prompt(self) -> str:
        """Get system prompt describing available computer use tools."""
        return f"""You have access to a computer environment with these tools:
- run_python(code): Execute Python code, returns stdout/stderr
- run_shell(command): Execute shell command, returns output
- read_file(path): Read file contents
- write_file(path, content): Write to a file
- list_files(directory): List files in directory
- search_files(pattern): Find files matching glob pattern
- browse_web(url): Fetch and extract text from a URL

To use a tool, output a JSON block:
{{"tool": "tool_name", "arguments": {{"param": "value"}}}}

The working directory is: {self.working_dir}
"""

    def get_state_summary(self) -> str:
        """Get a summary of the current computer state."""
        lines = [f"Working directory: {self.working_dir}"]
        if self.state.command_history:
            lines.append(f"Commands run: {len(self.state.command_history)}")
        if self.state.files_modified:
            lines.append(f"Files modified: {', '.join(set(self.state.files_modified))}")
        if self.state.open_urls:
            lines.append(f"URLs visited: {len(self.state.open_urls)}")
        return "\n".join(lines)

    def cleanup(self) -> None:
        """Clean up temporary resources."""
        self.code_interpreter.cleanup()
        if os.path.exists(self.working_dir) and self.working_dir.startswith(tempfile.gettempdir()):
            import shutil
            shutil.rmtree(self.working_dir, ignore_errors=True)
