"""
Claude Code Client — Tool wrapper for `claude -p` headless mode.

This is the TOOLS layer for invoking Claude Code programmatically.
The Coder Agent uses this to run coding sessions.
"""

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("tools.claude_code")


@dataclass
class ClaudeCodeResult:
    """Result from a claude -p invocation."""
    success: bool
    result_text: str
    session_id: str | None = None
    cost_usd: float | None = None
    duration_ms: int | None = None
    error: str | None = None

    @property
    def summary(self) -> str:
        cost_str = f"${self.cost_usd:.4f}" if self.cost_usd else "unknown"
        dur_str = f"{self.duration_ms / 1000:.1f}s" if self.duration_ms else "unknown"
        status = "✓" if self.success else "✗"
        return f"{status} Cost: {cost_str} | Duration: {dur_str}"


class ClaudeCodeClient:
    """
    Wrapper for Claude Code CLI (`claude -p`) headless mode.

    This is what lets our Coder agent actually write code, edit files,
    and run tests in the Django repo.
    """

    def __init__(
        self,
        working_dir: Path,
        allowed_tools: list[str] | None = None,
        max_turns: int = 30,
    ):
        self.working_dir = working_dir
        self.allowed_tools = allowed_tools or ["Read", "Edit", "Write", "Bash"]
        self.max_turns = max_turns

    def run(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_turns: int | None = None,
        timeout: int = 600,
    ) -> ClaudeCodeResult:
        """
        Run a claude -p session with the given prompt.

        Args:
            prompt: The task description for Claude Code
            system_prompt: Additional system prompt to append
            max_turns: Override max turns for this session
            timeout: Max seconds before killing the process

        Returns:
            ClaudeCodeResult with the session output
        """
        cmd = [
            "claude", "-p", prompt,
            "--output-format", "json",
            "--max-turns", str(max_turns or self.max_turns),
            "--allowedTools", ",".join(self.allowed_tools),
        ]

        if system_prompt:
            cmd.extend(["--append-system-prompt", system_prompt])

        logger.info("Running Claude Code session (max %d turns, timeout %ds)...", max_turns or self.max_turns, timeout)
        logger.debug("Prompt (first 200 chars): %s", prompt[:200])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.working_dir),
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            logger.error("Claude Code session timed out after %ds", timeout)
            return ClaudeCodeResult(
                success=False, result_text="",
                error=f"Session timed out after {timeout}s",
            )
        except FileNotFoundError:
            logger.error("'claude' CLI not found. Install: npm install -g @anthropic-ai/claude-code")
            return ClaudeCodeResult(
                success=False, result_text="",
                error="claude CLI not found",
            )

        if result.returncode != 0:
            logger.error("Claude Code failed (exit %d): %s", result.returncode, result.stderr[:500])
            return ClaudeCodeResult(
                success=False, result_text=result.stdout,
                error=result.stderr[:1000],
            )

        # Parse JSON output
        try:
            data = json.loads(result.stdout)
            return ClaudeCodeResult(
                success=data.get("subtype") == "success",
                result_text=data.get("result", ""),
                session_id=data.get("session_id"),
                cost_usd=data.get("total_cost_usd"),
                duration_ms=data.get("duration_ms"),
            )
        except json.JSONDecodeError:
            # If not JSON, treat raw stdout as result
            return ClaudeCodeResult(
                success=result.returncode == 0,
                result_text=result.stdout,
            )

    def run_with_continuation(
        self,
        prompts: list[str],
        system_prompt: str | None = None,
        timeout_per_step: int = 300,
    ) -> list[ClaudeCodeResult]:
        """
        Run a multi-step Claude Code session using --continue.

        Each prompt continues from the previous session context.
        Useful for: write code → run tests → fix failures → commit.
        """
        results = []

        for i, prompt in enumerate(prompts):
            cmd = [
                "claude", "-p", prompt,
                "--output-format", "json",
                "--max-turns", str(self.max_turns),
                "--allowedTools", ",".join(self.allowed_tools),
            ]

            if system_prompt and i == 0:
                cmd.extend(["--append-system-prompt", system_prompt])

            if i > 0:
                cmd.append("--continue")

            logger.info("Claude Code step %d/%d: %s", i + 1, len(prompts), prompt[:80])

            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True,
                    cwd=str(self.working_dir), timeout=timeout_per_step,
                )
                try:
                    data = json.loads(result.stdout)
                    cc_result = ClaudeCodeResult(
                        success=data.get("subtype") == "success",
                        result_text=data.get("result", ""),
                        session_id=data.get("session_id"),
                        cost_usd=data.get("total_cost_usd"),
                        duration_ms=data.get("duration_ms"),
                    )
                except json.JSONDecodeError:
                    cc_result = ClaudeCodeResult(
                        success=result.returncode == 0,
                        result_text=result.stdout,
                    )
                results.append(cc_result)

                if not cc_result.success:
                    logger.warning("Step %d failed, stopping continuation", i + 1)
                    break

            except subprocess.TimeoutExpired:
                results.append(ClaudeCodeResult(
                    success=False, result_text="",
                    error=f"Step {i + 1} timed out",
                ))
                break

        return results
