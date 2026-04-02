"""
OpenCode Client — Tool wrapper for `opencode run` headless mode.

Replaces the previous `claude -p` invocation.
The Coder Agent uses this to run coding sessions on the Django repo.
"""

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("tools.claude_code")

OPENCODE_MODEL = "opencode/kimi-k2.5"


@dataclass
class ClaudeCodeResult:
    """Result from an opencode run invocation."""
    success: bool
    result_text: str
    session_id: str | None = None
    cost_usd: float | None = None
    duration_ms: int | None = None
    error: str | None = None

    @property
    def summary(self) -> str:
        status = "✓" if self.success else "✗"
        return f"{status} opencode/kimi-k2.5 | success={self.success}"


class ClaudeCodeClient:
    """
    Wrapper for OpenCode CLI (`opencode run`) headless mode.
    Replaces `claude -p` for writing patches in the Django repo.
    """

    def __init__(
        self,
        working_dir: Path,
        allowed_tools: list[str] | None = None,
        max_turns: int = 30,
    ):
        self.working_dir = working_dir
        self.model = OPENCODE_MODEL
        # allowed_tools / max_turns kept for API compat but opencode run handles tools internally

    def run(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_turns: int | None = None,
        timeout: int = 600,
    ) -> ClaudeCodeResult:
        """
        Run an opencode run session with the given prompt.

        Args:
            prompt: The task description
            system_prompt: Additional system context prepended to prompt
            max_turns: Unused (opencode manages turns internally)
            timeout: Max seconds before killing the process

        Returns:
            ClaudeCodeResult with the session output
        """
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n---\n\n{prompt}"

        cmd = ["opencode", "run", "--model", self.model, full_prompt]

        logger.info("Running opencode session (model=%s, timeout=%ds)...", self.model, timeout)
        logger.debug("Prompt (first 200 chars): %s", full_prompt[:200])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.working_dir),
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            logger.error("opencode session timed out after %ds", timeout)
            return ClaudeCodeResult(
                success=False, result_text="",
                error=f"Session timed out after {timeout}s",
            )
        except FileNotFoundError:
            logger.error("'opencode' CLI not found. Install: brew install anomalyco/tap/opencode")
            return ClaudeCodeResult(
                success=False, result_text="",
                error="opencode CLI not found",
            )

        if result.returncode != 0:
            logger.error("opencode failed (exit %d): %s", result.returncode, result.stderr[:500])
            return ClaudeCodeResult(
                success=False, result_text=result.stdout,
                error=result.stderr[:1000],
            )

        cleaned = result.stdout.strip()

        return ClaudeCodeResult(
            success=result.returncode == 0,
            result_text=cleaned,
        )

    def run_with_continuation(
        self,
        prompts: list[str],
        system_prompt: str | None = None,
        timeout_per_step: int = 300,
    ) -> list[ClaudeCodeResult]:
        """
        Run multi-step prompts sequentially.
        Each prompt is a fresh opencode run call (opencode run has no --continue flag).
        """
        results = []

        for i, prompt in enumerate(prompts):
            full_prompt = prompt
            if system_prompt and i == 0:
                full_prompt = f"{system_prompt}\n\n---\n\n{prompt}"

            logger.info("opencode step %d/%d: %s", i + 1, len(prompts), prompt[:80])
            result = self.run(full_prompt, timeout=timeout_per_step)
            results.append(result)

            if not result.success:
                logger.warning("Step %d failed, stopping continuation", i + 1)
                break

        return results
