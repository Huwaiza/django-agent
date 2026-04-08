"""
Base Agent — The AI brain that powers all agents.

Uses OpenCode CLI (`opencode run`) as the LLM backend.
Model: opencode/kimi-k2.5 via OpenCode Zen.
"""

import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("agents.base")

# Model — opencode/kimi-k2.5 via OpenCode Zen
MODEL_FAST = "opencode/kimi-k2.5"
MODEL_DEEP = "opencode/kimi-k2.5"

# Cost tracking — Zen subscription, treat as 0
COST_PER_1M = {
    MODEL_FAST: {"input": 0.0, "output": 0.0},
    MODEL_DEEP: {"input": 0.0, "output": 0.0},
}


@dataclass
class TokenUsage:
    """Token usage for a single API call."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    model: str = MODEL_FAST

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> float:
        return 0.0


@dataclass
class AgentResponse:
    """Structured response from an agent."""
    raw_text: str
    parsed: Optional[dict] = None
    reasoning: str = ""
    confidence: float = 0.0
    usage: TokenUsage = field(default_factory=TokenUsage)

    @property
    def succeeded(self) -> bool:
        return self.parsed is not None

    @property
    def tokens_used(self) -> int:
        return self.usage.total_tokens

    @property
    def cost_usd(self) -> float:
        return self.usage.cost_usd


class TokenBudget:
    """
    Global token budget tracker across all agents.
    Cost is 0 on Zen subscription — budget never blocks.
    """

    def __init__(self, max_cost_per_cycle: float = 999.0, max_cost_per_day: float = 999.0):
        self.max_cost_per_cycle = max_cost_per_cycle
        self.max_cost_per_day = max_cost_per_day
        self.cycle_cost = 0.0
        self.daily_cost = 0.0
        self.call_log: list[dict] = []

    def record(self, agent_name: str, usage: TokenUsage) -> None:
        self.call_log.append({
            "agent": agent_name, "model": usage.model,
            "input": usage.input_tokens, "output": usage.output_tokens, "cost": 0.0,
        })

    def check_budget(self) -> tuple[bool, str]:
        return True, "OK"

    def reset_cycle(self) -> None:
        self.call_log.clear()

    def reset_daily(self) -> None:
        pass

    @property
    def summary(self) -> str:
        calls = len(self.call_log)
        by_agent: dict[str, int] = {}
        for entry in self.call_log:
            by_agent[entry["agent"]] = by_agent.get(entry["agent"], 0) + 1
        breakdown = ", ".join(f"{k}={v} calls" for k, v in sorted(by_agent.items()))
        return f"Total calls: {calls} | {breakdown}"


_global_budget = TokenBudget()

def get_budget() -> TokenBudget:
    return _global_budget


class BaseAgent:
    """
    Base class for all AI agents.
    Uses `opencode run` CLI to call opencode/kimi-k2.5.
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        model: str = MODEL_FAST,
        api_key: Optional[str] = None,  # unused, kept for compat
        skill_path: Optional[Path] = None,
        use_skill: bool = True,
    ):
        self.name = name
        self.model = model
        self.skill_path = skill_path
        self.use_skill = use_skill
        self._base_system_prompt = system_prompt
        self.system_prompt = self._build_system_prompt()
        self.budget = get_budget()

        from agents.memory import get_memory
        self.memory = get_memory()

        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.call_count = 0

    def _build_system_prompt(self) -> str:
        prompt = self._base_system_prompt
        if self.use_skill and self.skill_path and self.skill_path.exists():
            prompt += f"\n\n--- LEARNED PATTERNS (from SKILL.md) ---\n{self.skill_path.read_text()}"
        return prompt

    def reload_skill(self) -> None:
        self.system_prompt = self._build_system_prompt()
        logger.info("Agent '%s' reloaded SKILL.md", self.name)

    def _call_opencode(self, full_prompt: str, timeout: int = 120) -> str:
        """Invoke `opencode run` and return the raw text output."""
        cmd = ["opencode", "run", "--model", self.model, full_prompt]
        logger.debug("Agent '%s' calling opencode run (model=%s)", self.name, self.model)
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd="/tmp",  # neutral dir — prevents opencode from using repo context
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            logger.error("Agent '%s' opencode run timed out", self.name)
            return ""
        except FileNotFoundError:
            logger.error("'opencode' CLI not found. Install: brew install anomalyco/tap/opencode")
            return ""

        if result.returncode != 0:
            logger.error("Agent '%s' opencode error: %s", self.name, result.stderr[:300])
            return ""

        return result.stdout.strip()

    def think(
        self,
        user_message: str,
        context: str = "",
        temperature: float = 0.3,
        max_tokens: int = 1024,
        response_format: str = "json",
        timeout: int = 120,
    ) -> AgentResponse:
        """Core reasoning method."""
        if response_format == "json":
            full_message = (
                "OUTPUT FORMAT: You MUST respond with ONLY a valid JSON object or array. "
                "No prose, no explanation, no markdown, no backticks. Start your response with { or [.\n\n"
            )
        else:
            full_message = ""

        full_message += (
            f"YOUR ROLE: {self.system_prompt}\n\n"
        )
        if context:
            full_message += f"CONTEXT:\n{context}\n\n---\n\n"
        full_message += user_message
        if response_format == "json":
            full_message += "\n\nRemember: respond with ONLY the JSON. Start with { or [."

        raw_text = self._call_opencode(full_message, timeout=timeout)
        usage = TokenUsage(model=self.model)
        self._record_usage(usage)

        parsed = None
        if response_format == "json":
            parsed = self._parse_json(raw_text)

        return AgentResponse(raw_text=raw_text, parsed=parsed, usage=usage)

    def converse(self, messages: list[dict], temperature: float = 0.3, max_tokens: int = 1024) -> AgentResponse:
        """Multi-turn conversation — flattened into a single prompt for opencode run."""
        parts = [f"SYSTEM INSTRUCTIONS:\n{self.system_prompt}\n"]
        for msg in messages:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            parts.append(f"{role}:\n{content}")
        full_prompt = "\n\n".join(parts)

        raw_text = self._call_opencode(full_prompt)
        usage = TokenUsage(model=self.model)
        self._record_usage(usage)
        return AgentResponse(raw_text=raw_text, usage=usage)

    def _record_usage(self, usage: TokenUsage) -> None:
        self.call_count += 1
        self.budget.record(self.name, usage)
        logger.debug("Agent '%s' call #%d complete", self.name, self.call_count)

    @staticmethod
    def _parse_json(text: str) -> Optional[dict]:
        import re
        clean = text.strip()

        # Strip markdown fences
        if clean.startswith("```"):
            clean = re.sub(r"^```[a-z]*\n?", "", clean)
            clean = re.sub(r"\n?```$", "", clean)
            clean = clean.strip()

        # Try direct parse first
        try:
            return json.loads(clean)
        except (json.JSONDecodeError, ValueError):
            pass

        # Extract JSON from prose — try array first, then object
        # (array pattern must come first to avoid matching {} inside [{}])
        for pattern in [r"\[[\s\S]*\]", r"\{[\s\S]*\}"]:
            match = re.search(pattern, clean)
            if match:
                try:
                    return json.loads(match.group())
                except (json.JSONDecodeError, ValueError):
                    pass

        return None

    @property
    def cost_summary(self) -> str:
        return f"{self.name}: {self.call_count} calls (opencode/kimi-k2.5)"
