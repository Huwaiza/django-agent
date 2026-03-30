"""
Base Agent — The AI brain that powers all agents.

Optimized for token efficiency:
- Tracks input/output tokens and cost per call
- Cumulative cost tracking per agent and globally
- SKILL.md injection is opt-in, not default
- max_tokens tuned per call, not blanket 4096
- Global TokenBudget enforces cycle/daily spend limits
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger("agents.base")

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

# Model tiers — use the cheapest model that gets the job done
MODEL_FAST = "claude-sonnet-4-20250514"   # Routine: triage, PR body, review classification
MODEL_DEEP = "claude-opus-4-20250514"     # Complex: orchestrator decisions, self-review

# Cost per 1M tokens (USD)
COST_PER_1M = {
    MODEL_FAST: {"input": 3.00, "output": 15.00},
    MODEL_DEEP: {"input": 15.00, "output": 75.00},
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
        rates = COST_PER_1M.get(self.model, COST_PER_1M[MODEL_FAST])
        input_cost = (self.input_tokens / 1_000_000) * rates["input"]
        output_cost = (self.output_tokens / 1_000_000) * rates["output"]
        cache_discount = (self.cache_read_tokens / 1_000_000) * rates["input"] * 0.9
        return input_cost + output_cost - cache_discount


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
    Prevents runaway costs by enforcing cycle and daily limits.
    """

    def __init__(self, max_cost_per_cycle: float = 2.0, max_cost_per_day: float = 10.0):
        self.max_cost_per_cycle = max_cost_per_cycle
        self.max_cost_per_day = max_cost_per_day
        self.cycle_cost = 0.0
        self.daily_cost = 0.0
        self.call_log: list[dict] = []

    def record(self, agent_name: str, usage: TokenUsage) -> None:
        cost = usage.cost_usd
        self.cycle_cost += cost
        self.daily_cost += cost
        self.call_log.append({
            "agent": agent_name, "model": usage.model,
            "input": usage.input_tokens, "output": usage.output_tokens, "cost": cost,
        })

    def check_budget(self) -> tuple[bool, str]:
        if self.cycle_cost >= self.max_cost_per_cycle:
            return False, f"Cycle budget exceeded: ${self.cycle_cost:.4f} >= ${self.max_cost_per_cycle}"
        if self.daily_cost >= self.max_cost_per_day:
            return False, f"Daily budget exceeded: ${self.daily_cost:.4f} >= ${self.max_cost_per_day}"
        return True, "OK"

    def reset_cycle(self) -> None:
        self.cycle_cost = 0.0
        self.call_log.clear()

    def reset_daily(self) -> None:
        self.daily_cost = 0.0

    @property
    def summary(self) -> str:
        by_agent: dict[str, float] = {}
        for entry in self.call_log:
            by_agent[entry["agent"]] = by_agent.get(entry["agent"], 0) + entry["cost"]
        parts = [f"Cycle: ${self.cycle_cost:.4f} | Daily: ${self.daily_cost:.4f}"]
        if by_agent:
            breakdown = ", ".join(f"{k}=${v:.4f}" for k, v in sorted(by_agent.items(), key=lambda x: -x[1]))
            parts.append(f"By agent: {breakdown}")
        return " | ".join(parts)


_global_budget = TokenBudget()

def get_budget() -> TokenBudget:
    return _global_budget


class BaseAgent:
    """
    Base class for all AI agents.

    Token optimization:
    - max_tokens defaults to 1024 (most JSON responses are <500 tokens)
    - SKILL.md only injected when use_skill=True
    - Every call tracked in global TokenBudget
    - Budget check before every API call — refuses if over limit
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        model: str = MODEL_FAST,
        api_key: Optional[str] = None,
        skill_path: Optional[Path] = None,
        use_skill: bool = True,
    ):
        self.name = name
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.skill_path = skill_path
        self.use_skill = use_skill
        self._base_system_prompt = system_prompt
        self.system_prompt = self._build_system_prompt()
        self.budget = get_budget()

        # Memory access — all agents share the same memory instance
        from agents.memory import get_memory
        self.memory = get_memory()

        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.call_count = 0

        if not self.api_key:
            logger.warning("No ANTHROPIC_API_KEY — agent '%s' will fail on API calls.", name)

    def _build_system_prompt(self) -> str:
        prompt = self._base_system_prompt
        if self.use_skill and self.skill_path and self.skill_path.exists():
            prompt += f"\n\n--- LEARNED PATTERNS (from SKILL.md) ---\n{self.skill_path.read_text()}"
        return prompt

    def reload_skill(self) -> None:
        self.system_prompt = self._build_system_prompt()
        logger.info("Agent '%s' reloaded SKILL.md", self.name)

    def think(
        self,
        user_message: str,
        context: str = "",
        temperature: float = 0.3,
        max_tokens: int = 1024,
        response_format: str = "json",
    ) -> AgentResponse:
        """
        Core reasoning method. max_tokens defaults to 1024 — override higher
        only when you know the response will be long (e.g., Learner consolidation).
        """
        full_message = user_message
        if context:
            full_message = f"{context}\n\n---\n\n{user_message}"

        if response_format == "json":
            full_message += "\n\nRespond ONLY with valid JSON. No markdown, no backticks, no preamble."

        within_budget, reason = self.budget.check_budget()
        if not within_budget:
            logger.warning("Agent '%s' blocked by budget: %s", self.name, reason)
            return AgentResponse(raw_text=f"Budget exceeded: {reason}")

        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": self.system_prompt,
            "messages": [{"role": "user", "content": full_message}],
        }

        try:
            resp = requests.post(
                ANTHROPIC_API_URL,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                },
                json=payload, timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.error("Agent '%s' API error: %s", self.name, e)
            return AgentResponse(raw_text=f"API error: {e}")

        raw_text = "".join(b["text"] for b in data.get("content", []) if b.get("type") == "text")

        usage_data = data.get("usage", {})
        usage = TokenUsage(
            input_tokens=usage_data.get("input_tokens", 0),
            output_tokens=usage_data.get("output_tokens", 0),
            cache_read_tokens=usage_data.get("cache_read_input_tokens", 0),
            model=self.model,
        )
        self._record_usage(usage)

        parsed = None
        if response_format == "json":
            parsed = self._parse_json(raw_text)

        return AgentResponse(raw_text=raw_text, parsed=parsed, usage=usage)

    def converse(self, messages: list[dict], temperature: float = 0.3, max_tokens: int = 1024) -> AgentResponse:
        """Multi-turn conversation."""
        within_budget, reason = self.budget.check_budget()
        if not within_budget:
            return AgentResponse(raw_text=f"Budget exceeded: {reason}")

        try:
            resp = requests.post(
                ANTHROPIC_API_URL,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": self.model, "max_tokens": max_tokens,
                    "temperature": temperature, "system": self.system_prompt,
                    "messages": messages,
                },
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            return AgentResponse(raw_text=f"API error: {e}")

        raw_text = "".join(b["text"] for b in data.get("content", []) if b.get("type") == "text")
        usage_data = data.get("usage", {})
        usage = TokenUsage(
            input_tokens=usage_data.get("input_tokens", 0),
            output_tokens=usage_data.get("output_tokens", 0),
            model=self.model,
        )
        self._record_usage(usage)
        return AgentResponse(raw_text=raw_text, usage=usage)

    def _record_usage(self, usage: TokenUsage) -> None:
        self.total_input_tokens += usage.input_tokens
        self.total_output_tokens += usage.output_tokens
        self.total_cost += usage.cost_usd
        self.call_count += 1
        self.budget.record(self.name, usage)
        logger.debug(
            "%s: %d in + %d out ($%.4f) | cumulative: $%.4f",
            self.name, usage.input_tokens, usage.output_tokens,
            usage.cost_usd, self.total_cost,
        )

    @staticmethod
    def _parse_json(text: str) -> Optional[dict]:
        clean = text.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
        try:
            return json.loads(clean.strip())
        except (json.JSONDecodeError, IndexError):
            return None

    @property
    def cost_summary(self) -> str:
        return f"{self.name}: {self.call_count} calls, {self.total_input_tokens:,} in + {self.total_output_tokens:,} out, ${self.total_cost:.4f}"
