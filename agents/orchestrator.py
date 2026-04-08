"""
Orchestrator Agent — The AI coordinator that runs the full contribution pipeline.

This is the MASTER AGENT. It:
1. Decides what to do on each cycle (scout for new tickets? check PR reviews? etc.)
2. Invokes sub-agents (Scout, Picker, Coder, Reviewer Handler)
3. Reviews each sub-agent's output before passing it downstream
4. Maintains state across runs (what's in progress, what's waiting for review)
5. Implements quality gates and circuit breakers
6. Escalates to human when needed

The Orchestrator itself is an AI agent — it uses Claude to REASON about
what the system should do, rather than following a hardcoded state machine.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from agents.base import BaseAgent, AgentResponse, MODEL_DEEP, MODEL_FAST
from agents.scout import ScoutAgent, TicketEvaluation
from agents.coder import CoderAgent, CodingResult
from agents.pr_maker import PRMakerAgent
from agents.review_handler import ReviewHandlerAgent, ReviewAction
from agents.learner import LearnerAgent
from agents.escalator import EscalatorAgent, EscalationEvent
from config.prompts import ORCHESTRATOR_SYSTEM_PROMPT, PICKER_SYSTEM_PROMPT

logger = logging.getLogger("agents.orchestrator")


@dataclass
class SystemState:
    """Current state of the entire contribution system."""
    # Tickets and PRs
    candidates: list[TicketEvaluation] = field(default_factory=list)
    active_tickets: list[dict] = field(default_factory=list)  # being worked on
    open_prs: list[dict] = field(default_factory=list)  # submitted, awaiting review
    merged_prs: list[dict] = field(default_factory=list)
    rejected_prs: list[dict] = field(default_factory=list)

    # Health metrics
    consecutive_rejections: int = 0
    total_runs: int = 0
    last_run_at: Optional[str] = None
    circuit_breaker_active: bool = False

    # Learnings
    skill_md_version: int = 0

    # Scout history — every scout run appended, never overwritten
    scout_history: list[dict] = field(default_factory=list)
    last_scout_at: Optional[str] = None

    def to_context_string(self) -> str:
        """Format state for the Orchestrator AI to read."""
        ready_for_pr = [t for t in self.active_tickets if t.get("status") == "ready_for_pr"]
        in_progress = [t for t in self.active_tickets if t.get("status") not in ("ready_for_pr", "coding_failed")]

        return f"""
SYSTEM STATE (as of {datetime.now().isoformat()}):

Tickets ready to submit as PR: {len(ready_for_pr)}
{self._format_items(ready_for_pr)}

Tickets being coded: {len(in_progress)}
{self._format_items(in_progress)}

Open PRs (awaiting review): {len(self.open_prs)}
{self._format_items(self.open_prs)}

Merged PRs (lifetime): {len(self.merged_prs)}
Rejected PRs (lifetime): {len(self.rejected_prs)}
Consecutive rejections: {self.consecutive_rejections}
Circuit breaker active: {self.circuit_breaker_active}
Total orchestrator runs: {self.total_runs}
Last run: {self.last_run_at or 'never'}
SKILL.md version: {self.skill_md_version}

Available candidates from last scout: {len(self.candidates)}
""".strip()

    @staticmethod
    def _format_items(items: list[dict]) -> str:
        if not items:
            return "  (none)"
        lines = []
        for item in items:
            status = item.get("status", "")
            status_str = f" [{status}]" if status else ""
            lines.append(f"  - #{item.get('ticket_id', '?')}: {item.get('summary', '?')[:50]}{status_str}")
        return "\n".join(lines)


ORCHESTRATOR_DECISION_PROMPT = """\
Based on the current system state, decide what actions to take in this cycle.

{state_context}

Choose ONE OR MORE actions from:
- SCOUT: Run the Scout agent to discover new ticket candidates
- PICK_AND_CODE: Select a ticket from candidates and start working on it
- SUBMIT_PR: Push branch and open PR for tickets with status "ready_for_pr"
- CHECK_REVIEWS: Check open PRs for new reviewer comments
- RESPOND_TO_REVIEWS: Respond to pending reviewer comments
- LEARN: Extract lessons from recent merges/rejections
- PAUSE: Do nothing this cycle (e.g., waiting for reviews, circuit breaker active)
- ESCALATE: Something needs human attention

Respond with a JSON object:
{{
    "actions": ["<action1>", "<action2>"],
    "reasoning": "<why these actions, in 2-3 sentences>",
    "priority_action": "<which action to do first>",
    "should_scout": <true if candidates list is empty or stale>,
    "confidence": <0.0-1.0>
}}

Rules:
- If circuit_breaker_active is true, ONLY action is PAUSE or ESCALATE
- If we have 0 candidates and 0 active tickets, we MUST SCOUT
- If tickets are "ready_for_pr", ALWAYS include SUBMIT_PR — don't leave patches sitting
- If we have open PRs, always CHECK_REVIEWS
- Don't PICK_AND_CODE if we already have 3+ active tickets
- If consecutive_rejections >= 3, activate circuit breaker and ESCALATE
"""

PICKER_DECISION_PROMPT = """\
Here are the candidate tickets evaluated by the Scout agent. Select the SINGLE BEST \
ticket to work on next.

{candidates_context}

Current system context:
- Components we've successfully contributed to: {successful_components}
- Components where PRs were recently rejected: {rejected_components}
- Active tickets (don't pick the same component if possible): {active_components}

Respond with a JSON object:
{{
    "selected_ticket_id": <int>,
    "reasoning": "<why this ticket over others, 2-3 sentences>",
    "confidence": <0.0-1.0>,
    "expected_complexity": "<trivial|simple|moderate>",
    "fallback_ticket_id": <int or null if no good fallback>
}}
"""


class PickerAgent(BaseAgent):
    """AI agent that selects the best ticket from Scout's candidates."""

    def __init__(self, skill_path: Path | None = None, api_key: str | None = None):
        super().__init__(
            name="picker",
            system_prompt=PICKER_SYSTEM_PROMPT,
            model=MODEL_FAST,
            api_key=api_key,
            skill_path=skill_path,
            use_skill=False,  # TOKEN OPTIMIZATION: Picker decides from candidate data, not SKILL.md
        )

    def pick(
        self,
        candidates: list[TicketEvaluation],
        successful_components: list[str] | None = None,
        rejected_components: list[str] | None = None,
        active_components: list[str] | None = None,
    ) -> dict:
        """Select the best ticket to work on from Scout's evaluations."""
        # Format candidates for the AI
        candidates_text = []
        for e in candidates:
            if not e.is_candidate:
                continue
            candidates_text.append(
                f"Ticket #{e.ticket_id} [{e.verdict}, {e.score}/100]\n"
                f"  Summary: {e.ticket.summary}\n"
                f"  Component: {e.ticket.component}\n"
                f"  Type: {e.ticket.ticket_type}\n"
                f"  Complexity: {e.estimated_complexity}\n"
                f"  Clarity: {e.clarity}\n"
                f"  Fix approach: {e.fix_approach}\n"
                f"  Risks: {', '.join(e.risk_factors) or 'none'}\n"
                f"  Reasoning: {e.reasoning}"
            )

        if not candidates_text:
            return {"selected_ticket_id": None, "reasoning": "No viable candidates", "confidence": 0}

        prompt = PICKER_DECISION_PROMPT.format(
            candidates_context="\n\n".join(candidates_text),
            successful_components=", ".join(successful_components or ["none yet"]),
            rejected_components=", ".join(rejected_components or ["none yet"]),
            active_components=", ".join(active_components or ["none"]),
        )

        response = self.think(
            user_message=prompt,
            temperature=0.2,
            max_tokens=512,  # TOKEN OPTIMIZATION: picker response is ~200 tokens JSON
        )

        if response.succeeded:
            return response.parsed
        return {"selected_ticket_id": None, "reasoning": "Picker failed to decide", "confidence": 0}


class Orchestrator(BaseAgent):
    """
    Master AI agent that coordinates the entire contribution pipeline.

    On each cycle:
    1. Reads current system state
    2. AI decides what actions to take (not hardcoded if/else!)
    3. Executes actions by invoking sub-agents
    4. Updates state
    5. Logs everything for human review

    The key difference from a script: the Orchestrator REASONS about what to do.
    If there are 3 open PRs with pending reviews, it prioritizes responding to those
    over opening new tickets. If we just got rejected, it reasons about why and
    adjusts strategy. A script would need all these cases hardcoded.
    """

    def __init__(
        self,
        skill_path: Path | None = None,
        api_key: str | None = None,
        state_path: Path | None = None,
        repo_path: Path | None = None,
        github_fork: str | None = None,
    ):
        super().__init__(
            name="orchestrator",
            system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
            model=MODEL_FAST,  # TOKEN OPTIMIZATION: coordination decisions don't need Opus
            api_key=api_key,
            skill_path=skill_path,
            use_skill=False,  # TOKEN OPTIMIZATION: orchestrator coordinates, doesn't code
        )

        self.state_path = state_path or Path("db/orchestrator_state.json")
        self.repo_path = repo_path
        self.github_fork = github_fork
        self.state = self._load_state()

        # Sub-agents
        self.scout = ScoutAgent(skill_path=skill_path, api_key=api_key)
        self.picker = PickerAgent(skill_path=skill_path, api_key=api_key)
        self.coder = CoderAgent(
            repo_path=repo_path or Path.cwd(),
            skill_path=skill_path,
            api_key=api_key,
        ) if repo_path else None
        self.pr_maker = PRMakerAgent(
            repo_path=repo_path or Path.cwd(),
            github_fork=github_fork or "",
            api_key=api_key,
        ) if repo_path and github_fork else None
        self.review_handler = ReviewHandlerAgent(
            repo_path=repo_path,
            api_key=api_key,
        )
        self.learner = LearnerAgent(
            skill_path=skill_path or Path("skills/django-contributor/SKILL.md"),
            api_key=api_key,
        )
        self.escalator = EscalatorAgent(api_key=api_key)

        # Collect all sub-agents for cost reporting
        self._all_agents = [
            self.scout, self.picker, self.coder, self.pr_maker,
            self.review_handler, self.learner, self.escalator,
        ]

    def run_cycle(self) -> dict:
        """
        Execute one orchestration cycle.

        Returns a summary of what happened this cycle.
        """
        self.state.total_runs += 1
        self.state.last_run_at = datetime.now().isoformat()
        cycle_log = {"run": self.state.total_runs, "actions": [], "timestamp": self.state.last_run_at}

        # Reset short-term memory for this cycle
        self.memory.reset_cycle()
        self.memory.short.set("cycle_number", self.state.total_runs, agent="orchestrator")

        # Step 1: AI decides what to do
        logger.info("── Orchestrator Cycle #%d ──", self.state.total_runs)
        decision = self._decide_actions()
        cycle_log["decision"] = decision

        if not decision or "actions" not in decision:
            logger.error("Orchestrator failed to decide actions")
            cycle_log["error"] = "Decision failed"
            return cycle_log

        logger.info("Decision: %s (confidence: %.0f%%)", decision["actions"], decision.get("confidence", 0) * 100)
        logger.info("Reasoning: %s", decision.get("reasoning", ""))

        # Step 2: Execute each action
        for action in decision["actions"]:
            # Check budget before each action
            within_budget, budget_reason = self.budget.check_budget()
            if not within_budget:
                logger.warning("Budget exceeded, skipping remaining actions: %s", budget_reason)
                cycle_log["budget_exceeded"] = budget_reason
                break

            logger.info("Executing action: %s", action)
            result = self._execute_action(action)
            cycle_log["actions"].append({"action": action, "result": result})

        # Step 3: Log token budget
        cycle_log["budget"] = self.budget.summary
        logger.info("💰 %s", self.budget.summary)

        # Log per-agent costs
        for agent in self._all_agents:
            if agent and agent.call_count > 0:
                logger.info("   %s", agent.cost_summary)

        # Step 4: Save state and reset cycle budget
        self._save_state()
        self.budget.reset_cycle()

        return cycle_log

    def _decide_actions(self) -> dict | None:
        """Ask the AI what actions to take this cycle."""
        memory_ctx = self.memory.long.to_context_string()
        prompt = ORCHESTRATOR_DECISION_PROMPT.format(
            state_context=self.state.to_context_string() + "\n\n" + memory_ctx
        )

        response = self.think(
            user_message=prompt,
            temperature=0.2,
            max_tokens=1024,
        )

        if response.succeeded:
            return response.parsed

        # Fallback: if AI decision fails, use simple heuristics
        logger.warning("AI decision failed, using fallback heuristics")
        ready_for_pr = [t for t in self.state.active_tickets if t.get("status") == "ready_for_pr"]
        if ready_for_pr:
            return {"actions": ["SUBMIT_PR"], "reasoning": "fallback: tickets ready for PR", "confidence": 0.8}
        viable_candidates = [e for e in self.state.candidates if e.is_candidate]
        if viable_candidates:
            return {"actions": ["PICK_AND_CODE"], "reasoning": f"fallback: {len(viable_candidates)} saved candidates, skipping scout", "confidence": 0.8}
        if self.state.open_prs:
            return {"actions": ["CHECK_REVIEWS"], "reasoning": "fallback: has open PRs", "confidence": 0.5}
        return {"actions": ["SCOUT"], "reasoning": "fallback: no candidates, need to scout", "confidence": 0.5}

    def _execute_action(self, action: str) -> dict:
        """Execute a single action by invoking the appropriate sub-agent."""
        if action == "SCOUT":
            return self._action_scout()
        elif action == "PICK_AND_CODE":
            return self._action_pick_and_code()
        elif action == "SUBMIT_PR":
            return self._action_submit_pr()
        elif action == "CHECK_REVIEWS":
            return self._action_check_reviews()
        elif action == "LEARN":
            return self._action_learn()
        elif action == "PAUSE":
            logger.info("Pausing this cycle.")
            return {"status": "paused"}
        elif action == "ESCALATE":
            return self._action_escalate()
        else:
            logger.warning("Unknown action: %s", action)
            return {"status": "unknown_action", "action": action}

    def _action_scout(self) -> dict:
        """Run the Scout agent to find new candidates."""
        try:
            evaluations = self.scout.discover(deep_eval_limit=10)

            # IDs already acted on — don't resurface as candidates
            acted_ids = (
                {t.get("ticket_id") for t in self.state.active_tickets}
                | {p.get("ticket_id") for p in self.state.open_prs}
                | {p.get("ticket_id") for p in self.state.merged_prs}
                | {p.get("ticket_id") for p in self.state.rejected_prs}
            )
            new_evals = [e for e in evaluations if e.ticket_id not in acted_ids]

            # Merge: existing candidates keyed by ticket_id; new run overwrites stale entries
            existing_by_id = {e.ticket_id: e for e in self.state.candidates}
            for e in new_evals:
                existing_by_id[e.ticket_id] = e
            self.state.candidates = list(existing_by_id.values())

            # Append to history — never overwrite past runs
            self.state.scout_history.append({
                "scanned_at": datetime.now().isoformat(),
                "evaluated": len(evaluations),
                "new_candidate_ids": [e.ticket_id for e in new_evals if e.is_candidate],
                "results": [
                    {"ticket_id": e.ticket_id, "verdict": e.verdict,
                     "score": e.score, "summary": e.ticket.summary[:60]}
                    for e in evaluations
                ],
            })
            self.state.last_scout_at = datetime.now().isoformat()

            picks = [e for e in evaluations if e.verdict == "PICK"]
            maybes = [e for e in evaluations if e.verdict == "MAYBE"]

            # Store in short-term memory (available to Picker this cycle)
            self.memory.short.set("scout_picks", len(picks), agent="scout")
            self.memory.short.set("scout_maybes", len(maybes), agent="scout")

            # Store each evaluation in contextual memory (persists for Coder/ReviewHandler)
            for e in evaluations:
                if e.is_candidate:
                    self.memory.ctx.set("ticket", str(e.ticket_id), "scout_verdict", e.verdict, agent="scout")
                    self.memory.ctx.set("ticket", str(e.ticket_id), "scout_score", e.score, agent="scout")
                    self.memory.ctx.set("ticket", str(e.ticket_id), "fix_approach", e.fix_approach or "", agent="scout")
                    self.memory.ctx.set("ticket", str(e.ticket_id), "component", e.ticket.component, agent="scout")
                    self.memory.ctx.set("ticket", str(e.ticket_id), "summary", e.ticket.summary, agent="scout")

            logger.info("Scout found %d PICKs and %d MAYBEs", len(picks), len(maybes))
            return {
                "status": "success",
                "total_evaluated": len(evaluations),
                "picks": len(picks),
                "maybes": len(maybes),
            }
        except Exception as e:
            logger.error("Scout failed: %s", e)
            return {"status": "error", "error": str(e)}

    def _action_pick_and_code(self) -> dict:
        """Use Picker agent to select a ticket, then invoke the Coder agent."""
        viable = [e for e in self.state.candidates if e.is_candidate]
        if not viable:
            logger.info("No viable candidates to pick from")
            return {"status": "no_candidates"}

        # AI Picker selects the best ticket — fed by long-term memory
        decision = self.picker.pick(
            candidates=viable,
            successful_components=self.memory.long.get_successful_components(),
            rejected_components=self.memory.long.get_rejected_components(),
            active_components=[t.get("component") for t in self.state.active_tickets],
        )

        selected_id = decision.get("selected_ticket_id")
        if not selected_id:
            logger.info("Picker couldn't select a ticket: %s", decision.get("reasoning"))
            return {"status": "picker_declined", "reasoning": decision.get("reasoning")}

        logger.info(
            "Picker selected #%d (confidence: %.0f%%): %s",
            selected_id, decision.get("confidence", 0) * 100, decision.get("reasoning", ""),
        )

        # Store picker decision in contextual memory
        self.memory.ctx.set("ticket", str(selected_id), "picker_reasoning", decision.get("reasoning", ""), agent="picker")
        self.memory.ctx.set("ticket", str(selected_id), "picker_confidence", decision.get("confidence", 0), agent="picker")
        self.memory.short.set("picked_ticket", selected_id, agent="picker")

        # Find the full evaluation for the selected ticket
        selected_eval = next((e for e in viable if e.ticket_id == selected_id), None)
        if not selected_eval:
            return {"status": "error", "error": f"Selected ticket #{selected_id} not found in candidates"}

        # Add to active tickets
        self.state.active_tickets.append({
            "ticket_id": selected_id,
            "summary": selected_eval.ticket.summary,
            "component": selected_eval.ticket.component,
            "fix_approach": selected_eval.fix_approach,
            "needs_docs": getattr(selected_eval.ticket, "needs_docs", False),
            "picked_at": datetime.now().isoformat(),
            "status": "coding",
        })

        # ── Invoke the Coder Agent ──
        if not self.coder:
            logger.warning("No repo_path configured — cannot run Coder agent")
            return {
                "status": "picked_but_no_coder",
                "ticket_id": selected_id,
                "summary": selected_eval.ticket.summary,
                "fix_approach": selected_eval.fix_approach,
                "note": "Set --repo to enable the Coder agent",
            }

        coding_result = self.coder.code_fix(selected_eval)

        # Update ticket status based on coding result
        active_entry = next(
            (t for t in self.state.active_tickets if t["ticket_id"] == selected_id), None
        )

        # Store coding result in contextual memory (persists for ReviewHandler)
        self.memory.ctx.set("ticket", str(selected_id), "coding_verdict", coding_result.self_review_verdict, agent="coder")
        self.memory.ctx.set("ticket", str(selected_id), "coding_score", coding_result.self_review_score, agent="coder")
        self.memory.ctx.set("ticket", str(selected_id), "branch_name", coding_result.branch_name, agent="coder")
        self.memory.short.set("coding_result", {
            "ticket_id": selected_id, "verdict": coding_result.self_review_verdict,
            "score": coding_result.self_review_score, "ready": coding_result.ready_for_pr,
        }, agent="coder")

        if coding_result.ready_for_pr:
            if active_entry:
                active_entry["status"] = "ready_for_pr"
                active_entry["branch_name"] = coding_result.branch_name
                active_entry["self_review_score"] = coding_result.self_review_score
                active_entry["self_review_summary"] = coding_result.self_review_summary
                active_entry["diff_stat"] = coding_result.diff_stat
            logger.info("✅ Patch ready for PR: #%d on branch %s", selected_id, coding_result.branch_name)
        else:
            if active_entry:
                active_entry["status"] = "coding_failed"
                active_entry["error"] = coding_result.error or coding_result.self_review_summary
            self.state.consecutive_rejections += 1
            # Record failure in long-term memory
            self.memory.long.record_pr_outcome(
                ticket_id=selected_id,
                component=selected_eval.ticket.component,
                outcome="coding_failed",
                self_review_score=coding_result.self_review_score,
                reviewer_feedback=coding_result.self_review_summary,
            )
            if self.state.consecutive_rejections >= 3:
                self.state.circuit_breaker_active = True
                logger.warning("🔴 Circuit breaker ACTIVATED after %d consecutive failures", self.state.consecutive_rejections)

        return {
            "status": "coded" if coding_result.success else "coding_failed",
            "ticket_id": selected_id,
            "branch_name": coding_result.branch_name,
            "ready_for_pr": coding_result.ready_for_pr,
            "self_review_verdict": coding_result.self_review_verdict,
            "self_review_score": coding_result.self_review_score,
            "test_passed": coding_result.test_passed,
            "fix_iterations": coding_result.fix_iterations,
            "cost_usd": coding_result.total_cost_usd,
            "error": coding_result.error,
        }

    def _action_check_reviews(self) -> dict:
        """Check open PRs for CI failures and reviewer comments."""
        if not self.state.open_prs:
            logger.info("No open PRs to check")
            return {"status": "no_open_prs"}

        # ── Step 1: Fix CI failures first ──
        ci_results = []
        for pr in self.state.open_prs:
            pr_number = pr.get("pr_number")
            if not pr_number:
                continue
            ticket_id = pr.get("ticket_id", 0)
            summary = pr.get("summary", "")
            branch = pr.get("branch", f"ticket_{ticket_id}")
            logger.info("Checking CI for PR #%d...", pr_number)
            fixes = self.review_handler.fix_ci_failures(
                pr_number=pr_number,
                ticket_id=ticket_id,
                summary=summary,
                branch_name=branch,
            )
            if fixes:
                ci_results.extend(fixes)
                for fix in fixes:
                    logger.info("  CI fix '%s': fixed=%s", fix["check_name"], fix["fixed"])

        # ── Step 2: Handle human reviewer comments ──
        logger.info("Checking %d open PRs for reviewer comments...", len(self.state.open_prs))
        actions = self.review_handler.check_and_handle_reviews(self.state.open_prs)

        escalations, approvals, auto_handled = [], [], []

        for action in actions:
            if action.escalated:
                escalations.append(action)
                self.escalator.escalate(EscalationEvent(
                    tier=2,
                    title=f"Reviewer comment needs attention on #{action.comment.ticket_id}",
                    detail=f"Category: {action.category}\nComment: {action.comment.comment_text[:300]}",
                    ticket_id=action.comment.ticket_id,
                    pr_url=f"https://github.com/django/django/pull/{action.comment.pr_number}",
                    suggested_action="Review the comment and decide how to respond.",
                ))
            elif action.category == "approval":
                approvals.append(action)
                pr_entry = next((p for p in self.state.open_prs if p.get("pr_number") == action.comment.pr_number), None)
                if pr_entry:
                    self.state.merged_prs.append(pr_entry)
                    self.state.open_prs.remove(pr_entry)
                    self.state.consecutive_rejections = 0
                    # Record in long-term memory
                    self.memory.long.record_pr_outcome(
                        ticket_id=pr_entry.get("ticket_id", 0),
                        component=pr_entry.get("component", ""),
                        outcome="merged",
                        pr_number=pr_entry.get("pr_number"),
                        self_review_score=pr_entry.get("self_review_score", 0),
                    )
            elif action.category == "rejection":
                pr_entry = next((p for p in self.state.open_prs if p.get("pr_number") == action.comment.pr_number), None)
                if pr_entry:
                    pr_entry["rejection_reason"] = action.comment.comment_text[:200]
                    self.state.rejected_prs.append(pr_entry)
                    self.state.open_prs.remove(pr_entry)
                    self.state.consecutive_rejections += 1
                    # Record in long-term memory
                    self.memory.long.record_pr_outcome(
                        ticket_id=pr_entry.get("ticket_id", 0),
                        component=pr_entry.get("component", ""),
                        outcome="rejected",
                        pr_number=pr_entry.get("pr_number"),
                        reviewer_feedback=action.comment.comment_text[:500],
                    )
                    if self.state.consecutive_rejections >= 3:
                        self.state.circuit_breaker_active = True
            else:
                auto_handled.append(action)

            # Feed every meaningful reviewer interaction to the Learner
            if action.category not in ("informational",):
                self.learner.extract_lessons(
                    ticket_id=action.comment.ticket_id,
                    summary=action.comment.summary,
                    component="",
                    outcome=action.category,
                    review_thread=action.comment.comment_text,
                    pr_number=action.comment.pr_number,
                )

        # Reload SKILL.md for agents that use it (Learner may have updated it)
        self.scout.reload_skill()
        if self.coder:
            self.coder.reload_skill()

        logger.info("Review check: %d auto-handled, %d approvals, %d escalated",
                     len(auto_handled), len(approvals), len(escalations))
        return {
            "status": "reviewed", "total_comments": len(actions),
            "auto_handled": len(auto_handled), "approvals": len(approvals),
            "escalated": len(escalations),
            "ci_fixes": len([f for f in ci_results if f.get("fixed")]),
            "ci_failures_remaining": len([f for f in ci_results if not f.get("fixed")]),
        }

    def _action_submit_pr(self) -> dict:
        """Submit PRs for all tickets with status 'ready_for_pr'."""
        if not self.pr_maker:
            logger.warning("PR-Maker not configured — need --repo and --fork")
            return {"status": "no_pr_maker", "note": "Set --repo and --fork to enable PR submission"}

        ready = [t for t in self.state.active_tickets if t.get("status") == "ready_for_pr"]
        if not ready:
            logger.info("No tickets ready for PR submission")
            return {"status": "nothing_to_submit"}

        results = []
        for ticket_entry in ready:
            ticket_id = ticket_entry["ticket_id"]
            summary = ticket_entry.get("summary", "")
            component = ticket_entry.get("component", "")
            branch = ticket_entry.get("branch_name", f"ticket_{ticket_id}")

            # Build a minimal CodingResult for the PR-Maker
            # (the real CodingResult was consumed in the coding step,
            #  but the essential fields are preserved in the state entry)
            from agents.coder import CodingResult
            coding_result = CodingResult(
                ticket_id=ticket_id,
                branch_name=branch,
                success=True,
                diff="",  # not needed for PR submission
                diff_stat=ticket_entry.get("diff_stat", ""),
                test_passed=True,
                self_review_verdict="APPROVE",
                self_review_score=ticket_entry.get("self_review_score", 0),
                self_review_issues=[],
                self_review_summary=ticket_entry.get("self_review_summary", "Patch approved by self-review."),
                coding_cost_usd=0,
                review_cost_usd=0,
                fix_iterations=0,
            )

            logger.info("Submitting PR for #%d on branch %s...", ticket_id, branch)
            pr_result = self.pr_maker.submit_pr(
                coding_result=coding_result,
                ticket_summary=summary,
                component=component,
                needs_docs=ticket_entry.get("needs_docs", False),
            )

            if pr_result["success"]:
                # Move from active_tickets to open_prs
                ticket_entry["status"] = "pr_submitted"
                self.state.open_prs.append({
                    "ticket_id": ticket_id,
                    "summary": summary,
                    "component": component,
                    "pr_url": pr_result.get("pr_url"),
                    "pr_number": pr_result.get("pr_number"),
                    "branch": branch,
                    "submitted_at": datetime.now().isoformat(),
                })
                # Reset consecutive rejections on successful submission
                self.state.consecutive_rejections = 0
                logger.info("✅ PR submitted: %s", pr_result.get("pr_url"))
            else:
                ticket_entry["status"] = "pr_failed"
                ticket_entry["error"] = pr_result.get("error")
                logger.error("PR submission failed for #%d: %s", ticket_id, pr_result.get("error"))

            results.append(pr_result)

        # Clean up submitted tickets from active list
        self.state.active_tickets = [
            t for t in self.state.active_tickets
            if t.get("status") != "pr_submitted"
        ]

        submitted = sum(1 for r in results if r["success"])
        failed = sum(1 for r in results if not r["success"])
        return {"status": "submitted", "submitted": submitted, "failed": failed, "details": results}

    def _action_learn(self) -> dict:
        """Extract lessons from recent merges and rejections via Learner agent."""
        lessons_extracted = 0

        for pr in self.state.merged_prs:
            if pr.get("lessons_extracted"):
                continue
            lessons = self.learner.extract_lessons(
                ticket_id=pr.get("ticket_id", 0),
                summary=pr.get("summary", ""),
                component=pr.get("component", ""),
                outcome="merged",
                review_thread=f"PR was approved and merged. Score: {pr.get('self_review_score', 'N/A')}",
                pr_number=pr.get("pr_number"),
            )
            pr["lessons_extracted"] = True
            lessons_extracted += len(lessons)

        for pr in self.state.rejected_prs:
            if pr.get("lessons_extracted"):
                continue
            lessons = self.learner.extract_lessons(
                ticket_id=pr.get("ticket_id", 0),
                summary=pr.get("summary", ""),
                component=pr.get("component", ""),
                outcome="rejected",
                review_thread=pr.get("rejection_reason", "No reason captured."),
                pr_number=pr.get("pr_number"),
            )
            pr["lessons_extracted"] = True
            lessons_extracted += len(lessons)

        self.learner.consolidate_skill()

        # Reload SKILL.md for agents that use it
        self.scout.reload_skill()
        if self.coder:
            self.coder.reload_skill()

        self.state.skill_md_version += 1
        logger.info("Learned %d lessons (SKILL.md v%d)", lessons_extracted, self.state.skill_md_version)
        return {"status": "learned", "lessons_extracted": lessons_extracted, "skill_version": self.state.skill_md_version}

    def _action_escalate(self) -> dict:
        """Escalate to human operator via Escalator agent."""
        if self.state.circuit_breaker_active:
            reason = f"Circuit breaker active after {self.state.consecutive_rejections} consecutive rejections"
            tier = 3
        else:
            reason = "AI orchestrator requested human review"
            tier = 2

        result = self.escalator.escalate(EscalationEvent(
            tier=tier,
            title="Django Agent needs human attention",
            detail=reason,
            suggested_action="Review recent PR rejections and adjust strategy.",
        ))
        logger.warning("ESCALATION [Tier %d]: %s", tier, reason)
        return {"status": "escalated", "tier": tier, "reason": reason, "notification": result}

    # ── State persistence ──

    def _load_state(self) -> SystemState:
        """Load state from disk or create fresh."""
        if self.state_path.exists():
            try:
                data = json.loads(self.state_path.read_text())
                state = SystemState()
                state.active_tickets = data.get("active_tickets", [])
                state.open_prs = data.get("open_prs", [])
                state.merged_prs = data.get("merged_prs", [])
                state.rejected_prs = data.get("rejected_prs", [])
                state.consecutive_rejections = data.get("consecutive_rejections", 0)
                state.total_runs = data.get("total_runs", 0)
                state.last_run_at = data.get("last_run_at")
                state.circuit_breaker_active = data.get("circuit_breaker_active", False)
                state.skill_md_version = data.get("skill_md_version", 0)
                state.scout_history = data.get("scout_history", [])
                state.last_scout_at = data.get("last_scout_at")

                # Restore saved candidates so we skip re-scouting
                raw_candidates = data.get("candidates", [])
                if raw_candidates:
                    from tools.trac_client import TracTicket
                    restored = []
                    for c in raw_candidates:
                        ticket = TracTicket(
                            ticket_id=c["ticket_id"],
                            summary=c.get("summary", ""),
                            component=c.get("component", ""),
                            ticket_type=c.get("ticket_type", ""),
                            severity="", version="", owner="", reporter="",
                            status="", stage="", has_patch=False,
                            needs_better_patch=False, needs_tests=False,
                            needs_docs=False, easy_picking=True,
                        )
                        from agents.base import AgentResponse, TokenUsage
                        restored.append(TicketEvaluation(
                            ticket_id=c["ticket_id"],
                            ticket=ticket,
                            verdict=c.get("verdict", "PICK"),
                            score=c.get("score", 0),
                            reasoning=c.get("reasoning", ""),
                            risk_factors=c.get("risk_factors", []),
                            fix_approach=c.get("fix_approach"),
                            estimated_complexity=c.get("complexity", "simple"),
                            someone_working=False,
                            has_existing_pr=False,
                            clarity=c.get("clarity", "clear"),
                            component_depth=c.get("component_depth", "surface"),
                            raw_response=AgentResponse(raw_text="", usage=TokenUsage()),
                        ))
                    state.candidates = restored
                    logger.info("Restored %d candidates from state file — skipping scout", len(restored))

                return state
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Failed to load state: %s, starting fresh", e)
        return SystemState()

    def _save_state(self) -> None:
        """Persist state to disk."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "active_tickets": self.state.active_tickets,
            "open_prs": self.state.open_prs,
            "merged_prs": self.state.merged_prs,
            "rejected_prs": self.state.rejected_prs,
            "consecutive_rejections": self.state.consecutive_rejections,
            "total_runs": self.state.total_runs,
            "last_run_at": self.state.last_run_at,
            "circuit_breaker_active": self.state.circuit_breaker_active,
            "skill_md_version": self.state.skill_md_version,
            "last_scout_at": self.state.last_scout_at,
            "scout_history": self.state.scout_history,
            "candidates": [
                {
                    "ticket_id": e.ticket_id,
                    "summary": e.ticket.summary,
                    "component": e.ticket.component,
                    "ticket_type": e.ticket.ticket_type,
                    "verdict": e.verdict,
                    "score": e.score,
                    "reasoning": e.reasoning,
                    "risk_factors": e.risk_factors,
                    "fix_approach": e.fix_approach,
                    "complexity": e.estimated_complexity,
                    "clarity": e.clarity,
                    "component_depth": e.component_depth,
                }
                for e in self.state.candidates
            ],
        }
        self.state_path.write_text(json.dumps(data, indent=2))

    def run_continuous(self, interval_minutes: int = 60) -> None:
        """Run the orchestrator continuously on a schedule."""
        import signal
        running = True

        def handle_stop(signum, frame):
            nonlocal running
            logger.info("Stopping orchestrator...")
            running = False

        signal.signal(signal.SIGTERM, handle_stop)
        signal.signal(signal.SIGINT, handle_stop)

        logger.info("Starting orchestrator loop (every %d min). Ctrl+C to stop.", interval_minutes)

        while running:
            try:
                cycle_log = self.run_cycle()
                logger.info("Cycle complete: %s", json.dumps(cycle_log, indent=2, default=str))
            except Exception:
                logger.exception("Orchestrator cycle failed")

            if not running:
                break

            logger.info("Sleeping %d minutes...", interval_minutes)
            for _ in range(interval_minutes * 60):
                if not running:
                    break
                time.sleep(1)

        logger.info("Orchestrator stopped.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Orchestrator — Coordinate Django contribution agents")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--once", action="store_true", help="Run a single cycle")
    parser.add_argument("--watch", type=int, metavar="MINUTES", help="Run continuously every N minutes")
    parser.add_argument("--skill", type=str, default="skills/django-contributor/SKILL.md")
    parser.add_argument("--state", type=str, default="db/orchestrator_state.json")
    parser.add_argument("--repo", type=str, help="Path to Django repo clone (required for coding)")
    parser.add_argument("--fork", type=str, help="Your GitHub fork (e.g. 'yourusername/django', required for PRs)")
    parser.add_argument("--budget-cycle", type=float, default=2.0, help="Max USD per cycle (default: $2)")
    parser.add_argument("--budget-daily", type=float, default=10.0, help="Max USD per day (default: $10)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # Configure budget
    from agents.base import get_budget
    budget = get_budget()
    budget.max_cost_per_cycle = args.budget_cycle
    budget.max_cost_per_day = args.budget_daily

    skill_path = Path(args.skill) if Path(args.skill).exists() else None
    state_path = Path(args.state)
    repo_path = Path(args.repo) if args.repo else None

    orchestrator = Orchestrator(
        skill_path=skill_path, state_path=state_path,
        repo_path=repo_path, github_fork=args.fork,
    )

    if args.watch:
        orchestrator.run_continuous(interval_minutes=args.watch)
    elif args.once:
        result = orchestrator.run_cycle()
        print(json.dumps(result, indent=2, default=str))
    else:
        # Default: single cycle
        result = orchestrator.run_cycle()
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
