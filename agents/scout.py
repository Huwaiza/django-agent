"""
Scout Agent (AI-Powered) — Uses Claude to discover and evaluate Django Trac tickets.

Unlike a scraper with hardcoded heuristics, this agent:
- READS each ticket's description and full comment thread
- REASONS about whether it's a good contribution target
- EXPLAINS its assessment (useful for auditing and learning)
- IMPROVES over time via SKILL.md context from the Learner agent

The Python code here handles Trac API access and orchestration.
The INTELLIGENCE lives in the Claude API call that evaluates each ticket.
"""

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base import BaseAgent, AgentResponse, MODEL_FAST, MODEL_DEEP
from config.prompts import SCOUT_SYSTEM_PROMPT
from tools.trac_client import TracClient, TracTicket

logger = logging.getLogger("agents.scout")

# The JSON schema the Scout agent must return for each ticket evaluation.
# This is the CONTRACT between the Scout and the Picker.
EVALUATION_REQUEST_TEMPLATE = """\
Evaluate this Django Trac ticket as a potential contribution target.

{ticket_context}

Analyze this ticket and respond with a JSON object containing:
{{
    "verdict": "PICK" | "SKIP" | "MAYBE",
    "score": <0-100 integer>,
    "reasoning": "<2-3 sentences explaining your assessment>",
    "risk_factors": ["<list of concerns>"],
    "fix_approach_sketch": "<1-2 sentences on how you'd approach the fix, or null if SKIP>",
    "estimated_complexity": "trivial" | "simple" | "moderate" | "complex",
    "someone_actively_working": <true/false>,
    "has_existing_pr": <true/false>,
    "clarity_of_problem": "clear" | "somewhat_clear" | "unclear",
    "component_depth": "surface" | "moderate" | "deep_internals"
}}

Be strict. Only give "PICK" verdict to tickets where:
- The problem is clearly defined
- Nobody is actively working on it (check comment thread carefully)
- The fix scope is manageable for a single PR
- You can sketch a reasonable approach

Give "SKIP" for tickets that are:
- Triage Stage is "Unreviewed" — the Django core team has not accepted it yet,
  working on it risks wasted effort if it gets closed as invalid
- Actively being worked on by someone
- Too vague or controversial
- Require deep framework internals knowledge
- Have an existing PR linked in comments

Give "MAYBE" for tickets that look promising but have some uncertainty.
"""

BATCH_TRIAGE_TEMPLATE = """\
You are triaging a batch of Django Trac tickets. For each ticket, provide a QUICK assessment \
(don't overthink — we'll do deep evaluation on the shortlisted ones).

Here are the tickets (summary only):

{ticket_summaries}

For each ticket, respond with a JSON array where each element is:
{{
    "ticket_id": <int>,
    "quick_verdict": "PROMISING" | "SKIP" | "NEEDS_DETAIL",
    "one_line_reason": "<why>"
}}

Skip tickets where:
- The type is "New feature" (usually too large for easy pickings)
- The summary suggests deep ORM/migration internals
- There is clear recent activity (comments in the last few weeks) showing someone is mid-PR

Mark as PROMISING:
- Bug fixes with clear, specific summaries
- Cleanup/optimization tasks
- Documentation improvements
- Small behavioral fixes

Mark as NEEDS_DETAIL for anything ambiguous — we'll fetch the full ticket to decide.
"""


@dataclass
class TicketEvaluation:
    """AI evaluation of a single ticket."""
    ticket_id: int
    ticket: TracTicket
    verdict: str  # PICK, SKIP, MAYBE
    score: int
    reasoning: str
    risk_factors: list[str]
    fix_approach: str | None
    estimated_complexity: str
    someone_working: bool
    has_existing_pr: bool
    clarity: str
    component_depth: str
    raw_response: AgentResponse

    @property
    def is_candidate(self) -> bool:
        return self.verdict in ("PICK", "MAYBE") and self.score >= 40


class ScoutAgent(BaseAgent):
    """
    AI-powered Scout that discovers and evaluates Django contribution opportunities.

    Architecture:
    1. TracClient (tool) fetches raw ticket data from Trac
    2. ScoutAgent (AI) reads and reasons about each ticket
    3. Returns structured evaluations the Picker agent can act on

    Two-phase evaluation for efficiency:
    - Phase 1: BATCH TRIAGE — AI does quick pass on summaries (cheap, fast)
    - Phase 2: DEEP EVAL — AI reads full ticket + comments for shortlisted ones (thorough)
    """

    def __init__(self, skill_path: Path | None = None, api_key: str | None = None):
        super().__init__(
            name="scout",
            system_prompt=SCOUT_SYSTEM_PROMPT,
            model=MODEL_FAST,  # Sonnet for triage, switch to Opus for deep eval if needed
            api_key=api_key,
            skill_path=skill_path,
        )
        self.trac = TracClient(rate_limit_seconds=1.0)

    def discover(self, deep_eval_limit: int = 20, stop_on_first_pick: bool = False) -> list[TicketEvaluation]:
        """
        Full discovery pipeline:
        1. Fetch all easy-picking tickets from Trac (tool)
        2. AI batch-triages the summaries (fast, cheap)
        3. Fetch full details for promising tickets (tool)
        4. AI deep-evaluates each promising ticket (thorough)

        Args:
            deep_eval_limit: Max tickets to deep-evaluate
            stop_on_first_pick: Stop as soon as one PICK is found

        Returns:
            List of TicketEvaluation objects, sorted by score descending
        """
        # Phase 0: Fetch ticket list from Trac
        logger.info("Phase 0: Fetching easy-picking tickets from Trac...")
        tickets = self.trac.fetch_easy_pickings()
        if not tickets:
            logger.warning("No tickets found on Trac")
            return []

        logger.info("Found %d raw tickets", len(tickets))

        # Phase 1: AI batch triage (cheap — just summaries)
        logger.info("Phase 1: AI batch triage on %d ticket summaries...", len(tickets))
        shortlist = self._batch_triage(tickets)
        logger.info("Batch triage shortlisted %d tickets for deep evaluation", len(shortlist))

        # Phase 2: Fetch full details + AI deep evaluation
        shortlist = shortlist[:deep_eval_limit]
        logger.info("Phase 2: Deep-evaluating %d tickets (fetching details)...", len(shortlist))

        evaluations = []
        for ticket in shortlist:
            try:
                # Tool: fetch full ticket with comments
                hydrated = self.trac.fetch_ticket_detail(ticket)

                # AI: deep evaluation
                evaluation = self._deep_evaluate(hydrated)
                evaluations.append(evaluation)

                logger.info(
                    "  #%d [%s, %d/100] %s",
                    evaluation.ticket_id,
                    evaluation.verdict,
                    evaluation.score,
                    ticket.summary[:50],
                )

                if stop_on_first_pick and evaluation.verdict == "PICK":
                    logger.info("Found a PICK — stopping early (--stop-on-pick mode)")
                    break

            except Exception as e:
                logger.error("Failed to evaluate #%d: %s", ticket.ticket_id, e)

        # Sort by score
        evaluations.sort(key=lambda e: e.score, reverse=True)
        return evaluations

    # Django Trac triage stages that are safe to work on.
    # "Unreviewed" means the ticket has not been accepted by the core team yet —
    # working on it risks wasted effort if the ticket is later closed as invalid.
    WORKABLE_STAGES = {
        "Accepted",
        "Ready for checkin",
        "Someday/Maybe",
    }

    def _batch_triage(self, tickets: list[TracTicket], chunk_size: int = 20) -> list[TracTicket]:
        """
        Phase 1: Fast heuristic triage — no AI, no timeouts.

        Hard-filters tickets whose Triage Stage is "Unreviewed" — these haven't
        been accepted by the Django core team and should not be claimed.
        Everything else passes through to Phase 2 deep eval.
        """
        before = len(tickets)
        tickets = [t for t in tickets if t.stage.strip() not in ("Unreviewed", "")]
        dropped = before - len(tickets)
        if dropped:
            logger.info(
                "Heuristic triage: dropped %d Unreviewed tickets, %d remain for deep eval",
                dropped, len(tickets),
            )
        else:
            logger.info("Heuristic triage: all %d tickets already past Unreviewed stage", len(tickets))
        return tickets

    def _deep_evaluate(self, ticket: TracTicket) -> TicketEvaluation:
        """
        Phase 2: AI thoroughly evaluates a single ticket.

        Reads the full description, all comments, and reasons about:
        - Is anyone working on this?
        - Is the problem clearly defined?
        - What's the fix approach?
        - What are the risks?
        """
        # Hard gate — never evaluate Unreviewed tickets regardless of how we got here.
        # fetch_ticket_detail() populates the stage from the Trac page, so this is
        # always authoritative even when called via --ticket or evaluate_single().
        if ticket.stage.strip() == "Unreviewed":
            logger.info(
                "  #%d SKIP — Triage Stage is Unreviewed (not accepted by core team)",
                ticket.ticket_id,
            )
            return TicketEvaluation(
                ticket_id=ticket.ticket_id,
                ticket=ticket,
                verdict="SKIP",
                score=0,
                reasoning="Triage Stage is Unreviewed — ticket has not been accepted by the Django core team.",
                risk_factors=["unreviewed"],
                fix_approach=None,
                estimated_complexity="unknown",
                someone_working=False,
                has_existing_pr=False,
                clarity="unclear",
                component_depth="unknown",
                raw_response=AgentResponse(raw_text=""),
            )

        prompt = EVALUATION_REQUEST_TEMPLATE.format(
            ticket_context=ticket.to_context_string()
        )

        response = self.think(
            user_message=prompt,
            temperature=0.2,
            max_tokens=768,
            timeout=180,
        )

        if not response.succeeded:
            # Conservative fallback — if we can't evaluate, skip
            return TicketEvaluation(
                ticket_id=ticket.ticket_id,
                ticket=ticket,
                verdict="SKIP",
                score=0,
                reasoning=f"Evaluation failed: could not parse AI response",
                risk_factors=["evaluation_failure"],
                fix_approach=None,
                estimated_complexity="unknown",
                someone_working=False,
                has_existing_pr=False,
                clarity="unclear",
                component_depth="unknown",
                raw_response=response,
            )

        data = response.parsed
        return TicketEvaluation(
            ticket_id=ticket.ticket_id,
            ticket=ticket,
            verdict=data.get("verdict", "SKIP"),
            score=data.get("score", 0),
            reasoning=data.get("reasoning", ""),
            risk_factors=data.get("risk_factors", []),
            fix_approach=data.get("fix_approach_sketch"),
            estimated_complexity=data.get("estimated_complexity", "unknown"),
            someone_working=data.get("someone_actively_working", False),
            has_existing_pr=data.get("has_existing_pr", False),
            clarity=data.get("clarity_of_problem", "unclear"),
            component_depth=data.get("component_depth", "unknown"),
            raw_response=response,
        )

    def evaluate_single(self, ticket_id: int) -> TicketEvaluation:
        """Evaluate a specific ticket by ID (useful for manual testing)."""
        # Create a minimal ticket to fetch details
        ticket = TracTicket(
            ticket_id=ticket_id,
            summary="", component="", ticket_type="", severity="",
            version="", owner="", reporter="", status="", stage="",
            has_patch=False, needs_better_patch=False, needs_tests=False,
            needs_docs=False, easy_picking=True,
        )
        hydrated = self.trac.fetch_ticket_detail(ticket)
        return self._deep_evaluate(hydrated)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="AI Scout Agent — Discover Django contribution opportunities")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-n", "--limit", type=int, default=10, help="Max tickets to deep-evaluate")
    parser.add_argument("--ticket", type=int, help="Evaluate a specific ticket ID")
    parser.add_argument("--skill", type=str, default="skills/django-contributor/SKILL.md")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    skill_path = Path(args.skill)
    if not skill_path.exists():
        skill_path = None

    scout = ScoutAgent(skill_path=skill_path)

    if args.ticket:
        logger.info("Evaluating single ticket #%d...", args.ticket)
        evaluation = scout.evaluate_single(args.ticket)
        print(f"\nTicket #{evaluation.ticket_id}: {evaluation.ticket.summary}")
        print(f"Verdict: {evaluation.verdict} ({evaluation.score}/100)")
        print(f"Reasoning: {evaluation.reasoning}")
        print(f"Complexity: {evaluation.estimated_complexity}")
        print(f"Fix approach: {evaluation.fix_approach}")
        print(f"Risks: {', '.join(evaluation.risk_factors) or 'None identified'}")
        return

    evaluations = scout.discover(deep_eval_limit=args.limit)

    print(f"\n{'='*80}")
    print(f" AI SCOUT RESULTS — {len(evaluations)} tickets evaluated")
    print(f"{'='*80}\n")

    picks = [e for e in evaluations if e.verdict == "PICK"]
    maybes = [e for e in evaluations if e.verdict == "MAYBE"]
    skips = [e for e in evaluations if e.verdict == "SKIP"]

    if picks:
        print(f"✅ PICK ({len(picks)} tickets ready to work on):\n")
        for e in picks:
            print(f"  #{e.ticket_id} [{e.score}/100] {e.ticket.summary[:60]}")
            print(f"    Reasoning: {e.reasoning}")
            print(f"    Approach: {e.fix_approach}")
            print(f"    Complexity: {e.estimated_complexity} | Clarity: {e.clarity}")
            print()

    if maybes:
        print(f"🟡 MAYBE ({len(maybes)} tickets worth considering):\n")
        for e in maybes:
            print(f"  #{e.ticket_id} [{e.score}/100] {e.ticket.summary[:60]}")
            print(f"    Reasoning: {e.reasoning}")
            if e.risk_factors:
                print(f"    Risks: {', '.join(e.risk_factors)}")
            print()

    print(f"⏭️  Skipped: {len(skips)} tickets")
    print(f"\nTotal API tokens used: ~{sum(e.raw_response.tokens_used for e in evaluations):,}")


if __name__ == "__main__":
    main()
