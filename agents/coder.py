"""
Coder Agent — Writes production-quality Django patches using Claude Code.

This agent:
1. Receives a picked ticket + fix approach from the Picker
2. Creates a git branch: ticket_XXXXX
3. Invokes Claude Code (`claude -p`) with full ticket context + SKILL.md
4. Claude Code reads Django source, writes the fix, writes tests, updates docs
5. Runs the targeted test suite to verify
6. SELF-REVIEW GATE: A separate Claude instance reviews the diff as a Django reviewer
7. If self-review passes → ready for PR. If not → Claude Code fixes the issues.
8. Returns a structured result: branch name, diff, test results, confidence

Architecture:
- CoderAgent (AI reasoning) decides WHAT to do and REVIEWS the result
- ClaudeCodeClient (tool) does the actual coding via `claude -p`
- GitClient (tool) manages branches and commits
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from agents.base import BaseAgent, AgentResponse, MODEL_DEEP, MODEL_FAST
from config.prompts import CODER_SYSTEM_PROMPT
from agents.scout import TicketEvaluation
from tools.claude_code_client import ClaudeCodeClient, ClaudeCodeResult
from tools.git_client import GitClient

logger = logging.getLogger("agents.coder")


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

CODING_TASK_PROMPT = """\
You are working on Django's codebase (https://github.com/django/django).
Your task is to fix the following Trac ticket.

{ticket_context}

SUGGESTED FIX APPROACH (from our analysis):
{fix_approach}

YOUR INSTRUCTIONS:
1. First, read the relevant source files for the component: {component}
   - Find the code related to the bug/issue described above
   - Read the full file(s) to understand the surrounding context

2. Write the MINIMAL fix — change only what's necessary, no unrelated refactoring

3. Write a regression test that:
   - FAILS without your fix (verify this mentally)
   - PASSES with your fix
   - Goes in the correct test file for this component
   - Uses Django's test conventions (self.assertIs for booleans, etc.)

4. If the fix changes user-facing behavior, update docs in docs/ (RST format)

5. Run the targeted test suite to make sure nothing breaks:
   python -m django test {test_module} --settings=tests.test_settings

6. Stage and commit with message: "Fixed #{ticket_id} -- {summary}"

IMPORTANT RULES:
- Follow PEP 8 + Django's coding style
- Use self.assertIs() not assertTrue() for boolean checks
- Use self.assertQuerySetEqual() for QuerySet comparisons
- Import ordering: stdlib → third-party → Django → local
- Keep the commit to a SINGLE logical change
- Do NOT modify any test infrastructure or unrelated files

{skill_context}
"""

SELF_REVIEW_PROMPT = """\
You are a senior Django core reviewer. Review this pull request diff and decide \
if it's ready to submit.

TICKET CONTEXT:
{ticket_context}

GIT DIFF:
```
{diff}
```

FILES CHANGED:
{diff_stat}

Evaluate this patch against Django's contribution standards:

1. CORRECTNESS: Does the fix actually solve the described problem?
2. MINIMAL SCOPE: Are there any unrelated changes? Anything that should be removed?
3. TEST QUALITY: Is there a regression test? Does it test the right thing?
4. STYLE: Does it follow Django's coding conventions?
5. DOCS: If behavior changed, are docs updated?
6. COMMIT MESSAGE: Does it follow "Fixed #XXXXX -- Description." format?
7. EDGE CASES: Are there obvious edge cases the fix misses?
8. BACKWARD COMPAT: Could this break existing code?

Respond with JSON:
{{
    "verdict": "APPROVE" | "REQUEST_CHANGES" | "REJECT",
    "score": <0-100>,
    "issues": [
        {{
            "severity": "blocker" | "major" | "minor" | "nit",
            "file": "<filename>",
            "description": "<what's wrong>",
            "suggestion": "<how to fix it>"
        }}
    ],
    "strengths": ["<what's good about this patch>"],
    "summary": "<1-2 sentence overall assessment>",
    "ready_to_submit": <true/false>
}}

Be strict. Django reviewers are thorough. Only APPROVE if you'd stake your \
reputation on this patch being accepted by a Django core dev.
"""

FIX_REVIEW_ISSUES_PROMPT = """\
Your patch was self-reviewed and the following issues were found.
Fix them now.

ISSUES TO FIX:
{issues}

Fix each issue, then run the tests again to make sure nothing broke.
Amend the commit with the fixes (git commit --amend).
"""


@dataclass
class CodingResult:
    """Result from a complete coding session."""
    ticket_id: int
    branch_name: str
    success: bool
    diff: str
    diff_stat: str
    test_passed: bool
    self_review_verdict: str  # APPROVE, REQUEST_CHANGES, REJECT
    self_review_score: int
    self_review_issues: list[dict]
    self_review_summary: str
    coding_cost_usd: float
    review_cost_usd: float
    fix_iterations: int  # how many times we iterated on review feedback
    error: str | None = None

    @property
    def ready_for_pr(self) -> bool:
        return self.success and self.self_review_verdict == "APPROVE" and self.test_passed

    @property
    def total_cost_usd(self) -> float:
        return self.coding_cost_usd + self.review_cost_usd


# ---------------------------------------------------------------------------
# Component → test module mapping
# ---------------------------------------------------------------------------

COMPONENT_TEST_MAP = {
    "Admin": "admin_views",
    "contrib.admin": "admin_views",
    "Database layer (models, ORM)": "model_fields",
    "Forms": "forms_tests",
    "Template system": "template_tests",
    "Generic views": "generic_views",
    "HTTP handling": "requests_tests",
    "Core (URLs, Middleware, etc.)": "urlpatterns",
    "Internationalization": "i18n",
    "Migrations": "migrations",
    "File uploads/storage": "file_storage",
    "contrib.auth": "auth_tests",
    "contrib.staticfiles": "staticfiles_tests",
    "Testing framework": "test_runner",
    "Management commands": "admin_scripts",
    "Serialization": "serializers",
    "Validators": "validators",
    "Documentation": "docs",
}


class CoderAgent(BaseAgent):
    """
    AI agent that writes Django patches using Claude Code.

    Architecture:
    - The CoderAgent (this class) handles REASONING: constructing the task prompt,
      reading SKILL.md, and running the self-review gate.
    - Claude Code (via ClaudeCodeClient) does the CODING: reading source files,
      writing the fix, writing tests, running tests.
    - Git (via GitClient) handles BRANCH MANAGEMENT: creating branches, committing,
      diffing.

    Flow:
        Ticket → create branch → Claude Code writes fix → run tests →
        self-review gate → (fix issues if needed) → ready for PR
    """

    MAX_FIX_ITERATIONS = 2  # Max times we'll try to fix self-review issues

    def __init__(
        self,
        repo_path: Path,
        skill_path: Path | None = None,
        api_key: str | None = None,
        max_coding_turns: int = 30,
    ):
        super().__init__(
            name="coder",
            system_prompt=CODER_SYSTEM_PROMPT,
            model=MODEL_DEEP,  # Self-review uses Opus for strict evaluation
            api_key=api_key,
            skill_path=skill_path,
        )
        self.repo_path = repo_path
        self.git = GitClient(repo_path)
        self.claude_code = ClaudeCodeClient(
            working_dir=repo_path,
            allowed_tools=["Read", "Edit", "Write", "Bash"],
            max_turns=max_coding_turns,
        )

    def code_fix(self, evaluation: TicketEvaluation) -> CodingResult:
        """
        Full coding pipeline for a single ticket.

        Steps:
        1. Create git branch
        2. Build task prompt with ticket context + SKILL.md
        3. Invoke Claude Code to write the fix
        4. Get diff and run self-review
        5. If self-review finds issues, invoke Claude Code again to fix them
        6. Return structured result
        """
        ticket = evaluation.ticket
        ticket_id = ticket.ticket_id
        branch_name = f"ticket_{ticket_id}"

        logger.info("═" * 60)
        logger.info("Starting coding session for #%d: %s", ticket_id, ticket.summary[:50])
        logger.info("Component: %s | Complexity: %s", ticket.component, evaluation.estimated_complexity)
        logger.info("═" * 60)

        total_coding_cost = 0.0
        total_review_cost = 0.0

        # ── Step 1: Create branch ──
        logger.info("Step 1: Creating branch '%s'...", branch_name)
        branch_result = self.git.create_branch(branch_name)
        if not branch_result.success:
            return self._error_result(ticket_id, branch_name, f"Failed to create branch: {branch_result.stderr}")

        # ── Step 2: Build the coding prompt ──
        test_module = COMPONENT_TEST_MAP.get(ticket.component, "tests")
        skill_context = ""
        if self.skill_path and self.skill_path.exists():
            skill_context = f"\nLEARNED PATTERNS FROM PAST REVIEWS:\n{self.skill_path.read_text()}"

        coding_prompt = CODING_TASK_PROMPT.format(
            ticket_context=ticket.to_context_string(),
            fix_approach=evaluation.fix_approach or "No specific approach suggested — analyze the code and determine the best fix.",
            component=ticket.component,
            test_module=test_module,
            ticket_id=ticket_id,
            summary=ticket.summary,
            skill_context=skill_context,
        )

        # ── Step 3: Invoke Claude Code ──
        logger.info("Step 2: Invoking Claude Code to write the fix...")
        coding_result = self.claude_code.run(
            prompt=coding_prompt,
            timeout=600,  # 10 min max for coding session
        )

        total_coding_cost += coding_result.cost_usd or 0
        logger.info("Claude Code session: %s", coding_result.summary)

        if not coding_result.success:
            self.git.stash_and_reset()
            return self._error_result(ticket_id, branch_name, f"Claude Code failed: {coding_result.error}")

        # ── Step 4: Get diff for review ──
        logger.info("Step 3: Getting diff for self-review...")
        diff_result = self.git.get_diff("main")
        diff_stat_result = self.git.get_diff_stat("main")

        if not diff_result.stdout:
            self.git.stash_and_reset()
            return self._error_result(ticket_id, branch_name, "No changes produced — Claude Code didn't write anything")

        # ── Step 5: Self-review gate ──
        logger.info("Step 4: Running self-review gate...")
        review_response = self._self_review(
            ticket_context=ticket.to_context_string(),
            diff=diff_result.stdout,
            diff_stat=diff_stat_result.stdout,
        )
        total_review_cost += review_response.tokens_used * 0.000015  # rough estimate

        if not review_response.succeeded:
            logger.warning("Self-review failed to parse — treating as REQUEST_CHANGES")
            review_data = {"verdict": "REQUEST_CHANGES", "score": 0, "issues": [], "summary": "Review parse failed", "ready_to_submit": False}
        else:
            review_data = review_response.parsed

        verdict = review_data.get("verdict", "REJECT")
        review_score = review_data.get("score", 0)
        issues = review_data.get("issues", [])

        logger.info("Self-review: %s (%d/100) — %s", verdict, review_score, review_data.get("summary", ""))

        # ── Step 6: Fix issues if needed (up to MAX_FIX_ITERATIONS) ──
        fix_iterations = 0
        while verdict == "REQUEST_CHANGES" and fix_iterations < self.MAX_FIX_ITERATIONS:
            fix_iterations += 1
            blockers = [i for i in issues if i.get("severity") in ("blocker", "major")]

            if not blockers:
                logger.info("Only minor/nit issues — acceptable, proceeding.")
                verdict = "APPROVE"
                break

            logger.info("Iteration %d: Fixing %d blocker/major issues...", fix_iterations, len(blockers))
            issues_text = "\n".join(
                f"- [{i['severity']}] {i.get('file', '?')}: {i['description']}\n  Suggestion: {i.get('suggestion', 'N/A')}"
                for i in blockers
            )

            fix_result = self.claude_code.run(
                prompt=FIX_REVIEW_ISSUES_PROMPT.format(issues=issues_text),
                timeout=300,
            )
            total_coding_cost += fix_result.cost_usd or 0

            if not fix_result.success:
                logger.warning("Fix iteration %d failed: %s", fix_iterations, fix_result.error)
                break

            # Re-get diff and re-review
            diff_result = self.git.get_diff("main")
            diff_stat_result = self.git.get_diff_stat("main")

            review_response = self._self_review(
                ticket_context=ticket.to_context_string(),
                diff=diff_result.stdout,
                diff_stat=diff_stat_result.stdout,
            )
            total_review_cost += review_response.tokens_used * 0.000015

            if review_response.succeeded:
                review_data = review_response.parsed
                verdict = review_data.get("verdict", "REJECT")
                review_score = review_data.get("score", 0)
                issues = review_data.get("issues", [])
                logger.info("Re-review after fixes: %s (%d/100)", verdict, review_score)

        # ── Step 7: Build result ──
        test_passed = "FAIL" not in coding_result.result_text.upper() if coding_result.result_text else False

        result = CodingResult(
            ticket_id=ticket_id,
            branch_name=branch_name,
            success=verdict in ("APPROVE", "REQUEST_CHANGES"),
            diff=diff_result.stdout,
            diff_stat=diff_stat_result.stdout,
            test_passed=test_passed,
            self_review_verdict=verdict,
            self_review_score=review_score,
            self_review_issues=issues,
            self_review_summary=review_data.get("summary", ""),
            coding_cost_usd=total_coding_cost,
            review_cost_usd=total_review_cost,
            fix_iterations=fix_iterations,
        )

        if result.ready_for_pr:
            logger.info("✅ Patch READY FOR PR (score: %d/100, cost: $%.4f)", review_score, result.total_cost_usd)
        else:
            logger.warning(
                "⚠️  Patch NOT ready: verdict=%s, score=%d, tests=%s",
                verdict, review_score, "passed" if test_passed else "FAILED",
            )
            # Don't leave a broken branch checked out
            if verdict == "REJECT":
                self.git.stash_and_reset()

        return result

    def _self_review(self, ticket_context: str, diff: str, diff_stat: str) -> AgentResponse:
        """
        Self-review gate: A separate AI pass that reviews the diff as a Django reviewer.

        This is the quality gate. It uses the CoderAgent's own Claude API connection
        (Opus) — NOT Claude Code — to do a pure reasoning review of the diff.
        """
        # Truncate very large diffs to avoid context limit
        max_diff_chars = 30000
        if len(diff) > max_diff_chars:
            diff = diff[:max_diff_chars] + f"\n\n... [truncated, {len(diff)} total chars]"

        prompt = SELF_REVIEW_PROMPT.format(
            ticket_context=ticket_context,
            diff=diff,
            diff_stat=diff_stat,
        )

        return self.think(
            user_message=prompt,
            temperature=0.2,
            max_tokens=2048,
        )

    @staticmethod
    def _error_result(ticket_id: int, branch_name: str, error: str) -> CodingResult:
        logger.error("Coding failed for #%d: %s", ticket_id, error)
        return CodingResult(
            ticket_id=ticket_id,
            branch_name=branch_name,
            success=False,
            diff="",
            diff_stat="",
            test_passed=False,
            self_review_verdict="REJECT",
            self_review_score=0,
            self_review_issues=[],
            self_review_summary=error,
            coding_cost_usd=0,
            review_cost_usd=0,
            fix_iterations=0,
            error=error,
        )


# ---------------------------------------------------------------------------
# CLI for manual testing
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Coder Agent — Write Django patches with AI")
    parser.add_argument("--repo", type=str, required=True, help="Path to Django repo clone")
    parser.add_argument("--ticket-json", type=str, help="Path to TicketEvaluation JSON (from Scout)")
    parser.add_argument("--skill", type=str, default="skills/django-contributor/SKILL.md")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    skill_path = Path(args.skill) if Path(args.skill).exists() else None
    repo_path = Path(args.repo)

    if not repo_path.exists():
        logger.error("Repo path does not exist: %s", repo_path)
        return

    coder = CoderAgent(repo_path=repo_path, skill_path=skill_path)

    if args.ticket_json:
        # Load evaluation from JSON file (produced by Scout/Picker)
        with open(args.ticket_json) as f:
            data = json.load(f)
        logger.info("Loaded ticket evaluation from %s", args.ticket_json)
        # Would need to reconstruct TicketEvaluation from JSON — for now, log and exit
        logger.info("Ticket: #%s — %s", data.get("ticket_id"), data.get("summary"))
        logger.info("Run via orchestrator for full pipeline integration.")
    else:
        logger.info("Use --ticket-json to provide a ticket evaluation, or run via the orchestrator.")
        logger.info("Example: python -m agents.coder --repo /path/to/django --ticket-json evaluation.json")


if __name__ == "__main__":
    main()
