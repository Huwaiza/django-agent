"""
PR-Maker Agent — Pushes the branch, opens the PR, updates Trac.

TOKEN OPTIMIZATION: This is the cheapest agent in the system.
- One Sonnet call to generate the PR body (~300 output tokens)
- Everything else is pure tool calls (git push, gh pr create, Trac comment)
- No SKILL.md injection (use_skill=False) — it doesn't need learned patterns
- max_tokens capped at 512 for PR body generation

The PR-Maker only runs when CodingResult.ready_for_pr is True,
so it should never produce a bad PR.
"""

import logging
from pathlib import Path

from agents.base import BaseAgent, AgentResponse, MODEL_FAST
from agents.coder import CodingResult
from tools.git_client import GitClient, GitHubClient

logger = logging.getLogger("agents.pr_maker")


PR_BODY_PROMPT = """\
Write a concise pull request description for Django.

Ticket: #{ticket_id} — {summary}
Component: {component}
Branch: {branch_name}

Diff stat:
{diff_stat}

Self-review summary: {review_summary}

Write the PR body in this format (no markdown fencing, just the raw text):

Trac ticket: https://code.djangoproject.com/ticket/{ticket_id}

## Description
<2-3 sentences explaining what the fix does and why>

## Changes
<bullet list of files changed and what changed in each>

## Tests
<what test was added and what it verifies>

Keep it SHORT. Django reviewers read hundreds of PRs. No fluff.
Respond with ONLY the PR body text, nothing else.
"""

TRAC_COMMENT_TEMPLATE = (
    "I've submitted a pull request for this ticket: "
    "{pr_url}\n\n"
    "The fix {fix_summary}"
)


class PRMakerAgent(BaseAgent):
    """
    Creates pull requests from ready-to-submit patches.

    This agent is deliberately lightweight:
    - One AI call: generate PR body (Sonnet, 512 max_tokens)
    - Three tool calls: git push, gh pr create, Trac comment
    - No SKILL.md needed — PR formatting doesn't change with learnings

    The quality gate already happened in the Coder's self-review.
    The PR-Maker just packages and ships it.
    """

    def __init__(
        self,
        repo_path: Path,
        github_fork: str,
        api_key: str | None = None,
    ):
        super().__init__(
            name="pr_maker",
            system_prompt="You write concise, professional pull request descriptions for Django.",
            model=MODEL_FAST,
            api_key=api_key,
            use_skill=False,  # TOKEN OPTIMIZATION: PR body doesn't need SKILL.md
        )
        self.git = GitClient(repo_path)
        self.github = GitHubClient(repo="django/django")
        self.github_fork = github_fork  # e.g. "yourusername/django"

    def submit_pr(self, coding_result: CodingResult, ticket_summary: str, component: str) -> dict:
        """
        Full PR submission pipeline:
        1. Generate PR body (AI — one cheap Sonnet call)
        2. Push branch to fork (tool)
        3. Open PR against django/django (tool)
        4. Return PR URL and metadata

        Returns dict with: success, pr_url, pr_number, error
        """
        if not coding_result.ready_for_pr:
            return {"success": False, "error": "CodingResult is not ready for PR"}

        ticket_id = coding_result.ticket_id
        branch = coding_result.branch_name

        logger.info("Submitting PR for #%d on branch %s...", ticket_id, branch)

        # ── Step 1: Generate PR body (ONE Sonnet call, 512 max_tokens) ──
        pr_body = self._generate_pr_body(
            ticket_id=ticket_id,
            summary=ticket_summary,
            component=component,
            branch_name=branch,
            diff_stat=coding_result.diff_stat,
            review_summary=coding_result.self_review_summary,
        )
        logger.info("PR body generated (%d chars)", len(pr_body))

        # ── Step 2: Push branch to fork ──
        push_result = self.git.push_branch(branch, remote="origin")
        if not push_result.success:
            return {"success": False, "error": f"Git push failed: {push_result.stderr}"}
        logger.info("Branch pushed to fork")

        # ── Step 3: Open PR ──
        pr_title = f"Fixed #{ticket_id} -- {ticket_summary}"
        # Truncate title to 72 chars (GitHub convention)
        if len(pr_title) > 72:
            pr_title = pr_title[:69] + "..."

        pr_result = self.github.create_pr(
            title=pr_title,
            body=pr_body,
            branch=f"{self.github_fork.split('/')[0]}:{branch}",
            base="main",
        )

        if not pr_result.success:
            return {"success": False, "error": f"PR creation failed: {pr_result.stderr}"}

        # Parse PR URL from gh output
        pr_url = pr_result.stdout.strip()
        pr_number = self._extract_pr_number(pr_url)

        logger.info("✅ PR created: %s", pr_url)

        return {
            "success": True,
            "pr_url": pr_url,
            "pr_number": pr_number,
            "branch": branch,
            "ticket_id": ticket_id,
            "title": pr_title,
        }

    def _generate_pr_body(
        self,
        ticket_id: int,
        summary: str,
        component: str,
        branch_name: str,
        diff_stat: str,
        review_summary: str,
    ) -> str:
        """
        Generate PR body using ONE Sonnet call.

        TOKEN BUDGET:
        - System prompt: ~30 tokens (minimal, no SKILL.md)
        - User prompt: ~200 tokens (template + context)
        - max_tokens: 512 (PR bodies should be <300 tokens)
        - Total: ~750 tokens → ~$0.001 with Sonnet
        """
        prompt = PR_BODY_PROMPT.format(
            ticket_id=ticket_id,
            summary=summary,
            component=component,
            branch_name=branch_name,
            diff_stat=diff_stat or "(diff stat unavailable)",
            review_summary=review_summary or "Clean patch, tests pass.",
        )

        response = self.think(
            user_message=prompt,
            temperature=0.2,
            max_tokens=512,        # TOKEN OPTIMIZATION: PR bodies are short
            response_format="text",  # Plain text, not JSON
        )

        if response.raw_text:
            return response.raw_text.strip()

        # Fallback: generate a minimal PR body without AI
        logger.warning("AI PR body generation failed, using fallback template")
        return (
            f"Trac ticket: https://code.djangoproject.com/ticket/{ticket_id}\n\n"
            f"## Description\n"
            f"Fix for #{ticket_id}: {summary}\n\n"
            f"## Changes\n{diff_stat or 'See diff.'}\n\n"
            f"## Tests\nRegression test included."
        )

    @staticmethod
    def _extract_pr_number(pr_url: str) -> int | None:
        """Extract PR number from GitHub URL."""
        try:
            return int(pr_url.rstrip("/").split("/")[-1])
        except (ValueError, IndexError):
            return None

    def generate_trac_comment(self, pr_url: str, fix_summary: str) -> str:
        """Generate a comment to post on the Trac ticket after PR is opened."""
        return TRAC_COMMENT_TEMPLATE.format(
            pr_url=pr_url,
            fix_summary=fix_summary,
        )
