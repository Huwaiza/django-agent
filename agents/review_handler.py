"""
Review Handler Agent — Monitors PR comments, classifies, responds, or escalates.

Flow:
1. GitHubClient fetches new comments on open PRs (tool)
2. AI classifies each comment: style_fix, logic_question, requested_change,
   bug_found, architectural_concern, approval, rejection
3. For auto-handleable types → AI drafts a response + Claude Code fixes the code
4. For architectural concerns → escalates to human
5. Feeds all reviewer interactions to the Learner agent
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from agents.base import BaseAgent, AgentResponse, MODEL_FAST
from config.prompts import REVIEWER_HANDLER_SYSTEM_PROMPT
from tools.git_client import GitHubClient, GitClient
from tools.claude_code_client import ClaudeCodeClient

logger = logging.getLogger("agents.review_handler")


CLASSIFY_COMMENT_PROMPT = """\
Classify this PR review comment from a Django reviewer.

PR: #{ticket_id} — {summary}
Reviewer: {reviewer}
Comment:
{comment_text}

Respond with JSON:
{{
    "category": "style_fix" | "logic_question" | "requested_change" | "bug_found" | "architectural_concern" | "approval" | "rejection" | "informational",
    "can_auto_handle": <true if we can respond and fix without human>,
    "severity": "trivial" | "minor" | "major" | "blocker",
    "summary": "<1 sentence summary of what reviewer wants>",
    "suggested_action": "<what we should do>"
}}
"""

DRAFT_RESPONSE_PROMPT = """\
Draft a response to this Django PR review comment.

PR: #{ticket_id} — {summary}
Reviewer: {reviewer}
Their comment: {comment_text}
Category: {category}

Rules:
- Keep it SHORT (2-3 sentences max)
- Be respectful and grateful — these are volunteers
- If acknowledging a fix: "Thanks, fixed in the latest push."
- If explaining reasoning: be concise and reference the ticket
- NEVER argue more than once on the same point
- NEVER use AI-sounding phrases

Respond with JSON:
{{
    "response_text": "<your drafted response>",
    "needs_code_change": <true if we need to push a fix>,
    "code_change_description": "<what to change, or null>"
}}
"""

CODE_FIX_PROMPT = """\
A Django PR reviewer requested a change. Apply the fix.

Reviewer's request: {change_description}

File context: This is on branch {branch_name} for ticket #{ticket_id}.

Apply the fix, run tests for the affected module, and amend the commit.
"""

CI_FIX_PROMPT = """\
A Django PR CI check failed. Diagnose and fix it.

PR: #{ticket_id} — {summary}
Branch: {branch_name}
Failed check: {check_name}

CI failure log:
{failure_log}

Instructions:
- Read the exact error messages in the log above
- Find the relevant file(s) in the repo and fix them
- Do NOT change any Python logic — only fix the issue the CI is reporting
- After fixing, amend the commit (git commit --amend --no-edit)
- Do NOT push — the caller will push
"""


@dataclass
class ReviewComment:
    """A single review comment on a PR."""
    pr_number: int
    ticket_id: int
    summary: str
    reviewer: str
    comment_text: str
    comment_id: str = ""


@dataclass
class ReviewAction:
    """What the handler decided to do about a comment."""
    comment: ReviewComment
    category: str
    can_auto_handle: bool
    severity: str
    response_text: str | None
    needs_code_change: bool
    code_change_applied: bool = False
    escalated: bool = False
    raw_classification: dict | None = None


class ReviewHandlerAgent(BaseAgent):
    """
    Monitors open PRs, classifies reviewer comments, responds appropriately.

    Auto-handles:
    - style_fix: push fix + short acknowledgment
    - logic_question: explain reasoning
    - requested_change: apply change + acknowledge
    - bug_found: fix + thank reviewer
    - approval: log success
    - informational: no action needed

    Escalates:
    - architectural_concern: needs human judgment
    - rejection: needs human to decide next steps
    """

    def __init__(
        self,
        repo_path: Path | None = None,
        api_key: str | None = None,
    ):
        super().__init__(
            name="review_handler",
            system_prompt=REVIEWER_HANDLER_SYSTEM_PROMPT,
            model=MODEL_FAST,
            api_key=api_key,
            use_skill=False,  # Handler doesn't need SKILL.md — it reads reviewer comments directly
        )
        self.github = GitHubClient(repo="django/django")
        self.git = GitClient(repo_path) if repo_path else None
        self.claude_code = ClaudeCodeClient(
            working_dir=repo_path or Path.cwd(),
            allowed_tools=["Read", "Edit", "Write", "Bash"],
            max_turns=15,
        ) if repo_path else None

    def fix_ci_failures(self, pr_number: int, ticket_id: int, summary: str, branch_name: str) -> list[dict]:
        """
        Fetch CI check failures for a PR, fix each one via opencode, amend + push.

        Returns list of dicts: {check_name, fixed, error}
        """
        import subprocess

        if not self.git:
            return [{"check_name": "all", "fixed": False, "error": "No repo_path configured"}]

        # 1. Get failed checks
        result = subprocess.run(
            ["gh", "pr", "checks", str(pr_number), "--repo", "django/django",
             "--json", "name,state,link"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            return [{"check_name": "all", "fixed": False, "error": result.stderr.strip()}]

        import json as _json
        try:
            checks = _json.loads(result.stdout)
        except Exception:
            return [{"check_name": "all", "fixed": False, "error": "Could not parse gh output"}]

        failed = [c for c in checks if c.get("state", "").upper() in ("FAIL", "FAILURE", "ERROR")]
        if not failed:
            logger.info("No failed CI checks on PR #%d", pr_number)
            return []

        results = []
        for check in failed:
            check_name = check.get("name", "unknown")
            details_url = check.get("link", "")
            logger.info("Fetching failure log for check '%s'...", check_name)

            # Fetch the run log via gh
            log_result = subprocess.run(
                ["gh", "run", "view", "--log-failed", "--repo", "django/django",
                 details_url.split("/job/")[0].split("/runs/")[-1].split("/")[0]
                 if "/runs/" in details_url else ""],
                capture_output=True, text=True, timeout=30,
            )
            failure_log = log_result.stdout[:4000] if log_result.stdout else "(no log available)"

            if not failure_log.strip() or failure_log == "(no log available)":
                # Try fetching the run ID from the URL differently
                import re
                run_id_match = re.search(r"/runs/(\d+)", details_url)
                if run_id_match:
                    log_result2 = subprocess.run(
                        ["gh", "run", "view", run_id_match.group(1), "--log-failed",
                         "--repo", "django/django"],
                        capture_output=True, text=True, timeout=30,
                    )
                    failure_log = log_result2.stdout[:4000] if log_result2.stdout else failure_log

            logger.info("CI log (%d chars) fetched for '%s'", len(failure_log), check_name)

            # Checkout the branch
            self.git._git("checkout", branch_name)

            # Ask opencode to fix it
            fix_prompt = CI_FIX_PROMPT.format(
                ticket_id=ticket_id,
                summary=summary,
                branch_name=branch_name,
                check_name=check_name,
                failure_log=failure_log,
            )

            if self.claude_code:
                fix_result = self.claude_code.run(prompt=fix_prompt, timeout=300)
                fixed = fix_result.success
                error = fix_result.error if not fixed else None
            else:
                fixed = False
                error = "No claude_code client configured"

            if fixed:
                # Push the fix
                push = self.git._git("push", "origin", branch_name, "--force-with-lease", "--force-if-includes")
                if not push.success:
                    fixed = False
                    error = f"Push failed: {push.stderr}"
                else:
                    logger.info("Fix for '%s' pushed to branch %s", check_name, branch_name)

            results.append({"check_name": check_name, "fixed": fixed, "error": error})

        return results

    def check_and_handle_reviews(self, open_prs: list[dict]) -> list[ReviewAction]:
        """
        Check all open PRs for new comments and handle them.

        Args:
            open_prs: List of dicts with pr_number, ticket_id, summary

        Returns:
            List of ReviewAction for each comment processed
        """
        all_actions = []

        for pr in open_prs:
            pr_number = pr.get("pr_number")
            ticket_id = pr.get("ticket_id")
            summary = pr.get("summary", "")

            if not pr_number:
                continue

            logger.info("Checking PR #%d for new comments...", pr_number)

            # Fetch comments (tool)
            comments_result = self.github.get_pr_comments(pr_number)
            if not comments_result.success:
                logger.warning("Failed to fetch comments for PR #%d: %s", pr_number, comments_result.stderr)
                continue

            # Parse comments
            raw_comments = comments_result.stdout.strip().split("\n") if comments_result.stdout.strip() else []
            if not raw_comments:
                logger.debug("No comments on PR #%d", pr_number)
                continue

            for comment_text in raw_comments:
                if not comment_text.strip():
                    continue

                comment = ReviewComment(
                    pr_number=pr_number,
                    ticket_id=ticket_id,
                    summary=summary,
                    reviewer="(from GitHub)",
                    comment_text=comment_text.strip(),
                )

                action = self._handle_single_comment(comment)
                all_actions.append(action)

        return all_actions

    def _handle_single_comment(self, comment: ReviewComment) -> ReviewAction:
        """Classify and handle a single reviewer comment."""

        # Step 1: Classify the comment (AI)
        classify_prompt = CLASSIFY_COMMENT_PROMPT.format(
            ticket_id=comment.ticket_id,
            summary=comment.summary,
            reviewer=comment.reviewer,
            comment_text=comment.comment_text,
        )

        classification = self.think(
            user_message=classify_prompt,
            temperature=0.2,
            max_tokens=512,
        )

        if not classification.succeeded:
            logger.warning("Failed to classify comment on PR #%d", comment.pr_number)
            return ReviewAction(
                comment=comment, category="unknown", can_auto_handle=False,
                severity="unknown", response_text=None, needs_code_change=False,
                escalated=True,
            )

        data = classification.parsed
        category = data.get("category", "unknown")
        can_auto = data.get("can_auto_handle", False)
        severity = data.get("severity", "unknown")

        logger.info(
            "  PR #%d comment classified: %s (severity=%s, auto=%s)",
            comment.pr_number, category, severity, can_auto,
        )

        # Step 2: Decide what to do
        if category in ("architectural_concern", "rejection"):
            # Escalate — needs human judgment
            return ReviewAction(
                comment=comment, category=category, can_auto_handle=False,
                severity=severity, response_text=None, needs_code_change=False,
                escalated=True, raw_classification=data,
            )

        if category == "approval":
            return ReviewAction(
                comment=comment, category=category, can_auto_handle=True,
                severity="trivial", response_text=None, needs_code_change=False,
                raw_classification=data,
            )

        if category == "informational":
            return ReviewAction(
                comment=comment, category=category, can_auto_handle=True,
                severity="trivial", response_text=None, needs_code_change=False,
                raw_classification=data,
            )

        # Step 3: Draft a response (AI)
        draft_prompt = DRAFT_RESPONSE_PROMPT.format(
            ticket_id=comment.ticket_id,
            summary=comment.summary,
            reviewer=comment.reviewer,
            comment_text=comment.comment_text,
            category=category,
        )

        draft = self.think(user_message=draft_prompt, temperature=0.3, max_tokens=512)

        response_text = None
        needs_code = False
        code_desc = None

        if draft.succeeded:
            response_text = draft.parsed.get("response_text")
            needs_code = draft.parsed.get("needs_code_change", False)
            code_desc = draft.parsed.get("code_change_description")

        # Step 4: Apply code change if needed (Claude Code)
        code_applied = False
        if needs_code and code_desc and self.claude_code:
            fix_prompt = CODE_FIX_PROMPT.format(
                change_description=code_desc,
                branch_name=f"ticket_{comment.ticket_id}",
                ticket_id=comment.ticket_id,
            )
            fix_result = self.claude_code.run(prompt=fix_prompt, timeout=300)
            code_applied = fix_result.success
            if code_applied:
                logger.info("  Code fix applied for PR #%d", comment.pr_number)
            else:
                logger.warning("  Code fix failed for PR #%d: %s", comment.pr_number, fix_result.error)

        # Step 5: Post the response on GitHub
        if response_text:
            post_result = self.github.add_pr_comment(comment.pr_number, response_text)
            if post_result.success:
                logger.info("  Response posted on PR #%d", comment.pr_number)
            else:
                logger.warning("  Failed to post response on PR #%d", comment.pr_number)

        return ReviewAction(
            comment=comment, category=category, can_auto_handle=can_auto,
            severity=severity, response_text=response_text,
            needs_code_change=needs_code, code_change_applied=code_applied,
            raw_classification=data,
        )
