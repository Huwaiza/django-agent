"""
Learner Agent — Extracts lessons from PR reviews and upgrades SKILL.md.

After every PR review cycle (merge, rejection, or comment thread), this agent:
1. Reads the reviewer's comments
2. Uses Claude to extract structured, reusable lessons
3. Appends them to SKILL.md so all other agents get smarter
4. Deduplicates and consolidates lessons periodically

This is the COMPOUND ADVANTAGE — every PR makes the next one better.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from agents.base import BaseAgent, MODEL_FAST
from config.prompts import LEARNER_SYSTEM_PROMPT

logger = logging.getLogger("agents.learner")


EXTRACT_LESSONS_PROMPT = """\
A pull request to Django was just {outcome}. Analyze the review thread and extract \
reusable lessons for our contribution system.

PR Context:
- Ticket: #{ticket_id} — {summary}
- Component: {component}
- Outcome: {outcome}

Review thread:
{review_thread}

Extract lessons as a JSON object:
{{
    "lessons": [
        {{
            "category": "coding_pattern" | "test_pattern" | "reviewer_preference" | \
"component_knowledge" | "anti_pattern" | "process_pattern",
            "description": "<concise, actionable instruction — something a code-writing agent can follow>",
            "component": "<Django component this applies to, or 'all'>",
            "confidence": "high" | "medium" | "low",
            "source_pr": "<PR number>",
            "source_reviewer": "<reviewer name if identifiable>"
        }}
    ],
    "meta_insights": "<any higher-level observation about what happened, 1-2 sentences>"
}}

Rules:
- Write lessons as INSTRUCTIONS, not observations. "Use self.assertIs() for booleans" \
not "The reviewer prefers assertIs".
- Include the reviewer's name when possible — it helps calibrate future interactions.
- If the PR was rejected, focus on WHAT WENT WRONG and HOW TO AVOID IT NEXT TIME.
- If the PR was merged, focus on WHAT WE DID RIGHT that we should keep doing.
- Be specific to Django. "Write good tests" is useless. "Place regression tests in \
tests/admin_views/test_*.py for admin component bugs" is useful.
"""

CONSOLIDATE_PROMPT = """\
Here is the current content of our SKILL.md file that contains lessons learned from \
Django PR reviews:

{current_skill}

Consolidate this file:
1. Merge duplicate or near-duplicate lessons into single, stronger entries
2. Remove contradictions (keep the more recent lesson)
3. Group by component and category
4. Keep the file under 400 lines
5. Preserve the source/PR references

Return the consolidated SKILL.md content as plain text (not JSON, not code-fenced).
Start with the "# Django Contribution Skill" header.
"""


class LearnerAgent(BaseAgent):
    """
    AI agent that learns from PR review feedback and maintains SKILL.md.

    The learning loop:
    1. Reviewer comments on our PR → extracted by Review Handler
    2. Learner reads the comments and extracts structured lessons
    3. Lessons are appended to SKILL.md
    4. All other agents reload SKILL.md and benefit from the new knowledge
    5. Next PR is better because of it

    This is what makes the system compound over time.
    """

    def __init__(self, skill_path: Path, api_key: str | None = None):
        super().__init__(
            name="learner",
            system_prompt=LEARNER_SYSTEM_PROMPT,
            model=MODEL_FAST,
            api_key=api_key,
            skill_path=skill_path,
        )
        self.skill_path = skill_path

    def extract_lessons(
        self,
        ticket_id: int,
        summary: str,
        component: str,
        outcome: str,  # "merged", "rejected", "changes_requested"
        review_thread: str,
        pr_number: int | None = None,
    ) -> list[dict]:
        """
        Extract structured lessons from a PR review thread.

        Returns list of lesson dicts and appends them to SKILL.md.
        """
        prompt = EXTRACT_LESSONS_PROMPT.format(
            ticket_id=ticket_id,
            summary=summary,
            component=component,
            outcome=outcome,
            review_thread=review_thread,
        )

        response = self.think(user_message=prompt, temperature=0.3)

        if not response.succeeded:
            logger.warning("Lesson extraction failed for #%d", ticket_id)
            return []

        lessons = response.parsed.get("lessons", [])
        meta = response.parsed.get("meta_insights", "")

        if lessons:
            self._append_to_skill(lessons, ticket_id, outcome, meta)

            # Store reviewer patterns in long-term memory
            for lesson in lessons:
                if lesson.get("category") == "reviewer_preference" and lesson.get("source_reviewer"):
                    self.memory.long.record_reviewer_pattern(
                        reviewer=lesson["source_reviewer"],
                        pattern_type=lesson.get("category", "preference"),
                        description=lesson["description"],
                        confidence=lesson.get("confidence", "medium"),
                        source_pr=pr_number,
                    )

            logger.info(
                "Extracted %d lessons from #%d (%s): %s",
                len(lessons), ticket_id, outcome, meta,
            )

        return lessons

    def _append_to_skill(
        self,
        lessons: list[dict],
        ticket_id: int,
        outcome: str,
        meta: str,
    ) -> None:
        """Append new lessons to the SKILL.md file."""
        self.skill_path.parent.mkdir(parents=True, exist_ok=True)

        # Create SKILL.md if it doesn't exist
        if not self.skill_path.exists():
            self.skill_path.write_text(
                "# Django Contribution Skill\n\n"
                "Lessons learned from PR reviews. This file is automatically "
                "maintained by the Learner agent.\n\n"
                "---\n\n"
            )

        # Format and append
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        block = f"\n## Lesson — Ticket #{ticket_id} ({outcome}) [{timestamp}]\n\n"

        if meta:
            block += f"*{meta}*\n\n"

        for lesson in lessons:
            confidence = lesson.get("confidence", "medium")
            component = lesson.get("component", "all")
            category = lesson.get("category", "general")
            source = lesson.get("source_reviewer", "unknown reviewer")

            block += (
                f"- **[{category}]** [{component}] {lesson['description']} "
                f"(confidence: {confidence}, via {source})\n"
            )

        block += "\n"

        with open(self.skill_path, "a") as f:
            f.write(block)

        logger.info("Appended %d lessons to %s", len(lessons), self.skill_path)

    def consolidate_skill(self) -> None:
        """
        Periodically consolidate SKILL.md to remove duplicates and keep it manageable.

        Uses Claude to intelligently merge lessons rather than doing dumb deduplication.
        """
        if not self.skill_path.exists():
            return

        current = self.skill_path.read_text()
        line_count = len(current.splitlines())

        if line_count < 200:
            logger.info("SKILL.md is %d lines, no consolidation needed", line_count)
            return

        logger.info("Consolidating SKILL.md (%d lines)...", line_count)

        prompt = CONSOLIDATE_PROMPT.format(current_skill=current)
        response = self.think(
            user_message=prompt,
            response_format="text",
            temperature=0.2,
            max_tokens=8192,
        )

        if response.raw_text and len(response.raw_text) > 100:
            # Backup current version
            backup_path = self.skill_path.with_suffix(f".md.bak.{datetime.now():%Y%m%d%H%M}")
            backup_path.write_text(current)

            # Write consolidated version
            self.skill_path.write_text(response.raw_text)
            new_lines = len(response.raw_text.splitlines())
            logger.info("Consolidated SKILL.md: %d → %d lines (backup: %s)", line_count, new_lines, backup_path)
        else:
            logger.warning("Consolidation produced empty result, keeping original")
