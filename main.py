#!/usr/bin/env python3
"""
Django Contribution Agent System — Main entry point.

Usage:
    # Run a single orchestration cycle
    python main.py --once

    # Run continuously every 60 minutes
    python main.py --watch 60

    # Run just the Scout to see candidates
    python main.py --scout-only

    # Evaluate a specific Trac ticket
    python main.py --evaluate 35421

Environment variables:
    ANTHROPIC_API_KEY    Your Anthropic API key (required)
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(
        description="Django Contribution Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--once", action="store_true", help="Run single orchestration cycle")
    group.add_argument("--watch", type=int, metavar="MIN", help="Run continuously every N minutes")
    group.add_argument("--scout-only", action="store_true", help="Just run the Scout agent")
    group.add_argument("--evaluate", type=int, metavar="TICKET_ID", help="Evaluate a specific ticket")
    group.add_argument("--fix-pr", type=int, metavar="PR_NUMBER", help="Fix CI failures on an open PR and push")

    parser.add_argument("--skill", default="skills/django-contributor/SKILL.md", help="Path to SKILL.md")
    parser.add_argument("--stop-on-pick", action="store_true", help="Stop scouting as soon as one PICK is found")
    parser.add_argument("--keep-picking", type=int, metavar="N", default=20, help="Max tickets to deep-evaluate (default: 20)")
    parser.add_argument("--state", default="db/orchestrator_state.json", help="Path to state file")
    parser.add_argument("--ticket", type=int, metavar="TICKET_ID", help="Code a specific ticket ID directly (skips scout + picker)")
    parser.add_argument("--repo", type=str, help="Path to Django repo clone (required for coding + PRs)")
    parser.add_argument("--fork", type=str, help="Your GitHub fork e.g. 'yourusername/django' (required for PRs)")
    parser.add_argument("--budget-cycle", type=float, default=2.0, help="Max USD spend per cycle (default: $2)")
    parser.add_argument("--budget-daily", type=float, default=10.0, help="Max USD spend per day (default: $10)")
    args = parser.parse_args()

    # Logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("main")

    # Configure global token budget
    from agents.base import get_budget
    budget = get_budget()
    budget.max_cost_per_cycle = args.budget_cycle
    budget.max_cost_per_day = args.budget_daily

    skill_path = Path(args.skill) if Path(args.skill).exists() else None

    if args.fix_pr:
        from agents.review_handler import ReviewHandlerAgent
        from tools.trac_client import TracClient, TracTicket

        if not args.repo:
            logger.error("--repo is required for --fix-pr")
            sys.exit(1)

        repo_path = Path(args.repo)

        # Fetch PR info to get ticket_id and branch
        import subprocess as _sp
        pr_info = _sp.run(
            ["gh", "pr", "view", str(args.fix_pr), "--repo", "django/django",
             "--json", "title,headRefName,number"],
            capture_output=True, text=True,
        )
        if pr_info.returncode != 0:
            logger.error("Could not fetch PR #%d: %s", args.fix_pr, pr_info.stderr)
            sys.exit(1)

        import json as _json
        pr_data = _json.loads(pr_info.stdout)
        branch_name = pr_data["headRefName"]
        pr_title = pr_data["title"]

        # Extract ticket_id from branch name (ticket_XXXXX) or PR title
        import re as _re
        m = _re.search(r"(\d{4,6})", branch_name) or _re.search(r"#(\d{4,6})", pr_title)
        ticket_id = int(m.group(1)) if m else 0

        # Get summary from PR title
        summary = _re.sub(r"^Fixed #\d+ -- ", "", pr_title).rstrip(".")

        logger.info("Fixing CI failures on PR #%d (branch: %s, ticket: #%d)",
                    args.fix_pr, branch_name, ticket_id)

        handler = ReviewHandlerAgent(repo_path=repo_path)
        results = handler.fix_ci_failures(
            pr_number=args.fix_pr,
            ticket_id=ticket_id,
            summary=summary,
            branch_name=branch_name,
        )

        print(json.dumps(results, indent=2, default=str))

    elif args.scout_only:
        from agents.scout import ScoutAgent
        scout = ScoutAgent(skill_path=skill_path)
        evaluations = scout.discover(
            deep_eval_limit=args.keep_picking,
            stop_on_first_pick=args.stop_on_pick,
        )

        picks = [e for e in evaluations if e.verdict == "PICK"]
        maybes = [e for e in evaluations if e.verdict == "MAYBE"]

        print(f"\n{'='*60}")
        print(f" Scout Results: {len(picks)} PICKs, {len(maybes)} MAYBEs")
        print(f"{'='*60}\n")

        for e in evaluations:
            if e.is_candidate:
                print(f"  [{e.verdict}] #{e.ticket_id} ({e.score}/100) {e.ticket.summary[:50]}")
                print(f"    → {e.reasoning}")
                print()

        # Merge candidates into state — never overwrite existing ones
        import json as _json
        from datetime import datetime as _dt
        state_path = Path(args.state)
        state_path.parent.mkdir(exist_ok=True)
        try:
            state = _json.loads(state_path.read_text()) if state_path.exists() else {}
        except Exception:
            state = {}

        # IDs already coded or submitted — don't resurface them as candidates
        acted_ids = {
            t.get("ticket_id") for t in state.get("active_tickets", [])
        } | {
            p.get("ticket_id") for p in state.get("open_prs", [])
        } | {
            p.get("ticket_id") for p in state.get("merged_prs", [])
        } | {
            p.get("ticket_id") for p in state.get("rejected_prs", [])
        }

        new_candidates = [
            {"ticket_id": e.ticket_id, "summary": e.ticket.summary,
             "verdict": e.verdict, "score": e.score, "reasoning": e.reasoning,
             "fix_approach": e.fix_approach, "complexity": e.estimated_complexity,
             "found_at": _dt.now().isoformat()}
            for e in evaluations if e.is_candidate and e.ticket_id not in acted_ids
        ]

        # Merge: keep existing candidates not seen in this run, add/update new ones
        existing_by_id = {c["ticket_id"]: c for c in state.get("candidates", [])}
        for c in new_candidates:
            existing_by_id[c["ticket_id"]] = c  # new scan result overwrites stale entry
        merged_candidates = list(existing_by_id.values())

        # Append a record to scout_history so every run is preserved
        history_entry = {
            "scanned_at": _dt.now().isoformat(),
            "evaluated": len(evaluations),
            "new_candidates": [c["ticket_id"] for c in new_candidates],
            "results": [
                {"ticket_id": e.ticket_id, "verdict": e.verdict,
                 "score": e.score, "summary": e.ticket.summary[:60]}
                for e in evaluations
            ],
        }
        state.setdefault("scout_history", []).append(history_entry)

        state["candidates"] = merged_candidates
        state["last_scout_at"] = _dt.now().isoformat()
        state_path.write_text(_json.dumps(state, indent=2))
        logger.info(
            "Merged %d new candidates (total pool: %d) into %s",
            len(new_candidates), len(merged_candidates), state_path,
        )

    elif args.evaluate:
        from agents.scout import ScoutAgent
        scout = ScoutAgent(skill_path=skill_path)
        e = scout.evaluate_single(args.evaluate)

        print(f"\nTicket #{e.ticket_id}: {e.ticket.summary}")
        print(f"Verdict: {e.verdict} ({e.score}/100)")
        print(f"Complexity: {e.estimated_complexity}")
        print(f"Clarity: {e.clarity}")
        print(f"Component depth: {e.component_depth}")
        print(f"Someone working: {e.someone_working}")
        print(f"Reasoning: {e.reasoning}")
        print(f"Fix approach: {e.fix_approach}")
        print(f"Risks: {', '.join(e.risk_factors) or 'None'}")

    elif args.once or args.watch:
        from agents.orchestrator import Orchestrator

        repo_path = Path(args.repo) if args.repo else None
        orchestrator = Orchestrator(
            skill_path=skill_path,
            state_path=Path(args.state),
            repo_path=repo_path,
            github_fork=args.fork,
        )

        # --ticket: inject a specific ticket directly, bypassing scout + picker
        if args.ticket:
            from agents.scout import TicketEvaluation
            from tools.trac_client import TracClient, TracTicket
            from agents.base import AgentResponse, TokenUsage

            logger.info("--ticket %d specified, fetching ticket details...", args.ticket)
            trac = TracClient()
            ticket = TracTicket(
                ticket_id=args.ticket, summary="", component="", ticket_type="",
                severity="", version="", owner="", reporter="", status="", stage="",
                has_patch=False, needs_better_patch=False, needs_tests=False,
                needs_docs=False, easy_picking=True,
            )
            ticket = trac.fetch_ticket_detail(ticket)
            logger.info("Ticket #%d: %s", args.ticket, ticket.summary)

            # Force PICK regardless of AI opinion — user explicitly chose this ticket
            evaluation = TicketEvaluation(
                ticket_id=args.ticket,
                ticket=ticket,
                verdict="PICK",
                score=80,
                reasoning="Manually specified via --ticket flag",
                risk_factors=[],
                fix_approach=None,
                estimated_complexity="simple",
                someone_working=False,
                has_existing_pr=False,
                clarity="clear",
                component_depth="surface",
                raw_response=AgentResponse(raw_text="", usage=TokenUsage()),
            )
            orchestrator.state.candidates = [evaluation]
            # Clear active tickets for this ticket so coder doesn't think it's already in progress
            orchestrator.state.active_tickets = [
                t for t in orchestrator.state.active_tickets
                if t.get("ticket_id") != args.ticket
            ]
            logger.info("Injected #%d as PICK — going straight to PICK_AND_CODE", args.ticket)

            # Step 1: Code it
            code_result = orchestrator._action_pick_and_code()
            orchestrator._save_state()
            logger.info("Coding result: %s", code_result)

            # Step 2: If ready, submit PR immediately
            ready_for_pr = [t for t in orchestrator.state.active_tickets if t.get("status") == "ready_for_pr"]
            if ready_for_pr:
                logger.info("Patch ready — submitting PR now...")
                pr_result = orchestrator._action_submit_pr()
                orchestrator._save_state()
                print(json.dumps(pr_result, indent=2, default=str))
            else:
                print(json.dumps(code_result, indent=2, default=str))

        elif args.watch:
            orchestrator.run_continuous(interval_minutes=args.watch)
        else:
            result = orchestrator.run_cycle()
            print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
