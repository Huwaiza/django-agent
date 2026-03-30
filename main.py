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

    parser.add_argument("--skill", default="skills/django-contributor/SKILL.md", help="Path to SKILL.md")
    parser.add_argument("--state", default="db/orchestrator_state.json", help="Path to state file")
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

    # Check API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.error(
            "ANTHROPIC_API_KEY environment variable not set.\n"
            "Export it: export ANTHROPIC_API_KEY='sk-ant-...'"
        )
        sys.exit(1)

    # Configure global token budget
    from agents.base import get_budget
    budget = get_budget()
    budget.max_cost_per_cycle = args.budget_cycle
    budget.max_cost_per_day = args.budget_daily

    skill_path = Path(args.skill) if Path(args.skill).exists() else None

    if args.scout_only:
        from agents.scout import ScoutAgent
        scout = ScoutAgent(skill_path=skill_path)
        evaluations = scout.discover(deep_eval_limit=10)

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

        if args.watch:
            orchestrator.run_continuous(interval_minutes=args.watch)
        else:
            result = orchestrator.run_cycle()
            print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
