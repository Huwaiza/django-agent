"""
Tests for the Scout agent pipeline.

Tests cover:
- JSON parsing (clean JSON, prose-wrapped JSON, markdown fences, malformed)
- _batch_triage (PROMISING/SKIP/NEEDS_DETAIL, chunk failures, all-skip, all-promising)
- _deep_evaluate (PICK, MAYBE, SKIP, parse failure fallback)
- discover() with stop_on_first_pick and continuous modes
- CLI flags: --scout-only, --stop-on-pick, --keep-picking

Run:
    source venv/bin/activate && python -m pytest tests/test_scout.py -v
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base import AgentResponse, BaseAgent, TokenUsage
from agents.scout import (
    BATCH_TRIAGE_TEMPLATE,
    EVALUATION_REQUEST_TEMPLATE,
    ScoutAgent,
    TicketEvaluation,
)
from tools.trac_client import TracTicket


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ticket(ticket_id: int, summary: str = "Test ticket", owner: str = "",
                ticket_type: str = "Bug") -> TracTicket:
    return TracTicket(
        ticket_id=ticket_id,
        summary=summary,
        component="Core",
        ticket_type=ticket_type,
        severity="Normal",
        version="5.0",
        owner=owner,
        reporter="someone",
        status="new",
        stage="Accepted",
        has_patch=False,
        needs_better_patch=False,
        needs_tests=False,
        needs_docs=False,
        easy_picking=True,
        description="A test ticket description.",
    )


def make_eval(ticket_id: int, verdict: str, score: int) -> TicketEvaluation:
    ticket = make_ticket(ticket_id)
    return TicketEvaluation(
        ticket_id=ticket_id,
        ticket=ticket,
        verdict=verdict,
        score=score,
        reasoning="Test reasoning",
        risk_factors=[],
        fix_approach="Test approach",
        estimated_complexity="simple",
        someone_working=False,
        has_existing_pr=False,
        clarity="clear",
        component_depth="surface",
        raw_response=AgentResponse(raw_text="{}", usage=TokenUsage()),
    )


def mock_agent_response(data) -> AgentResponse:
    raw = json.dumps(data)
    return AgentResponse(raw_text=raw, parsed=data, usage=TokenUsage())


def mock_failed_response(raw_text: str = "I need more info") -> AgentResponse:
    return AgentResponse(raw_text=raw_text, parsed=None, usage=TokenUsage())


# ---------------------------------------------------------------------------
# 1. JSON parsing tests (BaseAgent._parse_json)
# ---------------------------------------------------------------------------

class TestParseJson:

    def test_clean_json_object(self):
        result = BaseAgent._parse_json('{"verdict": "PICK", "score": 85}')
        assert result == {"verdict": "PICK", "score": 85}

    def test_clean_json_array(self):
        result = BaseAgent._parse_json('[{"ticket_id": 1, "quick_verdict": "PROMISING"}]')
        assert result == [{"ticket_id": 1, "quick_verdict": "PROMISING"}]

    def test_markdown_fence_json(self):
        text = '```json\n{"verdict": "PICK", "score": 70}\n```'
        result = BaseAgent._parse_json(text)
        assert result == {"verdict": "PICK", "score": 70}

    def test_markdown_fence_no_lang(self):
        text = '```\n{"verdict": "MAYBE", "score": 55}\n```'
        result = BaseAgent._parse_json(text)
        assert result == {"verdict": "MAYBE", "score": 55}

    def test_json_wrapped_in_prose(self):
        # Kimi's common pattern: explanation then JSON
        text = (
            "Based on my analysis, here is the evaluation:\n\n"
            '{"verdict": "PICK", "score": 82, "reasoning": "Clear bug"}'
        )
        result = BaseAgent._parse_json(text)
        assert result is not None
        assert result["verdict"] == "PICK"
        assert result["score"] == 82

    def test_json_array_wrapped_in_prose(self):
        text = (
            "Here are the triage results:\n\n"
            '[{"ticket_id": 123, "quick_verdict": "PROMISING", "one_line_reason": "bug"}]'
        )
        result = BaseAgent._parse_json(text)
        assert result is not None
        assert result[0]["ticket_id"] == 123

    def test_completely_malformed(self):
        result = BaseAgent._parse_json("This is just plain text with no JSON at all.")
        assert result is None

    def test_empty_string(self):
        result = BaseAgent._parse_json("")
        assert result is None

    def test_whitespace_only(self):
        result = BaseAgent._parse_json("   \n\n  ")
        assert result is None

    def test_partial_json(self):
        result = BaseAgent._parse_json('{"verdict": "PICK"')
        assert result is None

    def test_nested_json(self):
        data = {"verdict": "PICK", "risk_factors": ["low risk"], "score": 90}
        result = BaseAgent._parse_json(json.dumps(data))
        assert result == data


# ---------------------------------------------------------------------------
# 2. _batch_triage tests
# ---------------------------------------------------------------------------

class TestBatchTriage:

    def setup_method(self):
        self.scout = ScoutAgent.__new__(ScoutAgent)
        self.scout.name = "scout"
        self.scout.model = "opencode/kimi-k2.5"
        self.scout.call_count = 0
        self.scout.total_cost = 0.0
        self.scout.total_input_tokens = 0
        self.scout.total_output_tokens = 0
        self.scout.budget = MagicMock()
        self.scout.budget.check_budget.return_value = (True, "OK")
        self.scout.memory = MagicMock()
        self.scout.system_prompt = "You are a scout."
        self.scout.trac = MagicMock()

    def _make_triage_response(self, tickets, verdicts: dict) -> AgentResponse:
        """verdicts: {ticket_id: "PROMISING"|"SKIP"|"NEEDS_DETAIL"}"""
        data = [
            {"ticket_id": t.ticket_id, "quick_verdict": verdicts.get(t.ticket_id, "SKIP"),
             "one_line_reason": "test"}
            for t in tickets
        ]
        return mock_agent_response(data)

    def test_all_promising(self):
        tickets = [make_ticket(i) for i in range(1, 6)]
        verdicts = {i: "PROMISING" for i in range(1, 6)}
        with patch.object(self.scout, "think", return_value=self._make_triage_response(tickets, verdicts)):
            result = self.scout._batch_triage(tickets, chunk_size=20)
        assert len(result) == 5

    def test_all_skip(self):
        tickets = [make_ticket(i) for i in range(1, 6)]
        verdicts = {i: "SKIP" for i in range(1, 6)}
        with patch.object(self.scout, "think", return_value=self._make_triage_response(tickets, verdicts)):
            result = self.scout._batch_triage(tickets, chunk_size=20)
        assert len(result) == 0

    def test_mixed_verdicts(self):
        tickets = [make_ticket(i) for i in range(1, 6)]
        verdicts = {1: "PROMISING", 2: "SKIP", 3: "NEEDS_DETAIL", 4: "SKIP", 5: "PROMISING"}
        with patch.object(self.scout, "think", return_value=self._make_triage_response(tickets, verdicts)):
            result = self.scout._batch_triage(tickets, chunk_size=20)
        ids = [t.ticket_id for t in result]
        assert 1 in ids
        assert 3 in ids
        assert 5 in ids
        assert 2 not in ids
        assert 4 not in ids

    def test_chunk_failure_falls_back_to_including_all(self):
        tickets = [make_ticket(i) for i in range(1, 6)]
        with patch.object(self.scout, "think", return_value=mock_failed_response()):
            result = self.scout._batch_triage(tickets, chunk_size=20)
        # On failure, all tickets in the chunk are included
        assert len(result) == 5

    def test_chunking_processes_all(self):
        tickets = [make_ticket(i) for i in range(1, 61)]  # 60 tickets
        call_count = []

        def fake_think(user_message, **kwargs):
            chunk_tickets = [t for t in tickets if f"#{t.ticket_id} " in user_message]
            call_count.append(len(chunk_tickets))
            data = [{"ticket_id": t.ticket_id, "quick_verdict": "PROMISING", "one_line_reason": "ok"}
                    for t in chunk_tickets]
            return mock_agent_response(data)

        with patch.object(self.scout, "think", side_effect=fake_think):
            result = self.scout._batch_triage(tickets, chunk_size=20)

        assert len(call_count) == 3  # 60 / 20 = 3 chunks
        assert len(result) == 60

    def test_partial_chunk_failure_keeps_others(self):
        tickets = [make_ticket(i) for i in range(1, 41)]  # 40 tickets = 2 chunks of 20
        chunk1 = tickets[:20]
        chunk2 = tickets[20:]

        def fake_think(user_message, **kwargs):
            if f"#{chunk1[0].ticket_id} " in user_message:
                # chunk 1 fails
                return mock_failed_response()
            else:
                # chunk 2 succeeds with all PROMISING
                data = [{"ticket_id": t.ticket_id, "quick_verdict": "PROMISING", "one_line_reason": "ok"}
                        for t in chunk2]
                return mock_agent_response(data)

        with patch.object(self.scout, "think", side_effect=fake_think):
            result = self.scout._batch_triage(tickets, chunk_size=20)

        # chunk1 all included (fallback) + chunk2 all promising = 40 total
        assert len(result) == 40


# ---------------------------------------------------------------------------
# 3. _deep_evaluate tests
# ---------------------------------------------------------------------------

class TestDeepEvaluate:

    def setup_method(self):
        self.scout = ScoutAgent.__new__(ScoutAgent)
        self.scout.name = "scout"
        self.scout.model = "opencode/kimi-k2.5"
        self.scout.call_count = 0
        self.scout.total_cost = 0.0
        self.scout.total_input_tokens = 0
        self.scout.total_output_tokens = 0
        self.scout.budget = MagicMock()
        self.scout.budget.check_budget.return_value = (True, "OK")
        self.scout.memory = MagicMock()
        self.scout.system_prompt = "You are a scout."
        self.scout.trac = MagicMock()

    def _eval_response(self, verdict: str, score: int) -> AgentResponse:
        data = {
            "verdict": verdict,
            "score": score,
            "reasoning": "Test reasoning",
            "risk_factors": ["none"],
            "fix_approach_sketch": "Test fix",
            "estimated_complexity": "simple",
            "someone_actively_working": False,
            "has_existing_pr": False,
            "clarity_of_problem": "clear",
            "component_depth": "surface",
        }
        return mock_agent_response(data)

    def test_pick_verdict(self):
        ticket = make_ticket(100)
        with patch.object(self.scout, "think", return_value=self._eval_response("PICK", 85)):
            result = self.scout._deep_evaluate(ticket)
        assert result.verdict == "PICK"
        assert result.score == 85
        assert result.is_candidate is True

    def test_maybe_verdict(self):
        ticket = make_ticket(101)
        with patch.object(self.scout, "think", return_value=self._eval_response("MAYBE", 60)):
            result = self.scout._deep_evaluate(ticket)
        assert result.verdict == "MAYBE"
        assert result.is_candidate is True  # score >= 40

    def test_maybe_low_score_not_candidate(self):
        ticket = make_ticket(102)
        with patch.object(self.scout, "think", return_value=self._eval_response("MAYBE", 30)):
            result = self.scout._deep_evaluate(ticket)
        assert result.verdict == "MAYBE"
        assert result.is_candidate is False  # score < 40

    def test_skip_verdict(self):
        ticket = make_ticket(103)
        with patch.object(self.scout, "think", return_value=self._eval_response("SKIP", 20)):
            result = self.scout._deep_evaluate(ticket)
        assert result.verdict == "SKIP"
        assert result.is_candidate is False

    def test_parse_failure_returns_skip(self):
        ticket = make_ticket(104)
        with patch.object(self.scout, "think", return_value=mock_failed_response()):
            result = self.scout._deep_evaluate(ticket)
        assert result.verdict == "SKIP"
        assert result.score == 0
        assert "Evaluation failed" in result.reasoning

    def test_missing_fields_use_defaults(self):
        ticket = make_ticket(105)
        # Response with only verdict and score, missing other fields
        partial = AgentResponse(
            raw_text='{"verdict": "PICK", "score": 75}',
            parsed={"verdict": "PICK", "score": 75},
            usage=TokenUsage(),
        )
        with patch.object(self.scout, "think", return_value=partial):
            result = self.scout._deep_evaluate(ticket)
        assert result.verdict == "PICK"
        assert result.score == 75
        assert result.risk_factors == []
        assert result.fix_approach is None


# ---------------------------------------------------------------------------
# 4. discover() — stop_on_first_pick and keep_picking modes
# ---------------------------------------------------------------------------

class TestDiscoverModes:

    def setup_method(self):
        self.scout = ScoutAgent.__new__(ScoutAgent)
        self.scout.name = "scout"
        self.scout.model = "opencode/kimi-k2.5"
        self.scout.call_count = 0
        self.scout.total_cost = 0.0
        self.scout.total_input_tokens = 0
        self.scout.total_output_tokens = 0
        self.scout.budget = MagicMock()
        self.scout.budget.check_budget.return_value = (True, "OK")
        self.scout.memory = MagicMock()
        self.scout.system_prompt = "You are a scout."
        self.scout.trac = MagicMock()
        self.scout.skill_path = None
        self.scout.use_skill = False

    def _patch_discover(self, evaluations: list[TicketEvaluation]):
        """Patch the full pipeline to return given evaluations."""
        tickets = [e.ticket for e in evaluations]

        triage_data = [{"ticket_id": t.ticket_id, "quick_verdict": "PROMISING",
                        "one_line_reason": "ok"} for t in tickets]

        eval_map = {e.ticket_id: e for e in evaluations}

        def fake_trac_fetch():
            return tickets

        def fake_trac_detail(ticket):
            return ticket

        def fake_think(user_message, **kwargs):
            # Return triage data for batch, eval data for single
            if "quick_verdict" in user_message or "PROMISING" in user_message or "batch" in user_message.lower():
                return mock_agent_response(triage_data)
            # deep eval
            for tid, ev in eval_map.items():
                if str(tid) in user_message:
                    data = {
                        "verdict": ev.verdict, "score": ev.score,
                        "reasoning": ev.reasoning, "risk_factors": ev.risk_factors,
                        "fix_approach_sketch": ev.fix_approach,
                        "estimated_complexity": ev.estimated_complexity,
                        "someone_actively_working": ev.someone_working,
                        "has_existing_pr": ev.has_existing_pr,
                        "clarity_of_problem": ev.clarity,
                        "component_depth": ev.component_depth,
                    }
                    return mock_agent_response(data)
            return mock_failed_response()

        self.scout.trac.fetch_easy_pickings = fake_trac_fetch
        self.scout.trac.fetch_ticket_detail = fake_trac_detail
        self.scout.think = MagicMock(side_effect=fake_think)

    def test_discover_returns_all_by_default(self):
        evals = [
            make_eval(1, "PICK", 90),
            make_eval(2, "MAYBE", 60),
            make_eval(3, "SKIP", 20),
            make_eval(4, "PICK", 80),
        ]
        self._patch_discover(evals)
        result = self.scout.discover(deep_eval_limit=10)
        assert len(result) == 4

    def test_discover_sorted_by_score(self):
        evals = [
            make_eval(1, "SKIP", 10),
            make_eval(2, "PICK", 90),
            make_eval(3, "MAYBE", 60),
        ]
        self._patch_discover(evals)
        result = self.scout.discover(deep_eval_limit=10)
        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_discover_stop_on_first_pick(self):
        """stop_on_first_pick=True should stop deep-eval as soon as one PICK is found."""
        evaluated = []

        tickets = [make_ticket(i) for i in range(1, 6)]
        # ticket 3 is the first PICK
        verdicts = {1: "SKIP", 2: "MAYBE", 3: "PICK", 4: "PICK", 5: "MAYBE"}
        scores =   {1: 10,     2: 50,      3: 85,     4: 80,     5: 55}

        def fake_trac_fetch():
            return tickets

        def fake_trac_detail(t):
            return t

        triage_data = [{"ticket_id": t.ticket_id, "quick_verdict": "PROMISING",
                        "one_line_reason": "ok"} for t in tickets]

        def fake_think(user_message, **kwargs):
            # Batch triage prompts contain "quick_verdict" in the template
            if "quick_verdict" in user_message:
                return mock_agent_response(triage_data)
            # Deep eval — match by ticket ID in prompt
            for t in tickets:
                if f"TICKET #{t.ticket_id}" in user_message or f"#{t.ticket_id}:" in user_message:
                    evaluated.append(t.ticket_id)
                    v = verdicts[t.ticket_id]
                    s = scores[t.ticket_id]
                    data = {
                        "verdict": v, "score": s, "reasoning": "r",
                        "risk_factors": [], "fix_approach_sketch": "f",
                        "estimated_complexity": "simple",
                        "someone_actively_working": False, "has_existing_pr": False,
                        "clarity_of_problem": "clear", "component_depth": "surface",
                    }
                    return mock_agent_response(data)
            return mock_failed_response()

        self.scout.trac.fetch_easy_pickings = fake_trac_fetch
        self.scout.trac.fetch_ticket_detail = fake_trac_detail
        self.scout.think = MagicMock(side_effect=fake_think)

        result = self.scout.discover(deep_eval_limit=10, stop_on_first_pick=True)

        picks = [r for r in result if r.verdict == "PICK"]
        assert len(picks) >= 1
        # Should not have evaluated tickets 4 and 5
        assert 4 not in evaluated
        assert 5 not in evaluated

    def test_discover_keep_picking_respects_limit(self):
        """deep_eval_limit should cap how many tickets are deep-evaluated."""
        tickets = [make_ticket(i) for i in range(1, 21)]  # 20 tickets

        def fake_trac_fetch():
            return tickets

        def fake_trac_detail(t):
            return t

        triage_data = [{"ticket_id": t.ticket_id, "quick_verdict": "PROMISING",
                        "one_line_reason": "ok"} for t in tickets]

        evaluated = []

        def fake_think(user_message, **kwargs):
            promising_match = sum(1 for t in tickets if f"#{t.ticket_id} " in user_message)
            if promising_match > 3:
                return mock_agent_response(triage_data)
            for t in tickets:
                if str(t.ticket_id) in user_message:
                    evaluated.append(t.ticket_id)
                    data = {
                        "verdict": "PICK", "score": 80, "reasoning": "r",
                        "risk_factors": [], "fix_approach_sketch": "f",
                        "estimated_complexity": "simple",
                        "someone_actively_working": False, "has_existing_pr": False,
                        "clarity_of_problem": "clear", "component_depth": "surface",
                    }
                    return mock_agent_response(data)
            return mock_failed_response()

        self.scout.trac.fetch_easy_pickings = fake_trac_fetch
        self.scout.trac.fetch_ticket_detail = fake_trac_detail
        self.scout.think = MagicMock(side_effect=fake_think)

        result = self.scout.discover(deep_eval_limit=5)
        assert len(result) <= 5

    def test_discover_no_tickets_returns_empty(self):
        self.scout.trac.fetch_easy_pickings = MagicMock(return_value=[])
        result = self.scout.discover()
        assert result == []

    def test_discover_all_skip_returns_empty_candidates(self):
        evals = [make_eval(i, "SKIP", 10) for i in range(1, 4)]
        self._patch_discover(evals)
        result = self.scout.discover(deep_eval_limit=10)
        picks = [r for r in result if r.verdict == "PICK"]
        maybes = [r for r in result if r.verdict == "MAYBE"]
        assert len(picks) == 0
        assert len(maybes) == 0


# ---------------------------------------------------------------------------
# 5. is_candidate property
# ---------------------------------------------------------------------------

class TestIsCandidate:

    def test_pick_high_score(self):
        e = make_eval(1, "PICK", 85)
        assert e.is_candidate is True

    def test_pick_low_score(self):
        e = make_eval(2, "PICK", 20)
        assert e.is_candidate is False  # score < 40

    def test_maybe_above_threshold(self):
        e = make_eval(3, "MAYBE", 40)
        assert e.is_candidate is True

    def test_maybe_below_threshold(self):
        e = make_eval(4, "MAYBE", 39)
        assert e.is_candidate is False

    def test_skip_always_false(self):
        e = make_eval(5, "SKIP", 90)
        assert e.is_candidate is False
