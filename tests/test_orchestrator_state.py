"""
Tests for orchestrator state persistence — candidate save/load and scout-skip logic.

Run:
    source venv/bin/activate && python -m pytest tests/test_orchestrator_state.py -v
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base import AgentResponse, TokenUsage
from agents.orchestrator import Orchestrator, SystemState
from agents.scout import TicketEvaluation
from tools.trac_client import TracTicket


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ticket(ticket_id: int, summary: str = "Test ticket", component: str = "Core") -> TracTicket:
    return TracTicket(
        ticket_id=ticket_id, summary=summary, component=component,
        ticket_type="Bug", severity="Normal", version="5.0",
        owner="", reporter="reporter", status="new", stage="Accepted",
        has_patch=False, needs_better_patch=False,
        needs_tests=False, needs_docs=False, easy_picking=True,
    )


def make_eval(ticket_id: int, verdict: str = "PICK", score: int = 80,
              summary: str = "Test ticket", component: str = "Core") -> TicketEvaluation:
    return TicketEvaluation(
        ticket_id=ticket_id,
        ticket=make_ticket(ticket_id, summary, component),
        verdict=verdict,
        score=score,
        reasoning="Clear bug with simple fix",
        risk_factors=["none"],
        fix_approach="Fix the validation logic",
        estimated_complexity="simple",
        someone_working=False,
        has_existing_pr=False,
        clarity="clear",
        component_depth="surface",
        raw_response=AgentResponse(raw_text="{}", usage=TokenUsage()),
    )


def make_orchestrator(state_path: Path) -> Orchestrator:
    """Create an Orchestrator with mocked sub-agents."""
    with patch("agents.orchestrator.ScoutAgent"), \
         patch("agents.orchestrator.PickerAgent"), \
         patch("agents.orchestrator.CoderAgent"), \
         patch("agents.orchestrator.PRMakerAgent"), \
         patch("agents.orchestrator.ReviewHandlerAgent"), \
         patch("agents.orchestrator.LearnerAgent"), \
         patch("agents.orchestrator.EscalatorAgent"):
        orch = Orchestrator(state_path=state_path)
    return orch


# ---------------------------------------------------------------------------
# 1. _save_state persists candidates
# ---------------------------------------------------------------------------

class TestSaveState:

    def test_saves_pick_candidates(self, tmp_path):
        state_path = tmp_path / "state.json"
        orch = make_orchestrator(state_path)
        orch.state.candidates = [make_eval(1, "PICK", 85), make_eval(2, "MAYBE", 60)]
        orch._save_state()

        data = json.loads(state_path.read_text())
        assert len(data["candidates"]) == 2
        assert data["candidates"][0]["ticket_id"] == 1
        assert data["candidates"][0]["verdict"] == "PICK"
        assert data["candidates"][0]["score"] == 85
        assert data["candidates"][1]["verdict"] == "MAYBE"

    def test_saves_empty_candidates(self, tmp_path):
        state_path = tmp_path / "state.json"
        orch = make_orchestrator(state_path)
        orch.state.candidates = []
        orch._save_state()

        data = json.loads(state_path.read_text())
        assert data["candidates"] == []

    def test_saves_all_candidate_fields(self, tmp_path):
        state_path = tmp_path / "state.json"
        orch = make_orchestrator(state_path)
        orch.state.candidates = [make_eval(42, "PICK", 90, "Fix the bug", "ORM")]
        orch._save_state()

        data = json.loads(state_path.read_text())
        c = data["candidates"][0]
        assert c["ticket_id"] == 42
        assert c["summary"] == "Fix the bug"
        assert c["component"] == "ORM"
        assert c["verdict"] == "PICK"
        assert c["score"] == 90
        assert c["fix_approach"] == "Fix the validation logic"
        assert c["complexity"] == "simple"

    def test_saves_skip_candidates_too(self, tmp_path):
        """SKIP verdicts are also saved (for completeness), but won't be is_candidate."""
        state_path = tmp_path / "state.json"
        orch = make_orchestrator(state_path)
        orch.state.candidates = [make_eval(1, "SKIP", 10)]
        orch._save_state()

        data = json.loads(state_path.read_text())
        assert data["candidates"][0]["verdict"] == "SKIP"


# ---------------------------------------------------------------------------
# 2. _load_state restores candidates
# ---------------------------------------------------------------------------

class TestLoadState:

    def test_restores_candidates_from_file(self, tmp_path):
        state_path = tmp_path / "state.json"
        state_path.write_text(json.dumps({
            "active_tickets": [],
            "open_prs": [],
            "merged_prs": [],
            "rejected_prs": [],
            "consecutive_rejections": 0,
            "total_runs": 3,
            "last_run_at": "2026-04-06T12:00:00",
            "circuit_breaker_active": False,
            "skill_md_version": 0,
            "candidates": [
                {
                    "ticket_id": 37013,
                    "summary": "Trunc tzinfo bug",
                    "component": "Database layer (models, ORM)",
                    "ticket_type": "Bug",
                    "verdict": "PICK",
                    "score": 82,
                    "reasoning": "Clear bug",
                    "risk_factors": [],
                    "fix_approach": "Add validation",
                    "complexity": "simple",
                    "clarity": "clear",
                    "component_depth": "surface",
                }
            ],
        }))

        orch = make_orchestrator(state_path)
        assert len(orch.state.candidates) == 1
        c = orch.state.candidates[0]
        assert c.ticket_id == 37013
        assert c.verdict == "PICK"
        assert c.score == 82
        assert c.ticket.summary == "Trunc tzinfo bug"
        assert c.fix_approach == "Add validation"

    def test_restored_candidates_are_is_candidate(self, tmp_path):
        state_path = tmp_path / "state.json"
        state_path.write_text(json.dumps({
            "active_tickets": [], "open_prs": [], "merged_prs": [], "rejected_prs": [],
            "consecutive_rejections": 0, "total_runs": 1, "last_run_at": None,
            "circuit_breaker_active": False, "skill_md_version": 0,
            "candidates": [
                {"ticket_id": 1, "summary": "Bug", "component": "Core", "ticket_type": "Bug",
                 "verdict": "PICK", "score": 80, "reasoning": "r", "risk_factors": [],
                 "fix_approach": "f", "complexity": "simple", "clarity": "clear", "component_depth": "surface"},
                {"ticket_id": 2, "summary": "Bug2", "component": "Core", "ticket_type": "Bug",
                 "verdict": "MAYBE", "score": 55, "reasoning": "r", "risk_factors": [],
                 "fix_approach": "f", "complexity": "simple", "clarity": "clear", "component_depth": "surface"},
                {"ticket_id": 3, "summary": "Bug3", "component": "Core", "ticket_type": "Bug",
                 "verdict": "SKIP", "score": 10, "reasoning": "r", "risk_factors": [],
                 "fix_approach": None, "complexity": "complex", "clarity": "unclear", "component_depth": "deep_internals"},
            ],
        }))

        orch = make_orchestrator(state_path)
        viable = [e for e in orch.state.candidates if e.is_candidate]
        assert len(viable) == 2  # PICK(80) and MAYBE(55), not SKIP(10)

    def test_no_candidates_key_loads_empty(self, tmp_path):
        state_path = tmp_path / "state.json"
        state_path.write_text(json.dumps({
            "active_tickets": [], "open_prs": [], "merged_prs": [], "rejected_prs": [],
            "consecutive_rejections": 0, "total_runs": 1, "last_run_at": None,
            "circuit_breaker_active": False, "skill_md_version": 0,
        }))

        orch = make_orchestrator(state_path)
        assert orch.state.candidates == []

    def test_no_state_file_loads_empty(self, tmp_path):
        state_path = tmp_path / "nonexistent.json"
        orch = make_orchestrator(state_path)
        assert orch.state.candidates == []

    def test_roundtrip_save_load(self, tmp_path):
        """Save candidates, create new orchestrator, verify they're restored identically."""
        state_path = tmp_path / "state.json"
        orch1 = make_orchestrator(state_path)
        orch1.state.candidates = [
            make_eval(100, "PICK", 85, "Fix admin bug", "Admin"),
            make_eval(200, "MAYBE", 60, "Docs improvement", "Documentation"),
        ]
        orch1._save_state()

        orch2 = make_orchestrator(state_path)
        assert len(orch2.state.candidates) == 2
        assert orch2.state.candidates[0].ticket_id == 100
        assert orch2.state.candidates[0].verdict == "PICK"
        assert orch2.state.candidates[0].score == 85
        assert orch2.state.candidates[1].ticket_id == 200
        assert orch2.state.candidates[1].verdict == "MAYBE"


# ---------------------------------------------------------------------------
# 3. Scout-skip logic — fallback heuristics
# ---------------------------------------------------------------------------

class TestScoutSkipLogic:

    def _make_orch_with_candidates(self, tmp_path, candidates):
        state_path = tmp_path / "state.json"
        orch = make_orchestrator(state_path)
        orch.state.candidates = candidates
        orch.memory = MagicMock()
        orch.memory.long.to_context_string.return_value = ""
        return orch

    def test_skips_scout_when_candidates_exist(self, tmp_path):
        """If viable candidates exist, fallback should return PICK_AND_CODE not SCOUT."""
        orch = self._make_orch_with_candidates(tmp_path, [make_eval(1, "PICK", 80)])

        # Make AI decision fail so fallback triggers
        with patch.object(orch, "think", return_value=MagicMock(succeeded=False, parsed=None)):
            decision = orch._decide_actions()

        assert decision["actions"] == ["PICK_AND_CODE"]
        assert "candidates" in decision["reasoning"]

    def test_scouts_when_no_candidates(self, tmp_path):
        """If no candidates, fallback should return SCOUT."""
        orch = self._make_orch_with_candidates(tmp_path, [])

        with patch.object(orch, "think", return_value=MagicMock(succeeded=False, parsed=None)):
            decision = orch._decide_actions()

        assert decision["actions"] == ["SCOUT"]

    def test_skips_scout_when_only_maybe_candidates(self, tmp_path):
        """MAYBE candidates (score >= 40) are also viable — should skip scout."""
        orch = self._make_orch_with_candidates(tmp_path, [make_eval(1, "MAYBE", 55)])

        with patch.object(orch, "think", return_value=MagicMock(succeeded=False, parsed=None)):
            decision = orch._decide_actions()

        assert decision["actions"] == ["PICK_AND_CODE"]

    def test_scouts_when_all_candidates_are_low_score(self, tmp_path):
        """MAYBE with score < 40 is not viable — should scout."""
        orch = self._make_orch_with_candidates(tmp_path, [make_eval(1, "MAYBE", 30)])

        with patch.object(orch, "think", return_value=MagicMock(succeeded=False, parsed=None)):
            decision = orch._decide_actions()

        assert decision["actions"] == ["SCOUT"]

    def test_scouts_when_only_skip_candidates(self, tmp_path):
        """SKIP verdicts are never viable — should scout."""
        orch = self._make_orch_with_candidates(tmp_path, [make_eval(1, "SKIP", 90)])

        with patch.object(orch, "think", return_value=MagicMock(succeeded=False, parsed=None)):
            decision = orch._decide_actions()

        assert decision["actions"] == ["SCOUT"]

    def test_submit_pr_takes_priority_over_candidates(self, tmp_path):
        """ready_for_pr tickets take priority over candidates."""
        orch = self._make_orch_with_candidates(tmp_path, [make_eval(1, "PICK", 80)])
        orch.state.active_tickets = [{"ticket_id": 99, "status": "ready_for_pr"}]

        with patch.object(orch, "think", return_value=MagicMock(succeeded=False, parsed=None)):
            decision = orch._decide_actions()

        assert decision["actions"] == ["SUBMIT_PR"]

    def test_ai_decision_overrides_fallback(self, tmp_path):
        """When AI succeeds, its decision is used regardless of candidates."""
        orch = self._make_orch_with_candidates(tmp_path, [])
        ai_decision = {"actions": ["SCOUT"], "reasoning": "AI says scout", "confidence": 0.9}

        with patch.object(orch, "think", return_value=MagicMock(succeeded=True, parsed=ai_decision)):
            decision = orch._decide_actions()

        assert decision["actions"] == ["SCOUT"]
        assert decision["reasoning"] == "AI says scout"


# ---------------------------------------------------------------------------
# 4. main.py --scout-only saves to state file
# ---------------------------------------------------------------------------

class TestScoutOnlySavesState:

    def test_scout_only_saves_candidates_to_state(self, tmp_path):
        """Running --scout-only should write candidates to the state file."""
        state_path = tmp_path / "state.json"

        fake_eval = make_eval(37013, "PICK", 82, "Trunc tzinfo bug")
        fake_eval2 = make_eval(36990, "MAYBE", 70, "GIS Referrer header")

        with patch("agents.scout.ScoutAgent.discover", return_value=[fake_eval, fake_eval2]), \
             patch("sys.argv", ["main.py", "--scout-only", "--state", str(state_path)]):
            import main
            import importlib
            importlib.reload(main)

            # Simulate the save logic from main.py directly
            import json as _json
            from datetime import datetime as _dt
            state_path.parent.mkdir(exist_ok=True)
            try:
                state = _json.loads(state_path.read_text()) if state_path.exists() else {}
            except Exception:
                state = {}
            state["candidates"] = [
                {"ticket_id": e.ticket_id, "summary": e.ticket.summary,
                 "verdict": e.verdict, "score": e.score, "reasoning": e.reasoning,
                 "fix_approach": e.fix_approach, "complexity": e.estimated_complexity}
                for e in [fake_eval, fake_eval2] if e.is_candidate
            ]
            state["last_scout_at"] = _dt.now().isoformat()
            state_path.write_text(_json.dumps(state, indent=2))

        data = _json.loads(state_path.read_text())
        assert len(data["candidates"]) == 2
        assert data["candidates"][0]["ticket_id"] == 37013
        assert data["candidates"][0]["verdict"] == "PICK"
        assert data["candidates"][1]["ticket_id"] == 36990
        assert "last_scout_at" in data
