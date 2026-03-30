"""
Memory Management System — Short-term, Long-term, and Contextual memory.

Three memory tiers that any agent can read/write:

SHORT-TERM MEMORY
  - Lives for one orchestration cycle, then resets
  - Holds: what Scout found this run, what Coder produced, decisions made
  - Purpose: agents within a cycle can see what other agents just did
  - Implementation: in-memory dict

LONG-TERM MEMORY
  - Persists forever across all runs (SQLite-backed)
  - Holds: ticket outcomes, PR history, reviewer patterns, component stats
  - Purpose: Picker knows "we've merged 3 Admin PRs" and "ORM rejected us twice"
  - Implementation: SQLite tables with structured queries

CONTEXTUAL MEMORY
  - Per-entity context (ticket, PR, reviewer, component)
  - Any agent can store and retrieve context about a specific entity
  - Purpose: Review Handler recalls the Coder's approach for ticket #35421
  - Implementation: SQLite key-value with entity_type + entity_id indexing
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("memory")

DB_PATH = Path("db/memory.sqlite")


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _init_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        -- Long-term: PR outcome history
        CREATE TABLE IF NOT EXISTS pr_history (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_id       INTEGER NOT NULL,
            pr_number       INTEGER,
            component       TEXT,
            outcome         TEXT NOT NULL,  -- merged, rejected, abandoned
            self_review_score INTEGER,
            reviewer_feedback TEXT,
            lessons_extracted BOOLEAN DEFAULT 0,
            created_at      TEXT DEFAULT (datetime('now'))
        );

        -- Long-term: component performance stats (materialized)
        CREATE TABLE IF NOT EXISTS component_stats (
            component       TEXT PRIMARY KEY,
            total_prs       INTEGER DEFAULT 0,
            merged          INTEGER DEFAULT 0,
            rejected        INTEGER DEFAULT 0,
            avg_review_score REAL DEFAULT 0,
            last_pr_at      TEXT,
            updated_at      TEXT DEFAULT (datetime('now'))
        );

        -- Long-term: reviewer patterns
        CREATE TABLE IF NOT EXISTS reviewer_patterns (
            reviewer        TEXT NOT NULL,
            pattern_type    TEXT NOT NULL,  -- preference, style, tone
            description     TEXT NOT NULL,
            confidence      TEXT DEFAULT 'medium',
            source_pr       INTEGER,
            created_at      TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (reviewer, description)
        );

        -- Contextual: per-entity key-value store
        CREATE TABLE IF NOT EXISTS context_store (
            entity_type     TEXT NOT NULL,  -- ticket, pr, component, reviewer
            entity_id       TEXT NOT NULL,
            key             TEXT NOT NULL,
            value           TEXT NOT NULL,
            agent           TEXT,           -- which agent wrote this
            created_at      TEXT DEFAULT (datetime('now')),
            updated_at      TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (entity_type, entity_id, key)
        );

        CREATE INDEX IF NOT EXISTS idx_pr_history_component ON pr_history(component);
        CREATE INDEX IF NOT EXISTS idx_pr_history_outcome ON pr_history(outcome);
        CREATE INDEX IF NOT EXISTS idx_context_entity ON context_store(entity_type, entity_id);
    """)
    conn.commit()


# ═══════════════════════════════════════════════════════════════════════
# SHORT-TERM MEMORY — in-memory, resets each cycle
# ═══════════════════════════════════════════════════════════════════════

class ShortTermMemory:
    """
    Current cycle context. Any agent can write observations,
    any other agent in the same cycle can read them.

    Resets at the start of each orchestration cycle.
    """

    def __init__(self):
        self._store: dict[str, Any] = {}
        self._events: list[dict] = []

    def set(self, key: str, value: Any, agent: str = "") -> None:
        """Store a value for this cycle."""
        self._store[key] = value
        self._events.append({
            "action": "set", "key": key, "agent": agent,
            "timestamp": datetime.now().isoformat(),
        })

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from this cycle."""
        return self._store.get(key, default)

    def append(self, key: str, value: Any, agent: str = "") -> None:
        """Append to a list under key (creates list if doesn't exist)."""
        if key not in self._store:
            self._store[key] = []
        self._store[key].append(value)

    def get_all(self) -> dict[str, Any]:
        """Get entire cycle context (for orchestrator summaries)."""
        return dict(self._store)

    def get_events(self) -> list[dict]:
        """Get ordered list of everything that happened this cycle."""
        return list(self._events)

    def reset(self) -> None:
        """Clear everything — called at start of each cycle."""
        self._store.clear()
        self._events.clear()

    def to_context_string(self) -> str:
        """Format for injection into an agent's prompt."""
        if not self._store:
            return "(no short-term memory this cycle)"
        lines = ["SHORT-TERM MEMORY (this cycle):"]
        for key, value in self._store.items():
            if isinstance(value, list):
                lines.append(f"  {key}: {len(value)} items")
                for item in value[-3:]:  # last 3 items only to save tokens
                    lines.append(f"    - {str(item)[:100]}")
            else:
                lines.append(f"  {key}: {str(value)[:200]}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# LONG-TERM MEMORY — SQLite-backed, persists forever
# ═══════════════════════════════════════════════════════════════════════

class LongTermMemory:
    """
    Persistent knowledge across all runs.

    Stores PR outcomes, component success rates, and reviewer patterns.
    Agents query this to make informed decisions — e.g., the Picker
    knows "we've merged 3 Admin PRs and been rejected on ORM twice."
    """

    def __init__(self):
        self._conn = _get_conn()
        _init_tables(self._conn)

    def record_pr_outcome(
        self,
        ticket_id: int,
        component: str,
        outcome: str,
        pr_number: int | None = None,
        self_review_score: int = 0,
        reviewer_feedback: str = "",
    ) -> None:
        """Record a PR outcome (merged/rejected/abandoned)."""
        self._conn.execute("""
            INSERT INTO pr_history (ticket_id, pr_number, component, outcome,
                                    self_review_score, reviewer_feedback)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (ticket_id, pr_number, component, outcome, self_review_score, reviewer_feedback))
        self._update_component_stats(component)
        self._conn.commit()

    def get_component_stats(self, component: str | None = None) -> list[dict]:
        """Get success/failure stats per component."""
        if component:
            rows = self._conn.execute(
                "SELECT * FROM component_stats WHERE component = ?", (component,)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM component_stats ORDER BY total_prs DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_successful_components(self) -> list[str]:
        """Components where we've had PRs merged."""
        rows = self._conn.execute(
            "SELECT component FROM component_stats WHERE merged > 0 ORDER BY merged DESC"
        ).fetchall()
        return [r["component"] for r in rows]

    def get_rejected_components(self, recent_n: int = 5) -> list[str]:
        """Components with recent rejections."""
        rows = self._conn.execute("""
            SELECT component FROM pr_history
            WHERE outcome = 'rejected'
            ORDER BY created_at DESC LIMIT ?
        """, (recent_n,)).fetchall()
        return list(set(r["component"] for r in rows))

    def get_merge_rate(self) -> float:
        """Overall merge rate across all PRs."""
        row = self._conn.execute("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN outcome = 'merged' THEN 1 ELSE 0 END) as merged
            FROM pr_history
        """).fetchone()
        if row["total"] == 0:
            return 0.0
        return row["merged"] / row["total"]

    def record_reviewer_pattern(
        self,
        reviewer: str,
        pattern_type: str,
        description: str,
        confidence: str = "medium",
        source_pr: int | None = None,
    ) -> None:
        """Record a reviewer preference or pattern."""
        self._conn.execute("""
            INSERT OR REPLACE INTO reviewer_patterns
                (reviewer, pattern_type, description, confidence, source_pr)
            VALUES (?, ?, ?, ?, ?)
        """, (reviewer, pattern_type, description, confidence, source_pr))
        self._conn.commit()

    def get_reviewer_patterns(self, reviewer: str | None = None) -> list[dict]:
        """Get known patterns for a reviewer (or all reviewers)."""
        if reviewer:
            rows = self._conn.execute(
                "SELECT * FROM reviewer_patterns WHERE reviewer = ?", (reviewer,)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM reviewer_patterns").fetchall()
        return [dict(r) for r in rows]

    def to_context_string(self) -> str:
        """Format long-term stats for injection into agent prompts."""
        stats = self.get_component_stats()
        merge_rate = self.get_merge_rate()
        reviewer_count = self._conn.execute("SELECT COUNT(DISTINCT reviewer) FROM reviewer_patterns").fetchone()[0]

        lines = [f"LONG-TERM MEMORY (all-time):"]
        lines.append(f"  Overall merge rate: {merge_rate:.0%}")
        lines.append(f"  Known reviewer patterns: {reviewer_count}")

        if stats:
            lines.append("  Component track record:")
            for s in stats[:8]:
                lines.append(
                    f"    {s['component']}: {s['merged']}/{s['total_prs']} merged "
                    f"(avg score: {s['avg_review_score']:.0f})"
                )
        return "\n".join(lines)

    def _update_component_stats(self, component: str) -> None:
        """Recompute stats for a component from pr_history."""
        self._conn.execute("""
            INSERT INTO component_stats (component, total_prs, merged, rejected, avg_review_score, last_pr_at, updated_at)
            SELECT
                component,
                COUNT(*),
                SUM(CASE WHEN outcome = 'merged' THEN 1 ELSE 0 END),
                SUM(CASE WHEN outcome = 'rejected' THEN 1 ELSE 0 END),
                AVG(self_review_score),
                MAX(created_at),
                datetime('now')
            FROM pr_history WHERE component = ?
            ON CONFLICT(component) DO UPDATE SET
                total_prs = excluded.total_prs,
                merged = excluded.merged,
                rejected = excluded.rejected,
                avg_review_score = excluded.avg_review_score,
                last_pr_at = excluded.last_pr_at,
                updated_at = excluded.updated_at
        """, (component,))


# ═══════════════════════════════════════════════════════════════════════
# CONTEXTUAL MEMORY — per-entity key-value, SQLite-backed
# ═══════════════════════════════════════════════════════════════════════

class ContextualMemory:
    """
    Per-entity context that any agent can store and retrieve.

    When the Scout evaluates ticket #35421, it stores:
        ctx.set("ticket", "35421", "fix_approach", "Patch the timezone conversion in admin widget")
        ctx.set("ticket", "35421", "scout_verdict", "PICK")

    Later, when the Review Handler sees a comment on PR for #35421, it retrieves:
        approach = ctx.get("ticket", "35421", "fix_approach")
        # Now it knows the original intent and can explain it to the reviewer
    """

    def __init__(self):
        self._conn = _get_conn()
        _init_tables(self._conn)

    def set(self, entity_type: str, entity_id: str, key: str, value: Any, agent: str = "") -> None:
        """Store context for an entity."""
        serialized = json.dumps(value) if not isinstance(value, str) else value
        self._conn.execute("""
            INSERT INTO context_store (entity_type, entity_id, key, value, agent, updated_at)
            VALUES (?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(entity_type, entity_id, key) DO UPDATE SET
                value = excluded.value,
                agent = excluded.agent,
                updated_at = excluded.updated_at
        """, (entity_type, str(entity_id), key, serialized, agent))
        self._conn.commit()

    def get(self, entity_type: str, entity_id: str, key: str, default: Any = None) -> Any:
        """Retrieve a specific context value."""
        row = self._conn.execute("""
            SELECT value FROM context_store
            WHERE entity_type = ? AND entity_id = ? AND key = ?
        """, (entity_type, str(entity_id), key)).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except (json.JSONDecodeError, TypeError):
            return row["value"]

    def get_all_for_entity(self, entity_type: str, entity_id: str) -> dict[str, Any]:
        """Get all context for an entity — useful when an agent needs full picture."""
        rows = self._conn.execute("""
            SELECT key, value, agent, updated_at FROM context_store
            WHERE entity_type = ? AND entity_id = ?
        """, (entity_type, str(entity_id))).fetchall()
        result = {}
        for row in rows:
            try:
                result[row["key"]] = json.loads(row["value"])
            except (json.JSONDecodeError, TypeError):
                result[row["key"]] = row["value"]
        return result

    def to_context_string(self, entity_type: str, entity_id: str) -> str:
        """Format entity context for injection into agent prompt."""
        data = self.get_all_for_entity(entity_type, entity_id)
        if not data:
            return f"(no contextual memory for {entity_type}:{entity_id})"
        lines = [f"CONTEXT for {entity_type} #{entity_id}:"]
        for key, value in data.items():
            lines.append(f"  {key}: {str(value)[:200]}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# UNIFIED MEMORY — single interface for all three tiers
# ═══════════════════════════════════════════════════════════════════════

class Memory:
    """
    Unified memory interface for the agent system.

    Usage:
        memory = Memory()

        # Short-term (this cycle only)
        memory.short.set("scout_candidates", [...], agent="scout")
        candidates = memory.short.get("scout_candidates")

        # Long-term (forever)
        memory.long.record_pr_outcome(ticket_id=35421, component="Admin", outcome="merged")
        stats = memory.long.get_component_stats("Admin")

        # Contextual (per entity)
        memory.ctx.set("ticket", "35421", "fix_approach", "Patch the widget")
        approach = memory.ctx.get("ticket", "35421", "fix_approach")

        # Reset short-term at cycle start
        memory.reset_cycle()

        # Get formatted summary for agent prompts
        summary = memory.to_context_string(ticket_id=35421)
    """

    def __init__(self):
        self.short = ShortTermMemory()
        self.long = LongTermMemory()
        self.ctx = ContextualMemory()

    def reset_cycle(self) -> None:
        """Reset short-term memory at the start of each cycle."""
        self.short.reset()

    def to_context_string(self, ticket_id: int | None = None) -> str:
        """
        Build a combined memory context string for agent prompts.

        Token-efficient: only includes relevant sections.
        """
        sections = []

        # Short-term (only if there's data this cycle)
        if self.short.get_all():
            sections.append(self.short.to_context_string())

        # Long-term (only the summary stats, not raw data)
        long_ctx = self.long.to_context_string()
        if "0%" not in long_ctx or "merge rate" not in long_ctx:
            sections.append(long_ctx)

        # Contextual (only if a specific ticket is provided)
        if ticket_id:
            ticket_ctx = self.ctx.to_context_string("ticket", str(ticket_id))
            if "no contextual memory" not in ticket_ctx:
                sections.append(ticket_ctx)

        return "\n\n".join(sections) if sections else "(no memory available)"


# Global memory instance
_global_memory: Memory | None = None


def get_memory() -> Memory:
    global _global_memory
    if _global_memory is None:
        _global_memory = Memory()
    return _global_memory
