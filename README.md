# Django Open-Source Contribution Multi-Agent System

An autonomous AI agent system that discovers Django tickets, writes patches, submits PRs, responds to reviewer comments, and learns from every interaction.

---

## Prerequisites

- Python 3.12+
- Django repo cloned at `/Users/huwaizatahir/personal/work/django`
- Fork at `github.com/Huwaiza/django`
- `gh` CLI authenticated (`gh auth status`)
- `opencode` CLI installed and authenticated with Zen (`opencode auth list`)
- Django CLA signed at https://www.djangoproject.com/foundation/cla/

---

## Setup

```bash
cd django-agent
source venv/bin/activate
source .env
```

No API key needed — all AI calls go through OpenCode Zen using `opencode/kimi-k2.5`.

---

## Usage

### Scout only — find tickets, no code changes

```bash
# Default: evaluate up to 20 tickets
source .env && python main.py --scout-only -v

# Stop the moment one PICK is found (fastest)
source .env && python main.py --scout-only --stop-on-pick -v

# Evaluate more tickets (keep picking through the list)
source .env && python main.py --scout-only --keep-picking 50 -v
```

### Evaluate a specific ticket

```bash
source .env && python main.py --evaluate 37013 -v
```

### Full cycle (scout → pick → code → self-review → PR)

```bash
source .env && python main.py --once \
  --repo /Users/huwaizatahir/personal/work/django \
  --fork Huwaiza/django \
  -v
```

### Full cycle on a specific ticket (skips scout + picker)

```bash
source .env && python main.py --once \
  --repo /Users/huwaizatahir/personal/work/django \
  --fork Huwaiza/django \
  --ticket 37013 \
  -v
```

Fetches the ticket from Trac, evaluates it, and goes straight to coding. Use this after `--scout-only` finds a PICK you want to act on immediately.

### Continuous mode (every 60 minutes)

```bash
source .env && python main.py --watch 60 \
  --repo /Users/huwaizatahir/personal/work/django \
  --fork Huwaiza/django \
  -v
```

### Background daemon

```bash
nohup bash -c 'source .env && python main.py --watch 60 \
  --repo /Users/huwaizatahir/personal/work/django \
  --fork Huwaiza/django \
  -v' >> logs/agent.log 2>&1 &

tail -f logs/agent.log   # watch it
kill $(pgrep -f "main.py --watch")  # stop it
```

### Fix CI failures on an open PR

After a PR is submitted, if a CI check (e.g. docs lint) fails, run this to automatically diagnose and push a fix:

```bash
source .env && python main.py --fix-pr 21064 \
  --repo /Users/huwaizatahir/personal/work/django \
  -v
```

The agent will:
1. Fetch the list of failed CI checks from GitHub
2. Download the failure log for each failing check
3. Run `opencode run` to fix the issue in the repo
4. Amend the commit and push with `--force-with-lease --force-if-includes`

### All CLI flags

```
--once                  Single orchestration cycle
--watch MIN             Continuous, every MIN minutes
--scout-only            Just discover tickets (no coding)
--evaluate TICKET_ID    Evaluate one specific ticket
--fix-pr PR_NUMBER      Fix CI failures on an open PR and push
--stop-on-pick          Stop scouting as soon as one PICK is found
--keep-picking N        Max tickets to deep-evaluate (default: 20)
--ticket TICKET_ID      Code a specific ticket directly (skips scout + picker)
--repo PATH             Path to Django repo clone
--fork USER/django      Your GitHub fork (e.g. Huwaiza/django)
--skill PATH            Path to SKILL.md
--state PATH            Path to state file
-v                      Verbose logging
```

---

## How Ticket Picking Works

### Phase 0 — Fetch from Trac

Fetches all open tickets (`status=new|assigned`, `has_patch=0` or `needs_better_patch=1`) ordered by most recently modified. This gives ~900 tickets.

### Phase 1 — Heuristic triage (instant, no AI)

Fast Python filter that drops obvious non-starters:

**Skipped automatically:**
- Type is `New feature` (too large, needs design consensus)
- Summary contains large-scope keywords: `"add support for"`, `"implement "`, `"full text search"`, `"composite foreign"`, `"async transaction"`, `"drag and drop"`, `"datepicker"`, etc.

**Passes through:**
- Bug fixes
- Documentation improvements
- Cleanup / small behavioral fixes
- Tickets with or without owners (having an owner doesn't mean it's being actively worked on)

This reduces ~900 → ~530 tickets without any AI calls.

### Phase 2 — AI deep evaluation (Kimi K2.5)

For each of the top N tickets (default 20, set with `--keep-picking`):
1. Fetches full ticket HTML + comment thread from Trac
2. Kimi reads the full context and returns a structured JSON verdict

**Verdict criteria:**

| Verdict | Meaning | Score threshold |
|---------|---------|----------------|
| `PICK` | Clear bug, manageable scope, nobody actively mid-PR | ≥ 40 |
| `MAYBE` | Promising but some uncertainty | ≥ 40 |
| `SKIP` | Too complex, vague, or actively being worked on | any |

Only `PICK` and `MAYBE` with score ≥ 40 become candidates (`is_candidate = True`).

**What Kimi checks:**
- Is the problem clearly defined?
- Is there an existing PR in the comments?
- Is the fix scope manageable in a single PR?
- What's the fix approach?
- How deep into Django internals does it go?

### How to guarantee picks

If scout returns 0 PICKs, the most effective levers are:

1. **Evaluate more tickets** — increase `--keep-picking`:
   ```bash
   python main.py --scout-only --keep-picking 50 -v
   ```

2. **Target a specific ticket you know is good**:
   ```bash
   python main.py --evaluate 37013 -v
   ```

3. **Use `--stop-on-pick`** to exit immediately once found:
   ```bash
   python main.py --scout-only --stop-on-pick --keep-picking 50 -v
   ```

4. **Known good tickets** from recent runs:
   - `#37013` — Omitting tzinfo in Trunc/Extract raises wrong error (PICK, 82/100)
   - `#36990` — GIS form requires Referrer header (MAYBE, 70/100)
   - `#36494` — JSONField lookup failures with expression RHS (MAYBE, 65/100)

---

## Architecture

```
                    ORCHESTRATOR (kimi-k2.5)
                    ↓ Reads state → decides actions
        ┌───────────┼───────────┬───────────┬───────────┐
        ▼           ▼           ▼           ▼           ▼
     SCOUT       PICKER      CODER      PR-MAKER   REVIEW-HANDLER
  (kimi-k2.5) (kimi-k2.5) (kimi-k2.5) (kimi-k2.5) (kimi-k2.5)
        │                     │                        │
        ▼                     ▼                        ▼
     TOOLS (Python, no AI)                          LEARNER
  TracClient | GitClient | OpenCodeClient        (kimi-k2.5)
                                                      │
                                              SKILL.md grows
```

### Agent table

| Agent | What it does |
|-------|-------------|
| Orchestrator | Reads state, decides: SCOUT / PICK_AND_CODE / SUBMIT_PR / CHECK_REVIEWS / LEARN / PAUSE |
| Scout | Phase 1 heuristic triage + Phase 2 Kimi deep eval |
| Picker | Selects best candidate from PICK/MAYBE list |
| Coder | Runs `opencode run` on the Django repo to write fix + tests |
| PR-Maker | Generates PR body, pushes branch, opens PR via `gh` |
| Review Handler | Fixes CI failures (--fix-pr), fetches PR comments, classifies, responds or fixes |
| Learner | Extracts lessons from reviewer feedback → appends to SKILL.md |
| Escalator | Sends email/webhook if circuit breaker fires or human needed |

### Model

All agents use `opencode/kimi-k2.5` via OpenCode Zen (your subscription). No Anthropic API key required.

---

## The Full Pipeline (One Cycle)

```
1. ORCHESTRATOR reads state → decides SCOUT

2. SCOUT
   Phase 0: Fetch ~900 tickets from Trac (ordered by modified desc)
   Phase 1: Heuristic filter → ~530 remain (instant, no AI)
   Phase 2: Kimi deep-evaluates top 20 → returns PICKs/MAYBEs

3. ORCHESTRATOR sees candidates → PICK_AND_CODE

4. PICKER selects best ticket by score + component history

5. CODER
   - git checkout -b ticket_XXXXX
   - opencode run writes fix + tests in Django repo
   - Self-review gate: Kimi reviews the diff
   - If REQUEST_CHANGES: fix + re-review (max 2x)
   - If APPROVE: mark ready_for_pr

6. ORCHESTRATOR → SUBMIT_PR

7. PR-MAKER
   - Generate PR body
   - git push origin ticket_XXXXX
   - gh pr create → PR live on django/django

8. Next cycle: ORCHESTRATOR → CHECK_REVIEWS

9. REVIEW-HANDLER
   - Fetch CI check failures → auto-fix + force-push (--fix-pr)
   - Fetch comments via gh CLI
   - Classify: style_fix → auto-fix | concern → ESCALATE
   - Feed to LEARNER

10. LEARNER → append lessons to SKILL.md → all agents reload
```

---

## File Structure

```
django-agent/
├── main.py                        # CLI entry point
├── .env                           # OpenCode/API keys (git-ignored)
├── venv/                          # Python virtualenv
│
├── agents/
│   ├── base.py                    # BaseAgent — wraps opencode run subprocess
│   ├── scout.py                   # ScoutAgent: heuristic triage + Kimi deep eval
│   ├── orchestrator.py            # Orchestrator + PickerAgent
│   ├── coder.py                   # CoderAgent: opencode run + self-review
│   ├── pr_maker.py                # PRMakerAgent
│   ├── review_handler.py          # ReviewHandlerAgent
│   ├── learner.py                 # LearnerAgent → SKILL.md
│   ├── escalator.py               # EscalatorAgent
│   └── memory.py                  # 3-tier memory (short/long/contextual, SQLite)
│
├── tools/
│   ├── trac_client.py             # Trac CSV + HTML scraping
│   ├── git_client.py              # git + gh CLI wrapper
│   └── claude_code_client.py      # opencode run wrapper (for Coder agent)
│
├── config/
│   └── prompts.py                 # System prompts for all agents
│
├── skills/django-contributor/
│   └── SKILL.md                   # Self-updating knowledge base
│
├── db/
│   ├── memory.sqlite              # Long-term memory (PR history, component stats)
│   └── orchestrator_state.json    # Persisted state between cycles
│
├── tests/
│   └── test_scout.py              # 34 tests for scout pipeline
│
└── logs/                          # Agent run logs
```

---

## Safety Rails

**Circuit breaker**: 3 consecutive coding failures → system pauses → escalates.

**Self-review gate**: Every diff reviewed by Kimi before PR submission. REQUEST_CHANGES → fix → re-review (max 2x). REJECT → branch abandoned.

**PR-Maker gatekeeping**: Will not submit if `ready_for_pr=False`.

**Fallback heuristics**: If Orchestrator AI call fails, deterministic fallback: submit PRs if ready → check reviews → scout. Never freezes.

---

## Running Tests

```bash
source venv/bin/activate
python -m pytest tests/test_scout.py -v
```

34 tests covering: JSON parsing, batch triage, deep eval, stop-on-pick, keep-picking, is_candidate.
