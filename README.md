# Django Open-Source Contribution Multi-Agent System

An autonomous AI agent system that discovers Django tickets, writes patches, submits PRs, responds to reviewer comments, and learns from every interaction — getting smarter with every cycle.

---

## Prerequisites

You need these already set up (you have all of these):

- Python 3.12+
- Django repo cloned and forked on GitHub
- `gh` CLI authenticated (`gh auth status`)
- `claude` CLI installed (`npm install -g @anthropic-ai/claude-code`)
- Anthropic API key
- Django CLA signed at https://www.djangoproject.com/foundation/cla/

---

## Setup

### 1. Extract and install dependencies

```bash
tar xzf django-agent.tar.gz
cd django-agent
pip install requests beautifulsoup4
```

### 2. Environment variables

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
# Required
export ANTHROPIC_API_KEY='sk-ant-...'

# For escalation notifications (optional but recommended)
export MAILGUN_API_KEY='your-key'
export MAILGUN_DOMAIN='your-domain.mailgun.org'
export ESCALATION_EMAIL='you@email.com'

# For WhatsApp/Slack urgent alerts (optional)
export ESCALATION_WEBHOOK_URL='https://your-webhook'
```

### 3. Verify your Django repo

```bash
cd ~/your-path-to/django
git remote -v
```

You should see:

```
origin    git@github.com:YOUR_USERNAME/django.git (push)
upstream  git@github.com:django/django.git (fetch)
```

If you only have `origin` pointing to your fork, add upstream:

```bash
git remote add upstream git@github.com:django/django.git
```

Make sure you're clean and up to date:

```bash
git checkout main
git pull upstream main
```

### 4. Verify tools

```bash
gh auth status          # Should show your GitHub account
claude --version        # Should show Claude Code version
```

---

## gh CLI vs GitHub MCP — Which One Does This System Use?

**Short answer: `gh` CLI for the automation pipeline, GitHub MCP stays for Claude Code sessions.**

The system uses `gh` CLI (`tools/git_client.py`) for all scripted GitHub operations: creating PRs, fetching comments, posting responses. This is the right choice because:

- `gh` runs as a subprocess from Python — no server dependency, no connection state to manage
- Predictable error handling via exit codes and stderr
- Works identically in headless/cron/background mode
- Your PAT auth is already configured via `gh auth login`

GitHub MCP stays useful inside Claude Code (`claude -p`) sessions — when the Coder agent runs, Claude Code can use the GitHub MCP server to browse PRs, read code on GitHub, and look up related issues. You already have this configured with `--scope user` and SAML SSO for `bvs-xiangqi`.

**You don't need to change anything.** Both work together. The Python tools layer uses `gh`, the Claude Code sessions can access GitHub MCP if available.

---

## Usage

### Quick test — Scout only (reads Trac, no code changes)

```bash
cd ~/django-agent
python main.py --scout-only -v
```

Costs ~$0.01. Hits Trac, AI-evaluates tickets, shows you candidates. If this works, everything is connected.

### Evaluate a specific ticket

```bash
python main.py --evaluate 35421 -v
```

Fetches the ticket, reads the full comment thread, gives you AI's assessment: verdict, score, complexity, fix approach, risks.

### Single full cycle (scout → pick → code → self-review → PR)

```bash
python main.py --once \
  --repo ~/your-path-to/django \
  --fork YOUR_USERNAME/django \
  --budget-cycle 0.50 \
  -v
```

Start with `--budget-cycle 0.50` for the first run. The system will:

1. Scout Trac for easy-picking tickets
2. AI picks the best candidate
3. Claude Code writes the fix + tests on a new branch
4. Self-review gate checks the diff (Opus)
5. If approved → pushes branch and opens PR on `django/django`

### Continuous mode (hourly)

```bash
python main.py --watch 60 \
  --repo ~/your-path-to/django \
  --fork YOUR_USERNAME/django \
  --budget-daily 5.00 \
  -v
```

### Background daemon

```bash
nohup python main.py --watch 60 \
  --repo ~/your-path-to/django \
  --fork YOUR_USERNAME/django \
  --budget-daily 5.00 \
  -v >> logs/agent.log 2>&1 &

# Check on it
tail -f logs/agent.log

# Stop it
kill $(pgrep -f "main.py --watch")
```

### All CLI options

```
--once                  Single orchestration cycle
--watch MIN             Continuous, every MIN minutes
--scout-only            Just discover tickets (no coding)
--evaluate TICKET_ID    Evaluate one specific ticket
--repo PATH             Path to your Django clone
--fork USER/django      Your GitHub fork name
--skill PATH            Path to SKILL.md (default: skills/django-contributor/SKILL.md)
--state PATH            Path to state file (default: db/orchestrator_state.json)
--budget-cycle USD      Max spend per cycle (default: $2.00)
--budget-daily USD      Max spend per day (default: $10.00)
-v                      Verbose logging
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR (Sonnet)                       │
│  AI reads system state → decides actions → dispatches agents  │
│                                                                │
│  Actions: SCOUT | PICK_AND_CODE | SUBMIT_PR | CHECK_REVIEWS   │
│           LEARN | PAUSE | ESCALATE                             │
└───────┬────────┬──────────┬──────────┬──────────┬────────────┘
        │        │          │          │          │
        ▼        ▼          ▼          ▼          ▼
   ┌────────┐┌───────┐┌─────────┐┌──────────┐┌──────────┐
   │ SCOUT  ││PICKER ││ CODER   ││PR-MAKER  ││ REVIEW   │
   │Sonnet  ││Sonnet ││Opus +   ││Sonnet    ││ HANDLER  │
   │SKILL.md││       ││claude -p││          ││Sonnet    │
   │        ││       ││SKILL.md ││          ││          │
   └───┬────┘└───┬───┘└────┬────┘└────┬─────┘└────┬─────┘
       │         │         │          │            │
       │         │         │          │            ▼
       │         │         │          │     ┌──────────┐
       │         │         │          │     │ LEARNER  │
       │         │         │          │     │ Sonnet   │──→ SKILL.md grows
       │         │         │          │     │ SKILL.md │    (all agents reload)
       │         │         │          │     └──────────┘
       │         │         │          │
       ▼         ▼         ▼          ▼
   ┌─────────────────────────────────────────────────┐
   │              TOOLS (Python, no AI)               │
   │  TracClient  │  GitClient  │  ClaudeCodeClient   │
   │  (requests)  │  (gh CLI)   │  (claude -p)        │
   └─────────────────────────────────────────────────┘
                          │
                          ▼
                   ┌──────────┐
                   │ESCALATOR │──→ Email (Mailgun)
                   │ Sonnet   │──→ Webhook (WhatsApp/Slack)
                   └──────────┘
```

### Agent Details

| Agent | Model | SKILL.md | max_tokens | Cost/call | What it does |
|-------|-------|----------|------------|-----------|-------------|
| Orchestrator | Sonnet | No | 1024 | ~$0.005 | Reads state, decides what actions to take |
| Scout (triage) | Sonnet | Yes | 2048 | ~$0.01 | Batch-triages all ticket summaries |
| Scout (deep) | Sonnet | Yes | 768 | ~$0.005 | Deep-evaluates shortlisted tickets |
| Picker | Sonnet | No | 512 | ~$0.003 | Selects best ticket from candidates |
| Coder | Opus | Yes | 1024 | ~$0.05 | Self-review gate for the diff |
| Coder (claude -p) | — | — | — | ~$0.10 | Actually writes the fix + tests |
| PR-Maker | Sonnet | No | 512 | ~$0.001 | Generates PR body text |
| Review Handler | Sonnet | No | 512 | ~$0.003 | Classifies + responds to PR comments |
| Learner | Sonnet | Yes | 1024 | ~$0.005 | Extracts lessons from reviews |
| Escalator | Sonnet | No | 512 | ~$0.002 | Composes human notifications |

**Estimated cost per full PR: ~$0.16**
**At 2 PRs/day: ~$0.32/day**

### Token Optimization

Only 3 of 7 agents load SKILL.md (Scout, Coder, Learner — the ones that need Django patterns). The other 4 have `use_skill=False` to save input tokens.

Every agent call checks the global `TokenBudget` before firing. If cycle or daily limit is hit, the call is blocked and the orchestrator skips remaining actions.

`max_tokens` is tuned per agent. PR-Maker and Picker use 512 (their JSON responses are ~200 tokens). Scout deep-eval uses 768. Nobody uses the 4096 default.

---

## The Full Pipeline (One Cycle)

```
1. ORCHESTRATOR reads system state
   → AI decides: "No candidates, need to SCOUT"

2. SCOUT
   → TracClient fetches easy-picking tickets from Trac (tool)
   → AI batch-triages summaries: 50 tickets → 12 PROMISING (cheap Sonnet call)
   → For each of the 12: fetch full ticket + comments (tool)
   → AI deep-evaluates each: 4 PICK, 3 MAYBE, 5 SKIP

3. ORCHESTRATOR sees candidates
   → AI decides: "We have 4 PICKs, do PICK_AND_CODE"

4. PICKER
   → AI compares 4 candidates vs our history
   → Selects #35421 (Admin bug, score 82/100, simple complexity)

5. CODER
   → git checkout -b ticket_35421 (tool)
   → Builds prompt: ticket context + SKILL.md + fix approach
   → claude -p writes the fix + tests (Claude Code, ~10 min)
   → git diff main (tool)
   → Self-review gate: Opus reviews the diff as a Django reviewer
   → Verdict: APPROVE (85/100) or REQUEST_CHANGES → fix → re-review (max 2x)

6. ORCHESTRATOR sees ready_for_pr
   → AI decides: "Patch ready, do SUBMIT_PR"

7. PR-MAKER
   → AI generates PR body (1 Sonnet call, 512 tokens)
   → git push origin ticket_35421 (tool)
   → gh pr create (tool)
   → PR is live on django/django

8. ORCHESTRATOR on next cycle sees open PRs
   → AI decides: "CHECK_REVIEWS"

9. REVIEW HANDLER
   → Fetches PR comments via gh CLI (tool)
   → AI classifies each: style_fix → auto-fix + respond
   → AI classifies: architectural_concern → ESCALATE
   → Every comment → LEARNER extracts lessons → SKILL.md updated
   → All agents with SKILL.md reload it

10. LEARNER
    → Merges/rejections → extract patterns
    → "ALWAYS use self.assertIs() for booleans" added to SKILL.md
    → Next PR's Coder reads it → gets it right the first time
```

---

## File Reference

```
django-agent/
├── main.py                              # CLI entry point
│
├── agents/                              # AI AGENTS (Claude-powered)
│   ├── base.py                          # BaseAgent + TokenBudget + cost tracking
│   ├── scout.py                         # ScoutAgent: 2-phase ticket discovery
│   ├── orchestrator.py                  # Orchestrator + PickerAgent
│   ├── coder.py                         # CoderAgent: claude -p + self-review gate
│   ├── pr_maker.py                      # PRMakerAgent: generates body, pushes, opens PR
│   ├── review_handler.py               # ReviewHandlerAgent: classifies + responds to comments
│   ├── learner.py                       # LearnerAgent: extracts lessons → SKILL.md
│   └── escalator.py                     # EscalatorAgent: email/webhook notifications
│
├── tools/                               # TOOLS (Python, no AI)
│   ├── trac_client.py                   # Trac API: fetch tickets, parse HTML
│   ├── git_client.py                    # Git + GitHub: branches, PRs, comments via gh CLI
│   └── claude_code_client.py            # Claude Code: wraps claude -p headless mode
│
├── config/
│   └── prompts.py                       # System prompts for all 7 agent roles
│
├── skills/
│   └── django-contributor/
│       └── SKILL.md                     # Grows automatically via Learner agent
│
├── db/
│   └── orchestrator_state.json          # Persisted state (created on first run)
│
├── logs/                                # Agent run logs
└── tests/
```

---

## Safety Rails

**Circuit breaker**: 3 consecutive coding failures → system pauses → escalates to you.

**Token budget**: Every API call checked against cycle ($2) and daily ($10) limits. Configurable via `--budget-cycle` and `--budget-daily`. If blown, remaining actions in the cycle are skipped.

**Self-review gate**: No PR gets submitted without an Opus-level review of the diff. If the review says REQUEST_CHANGES, the Coder fixes issues and re-reviews (max 2 iterations). If REJECT, the branch is abandoned.

**PR-Maker gatekeeping**: Refuses to submit any `CodingResult` where `ready_for_pr=False` (tests failed, or self-review didn't approve).

**Fallback heuristics**: If the Orchestrator's AI decision call fails, deterministic heuristics kick in: submit PRs if ready, check reviews if PRs are open, scout if nothing else to do. The system never freezes.

**Escalation tiers**: Tier 1 (email) for info. Tier 2 (email) for attention. Tier 3 (email + webhook) for urgent. You control when and how you're notified.

---

## Recommended First Run

```bash
# Step 1: Verify Trac access works
python main.py --scout-only -v

# Step 2: Evaluate a ticket you know about
python main.py --evaluate TICKET_NUMBER -v

# Step 3: First real cycle with low budget
python main.py --once \
  --repo ~/your-path-to/django \
  --fork YOUR_USERNAME/django \
  --budget-cycle 0.50 \
  -v

# Step 4: Check the state file
cat db/orchestrator_state.json | python -m json.tool

# Step 5: If everything looks good, go continuous
python main.py --watch 60 \
  --repo ~/your-path-to/django \
  --fork YOUR_USERNAME/django \
  -v
```
