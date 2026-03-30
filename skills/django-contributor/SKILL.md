# Django Contribution Skill

Lessons learned from PR reviews. This file is automatically maintained by the Learner agent.
New entries are appended after every PR review cycle. Periodically consolidated to remove duplicates.

---

## Baseline Knowledge (seeded from Django contribution docs)

### Commit Messages
- **[process_pattern]** [all] Commit messages MUST follow format: `Fixed #XXXXX -- Description.` with the ticket number and a clear one-line summary. (confidence: high, via Django docs)

### Test Conventions
- **[test_pattern]** [all] Use `self.assertIs(value, True)` and `self.assertIs(value, False)` instead of `self.assertTrue(value)` or `self.assertFalse(value)` for boolean assertions. (confidence: high, via Django test style guide)
- **[test_pattern]** [all] Use `self.assertQuerySetEqual()` for comparing QuerySets, not manual list comparisons. (confidence: high, via Django test style guide)
- **[test_pattern]** [all] Every bug fix PR MUST include a regression test that fails without the fix and passes with it. (confidence: high, via Django docs)

### Code Style
- **[coding_pattern]** [all] Follow PEP 8 strictly. Django also enforces specific import ordering: stdlib → third-party → Django → local app. (confidence: high, via Django docs)
- **[coding_pattern]** [all] Use `ruff check` and `ruff format` before committing. (confidence: high, via Django docs)

### PR Process
- **[process_pattern]** [all] All PRs must target the `main` branch unless explicitly backporting. (confidence: high, via Django docs)
- **[process_pattern]** [all] Update documentation in `docs/` directory (RST format) if the fix changes any user-facing behavior. (confidence: high, via Django docs)
- **[process_pattern]** [all] Run the targeted test suite before submitting: `python -m pytest tests/<module>/ -x`. (confidence: high, via Django docs)

### Trac Ticket Workflow
- **[process_pattern]** [all] Comment on the Trac ticket before starting work to signal your intent. (confidence: high, via Django docs)
- **[process_pattern]** [all] After opening a PR, update the Trac ticket: set `Has patch` flag and add the PR link in a comment. (confidence: high, via Django docs)
- **[process_pattern]** [all] Never mark your own ticket as "Ready for checkin" — wait for a reviewer to do it. (confidence: high, via Django docs)

---

## Learned Patterns (auto-populated by Learner agent)

*Entries below are added automatically after each PR review cycle.*
