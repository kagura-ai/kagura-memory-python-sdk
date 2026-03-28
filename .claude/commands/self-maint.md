---
description: Audit .claude/ configuration against current codebase
---

Audit the `.claude/` directory against the current codebase state.

## Audit Steps

### 1. Rules Audit (.claude/rules/)
- Verify referenced imports, packages, patterns still exist in code
- Check if new patterns have emerged that rules don't cover

### 2. Commands Audit (.claude/commands/)
- Verify CLI commands work (uv run pytest, uv run ruff, etc.)
- Check file paths and tool references are valid

### 3. Settings Audit (.claude/settings.json)
- Verify hook commands work
- Check permission allowlist covers dev workflow

### 4. CLAUDE.md Consistency
- Verify references match actual .claude/ contents

## Output Format

| # | Severity | File | Issue | Proposed Fix |
|---|----------|------|-------|-------------|

After presenting findings, ask which fixes to apply.
