---
name: "source-command-self-maint"
description: "Audit .claude/ configuration against current codebase"
---

# source-command-self-maint

Use this skill when the user asks to run the migrated source command `self-maint`.

## Command Template

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

### 4. AGENTS.md Consistency
- Verify AGENTS.md mirrors CLAUDE.md and references match actual `.claude/` contents

## Output Format

| # | Severity | File | Issue | Proposed Fix |
|---|----------|------|-------|-------------|

After presenting findings, ask which fixes to apply.
