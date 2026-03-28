---
paths:
  - "**"
---

# Development Workflow

## Flow

1. **Start**: `gh issue view <number>` → understand the task
2. **Branch**: `git checkout -b {issue-number}-{type}/{description} main`
3. **Implement**: Write code keeping self-review criteria in mind from the start
4. **Quality**: Run `/quality` (lint, type-check, tests)
5. **Simplify**: Run `/simplify` — review for code reuse, quality, and efficiency
6. **Self-review**: Run `/self-review` — fix all `[C]` and `[W]` findings
7. **PR**: Create PR linking the issue. Only when user explicitly asks.
8. **Merge**: Only after user approval. Never auto-merge.

## Commit Discipline

- Commit frequently with meaningful messages
- Follow Conventional Commits: `<type>(<scope>): <subject>`
- Each commit should be atomic — one logical change per commit
- Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`

## PR Rules

- Run `/quality` → `/simplify` → `/self-review` before every PR
- PR title: Under 70 characters, Conventional Commits format
- Squash merge to `main`
- Delete branch after merge
- Never skip self-review before merge
