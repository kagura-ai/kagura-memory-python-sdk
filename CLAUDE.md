# Kagura Memory SDK - Development Guide

## Overview

Python SDK for Kagura Memory Cloud. Three clients:
- `KaguraClient` — MCP client for memory and context operations
- `KaguraAgent` — LLM-powered session analysis with hooks/skills API
- `ResourceClient` — REST client for resource token management and data ingestion

## Development Workflow

See `.claude/rules/development-workflow.md` for full flow (auto-loaded).

**Key sequence**: Issue → Branch → Implement → `/quality` → `/simplify` → `/self-review` → PR → Merge

## Commands

```bash
uv run pytest                    # Tests
uv run ruff check src/ tests/   # Lint
uv run ruff format src/ tests/  # Format
uv run pyright src/              # Type check
```

**Slash commands**: `/quality` → `/simplify` → `/self-review` → PR, `/release patch|minor|major` for releases

## Branch Strategy

- Branch from `main` for every issue
- Branch naming: `{issue-number}-{type}/{description}`
- Squash merge to `main`

## References

- [Main project](https://github.com/kagura-ai/memory-cloud) — Kagura Memory Cloud server
- `.claude/rules/` — Coding standards (auto-loaded)
