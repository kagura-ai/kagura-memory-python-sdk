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

**Slash commands**: `/quality` → `/simplify` → `/self-review` → PR, `/release patch|minor|major` for releases, `/onboarding` for setup, `/kagura-guide` for reference

## Key SDK Features

- Context management: `create_context`, `update_context` (with `resource_id`, `is_public`)
- Search tuning: `update_search_config` (semantic/bm25 weights, reranking)
- Resource ingestion: `setup_resource`, `ingest_event`, `ingest_events`
- Resource stats/schema: `get_resource_impact`, `get_resource_schema`
- Agent hooks/skills: `@agent.hook("before_process")`, `@agent.skill("name")`
- Ollama support: `model="ollama/qwen3:30b"` for local LLMs
- CLI: `kagura resource setup/import/stats/schema`, `kagura context search-config`

## Branch Strategy

- Branch from `main` for every issue
- Branch naming: `{issue-number}-{type}/{description}`
- Squash merge to `main`

## References

- [Main project](https://github.com/kagura-ai/memory-cloud) — Kagura Memory Cloud server
- `.claude/rules/` — Coding standards (auto-loaded)
