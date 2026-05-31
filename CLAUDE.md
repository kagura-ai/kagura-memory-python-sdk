# Kagura Memory SDK - Development Guide

## Overview

Python SDK for Kagura Memory Cloud. Four clients:
- `KaguraClient` — MCP client for memory and context operations
- `KaguraAgent` — LLM-powered session analysis with hooks/skills API
- `ResourceClient` — REST client for resource token management and data ingestion
- `FilesClient` — REST + presigned PUT client for file uploads with sha256 integrity binding (server v0.15.1+)

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

**Slash commands**: `/quality` → `/simplify` → `/self-review` → PR, `/release patch|minor|major` for releases, `/kagura-memory:guide` for reference

## Key SDK Features

- Context management: `create_context`, `update_context` (with `resource_id`, `is_public`)
- Search tuning: `update_search_config` (semantic/bm25 weights, reranking)
- Resource ingestion: `setup_resource`, `ingest_event`, `ingest_events`
- Resource stats/schema: `get_resource_impact`, `get_resource_schema`
- **File uploads**: `FilesClient.upload/download_url/delete/list` with R2 sha256 binding (server v0.15.1+)
- Agent hooks/skills: `@agent.hook("before_process")`, `@agent.skill("name")`
- Ollama support: `model="ollama/qwen3:30b"` for local LLMs
- CLI: `kagura auth login/refresh/status`, `kagura setup claude`, `kagura doctor` (setup/auth/MCP/connectivity + LLM-provider key/model diagnostics), `kagura ingest <file|url>`, `kagura resource setup/import/stats/schema`, `kagura files upload/list/delete/download-url`, `kagura context search-config`, `kagura sleep history/report/rollback`

## Branch Strategy

- Branch from `main` for every issue
- Branch naming: `{issue-number}-{type}/{description}`
- Squash merge to `main`

## References

- [Main project](https://github.com/kagura-ai/memory-cloud) — Kagura Memory Cloud server
- `.claude/rules/` — Coding standards (auto-loaded)
