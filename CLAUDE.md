# Kagura Memory SDK - Development Guide

## Overview

Python SDK for Kagura Memory Cloud. Seven clients:
- `KaguraClient` — MCP client for memory and context operations
- `KaguraAgent` — LLM-powered session analysis with hooks/skills API
- `ResourceClient` — REST client for resource token management and data ingestion
- `FilesClient` — REST + presigned PUT client for file uploads with sha256 integrity binding (server v0.15.1+)
- `SecretClient` — REST client for the zero-knowledge secret store (age recipient encryption, local decrypt; `[secret]` extra, server v0.39.0+)
- `WorkspaceClient` — REST client for workspace member/invitation management (owner API key only, OAuth rejected; server v0.42.0+)
- `AgentsClient` — REST companion for agent bootstrap (`POST /api/v1/agents/{agent_id}/bootstrap`; server v0.49.0+)

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
- **File uploads**: `FilesClient.upload/download_url/delete/list` with R2 sha256 binding (server v0.15.1+); optional `binding_context_id` binds a file to an owning context for ACL + nullable `FileObject.context_id` (server v0.41.0+). Against v0.41.0 the file-id endpoints require `workspace_id` on the query, so `download_url`/`delete` take a **required** `context_id` and the PUT sends the checksum header only when the presign signed it (#226)
- **Zero-knowledge secrets**: `SecretClient` + `kagura secret` — age recipient encryption, local decrypt; private key in OS keychain via `pyrage`/`keyring` (`[secret]` extra, server v0.39.0+). Owner-only hard delete via `SecretClient.delete_secret` / `kagura secret delete` (server v0.41.0+)
- **Workspace member management**: `WorkspaceClient` + `kagura workspace member|invite` — owner-key-only (server v0.42.0+, memory-cloud#1164). Assignable roles member|admin|viewer (owner → 422); member/viewer invites require `allowed_context_ids`; `expires_in_days` presets 7/30/90/365; invitation ids are int with no status field; programmatic invite list nulls `token`/`invitation_url`
- **Owner-provisioned member keys**: `WorkspaceClient.mint_member_key/list_member_keys/revoke_member_key` + `kagura auth create-key/list-keys/revoke-key` (server v0.42.0+, memory-cloud#1165). Member/viewer targets only, never self; `expires_days` 1-3650 required; plaintext returned once (force-hidden after); revoke is soft
- **Agent bootstrap**: `KaguraClient.get_agent_bootstrap` (MCP) + `AgentsClient.bootstrap` (REST) — one-call session-start rehydration composing pinned + trusted recall + upcoming + state, fail-soft per component with top-level `degraded` (server v0.49.0+, RFC-0002 P0-3, memory-cloud#1276)
- Agent hooks/skills: `@agent.hook("before_process")`, `@agent.skill("name")`
- Ollama support: `model="ollama/qwen3:30b"` for local LLMs
- CLI: `kagura doctor`, `kagura auth login/refresh/status/list/create-key/list-keys/revoke-key`, `kagura setup claude`, `kagura ingest <file|url>`, `kagura resource setup/import/stats/schema`, `kagura files upload/list/delete/download-url`, `kagura secret keygen/put/get/grant/rotate/delete/exec`, `kagura workspace member list/add/set-role/remove`, `kagura workspace invite create/list/revoke`, `kagura context search-config`, `kagura sleep history/report/rollback`

## Branch Strategy

- Branch from `main` for every issue
- Branch naming: `{issue-number}-{type}/{description}`
- Squash merge to `main`

## References

- [Main project](https://github.com/kagura-ai/memory-cloud) — Kagura Memory Cloud server
- `.claude/rules/` — Coding standards (auto-loaded)
