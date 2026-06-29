# Contributing

## Setup

```bash
git clone https://github.com/kagura-ai/kagura-memory-python-sdk.git
cd kagura-memory-python-sdk
uv sync --dev
cp .kagura.json.example .kagura.json
# Edit .kagura.json — set api_key and mcp_url
```

## Development with Claude Code

This project is developed with [Claude Code](https://claude.com/claude-code). Slash commands automate the workflow:

| Command | Description |
|---------|-------------|
| `/quality` | Run lint, format, type check, tests |
| `/simplify` | Review for code reuse and efficiency |
| `/self-review` | Pre-PR review (correctness, security, coverage, etc.) |
| `/self-maint` | Audit `.claude/` config against codebase |
| `/test` | Run the test suite |
| `/release patch\|minor\|major` | Bump version, tag, publish, create GitHub Release |
| `/kagura-memory:guide` | SDK usage reference (clients, models, CLI) |

### Recommended flow

```
# ... implement ...
/quality            # All checks pass?
/simplify           # Any improvements?
/self-review        # Ready for PR?
```

### MCP Integration

Copy `.mcp.json.example` to `.mcp.json` and fill in your credentials to use Kagura Memory as an MCP server in Claude Code.

## Workflow

1. Pick an issue from [Issue Tracker](https://github.com/kagura-ai/kagura-memory-python-sdk/issues)
2. Create a branch: `git checkout -b {issue-number}-{type}/{description} main`
3. Implement (commit frequently, Conventional Commits)
4. Run `/quality` → `/simplify` → `/self-review`
5. Push and create a PR
6. CI must pass (lint + test on Python 3.11/3.12/3.13 + codecov/patch)
7. Squash merge to `main`

## Commit Convention

```
<type>(<scope>): <subject>
```

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `ci`

## Quality Checks

```bash
uv run ruff check src/ tests/   # Lint
uv run ruff format src/ tests/  # Format
uv run pyright src/              # Type check
uv run pytest tests/ -v          # Test
```

All PRs must pass codecov/patch — every changed line needs a test.

## Code Style

- Python 3.11+, type hints required
- Google-style docstrings on public functions
- Use `VerboseLogger` instead of `print()`
- Async HTTP with `httpx.AsyncClient`
- Custom exceptions from `KaguraError` hierarchy
- Hooks/skills accept both sync and async callables
- Always close agents/clients in tests (`async with` or `try/finally`)

## SDK Architecture

| Client | Protocol | Purpose |
|--------|----------|---------|
| `KaguraClient` | MCP (JSON-RPC) | Memory + context operations |
| `KaguraAgent` | MCP + LLM | AI-powered session analysis with hooks/skills |
| `ResourceClient` | REST API | Resource token management + data ingestion |
| `FilesClient` | REST + presigned PUT | File uploads with R2 sha256 checksum binding |
| `SecretClient` | REST + age crypto | Zero-knowledge secrets — age recipient encryption, local decrypt (server stores only ciphertext) |
| `FileIngestor` | CLI + SDK | Document ingestion (PDF → memory graph + R2 archive) |
