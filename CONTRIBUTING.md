# Contributing

## Setup

```bash
git clone https://github.com/kagura-ai/kagura-memory-python-sdk.git
cd kagura-memory-python-sdk
uv sync --dev
```

## Development with Claude Code

This project is optimized for [Claude Code](https://claude.com/claude-code) development. Slash commands automate the entire workflow:

| Command | Description |
|---------|-------------|
| `/workflow` | Show current state and next step |
| `/quality` | Run lint, format, type check, tests |
| `/simplify` | Review for code reuse and efficiency |
| `/self-review` | Pre-PR self-review with severity levels |
| `/release patch\|minor\|major` | Bump version, tag, publish |

### Recommended flow

```
/workflow          # Where am I? What's next?
# ... implement ...
/quality           # All checks pass?
/simplify          # Any improvements?
/self-review       # Ready for PR?
```

### MCP Integration

Copy `.mcp.json.example` to `.mcp.json` and fill in your credentials to enable Kagura Memory as an MCP server in Claude Code.

## Workflow

1. Pick an issue from [Issue Tracker](https://github.com/kagura-ai/kagura-memory-python-sdk/issues)
2. Create a branch: `git checkout -b {issue-number}-{type}/{description} main`
3. Implement (commit frequently, Conventional Commits)
4. Push and create a PR
5. CI must pass (lint + test on Python 3.11/3.12/3.13)
6. Squash merge to `main`

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

## Code Style

- Python 3.11+, type hints required
- Google-style docstrings on public functions
- Use `VerboseLogger` instead of `print()`
- Async HTTP with `httpx.AsyncClient`
- Custom exceptions from `KaguraError` hierarchy
