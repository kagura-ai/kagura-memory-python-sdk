---
paths:
  - "src/kagura_memory/__init__.py"
  - "pyproject.toml"
---

# Version Management

## Single Source of Truth

- Version is defined ONLY in `src/kagura_memory/__init__.py` as `__version__`
- `pyproject.toml` uses `dynamic = ["version"]` with `[tool.hatch.version]` to read it
- Never set `version = "..."` directly in `pyproject.toml`

## SemVer Rules

Format: `MAJOR.MINOR.PATCH` (e.g., `0.2.3`)

| Change type | Bump | Example |
|-------------|------|---------|
| Breaking API change | MAJOR | Removing a public method |
| New feature (backward compatible) | MINOR | Adding a new CLI command |
| Bug fix, docs, refactor | PATCH | Fixing a typo in error message |

While `MAJOR` is `0`, breaking changes bump MINOR instead.

## Conventional Commits → Version

| Commit prefix | Version bump |
|---------------|-------------|
| `feat` | MINOR (or PATCH if 0.x) |
| `fix`, `perf` | PATCH |
| `BREAKING CHANGE` footer or `!` suffix | MAJOR (or MINOR if 0.x) |
| `docs`, `chore`, `refactor`, `test`, `ci` | No bump |

## Release Flow

Use `/release patch|minor|major` to:
1. Bump `__version__` in `__init__.py`
2. `uv lock`
3. Commit: `chore(release): vX.Y.Z`
4. Tag: `vX.Y.Z`
5. Push commit + tag → CI publishes to PyPI
