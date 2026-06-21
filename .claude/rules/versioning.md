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
2. Sync the plugin manifests (`.claude-plugin/plugin.json`, `.claude-plugin/marketplace.json`) to the same version — they ship from this repo and `tests/test_plugin.py` fails CI if they drift from `__version__`
3. `uv lock`
4. Commit: `chore(release): vX.Y.Z`
5. Tag: `vX.Y.Z`
6. Push commit + tag → CI publishes to PyPI

### Test release

Manual trigger via GitHub Actions → `Publish to PyPI` → Run workflow → `testpypi`

## Release Infrastructure (configured)

- **PyPI Trusted Publisher**: `kagura-ai/kagura-memory-python-sdk` / `publish.yml` / environment `pypi`
- **TestPyPI Trusted Publisher**: same, environment `testpypi`
- **GitHub Environments**: `pypi` and `testpypi` configured in repo settings
- No API tokens needed — uses OIDC authentication
