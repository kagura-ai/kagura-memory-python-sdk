---
description: Bump version, commit, tag, and push a release
arguments:
  - name: level
    description: "patch, minor, or major"
    required: true
---

Release the SDK with a version bump.

## Prerequisites

Before releasing, ensure:
1. `/quality` has been run and passes
2. `/self-maint` has been run (no stale .claude/ config)
3. Current branch is `main`

## Steps

### 1. Validate preconditions

- Argument: `$ARGUMENTS` (must be `patch`, `minor`, or `major`)
- Verify current branch is `main` (`git branch --show-current`). Abort if not on main.

### 2. Read current version

Read `src/kagura_memory/__init__.py` and extract `__version__`.

### 3. Calculate new version

Apply SemVer bump:
- `patch`: 0.2.2 → 0.2.3
- `minor`: 0.2.2 → 0.3.0
- `major`: 0.2.2 → 1.0.0

### 4. Update version

Edit `__version__` in `src/kagura_memory/__init__.py` to the new version.

### 4b. Update MIN_SERVER_VERSION (if needed)

Check if `MIN_SERVER_VERSION` in `src/kagura_memory/client.py` needs updating for this release (e.g., if new methods require a newer server version). Update if necessary.

### 5. Update lock file

```bash
uv lock
```

### 6. Verify build

```bash
uv build
```

### 7. Commit and tag

```bash
git add src/kagura_memory/__init__.py uv.lock
git commit -m "chore(release): vX.Y.Z"
git tag vX.Y.Z
```

### 8. Push

```bash
git push && git push --tags
```

This triggers the publish workflow to deploy to PyPI.

### 9. Create GitHub Release with notes

1. Run `git log --oneline vPREVIOUS..vX.Y.Z` to review all commits in this release.
2. Write release notes in the following format, then create the release with `gh release create`:

```markdown
## Highlights

One-line summary of the release theme.

### New features

- **Feature name** — Description (#issue)
  ```python
  # Short code example showing usage
  ```

### Improvements

- Bullet list of non-feature changes (perf, validation, refactor, etc.)

### Bug fixes

- Bullet list of fixes (omit section if none)

## What's Changed
* PR title by @author in URL (auto-generated section)
```

Rules:
- Include code examples for every new public API (method, parameter, CLI command)
- Omit empty sections (e.g., skip "Bug fixes" if there are none)
- Keep descriptions concise — one line per item, details go in code examples
- Reference issue/PR numbers with `#N`
- **Breaking changes check**: If any commit contains breaking API/MCP changes (field renames, tool removals, endpoint changes), add a `## Migration` section at the top of the release notes with:
  - What changed (before → after)
  - What clients/users need to update
  - Example of the new usage

### 10. Report

Print the new version, link to the GitHub Release, and the Actions run.
