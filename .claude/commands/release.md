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
2. Current branch is `main`

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

Generate release notes from commits since the last tag:

```bash
gh release create vX.Y.Z --generate-notes --title "vX.Y.Z"
```

Review the auto-generated notes. If needed, edit to add a summary section at the top highlighting the key changes (new features, breaking changes, etc.).

### 10. Report

Print the new version, link to the GitHub Release, and the Actions run.
