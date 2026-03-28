Perform a pre-PR self-review of all changes.

## Severity scale

| Level | Marker | Meaning | Action |
|-------|--------|---------|--------|
| Critical | `[C]` | Security hole, data loss, crash | Must fix |
| Warning | `[W]` | Bug risk, missing guard | Should fix |
| Info | `[I]` | Style, readability | Fix or justify |

## Mindset
- Review as if seeing the code for the first time
- Hunt for problems — don't wait for them to appear
- If problem found, provide the fix (not just the complaint)

## Steps

### 1. Review the diff
```bash
git diff main...HEAD
```

### 2. Security
- No hardcoded API keys, passwords, or tokens
- HTTPS enforcement on MCP URLs
- API keys not stored as instance attributes
- No secrets written to os.environ

### 3. Error handling
- Auth errors (KaguraAuthError) always surface to caller
- Rate limit detection uses typed exceptions (not string matching)
- External HTTP calls have timeouts

### 4. Coding standards
- Type hints on all function signatures
- No `print()` (use VerboseLogger)
- async/await patterns correct

### 5. Breaking changes
- New parameters are optional (backward compatible)
- Return types unchanged

### 6. Testing
- New code has tests
- Existing tests pass: `uv run pytest tests/ -v`

## Output format

```markdown
## Self-review results

### Files changed: X files

### Findings

`[C]` file:line - Description
  **Fix:** ...

`[W]` file:line - Description
  **Fix:** ...

### Verdict: Ready for PR / Needs fixes
```
