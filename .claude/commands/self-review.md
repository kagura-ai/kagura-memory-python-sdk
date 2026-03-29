Perform a pre-PR self-review of all changes.

## Role

You are an AI code reviewer. Review only the modified code in the diff. Prioritize high-signal feedback over quantity.

## Review the diff

```bash
git diff main...HEAD
```

If no diff, review `git diff HEAD~1`.

## Review for (in priority order)

1. **Correctness** — bugs, edge cases, broken logic, unreachable code
2. **Security** — API key leaks, secrets in logs, unsafe defaults, injection risks
3. **Error handling** — exceptions swallowed, except blocks that can raise, auth errors not surfacing, hooks/callbacks missing on early-return paths
4. **Test coverage** — every changed/added code path must have a test. Check: new branches, error paths, edge cases. codecov/patch WILL fail if you skip this.
5. **Performance** — unnecessary work, N+1 patterns, blocking calls in async context
6. **Consistency** — patterns used differently across files, naming inconsistencies with existing code
7. **Resource cleanup** — unclosed clients/connections in all code paths including error paths (use try/finally)

## Rules

- Only comment on lines that were changed in the diff
- Be specific and actionable — suggest concrete fixes
- Do not nitpick formatting (ruff handles it)
- Do not comment on unchanged code unless needed for context
- Assume linters, type checkers, and formatters are already in place

## Severity

| Marker | Meaning | Action |
|--------|---------|--------|
| `[C]` | Bug, security hole, data loss | Must fix before merge |
| `[W]` | Risk, missing guard, edge case | Should fix |
| `[I]` | Readability, minor improvement | Fix or justify |

## Output

```markdown
## Self-review: X files changed

`[C]` file:line — Description. **Fix:** concrete suggestion.

`[W]` file:line — Description. **Fix:** concrete suggestion.

### Verdict: Ready / Needs fixes
```

If no significant issues: state "Changes look good" with a brief summary.
