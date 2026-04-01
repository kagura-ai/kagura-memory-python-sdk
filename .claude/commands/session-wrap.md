---
description: Summarize session findings and save to memory before ending
---

Review this conversation session and save important findings to memory.

## Steps

### 1. Scan the conversation

Identify and categorize:

| Category | What to look for |
|----------|-----------------|
| **Decisions** | Architecture choices, approach selected/rejected, trade-offs evaluated |
| **Findings** | Test results, benchmarks, performance numbers, baselines |
| **Issues** | Bugs found, limitations discovered, open questions |
| **Changes** | Code committed, PRs created/merged, releases made |
| **Ideas** | Feature requests, future improvements mentioned but not acted on |

### 2. Save to Kagura Memory (if available)

For each finding worth preserving across conversations, call `remember` with:
- `summary`: Concise conclusion (not process). Include numbers.
- `type`: decision | note | bug-fix | feature | learning
- `importance`: 0.9 for architecture decisions, 0.7-0.8 for findings/baselines, 0.5-0.6 for ideas
- `tags`: Include `category:{domain}` + key terms
- `context_id`: Use the most relevant context. Ask user if unclear.

Skip if:
- Already remembered earlier in this session
- Trivial or ephemeral (debugging steps, typos fixed)
- Derivable from git history

### 3. Save to local memory

Update `memory/` files for information useful in future conversations:
- Project state changes (version bumps, milestone progress)
- User preferences or workflow patterns observed
- Technical context that won't be in Kagura Memory

### 4. Output summary

```markdown
## Session Summary

### Decisions
- [list architecture/approach decisions made]

### Results
- [test results, benchmarks, key numbers]

### Issues Found
- [bugs, limitations, open questions]

### Changes Made
- [commits, PRs, releases]

### Remembered
- [what was saved to Kagura Memory / local memory]

### Next Steps
- [what remains to be done]
```
