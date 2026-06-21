---
name: doctor
description: Diagnose the local Kagura Memory CLI setup (install, auth, MCP wiring). Use before other kagura skills, or whenever a kagura command reports a blocked or misconfigured environment.
---

# kagura doctor — environment diagnosis

Run the Kagura Memory CLI's self-check and report what must be fixed. This skill
is a thin wrapper — it shells out to the already-installed `kagura` CLI and never
reimplements the checks.

## Preflight

Confirm the CLI is installed: `kagura --version`. If it reports "command not
found", tell the user to install it — `uv tool install kagura-memory` (or
`pip install kagura-memory`) — and stop. This skill is instructions only; it
cannot install the package itself.

## Run

```bash
kagura doctor
```

## Consume the result

- **Exit 0** → the environment is healthy; relay the summary.
- **Non-zero** → relay exactly what `doctor` reports is wrong and its suggested
  fix. The common ones are "not authenticated" → run the `auth` skill
  (`kagura auth login`) and "Claude Code not wired" → run the `setup` skill
  (`kagura setup claude`). Do not guess beyond what `doctor` prints.
