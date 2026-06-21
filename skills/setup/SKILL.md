---
name: setup
description: Wire Claude Code to Kagura Memory Cloud by writing the MCP config. Use during onboarding to connect this machine's Claude Code to a Kagura workspace.
---

# kagura setup claude — onboarding

Connect Claude Code to Kagura Memory via the refresh-aware `kagura-mcp` stdio
proxy. Thin wrapper around the installed `kagura` CLI.

## Preflight

- `kagura --version`; if missing → `uv tool install kagura-memory` (or
  `pip install kagura-memory`), then stop.
- Requires authentication. Run `kagura auth status`; if not logged in, run the
  `auth` skill first (`kagura auth login`), or pass `--profile <name>` for an
  existing profile.

## Run

```bash
kagura setup claude                  # wire the refresh-aware kagura-mcp proxy
kagura setup claude --profile <name> # bind to a named OAuth profile
```

## Consume the result

- Relay which `.mcp.json` was written. A "kagura-mcp not on PATH" message is a
  warning, not a failure.
- Suggest `kagura doctor` (the `doctor` skill) to confirm the wiring end-to-end.
