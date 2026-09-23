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

## Coexisting with the memory-cloud `kagura-memory` plugin

The memory-cloud repo ships its own Claude Code plugin, `kagura-memory` (distinct
from this `kagura-cli` plugin). It adds tool-guardrail hooks and, from memory-cloud
v0.75.0, a `/kagura-memory:setup` command that checks the MCP entry and configures
those hooks. The two overlap in three places:

- **One MCP entry named `kagura-memory`.** `kagura setup claude` writes it into the
  project `.mcp.json`; the plugin's docs and `/kagura-memory:setup` expect an entry
  with that name and do not add a second one. Keep a single entry per scope — a second
  one (e.g. `claude mcp add kagura-memory ...` at user scope) shadows or is shadowed by
  the project entry. When the entry is missing, `/kagura-memory:setup` sends the user
  back to `kagura setup claude --profile <name>`; run that first, then the plugin
  command.
- **Two SessionStart injections.** `kagura setup claude` adds `kagura recall`
  (SessionStart) and `kagura remember` (PostToolUse) hooks to `.claude/settings.json`;
  the plugin adds its own SessionStart, PreToolUse, PostToolUse and PostToolUseFailure
  guardrail hooks. They do
  different jobs (memory sync vs. guardrail delivery) and both run — neither replaces
  the other.
- **Credentials.** The plugin's hooks authenticate with a static API key entered in
  the plugin's `userConfig`. The OAuth `kagura-mcp` entry this skill writes holds no
  key, so enabling the hooks needs a separate user API key (an agent-bound key can
  read guardrails but not author them).

With the hooks active, the plugin recommends `?guardrails=off` on the MCP URL so the
server does not also send a guardrail digest. The `kagura-mcp` entry cannot carry that
query through the profile; `/kagura-memory:setup` explains the `--server` override, and
re-running `kagura setup claude --profile <name>` rewrites the entry, so re-apply it
afterwards. To check what the hooks will load, run `kagura guardrails load`.
