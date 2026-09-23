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

Options (all optional; the defaults write the same files as before):

- `--scope project|user` — where the `kagura-memory` MCP entry goes. `project`
  (default) is `<project>/.mcp.json`. `user` serves every project on the machine
  and is the better fit for `--profile` when one profile serves them all; it is
  written through `claude mcp add-json --scope user` (without `claude` on PATH,
  setup prints that command and stops). With an API key instead of `--profile`,
  the user-scope entry sends `Bearer ${KAGURA_MCP_API_KEY}`, which Claude Code
  fills in when it connects, so the key is never on a command line or in
  `~/.claude.json`: the user sets `KAGURA_MCP_API_KEY` in the environment that
  starts Claude Code. Claude Code uses the entry from the
  strongest scope (local > project > user), so setup warns when a stronger one
  would hide the new entry, and with `-y` exits 1 without writing.
- `--guardrails off|<context-uuid>` (server v0.74.0+) and `--tool-profile <name>`
  (server v0.73.0+: `full` or `core`) — put `?guardrails=` / `?profile=` on the
  upstream MCP URL (as `kagura-mcp` args for `--profile`). `off` also removes the
  `guardrails` block from `get_context_info`: use it only once hooks (such as the
  memory-cloud `kagura-memory` plugin's) deliver guardrails. Changing either never
  forces a re-login; a re-run without them drops them, so pass them again.
- `--no-session-hook`, `--no-sync-hook`, `--no-commands` — skip the SessionStart
  recall hook (trusted memories only), the `.claude/memory` sync hook, or
  `/kagura-recall` · `/kagura-remember`. Re-running with one removes only what
  setup itself wrote. With the `kagura-memory` plugin installed, an interactive
  run offers to skip the commands it duplicates, asks separately about the
  session hook (default keep: the plugin has no automatic recall), and prints the
  plugin's `server_url` / `context_id` settings.

## Consume the result

- With `--scope user` the entry lives in `~/.claude.json` (via `claude mcp
  add-json`), not in `.mcp.json`. For an API key, relay setup's
  `KAGURA_MCP_API_KEY` note, and never print or ask for the key yourself: the
  user sets the variable. A shadowing warning names the scope that wins
  and the `claude mcp remove` command for it: relay it, and never remove an
  entry on the user's behalf.
- Relay which `.mcp.json` was written. A "kagura-mcp not on PATH" message is a
  warning, not a failure.
- Suggest `kagura doctor` (the `doctor` skill) to confirm the wiring end-to-end.

## Coexisting with the memory-cloud `kagura-memory` plugin

The memory-cloud repo ships its own Claude Code plugin, `kagura-memory` (distinct
from this `kagura-cli` plugin). It adds tool-guardrail hooks and, from memory-cloud
v0.75.0, a `/kagura-memory:setup` command that checks the MCP entry and configures
those hooks. The two overlap in three places:

- **One MCP entry named `kagura-memory`.** `kagura setup claude` writes it into the
  project `.mcp.json` (or, with `--scope user`, the user scope); the plugin's docs and
  `/kagura-memory:setup` expect an entry with that name and do not add a second one.
  Keep a single entry per scope — a second one (e.g. `claude mcp add kagura-memory ...`
  at another scope) shadows or is shadowed by it; setup checks every scope and warns.
  When the entry is missing, `/kagura-memory:setup` sends the user back to
  `kagura setup claude --profile <name>`; run that first, then the plugin command.
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
server does not also send a guardrail digest. Re-run `kagura setup claude --profile
<name> --guardrails off`: the entry then runs `kagura-mcp --guardrails off`, which adds
the query at run time, so no `--server` override (which pins the host) is needed. Pass
`--guardrails off` again on every later re-run; setup notes it when a re-run drops it.
memory-cloud v0.76.0's `/kagura-memory:setup` reads the query only from `--server` or
the profile's URL, so it may still report `guardrails=off` as missing and offer a
`--server` override: decline it. `kagura guardrails load <the plugin's context_id>` shows that context's set
as the CLI's own credential sees it (the bare command uses `.kagura.json`'s context,
which may be a different one). The hooks use their own API key, and an agent binding
filters the set per credential, so what they load can differ from that output.
