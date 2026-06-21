---
name: auth
description: Manage Kagura Memory authentication — log in (OAuth device flow), check status, refresh tokens, and list or switch profiles. Use when the user wants to sign in to Kagura, see which account/workspace is active, or manage multiple profiles.
---

# kagura auth — authentication

Drive the Kagura Memory CLI's auth commands. Thin wrapper around the installed
`kagura` CLI — no secrets are handled here.

## Preflight

`kagura --version`; if missing, tell the user to `uv tool install kagura-memory`
(or `pip install kagura-memory`) and stop.

## Run (choose by intent)

```bash
kagura auth status                 # who am I — profile, workspace, scope, expiry
kagura auth login                  # OAuth device flow (add --no-browser on SSH/headless)
kagura auth list                   # all stored profiles; default marked with *
kagura auth use <name>             # set the default profile deliberately
kagura auth refresh                # rotate the access token
kagura auth logout                 # revoke + delete a profile
```

## Consume the result

- Relay the CLI output verbatim where it matters (status table, login URL/code).
- On "not authenticated", run `kagura auth login`.
- **Multi-profile safety:** if several profiles exist, surface which profile and
  workspace are active so the user does not operate on the wrong account; use
  `kagura auth use <name>` or `--profile <name>` to pick one.
