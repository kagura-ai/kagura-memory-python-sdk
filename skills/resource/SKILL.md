---
name: resource
description: Manage Kagura resource tokens and push external data (Slack/CI/CRM events) into a context. Use for resource token create/list/update/revoke, resource setup/import, and impact/schema stats.
---

# kagura resource — resource tokens & external ingestion

Drive the `kagura resource` command group. Thin wrapper around the installed
`kagura` CLI.

## Preflight

- `kagura --version`; if missing → `uv tool install kagura-memory` (or
  `pip install kagura-memory`), then stop.
- Requires authentication. Run `kagura auth status`; if not authed, run the
  `auth` skill first.

## Run (choose by intent)

```bash
kagura resource setup -r <id> [--name <ctx>]    # provision a resource binding
kagura resource import <file>                   # bulk-import events
kagura resource stats                           # resource impact / usage
kagura resource schema                          # inferred event schema
kagura resource tokens list|create|update|revoke
```

## Consume the result

- Relay the CLI output. **Resource tokens are secrets:** when one is created it
  is shown once and not stored — surface it to the user and remind them to save
  it now. Prefer `revoke` over leaving stale tokens active.
- `setup` names the new context after the resource id unless `--name` is
  given (lowercase letters, digits, hyphens and underscores; max 100
  characters). Pass `--name` when the id is longer than that, or when the
  server refuses with `Context '<name>' already exists in this workspace` — a
  context of that name is already there. `--summary` is ignored — the
  server has none to set — so set one afterwards with
  `kagura context update <context_id> --summary ...` (context owner only).
- **Creation is plan-gated**: `setup` and `tokens create` need the workspace
  plan's `resources` feature, and making a context public needs
  `public_contexts` (memory-cloud v0.68.0+, XL only by default; earlier servers
  gated them on the plan's shared contexts and token cap). Without it the
  server refuses (`plan_required` / `FEAT-001`): tell the user the plan does
  not allow it instead of retrying, and suggest an upgrade only when the
  refusal names a `required_plan`. A `tokens create` refused at the
  active-token cap ("Token limit reached"; `QUOTA-001` with
  `quota_type: resource_tokens` on v0.75.0+) means revoking unused tokens
  first. Existing resources, tokens and public contexts keep working.
