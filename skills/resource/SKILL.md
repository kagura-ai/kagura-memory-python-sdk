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
kagura resource setup ...                       # provision a resource binding
kagura resource import <file>                   # bulk-import events
kagura resource stats                           # resource impact / usage
kagura resource schema                          # inferred event schema
kagura resource tokens list|create|update|revoke
```

## Consume the result

- Relay the CLI output. **Resource tokens are secrets:** when one is created it
  is shown once and not stored — surface it to the user and remind them to save
  it now. Prefer `revoke` over leaving stale tokens active.
- **Creation is plan-gated** (memory-cloud v0.68.0+): `setup` and
  `tokens create` need the workspace plan's `resources` feature, and making a
  context public needs `public_contexts`. Without it the server refuses
  (`plan_required` / `FEAT-001`): tell the user the plan must be upgraded
  instead of retrying. Existing resources, tokens and public contexts keep
  working.
