---
name: memory
description: Store and search memories with the `kagura` CLI — `kagura remember` (with `--tags`, `--details`, `--location`) and `kagura recall` (`--trusted-only`, `--rerank/--no-rerank`), plus `reference`, `update-memory` and `forget`. Use when the user wants to save a fact, decision or note to Kagura Memory, or search what is already stored.
---

# kagura memory — remember and recall

Drive the core memory verbs — every command prints one JSON object to stdout.
Thin wrapper around the installed `kagura` CLI.

## Preflight

- `kagura --version`; if missing → `uv tool install kagura-memory` (or
  `pip install kagura-memory`), then stop.
- Requires authentication. Run `kagura auth status`; if not authed, run the
  `auth` skill first.
- Requires a target context: `-c/--context-id <id>` on the command, else the
  `context_id` in `.kagura.json` (the working directory's, else
  `~/.kagura.json`), else `KAGURA_CONTEXT_ID` when neither file exists.
  Without any of these the CLI stops with "context_id required" — ask the user
  which context or run `kagura context list`. Never read `.kagura.json`
  yourself: it may hold an API key.

## Run

```bash
# store
kagura remember -s "FastAPI DI pattern" --content "Use Depends() for ..."
kagura remember -s "OAuth2 setup" --content "..." --tags "auth,oauth" -i 0.8
kagura remember -s "Coffee with Sato" --content "..." --location "35.68,139.76,Tokyo HQ"
kagura remember -s "Site visit" --content "..." \
  --details '{"location": {"lat": 35.68, "lon": 139.76}, "client": "acme"}'

# search
kagura recall "FastAPI dependency injection"              # top 5 (default)
kagura recall "error handling pattern" -k 10 -c <ctx>
kagura recall "project context" --trusted-only             # reads that steer your next actions
kagura recall "latency-sensitive lookup" --no-rerank

# one memory in full / revise / delete
kagura reference -m <uuid>
kagura update-memory -m <uuid> -s "updated summary" --tags "auth,oauth"
kagura update-memory -m <uuid> --location "35.68,139.76,Tokyo HQ" --merge-details   # keeps the other details keys
kagura update-memory -m <uuid> --details '{"location": {"lat": 35.68, "lon": 139.76}, "client": "acme"}'   # replaces them all
kagura forget -m <uuid>                                    # soft delete, one memory
kagura forget -q "outdated test data" -k 5                 # bulk delete by search
```

Guardrails for `--details` / `--location` (`remember` and `update-memory`):

- `lat` / `lon` inside `--details` must be JSON **numbers** — `{"lat": 35.68}`,
  never `{"lat": "35.68"}`. The server rejects string-typed coordinates with a
  422 by design.
- `--location` takes exactly `lat,lon` or `lat,lon,label` (two or three
  comma-separated fields); it is a shorthand for `details.location`.
- `--location` together with a `location` key inside `--details` is a usage
  error, not a silent merge — pick one.
- A blank or whitespace-only `--details`, `--location` or `--tags` is treated
  as if the flag were omitted — nothing is sent, so an empty shell variable is
  safe; on `update-memory` the stored value is left unchanged (the CLI cannot
  clear tags). Invalid JSON, or JSON that is not an object, in `--details` is a
  usage error.
- On `update-memory`, `--details` / `--location` **replace** the memory's
  `details` wholesale — the server does not deep-merge — so a bare `--location`
  drops every other key (the memory keeps its place on the WHERE axis, but
  loses the rest) and `--details '{}'` clears them. Add `--merge-details` to
  keep the keys you do not mention: it reads the memory first (`reference`, so
  two calls, not one atomic update) and shallow-merges the top-level keys. It
  needs `-m`, cannot remove a key, and writes nothing when the server's bounded
  `reference` reply left `details` out (memory-cloud 0.78.0+ on a large
  memory) — then re-send the complete object with `--details` and without
  `--merge-details`.

Search options:

- `--trusted-only` excludes connector-ingested / external memories. Use it for
  reads that steer the agent's next actions (as the SessionStart hook does).
- `--rerank` / `--no-rerank` is tri-state: no flag follows the context's search
  config; `--rerank` applies only when the context enables reranking;
  `--no-rerank` always skips it.

## Consume the result

- `remember` → `memory_id` (keep it: `reference`, `update-memory` and `forget`
  take it as `-m`) and `scope`. An optional `lint` list (`[{code, hint}]`, e.g.
  `summary_short`, `no_tags`) means the memory was stored but will recall
  poorly — offer an `update-memory` with a better summary or tags.
- `recall` → `results` (ranked; each carries `memory_id` and `summary`),
  `count`, `confidence` and `related_tags` (`[{tag, count}]`). Optional keys
  are absent when they do not apply: `degraded` / `degraded_reason` means the
  semantic half of search was unavailable and the results are keyword-only,
  so an empty result means "search impaired", not "nothing stored" — retry
  later.
- `reference` → the full memory under `memory` (`summary`, `content`, …).
- `update-memory` → `memory_id`, `operation`, `re_embedded`, `scope`, plus the
  same optional `lint` as `remember` (re-check it after fixing a hint);
  `supersede_candidate_dismissed` is the only confirmation that
  `--dismiss-supersede-candidate` applied — absent when the memory had no live
  candidate or the server predates v0.65.0.
- `forget` → `deleted_count` and `memory_ids`. `deleted_count` is the outcome,
  not the request: `0` after `-m` means the server skipped it (a memory the
  credential may not delete, such as a tool guardrail) — report it as not
  deleted; a `-q` sweep can delete fewer than `-k` for the same reason.
  Deletion is soft and stays recoverable until the deployment's retention
  window passes; `-q` is refused while recall is degraded. Confirm with the
  user before a `-q` bulk delete.
