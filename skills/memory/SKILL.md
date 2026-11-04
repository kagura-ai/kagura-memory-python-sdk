---
name: memory
description: Store and search memories with the `kagura` CLI — `kagura remember` (with `--tags`, `--details`, `--location`) and `kagura recall` (`--trusted-only`, `--rerank/--no-rerank`, `--verbosity`, `--full`, `--max-chars`), plus `reference`, `update-memory` and `forget` (`--ids` for several), and `bootstrap` / `changes` to open a session. Use when the user wants to save a fact, decision or note to Kagura Memory, or search what is already stored.
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
kagura recall "deploy decisions" --verbosity full          # uncut summaries + tags
kagura recall "deploy runbook" --full 1                    # content on the top hit only

# one memory in full / revise / delete
kagura reference -m <uuid>
kagura update-memory -m <uuid> -s "updated summary" --tags "auth,oauth"
kagura update-memory -m <uuid> --location "35.68,139.76,Tokyo HQ" --merge-details   # keeps the other details keys
kagura update-memory -m <uuid> --details '{"location": {"lat": 35.68, "lon": 139.76}, "client": "acme"}'   # replaces them all
kagura update-memory -m <uuid> --add-tag status:done --remove-tag status:open --expected-version 3
kagura forget -m <uuid>                                    # soft delete, one memory
kagura forget --ids "<uuid>,<uuid>"                        # up to 50, one result row per id
kagura forget -q "outdated test data" -k 5                 # bulk delete by search

# session start and what changed (read-only, no search)
kagura bootstrap --since 7d --format index --max-chars 5000
kagura changes 2026-11-01T00:00:00Z --by-me
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
  keep the keys you do not mention: it sends them as a `details_patch`, so the
  server merges the top-level keys in one call (memory-cloud 0.140.0+) —
  `location` itself is replaced whole, label included, so re-send
  `lat,lon,label` to keep a label — and a key given as `null` is deleted. It
  needs `-m`. `--details-patch '<json>'` sends a patch you write yourself.
  An older server gets a fallback with a `note:` on stderr: the CLI reads the
  memory first (`reference`, so two calls, not one atomic update) and writes
  the merged object; it writes nothing when the bounded `reference` reply left
  `details` out (memory-cloud 0.78.0+ on a large memory) — then re-send the
  complete object with `--details` and without `--merge-details`. When it
  exits 1 saying the server ignored other flags, relay which ones.
- `--add-tag` / `--remove-tag` (repeatable, memory-cloud 0.140.0+) edit the
  stored tags without resending them; `--tags` replaces the whole list.
  `--expected-version N` (0.113.0+) writes nothing when another edit landed
  since you read the memory, and prints the `Current version`: read it again
  (`reference`) and ask before retrying.

Search options:

- `--trusted-only` excludes connector-ingested / external memories. Use it for
  reads that steer the agent's next actions (as the SessionStart hook does).
- `--rerank` / `--no-rerank` is tri-state: no flag follows the context's search
  config; `--rerank` applies only when the context enables reranking;
  `--no-rerank` always skips it.
- `--verbosity full` returns each hit's uncut summary, `tags`,
  `context_summary` and `scope`. Without it (memory-cloud 0.108.0+) hits are
  compact: a 160-character summary head (`summary_cut: true` when cut), no
  tags.
- `--full N` puts `content` (up to 1,500 characters) on the top N hits;
  `--full 0` returns summaries only. Without it the server's default applies
  (3 unless the deployment changes it; memory-cloud 0.114.0+). The
  SessionStart hook passes `--verbosity full --full 0`: uncut summaries with
  tags, no content.
- `--max-chars N` (10000-100000, default 20000) bounds the whole reply.
- A server that predates one of these three flags refuses it and names it;
  `kagura recall` then retries once without the ones named and prints a
  `note:` on stderr (0.108.0–0.113.x keep `--verbosity full`).

## Consume the result

- `remember` → `memory_id` (keep it: `reference`, `update-memory` and `forget`
  take it as `-m`) and `scope`. An optional `lint` list (`[{code, hint}]`, e.g.
  `summary_short`, `no_tags`) means the memory was stored but will recall
  poorly — offer an `update-memory` with a better summary or tags.
- `recall` → `confidence` (read `confidence.level` first: `none` / `low` means
  probably not stored), `results` (ranked; each carries `memory_id` and
  `summary`) and `count`. Optional keys are absent when they do not apply:
  `related_tags` (`[{tag, count}]`), a hit's `tags` and `content`;
  `content_truncated` / `content_next_offset` mean `reference` has the rest;
  `truncated` / `context_summary_omitted` / `full_applied` mean `--max-chars`
  cut the reply; `degraded` / `degraded_reason` means the semantic half of
  search was unavailable and the results are keyword-only, so an empty result
  means "search impaired", not "nothing stored" — retry later.
- `reference` → the memory under `memory` (`summary`, `content`, `details`, …).
  On memory-cloud 0.78.0+ a large memory comes back bounded (about 20,000
  characters by default): check `content_truncated` / `content_omitted` /
  `details_omitted` before treating `content` or `details` as complete. The
  CLI cannot page the rest; the MCP `reference` tool can (`max_chars`,
  `*_offset`).
- `update-memory` → `memory_id`, `operation`, `re_embedded`, `scope`,
  `version` (memory-cloud 0.113.0+), plus the same optional `lint` as
  `remember` (re-check it after fixing a hint);
  `supersede_candidate_dismissed` is the only confirmation that
  `--dismiss-supersede-candidate` applied — absent when the memory had no live
  candidate or the server predates v0.65.0.
- `forget` → `deleted_count` and `memory_ids`. `deleted_count` is the outcome,
  not the request: `0` after `-m` means the server skipped it (a memory the
  credential may not delete, such as a tool guardrail) — report it as not
  deleted; a `-q` sweep can delete fewer than `-k` for the same reason.
  Deletion is soft and stays recoverable until the deployment's retention
  window passes; `-q` is refused while recall is degraded. Confirm with the
  user before a `-q` bulk delete. `--ids` → `succeeded` and one row per id
  (`deleted`, `skipped`: not found or not yours, or `error`); resend only the
  ids whose `error` is `not_attempted`. When every id failed the command exits
  1 and prints those rows on stderr.
- `remember` / `update-memory` (memory-cloud 0.142.0+) may carry
  `plan_headroom`: a plan cap at 80 % (`warning`) or 95 % (`critical`) of its
  limit, with a `help` sentence. Relay a `critical` one to the user; it is
  shown at most hourly, so its absence does not mean there is room.
- `bootstrap` → `context`, `guardrails` and `components` (`pinned`,
  `upcoming`, `changes`, each with its own `status`); `degraded: true` means
  a lane failed. `changes` → `changes` (oldest first), `has_more` and
  `next_cursor` (pass it as `--cursor`).
