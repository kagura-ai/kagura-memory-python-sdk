---
name: ingest
description: Ingest a document or URL into Kagura Memory — extract text, chunk it, and store an overview plus per-section memories. Use when the user wants to add a PDF, Office/HTML/EPUB file, audio/video, or a web page to their memory graph.
---

# kagura ingest — document ingestion

Drive `kagura ingest` to turn a file or URL into a structured set of memories.
Thin wrapper around the installed `kagura` CLI.

## Preflight

- `kagura --version`; if missing → `uv tool install kagura-memory` (or
  `pip install kagura-memory`), then stop.
- Heavy parsers are opt-in extras (e.g. `pip install 'kagura-memory[ingest-pdf]'`
  or `[ingest-all]`); a missing parser surfaces a clear "install ..." error from
  the CLI — relay that hint rather than guessing.
- Requires authentication and a target context. Run `kagura auth status`; if not
  authed, run the `auth` skill first.

## Run

```bash
kagura ingest ./report.pdf                  # local file
kagura ingest https://example.com/page      # URL
kagura ingest ./report.pdf --dry-run        # preview cost/sections; no LLM or backend write
kagura ingest ./report.pdf --json           # machine-readable result for scripting
```

## Consume the result

- Relay the overview + per-section counts and any per-section error records.
- On a missing-extra error, surface the exact `pip install` command the CLI
  prints. Use `--dry-run` first when the user wants a cost/size estimate.
