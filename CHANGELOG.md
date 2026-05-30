# Changelog

See [GitHub Releases](https://github.com/kagura-ai/kagura-memory-python-sdk/releases) for all release notes.

## Unreleased

### Added

- **Browser-rendered URL fetch** (#145): opt-in `render=True` on
  `FileIngestor.ingest()` / `estimate_cost()` (and `kagura ingest --render`)
  drives a headless Chromium via Playwright to load and render JS-heavy / SPA
  pages, then hands the rendered HTML to the existing HTML extractor → chunker
  → provider pipeline. Default off. Requires the new `[ingest-browser]` extra
  (`pip install 'kagura-memory[ingest-browser]'` + `playwright install
  chromium`); a missing dependency surfaces as a `step="fetch"` error on the
  `IngestResult`, not an uncaught exception. SSRF-hardened at the browser
  layer: every request (navigation, redirect, XHR/fetch, sub-resource) is
  re-resolved against the same RFC1918/loopback/link-local/IMDS denylist and
  aborted if it targets an internal IP; `http://` sub-resources are gated by
  `allow_http`. The rendered-HTML size cap (`max_bytes`) and navigation
  timeout (`read_timeout`) bound the worst case. See
  `examples/ingest_rendered_url.py`.
- **Document ingestion beyond PDF** (#144): structural extractors for plain
  text / Markdown (stdlib, no extra), HTML (`[ingest-html]`), Word `.docx`
  (`[ingest-docx]`), Excel `.xlsx` (`[ingest-xlsx]`), PowerPoint `.pptx`
  (`[ingest-pptx]`), and EPUB (`[ingest-epub]`, reuses PyMuPDF). `[ingest-all]`
  installs them all. Each extractor is pure-parse (no network/LLM), enforces
  decompression-bomb caps, and surfaces a clear `KaguraIngestError` when its
  optional dependency is missing. `kagura ingest` / `FileIngestor` dispatch
  automatically by MIME type, suffix, or magic bytes. See
  `examples/ingest_documents.py`.
