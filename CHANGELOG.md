# Changelog

See [GitHub Releases](https://github.com/kagura-ai/kagura-memory-python-sdk/releases) for all release notes.

## Unreleased

### Added

- **Audio/video transcription via Gemini** (#147): `ingest("talk.mp3")` (and
  `.wav` / `.m4a` / `.mp4`-with-audio) now routes the file to a Gemini
  transcription path (`gemini/gemini-2.5-flash`) that returns timestamped
  `{start, end, text}` segments, assembled into time-windowed sections, then
  runs the normal chunk → summarize → remember pipeline. Audio has no parseable
  text — the transcript is *generated* by an LLM, so this is a Provider-layer
  concern (a new `ingest/_audio.py`), NOT an Extractor. v1 is a single inline
  request ≤ ~20 MB (Gemini's inline limit); cost is surfaced from the provider's
  real token usage (audio is metered at ~32 tokens/sec inside `prompt_tokens`).
  Opt-in `[ingest-audio]` extra; requires `GEMINI_API_KEY`. Chapter detection
  and ffmpeg-based chunking of longer media are deferred follow-ups. See
  `examples/ingest_audio.py`.
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
