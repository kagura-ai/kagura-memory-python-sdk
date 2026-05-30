# Changelog

See [GitHub Releases](https://github.com/kagura-ai/kagura-memory-python-sdk/releases) for all release notes.

## Unreleased

### Added

- **Context-aware ingest summarization steering** (#148): ingest summaries are
  no longer context-blind. `FileIngestor.ingest()` gains a keyword-only
  `steering=None` argument; when omitted, the destination context's own
  owner-authored configuration steers the summaries, resolved by precedence
  `caller steering > context_info.instructions > context.summary > None`. The
  resolved string is injected as a clearly-demarcated, **non-overriding**
  trusted `<domain_context>` block appended *after* the fixed summarization
  prompt — it focuses terminology/scope but never replaces the task and never
  reopens the prompt-injection surface (the document body always stays data in
  the `user` role; §8.3 preserved). The signal source is owner/editor-authored
  workspace config, never document content; in a shared context the steering
  author may differ from the ingesting member, hence the non-overriding
  demarcation. `get_context_info` is fetched at most once per
  `(client, context_id)` and cached on `KaguraClient` (failures cache `None`),
  so steering adds no per-section round-trips; a fetch failure logs a warning
  and proceeds with `steering=None` rather than failing the ingest. Whitespace-
  only steering is treated as absent and the resolved value is truncated to
  2000 characters. Recall-side steering, vision steering, named profiles, and
  mid-session cache invalidation are deferred follow-ups.
- **Audio/video transcription via Gemini** (#147): `ingest("talk.mp3")` (and
  `.wav` / `.m4a` / `.mp4`-with-audio) now routes the file to a Gemini
  transcription path (`gemini/gemini-2.5-flash`) that returns timestamped
  `{start, end, text}` segments, assembled into time-windowed sections, then
  runs the normal chunk → summarize → remember pipeline. Audio has no parseable
  text — the transcript is *generated* by an LLM, so this is a Provider-layer
  concern (a new `ingest/_audio.py`), NOT an Extractor. v1 is a single inline
  request: media is sent base64-encoded, so the effective cap is ~15 MiB of raw
  audio/video (the encoded payload must fit Gemini's ~20 MiB inline limit), and
  larger files are rejected up front. Cost is surfaced from the provider's real
  token usage (audio is metered at ~32 tokens/sec inside `prompt_tokens`).
  Opt-in `[ingest-audio]` extra; requires `GEMINI_API_KEY`. Chapter detection
  and ffmpeg-based chunking of longer media are deferred follow-ups. See
  `examples/ingest_audio.py`.
- **YouTube transcript ingestion** (#146): `ingest("https://youtube.com/watch?v=...")`
  now resolves a single video's captions into a memory graph. YouTube URLs are
  auto-detected by host (`youtube.com`, `youtu.be`, `m.youtube.com`, including
  `watch?v=`, `youtu.be/`, and `shorts/` forms) and routed to a transcript
  source resolver that formats the captions as time-windowed Markdown
  (`# <title>` + `## [mm:ss]` sections), flowing through the existing
  chunk → summarize → remember pipeline. Manual captions are preferred,
  auto-generated captions are the fallback; the video title/channel come from
  YouTube oEmbed (best-effort — failure degrades the title to the video id, it
  never fails the ingest). Opt-in via the `[ingest-youtube]` extra
  (`youtube-transcript-api`, no API key); also bundled in `[ingest-all]`.
  Playlists/channels, caption-disabled, age-restricted, and unavailable videos
  raise an actionable error. Chapters are deferred to a follow-up. See
  `examples/ingest_youtube.py`.
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
