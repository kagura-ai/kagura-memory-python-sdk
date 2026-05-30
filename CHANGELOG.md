# Changelog

See [GitHub Releases](https://github.com/kagura-ai/kagura-memory-python-sdk/releases) for all release notes.

## Unreleased

### Added

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
- **Document ingestion beyond PDF** (#144): structural extractors for plain
  text / Markdown (stdlib, no extra), HTML (`[ingest-html]`), Word `.docx`
  (`[ingest-docx]`), Excel `.xlsx` (`[ingest-xlsx]`), PowerPoint `.pptx`
  (`[ingest-pptx]`), and EPUB (`[ingest-epub]`, reuses PyMuPDF). `[ingest-all]`
  installs them all. Each extractor is pure-parse (no network/LLM), enforces
  decompression-bomb caps, and surfaces a clear `KaguraIngestError` when its
  optional dependency is missing. `kagura ingest` / `FileIngestor` dispatch
  automatically by MIME type, suffix, or magic bytes. See
  `examples/ingest_documents.py`.
