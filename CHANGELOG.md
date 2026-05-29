# Changelog

See [GitHub Releases](https://github.com/kagura-ai/kagura-memory-python-sdk/releases) for all release notes.

## Unreleased

### Added

- **Document ingestion beyond PDF** (#144): structural extractors for plain
  text / Markdown (stdlib, no extra), HTML (`[ingest-html]`), Word `.docx`
  (`[ingest-docx]`), Excel `.xlsx` (`[ingest-xlsx]`), PowerPoint `.pptx`
  (`[ingest-pptx]`), and EPUB (`[ingest-epub]`, reuses PyMuPDF). `[ingest-all]`
  installs them all. Each extractor is pure-parse (no network/LLM), enforces
  decompression-bomb caps, and surfaces a clear `KaguraIngestError` when its
  optional dependency is missing. `kagura ingest` / `FileIngestor` dispatch
  automatically by MIME type, suffix, or magic bytes. See
  `examples/ingest_documents.py`.
