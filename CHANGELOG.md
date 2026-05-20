# Changelog

See [GitHub Releases](https://github.com/kagura-ai/kagura-memory-python-sdk/releases) for all release notes.

## Unreleased

### Added
- `FileIngestor.ingest()` now accepts `details_extra: dict[str, Any] | None` to stamp caller-supplied keys onto both overview and section `memory.details`. Collisions with SDK reserved keys (e.g. `file_id`, `parent_id`) raise `ValueError` at entry. (#120)
