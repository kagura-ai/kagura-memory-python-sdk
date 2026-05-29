# Changelog

See [GitHub Releases](https://github.com/kagura-ai/kagura-memory-python-sdk/releases) for all release notes.

## Unreleased

### Added

- `KaguraClient.list_memories()` — paginated, newest-first memory listing
  (`GET /api/v1/memory/list`) with optional `context_id`, `q` substring filter,
  `scope`, `type`, `limit`, and `offset`. `q` is stripped and whitespace-only
  values are treated as no filter, matching the server (memory-cloud #580).
  New `MemoryListItem` / `MemoryListResponse` models. (#143)
