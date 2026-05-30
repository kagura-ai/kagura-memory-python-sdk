# Changelog

See [GitHub Releases](https://github.com/kagura-ai/kagura-memory-python-sdk/releases) for all release notes.

## v0.25.0

### Added

- **`kagura-mcp` refresh-aware stdio MCP proxy** (#101, core): a new console
  script (`kagura_memory.mcp_proxy:main`) that Claude Code spawns as a stdio
  MCP child. It owns `~/.kagura/credentials.json` and transparently forwards
  every JSON-RPC message to memory-cloud's HTTP `/mcp` endpoint with an
  always-fresh OAuth bearer — fixing the silent 401 that occurs when a
  short-lived `access_token` is baked into a static `.mcp.json` `headers`
  block. The proxy is a thin pass-through pump (not a tool-registering server),
  captures and replays the upstream `mcp-session-id`, and on an upstream `401`
  forces one token refresh + retry, surfacing an actionable "run `kagura auth
  login`" MCP error if the refresh itself fails. Point a `.mcp.json` entry at
  it with `{"type": "stdio", "command": "kagura-mcp", "args": ["--profile",
  "default"]}`.
- **`KaguraOAuth.force_refresh()`**: unconditional token refresh (ignores the
  skew window), coalesced through the same in-process lock as skew-driven
  refreshes. Backs the proxy's 401-retry.
- **Cross-process credentials locking**: `update_profile()` now wraps its
  read-modify-write in a POSIX `fcntl` advisory lock (on a sibling
  `credentials.json.lock`), so concurrent writers in different processes
  (e.g. multiple `kagura-mcp` children) cannot lose an update. Windows is a
  documented no-op pending a follow-up (the `msvcrt` semantics differ enough
  to warrant separate work).
- **`kagura setup claude --profile NAME`** (#157): wires Claude Code to the
  refresh-aware `kagura-mcp` proxy in one step. With `--profile`, `setup
  claude` writes the stdio `.mcp.json` form
  (`{"type": "stdio", "command": "kagura-mcp", "args": ["--profile", NAME]}`)
  bound to an OAuth profile from `kagura auth login` — no API key, no secret in
  the file, and the token refreshes automatically. It resolves the MCP URL and
  auth from the profile, verifies `kagura-mcp` is on `$PATH` (a warning, never a
  hard failure), and writes a `.kagura.json` without an `api_key`. Without
  `--profile`, the legacy long-lived API-key url form is unchanged
  (CI / service accounts). `--profile` and `--api-key` are mutually exclusive.
- **`kagura auth status` reports the Claude Code integration mode** (#157):
  when a `.mcp.json` exists in the current directory, `auth status` now reports
  whether the `kagura-memory` entry is `refresh-aware` (the `kagura-mcp` stdio
  proxy) or a `legacy static API-key token` (with a migration hint), so you can
  see at a glance which path a project is on.
- **Windows credentials-lock shim** (#158): `_filelock.file_lock` now has a
  Windows backend (`msvcrt.locking`) alongside the POSIX `fcntl` one, replacing
  the previous non-POSIX no-op. Because the Windows API has no shared-lock mode,
  `exclusive=False` is upgraded to an exclusive lock (concurrent readers
  serialize on Windows where they run in parallel on POSIX — the documented
  behavior difference), and a non-blocking `LK_NBLCK` retry loop emulates
  `flock`'s blocking acquire without `msvcrt`'s 10-second `LK_LOCK` timeout.
- **Cross-process refresh dedup over the network** (#158): concurrent
  `kagura-mcp` proxies no longer each hit `/oauth2/token`. `KaguraOAuth`'s
  refresh now acquires the cross-process advisory lock with non-blocking
  attempts on the event loop (so other coroutines keep running and a
  cancellation while waiting cannot leak the lock), re-reads the on-disk
  token, and **skips the network round-trip when another process already
  rotated it** — adopting the on-disk token instead. The skew-driven path and
  the 401-retry path use different "already rotated?" predicates: the skew path
  skips when the on-disk token is outside the skew window, while the 401 path
  skips only when the on-disk token *differs* from the rejected one (an
  identical token means nobody rotated yet, so a real refresh must fire — an
  `expires_at`-based skip there would loop on the rejected token). The lock is
  released synchronously (never offloaded) so a saturated executor cannot
  deadlock the release. Covered by a subprocess-based cross-process
  lock-contention test (no lost updates under `N`-way contention) plus
  deterministic in-process dedup tests with a negative control.

## v0.24.0

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
  2000 characters. The default (no-steering) path is **fully** non-breaking,
  including for custom `Provider` implementations written against the
  pre-steering signature: the `steering=` kwarg is omitted from the summarize
  call entirely when no steering is resolved, so such providers never receive
  an unexpected keyword. Recall-side steering, vision steering, named profiles,
  and mid-session cache invalidation are deferred follow-ups.
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
