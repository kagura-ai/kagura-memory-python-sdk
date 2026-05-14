<p align="center">
  <a href="https://github.com/kagura-ai/memory-cloud">
    <img src="https://raw.githubusercontent.com/kagura-ai/kagura-memory-python-sdk/main/assets/kagura-logo.svg" alt="Kagura Ai" width="300">
  </a>
  <br>
  <strong>Memory SDK</strong> — Python client for <a href="https://github.com/kagura-ai/memory-cloud">Kagura Memory Cloud</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/kagura-memory/"><img src="https://img.shields.io/pypi/v/kagura-memory" alt="PyPI version"></a>
  <a href="https://pypi.org/project/kagura-memory/"><img src="https://img.shields.io/pypi/pyversions/kagura-memory" alt="Python versions"></a>
  <a href="https://github.com/kagura-ai/kagura-memory-python-sdk/actions/workflows/ci.yml"><img src="https://github.com/kagura-ai/kagura-memory-python-sdk/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://codecov.io/gh/kagura-ai/kagura-memory-python-sdk"><img src="https://codecov.io/gh/kagura-ai/kagura-memory-python-sdk/graph/badge.svg" alt="codecov"></a>
  <a href="https://github.com/kagura-ai/kagura-memory-python-sdk/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/kagura-memory" alt="License: MIT"></a>
  <a href="https://modelcontextprotocol.io/"><img src="https://img.shields.io/badge/MCP-Streamable_HTTP-purple.svg" alt="MCP"></a>
  <a href="https://microsoft.github.io/pyright/"><img src="https://microsoft.github.io/pyright/img/pyright_badge.svg" alt="Checked with pyright"></a>
</p>

## What is this?

This SDK connects your Python code to [Kagura Memory Cloud](https://github.com/kagura-ai/memory-cloud), giving AI assistants the ability to **remember, search, and learn** from past interactions — and to **ingest documents** (PDFs, URLs) directly into a searchable memory graph. It provides four clients for different use cases:

| Client | Protocol | Use Case |
|--------|----------|----------|
| **`KaguraAgent`** | MCP + LLM | AI-powered — auto-decides what to remember/recall from conversations |
| **`KaguraClient`** | MCP (JSON-RPC) | Direct memory ops — remember, recall, explore, reference, forget |
| **`ResourceClient`** | REST API | External data ingestion — push data from Slack, CI/CD, CRM into Kagura |
| **`FilesClient`** | REST + presigned PUT | File uploads with sha256 integrity binding (R2) |
| **`FileIngestor`** | CLI + SDK | Document ingestion — PDF text → memory graph + R2 archive (Phase 1; image/PPT/Excel in Phase 2) |

## 60-second demo

Turn a PDF into a structured graph of memories — one overview memory plus per-section summaries, linked via `declared_link` edges. The original file is archived to your workspace's storage so you can always pull the bytes back.

```bash
pip install 'kagura-memory[ingest-pdf]'
kagura auth login
kagura ingest ./report.pdf
```

<p align="center">
  <img src="https://raw.githubusercontent.com/kagura-ai/kagura-memory-python-sdk/main/assets/ingest-demo.svg"
       alt="kagura ingest demo" width="720">
</p>

```bash
kagura recall "report findings" -k 5        # search across sections
kagura files download-url <file_id>          # short-lived GET on the original
```

Phase 1 ingests text content (PDF section text). The vision pipeline (Gemini 2.5 Flash by default) is wired through and can be exercised via the `FileIngestor` API, but the bundled PDF extractor does not yet emit page images — image-based OCR memories land in Phase 2. Pass `--no-vision` to skip the vision-provider configuration entirely, or `--dry-run` to see token / cost estimates without calling an LLM.

## Installation

```bash
pip install kagura-memory                   # core SDK
pip install 'kagura-memory[ingest-pdf]'     # adds PDF ingestion support
# or
uv add kagura-memory
```

## Quick Start

### Configuration

Copy the example and fill in your credentials:

```bash
cp .kagura.json.example .kagura.json
# Edit .kagura.json — set api_key and mcp_url
```

Used by the CLI (`kagura` commands) and `load_config()` in Python code:

```json
{
  "api_key": "kagura_your_api_key",
  "mcp_url": "http://localhost:8080/mcp/w/{workspace_id}",
  "model": "gpt-5.4-nano",
  "context_id": "auto"
}
```

Or use environment variables: `KAGURA_API_KEY`, `KAGURA_MCP_URL`, `KAGURA_MODEL`, `KAGURA_CONTEXT_ID`

> Get your API key from the [Kagura Memory Cloud](https://github.com/kagura-ai/memory-cloud) Web UI: **Integrations > API Keys**

### KaguraAgent — AI-Powered Memory

Let the AI analyze conversations and automatically decide what to remember and recall:

```python
from kagura_memory import KaguraAgent, Session, Message

agent = KaguraAgent(api_key="kagura_...", model="gpt-5.4-nano")

session = Session(messages=[
    Message(role="user", content="FastAPIでOAuth2を実装したい"),
    Message(role="assistant", content="Authlibを使うパターンが推奨です..."),
    Message(role="user", content="なるほど、これ覚えておいて"),
])

async with agent:
    result = await agent.process(session, deep=True, verbose=2)
    print(f"Remembered: {len(result.remembered)}, Recalled: {len(result.recalled)}")
```

Supports OpenAI, Claude, Gemini via [LiteLLM](https://github.com/BerriAI/litellm), and **Ollama** for local models:

```python
# Local LLM via Ollama (no cloud API key needed)
agent = KaguraAgent(api_key="kagura_...", model="ollama/qwen3:30b")
```

#### Ollama Local Model Requirements

| Model | Size | Context | Min VRAM | Recommended GPU |
|-------|------|---------|----------|-----------------|
| `qwen3:30b` (recommended) | 19 GB | 256K | 24 GB | RTX 4090 or equivalent |
| `qwen3:14b` | 9.3 GB | 40K | 16 GB | RTX 4080 or equivalent |

**Recommended minimum**: `qwen3:30b` on an RTX 4090 (24 GB VRAM) or equivalent.

Smaller models (< 30B parameters) may produce lower quality memory analysis — summaries may lack searchable keywords, and recall query generation may be less precise.

### KaguraClient — Direct Memory Operations

For programmatic control without LLM:

```python
from kagura_memory import KaguraClient

async with KaguraClient(api_key="kagura_...", mcp_url="https://...") as client:
    await client.remember(context_id="dev", summary="OAuth2 pattern", content="Use Authlib...")
    results = await client.recall(context_id="dev", query="OAuth2", k=5)
    await client.explore(context_id="dev", memory_id="uuid", depth=3)

    # Tag AND filter — match memories with ALL specified tags
    results = await client.recall(
        context_id="dev", query="budget",
        filters={"tags": ["予算", "2026"], "tags_match": "all"},
    )

    # Date range filter
    results = await client.recall(
        context_id="dev", query="recent decisions",
        filters={"created_after": "2026-03-01T00:00:00Z", "created_before": "2026-03-31T23:59:59Z"},
    )

    # Cross-context recall — search across multiple contexts at once
    results = await client.recall(
        query="authentication",
        context_ids=["ctx-uuid-1", "ctx-uuid-2"], k=10,
    )

    # Tag vocabulary — discover existing tag spellings before remember()/recall()
    tags = await client.list_tags(context_id="dev", sort="recent", prefix="auth")
    print([(t.tag, t.count) for t in tags.tags])

    # Merge contexts — copy all memories from source to target
    result = await client.merge_contexts(source_id="old-ctx", target_id="new-ctx")
    print(f"Merged {result['merged']} memories")

    # Merge and delete the source context
    result = await client.merge_contexts(
        source_id="old-ctx", target_id="new-ctx", delete_source=True,
    )

    # Workspace usage — check quota limits
    usage = await client.get_usage()
    print(f"Plan: {usage.plan}, Memories: {usage.memories.used}/{usage.memories.limit}")

    # Context info — includes search_config
    info = await client.get_context_info(context_id="dev")
    print(f"Search config: {info.context.search_config}")

    # Embedding status — check for failures
    status = await client.get_embedding_status()
    print(f"Embeddings: {status.total}, Failed: {len(status.failed_memories)}")

    # Per-memory stats — recall frequency, access patterns
    stats = await client.get_memory_stats(context_id="dev", sort_by="use_count", limit=10)

    # Duplicate detection — find similar memory pairs
    dupes = await client.find_duplicates(context_id="dev", threshold=0.90)
    print(f"Found {dupes.total_pairs} duplicate pairs")
```

### ResourceClient — External Data Ingestion

Push data from external systems into Kagura so AI can search it:

```python
from kagura_memory import ResourceClient, ResourceEventRequest

async with ResourceClient.from_mcp_url(api_key="kagura_...", mcp_url="http://localhost:8080/mcp/w/...") as client:
    # One-call setup: create public context + set resource_id + create token
    token = await client.setup_resource(resource_id="products", summary="Product catalog")
    print(f"Save this token: {token.token}")  # Shown only once!

    event = ResourceEventRequest(
        op="upsert", doc_id="SKU-001", version=1,
        payload={"name": "Wireless Headphones", "price": 79.99},
    )
    await client.ingest_event("products", token.token, event)

    # Check ingestion stats
    stats = await client.get_resource_impact("products")
    print(f"Memories: {stats.memory_count}, Tokens: {stats.token_count}")
```

See [`examples/`](examples/) for complete working examples.

### FilesClient — File Uploads with Checksum Binding

Upload files to the workspace's object store via short-lived presigned PUT URLs. The SDK binds the body's sha256 into the PUT signature so the server (memory-cloud v0.15.1+, `R2_CHECKSUM_BINDING_ENABLED=true`) can reject tampered uploads with `400 BadDigest`:

```python
from pathlib import Path
from kagura_memory import FilesClient

async with FilesClient.from_mcp_url(api_key="kagura_...", mcp_url="https://memory.kagura-ai.com/mcp") as client:
    # Upload from a Path (read fully into memory; server caps file size at 100 MiB)
    f = await client.upload(context_id="ctx-uuid", source=Path("./report.pdf"))
    print(f"Uploaded {f.id}, sha256={f.sha256}, size={f.size_bytes}")

    # Upload from bytes — filename is required (server enforces non-empty)
    f2 = await client.upload(context_id="ctx-uuid", source=b"...", filename="payload.bin")

    # Short-lived presigned GET URL
    url = await client.download_url(f.id)

    # List & delete
    page = await client.list(context_id="ctx-uuid", limit=50)
    await client.delete(f.id)
```

Re-uploading bytes whose sha256 already exists in the workspace returns the **existing `FileObject`** (idempotent dedup happy-path) — no exception.

## SDK ↔ memory-cloud Compatibility

| SDK | Min memory-cloud | Notes |
|---|---|---|
| 0.14.0 | 0.15.1 | `FilesClient` + R2 checksum binding (`x-amz-checksum-sha256` on PUT) |
| 0.13.x | 0.13.0 | Pre-`FilesClient` |

When pointing the SDK at a backend with `R2_CHECKSUM_BINDING_ENABLED=true`, the SDK must be v0.14.0+ — older versions don't send the signed checksum header and uploads fail with `HTTP 403 SignatureDoesNotMatch`.

> The `0.14.0` row above describes the next minor release; `__version__` is bumped from `0.13.0` to `0.14.0` by `/release minor` (see `.claude/rules/versioning.md`) at tag time, not in this feature branch.

## CLI

### Authentication (OAuth2 device flow)

Log in once with `kagura auth login` — the SDK stores credentials at
`~/.kagura/credentials.json` (mode 0600) and `KaguraClient()` plus all
`kagura` CLI commands pick them up automatically:

```bash
kagura auth login                                    # default scope: memory:read
kagura auth login --scope "memory:read memory:write" # also request write access
kagura auth login --no-browser                       # SSH / WSL2 / headless
kagura auth login --profile work                     # named profile for a second workspace

kagura auth status                                   # show profile, server, expiry, scope
kagura auth refresh                                  # manual token rotation
kagura auth refresh --scope "memory:write"           # incremental consent (re-runs device flow)
kagura auth token                                    # raw access_token to stdout (CI use)
kagura auth logout                                   # revoke + delete profile
kagura auth logout --all --yes                       # remove every profile
```

Two integration paths:

| You want… | Use |
|---|---|
| **CLI / `KaguraClient`** (terminal use, scripts, `KaguraAgent`) | `kagura auth login` — refresh happens automatically |
| **Claude Code MCP** (Claude Code reads `.mcp.json`) | `kagura setup claude` with a long-lived API key from the web UI |

Claude Code's MCP client reads its config once at startup and does
not refresh tokens — a refresh-aware MCP proxy daemon
(`kagura-mcp`) is tracked as a follow-up so that `kagura auth login`
can eventually power both paths from a single credentials file. For
now, use the long-lived API key path for Claude Code and the OAuth
path for everything else.

Credential resolution order when `KaguraClient()` is called with no
arguments: `KAGURA_API_KEY` env (CI / service accounts always win) →
`KAGURA_PROFILE` env or explicit `profile=` arg → `default_profile`
from `~/.kagura/credentials.json` → legacy `.kagura.json`.

### Other commands

```bash
# AI-powered (requires LLM API key)
kagura process -m "Remember: FastAPI uses Depends() for DI"

# Direct memory operations
kagura remember -s "FastAPI DI" --content "Use Depends()..." -c dev
kagura recall "dependency injection" -k 10
kagura explore -m "memory-uuid" --depth 3
kagura forget -m "memory-uuid"
kagura contexts

# Resource tokens
kagura resource tokens create -r products -d "Product sync"
kagura resource ingest -r products -k TOKEN --doc-id SKU-001 -V 1 -p '{"name":"Widget"}'
kagura resource ingest-batch -r products -k TOKEN -f events.json
kagura resource stats -r products
kagura resource schema -r products

# Sleep Maintenance — observability + rollback
kagura sleep history <context-id> --limit 5
kagura sleep report <context-id> <report-id>
kagura sleep rollback <context-id> <report-id> -y    # destructive: prompts unless --yes / -y is set

# File uploads (R2 checksum binding)
kagura files upload ./report.pdf -c <context-id>
kagura files list -c <context-id> --limit 50
kagura files download-url <file-id>
kagura files delete <file-id>

# Config
kagura config show
```

### Document ingestion (`kagura ingest`)

See the [60-second demo](#60-second-demo) above for the happy path. The full option surface:

```bash
# Local file or URL → one overview + N sections + R2 archive
kagura ingest ./report.pdf
kagura ingest https://example.com/report.pdf --tags "Q1,research"

# Preview cost / sections without calling any LLM
kagura ingest ./report.pdf --dry-run

# Skip vision-provider configuration entirely (Phase 1 already ingests
# text only; this becomes meaningful once Phase 2 image extraction lands)
kagura ingest ./report.pdf --no-vision

# Storage: skip the R2 archive (no file_id stamped on the overview memory)
kagura ingest ./report.pdf --no-archive

# Machine-readable output for scripts
kagura ingest ./report.pdf --json
```

Exit codes: `0` when the overview memory is created (per-section errors are still `0`, they show up in `result.errors`); `1` when the overview itself fails (corrupted PDF, network error, etc.); `0` for any `--dry-run` invocation.

Provider configuration (env vars, picked up automatically via `litellm`):
- `ANTHROPIC_API_KEY` — text summarization (default `claude/sonnet-4-6`)
- `GEMINI_API_KEY` — reserved for vision OCR (default `gemini/gemini-2.5-flash`); not invoked in Phase 1
- Override per invocation: `--text-provider {claude|gemini|ollama}`, `--vision-provider {claude|gemini|ollama}`

## Claude Code Integration

Use Kagura Memory as an MCP server in Claude Code:

```bash
cp .mcp.json.example .mcp.json
# Edit .mcp.json — set workspace_id and API key
```

Or use the CLI directly:

```bash
kagura process -m "今日の学び：FastAPIのDIはDepends()を使う"
```

## API Coverage

| Operation | SDK Client | Protocol | Auth |
|-----------|-----------|----------|------|
| Memory (remember/recall/forget/explore/reference) | `KaguraClient` | MCP | API Key |
| Context (create/update/list/get_context_info) | `KaguraClient` | MCP | API Key |
| Workspace (get_usage) | `KaguraClient` | MCP | API Key |
| Search config (update_search_config) | `KaguraClient` | MCP | API Key |
| Embedding status (get_embedding_status) | `KaguraClient` | REST | API Key |
| Memory stats (get_memory_stats) | `KaguraClient` | REST | API Key |
| Duplicate detection (find_duplicates) | `KaguraClient` | REST | API Key |
| Context delete | — | Web UI only | Session |
| Sleep Maintenance (history / report / rollback) | `KaguraClient` | MCP | API Key |
| Resource Token (create/list/update/revoke) | `ResourceClient` | REST API | API Key |
| Resource Event ingestion | `ResourceClient` | REST API | Resource Token |
| Resource Impact (stats) | `ResourceClient` | REST API | API Key |
| Resource Schema | `ResourceClient` | REST API | API Key |
| File upload / download-url / delete / list | `FilesClient` | REST + presigned PUT | API Key |
| Account erasure (GDPR Art.17 / APPI) | — | Web UI only | Session |

Context deletion and account erasure are intentionally Web UI only — destructive operations require session authentication and confirmation. `kagura sleep rollback` runs over the MCP API Key but is itself destructive (reverses edge creation, merges, importance updates, promotions, and archives) and the CLI requires `--yes` to skip the interactive confirmation. The server commits per-action without a Saga, so a 5xx response after partial success means SOME actions may have been reversed before the error surfaced — re-run `kagura sleep report` to inspect the post-failure state.

## Development

```bash
git clone https://github.com/kagura-ai/kagura-memory-python-sdk.git
cd kagura-memory-python-sdk
uv sync --dev
```

```bash
uv run ruff check src/ tests/   # Lint
uv run ruff format src/ tests/  # Format
uv run pyright src/              # Type check
uv run pytest tests/ -v          # Test
```

### Development with Claude Code

This project is developed with [Claude Code](https://claude.com/claude-code):

```
/onboarding      # Interactive setup — verify config, test connection
/workflow        # Check current state and next step
/quality         # Run all quality checks
/simplify        # Review for reuse, quality, efficiency
/self-review     # Pre-PR self-review
/self-maint      # Audit .claude/ config against codebase
/release <level> # Bump version, tag, push, create GitHub Release
/kagura-guide    # SDK usage reference
```

**Typical flow:** Issue → Branch → Implement → `/quality` → `/simplify` → `/self-review` → PR → Merge → `/release`

## Links

- [Kagura Memory Cloud](https://github.com/kagura-ai/memory-cloud) — the server this SDK connects to
- [Releases](https://github.com/kagura-ai/kagura-memory-python-sdk/releases) — changelogs
- [Issues](https://github.com/kagura-ai/kagura-memory-python-sdk/issues) — bug reports & feature requests

## License

MIT License — see [LICENSE](LICENSE) for details.
