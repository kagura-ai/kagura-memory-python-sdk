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

This SDK connects your Python code to [Kagura Memory Cloud](https://github.com/kagura-ai/memory-cloud), giving AI assistants the ability to **remember, search, and learn** from past interactions. It provides four clients for different use cases:

| Client | Protocol | Use Case |
|--------|----------|----------|
| **`KaguraAgent`** | MCP + LLM | AI-powered — auto-decides what to remember/recall from conversations |
| **`KaguraClient`** | MCP (JSON-RPC) | Direct memory ops — remember, recall, explore, reference, forget |
| **`ResourceClient`** | REST API | External data ingestion — push data from Slack, CI/CD, CRM into Kagura |
| **`FilesClient`** | REST + presigned PUT | File uploads with sha256 integrity binding (R2) |

## Installation

```bash
pip install kagura-memory
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
