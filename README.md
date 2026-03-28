<p align="center">
  <a href="https://github.com/kagura-ai/memory-cloud">
    <img src="assets/kagura-logo.svg" alt="Kagura Ai" width="300">
  </a>
</p>

<h1 align="center">Kagura Memory SDK</h1>

<p align="center">
  Python SDK for <a href="https://github.com/kagura-ai/memory-cloud">Kagura Memory Cloud</a> — give your AI applications persistent, searchable memory.
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

This SDK connects your Python code to [Kagura Memory Cloud](https://github.com/kagura-ai/memory-cloud), giving AI assistants the ability to **remember, search, and learn** from past interactions. It provides three clients for different use cases:

| Client | Protocol | Use Case |
|--------|----------|----------|
| **`KaguraAgent`** | MCP + LLM | AI-powered — auto-decides what to remember/recall from conversations |
| **`KaguraClient`** | MCP (JSON-RPC) | Direct memory ops — remember, recall, explore, reference, forget |
| **`ResourceClient`** | REST API | External data ingestion — push data from Slack, CI/CD, CRM into Kagura |

## Installation

```bash
pip install kagura-memory
# or
uv add kagura-memory
```

## Quick Start

### Configuration

Create `.kagura.json` in your project root (or `~/.kagura.json`):

```json
{
  "api_key": "kagura_your_api_key",
  "mcp_url": "https://memory.kagura-ai.com/mcp/w/{workspace_id}",
  "model": "gpt-5.4-nano",
  "context_id": "dev"
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

Supports OpenAI, Claude, Gemini, Ollama via [LiteLLM](https://github.com/BerriAI/litellm).

### KaguraClient — Direct Memory Operations

For programmatic control without LLM:

```python
from kagura_memory import KaguraClient

async with KaguraClient(api_key="kagura_...", mcp_url="https://...") as client:
    await client.remember(context_id="dev", summary="OAuth2 pattern", content="Use Authlib...")
    results = await client.recall(context_id="dev", query="OAuth2", k=5)
    await client.explore(context_id="dev", memory_id="uuid", depth=3)
```

### ResourceClient — External Data Ingestion

Push data from external systems into Kagura so AI can search it:

```python
from kagura_memory import ResourceClient, ResourceEventRequest

async with ResourceClient.from_mcp_url(api_key="kagura_...", mcp_url="https://...") as client:
    token = await client.create_token(resource_id="products", description="Product sync")
    print(f"Save this token: {token.token}")  # Shown only once!

    event = ResourceEventRequest(
        op="upsert", doc_id="SKU-001", version=1,
        payload={"name": "Wireless Headphones", "price": 79.99},
    )
    await client.ingest_event("products", token.token, event)
```

See [`examples/`](examples/) for complete working examples.

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
/workflow        # Check current state and next step
/quality         # Run all quality checks
/simplify        # Review for reuse, quality, efficiency
/self-review     # Pre-PR self-review
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
