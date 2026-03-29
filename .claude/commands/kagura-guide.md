---
description: Show Kagura Memory SDK usage guide (clients, models, CLI)
---

Show the user a concise SDK usage guide. Read the sections below and present the relevant parts based on context. If the user asked about a specific topic, focus on that; otherwise show the overview.

## SDK Overview

Kagura Memory SDK has three client classes:

| Class | Protocol | Purpose |
|-------|----------|---------|
| `KaguraClient` | MCP (JSON-RPC) | Low-level memory operations (remember, recall, explore, reference, forget) |
| `KaguraAgent` | MCP + LLM | AI-powered session analysis — auto-decides what to remember/recall |
| `ResourceClient` | REST API | Resource token management + external data ingestion |

## KaguraClient — Direct Memory Operations

```python
from kagura_memory import KaguraClient

async with KaguraClient(api_key="kagura_...", mcp_url="https://memory.kagura-ai.com/mcp") as client:
    # Store
    await client.remember(context_id="dev", summary="FastAPI DI", content="Use Depends()...", type="note", importance=0.7)

    # Search
    results = await client.recall(context_id="dev", query="dependency injection", k=5)

    # Graph traversal
    related = await client.explore(context_id="dev", memory_id="uuid-here", depth=2)

    # Full details
    memory = await client.reference(context_id="dev", memory_id="uuid-here")

    # Delete (soft, 30-day recovery)
    await client.forget(context_id="dev", memory_id="uuid-here")

    # Context management
    contexts = await client.list_contexts()
    await client.create_context(name="new-ctx", summary="My context")
    await client.update_context(context_id="uuid", summary="Updated")
    await client.update_context(context_id="uuid", resource_id="my-res", is_public=True)

    # Search config (tune hybrid search weights per context)
    await client.update_search_config(context_id="uuid", semantic_weight=0.5, bm25_weight=0.5)
```

## KaguraAgent — AI-Powered Analysis

```python
from kagura_memory import KaguraAgent, Session, Message

agent = KaguraAgent(api_key="kagura_...", model="gpt-5.4-nano")

session = Session(messages=[
    Message(role="user", content="FastAPIでOAuth2を実装したい"),
    Message(role="assistant", content="Authlibがおすすめです..."),
])

# AI auto-decides what to remember/recall
async with agent:
    result = await agent.process(session, deep=True, verbose=2)
    # result.remembered, result.recalled, result.explored, result.llm_usage
```

### Hooks & Skills

```python
agent = KaguraAgent(api_key="kagura_...", model="gpt-5.4-nano")

# Hooks — auto-triggered during process()
@agent.hook("before_process")
async def on_start(session, context_id):
    print(f"Processing {context_id}")

@agent.hook("on_remember")
def on_store(remembered):  # sync callables also work
    print(f"Stored {len(remembered)} memories")

# Events: before_process, after_process, on_remember, on_recall

# Skills — explicit invocation
@agent.skill("summarize")
async def summarize(context_id):
    return await agent.client.recall(context_id, "summary", k=50)

result = await agent.run_skill(skill_name="summarize", context_id="dev")
agent.list_skills()  # ["summarize"]
```

## ResourceClient — External Data Ingestion

```python
from kagura_memory import ResourceClient, ResourceEventRequest

client = ResourceClient.from_mcp_url(api_key="kagura_...", mcp_url="http://localhost:8080/mcp/w/...")

async with client:
    # One-call setup: create public context + set resource_id + create token
    token = await client.setup_resource(resource_id="products", summary="Product catalog")

    # Ingest events
    event = ResourceEventRequest(op="upsert", doc_id="SKU-001", version=1, payload={"name": "Widget"})
    await client.ingest_event("products", token.token, event)

    # Batch (max 100)
    events = [ResourceEventRequest(op="upsert", doc_id=f"SKU-{i}", version=1, payload={"name": f"Item {i}"}) for i in range(10)]
    await client.ingest_events("products", token.token, events)

    # Token management
    tokens = await client.list_tokens(resource_id="products")
    await client.revoke_token(token.id)
```

Note: Resource ingestion requires a public context (`is_public=True`) with `resource_id` set. `setup_resource()` handles this automatically.

## CLI Commands

```bash
# AI-powered
kagura process -m "Remember this: ..."
kagura process -m "Search for OAuth2" --deep

# Direct memory ops
kagura remember -s "summary" --content "full text" -c context_id
kagura recall "search query" -k 10
kagura explore -m memory-uuid --depth 3
kagura reference -m memory-uuid
kagura forget -m memory-uuid

# Context management
kagura context list
kagura context create -n my-project -s "Project context"
kagura context update CTX_UUID -s "Updated summary"
kagura context search-config CTX_UUID --semantic 0.5 --bm25 0.5

# Resource tokens
kagura resource tokens list
kagura resource tokens create -r resource_id -d "description"
kagura resource tokens update TOKEN_ID -q 2000
kagura resource tokens revoke TOKEN_ID
kagura resource ingest -r resource_id -k RESOURCE_TOKEN --doc-id DOC -V 1 -p '{"key":"value"}'
kagura resource ingest-batch -r resource_id -k RESOURCE_TOKEN -f events.json

# Config
kagura config show
```

## Configuration

`.kagura.json` (project root or home directory):
```json
{
  "api_key": "kagura_...",
  "mcp_url": "https://memory.kagura-ai.com/mcp",
  "model": "gpt-5.4-nano",
  "context_id": "dev",
  "llm_api_key": "sk-..."
}
```

Environment variables: `KAGURA_API_KEY`, `KAGURA_MCP_URL`, `KAGURA_MODEL`, `KAGURA_CONTEXT_ID`

## When to Use Which Client

| Scenario | Client |
|----------|--------|
| Store/search specific memories programmatically | `KaguraClient` |
| Let AI analyze a conversation and auto-manage memories | `KaguraAgent` |
| Push external data (products, Slack, CI/CD) into Kagura | `ResourceClient` |
| Quick CLI operations | `kagura` CLI |
