---
description: Interactive SDK setup guide — verify config, test connection, create context
---

Walk the user through setting up the Kagura Memory SDK step by step. Be interactive — check each step and report results before moving on. Do not overwrite existing configuration.

## Steps

### 1. Config check

Check if `.kagura.json` exists in the project root:

```bash
cat .kagura.json 2>/dev/null || echo "NOT_FOUND"
```

Also check environment variables:

```bash
echo "KAGURA_API_KEY=${KAGURA_API_KEY:+set}" "KAGURA_MCP_URL=${KAGURA_MCP_URL:+set}"
```

**If config exists**: Show the current config (mask API key). Ask if they want to continue with it.

**If no config**: Guide them to create `.kagura.json`:

```bash
cp .kagura.json.example .kagura.json
# Then edit: set api_key and mcp_url
```

Tell them:
- Get API key from Kagura Memory Cloud Web UI: **Integrations > API Keys**
- MCP URL format: `http://localhost:8080/mcp/w/{workspace_id}`
- Workspace ID is in the URL bar of the Web UI

### 2. Connection test

Verify the API key and MCP URL work:

```bash
uv run python -c "
import asyncio
from kagura_memory import KaguraClient
from kagura_memory.config import load_config

async def test():
    config = load_config()
    async with KaguraClient(api_key=config['api_key'], mcp_url=config['mcp_url']) as client:
        contexts = await client.list_contexts()
        print(f'Connected! {contexts[\"count\"]} contexts available.')
        for ctx in contexts.get('contexts', []):
            print(f'  - {ctx[\"name\"]} ({ctx[\"id\"][:8]}...)')

asyncio.run(test())
"
```

**If success**: Show available contexts and proceed.
**If error**: Help diagnose — is the server running? Is the API key correct? Is the MCP URL correct?

### 3. Context setup

If no contexts exist, offer to create one:

```bash
uv run kagura context create -n my-project -s "My first context"
```

If contexts exist, ask which one to use and suggest setting it as default in `.kagura.json`.

### 4. Quick test

Store and recall a test memory to verify everything works end-to-end:

```bash
uv run kagura remember -s "Onboarding test memory — SDK setup verified" --content "This memory was created during SDK onboarding to verify the connection works." -c CONTEXT_ID
```

Then recall it:

```bash
uv run kagura recall "onboarding test" -c CONTEXT_ID
```

Then clean up:

```bash
uv run kagura forget -m MEMORY_ID
```

### 5. Summary

Print a summary of what's configured:

```
✓ Config: .kagura.json loaded
✓ Connection: localhost:8080 (API key working)
✓ Context: my-project (CONTEXT_ID)
✓ Memory: store + recall + delete verified

You're ready to use Kagura Memory SDK!

Next steps:
- /kagura-guide — see full SDK usage reference
- kagura process -m "Remember this: ..." — AI-powered memory
- kagura recall "search query" — search memories
```

### 6. Optional: Resource Token setup

Ask if they want to set up external data ingestion:

```bash
uv run python -c "
import asyncio
from kagura_memory import ResourceClient
from kagura_memory.config import load_config

async def setup():
    config = load_config()
    async with ResourceClient.from_mcp_url(api_key=config['api_key'], mcp_url=config['mcp_url']) as rc:
        token = await rc.setup_resource(resource_id='my-resource', summary='My first resource')
        print(f'Resource token created! Save this token (shown only once):')
        print(f'  {token.token}')

asyncio.run(setup())
"
```
