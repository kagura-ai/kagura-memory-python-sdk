# Examples

Runnable scripts for each Kagura Memory SDK client. All read credentials
from the environment:

```bash
export KAGURA_API_KEY="kagura_..."
export KAGURA_MCP_URL="http://localhost:8080/mcp/w/{workspace_id}"
uv run python examples/<script>.py
```

| Script | Client | Shows |
|--------|--------|-------|
| [`client_basics.py`](client_basics.py) | `KaguraClient` | remember / recall / explore / reference / forget |
| [`client_advanced.py`](client_advanced.py) | `KaguraClient` | recall filters, cross-context recall, `list_tags`, `get_usage`, `get_memory_stats`, `find_duplicates`, `merge_contexts` |
| [`agent_session.py`](agent_session.py) | `KaguraAgent` | AI-powered session analysis (remember/recall decided by the LLM) |
| [`resource_tokens.py`](resource_tokens.py) | `ResourceClient` | resource setup, token lifecycle, single + batch event ingestion |
| [`files_upload.py`](files_upload.py) | `FilesClient` | upload (bytes + dedup), `download_url`, `list`, `delete` |
| [`ingest_pdf.py`](ingest_pdf.py) | `FileIngestor` | PDF → overview + section memories + R2 archive |

`ingest_pdf.py` needs the ingest extras (`pip install 'kagura-memory[ingest-pdf]'`)
and a text-LLM key (`ANTHROPIC_API_KEY` for the default `claude` provider).
`client_advanced.py`'s `list_tags()` needs memory-cloud server v0.15.4+.
