# Examples

Runnable scripts for the Kagura Memory SDK clients. All read credentials
from the environment:

```bash
export KAGURA_API_KEY="kagura_..."
export KAGURA_MCP_URL="http://localhost:8080/mcp/w/{workspace_id}"
uv run python examples/<script>.py
```

| Script | Client | Shows |
|--------|--------|-------|
| [`client_basics.py`](client_basics.py) | `KaguraClient` | remember / recall / explore / reference / forget |
| [`client_advanced.py`](client_advanced.py) | `KaguraClient` | recall filters, cross-context recall, `list_tags` (+ `with_tags` drill-down), `recall_nearby` (WHERE axis), `update_memory`, supersede/history, `get_usage`, `get_memory_stats`, `find_duplicates`, `merge_contexts` |
| [`agent_bootstrap.py`](agent_bootstrap.py) | `KaguraClient` / `AgentsClient` | one-call agent session-start rehydration (`get_agent_bootstrap`, MCP + REST; server v0.49.0+) |
| [`resource_tokens.py`](resource_tokens.py) | `ResourceClient` | resource setup, token lifecycle, single + batch event ingestion |
| [`files_upload.py`](files_upload.py) | `FilesClient` | upload (bytes + dedup), `download_url`, `list`, `delete` |
| [`ingest_pdf.py`](ingest_pdf.py) | `FileIngestor` | PDF → overview + section memories + R2 archive |
| [`ingest_documents.py`](ingest_documents.py) | `FileIngestor` | text/Markdown/HTML/DOCX/XLSX/PPTX/EPUB → memory graph |
| [`ingest_audio.py`](ingest_audio.py) | `FileIngestor` | audio/video → transcript → memory graph |
| [`ingest_youtube.py`](ingest_youtube.py) | `FileIngestor` | YouTube captions → memory graph |
| [`ingest_rendered_url.py`](ingest_rendered_url.py) | `FileIngestor` | JS-rendered web page → memory graph |

`ingest_pdf.py` needs the ingest extras (`pip install 'kagura-memory[ingest-pdf]'`)
and a text-LLM key (`ANTHROPIC_API_KEY` for the default `claude` provider).
`ingest_documents.py` works with any supported format — install the matching
extra (or `[ingest-all]`); plain text / Markdown need only the base
`[ingest]` extra (which carries `litellm` for summarization since v0.37.0).
`client_advanced.py` spans several server floors: `list_tags()` needs
memory-cloud v0.15.4+, its `with_tags` drill-down v0.17.2+, supersede
(`supersedes` / `include_superseded`) v0.45.0+, and `recall_nearby` /
`details.location` v0.53.0+. Against an older server those specific calls
return an MCP "tool not found"; the rest of the script still runs. Its
WHERE-axis block writes two demo memories and forgets them in a `finally`.

`SecretClient` (zero-knowledge secrets) has no standalone script — it is
CLI-first, since real use needs OS-keychain key custody and the put/get/grant
misuse guards. Drive it with `kagura secret` (needs the `[secret]` extra and
memory-cloud v0.39.0+); see the
[`kagura secret`](../README.md#zero-knowledge-secrets-kagura-secret) and
[`SecretClient`](../README.md#secretclient--zero-knowledge-secrets) sections of
the main README.
