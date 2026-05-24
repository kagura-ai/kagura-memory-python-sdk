#!/usr/bin/env python3
"""FileIngestor usage — turn a PDF into a memory graph + R2 archive.

A PDF becomes one overview memory plus per-section summaries linked by
``declared_link`` edges, and the original bytes are archived to the
workspace's object store (so ``download_url`` can pull them back later).

Phase 1 ingests text only. Requires the ingest extras and a text-LLM key:

    pip install 'kagura-memory[ingest-pdf]'
    export KAGURA_API_KEY="kagura_..."
    export KAGURA_MCP_URL="http://localhost:8080/mcp/w/{workspace_id}"
    export ANTHROPIC_API_KEY="sk-ant-..."          # default 'claude' text provider
    uv run python examples/ingest_pdf.py ./report.pdf
"""

import asyncio
import os
import sys

from kagura_memory import FileIngestor, FilesClient, KaguraClient


async def main():
    api_key = os.getenv("KAGURA_API_KEY")
    mcp_url = os.getenv("KAGURA_MCP_URL", "http://localhost:8080/mcp")
    source = sys.argv[1] if len(sys.argv) > 1 else "./report.pdf"

    if not api_key:
        print("Error: KAGURA_API_KEY required")
        sys.exit(1)

    async with (
        KaguraClient(api_key=api_key, mcp_url=mcp_url) as client,
        FilesClient.from_mcp_url(api_key=api_key, mcp_url=mcp_url) as files,
    ):
        contexts = await client.list_contexts()
        if not contexts.get("contexts"):
            print("No contexts found. Create one first.")
            return
        ctx = contexts["contexts"][0]["id"]
        print(f"Ingesting {source} into context {ctx}")

        # Pass files_client so the original is archived to R2.
        ingestor = FileIngestor(client=client, files_client=files)
        result = await ingestor.ingest(
            source,
            context_id=ctx,
            tags=["example", "ingest"],
        )

        if not result.success:
            print(f"Ingestion failed (no overview created). Errors: {result.errors}")
            sys.exit(1)

        print(f"Overview memory: {result.overview_id}")
        print(f"Sections:        {len(result.section_ids)}")
        print(f"Archived file:   {result.archived_file_id}")
        print(f"Est. cost (USD): {result.cost.est_usd}")
        if result.errors:
            print(f"Partial errors:  {len(result.errors)} (see result.errors)")


if __name__ == "__main__":
    asyncio.run(main())
