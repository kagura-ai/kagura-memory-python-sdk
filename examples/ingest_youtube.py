#!/usr/bin/env python3
"""FileIngestor usage — turn a YouTube video's transcript into a memory graph.

A YouTube URL is auto-detected and resolved to its captions (manual preferred,
auto-generated as fallback), formatted as time-windowed Markdown, then ingested
as one overview memory plus per-section summaries — the same flow as documents.
Only the transcript is fetched; the media stream is never downloaded.

Requires the YouTube ingest extra and a text-LLM key:

    pip install 'kagura-memory[ingest-youtube]'
    export KAGURA_API_KEY="kagura_..."
    export KAGURA_MCP_URL="http://localhost:8080/mcp/w/{workspace_id}"
    export ANTHROPIC_API_KEY="sk-ant-..."          # default 'claude' text provider
    uv run python examples/ingest_youtube.py "https://www.youtube.com/watch?v=jNQXAC9IVRw"

Notes:
    * Single videos only — playlists / channels are rejected.
    * Caption-disabled, age-restricted, or unavailable videos raise a clear error.
"""

import asyncio
import os
import sys

from kagura_memory import FileIngestor, KaguraClient


async def main() -> None:
    api_key = os.getenv("KAGURA_API_KEY")
    mcp_url = os.getenv("KAGURA_MCP_URL", "http://localhost:8080/mcp")
    url = sys.argv[1] if len(sys.argv) > 1 else "https://www.youtube.com/watch?v=jNQXAC9IVRw"

    if not api_key:
        print("Error: KAGURA_API_KEY required")
        sys.exit(1)

    async with KaguraClient(api_key=api_key, mcp_url=mcp_url) as client:
        contexts = await client.list_contexts()
        if not contexts.get("contexts"):
            print("No contexts found. Create one first.")
            return
        context_id = contexts["contexts"][0]["id"]

        ingestor = FileIngestor(client=client)
        result = await ingestor.ingest(url, context_id=context_id)

        if result.success:
            print(f"Ingested {url}")
            print(f"  overview memory: {result.overview_id}")
            print(f"  sections:        {len(result.section_ids)}")
        else:
            print(f"Ingest failed for {url}")
            for err in result.errors:
                print(f"  [{err.step}] {err.message}")


if __name__ == "__main__":
    asyncio.run(main())
