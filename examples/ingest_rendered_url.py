#!/usr/bin/env python3
"""Ingest a JS-heavy / SPA page via browser-rendered fetch (issue #145).

Some pages render their real content client-side with JavaScript, so a plain
HTTP fetch only sees an empty shell. Passing ``render=True`` drives a headless
Chromium (via Playwright) to load and render the page first, then hands the
rendered HTML to the normal HTML extractor → chunker → provider pipeline.

This path is OPT-IN and needs the browser extra plus the Chromium binary:

    pip install 'kagura-memory[ingest-browser]'
    playwright install chromium                    # one-time browser download
    export KAGURA_API_KEY="kagura_..."
    export KAGURA_MCP_URL="http://localhost:8080/mcp/w/{workspace_id}"
    export ANTHROPIC_API_KEY="sk-ant-..."          # default 'claude' text provider
    uv run python examples/ingest_rendered_url.py https://spa.example.com/

Security: the browser fetch re-resolves every request the page makes
(navigation, redirects, XHR/fetch, sub-resources) against the same SSRF
denylist as the plain fetcher, and enforces the rendered-HTML size cap and
the navigation timeout. ``render=True`` is ignored for local file paths.
"""

import asyncio
import os
import sys

from kagura_memory import FileIngestor, KaguraClient


async def main():
    api_key = os.getenv("KAGURA_API_KEY")
    mcp_url = os.getenv("KAGURA_MCP_URL", "http://localhost:8080/mcp")
    source = sys.argv[1] if len(sys.argv) > 1 else "https://example.com/"

    if not api_key:
        print("Error: KAGURA_API_KEY required")
        sys.exit(1)

    async with KaguraClient(api_key=api_key, mcp_url=mcp_url) as client:
        contexts = await client.list_contexts()
        if not contexts.get("contexts"):
            print("No contexts found. Create one first.")
            return
        ctx = contexts["contexts"][0]["id"]
        print(f"Rendering and ingesting {source} into context {ctx}")

        ingestor = FileIngestor(client=client)
        # render=True drives headless Chromium before extraction.
        result = await ingestor.ingest(
            source,
            context_id=ctx,
            tags=["example", "ingest", "rendered"],
            render=True,
        )

        if not result.success:
            # A missing [ingest-browser] extra surfaces here as a
            # step="fetch" error record (not an uncaught exception).
            print(f"Ingestion failed (no overview created). Errors: {result.errors}")
            sys.exit(1)

        print(f"Overview memory: {result.overview_id}")
        print(f"Sections:        {len(result.section_ids)}")
        if result.errors:
            print(f"Partial errors:  {len(result.errors)} (see result.errors)")


if __name__ == "__main__":
    asyncio.run(main())
