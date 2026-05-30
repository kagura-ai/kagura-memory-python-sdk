#!/usr/bin/env python3
"""FileIngestor usage — transcribe audio/video into a memory graph (Issue #147).

Audio and video files have NO parseable text — a transcript is *generated* by
Gemini (``gemini/gemini-2.5-flash``). ``FileIngestor.ingest("talk.mp3", ...)``
routes ``.mp3`` / ``.wav`` / ``.m4a`` / ``.mp4`` to that transcription path,
which returns timestamped segments assembled into time-windowed sections, then
runs the SAME chunk → summarize → remember pipeline as every other format.

v1 is a single inline request. Because the media is sent base64-encoded (~33%
larger than raw), the effective cap is ~15 MiB of RAW audio/video so the encoded
payload fits Gemini's ~20 MiB inline-request limit; larger files are rejected up
front. Splitting longer media with ffmpeg is a planned follow-up.

    pip install 'kagura-memory[ingest-audio]'
    export KAGURA_API_KEY="kagura_..."
    export KAGURA_MCP_URL="http://localhost:8080/mcp/w/{workspace_id}"
    export GEMINI_API_KEY="..."                    # required for transcription
    export ANTHROPIC_API_KEY="sk-ant-..."          # default 'claude' summarizer
    uv run python examples/ingest_audio.py ./talk.mp3
"""

import asyncio
import os
import sys

from kagura_memory import FileIngestor, KaguraClient


async def main():
    api_key = os.getenv("KAGURA_API_KEY")
    mcp_url = os.getenv("KAGURA_MCP_URL", "http://localhost:8080/mcp")
    source = sys.argv[1] if len(sys.argv) > 1 else "./talk.mp3"

    if not api_key:
        print("Error: KAGURA_API_KEY required")
        sys.exit(1)
    if not os.getenv("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY required for audio transcription")
        sys.exit(1)

    async with KaguraClient(api_key=api_key, mcp_url=mcp_url) as client:
        contexts = await client.list_contexts()
        if not contexts.get("contexts"):
            print("No contexts found. Create one first.")
            return
        ctx = contexts["contexts"][0]["id"]
        print(f"Transcribing + ingesting {source} into context {ctx}")

        ingestor = FileIngestor(client=client)
        result = await ingestor.ingest(
            source,
            context_id=ctx,
            tags=["example", "audio"],
        )

        if not result.success:
            print(f"Ingestion failed (no overview created). Errors: {result.errors}")
            sys.exit(1)

        print(f"Overview memory:   {result.overview_id}")
        print(f"Transcript sections: {len(result.section_ids)}")
        # Audio cost is real provider token usage (audio is metered at
        # ~32 tokens/sec inside prompt_tokens).
        print(f"Prompt tokens:     {result.cost.prompt_tokens}")
        print(f"Completion tokens: {result.cost.completion_tokens}")
        if result.errors:
            print(f"Partial errors:    {len(result.errors)} (see result.errors)")


if __name__ == "__main__":
    asyncio.run(main())
