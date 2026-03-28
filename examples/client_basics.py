#!/usr/bin/env python3
"""Basic KaguraClient usage — direct memory operations.

Usage:
    export KAGURA_API_KEY="kagura_..."
    export KAGURA_MCP_URL="http://localhost:8080/mcp/w/{workspace_id}"
    uv run python examples/client_basics.py
"""

import asyncio
import os
import sys

from kagura_memory import KaguraClient


async def main():
    api_key = os.getenv("KAGURA_API_KEY")
    mcp_url = os.getenv("KAGURA_MCP_URL", "http://localhost:8080/mcp")

    if not api_key:
        print("Error: KAGURA_API_KEY required")
        sys.exit(1)

    async with KaguraClient(api_key=api_key, mcp_url=mcp_url) as client:
        # List contexts
        contexts = await client.list_contexts()
        print(f"Contexts: {contexts.get('count', 0)}")
        if not contexts.get("contexts"):
            print("No contexts found. Create one first.")
            return

        ctx = contexts["contexts"][0]["id"]
        print(f"Using: {ctx}")

        # Remember
        result = await client.remember(
            context_id=ctx,
            summary="Python SDK test",
            content="Testing the Kagura Memory SDK client.",
            type="note",
            importance=0.3,
            tags=["test"],
        )
        memory_id = result["memory_id"]
        print(f"Remembered: {memory_id}")

        # Recall
        results = await client.recall(context_id=ctx, query="SDK test", k=3)
        for mem in results.get("results", []):
            print(f"  [{mem['score']:.2f}] {mem['summary'][:60]}")

        # Explore
        related = await client.explore(context_id=ctx, memory_id=memory_id)
        print(f"Related: {len(related.get('memories', []))} memories")

        # Reference
        detail = await client.reference(context_id=ctx, memory_id=memory_id)
        print(f"Detail: {detail.get('summary', '')}")

        # Cleanup
        await client.forget(context_id=ctx, memory_id=memory_id)
        print(f"Deleted: {memory_id}")


if __name__ == "__main__":
    asyncio.run(main())
