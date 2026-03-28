#!/usr/bin/env python3
"""Live SDK test against running Kagura Memory Cloud instance.

Usage:
    # Set environment variables or use .kagura.json
    export KAGURA_API_KEY="kagura_..."
    export KAGURA_MCP_URL="http://localhost:8080/mcp/w/{workspace_id}"

    python examples/test_sdk_live.py
"""

import asyncio
import os
import sys

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kagura_memory import KaguraClient


async def main():
    api_key = os.getenv("KAGURA_API_KEY")
    mcp_url = os.getenv("KAGURA_MCP_URL", "http://localhost:8080/mcp")

    if not api_key:
        print("Error: KAGURA_API_KEY environment variable required")
        print("Create an API key at http://localhost:3000 → Integrations → API Keys")
        sys.exit(1)

    print(f"Connecting to: {mcp_url}")

    async with KaguraClient(api_key=api_key, mcp_url=mcp_url) as client:
        # 1. List contexts
        print("\n--- list_contexts ---")
        contexts = await client.list_contexts()
        print(f"Found {contexts.get('count', 0)} contexts:")
        for ctx in contexts.get("contexts", []):
            print(f"  - {ctx['name']} ({ctx['id']})")

        if not contexts.get("contexts"):
            print("No contexts found. Create one first.")
            return

        context_id = contexts["contexts"][0]["id"]
        print(f"\nUsing context: {context_id}")

        # 2. Remember
        print("\n--- remember ---")
        result = await client.remember(
            context_id=context_id,
            summary="SDK live test memory",
            content="This memory was created by the SDK live test script.",
            type="note",
            importance=0.3,
            tags=["sdk-test", "live"],
        )
        memory_id = result.get("memory_id")
        print(f"Created memory: {memory_id}")

        # 3. Recall
        print("\n--- recall ---")
        results = await client.recall(
            context_id=context_id,
            query="SDK live test",
            k=3,
        )
        print(f"Found {results.get('count', 0)} results:")
        for mem in results.get("results", [])[:3]:
            print(f"  - [{mem.get('score', 0):.2f}] {mem.get('summary', '')[:60]}")

        # 4. Recall with rerank
        print("\n--- recall (with rerank) ---")
        results_rerank = await client.recall(
            context_id=context_id,
            query="SDK live test",
            k=3,
            use_rerank=True,
        )
        print(f"Found {results_rerank.get('count', 0)} results (reranked)")

        # 5. Get tool definitions
        print("\n--- get_tool_definitions ---")
        tools = await client.get_tool_definitions()
        print(f"Available tools: {[t['name'] for t in tools]}")

        # 6. Forget (cleanup)
        if memory_id:
            print("\n--- forget (cleanup) ---")
            await client.forget(context_id=context_id, memory_id=memory_id)
            print(f"Deleted memory: {memory_id}")

    print("\n✓ All SDK operations completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
