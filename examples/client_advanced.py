#!/usr/bin/env python3
"""Advanced KaguraClient usage — filters, cross-context, tags, stats.

Goes beyond examples/client_basics.py: recall filters (tags / date
range), cross-context recall, tag-vocabulary discovery, workspace usage,
per-memory stats, and duplicate detection. ``list_tags()`` needs
memory-cloud server v0.15.4+.

Usage:
    export KAGURA_API_KEY="kagura_..."
    export KAGURA_MCP_URL="http://localhost:8080/mcp/w/{workspace_id}"
    uv run python examples/client_advanced.py

    # Optional: also demonstrate merge_contexts (mutates data — copies
    # all memories from SOURCE into TARGET). Off by default.
    export KAGURA_MERGE_SOURCE="..." KAGURA_MERGE_TARGET="..."
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
        contexts = await client.list_contexts()
        ctx_ids = [c["id"] for c in contexts.get("contexts", [])]
        if not ctx_ids:
            print("No contexts found. Create one first.")
            return
        ctx = ctx_ids[0]
        print(f"Primary context: {ctx}")

        # Tag AND filter — match memories carrying ALL listed tags.
        tagged = await client.recall(
            context_id=ctx,
            query="budget",
            filters={"tags": ["予算", "2026"], "tags_match": "all"},
        )
        print(f"Tag-filtered hits: {len(tagged.get('results', []))}")

        # Date-range filter.
        recent = await client.recall(
            context_id=ctx,
            query="recent decisions",
            filters={"created_after": "2026-01-01T00:00:00Z"},
        )
        print(f"Date-filtered hits: {len(recent.get('results', []))}")

        # Cross-context recall — search several contexts at once.
        if len(ctx_ids) >= 2:
            spread = await client.recall(query="authentication", context_ids=ctx_ids[:2], k=10)
            print(f"Cross-context hits: {len(spread.get('results', []))}")

        # Tag vocabulary — discover existing spellings before remember()/recall().
        tags = await client.list_tags(context_id=ctx, sort="recent", limit=10)
        print(f"Known tags: {[(t.tag, t.count) for t in tags.tags]}")

        # Workspace usage — quota check.
        usage = await client.get_usage()
        print(f"Plan: {usage.plan}, memories {usage.memories.used}/{usage.memories.limit}")

        # Per-memory stats — recall frequency / access patterns.
        stats = await client.get_memory_stats(context_id=ctx, sort_by="use_count", limit=5)
        print(f"Top memories by use_count: {stats.total} total")

        # Duplicate detection — surface near-identical pairs.
        dupes = await client.find_duplicates(context_id=ctx, threshold=0.90)
        print(f"Duplicate pairs: {dupes.total_pairs} (scanned {dupes.memories_scanned})")

        # Merge contexts (opt-in, destructive — copies memories source -> target).
        src, dst = os.getenv("KAGURA_MERGE_SOURCE"), os.getenv("KAGURA_MERGE_TARGET")
        if src and dst:
            result = await client.merge_contexts(source_id=src, target_id=dst)
            print(f"Merged {result['merged']} memories into {dst}")


if __name__ == "__main__":
    asyncio.run(main())
