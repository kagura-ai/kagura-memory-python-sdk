#!/usr/bin/env python3
"""Advanced KaguraClient usage — filters, cross-context, tags, geo, history.

Goes beyond examples/client_basics.py: recall filters (tags / date
range), cross-context recall, tag-vocabulary discovery and faceted
drill-down, workspace usage, per-memory stats, duplicate detection, the
WHERE axis (``recall_nearby``), and supersede/history.

Server floors: ``list_tags()`` needs memory-cloud v0.15.4+, its
``with_tags`` drill-down v0.17.2+, ``supersedes`` / ``include_superseded``
v0.45.0+, and ``recall_nearby`` / ``details.location`` v0.53.0+. Against an
older server those calls raise ``KaguraConnectionError`` ("MCP error: Tool not
found") — MCP-level errors surface as a *connection* error even though the
connection is fine; it subclasses ``KaguraError``. This script does not catch
them, so it stops at the first surface the server lacks.

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

        # Faceted drill-down — restrict counts to memories carrying ALL listed
        # tags, aggregated server-side (no local index). The with_tags values
        # are themselves excluded from the returned vocabulary. Server v0.17.2+.
        if tags.tags:
            facet = tags.tags[0].tag
            drill = await client.list_tags(context_id=ctx, with_tags=[facet])
            print(f"Co-occurring with {facet!r}: {[t.tag for t in drill.tags]}")

        # Workspace usage — quota check.
        usage = await client.get_usage()
        print(f"Plan: {usage.plan}, memories {usage.memories.used}/{usage.memories.limit}")

        # Per-memory stats — recall frequency / access patterns.
        stats = await client.get_memory_stats(context_id=ctx, sort_by="use_count", limit=5)
        print(f"Top memories by use_count: {stats.total} total")

        # Duplicate detection — surface near-identical pairs.
        dupes = await client.find_duplicates(context_id=ctx, threshold=0.90)
        print(f"Duplicate pairs: {dupes.total_pairs} (scanned {dupes.memories_scanned})")

        # --- WHERE axis + supersede (writes two memories, cleans up after itself) ---
        # Attach a location. lat/lon must be JSON *numbers* — the server rejects
        # string-typed coordinates with a 422 by design.
        geo = await client.remember(
            context_id=ctx,
            summary="Example: coffee with Sato",
            content="WHERE-axis demo memory.",
            details={"location": {"lat": 35.68, "lon": 139.76, "label": "Tokyo HQ"}},
        )
        geo_id, newer_id = geo["memory_id"], None
        try:
            # Deterministic spatial query, nearest first — each result carries
            # distance_m. radius_m/k are keyword-only. Needs server v0.53.0+.
            near = await client.recall_nearby(context_id=ctx, lat=35.68, lon=139.76, radius_m=500)
            print(f"Nearby: {[(r['summary'], r['distance_m']) for r in near.get('results', [])]}")

            # update_memory REPLACES details wholesale — there is no deep-merge.
            # Spread the current value, or location is dropped and the memory
            # silently disappears from recall_nearby.
            current = (await client.reference(context_id=ctx, memory_id=geo_id))["memory"]
            await client.update_memory(
                context_id=ctx,
                memory_id=geo_id,
                details={**(current.get("details") or {}), "attendees": ["Sato"]},
            )

            # Supersede — the newer memory shadows the old one rather than
            # deleting it, so the history stays restorable. Server v0.45.0+.
            newer = await client.remember(
                context_id=ctx,
                summary="Example: coffee with Sato (rescheduled)",
                content="Supersede demo memory.",
                supersedes=geo_id,
            )
            newer_id = newer["memory_id"]

            # Read the shadowed version back. NOTE: include_superseded is a
            # TOP-LEVEL argument, not a filters key.
            shadowed = await client.recall(
                context_id=ctx, query="coffee with Sato", include_superseded=True
            )
            print(f"Hits including superseded: {len(shadowed.get('results', []))}")
        finally:
            # Keep the workspace clean even if the server predates a surface above.
            for mid in (newer_id, geo_id):
                if mid:
                    await client.forget(context_id=ctx, memory_id=mid)

        # Merge contexts (opt-in, destructive — copies memories source -> target).
        src, dst = os.getenv("KAGURA_MERGE_SOURCE"), os.getenv("KAGURA_MERGE_TARGET")
        if src and dst:
            result = await client.merge_contexts(source_id=src, target_id=dst)
            print(f"Merged {result['merged']} memories into {dst}")


if __name__ == "__main__":
    asyncio.run(main())
