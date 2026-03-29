#!/usr/bin/env python3
"""ResourceClient usage — setup, token management, and data ingestion.

Usage:
    export KAGURA_API_KEY="kagura_..."
    export KAGURA_MCP_URL="http://localhost:8080/mcp/w/{workspace_id}"
    uv run python examples/resource_tokens.py
"""

import asyncio
import os
import sys

from kagura_memory import ResourceClient, ResourceEventRequest


async def main():
    api_key = os.getenv("KAGURA_API_KEY")
    mcp_url = os.getenv("KAGURA_MCP_URL", "http://localhost:8080/mcp")

    if not api_key:
        print("Error: KAGURA_API_KEY required")
        sys.exit(1)

    async with ResourceClient.from_mcp_url(api_key=api_key, mcp_url=mcp_url) as client:
        # One-call setup: create public context + set resource_id + create token.
        # In production, save the token and reuse it — don't call setup_resource() every time.
        token = await client.setup_resource(
            resource_id="example-products",
            summary="Example product catalog",
            description="Example script token",
            quota_events_per_hour=100,
        )
        print(f"Setup complete! Token: {token.token[:30]}...")

        # Ingest single event
        event = ResourceEventRequest(
            op="upsert",
            doc_id="EXAMPLE-001",
            version=1,
            payload={"name": "Example Product", "price": 1980},
        )
        result = await client.ingest_event("example-products", token.token, event)
        print(f"Ingested: event_id={result.event_id}")

        # Batch ingest
        events = [
            ResourceEventRequest(
                op="upsert",
                doc_id=f"EXAMPLE-{i}",
                version=1,
                payload={"name": f"Product {i}", "price": i * 100},
            )
            for i in range(2, 6)
        ]
        batch = await client.ingest_events("example-products", token.token, events)
        print(f"Batch: {batch.created_count} created, {batch.failed_count} failed")

        # List tokens
        tokens = await client.list_tokens(resource_id="example-products")
        print(f"Tokens: {tokens.total}")

        # Cleanup
        await client.revoke_token(token.id)
        print("Token revoked.")


if __name__ == "__main__":
    asyncio.run(main())
