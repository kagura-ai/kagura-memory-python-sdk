#!/usr/bin/env python3
"""ResourceClient usage — token management and external data ingestion.

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
    resource_id = os.getenv("KAGURA_RESOURCE_ID", "sdk-test")

    if not api_key:
        print("Error: KAGURA_API_KEY required")
        sys.exit(1)

    async with ResourceClient.from_mcp_url(api_key=api_key, mcp_url=mcp_url) as client:
        # Create token
        token = await client.create_token(
            resource_id=resource_id,
            description="Example script token",
            quota_events_per_hour=100,
        )
        print(f"Token created: {token.token[:30]}...")

        # Ingest single event
        event = ResourceEventRequest(
            op="upsert",
            doc_id="EXAMPLE-001",
            version=1,
            payload={"name": "Example Product", "price": 1980},
        )
        result = await client.ingest_event(resource_id, token.token, event)
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
        batch = await client.ingest_events(resource_id, token.token, events)
        print(f"Batch: {batch.created_count} created, {batch.failed_count} failed")

        # List tokens
        tokens = await client.list_tokens(resource_id=resource_id)
        print(f"Tokens: {tokens.total}")

        # Cleanup
        await client.revoke_token(token.id)
        print("Token revoked.")


if __name__ == "__main__":
    asyncio.run(main())
