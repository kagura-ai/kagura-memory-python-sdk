#!/usr/bin/env python3
"""Agent bootstrap — one-call session-start rehydration (server v0.49.0+).

``get_agent_bootstrap`` composes context guide + pinned memories +
trusted-only recall (only when a query is supplied) + upcoming time
memories + agent run-state into a single envelope, so an agent starts a
session with one round-trip instead of five. Components are fail-soft:
a failing component reports status="error" while the rest still return,
with the top-level ``degraded`` flag set.

Shown here over both surfaces:
- ``KaguraClient.get_agent_bootstrap`` — the MCP tool
- ``AgentsClient.bootstrap`` — the REST companion
  (``POST /api/v1/agents/{agent_id}/bootstrap``), for API-key-only
  callers such as agent-bound member keys

Usage:
    export KAGURA_API_KEY="kagura_..."
    export KAGURA_MCP_URL="http://localhost:8080/mcp/w/{workspace_id}"
    export KAGURA_AGENT_ID="<agent uuid from the registry>"
    uv run python examples/agent_bootstrap.py
"""

import asyncio
import os
import sys

from kagura_memory import AgentsClient, KaguraClient


async def main():
    agent_id = os.getenv("KAGURA_AGENT_ID")
    if not agent_id:
        print("Error: KAGURA_AGENT_ID required (register the agent first)")
        sys.exit(1)

    # --- MCP surface -----------------------------------------------------
    async with KaguraClient() as client:
        bootstrap = await client.get_agent_bootstrap(
            agent_id,
            # context_id omitted → the agent's default binding
            session_id="example-session-001",
            query="current task and recent decisions",  # enables the recall component
            recall_k=5,
        )

        print(f"Agent:       {bootstrap.agent.name} ({bootstrap.agent.agent_id})")
        if bootstrap.context:
            print(f"Context:     {bootstrap.context.name}")
        print(f"Degraded:    {bootstrap.degraded}")
        for name, component in bootstrap.components.items():
            print(f"  {name:<9} status={component.get('status')}")

        pinned = bootstrap.components.get("pinned", {})
        for memory in pinned.get("memories", []):
            print(f"  pinned: {memory['summary']}")

    # --- REST companion ----------------------------------------------------
    # Same arguments, same envelope — for callers holding an API key but no
    # MCP session (e.g. an agent-bound member key minted for a deployed agent).
    async with AgentsClient.from_mcp_url() as agents:
        bootstrap = await agents.bootstrap(
            agent_id,
            include=["pinned", "state"],  # cheap subset, no recall
        )
        print(
            f"REST bootstrap degraded={bootstrap.degraded}, "
            f"components={sorted(bootstrap.components)}"
        )


if __name__ == "__main__":
    asyncio.run(main())
