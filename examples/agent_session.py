#!/usr/bin/env python3
"""KaguraAgent usage — AI-powered session analysis.

The agent analyzes conversations and auto-decides what to remember/recall.

Usage:
    export KAGURA_API_KEY="kagura_..."
    export KAGURA_MCP_URL="http://localhost:8080/mcp/w/{workspace_id}"
    export OPENAI_API_KEY="sk-..."  # or ANTHROPIC_API_KEY
    uv run python examples/agent_session.py
"""

import asyncio
import os
import sys

from kagura_memory import KaguraAgent, Message, Session


async def main():
    api_key = os.getenv("KAGURA_API_KEY")
    if not api_key:
        print("Error: KAGURA_API_KEY required")
        sys.exit(1)

    agent = KaguraAgent(
        api_key=api_key,
        mcp_url=os.getenv("KAGURA_MCP_URL", "http://localhost:8080/mcp"),
        model=os.getenv("KAGURA_MODEL", "gpt-5.4-nano"),
        context_id="auto",
    )

    session = Session(
        messages=[
            Message(role="user", content="FastAPIでOAuth2を実装したい"),
            Message(role="assistant", content="Authlibを使うパターンが推奨です。"),
            Message(role="user", content="なるほど、これ覚えておいて"),
        ]
    )

    async with agent:
        result = await agent.process(session, deep=True, verbose=2)

    print(f"\nRemembered: {len(result.remembered)}")
    print(f"Recalled:   {len(result.recalled)}")
    print(f"Explored:   {len(result.explored)}")
    print(f"Context:    {result.context_used}")
    if result.llm_usage:
        print(f"Tokens:     {result.llm_usage.total_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
