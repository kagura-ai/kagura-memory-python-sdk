"""Tests for KaguraAgent enhanced context and caching."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import time

from kagura_memory import KaguraAgent
from kagura_memory.models import Session, Message


@pytest.mark.asyncio
async def test_agent_caching_tools():
    """Test that tool definitions are cached and reused."""
    agent = KaguraAgent(api_key="test-key", model="gpt-5.4-nano")

    # Mock tool definitions
    mock_tools = [
        {"name": "remember", "description": "Store info"},
        {"name": "recall", "description": "Search info"},
    ]

    with patch.object(
        agent.client, "get_tool_definitions", new_callable=AsyncMock
    ) as mock_get_tools:
        mock_get_tools.return_value = mock_tools

        # First call - should fetch
        tools1 = await agent._get_tools_with_cache()
        assert tools1 == mock_tools
        assert mock_get_tools.call_count == 1

        # Second call - should use cache
        tools2 = await agent._get_tools_with_cache()
        assert tools2 == mock_tools
        assert mock_get_tools.call_count == 1  # Not called again

        # Verify cache entry exists with timestamp
        assert "tools" in agent._cache
        cached_data, cached_timestamp = agent._cache["tools"]
        assert cached_data == mock_tools
        assert cached_timestamp > 0

    await agent.close()


@pytest.mark.asyncio
async def test_agent_cache_expiration():
    """Test that cache expires after TTL."""
    agent = KaguraAgent(api_key="test-key", model="gpt-5.4-nano")
    agent._cache_ttl = 0.1  # 100ms TTL for testing

    mock_tools = [{"name": "remember", "description": "Store info"}]

    with patch.object(
        agent.client, "get_tool_definitions", new_callable=AsyncMock
    ) as mock_get_tools:
        mock_get_tools.return_value = mock_tools

        # First call
        await agent._get_tools_with_cache()
        assert mock_get_tools.call_count == 1

        # Verify cache entry exists
        assert "tools" in agent._cache

        # Wait for cache to expire
        time.sleep(0.15)

        # Second call - should fetch again (cache expired)
        await agent._get_tools_with_cache()
        assert mock_get_tools.call_count == 2

    await agent.close()


@pytest.mark.asyncio
async def test_build_enhanced_context():
    """Test building enhanced context with tools and contexts."""
    agent = KaguraAgent(api_key="test-key", model="gpt-5.4-nano")

    mock_tools = [{"name": "remember"}]
    mock_contexts = [{"id": "ctx-1", "name": "test"}]

    with patch.object(
        agent.client, "get_tool_definitions", new_callable=AsyncMock
    ) as mock_get_tools, patch.object(
        agent.client, "list_contexts", new_callable=AsyncMock
    ) as mock_list_contexts:
        mock_get_tools.return_value = mock_tools
        mock_list_contexts.return_value = {"contexts": mock_contexts}

        # Build enhanced context
        enhanced_ctx = await agent._build_enhanced_context()

        assert enhanced_ctx["tools"] == mock_tools
        assert enhanced_ctx["contexts"] == mock_contexts

    await agent.close()


@pytest.mark.asyncio
async def test_analyze_session_with_enhanced_context():
    """Test that _analyze_session uses enhanced context by default."""
    agent = KaguraAgent(api_key="test-key", model="gpt-5.4-nano", context_id="ctx-1")

    session = Session(messages=[Message(role="user", content="Test message")])

    mock_tools = [{"name": "remember"}]
    mock_contexts = [{"id": "ctx-1", "name": "test"}]

    with patch.object(
        agent.client, "get_tool_definitions", new_callable=AsyncMock
    ) as mock_get_tools, patch.object(
        agent.client, "list_contexts", new_callable=AsyncMock
    ) as mock_list_contexts, patch.object(
        agent, "_call_llm", new_callable=AsyncMock
    ) as mock_llm:
        mock_get_tools.return_value = mock_tools
        mock_list_contexts.return_value = {"contexts": mock_contexts}
        mock_llm.return_value = (
            {"should_remember": False, "should_recall": False},
            MagicMock(),
        )

        # Analyze with enhanced context (default)
        await agent._analyze_session(session, use_enhanced_context=True)

        # Verify enhanced context was built
        assert mock_get_tools.called
        assert mock_list_contexts.called

        # Verify LLM was called with enhanced prompt
        assert mock_llm.called
        call_args = mock_llm.call_args
        user_content = call_args[0][0][1]["content"]
        assert "Available Kagura Memory Tools:" in user_content

    await agent.close()


@pytest.mark.asyncio
async def test_independent_cache_timestamps():
    """Test that tools and contexts have independent cache timestamps (bug fix)."""
    agent = KaguraAgent(api_key="test-key", model="gpt-5.4-nano")

    mock_tools = [{"name": "remember"}]
    mock_contexts = [{"id": "ctx-1", "name": "test"}]

    with patch.object(
        agent.client, "get_tool_definitions", new_callable=AsyncMock
    ) as mock_get_tools, patch.object(
        agent.client, "list_contexts", new_callable=AsyncMock
    ) as mock_list_contexts:
        mock_get_tools.return_value = mock_tools
        mock_list_contexts.return_value = {"contexts": mock_contexts}

        # Fetch tools at t=0
        await agent._get_tools_with_cache()
        tools_entry_1 = agent._cache["tools"]
        tools_timestamp_1 = tools_entry_1[1]

        # Wait a bit
        time.sleep(0.05)

        # Fetch contexts at t=50ms
        await agent._get_contexts_with_cache()
        contexts_entry = agent._cache["contexts"]
        contexts_timestamp = contexts_entry[1]

        # Fetch tools again (should use cache)
        await agent._get_tools_with_cache()
        tools_entry_2 = agent._cache["tools"]
        tools_timestamp_2 = tools_entry_2[1]

        # CRITICAL: tools timestamp should NOT be affected by contexts fetch
        assert tools_timestamp_1 == tools_timestamp_2, (
            "Tools cache timestamp should not change when contexts are fetched"
        )
        assert contexts_timestamp > tools_timestamp_1, (
            "Contexts should have newer timestamp than tools"
        )

        # Verify each cache entry has its own data
        assert agent._cache["tools"][0] == mock_tools
        assert agent._cache["contexts"][0] == mock_contexts

    await agent.close()
