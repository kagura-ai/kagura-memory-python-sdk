"""Tests for KaguraAgent enhanced context and caching."""

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kagura_memory import KaguraAgent, KaguraAuthError
from kagura_memory.exceptions import KaguraLLMError
from kagura_memory.models import Message, Session


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

    with (
        patch.object(
            agent.client, "get_tool_definitions", new_callable=AsyncMock
        ) as mock_get_tools,
        patch.object(agent.client, "list_contexts", new_callable=AsyncMock) as mock_list_contexts,
    ):
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

    with (
        patch.object(
            agent.client, "get_tool_definitions", new_callable=AsyncMock
        ) as mock_get_tools,
        patch.object(agent.client, "list_contexts", new_callable=AsyncMock) as mock_list_contexts,
        patch.object(agent, "_call_llm", new_callable=AsyncMock) as mock_llm,
    ):
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

    with (
        patch.object(
            agent.client, "get_tool_definitions", new_callable=AsyncMock
        ) as mock_get_tools,
        patch.object(agent.client, "list_contexts", new_callable=AsyncMock) as mock_list_contexts,
    ):
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


# ============================================================================
# _call_llm tests
# ============================================================================


@pytest.mark.asyncio
async def test_call_llm_success():
    """_call_llm should parse JSON from LLM response."""
    agent = KaguraAgent(api_key="test", model="gpt-test")

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps({"should_remember": True})

    with patch("kagura_memory.agent.litellm") as mock_litellm:
        mock_litellm.acompletion = AsyncMock(return_value=mock_response)
        data, resp = await agent._call_llm([{"role": "user", "content": "test"}])

    assert data["should_remember"] is True
    assert resp == mock_response
    await agent.close()


@pytest.mark.asyncio
async def test_call_llm_auth_error():
    """_call_llm should raise KaguraLLMError on auth failure."""
    import litellm

    agent = KaguraAgent(api_key="test", model="gpt-test")

    with patch("kagura_memory.agent.litellm.acompletion", new_callable=AsyncMock) as mock:
        mock.side_effect = litellm.AuthenticationError(
            message="bad key", llm_provider="openai", model="gpt-test"
        )

        with pytest.raises(KaguraLLMError, match="authentication failed"):
            await agent._call_llm([{"role": "user", "content": "test"}])

    await agent.close()


@pytest.mark.asyncio
async def test_call_llm_json_decode_error():
    """_call_llm should raise KaguraLLMError on invalid JSON."""
    agent = KaguraAgent(api_key="test", model="gpt-test")

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "not json {"

    with patch("kagura_memory.agent.litellm.acompletion", new_callable=AsyncMock) as mock:
        mock.return_value = mock_response

        with pytest.raises(KaguraLLMError, match="Invalid JSON"):
            await agent._call_llm([{"role": "user", "content": "test"}])

    await agent.close()


# ============================================================================
# _analyze_session tests
# ============================================================================


@pytest.mark.asyncio
async def test_analyze_basic_prompt_fallback():
    """_analyze_session should fall back to basic prompt on connection error."""
    agent = KaguraAgent(api_key="test", model="gpt-test")

    from kagura_memory.exceptions import KaguraConnectionError

    with (
        patch.object(agent, "_build_enhanced_context", new_callable=AsyncMock) as mock_ctx,
        patch.object(agent, "_call_llm", new_callable=AsyncMock) as mock_llm,
    ):
        mock_ctx.side_effect = KaguraConnectionError("server down")
        mock_llm.return_value = (
            {"should_remember": False, "should_recall": True, "recall_queries": []},
            MagicMock(),
        )

        result = await agent._analyze_session(
            Session(messages=[Message(role="user", content="test")])
        )

        assert result.should_recall is True
        assert mock_llm.called

    await agent.close()


# ============================================================================
# _execute_remembers tests
# ============================================================================


@pytest.mark.asyncio
async def test_execute_remembers_success():
    """_execute_remembers should call client.remember for each memory."""
    agent = KaguraAgent(api_key="test", model="gpt-test")

    with patch.object(agent.client, "remember", new_callable=AsyncMock) as mock_rem:
        mock_rem.return_value = {"memory_id": "mem-1"}

        memories = [{"summary": "Test", "content": "Full content"}]
        remembered, actions = await agent._execute_remembers("ctx", memories)

        assert len(remembered) == 1
        assert remembered[0].memory_id == "mem-1"
        assert len(actions) == 1

    await agent.close()


@pytest.mark.asyncio
async def test_execute_remembers_auth_error():
    """_execute_remembers should propagate KaguraAuthError."""
    agent = KaguraAgent(api_key="test", model="gpt-test")

    with patch.object(agent.client, "remember", new_callable=AsyncMock) as mock_rem:
        mock_rem.side_effect = KaguraAuthError("bad key")

        with pytest.raises(KaguraAuthError):
            await agent._execute_remembers("ctx", [{"summary": "s", "content": "c"}])

    await agent.close()


# ============================================================================
# _select_context tests
# ============================================================================


@pytest.mark.asyncio
async def test_select_context_single():
    """_select_context should return the only context immediately."""
    agent = KaguraAgent(api_key="test", model="gpt-test")

    with patch.object(agent.client, "list_contexts", new_callable=AsyncMock) as mock:
        mock.return_value = {"contexts": [{"id": "ctx-1", "name": "only"}]}

        result = await agent._select_context(
            Session(messages=[Message(role="user", content="test")])
        )

    assert result == "ctx-1"
    await agent.close()


# ============================================================================
# process tests
# ============================================================================


@pytest.mark.asyncio
async def test_process_full_flow():
    """process() should run analyze -> recall -> remember."""
    agent = KaguraAgent(api_key="test", model="gpt-test", context_id="ctx")

    with (
        patch.object(agent, "_analyze_session", new_callable=AsyncMock) as mock_analyze,
        patch.object(agent, "_execute_recalls", new_callable=AsyncMock) as mock_recalls,
        patch.object(agent, "_execute_remembers", new_callable=AsyncMock) as mock_remembers,
    ):
        from kagura_memory.models import AnalysisResult

        mock_analyze.return_value = AnalysisResult(
            should_remember=True,
            memories_to_store=[{"summary": "s", "content": "c"}],
            should_recall=True,
            recall_queries=[{"query": "test"}],
        )
        mock_recalls.return_value = ([], [], ["recall: test"])
        mock_remembers.return_value = ([], ["remember: s"])

        session = Session(messages=[Message(role="user", content="hello")])
        result = await agent.process(session)

        assert result.context_used == "ctx"
        assert mock_recalls.called
        assert mock_remembers.called

    await agent.close()


@pytest.mark.asyncio
async def test_process_llm_failure_graceful():
    """process() should return empty result on LLM failure."""
    agent = KaguraAgent(api_key="test", model="gpt-test", context_id="ctx")

    with patch.object(agent, "_analyze_session", new_callable=AsyncMock) as mock:
        mock.side_effect = KaguraLLMError("LLM down")

        session = Session(messages=[Message(role="user", content="hello")])
        result = await agent.process(session)

        assert result.context_used == "ctx"
        assert len(result.remembered) == 0
        assert "error" in result.actions[0]

    await agent.close()


@pytest.mark.asyncio
async def test_process_context_required():
    """process() should raise ValueError if no context provided."""
    agent = KaguraAgent(api_key="test", model="gpt-test")

    with pytest.raises(ValueError, match="context_id is required"):
        await agent.process(Session(messages=[Message(role="user", content="hi")]))

    await agent.close()
