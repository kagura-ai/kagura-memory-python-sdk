"""Tests for KaguraClient."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from kagura_memory import KaguraAuthError, KaguraClient, KaguraConnectionError

# ============================================================================
# HTTPS enforcement (C-3)
# ============================================================================


def test_rejects_http_url():
    """HTTP URLs (non-localhost) should raise ValueError."""
    with pytest.raises(ValueError, match="must use HTTPS"):
        KaguraClient(api_key="test", mcp_url="http://evil.com/mcp")


def test_allows_https_url():
    """HTTPS URLs should be accepted."""
    client = KaguraClient(api_key="test", mcp_url="https://memory.kagura-ai.com/mcp")
    assert client.mcp_url == "https://memory.kagura-ai.com/mcp"


def test_allows_localhost_http():
    """HTTP localhost should be allowed for development."""
    client = KaguraClient(api_key="test", mcp_url="http://localhost:8080/mcp")
    assert client.mcp_url == "http://localhost:8080/mcp"


def test_allows_127_http():
    """HTTP 127.0.0.1 should be allowed for development."""
    client = KaguraClient(api_key="test", mcp_url="http://127.0.0.1:8080/mcp")
    assert client.mcp_url == "http://127.0.0.1:8080/mcp"


# ============================================================================
# API key not stored (C-1)
# ============================================================================


def test_api_key_not_on_instance():
    """API key should not be accessible as instance attribute."""
    client = KaguraClient(api_key="secret-key", mcp_url="https://test.com/mcp")
    assert not hasattr(client, "api_key")


# ============================================================================
# Auth error handling
# ============================================================================


@pytest.mark.asyncio
async def test_auth_error_on_401_initialize():
    """401 during session init should raise KaguraAuthError."""
    client = KaguraClient(api_key="bad-key", mcp_url="https://test.com/mcp")

    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "401", request=MagicMock(), response=mock_response
    )

    with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        with pytest.raises(KaguraAuthError, match="Authentication failed"):
            await client._initialize_session()

    await client.close()


@pytest.mark.asyncio
async def test_connection_error_on_network_failure():
    """Network failure should raise KaguraConnectionError."""
    client = KaguraClient(api_key="test", mcp_url="https://test.com/mcp")

    with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = httpx.ConnectError("Connection refused")

        with pytest.raises(KaguraConnectionError, match="Connection failed"):
            await client._initialize_session()

    await client.close()


@pytest.mark.asyncio
async def test_connection_error_on_missing_session_id():
    """Missing mcp-session-id header should raise KaguraConnectionError."""
    client = KaguraClient(api_key="test", mcp_url="https://test.com/mcp")

    mock_response = MagicMock()
    mock_response.json.return_value = {"jsonrpc": "2.0", "id": 1, "result": {}}
    mock_response.raise_for_status = MagicMock()
    mock_response.headers = {}  # No mcp-session-id

    with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        with pytest.raises(KaguraConnectionError, match="No session ID"):
            await client._initialize_session()

    await client.close()


# ============================================================================
# Tool definitions (existing tests)
# ============================================================================


@pytest.mark.asyncio
async def test_get_tool_definitions_success():
    """Test successful tool definitions retrieval."""
    client = KaguraClient(api_key="test-key", mcp_url="https://test.com/mcp")

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "tools": [
                {"name": "remember", "description": "Store information", "inputSchema": {}},
                {"name": "recall", "description": "Search memories", "inputSchema": {}},
            ]
        },
    }
    mock_response.raise_for_status = MagicMock()
    mock_response.headers = {"mcp-session-id": "test-session"}

    with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        tools = await client.get_tool_definitions()

        assert len(tools) == 2
        assert tools[0]["name"] == "remember"
        assert tools[1]["name"] == "recall"
        assert mock_post.call_count == 2  # initialize + tools/list

    await client.close()


@pytest.mark.asyncio
async def test_get_tool_definitions_empty():
    """Test tool definitions retrieval with empty response."""
    client = KaguraClient(api_key="test-key", mcp_url="https://test.com/mcp")

    mock_response = MagicMock()
    mock_response.json.return_value = {"jsonrpc": "2.0", "id": 1, "result": {}}
    mock_response.raise_for_status = MagicMock()
    mock_response.headers = {"mcp-session-id": "test-session"}

    with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        tools = await client.get_tool_definitions()
        assert tools == []

    await client.close()


# ============================================================================
# Request ID concurrency (I-2)
# ============================================================================


def test_request_id_increments():
    """Request IDs should be unique and incrementing."""
    client = KaguraClient(api_key="test", mcp_url="https://test.com/mcp")
    ids = [client._next_request_id() for _ in range(100)]
    assert ids == list(range(1, 101))
    assert len(set(ids)) == 100  # All unique


# ============================================================================
# Tool method tests
# ============================================================================


def _make_initialized_client():
    """Create a client with session already initialized."""
    client = KaguraClient(api_key="test", mcp_url="https://test.com/mcp")
    client._session_id = "pre-set-session"
    return client


@pytest.mark.asyncio
async def test_remember_with_tags():
    """remember() should include tags in arguments."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"memory_id": "abc"}
        await client.remember(
            context_id="ctx", summary="s", content="c", tags=["python", "fastapi"]
        )
        args = mock.call_args[0][1]
        assert args["tags"] == ["python", "fastapi"]

    await client.close()


@pytest.mark.asyncio
async def test_recall_with_rerank():
    """recall() should pass use_rerank when True."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"results": []}
        await client.recall(context_id="ctx", query="test", use_rerank=True)
        args = mock.call_args[0][1]
        assert args["use_rerank"] is True

    await client.close()


@pytest.mark.asyncio
async def test_recall_with_filters():
    """recall() should pass filters dict."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"results": []}
        await client.recall(context_id="ctx", query="test", filters={"type": "code"})
        args = mock.call_args[0][1]
        assert args["filters"] == {"type": "code"}

    await client.close()


@pytest.mark.asyncio
async def test_forget_by_memory_id():
    """forget() with memory_id should pass it in arguments."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"deleted": 1}
        await client.forget(context_id="ctx", memory_id="uuid-123")
        args = mock.call_args[0][1]
        assert args["memory_id"] == "uuid-123"
        assert "query" not in args

    await client.close()


@pytest.mark.asyncio
async def test_forget_by_query():
    """forget() with query should pass query and k in arguments."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"deleted": 3}
        await client.forget(context_id="ctx", query="old data", k=5)
        args = mock.call_args[0][1]
        assert args["query"] == "old data"
        assert args["k"] == 5

    await client.close()


@pytest.mark.asyncio
async def test_explore_calls_tool():
    """explore() should assemble correct arguments."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"memories": []}
        await client.explore(context_id="ctx", memory_id="seed", depth=3, min_weight=0.1)
        args = mock.call_args[0][1]
        assert args["memory_id"] == "seed"
        assert args["depth"] == 3
        assert args["min_weight"] == 0.1

    await client.close()


@pytest.mark.asyncio
async def test_reference_calls_tool():
    """reference() should pass context_id and memory_id."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"summary": "test"}
        await client.reference(context_id="ctx", memory_id="mem-1")
        args = mock.call_args[0][1]
        assert args["context_id"] == "ctx"
        assert args["memory_id"] == "mem-1"

    await client.close()


@pytest.mark.asyncio
async def test_call_tool_invalid_json():
    """_call_tool should raise KaguraConnectionError on invalid JSON response."""
    client = _make_initialized_client()

    with patch.object(client, "_make_jsonrpc_request", new_callable=AsyncMock) as mock:
        mock.return_value = {"content": [{"type": "text", "text": "not json{"}]}

        with pytest.raises(KaguraConnectionError, match="Invalid response"):
            await client._call_tool("remember", {})

    await client.close()


@pytest.mark.asyncio
async def test_jsonrpc_mcp_error():
    """_make_jsonrpc_request should raise on MCP error in response."""
    client = _make_initialized_client()

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "jsonrpc": "2.0",
        "id": 1,
        "error": {"message": "Tool not found"},
    }
    mock_response.raise_for_status = MagicMock()
    mock_response.headers = {"mcp-session-id": "test-session"}

    with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        with pytest.raises(KaguraConnectionError, match="MCP error"):
            await client._make_jsonrpc_request("tools/call", {"name": "bad"})

    await client.close()


@pytest.mark.asyncio
async def test_session_already_initialized():
    """_initialize_session should skip if session already set."""
    client = _make_initialized_client()

    with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
        await client._initialize_session()
        mock_post.assert_not_called()

    await client.close()


@pytest.mark.asyncio
async def test_context_manager():
    """async with should return client and close on exit."""
    async with KaguraClient(api_key="test", mcp_url="https://test.com/mcp") as client:
        assert isinstance(client, KaguraClient)
