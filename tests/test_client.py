"""Tests for KaguraClient."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from kagura_memory import KaguraAuthError, KaguraClient, KaguraConnectionError, KaguraQuotaError

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
async def test_recall_with_search_mode():
    """recall() should pass search_mode when specified."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"results": []}
        await client.recall(context_id="ctx", query="test", search_mode="keyword")
        args = mock.call_args[0][1]
        assert args["search_mode"] == "keyword"

    await client.close()


@pytest.mark.asyncio
async def test_recall_search_mode_not_sent_when_none():
    """recall() should not send search_mode when None."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"results": []}
        await client.recall(context_id="ctx", query="test")
        args = mock.call_args[0][1]
        assert "search_mode" not in args

    await client.close()


@pytest.mark.asyncio
async def test_recall_search_mode_invalid():
    """recall() should raise ValueError for invalid search_mode."""
    client = _make_initialized_client()

    with pytest.raises(ValueError, match="Invalid search_mode"):
        await client.recall(context_id="ctx", query="test", search_mode="invalid")

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


# ============================================================================
# Context management
# ============================================================================


@pytest.mark.asyncio
async def test_create_context():
    """create_context() should call tool with correct arguments."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.side_effect = [
            {"status": "success", "contexts": [], "count": 0, "limit": 20, "can_create": True},
            {"id": "uuid-1", "name": "new-ctx"},
        ]
        result = await client.create_context(
            name="new-ctx",
            summary="A test context",
            is_private=False,
        )
        assert result["name"] == "new-ctx"
        tool_name = mock.call_args[0][0]
        args = mock.call_args[0][1]
        assert tool_name == "create_context"
        assert args["name"] == "new-ctx"
        assert args["summary"] == "A test context"
        assert args["is_private"] is False

    await client.close()


@pytest.mark.asyncio
async def test_create_context_with_resource_id():
    """create_context() should pass resource_id when provided."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.side_effect = [
            {"status": "success", "contexts": [], "count": 0, "limit": 20, "can_create": True},
            {"id": "uuid-1", "name": "res-ctx"},
        ]
        await client.create_context(name="res-ctx", resource_id="my-resource")
        # Second call is create_context tool
        args = mock.call_args_list[1][0][1]
        assert args["resource_id"] == "my-resource"

    await client.close()


@pytest.mark.asyncio
async def test_create_context_with_embedding_model():
    """create_context() should pass embedding_model when provided."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.side_effect = [
            {"status": "success", "contexts": [], "count": 0, "limit": 20, "can_create": True},
            {"id": "uuid-1", "name": "emb-ctx"},
        ]
        await client.create_context(name="emb-ctx", embedding_model="qwen3-embedding:8b")
        args = mock.call_args_list[1][0][1]
        assert args["embedding_model"] == "qwen3-embedding:8b"

    await client.close()


@pytest.mark.asyncio
async def test_create_context_quota_exceeded():
    """create_context() should raise KaguraQuotaError when limit reached."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {
            "status": "success",
            "contexts": [],
            "count": 20,
            "limit": 20,
            "can_create": False,
        }

        with pytest.raises(KaguraQuotaError, match="Context limit reached"):
            await client.create_context(name="over-limit")

    await client.close()


@pytest.mark.asyncio
async def test_update_context_with_resource_id_and_is_public():
    """update_context() should pass resource_id and is_public."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"id": "uuid-1", "status": "success"}
        await client.update_context(context_id="uuid-1", resource_id="res-id", is_public=True)
        args = mock.call_args[0][1]
        assert args["resource_id"] == "res-id"
        assert args["is_public"] is True

    await client.close()


@pytest.mark.asyncio
async def test_create_context_minimal():
    """create_context() with only name should not send optional fields."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.side_effect = [
            {"status": "success", "contexts": [], "count": 0, "limit": 20, "can_create": True},
            {"id": "uuid-1", "name": "minimal"},
        ]
        await client.create_context(name="minimal")
        # Second call is create_context tool
        args = mock.call_args_list[1][0][1]
        assert args["name"] == "minimal"
        assert args["is_private"] is True
        assert "summary" not in args
        assert "display_name" not in args

    await client.close()


@pytest.mark.asyncio
async def test_delete_context():
    """delete_context() should call tool with context_id."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {
            "status": "success",
            "message": "Context 'test' has been soft-deleted.",
            "context_id": "ctx-123",
            "context_name": "test",
        }
        result = await client.delete_context(context_id="ctx-123")
        mock.assert_called_once_with("delete_context", {"context_id": "ctx-123"})
        assert result["status"] == "success"

    await client.close()


@pytest.mark.asyncio
async def test_update_context():
    """update_context() should call tool with correct arguments."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"id": "uuid-1", "summary": "updated"}
        result = await client.update_context(
            context_id="uuid-1",
            summary="updated",
            usage_guide="Store code only",
        )
        assert result["summary"] == "updated"
        tool_name = mock.call_args[0][0]
        args = mock.call_args[0][1]
        assert tool_name == "update_context"
        assert args["context_id"] == "uuid-1"
        assert args["summary"] == "updated"
        assert args["usage_guide"] == "Store code only"
        assert "display_name" not in args

    await client.close()


# ============================================================================
# Search config
# ============================================================================


@pytest.mark.asyncio
async def test_update_search_config():
    """update_search_config() should call tool with correct arguments."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"status": "success"}
        await client.update_search_config(
            context_id="uuid-1",
            semantic_weight=0.5,
            bm25_weight=0.5,
            fetch_factor=5,
        )
        tool_name = mock.call_args[0][0]
        args = mock.call_args[0][1]
        assert tool_name == "update_search_config"
        assert args["context_id"] == "uuid-1"
        assert args["semantic_weight"] == 0.5
        assert args["bm25_weight"] == 0.5
        assert "use_rerank" not in args
        assert "reranker_provider" not in args

    await client.close()


@pytest.mark.asyncio
async def test_update_search_config_with_rerank():
    """update_search_config() should pass rerank params."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"status": "success"}
        await client.update_search_config(
            context_id="uuid-1",
            use_rerank=True,
            reranker_provider="voyage",
            reranker_model="rerank-2",
        )
        args = mock.call_args[0][1]
        assert args["use_rerank"] is True
        assert args["reranker_provider"] == "voyage"
        assert args["reranker_model"] == "rerank-2"

    await client.close()


@pytest.mark.asyncio
async def test_update_search_config_minimal():
    """update_search_config() with only context_id should work."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"status": "success"}
        await client.update_search_config(context_id="uuid-1")
        args = mock.call_args[0][1]
        assert args == {"context_id": "uuid-1"}

    await client.close()


# ============================================================================
# list_embedding_models tests
# ============================================================================


@pytest.mark.asyncio
async def test_base_url_from_mcp_url():
    """_base_url should strip /mcp suffix."""
    async with KaguraClient(api_key="test", mcp_url="https://memory.kagura-ai.com/mcp") as client:
        assert client._base_url == "https://memory.kagura-ai.com"


@pytest.mark.asyncio
async def test_base_url_from_mcp_url_with_workspace():
    """_base_url should strip /mcp/w/{workspace_id}."""
    async with KaguraClient(
        api_key="test", mcp_url="https://memory.kagura-ai.com/mcp/w/ws-1"
    ) as client:
        assert client._base_url == "https://memory.kagura-ai.com"


@pytest.mark.asyncio
async def test_list_embedding_models():
    """list_embedding_models() should return parsed EmbeddingModelsResponse."""
    client = _make_initialized_client()

    response_data = {
        "models": [
            {
                "name": "text-embedding-3-small",
                "dimensions": 512,
                "provider": "openai",
                "available": True,
            },
            {
                "name": "qwen3-embedding:8b",
                "dimensions": 4096,
                "provider": "ollama",
                "available": False,
            },
        ],
        "default_model": "text-embedding-3-small",
    }

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = response_data
    mock_response.raise_for_status = MagicMock()

    with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        result = await client.list_embedding_models()

        assert result.default_model == "text-embedding-3-small"
        assert len(result.models) == 2
        assert result.models[0].name == "text-embedding-3-small"
        assert result.models[0].dimensions == 512
        assert result.models[0].provider == "openai"
        assert result.models[0].available is True
        assert result.models[1].available is False

        mock_get.assert_called_once_with("https://test.com/api/v1/system/embedding/models")

    await client.close()


@pytest.mark.asyncio
async def test_list_embedding_models_auth_error():
    """list_embedding_models() should raise KaguraAuthError on 401."""
    client = _make_initialized_client()

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 401

    with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "401", request=MagicMock(), response=mock_response
        )
        with pytest.raises(KaguraAuthError):
            await client.list_embedding_models()

    await client.close()


@pytest.mark.asyncio
async def test_list_embedding_models_http_error():
    """list_embedding_models() should raise KaguraConnectionError on non-401 HTTP error."""
    client = _make_initialized_client()

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 500

    with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=mock_response
        )
        with pytest.raises(KaguraConnectionError, match="HTTP 500"):
            await client.list_embedding_models()

    await client.close()


@pytest.mark.asyncio
async def test_list_embedding_models_invalid_response():
    """list_embedding_models() should raise KaguraConnectionError on invalid JSON schema."""
    client = _make_initialized_client()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"unexpected": "schema"}
    mock_response.raise_for_status = MagicMock()

    with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        with pytest.raises(KaguraConnectionError, match="Invalid response format"):
            await client.list_embedding_models()

    await client.close()


@pytest.mark.asyncio
async def test_list_embedding_models_connection_error():
    """list_embedding_models() should raise KaguraConnectionError on network failure."""
    client = _make_initialized_client()

    with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = httpx.ConnectError("Connection refused")
        with pytest.raises(KaguraConnectionError, match="Connection failed"):
            await client.list_embedding_models()

    await client.close()


@pytest.mark.asyncio
async def test_context_manager():
    """async with should return client and close on exit."""
    async with KaguraClient(api_key="test", mcp_url="https://test.com/mcp") as client:
        assert isinstance(client, KaguraClient)
