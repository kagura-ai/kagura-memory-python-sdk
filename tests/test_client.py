"""Tests for KaguraClient."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from kagura_memory import (
    MIN_SERVER_VERSION,
    ContextInfo,
    DuplicatesResponse,
    EmbeddingStatus,
    KaguraAuthError,
    KaguraClient,
    KaguraConnectionError,
    KaguraError,
    KaguraNotFoundError,
    KaguraQuotaError,
    MemoryListResponse,
    MemoryStatsResponse,
    RollbackResult,
    ServerInfo,
    SleepAction,
    SleepReport,
    SleepReportDetail,
    UsageInfo,
)
from tests.conftest import sleep_report_summary_dict

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


@pytest.mark.asyncio
async def test_initialize_session_non_401_http_error_surfaces_class_name():
    """5xx HTTPStatusError → wrapped via _exc_message so empty str(e) still renders (#130)."""
    client = KaguraClient(api_key="test", mcp_url="https://test.com/mcp")

    mock_response = MagicMock()
    mock_response.status_code = 503
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "", request=MagicMock(), response=mock_response
    )

    with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        with pytest.raises(KaguraConnectionError, match=r"HTTP 503:") as exc_info:
            await client._initialize_session()

    # The wrapper must not strand the prefix when str(e) is empty.
    msg = str(exc_info.value)
    assert msg != "HTTP 503: "
    assert msg.endswith("HTTPStatusError") or "HTTP 503: " in msg and len(msg.split(": ", 1)[1]) > 0

    await client.close()


@pytest.mark.asyncio
async def test_make_jsonrpc_request_connection_error_surfaces():
    """httpx.RequestError on JSON-RPC post → KaguraConnectionError via _exc_message (#130)."""
    client = KaguraClient(api_key="test", mcp_url="https://test.com/mcp")
    client._session_id = "stub-session"  # bypass initialize

    with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = httpx.ConnectError("network down")

        with pytest.raises(KaguraConnectionError, match="Connection failed: network down"):
            await client._make_jsonrpc_request("tools/list", {})

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
async def test_remember_with_source_uri():
    """remember() should include source_uri when provided."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"memory_id": "abc"}
        await client.remember(
            context_id="ctx", summary="s", content="c", source_uri="file:///foo.txt"
        )
        args = mock.call_args[0][1]
        assert args["source_uri"] == "file:///foo.txt"

    await client.close()


@pytest.mark.asyncio
async def test_remember_pass_through_keys_absent_when_none():
    """remember() should not send optional keys when None."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"memory_id": "abc"}
        await client.remember(context_id="ctx", summary="s", content="c")
        args = mock.call_args[0][1]
        assert "source_uri" not in args
        assert "source_type" not in args
        assert "context_summary" not in args
        assert "details" not in args
        assert "context" not in args
        assert "linked_memory_ids" not in args
        assert "linked_source_uris" not in args

    await client.close()


@pytest.mark.asyncio
async def test_remember_with_details():
    """remember() should pass details JSON dict through to MCP arguments."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"memory_id": "abc"}
        await client.remember(
            context_id="ctx",
            summary="s",
            content="c",
            details={
                "code_location": "src/auth.py:142",
                "related_issue": "#123",
                "tags_seen": ["oauth", "jwt"],
            },
        )
        args = mock.call_args[0][1]
        assert args["details"] == {
            "code_location": "src/auth.py:142",
            "related_issue": "#123",
            "tags_seen": ["oauth", "jwt"],
        }

    await client.close()


@pytest.mark.asyncio
async def test_remember_with_context_summary():
    """remember() should pass context_summary when provided."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"memory_id": "abc"}
        await client.remember(
            context_id="ctx",
            summary="s",
            content="c",
            context_summary="Why this memory matters and how to use it.",
        )
        args = mock.call_args[0][1]
        assert args["context_summary"] == "Why this memory matters and how to use it."

    await client.close()


@pytest.mark.asyncio
async def test_remember_with_source_type():
    """remember() should pass source_type alongside source_uri."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"memory_id": "abc"}
        await client.remember(
            context_id="ctx",
            summary="s",
            content="c",
            source_uri="https://example.com/doc.html",
            source_type="url",
        )
        args = mock.call_args[0][1]
        assert args["source_type"] == "url"
        assert args["source_uri"] == "https://example.com/doc.html"

    await client.close()


@pytest.mark.asyncio
async def test_remember_with_context_dict():
    """remember() should pass context dict (free-form provenance metadata)."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"memory_id": "abc"}
        await client.remember(
            context_id="ctx",
            summary="s",
            content="c",
            context={"issue": 80, "branch": "feat/example", "session_id": "s-1"},
        )
        args = mock.call_args[0][1]
        assert args["context"] == {
            "issue": 80,
            "branch": "feat/example",
            "session_id": "s-1",
        }

    await client.close()


@pytest.mark.asyncio
async def test_remember_combined_pass_through_payload():
    """remember() with all the new pass-through kwargs together."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"memory_id": "section-uuid"}
        await client.remember(
            context_id="ctx",
            summary="Doc — chapter 0: Intro",
            content="Chapter text...",
            type="document_section",
            importance=0.5,
            tags=["pdf", "user-recipe"],
            source_uri="https://example.com/doc.pdf",
            source_type="url",
            context_summary="Chapter 0 of doc.pdf",
            details={
                "parent_id": "overview-uuid",
                "role": "section",
                "section_index": 0,
            },
            linked_memory_ids=["overview-uuid"],
        )
        args = mock.call_args[0][1]
        assert args["type"] == "document_section"
        assert args["details"]["role"] == "section"
        assert args["details"]["parent_id"] == "overview-uuid"
        assert args["linked_memory_ids"] == ["overview-uuid"]
        assert args["source_type"] == "url"
        assert args["context_summary"] == "Chapter 0 of doc.pdf"

    await client.close()


@pytest.mark.asyncio
async def test_remember_with_linked_memory_ids():
    """remember() should include linked_memory_ids when provided."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"memory_id": "abc"}
        await client.remember(
            context_id="ctx",
            summary="s",
            content="c",
            linked_memory_ids=["mem-1", "mem-2"],
        )
        args = mock.call_args[0][1]
        assert args["linked_memory_ids"] == ["mem-1", "mem-2"]

    await client.close()


@pytest.mark.asyncio
async def test_remember_with_linked_source_uris():
    """remember() should include linked_source_uris when provided."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"memory_id": "abc"}
        await client.remember(
            context_id="ctx",
            summary="s",
            content="c",
            linked_source_uris=["vault://x", "vault://y"],
        )
        args = mock.call_args[0][1]
        assert args["linked_source_uris"] == ["vault://x", "vault://y"]

    await client.close()


@pytest.mark.asyncio
async def test_remember_with_empty_linked_memory_ids():
    """remember() should send linked_memory_ids even when empty list (not None)."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"memory_id": "abc"}
        await client.remember(context_id="ctx", summary="s", content="c", linked_memory_ids=[])
        args = mock.call_args[0][1]
        assert args["linked_memory_ids"] == []

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
async def test_recall_with_include_explore_hints():
    """recall() should send include_explore_hints when True."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"results": []}
        await client.recall(context_id="ctx", query="test", include_explore_hints=True)
        args = mock.call_args[0][1]
        assert args["include_explore_hints"] is True

    await client.close()


@pytest.mark.asyncio
async def test_recall_include_explore_hints_not_sent_by_default():
    """recall() should not send include_explore_hints when False (the default)."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"results": []}
        await client.recall(context_id="ctx", query="test")
        args = mock.call_args[0][1]
        assert "include_explore_hints" not in args

    await client.close()


@pytest.mark.asyncio
async def test_recall_search_mode_invalid():
    """recall() should raise ValueError for invalid search_mode."""
    client = _make_initialized_client()

    with pytest.raises(ValueError, match="Invalid search_mode"):
        await client.recall(context_id="ctx", query="test", search_mode="invalid")

    await client.close()


@pytest.mark.asyncio
async def test_recall_with_tags_match_filter():
    """recall() should pass tags_match in filters."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"results": []}
        await client.recall(
            context_id="ctx",
            query="budget",
            filters={"tags": ["予算", "2026"], "tags_match": "all"},
        )
        args = mock.call_args[0][1]
        assert args["filters"]["tags_match"] == "all"
        assert args["filters"]["tags"] == ["予算", "2026"]

    await client.close()


@pytest.mark.asyncio
async def test_recall_with_date_filters():
    """recall() should pass date range filters."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"results": []}
        await client.recall(
            context_id="ctx",
            query="recent",
            filters={
                "created_after": "2026-03-01T00:00:00Z",
                "created_before": "2026-03-31T23:59:59Z",
            },
        )
        args = mock.call_args[0][1]
        assert args["filters"]["created_after"] == "2026-03-01T00:00:00Z"
        assert args["filters"]["created_before"] == "2026-03-31T23:59:59Z"

    await client.close()


@pytest.mark.asyncio
async def test_recall_with_context_ids():
    """recall() with context_ids should send context_ids, not context_id."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"results": []}
        await client.recall(
            query="auth",
            context_ids=["ctx-1", "ctx-2"],
        )
        args = mock.call_args[0][1]
        assert args["context_ids"] == ["ctx-1", "ctx-2"]
        assert "context_id" not in args

    await client.close()


@pytest.mark.asyncio
async def test_recall_context_ids_validation():
    """recall() should reject context_ids with fewer than 2 or more than 20 IDs."""
    client = _make_initialized_client()

    with pytest.raises(ValueError, match="2–20 IDs"):
        await client.recall(query="test", context_ids=["only-one"])

    with pytest.raises(ValueError, match="2–20 IDs"):
        await client.recall(query="test", context_ids=[f"ctx-{i}" for i in range(21)])

    await client.close()


@pytest.mark.asyncio
async def test_recall_requires_context_id_or_context_ids():
    """recall() should raise ValueError when neither context_id nor context_ids."""
    client = _make_initialized_client()

    with pytest.raises(ValueError, match="Either context_id or context_ids"):
        await client.recall(query="test")

    await client.close()


@pytest.mark.asyncio
async def test_recall_both_context_id_and_context_ids():
    """recall() with both context_id and context_ids should use context_ids."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"results": []}
        await client.recall(
            context_id="single",
            query="test",
            context_ids=["ctx-1", "ctx-2"],
        )
        args = mock.call_args[0][1]
        assert args["context_ids"] == ["ctx-1", "ctx-2"]
        assert "context_id" not in args

    await client.close()


@pytest.mark.asyncio
async def test_recall_empty_query():
    """recall() should reject empty query string."""
    client = _make_initialized_client()

    with pytest.raises(ValueError, match="query must be a non-empty string"):
        await client.recall(context_id="ctx", query="")

    await client.close()


@pytest.mark.asyncio
async def test_recall_whitespace_only_query():
    """recall() should reject whitespace-only query string."""
    client = _make_initialized_client()

    with pytest.raises(ValueError, match="query must be a non-empty string"):
        await client.recall(context_id="ctx", query="   ")

    await client.close()


@pytest.mark.asyncio
async def test_recall_without_context_ids_sends_context_id():
    """recall() without context_ids should send context_id as before."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"results": []}
        await client.recall(context_id="my-ctx", query="test")
        args = mock.call_args[0][1]
        assert args["context_id"] == "my-ctx"
        assert "context_ids" not in args

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
async def test_setup_resource_basic():
    """Optional fields must be omitted when None so the server applies its own defaults."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {
            "context_id": "ctx-uuid",
            "context_name": "products",
            "resource_id": "products",
            "token": "kagura_resource_xyz",
            "token_id": 1,
        }
        result = await client.setup_resource(resource_id="products")
        tool_name = mock.call_args[0][0]
        args = mock.call_args[0][1]
        assert tool_name == "setup_resource"
        assert args["resource_id"] == "products"
        assert args["quota_events_per_hour"] == 1000
        assert "name" not in args
        assert "summary" not in args
        assert "description" not in args
        assert result["token"] == "kagura_resource_xyz"

    await client.close()


@pytest.mark.asyncio
async def test_setup_resource_with_all_args():
    """setup_resource() should forward all optional args when provided."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {}
        await client.setup_resource(
            resource_id="products",
            name="Product Catalog",
            summary="All product data",
            description="Catalog ingestion token",
            quota_events_per_hour=5000,
        )
        args = mock.call_args[0][1]
        assert args["resource_id"] == "products"
        assert args["name"] == "Product Catalog"
        assert args["summary"] == "All product data"
        assert args["description"] == "Catalog ingestion token"
        assert args["quota_events_per_hour"] == 5000

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
async def test_update_memory_by_id():
    """update_memory() with memory_id should pass correct arguments."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"status": "success", "memory_id": "mem-1"}
        result = await client.update_memory(
            context_id="ctx",
            memory_id="mem-1",
            summary="updated",
            importance=0.9,
            tags=["new-tag"],
            context_summary="why this matters",
        )
        args = mock.call_args[0][1]
        assert args["context_id"] == "ctx"
        assert args["memory_id"] == "mem-1"
        assert args["summary"] == "updated"
        assert args["importance"] == 0.9
        assert args["tags"] == ["new-tag"]
        assert args["context_summary"] == "why this matters"
        assert "external_id" not in args
        assert result["status"] == "success"

    await client.close()


@pytest.mark.asyncio
async def test_update_memory_upsert():
    """update_memory() with external_id should pass correct arguments."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"status": "success", "memory_id": "mem-new"}
        await client.update_memory(
            context_id="ctx",
            external_id="ext-key",
            summary="upserted",
            content="content",
            type="note",
        )
        args = mock.call_args[0][1]
        assert args["external_id"] == "ext-key"
        assert args["summary"] == "upserted"
        assert "memory_id" not in args

    await client.close()


@pytest.mark.asyncio
async def test_update_memory_requires_one_id():
    """update_memory() should reject when neither or both IDs provided."""
    client = _make_initialized_client()

    with pytest.raises(ValueError, match="exactly one"):
        await client.update_memory(context_id="ctx")

    with pytest.raises(ValueError, match="exactly one"):
        await client.update_memory(context_id="ctx", memory_id="m1", external_id="e1")

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


@pytest.mark.asyncio
async def test_update_context_is_locked():
    """update_context() should pass is_locked when specified."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"id": "uuid-1", "is_locked": True}
        await client.update_context(context_id="uuid-1", is_locked=True)
        args = mock.call_args[0][1]
        assert args["is_locked"] is True

        await client.update_context(context_id="uuid-1", is_locked=False)
        args = mock.call_args[0][1]
        assert args["is_locked"] is False

    await client.close()


# ============================================================================
# Context merge
# ============================================================================


@pytest.mark.asyncio
async def test_merge_contexts():
    """merge_contexts() should call tool with source and target IDs."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"merged": 42, "source_id": "src", "target_id": "tgt"}
        result = await client.merge_contexts(source_id="src", target_id="tgt")
        args = mock.call_args[0][1]
        assert args["source_id"] == "src"
        assert args["target_id"] == "tgt"
        assert "delete_source" not in args
        assert result["merged"] == 42

    await client.close()


@pytest.mark.asyncio
async def test_merge_contexts_with_delete_source():
    """merge_contexts() should pass delete_source when True."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"merged": 10}
        await client.merge_contexts(source_id="src", target_id="tgt", delete_source=True)
        args = mock.call_args[0][1]
        assert args["delete_source"] is True

    await client.close()


@pytest.mark.asyncio
async def test_merge_contexts_delete_source_not_sent_when_false():
    """merge_contexts() should not send delete_source when False."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"merged": 10}
        await client.merge_contexts(source_id="src", target_id="tgt", delete_source=False)
        args = mock.call_args[0][1]
        assert "delete_source" not in args

    await client.close()


@pytest.mark.asyncio
async def test_merge_contexts_same_ids():
    """merge_contexts() should reject same source and target IDs."""
    client = _make_initialized_client()

    with pytest.raises(ValueError, match="source_id and target_id must be different"):
        await client.merge_contexts(source_id="same", target_id="same")

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

        mock_get.assert_called_once_with(
            "https://test.com/api/v1/system/embedding/models", params=None
        )

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


# ============================================================================
# get_usage (MCP tool)
# ============================================================================


@pytest.mark.asyncio
async def test_get_usage():
    """get_usage() should return UsageInfo model."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {
            "status": "success",
            "plan": "pro",
            "memories": {"used": 100, "limit": 11000, "percentage": 0.9},
            "contexts": {"used": 3, "limit": 20},
            "members": {"used": 1, "limit": 10},
            "mcp_calls_per_day": {"limit": 50000},
        }
        result = await client.get_usage()
        assert isinstance(result, UsageInfo)
        assert result.plan == "pro"
        assert result.memories.used == 100
        assert result.memories.limit == 11000
        assert result.contexts.used == 3
        assert result.mcp_calls_per_day.limit == 50000
        mock.assert_called_once_with("get_usage", {})

    await client.close()


# ============================================================================
# get_context_info (MCP tool)
# ============================================================================


@pytest.mark.asyncio
async def test_get_context_info():
    """get_context_info() should return ContextInfo with search_config."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {
            "status": "success",
            "context": {
                "id": "uuid-1",
                "name": "test-ctx",
                "display_name": "Test Context",
                "summary": "A test context",
                "is_private": True,
                "is_locked": False,
                "embedding_model": "qwen3-embedding:8b",
                "embedding_dimensions": 4096,
                "search_config": {
                    "semantic_weight": 0.6,
                    "bm25_weight": 0.4,
                    "fetch_factor": 3,
                    "use_rerank": False,
                    "reranker_provider": "voyage",
                    "reranker_model": "rerank-2-lite",
                },
            },
            "workspace": {"id": "ws-1", "name": "My Workspace"},
            "stats": {
                "total_memories": 50,
                "working_memories": 5,
                "persistent_memories": 45,
            },
            "instructions": "Quick reference guide...",
        }
        result = await client.get_context_info(context_id="uuid-1")
        assert isinstance(result, ContextInfo)
        assert result.context.name == "test-ctx"
        assert result.context.search_config is not None
        assert result.context.search_config.semantic_weight == 0.6
        assert result.context.search_config.reranker_provider == "voyage"
        assert result.stats is not None
        assert result.stats.total_memories == 50
        args = mock.call_args[0][1]
        assert args["context_id"] == "uuid-1"
        assert args["include_details"] is True

    await client.close()


@pytest.mark.asyncio
async def test_get_context_info_without_details():
    """get_context_info() with include_details=False."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {
            "status": "success",
            "context": {"id": "uuid-1", "name": "test-ctx", "is_private": True},
        }
        result = await client.get_context_info(context_id="uuid-1", include_details=False)
        assert isinstance(result, ContextInfo)
        args = mock.call_args[0][1]
        assert args["include_details"] is False

    await client.close()


@pytest.mark.asyncio
async def test_get_context_info_cached_fetches_once():
    """_get_context_info_cached() should hit the MCP tool only once per context_id."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {
            "status": "success",
            "context": {"id": "uuid-1", "name": "test-ctx", "is_private": True},
            "instructions": "Steer toward billing terminology.",
        }
        first = await client._get_context_info_cached("uuid-1")
        second = await client._get_context_info_cached("uuid-1")

    assert first is second  # same cached object
    assert first is not None
    assert first.instructions == "Steer toward billing terminology."
    assert mock.call_count == 1  # second call served from cache

    await client.close()


@pytest.mark.asyncio
async def test_get_context_info_cached_degrades_on_failure():
    """A fetch failure caches and returns None (best-effort), never re-fetching."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.side_effect = KaguraError("context unreachable")
        first = await client._get_context_info_cached("uuid-missing")
        second = await client._get_context_info_cached("uuid-missing")

    assert first is None
    assert second is None
    assert mock.call_count == 1  # failure is cached; no retry storm

    await client.close()


@pytest.mark.asyncio
async def test_get_context_info_cached_degrades_on_malformed_payload():
    """A malformed server payload (pydantic ValidationError) degrades to None."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        # Missing the required `context` field → ContextInfo.model_validate raises
        # a pydantic ValidationError, which is NOT a KaguraError.
        mock.return_value = {"status": "success"}
        result = await client._get_context_info_cached("uuid-1")

    assert result is None  # did not crash; degraded to None

    await client.close()


# ============================================================================
# get_embedding_status (REST)
# ============================================================================


@pytest.mark.asyncio
async def test_get_embedding_status():
    """get_embedding_status() should return EmbeddingStatus model."""
    client = _make_initialized_client()

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "total": 100,
        "by_status": {"success": 98, "failed": 2},
        "failed_memories": [
            {
                "id": "mem-1",
                "summary": "broken",
                "embedding_error": "model unavailable",
                "created_at": "2026-04-01T00:00:00Z",
                "updated_at": None,
            }
        ],
    }
    mock_response.raise_for_status = MagicMock()

    with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        result = await client.get_embedding_status()
        assert isinstance(result, EmbeddingStatus)
        assert result.total == 100
        assert result.by_status["failed"] == 2
        assert len(result.failed_memories) == 1
        assert result.failed_memories[0].embedding_error == "model unavailable"
        mock_get.assert_called_once_with(
            "https://test.com/api/v1/workspace/embedding-status", params=None
        )

    await client.close()


# ============================================================================
# get_memory_stats (REST)
# ============================================================================


@pytest.mark.asyncio
async def test_get_memory_stats():
    """get_memory_stats() should return MemoryStatsResponse model."""
    client = _make_initialized_client()

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "memories": [
            {
                "id": "mem-1",
                "summary": "top memory",
                "type": "note",
                "importance": 0.9,
                "scope": "persistent",
                "use_count": 10,
                "access_count": 25,
                "last_used_at": "2026-04-03T00:00:00Z",
                "embedding_status": "success",
                "created_at": "2026-03-01T00:00:00Z",
            }
        ],
        "total": 1,
        "sort_by": "use_count",
        "sort_order": "desc",
    }
    mock_response.raise_for_status = MagicMock()

    with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        result = await client.get_memory_stats(context_id="ctx-1", sort_by="use_count", limit=10)
        assert isinstance(result, MemoryStatsResponse)
        assert result.total == 1
        assert result.memories[0].use_count == 10
        mock_get.assert_called_once_with(
            "https://test.com/api/v1/contexts/ctx-1/memory-stats",
            params={"sort_by": "use_count", "sort_order": "desc", "limit": 10, "offset": 0},
        )

    await client.close()


# ============================================================================
# find_duplicates (REST)
# ============================================================================


@pytest.mark.asyncio
async def test_find_duplicates():
    """find_duplicates() should return DuplicatesResponse model."""
    client = _make_initialized_client()

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "pairs": [
            {
                "memory_a": {
                    "id": "mem-1",
                    "summary": "foo",
                    "type": "note",
                    "created_at": "2026-03-01T00:00:00Z",
                },
                "memory_b": {
                    "id": "mem-2",
                    "summary": "foo bar",
                    "type": "note",
                    "created_at": "2026-03-02T00:00:00Z",
                },
                "similarity": 0.95,
            }
        ],
        "total_pairs": 1,
        "threshold": 0.90,
        "memories_scanned": 50,
    }
    mock_response.raise_for_status = MagicMock()

    with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        result = await client.find_duplicates(context_id="ctx-1", threshold=0.90, limit=25)
        assert isinstance(result, DuplicatesResponse)
        assert result.total_pairs == 1
        assert result.pairs[0].similarity == 0.95
        assert result.pairs[0].memory_a.id == "mem-1"
        mock_get.assert_called_once_with(
            "https://test.com/api/v1/contexts/ctx-1/duplicates",
            params={"threshold": 0.90, "limit": 25},
        )

    await client.close()


# ============================================================================
# list_memories (REST)
# ============================================================================


def _memory_list_response_mock():
    """A MagicMock httpx response shaped like GET /api/v1/memory/list."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "memories": [
            {
                "id": "mem-1",
                "summary": "first memory",
                "type": "note",
                "scope": "persistent",
                "importance": 0.7,
                "created_at": "2026-03-01T00:00:00Z",
                "updated_at": "2026-03-02T00:00:00Z",
            }
        ],
        "total": 1,
        "has_more": False,
    }
    mock_response.raise_for_status = MagicMock()
    return mock_response


@pytest.mark.asyncio
async def test_list_memories_with_filters():
    """list_memories() should return MemoryListResponse and forward filters."""
    client = _make_initialized_client()

    with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = _memory_list_response_mock()
        result = await client.list_memories(
            context_id="ctx-1", q="foo", scope="persistent", type="note", limit=10, offset=5
        )
        assert isinstance(result, MemoryListResponse)
        assert result.total == 1
        assert result.has_more is False
        assert result.memories[0].id == "mem-1"
        mock_get.assert_called_once_with(
            "https://test.com/api/v1/memory/list",
            params={
                "limit": 10,
                "offset": 5,
                "context_id": "ctx-1",
                "q": "foo",
                "scope": "persistent",
                "type": "note",
            },
        )

    await client.close()


@pytest.mark.asyncio
async def test_list_memories_defaults_omit_optional_params():
    """With no optional args, only limit/offset are sent."""
    client = _make_initialized_client()

    with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = _memory_list_response_mock()
        await client.list_memories()
        mock_get.assert_called_once_with(
            "https://test.com/api/v1/memory/list",
            params={"limit": 50, "offset": 0},
        )

    await client.close()


@pytest.mark.asyncio
async def test_list_memories_strips_q_and_omits_when_blank():
    """q is stripped; whitespace-only q is treated as None (omitted)."""
    client = _make_initialized_client()

    # Whitespace-only -> omitted entirely.
    with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = _memory_list_response_mock()
        await client.list_memories(q="   ")
        assert "q" not in mock_get.call_args.kwargs["params"]

    # Surrounding whitespace -> stripped to the trimmed value.
    with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = _memory_list_response_mock()
        await client.list_memories(q="  foo  ")
        assert mock_get.call_args.kwargs["params"]["q"] == "foo"

    await client.close()


@pytest.mark.asyncio
async def test_list_memories_window_params_forwarded():
    """list_memories() forwards trigger_from / trigger_until / order_by."""
    client = _make_initialized_client()

    with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = _memory_list_response_mock()
        await client.list_memories(
            context_id="ctx-1",
            trigger_from="2026-06-01T00:00:00",
            trigger_until="2026-07-01T00:00:00",
            order_by="trigger_from",
        )
        params = mock_get.call_args.kwargs["params"]
        assert params["trigger_from"] == "2026-06-01T00:00:00"
        assert params["trigger_until"] == "2026-07-01T00:00:00"
        assert params["order_by"] == "trigger_from"

    await client.close()


@pytest.mark.asyncio
async def test_list_memories_window_params_omitted_when_none():
    """list_memories() omits the window params when not provided."""
    client = _make_initialized_client()

    with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = _memory_list_response_mock()
        await client.list_memories(context_id="ctx-1")
        params = mock_get.call_args.kwargs["params"]
        assert "trigger_from" not in params
        assert "trigger_until" not in params
        assert "order_by" not in params

    await client.close()


# ============================================================================
# recall_upcoming (Time Memory, MCP)
# ============================================================================


@pytest.mark.asyncio
async def test_recall_upcoming_minimal():
    """recall_upcoming() with no bounds sends only context_id + the default k."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"results": []}
        await client.recall_upcoming(context_id="ctx")
        name, args = mock.call_args.args[0], mock.call_args.args[1]
        assert name == "recall_upcoming"
        assert args == {"context_id": "ctx", "k": 20}

    await client.close()


@pytest.mark.asyncio
async def test_recall_upcoming_maps_from_keyword_to_from_key():
    """recall_upcoming() maps the from_ param to the reserved-word "from" key."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"results": []}
        await client.recall_upcoming(
            context_id="ctx", from_="now", until="2026-07-01T00:00:00", k=5
        )
        args = mock.call_args.args[1]
        assert args["from"] == "now"
        assert "from_" not in args
        assert args["until"] == "2026-07-01T00:00:00"
        assert args["k"] == 5

    await client.close()


@pytest.mark.asyncio
async def test_recall_upcoming_omits_bounds_when_none():
    """recall_upcoming() omits from/until when not provided (k always sent)."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"results": []}
        await client.recall_upcoming(context_id="ctx")
        args = mock.call_args.args[1]
        assert "from" not in args
        assert "until" not in args
        assert args["k"] == 20

    await client.close()


# ============================================================================
# get_server_info / check_server_version
# ============================================================================


@pytest.mark.asyncio
async def test_get_server_info():
    """get_server_info() should return ServerInfo model."""
    client = _make_initialized_client()

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "name": "Kagura Memory Cloud",
        "version": "0.6.1",
        "description": "Remote MCP Server",
        "environment": "production",
        "features": {"neural_memory": True, "research_tools": True},
    }
    mock_response.raise_for_status = MagicMock()

    with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        result = await client.get_server_info()
        assert isinstance(result, ServerInfo)
        assert result.version == "0.6.1"
        assert result.features.neural_memory is True

    await client.close()


@pytest.mark.asyncio
async def test_check_server_version_ok(caplog):
    """check_server_version() should not warn when version meets minimum."""
    client = _make_initialized_client()

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "name": "Kagura Memory Cloud",
        "version": MIN_SERVER_VERSION,
        "features": {},
    }
    mock_response.raise_for_status = MagicMock()

    with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        import logging

        with caplog.at_level(logging.WARNING, logger="kagura_memory"):
            result = await client.check_server_version()
            assert result.version == MIN_SERVER_VERSION
            assert "below minimum" not in caplog.text

    await client.close()


@pytest.mark.asyncio
async def test_check_server_version_old(caplog):
    """check_server_version() should warn when server is too old."""
    client = _make_initialized_client()

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "name": "Kagura Memory Cloud",
        "version": "0.5.0",
        "features": {},
    }
    mock_response.raise_for_status = MagicMock()

    with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        import logging

        with caplog.at_level(logging.WARNING, logger="kagura_memory"):
            result = await client.check_server_version()
            assert result.version == "0.5.0"
            assert "is below" in caplog.text
            assert "tested minimum" in caplog.text

    await client.close()


@pytest.mark.asyncio
async def test_check_server_version_non_semver():
    """check_server_version() should not crash on non-semver version strings."""
    client = _make_initialized_client()

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "name": "Kagura Memory Cloud",
        "version": "0.6.1-rc1",
        "features": {},
    }
    mock_response.raise_for_status = MagicMock()

    with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        result = await client.check_server_version()
        assert result.version == "0.6.1-rc1"

    await client.close()


# ============================================================================
# _rest_get error handling
# ============================================================================


@pytest.mark.asyncio
async def test_rest_get_auth_error():
    """_rest_get should raise KaguraAuthError on 401."""
    client = _make_initialized_client()

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 401

    with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "401", request=MagicMock(), response=mock_response
        )
        with pytest.raises(KaguraAuthError):
            await client.get_embedding_status()

    await client.close()


@pytest.mark.asyncio
async def test_rest_get_connection_error():
    """_rest_get should raise KaguraConnectionError on network failure."""
    client = _make_initialized_client()

    with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = httpx.ConnectError("Connection refused")
        with pytest.raises(KaguraConnectionError, match="Connection failed"):
            await client.get_embedding_status()

    await client.close()


@pytest.mark.asyncio
async def test_context_manager():
    """async with should return client and close on exit."""
    async with KaguraClient(api_key="test", mcp_url="https://test.com/mcp") as client:
        assert isinstance(client, KaguraClient)


# ============================================================================
# Sleep Maintenance (issue #85)
# ============================================================================


@pytest.mark.asyncio
async def test_get_sleep_history_success():
    """get_sleep_history() returns a list of SleepReport models."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {
            "status": "success",
            "reports": [sleep_report_summary_dict("rid-1"), sleep_report_summary_dict("rid-2")],
            "count": 2,
        }
        result = await client.get_sleep_history(context_id="ctx-1", limit=5)

    assert len(result) == 2
    assert all(isinstance(r, SleepReport) for r in result)
    assert result[0].report_id == "rid-1"
    assert result[0].status == "completed"
    assert result[0].edges_created == 2
    mock.assert_called_once_with("get_sleep_history", {"context_id": "ctx-1", "limit": 5})

    await client.close()


@pytest.mark.asyncio
async def test_get_sleep_report_success():
    """get_sleep_report() flattens report + actions into SleepReportDetail."""
    client = _make_initialized_client()

    summary = sleep_report_summary_dict("rid-9")
    detail_extras = {
        "memories_flagged": 1,
        "embedding_calls_made": 4,
        "error_message": None,
        "edge_discovery_result": {"phase": "edge_discovery", "edges": 7},
        "dedup_result": None,
        "importance_result": None,
        "consolidation_result": None,
        "reindex_result": None,
    }
    actions = [
        {
            "id": "1",
            "phase": "edge_discovery",
            "action_type": "create_edge",
            "memory_id": "m-1",
            "target_id": "m-2",
            "details": {"weight": 0.9},
            "created_at": "2026-04-28T00:01:00",
        }
    ]
    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {
            "status": "success",
            "report": {**summary, **detail_extras},
            "actions": actions,
            "action_count": 1,
        }
        result = await client.get_sleep_report(context_id="ctx-1", report_id="rid-9")

    assert isinstance(result, SleepReportDetail)
    assert result.report_id == "rid-9"
    assert result.action_count == 1
    assert len(result.actions) == 1
    assert isinstance(result.actions[0], SleepAction)
    assert result.actions[0].action_type == "create_edge"
    assert result.actions[0].details == {"weight": 0.9}
    mock.assert_called_once_with("get_sleep_report", {"context_id": "ctx-1", "report_id": "rid-9"})

    await client.close()


@pytest.mark.asyncio
async def test_rollback_sleep_run_success():
    """rollback_sleep_run() returns RollbackResult on success."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {
            "status": "rolled_back",
            "report_id": "rid-9",
            "rollback_summary": {
                "edges_deleted": 5,
                "merges_reversed": 2,
                "importance_restored": 1,
                "promotions_reversed": 0,
                "archives_restored": 0,
                "errors": [],
            },
        }
        result = await client.rollback_sleep_run(context_id="ctx-1", report_id="rid-9")

    assert isinstance(result, RollbackResult)
    assert result.status == "rolled_back"
    assert result.rollback_summary.edges_deleted == 5
    assert result.rollback_summary.merges_reversed == 2
    assert result.rollback_summary.errors == []

    await client.close()


@pytest.mark.asyncio
async def test_get_sleep_history_auth_error_on_401():
    """HTTP 401 from MCP transport surfaces as KaguraAuthError."""
    client = _make_initialized_client()

    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "401", request=MagicMock(), response=mock_response
    )
    with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        with pytest.raises(KaguraAuthError):
            await client.get_sleep_history(context_id="ctx-1")

    await client.close()


@pytest.mark.asyncio
async def test_get_sleep_report_not_found_via_mcp_error():
    """MCP-level ``report_not_found`` surfaces as KaguraNotFoundError."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {
            "status": "error",
            "error": "report_not_found",
            "message": "Sleep report rid-x not found or not owned by you.",
        }
        with pytest.raises(KaguraNotFoundError, match="get_sleep_report"):
            await client.get_sleep_report(context_id="ctx-1", report_id="rid-x")

    await client.close()


@pytest.mark.asyncio
async def test_rollback_sleep_run_partial_failure():
    """``partial_rollback`` (some actions failed) raises KaguraError with code."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {
            "status": "error",
            "error": "partial_rollback",
            "message": "Rollback completed with 1 error(s).",
            "report_id": "rid-9",
            "rollback_summary": {
                "edges_deleted": 3,
                "merges_reversed": 0,
                "importance_restored": 0,
                "promotions_reversed": 0,
                "archives_restored": 0,
                "errors": ["Action 42 (merge): db error"],
            },
        }
        with pytest.raises(KaguraError, match="partial_rollback"):
            await client.rollback_sleep_run(context_id="ctx-1", report_id="rid-9")

    await client.close()


@pytest.mark.asyncio
async def test_get_sleep_history_connection_error_on_5xx():
    """HTTP 500 from MCP transport surfaces as KaguraConnectionError."""
    client = _make_initialized_client()

    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "500", request=MagicMock(), response=mock_response
    )
    with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        with pytest.raises(KaguraConnectionError):
            await client.get_sleep_history(context_id="ctx-1")

    await client.close()


# ============================================================================
# Edge CRUD tests
# ============================================================================


def _edge_dict(
    source_id: str = "src-uuid",
    target_id: str = "tgt-uuid",
    edge_type: str = "related_to",
    weight: float = 0.5,
    confidence: float = 1.0,
) -> dict:
    """Build a minimal server-shaped edge dict for use as ``_call_tool`` mock returns."""
    return {
        "source_id": source_id,
        "target_id": target_id,
        "edge_type": edge_type,
        "weight": weight,
        "confidence": confidence,
        "created_at": "2026-04-29T00:00:00",
        "last_updated": "2026-04-29T00:05:00",
    }


@pytest.mark.asyncio
async def test_list_edges_basic():
    """list_edges() should call tool and parse edges into Edge models."""
    from kagura_memory import Edge

    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {
            "memory_id": "mem-1",
            "edges": [_edge_dict(), _edge_dict(target_id="tgt-2", weight=0.8)],
            "count": 2,
        }
        edges = await client.list_edges(context_id="ctx-1", memory_id="mem-1")

        assert len(edges) == 2
        assert all(isinstance(e, Edge) for e in edges)
        assert edges[0].source_id == "src-uuid"
        assert edges[1].weight == 0.8

        tool_name = mock.call_args[0][0]
        args = mock.call_args[0][1]
        assert tool_name == "list_edges"
        assert args["context_id"] == "ctx-1"
        assert args["memory_id"] == "mem-1"
        assert args["min_weight"] == 0.0
        assert "edge_types" not in args
        assert "limit" not in args

    await client.close()


@pytest.mark.asyncio
async def test_list_edges_with_filters():
    """list_edges() should pass min_weight, edge_types, limit when provided."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"memory_id": "mem-1", "edges": [], "count": 0}
        await client.list_edges(
            context_id="ctx-1",
            memory_id="mem-1",
            min_weight=0.5,
            edge_types=["related_to", "depends_on"],
            limit=10,
        )
        args = mock.call_args[0][1]
        assert args["min_weight"] == 0.5
        assert args["edge_types"] == ["related_to", "depends_on"]
        assert args["limit"] == 10

    await client.close()


@pytest.mark.asyncio
async def test_list_edges_empty_response():
    """list_edges() should return [] when server returns no edges field."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"memory_id": "mem-1"}
        edges = await client.list_edges(context_id="ctx-1", memory_id="mem-1")
        assert edges == []

    await client.close()


@pytest.mark.asyncio
async def test_create_edge_basic():
    """create_edge() should call tool and return Edge model."""
    from kagura_memory import Edge

    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"edge": _edge_dict()}
        result = await client.create_edge(
            context_id="ctx-1",
            source_id="src-uuid",
            target_id="tgt-uuid",
        )

        assert isinstance(result, Edge)
        assert result.source_id == "src-uuid"
        assert result.target_id == "tgt-uuid"
        assert result.edge_type == "related_to"
        assert result.weight == 0.5
        assert result.confidence == 1.0

        tool_name = mock.call_args[0][0]
        args = mock.call_args[0][1]
        assert tool_name == "create_edge"
        assert args["context_id"] == "ctx-1"
        assert args["source_id"] == "src-uuid"
        assert args["target_id"] == "tgt-uuid"
        assert args["edge_type"] == "related_to"
        assert args["weight"] == 0.5
        assert args["confidence"] == 1.0

    await client.close()


@pytest.mark.asyncio
async def test_create_edge_custom_values():
    """create_edge() should pass custom edge_type, weight, confidence."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"edge": _edge_dict(edge_type="depends_on", weight=2.5, confidence=0.7)}
        result = await client.create_edge(
            context_id="ctx-1",
            source_id="a",
            target_id="b",
            edge_type="depends_on",
            weight=2.5,
            confidence=0.7,
        )
        assert result.edge_type == "depends_on"
        assert result.weight == 2.5
        assert result.confidence == 0.7

    await client.close()


@pytest.mark.asyncio
async def test_create_edge_self_loop_rejected():
    """create_edge() should raise ValueError when source_id == target_id."""
    client = _make_initialized_client()

    with pytest.raises(ValueError, match="self-loops are not allowed"):
        await client.create_edge(
            context_id="ctx-1",
            source_id="same-uuid",
            target_id="same-uuid",
        )

    await client.close()


@pytest.mark.asyncio
async def test_create_edge_accepts_unwrapped_response():
    """create_edge() should also handle the edge dict directly without 'edge' wrapper."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = _edge_dict()
        result = await client.create_edge(context_id="ctx-1", source_id="a", target_id="b")
        assert result.source_id == "src-uuid"

    await client.close()


@pytest.mark.asyncio
async def test_update_edge_weight_only():
    """update_edge() should send only weight when edge_type is None."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"edge": _edge_dict(weight=0.9)}
        result = await client.update_edge(
            context_id="ctx-1",
            source_id="a",
            target_id="b",
            weight=0.9,
        )
        assert result.weight == 0.9

        args = mock.call_args[0][1]
        assert args["weight"] == 0.9
        assert "edge_type" not in args

    await client.close()


@pytest.mark.asyncio
async def test_update_edge_type_only():
    """update_edge() should send only edge_type when weight is None."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"edge": _edge_dict(edge_type="depends_on")}
        await client.update_edge(
            context_id="ctx-1",
            source_id="a",
            target_id="b",
            edge_type="depends_on",
        )
        args = mock.call_args[0][1]
        assert args["edge_type"] == "depends_on"
        assert "weight" not in args

    await client.close()


@pytest.mark.asyncio
async def test_update_edge_both_fields():
    """update_edge() should send both weight and edge_type when both given."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"edge": _edge_dict(edge_type="learned_from", weight=1.5)}
        await client.update_edge(
            context_id="ctx-1",
            source_id="a",
            target_id="b",
            weight=1.5,
            edge_type="learned_from",
        )
        args = mock.call_args[0][1]
        assert args["weight"] == 1.5
        assert args["edge_type"] == "learned_from"

    await client.close()


@pytest.mark.asyncio
async def test_update_edge_neither_field():
    """update_edge() should send only the identifying triple when both are None."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"edge": _edge_dict()}
        await client.update_edge(
            context_id="ctx-1",
            source_id="a",
            target_id="b",
        )
        args = mock.call_args[0][1]
        assert args == {"context_id": "ctx-1", "source_id": "a", "target_id": "b"}

    await client.close()


@pytest.mark.asyncio
async def test_delete_edge_success():
    """delete_edge() should return True on success."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"deleted": True, "status": "success"}
        result = await client.delete_edge(context_id="ctx-1", source_id="a", target_id="b")
        assert result is True

        tool_name = mock.call_args[0][0]
        args = mock.call_args[0][1]
        assert tool_name == "delete_edge"
        assert args == {"context_id": "ctx-1", "source_id": "a", "target_id": "b"}

    await client.close()


@pytest.mark.asyncio
async def test_delete_edge_defaults_true_when_no_deleted_key():
    """delete_edge() should return True when no error and no 'deleted' key in response."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"status": "success"}
        result = await client.delete_edge(context_id="ctx-1", source_id="a", target_id="b")
        assert result is True

    await client.close()


def test_edge_model_ignores_extra_fields():
    """Edge model must accept (and silently drop) extra server fields.

    Guards against future server-side provenance additions
    (e.g. ``created_by``, ``origin``, ``frozen``) breaking older SDKs.
    """
    from kagura_memory import Edge

    edge = Edge.model_validate(
        {
            **_edge_dict(),
            "created_by": "user",
            "origin": "manual",
            "frozen": True,
            "future_unknown_field": {"nested": [1, 2, 3]},
        }
    )

    assert edge.source_id == "src-uuid"
    assert edge.weight == 0.5
    assert not hasattr(edge, "created_by")
    assert not hasattr(edge, "origin")


def test_edge_model_validates_weight_range():
    """Edge.weight must reject values outside [0.0, 3.0]."""
    from pydantic import ValidationError

    from kagura_memory import Edge

    # Below range
    with pytest.raises(ValidationError):
        Edge.model_validate({**_edge_dict(weight=-0.1)})

    # Above range
    with pytest.raises(ValidationError):
        Edge.model_validate({**_edge_dict(weight=3.5)})

    # Boundary values are valid
    Edge.model_validate({**_edge_dict(weight=0.0)})
    Edge.model_validate({**_edge_dict(weight=3.0)})


def test_edge_model_validates_confidence_range():
    """Edge.confidence must reject values outside [0.0, 1.0]."""
    from pydantic import ValidationError

    from kagura_memory import Edge

    with pytest.raises(ValidationError):
        Edge.model_validate({**_edge_dict(confidence=-0.1)})

    with pytest.raises(ValidationError):
        Edge.model_validate({**_edge_dict(confidence=1.5)})


@pytest.mark.asyncio
async def test_create_edge_surfaces_server_weight_error():
    """Server-side validation_error responses must raise KaguraError, not slip past
    as a Pydantic ValidationError on the error dict."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {
            "status": "error",
            "error": "validation_error",
            "message": "weight must be between 0.0 and 3.0.",
        }
        with pytest.raises(KaguraError, match="weight must be between"):
            await client.create_edge(
                context_id="ctx-1",
                source_id="a",
                target_id="b",
                weight=5.0,
            )

    await client.close()


@pytest.mark.asyncio
async def test_list_edges_surfaces_server_error():
    """list_edges() must raise on server-side error rather than returning []."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {
            "status": "error",
            "error": "memory_not_found",
            "message": "Memory not found.",
        }
        with pytest.raises(KaguraNotFoundError, match="list_edges"):
            await client.list_edges(context_id="ctx-1", memory_id="missing-mem")

    await client.close()


@pytest.mark.asyncio
async def test_update_edge_surfaces_server_error():
    """update_edge() must raise on server-side error rather than running model_validate
    on the error dict."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {
            "status": "error",
            "error": "edge_not_found",
            "message": "Edge does not exist.",
        }
        with pytest.raises(KaguraError, match="edge_not_found"):
            await client.update_edge(
                context_id="ctx-1",
                source_id="a",
                target_id="b",
                weight=0.7,
            )

    await client.close()


@pytest.mark.asyncio
async def test_delete_edge_surfaces_server_error():
    """delete_edge() must raise on server-side error rather than silently returning
    False, so callers can distinguish 'edge missing' from 'auth/permission failure'."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {
            "status": "error",
            "error": "edge_not_found",
            "message": "Edge does not exist.",
        }
        with pytest.raises(KaguraError, match="edge_not_found"):
            await client.delete_edge(context_id="ctx-1", source_id="a", target_id="b")

    await client.close()


@pytest.mark.asyncio
async def test_list_edges_raises_not_found_for_context_not_found():
    """list_edges() raises KaguraNotFoundError specifically for context_not_found code."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {
            "status": "error",
            "error": "context_not_found",
            "message": "Context not found.",
        }
        with pytest.raises(KaguraNotFoundError):
            await client.list_edges(context_id="ctx-missing", memory_id="mem-1")

    await client.close()


# ============================================================================
# list_tags (Issue #620)
# ============================================================================


def _list_tags_envelope(
    tags: list[dict] | None = None,
    *,
    context_id: str = "ctx-1",
    context_name: str = "my-project",
) -> dict:
    """Build a server-success envelope for the list_tags MCP tool."""
    tags = tags if tags is not None else []
    return {
        "status": "success",
        "context_id": context_id,
        "context_name": context_name,
        "tags": tags,
        "total": len(tags),
    }


@pytest.mark.asyncio
async def test_list_tags_basic():
    """list_tags() parses the server envelope into ListTagsResponse + TagInfo items."""
    from kagura_memory import ListTagsResponse, TagInfo

    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = _list_tags_envelope(
            [
                {"tag": "python", "count": 12, "last_used_at": "2026-05-01T10:00:00Z"},
                {"tag": "auth", "count": 3, "last_used_at": None},
            ]
        )
        result = await client.list_tags(context_id="ctx-1")

        assert isinstance(result, ListTagsResponse)
        assert result.context_id == "ctx-1"
        assert result.context_name == "my-project"
        assert result.total == 2
        assert len(result.tags) == 2
        assert all(isinstance(t, TagInfo) for t in result.tags)
        assert result.tags[0].tag == "python"
        assert result.tags[0].count == 12
        assert result.tags[0].last_used_at is not None
        assert result.tags[1].last_used_at is None

        tool_name, args = mock.call_args[0]
        assert tool_name == "list_tags"
        assert args["context_id"] == "ctx-1"
        assert args["limit"] == 50
        assert args["min_count"] == 1
        assert args["sort"] == "count"
        assert "prefix" not in args

    await client.close()


@pytest.mark.asyncio
async def test_list_tags_passes_all_params():
    """list_tags() forwards limit/min_count/sort/prefix to the MCP tool."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = _list_tags_envelope()
        await client.list_tags(
            context_id="ctx-1",
            limit=200,
            min_count=5,
            sort="recent",
            prefix="auth",
        )
        args = mock.call_args[0][1]
        assert args["limit"] == 200
        assert args["min_count"] == 5
        assert args["sort"] == "recent"
        assert args["prefix"] == "auth"

    await client.close()


@pytest.mark.asyncio
async def test_list_tags_empty_response():
    """list_tags() handles an empty tag list."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = _list_tags_envelope(context_name="empty-context")
        result = await client.list_tags(context_id="ctx-1")
        assert result.tags == []
        assert result.total == 0

    await client.close()


@pytest.mark.asyncio
async def test_list_tags_raises_not_found():
    """list_tags() raises KaguraNotFoundError on context_not_found."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {
            "status": "error",
            "error": "context_not_found",
            "message": "Context not found.",
        }
        with pytest.raises(KaguraNotFoundError, match="list_tags"):
            await client.list_tags(context_id="ctx-missing")

    await client.close()


@pytest.mark.asyncio
async def test_list_tags_surfaces_server_error():
    """list_tags() raises KaguraError on a generic server error code."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {
            "status": "error",
            "error": "invalid_argument",
            "message": "limit must be an integer between 1 and 500.",
        }
        with pytest.raises(KaguraError, match="invalid_argument"):
            await client.list_tags(context_id="ctx-1")

    await client.close()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"limit": 0}, "limit must be between"),
        ({"limit": 501}, "limit must be between"),
        ({"min_count": 0}, "min_count must be between"),
        ({"min_count": 10_001}, "min_count must be between"),
        ({"prefix": "x" * 201}, "prefix must be at most"),
    ],
)
@pytest.mark.asyncio
async def test_list_tags_arg_validation(kwargs, match):
    """list_tags() validates argument ranges client-side before issuing the call."""
    client = _make_initialized_client()
    try:
        with pytest.raises(ValueError, match=match):
            await client.list_tags(context_id="ctx-1", **kwargs)
    finally:
        await client.close()
