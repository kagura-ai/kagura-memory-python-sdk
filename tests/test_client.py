"""Tests for KaguraClient."""

import asyncio
import inspect
import json
import textwrap
import warnings
from datetime import UTC, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from kagura_memory import (
    MIN_SERVER_VERSION,
    Agent,
    AgentBinding,
    AgentBootstrapResponse,
    ContextInfo,
    DuplicatesResponse,
    EmbeddingStatus,
    KaguraAuthError,
    KaguraClient,
    KaguraConnectionError,
    KaguraError,
    KaguraNotFoundError,
    KaguraQuotaError,
    KaguraRateLimitError,
    MeasurementResult,
    MeasurementSeries,
    MemoryListItemLocation,
    MemoryListResponse,
    MemoryStatsResponse,
    RollbackResult,
    SeriesBucket,
    ServerFeatures,
    ServerInfo,
    SleepAction,
    SleepReport,
    SleepReportDetail,
    UsageInfo,
)
from tests.conftest import (
    SESSION_EXPIRED_BODY,
    FakeMcpServer,
    agent_binding_dict,
    agent_dict,
    bootstrap_envelope_dict,
    measurement_dict,
    measurement_series_dict,
    sleep_report_detail_dict,
    sleep_report_summary_dict,
)

# ============================================================================
# HTTPS enforcement (C-3)
# ============================================================================


@pytest.mark.parametrize(
    "mcp_url", ["http://evil.com/mcp", "HTTP://evil.com/mcp", " http://evil.com/mcp"]
)
def test_rejects_http_url(mcp_url):
    """HTTP URLs (non-localhost) should raise ValueError, in any case or padding (#274)."""
    with pytest.raises(ValueError, match="must use HTTPS"):
        KaguraClient(api_key="test", mcp_url=mcp_url)


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


@pytest.mark.asyncio
async def test_http_429_raises_rate_limit_with_retry_after():
    """A 429 surfaces as KaguraRateLimitError carrying the Retry-After seconds."""
    client = KaguraClient(api_key="test", mcp_url="https://test.com/mcp")

    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.headers = {"Retry-After": "12"}
    mock_response.json.return_value = {"detail": "slow down"}
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "429", request=MagicMock(), response=mock_response
    )

    with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        with pytest.raises(KaguraRateLimitError) as exc_info:
            await client._initialize_session()

    assert exc_info.value.retry_after == 12
    assert "slow down" in str(exc_info.value)
    await client.close()

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
# Expired MCP session recovery (#252)
# ============================================================================

_SESSION_EXPIRED_MESSAGE = SESSION_EXPIRED_BODY["error"]["message"]


def _client_on(handler) -> KaguraClient:
    client = KaguraClient(api_key="test", mcp_url="https://test.com/mcp")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


def _slow(server: FakeMcpServer):
    """``server.handler`` behind a short await, so concurrent calls interleave."""

    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.01)
        return server.handler(request)

    return handler


@pytest.mark.asyncio
async def test_expired_session_reinitializes_and_retries_once():
    """A 404 on a session request re-opens the session and retries the call once."""
    server = FakeMcpServer()
    client = _client_on(server.handler)
    try:
        assert await client._call_tool("list_contexts", {}) == {
            "status": "success",
            "session": "sess-1",
        }
        server.restart()  # idle-hour expiry or a deploy: every session is gone

        result = await client._call_tool("list_contexts", {})

        assert result == {"status": "success", "session": "sess-2"}
        assert server.calls()[2:] == [
            ("tools/call", "sess-1"),  # rejected: session gone
            ("initialize", None),  # exactly one re-initialize, without the stale id
            ("tools/call", "sess-2"),  # exactly one retry, on the new session
        ]
        assert client._session_id == "sess-2"

        # The recovered session is kept: no further initialize on the next call.
        await client._call_tool("list_contexts", {})
        assert server.methods().count("initialize") == 2
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_concurrent_calls_on_an_expired_session_share_one_reinitialize():
    """N in-flight calls that all hit the expired session open ONE new session, not N.

    ``Ingestor`` fans writes out with ``asyncio.gather``, so an ingest that
    spans a server restart would otherwise leave N-1 orphan sessions.
    """
    server = FakeMcpServer()
    client = _client_on(_slow(server))
    try:
        await client._call_tool("list_contexts", {})
        server.restart()

        results = await asyncio.gather(*[client._call_tool("list_contexts", {}) for _ in range(5)])

        assert results == [{"status": "success", "session": "sess-2"}] * 5
        assert server.methods().count("initialize") == 2  # the handshake + ONE re-open
        assert client._session_id == "sess-2"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_concurrent_first_calls_share_one_initialize():
    """Concurrent calls on a fresh client share the first handshake too."""
    server = FakeMcpServer()
    client = _client_on(_slow(server))
    try:
        await asyncio.gather(*[client._call_tool("list_contexts", {}) for _ in range(3)])
        assert server.methods() == ["initialize"] + ["tools/call"] * 3
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_expired_session_second_404_raises_with_server_message():
    """If the retry 404s too, raise KaguraConnectionError carrying the server's message."""
    server = FakeMcpServer()

    def handler(request: httpx.Request) -> httpx.Response:
        response = server.handler(request)
        server.restart()  # every session dies as soon as it is opened
        return response

    client = _client_on(handler)
    try:
        with pytest.raises(KaguraConnectionError, match=_SESSION_EXPIRED_MESSAGE) as exc_info:
            await client._call_tool("list_contexts", {})
        assert str(exc_info.value).startswith("HTTP 404: ")
        # initialize → call (404) → ONE re-initialize → ONE retry (404) → raise; no loop.
        assert server.methods() == ["initialize", "tools/call", "initialize", "tools/call"]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_expired_session_reinitialize_failure_raises_without_retry():
    """A failing re-initialize surfaces its own error; the call is not re-sent."""
    server = FakeMcpServer()
    client = _client_on(server.handler)
    try:
        await client._call_tool("list_contexts", {})
        server.restart()

        def down(request: httpx.Request) -> httpx.Response:
            if json.loads(request.content)["method"] == "initialize":
                server.record(request)
                return httpx.Response(503, text="Service Unavailable")
            return server.handler(request)

        await client._client.aclose()
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(down))
        with pytest.raises(KaguraConnectionError, match="HTTP 503"):
            await client._call_tool("list_contexts", {})
        assert server.methods()[2:] == ["tools/call", "initialize"]
        assert client._session_id is None  # next call starts a fresh session
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_401_on_session_request_raises_auth_error_without_reinitialize():
    """Only a 404 means "session gone": a 401 keeps its KaguraAuthError path."""
    server = FakeMcpServer()

    def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content)["method"] == "tools/call":
            server.record(request)
            return httpx.Response(401, json={"error": "invalid_token"})
        return server.handler(request)

    client = _client_on(handler)
    try:
        with pytest.raises(KaguraAuthError):
            await client._call_tool("list_contexts", {})
        assert server.methods() == ["initialize", "tools/call"]
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        pytest.param(
            400,
            {
                "jsonrpc": "2.0",
                "error": {"code": -32600, "message": "Invalid Request: missing method"},
                "id": None,
            },
            "HTTP 400: Invalid Request: missing method",
            id="jsonrpc-error",
        ),
        pytest.param(
            403,
            {
                "error": "workspace_mismatch",
                "error_description": "API key workspace does not match URL workspace. "
                "Use an API key scoped to this workspace.",
            },
            "HTTP 403: API key workspace does not match URL workspace",
            id="oauth-style-workspace-403",
        ),
    ],
)
async def test_mcp_4xx_surfaces_server_message(status: int, body: dict, expected: str):
    """A 4xx from the MCP endpoint reaches the caller as the server's message, not the reason."""
    server = FakeMcpServer()

    def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content)["method"] == "tools/call":
            return httpx.Response(status, json=body)
        return server.handler(request)

    client = _client_on(handler)
    try:
        with pytest.raises(KaguraConnectionError, match=expected):
            await client._call_tool("list_contexts", {})
    finally:
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
        assert "supersedes" not in args

    await client.close()


@pytest.mark.asyncio
async def test_remember_with_supersedes():
    """Issue #243: remember() forwards supersedes so the old memory is shadowed, not deleted."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"memory_id": "new-mem"}
        await client.remember(
            context_id="ctx",
            summary="s",
            content="c",
            supersedes="11111111-2222-3333-4444-555555555555",
        )
        args = mock.call_args[0][1]
        assert args["supersedes"] == "11111111-2222-3333-4444-555555555555"

    await client.close()


@pytest.mark.asyncio
async def test_recall_include_superseded():
    """Issue #243: superseded memories must be retrievable, or supersedes is a one-way door.

    ``include_superseded`` is a top-level tool argument, not a ``filters`` key.
    """
    client = _make_initialized_client()

    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = {"results": []}
            await client.recall(context_id="ctx", query="q", include_superseded=True)
            args = mock.call_args[0][1]
            assert args["include_superseded"] is True
            assert "include_superseded" not in args.get("filters", {})
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_recall_omits_include_superseded_by_default():
    """The default is the server's default — omit rather than send false."""
    client = _make_initialized_client()

    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = {"results": []}
            await client.recall(context_id="ctx", query="q")
            assert "include_superseded" not in mock.call_args[0][1]
    finally:
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


# delivery_mode constant + literal drift guard: assert both the public default
# and its literal value so a silent rename of either side fails fast.
_DEFAULT_DELIVERY_MODE = "on_recall"


@pytest.mark.asyncio
async def test_remember_with_delivery_mode_always():
    """remember() should pass delivery_mode when set to a non-default value."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"memory_id": "abc"}
        await client.remember(context_id="ctx", summary="s", content="c", delivery_mode="always")
        args = mock.call_args[0][1]
        assert args["delivery_mode"] == "always"

    await client.close()


@pytest.mark.asyncio
async def test_remember_delivery_mode_on_trigger():
    """remember() should pass delivery_mode='on_trigger' through verbatim."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"memory_id": "abc"}
        await client.remember(
            context_id="ctx", summary="s", content="c", delivery_mode="on_trigger"
        )
        args = mock.call_args[0][1]
        assert args["delivery_mode"] == "on_trigger"

    await client.close()


@pytest.mark.asyncio
async def test_remember_delivery_mode_default_not_sent():
    """remember() should omit delivery_mode when left at the default.

    The server applies ``server_default='on_recall'``; not sending the key keeps
    the SDK forward-compatible and avoids pinning the default into the payload.
    """
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"memory_id": "abc"}
        await client.remember(context_id="ctx", summary="s", content="c")
        args = mock.call_args[0][1]
        assert "delivery_mode" not in args

    await client.close()


@pytest.mark.asyncio
async def test_remember_delivery_mode_default_literal():
    """The remember() delivery_mode default is the literal 'on_recall'."""
    import inspect

    sig = inspect.signature(KaguraClient.remember)
    assert sig.parameters["delivery_mode"].default == _DEFAULT_DELIVERY_MODE
    assert _DEFAULT_DELIVERY_MODE == "on_recall"


@pytest.mark.asyncio
async def test_update_memory_with_delivery_mode():
    """update_memory() should pass delivery_mode to pin/unpin a memory."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"status": "success"}
        await client.update_memory(context_id="ctx", memory_id="mid", delivery_mode="always")
        args = mock.call_args[0][1]
        assert args["delivery_mode"] == "always"

    await client.close()


@pytest.mark.asyncio
async def test_update_memory_delivery_mode_not_sent_when_none():
    """update_memory() should omit delivery_mode when not provided."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"status": "success"}
        await client.update_memory(context_id="ctx", memory_id="mid", summary="s")
        args = mock.call_args[0][1]
        assert "delivery_mode" not in args

    await client.close()


@pytest.mark.asyncio
async def test_update_memory_sends_empty_values_that_clear_fields():
    """``""`` / ``{}`` / ``[]`` reach the wire: the server treats them as "clear".

    Only ``None`` means "leave unchanged", so a falsy-value shortcut in the
    argument builder would silently turn a clear into a no-op.
    """
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"status": "success", "memory_id": "mid"}
        await client.update_memory(
            context_id="ctx", memory_id="mid", context_summary="", details={}, tags=[]
        )
        args = mock.call_args[0][1]
        assert args["context_summary"] == ""
        assert args["details"] == {}
        assert args["tags"] == []

    await client.close()


def _load_pinned_response(*memory_ids: str, truncated: bool = False) -> dict:
    """The server's load_pinned envelope: the pinned set is under ``memories``."""
    return {
        "status": "success",
        "memories": [
            {
                "memory_id": mid,
                "summary": f"Guardrail {mid}",
                "context_summary": None,
                "type": "decision",
                "importance": 0.9,
                "delivery_mode": "always",
            }
            for mid in memory_ids
        ],
        "total_available": len(memory_ids),
        "truncated": truncated,
        "cap": 50,
        "context_id": "ctx",
        "context_name": "dev",
    }


@pytest.mark.asyncio
async def test_load_pinned_minimal():
    """load_pinned() should call the load_pinned MCP tool with context_id only."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = _load_pinned_response()
        result = await client.load_pinned(context_id="ctx")
        name, args = mock.call_args[0][0], mock.call_args[0][1]
        assert name == "load_pinned"
        assert args == {"context_id": "ctx"}
        assert result["truncated"] is False
        assert result["memories"] == []

    await client.close()


@pytest.mark.asyncio
async def test_load_pinned_with_cap():
    """load_pinned() should pass cap when provided."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = _load_pinned_response("m1", truncated=True)
        await client.load_pinned(context_id="ctx", cap=10)
        args = mock.call_args[0][1]
        assert args["cap"] == 10

    await client.close()


@pytest.mark.asyncio
async def test_load_pinned_cap_not_sent_when_none():
    """load_pinned() should omit cap when None (server default applies)."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = _load_pinned_response()
        await client.load_pinned(context_id="ctx")
        args = mock.call_args[0][1]
        assert "cap" not in args

    await client.close()


def _docstring_example(method) -> str:
    """Return the ``>>>`` / ``...`` lines of a method's docstring as source."""
    lines = []
    for raw in inspect.getdoc(method).splitlines():
        line = raw.strip()
        if line.startswith((">>> ", "... ")):
            lines.append(line[4:])
    return "\n".join(lines)


@pytest.mark.asyncio
@pytest.mark.parametrize("truncated", [False, True])
async def test_load_pinned_docstring_example_runs_against_server_shape(truncated):
    """The documented example must run against the real ``memories`` key (#257).

    It used to read ``pinned["results"]`` and raised ``KeyError`` against every
    server. When the first call is truncated the example re-calls with a larger
    cap, so both branches are exercised.
    """
    client = _make_initialized_client()
    source = _docstring_example(KaguraClient.load_pinned)
    assert source, "load_pinned docstring lost its example"
    namespace: dict = {}
    exec("async def _example(client, ctx):\n" + textwrap.indent(source, "    "), namespace)

    responses = {
        "load_pinned": [
            _load_pinned_response("m1", "m2", truncated=truncated),
            _load_pinned_response("m1", "m2", "m3"),
        ],
    }

    async def fake_call_tool(name, arguments):
        if name == "reference":
            return {"status": "success", "memory": {"memory_id": arguments["memory_id"]}}
        return responses[name].pop(0)

    with patch.object(
        client, "_call_tool", new_callable=AsyncMock, side_effect=fake_call_tool
    ) as mock:
        await namespace["_example"](client, "ctx")

    referenced = [c.args[1]["memory_id"] for c in mock.call_args_list if c.args[0] == "reference"]
    assert referenced == (["m1", "m2", "m3"] if truncated else ["m1", "m2"])
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
@pytest.mark.parametrize("use_rerank", [True, False])
@pytest.mark.parametrize(
    "target",
    [{"context_id": "ctx"}, {"context_ids": ["ctx-1", "ctx-2"]}],
    ids=["single", "cross-context"],
)
async def test_recall_sends_explicit_use_rerank(use_rerank, target):
    """Issue #251: an explicit use_rerank is sent as-is — False included.

    Since memory-cloud v0.69.0 (#1572) an omitted ``use_rerank`` follows the
    context's search config, so ``False`` must be sent — dropping it would let
    a rerank-enabled context rerank anyway.
    """
    client = _make_initialized_client()

    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = {"results": []}
            await client.recall(query="test", use_rerank=use_rerank, **target)
            args = mock.call_args[0][1]
            assert args["use_rerank"] is use_rerank
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_recall_omits_use_rerank_by_default():
    """Issue #251: the default (None) omits the key so the server follows the context config."""
    client = _make_initialized_client()

    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = {"results": []}
            await client.recall(context_id="ctx", query="test")
            assert "use_rerank" not in mock.call_args[0][1]
            await client.recall(context_id="ctx", query="test", use_rerank=None)
            assert "use_rerank" not in mock.call_args[0][1]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_recall_use_rerank_false_serializes_to_json_false():
    """Issue #251: the JSON-RPC body httpx puts on the wire carries ``"use_rerank": false``.

    A MockTransport captures the bytes httpx actually sends, so this pins the
    serialized request rather than the dict handed to ``post``.
    """
    sent: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request.content)
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"content": [{"type": "text", "text": '{"results": []}'}]},
            },
        )

    client = _make_initialized_client()
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    try:
        await client.recall(context_id="ctx", query="test", use_rerank=False)
        assert len(sent) == 1
        assert json.loads(sent[0])["params"]["arguments"]["use_rerank"] is False
    finally:
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
async def test_forget_requires_target():
    """forget() with neither memory_id nor query raises rather than sending a no-op."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        with pytest.raises(ValueError, match="memory_id or query"):
            await client.forget(context_id="ctx")
        mock.assert_not_called()

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
async def test_create_context_resource_id_is_deprecated_and_not_sent():
    """#273: the server's create_context never read resource_id, so it is not sent."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.side_effect = [
            {"status": "success", "contexts": [], "count": 0, "limit": 20, "can_create": True},
            {"id": "uuid-1", "name": "res-ctx"},
        ]
        with pytest.warns(DeprecationWarning, match=r"resource_id.*update_context") as record:
            await client.create_context(name="res-ctx", resource_id="my-resource")
        # Second call is create_context tool
        assert mock.call_args_list[1][0] == (
            "create_context",
            {"name": "res-ctx", "is_private": True},
        )
    # Attributed to the caller's line, not the SDK's.
    assert record[0].filename == __file__

    await client.close()


@pytest.mark.asyncio
async def test_create_context_without_resource_id_does_not_warn():
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.side_effect = [
            {"status": "success", "contexts": [], "count": 0, "limit": 20, "can_create": True},
            {"id": "uuid-1", "name": "ctx"},
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            await client.create_context(name="ctx")

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
@pytest.mark.parametrize(
    "quota_response",
    [
        # count/limit keys absent entirely
        {"status": "success", "contexts": [], "can_create": False},
        # count/limit present but null (JSON null — a distinct schema-drift form)
        {"status": "success", "contexts": [], "can_create": False, "count": None, "limit": None},
    ],
    ids=["missing-keys", "null-values"],
)
async def test_create_context_quota_exceeded_missing_count_limit(quota_response):
    """create_context() must raise KaguraQuotaError with a clean "(?/?)" message
    (not KeyError, not "(None/None)") when the quota response omits count/limit
    OR carries them as null (issue #183) — the message coerces both to "?"."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = quota_response

        with pytest.raises(KaguraQuotaError, match=r"Context limit reached \(\?/\?\)"):
            await client.create_context(name="over-limit")

    await client.close()


# Issue #255: the memory-cloud v0.73.0 slim list_contexts envelope (#1600).
# Items carry only id/name/is_private/is_locked/last_used_at; ``count`` is quota
# usage and ``total`` is the number of rows returned.
def _slim_list_contexts_envelope(*, count: int, limit: int, can_create: bool) -> dict:
    return {
        "status": "success",
        "contexts": [
            {
                "id": "ctx-1",
                "name": "dev",
                "is_private": True,
                "is_locked": False,
                "last_used_at": "2026-09-01T00:00:00Z",
            }
        ],
        "count": count,
        "total": 1,
        "limit": limit,
        "can_create": can_create,
    }


@pytest.mark.asyncio
async def test_list_contexts_default_sends_no_arguments():
    """Issue #255: a bare list_contexts() sends no options, so the server keeps its
    slim default shape."""
    client = _make_initialized_client()

    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = _slim_list_contexts_envelope(count=1, limit=20, can_create=True)
            await client.list_contexts()
            assert mock.call_args.args == ("list_contexts", {})
    finally:
        await client.close()


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"name_contains": "auth"}, {"name_contains": "auth"}),
        ({"include_summary": True}, {"include_summary": True}),
        ({"include_details": True}, {"include_details": True}),
        ({"include_stats": True}, {"include_stats": True}),
        (
            {
                "name_contains": "auth",
                "include_summary": True,
                "include_details": True,
                "include_stats": True,
            },
            {
                "name_contains": "auth",
                "include_summary": True,
                "include_details": True,
                "include_stats": True,
            },
        ),
    ],
    ids=["name_contains", "include_summary", "include_details", "include_stats", "all"],
)
@pytest.mark.asyncio
async def test_list_contexts_sends_only_the_options_set(kwargs, expected):
    """Issue #255: each list_contexts option is sent only when the caller sets it."""
    client = _make_initialized_client()

    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = _slim_list_contexts_envelope(count=1, limit=20, can_create=True)
            await client.list_contexts(**kwargs)
            assert mock.call_args.args == ("list_contexts", expected)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_list_contexts_omits_empty_name_contains():
    """Issue #255: an empty name_contains is not a filter, so it is not sent."""
    client = _make_initialized_client()

    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = _slim_list_contexts_envelope(count=1, limit=20, can_create=True)
            await client.list_contexts(name_contains="")
            assert mock.call_args.args == ("list_contexts", {})
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_list_contexts_options_are_keyword_only():
    """Issue #255: the flags are keyword-only, so a positional bool cannot misroute."""
    client = _make_initialized_client()

    try:
        with pytest.raises(TypeError):
            await client.list_contexts("auth")
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_list_contexts_passes_through_empty_account_hint():
    """Issue #255: the optional v0.75.0 ``hint`` (#1658) reaches the caller untouched."""
    client = _make_initialized_client()
    envelope = {
        "status": "success",
        "contexts": [],
        "count": 0,
        "total": 0,
        "limit": 3,
        "can_create": True,
        "hint": "No contexts are visible to you yet.",
    }

    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = envelope
            result = await client.list_contexts()
            assert result["hint"] == "No contexts are visible to you yet."
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_create_context_precheck_accepts_slim_list_contexts_response():
    """Issue #255: create_context's quota pre-check works on the v0.73.0 slim envelope.

    The pre-check only reads can_create/count/limit, and calls list_contexts with
    no options so it asks for the cheapest shape.
    """
    client = _make_initialized_client()

    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.side_effect = [
                _slim_list_contexts_envelope(count=1, limit=20, can_create=True),
                {"id": "uuid-1", "name": "new-ctx"},
            ]
            result = await client.create_context(name="new-ctx")
            assert result["name"] == "new-ctx"
            assert mock.call_args_list[0].args == ("list_contexts", {})
            assert mock.call_args_list[1].args[0] == "create_context"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_create_context_precheck_raises_quota_error_on_slim_response():
    """Issue #255: a slim envelope at its limit still raises KaguraQuotaError with count/limit."""
    client = _make_initialized_client()

    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = _slim_list_contexts_envelope(count=3, limit=3, can_create=False)
            with pytest.raises(KaguraQuotaError, match=r"Context limit reached \(3/3\)"):
                await client.create_context(name="over-limit")
            mock.assert_awaited_once()
    finally:
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
    """#273: the server requires name, so it defaults to resource_id; description is omitted."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {
            "context_id": "ctx-uuid",
            "context_name": "products",
            "resource_id": "products",
            "token": "kagura_resource_xyz",
            "token_id": 1,
        }
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            result = await client.setup_resource(resource_id="products")
        assert mock.call_args[0] == (
            "setup_resource",
            {"resource_id": "products", "name": "products", "quota_events_per_hour": 1000},
        )
        assert result["token"] == "kagura_resource_xyz"

    await client.close()


@pytest.mark.asyncio
async def test_setup_resource_with_all_args():
    """setup_resource() forwards an explicit name, description and quota."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {}
        await client.setup_resource(
            resource_id="products",
            name="product-catalog",
            description="Catalog ingestion token",
            quota_events_per_hour=5000,
        )
        assert mock.call_args[0][1] == {
            "resource_id": "products",
            "name": "product-catalog",
            "description": "Catalog ingestion token",
            "quota_events_per_hour": 5000,
        }

    await client.close()


@pytest.mark.asyncio
async def test_setup_resource_summary_is_deprecated_and_not_sent():
    """#273: the server's setup_resource has no summary, so it is not sent."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {}
        with pytest.warns(DeprecationWarning, match=r"summary.*update_context") as record:
            await client.setup_resource(resource_id="products", summary="All product data")
        assert "summary" not in mock.call_args[0][1]
    assert record[0].filename == __file__

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


@pytest.mark.parametrize(
    "id_kwargs",
    [{"memory_id": "mem-1"}, {"external_id": "ext-key"}],
    ids=["in-place", "upsert"],
)
@pytest.mark.asyncio
async def test_update_memory_passes_details(id_kwargs):
    """Issue #242: details is forwarded on both the in-place and upsert paths.

    The upsert path matters on its own: before this, ``update_memory(external_id=...)``
    could only produce a memory with no details at all, which made it useless as a
    de-duplication safety net.
    """
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"status": "success", "memory_id": "mem-1"}
        await client.update_memory(
            context_id="ctx",
            summary="s",
            content="c",
            type="note",
            details={"location": {"lat": 35.68, "lon": 139.76}},
            **id_kwargs,
        )
        args = mock.call_args[0][1]
        assert args["details"] == {"location": {"lat": 35.68, "lon": 139.76}}

    await client.close()


@pytest.mark.asyncio
async def test_update_memory_omits_details_when_none():
    """Issue #242: details is omitted when not provided, so existing callers are unaffected."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"status": "success", "memory_id": "mem-1"}
        await client.update_memory(context_id="ctx", memory_id="mem-1", summary="s")
        assert "details" not in mock.call_args[0][1]

    await client.close()


@pytest.mark.asyncio
async def test_update_memory_sends_dismiss_supersede_candidate():
    """Issue #255: a dismissal-only call sends the flag with memory_id and nothing else."""
    client = _make_initialized_client()

    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = {
                "status": "success",
                "memory_id": "mem-1",
                "supersede_candidate_dismissed": "mem-old",
            }
            result = await client.update_memory(
                context_id="ctx", memory_id="mem-1", dismiss_supersede_candidate=True
            )
            assert mock.call_args.args == (
                "update_memory",
                {"context_id": "ctx", "memory_id": "mem-1", "dismiss_supersede_candidate": True},
            )
            assert result["supersede_candidate_dismissed"] == "mem-old"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_update_memory_omits_dismiss_supersede_candidate_by_default():
    """Issue #255: the dismiss flag is not sent unless the caller sets it."""
    client = _make_initialized_client()

    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = {"status": "success", "memory_id": "mem-1"}
            await client.update_memory(
                context_id="ctx",
                memory_id="mem-1",
                summary="s",
                dismiss_supersede_candidate=False,
            )
            assert "dismiss_supersede_candidate" not in mock.call_args.args[1]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_update_memory_rejects_dismiss_with_external_id_before_network():
    """Issue #255: the server rejects the dismiss flag on an external_id upsert, so the
    SDK raises the same ValueError before any network call."""
    client = _make_initialized_client()

    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            with pytest.raises(ValueError, match="dismiss_supersede_candidate requires memory_id"):
                await client.update_memory(
                    context_id="ctx",
                    external_id="ext-key",
                    summary="upserted summary",
                    content="c",
                    type="note",
                    dismiss_supersede_candidate=True,
                )
            mock.assert_not_awaited()
    finally:
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
    """merge_contexts() should call tool with source/target context IDs."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"merged": 42, "source_id": "src", "target_id": "tgt"}
        result = await client.merge_contexts(source_id="src", target_id="tgt")
        args = mock.call_args[0][1]
        # memory-cloud #990 renamed the MCP params to disambiguate them from
        # the edge tools' source_id/target_id (which are memory UUIDs).
        assert args["source_context_id"] == "src"
        assert args["target_context_id"] == "tgt"
        assert "source_id" not in args
        assert "target_id" not in args
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
    """A malformed server payload (KaguraResponseError, #250) degrades to None."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        # Missing the required `context` field → get_context_info raises
        # KaguraResponseError (wrapping the pydantic ValidationError).
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

    try:
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
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_list_memories_window_params_omitted_when_none():
    """list_memories() omits the window params when not provided."""
    client = _make_initialized_client()

    try:
        with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _memory_list_response_mock()
            await client.list_memories(context_id="ctx-1")
            params = mock_get.call_args.kwargs["params"]
            assert "trigger_from" not in params
            assert "trigger_until" not in params
            assert "order_by" not in params
    finally:
        await client.close()


# ---- WHERE-axis bbox (memory-cloud #1334, server v0.54.0+) — Issue #254 ----

_BBOX_KEYS = ("lat_min", "lat_max", "lon_min", "lon_max")


@pytest.mark.asyncio
async def test_list_memories_bbox_all_bounds_forwarded():
    """list_memories() forwards all four bbox bounds when set."""
    client = _make_initialized_client()

    try:
        with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _memory_list_response_mock()
            await client.list_memories(
                context_id="ctx-1", lat_min=35.0, lat_max=36.0, lon_min=139.0, lon_max=140.0
            )
            params = mock_get.call_args.kwargs["params"]
            assert {k: params[k] for k in _BBOX_KEYS} == {
                "lat_min": 35.0,
                "lat_max": 36.0,
                "lon_min": 139.0,
                "lon_max": 140.0,
            }
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_list_memories_bbox_one_sided_bound():
    """A one-sided bound is forwarded alone — the other three stay off the wire."""
    client = _make_initialized_client()

    try:
        with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _memory_list_response_mock()
            await client.list_memories(lat_min=35.0)
            params = mock_get.call_args.kwargs["params"]
            assert params["lat_min"] == 35.0
            assert not {"lat_max", "lon_min", "lon_max"} & params.keys()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_list_memories_bbox_zero_bound_is_forwarded():
    """0.0 (the equator / prime meridian) is a real bound, not "unset"."""
    client = _make_initialized_client()

    try:
        with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _memory_list_response_mock()
            await client.list_memories(lat_min=0.0, lon_max=0)
            params = mock_get.call_args.kwargs["params"]
            assert params["lat_min"] == 0.0
            assert params["lon_max"] == 0
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_list_memories_bbox_antimeridian_box_passes_through():
    """lon_min > lon_max is the server's antimeridian-crossing box, not an error."""
    client = _make_initialized_client()

    try:
        with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _memory_list_response_mock()
            await client.list_memories(lon_min=170.0, lon_max=-170.0)
            params = mock_get.call_args.kwargs["params"]
            assert params["lon_min"] == 170.0
            assert params["lon_max"] == -170.0
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_list_memories_bbox_range_edges_are_forwarded():
    """±90 / ±180 are inside the range — the whole globe is a valid box."""
    client = _make_initialized_client()

    try:
        with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _memory_list_response_mock()
            await client.list_memories(lat_min=-90, lat_max=90, lon_min=-180, lon_max=180.0)
            params = mock_get.call_args.kwargs["params"]
            assert {k: params[k] for k in _BBOX_KEYS} == {
                "lat_min": -90,
                "lat_max": 90,
                "lon_min": -180,
                "lon_max": 180.0,
            }
    finally:
        await client.close()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"lat_min": -90.5}, "lat_min must be between -90 and 90"),
        ({"lat_max": 91}, "lat_max must be between -90 and 90"),
        ({"lon_min": -181.0}, "lon_min must be between -180 and 180"),
        ({"lon_max": 180.1}, "lon_max must be between -180 and 180"),
        ({"lat_min": float("nan")}, "lat_min must be between"),
        ({"lon_max": "140"}, "lon_max must be a number"),
        ({"lat_max": True}, "lat_max must be a number"),
    ],
)
@pytest.mark.asyncio
async def test_list_memories_bbox_bad_bound_rejected_locally(kwargs, match):
    """A bound the server would 422 raises ValueError before any request, like recall_nearby."""
    client = _make_initialized_client()

    try:
        with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
            with pytest.raises(ValueError, match=match):
                await client.list_memories(**kwargs)
            mock_get.assert_not_awaited()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_list_memories_bbox_omitted_when_none():
    """No bbox bound is sent when none is set (existing calls are unchanged)."""
    client = _make_initialized_client()

    try:
        with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _memory_list_response_mock()
            await client.list_memories(context_id="ctx-1")
            params = mock_get.call_args.kwargs["params"]
            assert not set(_BBOX_KEYS) & params.keys()
    finally:
        await client.close()


def test_list_memories_bbox_bounds_are_keyword_only():
    """The bbox bounds are keyword-only so they can never shift a positional caller."""
    import inspect

    params = inspect.signature(KaguraClient.list_memories).parameters
    for key in _BBOX_KEYS:
        assert params[key].kind is inspect.Parameter.KEYWORD_ONLY, key
        assert params[key].default is None


@pytest.mark.asyncio
async def test_list_memories_parses_item_location():
    """MemoryListItem.location carries the server's lat/lon; absent/None stays None."""
    client = _make_initialized_client()

    mock_response = _memory_list_response_mock()
    base = mock_response.json.return_value["memories"][0]
    mock_response.json.return_value["memories"] = [
        {**base, "id": "with-loc", "location": {"lat": 35.68, "lon": 139.76}},
        {**base, "id": "null-loc", "location": None},
        {**base, "id": "no-loc"},
    ]
    try:
        with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            result = await client.list_memories(lat_min=35.0)
            with_loc, null_loc, no_loc = result.memories
            assert isinstance(with_loc.location, MemoryListItemLocation)
            assert (with_loc.location.lat, with_loc.location.lon) == (35.68, 139.76)
            assert null_loc.location is None
            assert no_loc.location is None
    finally:
        await client.close()


# ============================================================================
# recall_upcoming (Time Memory, MCP)
# ============================================================================


@pytest.mark.asyncio
async def test_recall_upcoming_minimal():
    """recall_upcoming() with no bounds sends only context_id + the default k."""
    client = _make_initialized_client()

    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = {"results": []}
            await client.recall_upcoming(context_id="ctx")
            name, args = mock.call_args.args[0], mock.call_args.args[1]
            assert name == "recall_upcoming"
            assert args == {"context_id": "ctx", "k": 20}
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_recall_upcoming_maps_from_keyword_to_from_key():
    """recall_upcoming() maps the from_ param to the reserved-word "from" key."""
    client = _make_initialized_client()

    try:
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
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_recall_upcoming_omits_bounds_when_none():
    """recall_upcoming() omits from/until when not provided (k always sent)."""
    client = _make_initialized_client()

    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = {"results": []}
            await client.recall_upcoming(context_id="ctx")
            args = mock.call_args.args[1]
            assert "from" not in args
            assert "until" not in args
            assert args["k"] == 20
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_recall_upcoming_omits_include_details_by_default():
    """Issue #255: include_details is not sent unless set, so items keep the lean
    v0.73.0 ``trigger`` shape (#1599)."""
    client = _make_initialized_client()

    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = {"status": "success", "results": []}
            await client.recall_upcoming(context_id="ctx", include_details=False)
            assert "include_details" not in mock.call_args.args[1]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_recall_upcoming_sends_include_details_when_set():
    """Issue #255: include_details=True opts back into the full ``details`` per item."""
    client = _make_initialized_client()

    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = {"status": "success", "results": []}
            await client.recall_upcoming(context_id="ctx", include_details=True)
            assert mock.call_args.args == (
                "recall_upcoming",
                {"context_id": "ctx", "k": 20, "include_details": True},
            )
    finally:
        await client.close()


# ============================================================================
# recall_nearby (WHERE axis, MCP) — Issue #241
# ============================================================================


@pytest.mark.asyncio
async def test_recall_nearby_minimal():
    """recall_nearby() sends the point plus the documented radius/k defaults."""
    client = _make_initialized_client()

    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = {"results": []}
            await client.recall_nearby(context_id="ctx", lat=35.68, lon=139.76)
            name, args = mock.call_args.args[0], mock.call_args.args[1]
            assert name == "recall_nearby"
            assert args == {
                "context_id": "ctx",
                "lat": 35.68,
                "lon": 139.76,
                "radius_m": 1000,
                "k": 20,
            }
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_recall_nearby_passes_all_params():
    """recall_nearby() forwards an explicit radius and k."""
    client = _make_initialized_client()

    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = {"results": []}
            await client.recall_nearby(
                context_id="ctx", lat=-33.87, lon=151.21, radius_m=5000, k=50
            )
            args = mock.call_args.args[1]
            assert args["lat"] == -33.87
            assert args["lon"] == 151.21
            assert args["radius_m"] == 5000
            assert args["k"] == 50
    finally:
        await client.close()


@pytest.mark.parametrize(
    ("lat", "lon"),
    [(91.0, 0.0), (-91.0, 0.0), (0.0, 181.0), (0.0, -181.0)],
)
@pytest.mark.asyncio
async def test_recall_nearby_rejects_out_of_range_coordinates(lat, lon):
    """recall_nearby() rejects impossible coordinates locally instead of round-tripping a 422."""
    client = _make_initialized_client()

    try:
        with pytest.raises(ValueError, match="lat|lon"):
            await client.recall_nearby(context_id="ctx", lat=lat, lon=lon)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_recall_nearby_accepts_boundary_coordinates():
    """The poles and the antimeridian are valid points, not errors."""
    client = _make_initialized_client()

    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = {"results": []}
            await client.recall_nearby(context_id="ctx", lat=90.0, lon=180.0)
            await client.recall_nearby(context_id="ctx", lat=-90.0, lon=-180.0)
            assert mock.await_count == 2
    finally:
        await client.close()


# ============================================================================
# Measurement lane — HOW-MUCH axis (memory-cloud #1333, server v0.54.0+) — #254
# ============================================================================


@pytest.mark.asyncio
async def test_record_measurement_minimal_omits_optional_args():
    """record_measurement() sends only context_id/metric/value when the rest is None."""
    client = _make_initialized_client()

    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = measurement_dict(unit=None)
            result = await client.record_measurement("ctx", "weight_kg", 71.5)
            name, args = mock.call_args.args
            assert name == "record_measurement"
            assert args == {"context_id": "ctx", "metric": "weight_kg", "value": 71.5}
            assert isinstance(result, MeasurementResult)
            assert result.measurement_id == "cccccccc-dddd-eeee-ffff-000000000000"
            assert result.metric == "weight_kg"
            assert result.value == 71.5
            assert result.unit is None
            assert result.measured_at == datetime(2026, 9, 1, 7, 30, tzinfo=UTC)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_record_measurement_passes_all_args():
    """record_measurement() forwards measured_at (str as-is), unit and details."""
    client = _make_initialized_client()

    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = measurement_dict()
            result = await client.record_measurement(
                "ctx",
                "weight_kg",
                71.5,
                measured_at="2026-09-01T07:30:00Z",
                unit="kg",
                details={"device": "scale-1"},
            )
            args = mock.call_args.args[1]
            assert args == {
                "context_id": "ctx",
                "metric": "weight_kg",
                "value": 71.5,
                "measured_at": "2026-09-01T07:30:00Z",
                "unit": "kg",
                "details": {"device": "scale-1"},
            }
            assert result.unit == "kg"
    finally:
        await client.close()


@pytest.mark.parametrize(
    ("measured_at", "wire"),
    [
        # Naive = UTC on the server; sent without an offset.
        (datetime(2026, 9, 1, 7, 30), "2026-09-01T07:30:00"),
        # Aware keeps its offset; the server normalizes it to UTC.
        (
            datetime(2026, 9, 1, 16, 30, tzinfo=timezone(timedelta(hours=9))),
            "2026-09-01T16:30:00+09:00",
        ),
    ],
    ids=["naive", "aware"],
)
@pytest.mark.asyncio
async def test_record_measurement_serializes_datetime(measured_at, wire):
    """A datetime measured_at is sent as an ISO 8601 string (the tool takes strings)."""
    client = _make_initialized_client()

    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = measurement_dict()
            await client.record_measurement("ctx", "weight_kg", 71.5, measured_at=measured_at)
            assert mock.call_args.args[1]["measured_at"] == wire
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_record_measurement_accepts_int_value():
    """An int is a number — no float() coercion needed."""
    client = _make_initialized_client()

    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = measurement_dict(metric="reps", value=12.0, unit=None)
            await client.record_measurement("ctx", "reps", 12)
            assert mock.call_args.args[1]["value"] == 12
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_record_measurement_accepts_boundary_lengths():
    """A 64-char metric and a 32-char unit are the server's limits, not over them."""
    client = _make_initialized_client()

    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = measurement_dict()
            await client.record_measurement("ctx", "m" * 64, 1.0, unit="u" * 32)
            args = mock.call_args.args[1]
            assert len(args["metric"]) == 64
            assert len(args["unit"]) == 32
    finally:
        await client.close()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"metric": ""}, "metric"),
        ({"metric": "m" * 65}, "metric"),
        ({"metric": 42}, "metric"),
        ({"value": float("nan")}, "finite"),
        ({"value": float("inf")}, "finite"),
        ({"value": float("-inf")}, "finite"),
        # A JSON-sized int beyond float range makes isfinite() overflow.
        ({"value": 10**400}, "finite"),
        # bool is an int subclass, but True is never a measurement.
        ({"value": True}, "number"),
        # Strings are rejected, not coerced — the parameter is typed float.
        ({"value": "71.5"}, "number"),
        ({"value": None}, "number"),
        ({"unit": ""}, "unit"),
        ({"unit": "u" * 33}, "unit"),
    ],
    ids=[
        "empty-metric",
        "long-metric",
        "non-str-metric",
        "nan",
        "inf",
        "neg-inf",
        "overflow-int",
        "bool",
        "str-value",
        "none-value",
        "empty-unit",
        "long-unit",
    ],
)
@pytest.mark.asyncio
async def test_record_measurement_rejects_invalid_args(kwargs, match):
    """Invalid metric/value/unit fail locally, before any round-trip."""
    client = _make_initialized_client()
    call = {"context_id": "ctx", "metric": "weight_kg", "value": 71.5, **kwargs}

    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            with pytest.raises(ValueError, match=match):
                await client.record_measurement(**call)
            mock.assert_not_awaited()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_recall_series_minimal_omits_optional_args():
    """recall_series() sends only context_id/metric when the rest is None (server defaults)."""
    client = _make_initialized_client()

    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = measurement_series_dict(period="day")
            result = await client.recall_series("ctx", "weight_kg")
            name, args = mock.call_args.args
            assert name == "recall_series"
            assert args == {"context_id": "ctx", "metric": "weight_kg"}
            assert isinstance(result, MeasurementSeries)
            assert result.metric == "weight_kg"
            assert result.period == "day"
            assert result.agg == "avg"
            assert result.count == 2
            first = result.series[0]
            assert isinstance(first, SeriesBucket)
            assert first.bucket == datetime(2026, 8, 24, tzinfo=UTC)
            assert (first.value, first.count) == (72.0, 3)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_recall_series_passes_all_args():
    """recall_series() forwards period/agg and serializes datetime window bounds."""
    client = _make_initialized_client()

    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = measurement_series_dict(agg="max")
            await client.recall_series(
                "ctx",
                "weight_kg",
                period="week",
                agg="max",
                start=datetime(2026, 8, 1),
                end="2026-09-01T00:00:00Z",
            )
            assert mock.call_args.args[1] == {
                "context_id": "ctx",
                "metric": "weight_kg",
                "period": "week",
                "agg": "max",
                "start": "2026-08-01T00:00:00",
                "end": "2026-09-01T00:00:00Z",
            }
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_recall_series_empty_series():
    """An empty window is a valid, empty result — not an error."""
    client = _make_initialized_client()

    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = measurement_series_dict(series=[], count=0)
            result = await client.recall_series("ctx", "weight_kg")
            assert result.series == []
            assert result.count == 0
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_recall_series_tolerates_unknown_period_agg_and_extra_fields():
    """period/agg are Literal on input only; the response model stays forward-tolerant."""
    client = _make_initialized_client()

    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = measurement_series_dict(
                period="quarter", agg="p95", window={"days": 90}
            )
            result = await client.recall_series("ctx", "weight_kg")
            assert result.period == "quarter"
            assert result.agg == "p95"
    finally:
        await client.close()


@pytest.mark.parametrize("metric", ["", "m" * 65, None], ids=["empty", "long", "none"])
@pytest.mark.asyncio
async def test_recall_series_rejects_invalid_metric(metric):
    """recall_series() validates metric the same way record_measurement() does."""
    client = _make_initialized_client()

    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            with pytest.raises(ValueError, match="metric"):
                await client.recall_series("ctx", metric)
            mock.assert_not_awaited()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_record_measurement_surfaces_server_validation_error():
    """A server validation_error (e.g. a non-object details) raises KaguraError."""
    client = _make_initialized_client()

    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = {
                "status": "error",
                "error": "validation_error",
                "message": "'details' must be an object when provided",
            }
            with pytest.raises(KaguraError, match="validation_error"):
                await client.record_measurement("ctx", "weight_kg", 71.5)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_recall_series_surfaces_window_too_wide():
    """The 365-day window cap is the server's to enforce; its error surfaces as KaguraError."""
    client = _make_initialized_client()

    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = {
                "status": "error",
                "error": "validation_error",
                "message": "Window too wide: maximum lookback is 365 days",
            }
            with pytest.raises(KaguraError, match="Window too wide"):
                await client.recall_series(
                    "ctx", "weight_kg", start="2024-01-01T00:00:00", end="2026-01-01T00:00:00"
                )
    finally:
        await client.close()


# ============================================================================
# recall trust_tier filter (provenance, #173)
# ============================================================================


@pytest.mark.asyncio
async def test_recall_passes_trust_tier_filter():
    """recall() should pass a trust_tier filter straight through to the tool."""
    client = _make_initialized_client()

    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = {"results": []}
            await client.recall(context_id="ctx", query="auth", filters={"trust_tier": "trusted"})
            args = mock.call_args[0][1]
            assert args["filters"] == {"trust_tier": "trusted"}
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_recall_omits_filters_when_not_given():
    """recall() should not send a filters key when no filters are provided."""
    client = _make_initialized_client()

    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = {"results": []}
            await client.recall(context_id="ctx", query="auth")
            args = mock.call_args[0][1]
            assert "filters" not in args
    finally:
        await client.close()


# ============================================================================
# feedback (retrieval signal, #174)
# ============================================================================


@pytest.mark.asyncio
async def test_feedback_minimal():
    """feedback() should send context_id, memory_id, and helpful only."""
    client = _make_initialized_client()

    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = {"status": "ok"}
            result = await client.feedback(context_id="ctx", memory_id="mem", helpful=True)
            name, args = mock.call_args[0][0], mock.call_args[0][1]
            assert name == "feedback"
            assert args == {"context_id": "ctx", "memory_id": "mem", "helpful": True}
            assert result["status"] == "ok"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_feedback_with_query_and_note():
    """feedback() should pass optional query and note when provided."""
    client = _make_initialized_client()

    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = {"status": "ok"}
            await client.feedback(
                context_id="ctx",
                memory_id="mem",
                helpful=False,
                query="auth flow",
                note="off-topic result",
            )
            args = mock.call_args[0][1]
            assert args["helpful"] is False
            assert args["query"] == "auth flow"
            assert args["note"] == "off-topic result"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_feedback_omits_optionals_when_none():
    """feedback() should omit query and note when not provided."""
    client = _make_initialized_client()

    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = {"status": "ok"}
            await client.feedback(context_id="ctx", memory_id="mem", helpful=True)
            args = mock.call_args[0][1]
            assert "query" not in args
            assert "note" not in args
    finally:
        await client.close()


# ============================================================================
# set_state / get_state (agent session-state lane, #175)
# ============================================================================


@pytest.mark.asyncio
async def test_set_state_minimal():
    """set_state() should send context_id, key, and value without a TTL."""
    client = _make_initialized_client()

    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = {"status": "ok"}
            await client.set_state(context_id="ctx", key="step", value={"n": 3, "phase": "build"})
            name, args = mock.call_args[0][0], mock.call_args[0][1]
            assert name == "set_state"
            assert args == {
                "context_id": "ctx",
                "key": "step",
                "value": {"n": 3, "phase": "build"},
            }
            assert "ttl_seconds" not in args
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_set_state_with_ttl():
    """set_state() should pass ttl_seconds when provided."""
    client = _make_initialized_client()

    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = {"status": "ok"}
            await client.set_state(context_id="ctx", key="lock", value=True, ttl_seconds=300)
            args = mock.call_args[0][1]
            assert args["ttl_seconds"] == 300
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_get_state_single_key():
    """get_state() should send the key when reading one value."""
    client = _make_initialized_client()

    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = {"key": "step", "value": {"n": 3}}
            await client.get_state(context_id="ctx", key="step")
            name, args = mock.call_args[0][0], mock.call_args[0][1]
            assert name == "get_state"
            assert args == {"context_id": "ctx", "key": "step"}
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_get_state_lists_all_when_key_omitted():
    """get_state() should omit key to list all live entries for the context."""
    client = _make_initialized_client()

    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = {"entries": []}
            await client.get_state(context_id="ctx")
            args = mock.call_args[0][1]
            assert args == {"context_id": "ctx"}
            assert "key" not in args
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_feedback_surfaces_server_error():
    """feedback() must raise on a server-side error rather than returning the error dict."""
    client = _make_initialized_client()

    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = {
                "status": "error",
                "error": "memory_not_found",
                "message": "Memory not found.",
            }
            with pytest.raises(KaguraNotFoundError, match="feedback"):
                await client.feedback(context_id="ctx", memory_id="missing", helpful=True)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_set_state_surfaces_server_error():
    """set_state() must raise on a server-side error rather than returning the error dict."""
    client = _make_initialized_client()

    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = {
                "status": "error",
                "error": "context_not_found",
                "message": "Context not found.",
            }
            with pytest.raises(KaguraNotFoundError, match="set_state"):
                await client.set_state(context_id="missing", key="k", value="v")
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_get_state_surfaces_server_error():
    """get_state() must raise on a server-side error rather than returning the error dict."""
    client = _make_initialized_client()

    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = {
                "status": "error",
                "error": "context_not_found",
                "message": "Context not found.",
            }
            with pytest.raises(KaguraNotFoundError, match="get_state"):
                await client.get_state(context_id="missing")
    finally:
        await client.close()


# ============================================================================
# Issue #180 — unified MCP-error translation across the raw-dict family
# ============================================================================

# Every MCP tool method must translate a server {"status": "error", ...}
# response into a typed exception instead of returning it as data. Each entry
# invokes one method with minimal args; the patched _call_tool returns a
# context_not_found error, so the method must raise KaguraNotFoundError.
_ERROR_TRANSLATING_METHODS = [
    ("remember", lambda c: c.remember(context_id="c", summary="s", content="x")),
    ("recall", lambda c: c.recall(context_id="c", query="q")),
    ("recall_upcoming", lambda c: c.recall_upcoming(context_id="c")),
    ("recall_nearby", lambda c: c.recall_nearby(context_id="c", lat=0.0, lon=0.0)),
    ("record_measurement", lambda c: c.record_measurement("c", "m", 1.0)),
    ("recall_series", lambda c: c.recall_series("c", "m")),
    ("load_pinned", lambda c: c.load_pinned(context_id="c")),
    ("list_contexts", lambda c: c.list_contexts()),
    ("explore", lambda c: c.explore(context_id="c", memory_id="m")),
    ("reference", lambda c: c.reference(context_id="c", memory_id="m")),
    ("update_memory", lambda c: c.update_memory(context_id="c", memory_id="m", summary="s")),
    ("forget", lambda c: c.forget(context_id="c", memory_id="m")),
    ("delete_context", lambda c: c.delete_context(context_id="c")),
    ("update_context", lambda c: c.update_context(context_id="c", display_name="d")),
    ("setup_resource", lambda c: c.setup_resource(resource_id="r")),
    ("merge_contexts", lambda c: c.merge_contexts(source_id="a", target_id="b")),
    ("update_search_config", lambda c: c.update_search_config(context_id="c")),
    ("get_usage", lambda c: c.get_usage()),
    ("get_context_info", lambda c: c.get_context_info(context_id="c")),
    # create_context calls list_contexts() first; the mocked error surfaces there.
    ("create_context", lambda c: c.create_context(name="n")),
]


@pytest.mark.parametrize(
    "invoke",
    [m[1] for m in _ERROR_TRANSLATING_METHODS],
    ids=[m[0] for m in _ERROR_TRANSLATING_METHODS],
)
@pytest.mark.asyncio
async def test_mcp_methods_raise_on_server_error(invoke):
    """Issue #180: raw-dict tool methods translate MCP errors into exceptions, not data."""
    client = _make_initialized_client()

    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = {
                "status": "error",
                "error": "context_not_found",
                "message": "Context not found.",
            }
            with pytest.raises(KaguraNotFoundError):
                await invoke(client)
    finally:
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


# The ``features`` block memory-cloud v0.76.0 sends, unchanged in v0.77.0
# (backend/src/api/routes/system.py).
# ``extra="allow"`` keeps an untyped flag in ``model_extra`` and ``model_dump()``,
# so the tests below also assert ``model_extra`` / ``model_fields`` to prove each
# flag is a typed field.
_V076_FEATURES = {
    "neural_memory": True,
    "research_tools": False,
    "plan_page": True,
    "byok": True,
    "cost_display": False,
    "managed_connectors": True,
    "managed_llm": True,
    "referrals": False,
    "beta_invites": True,
    "reranking": True,
}
_V076_SEARCH_DEFAULTS = {
    "use_rerank": True,
    "reranker_provider": "self_hosted",
    "reranker_model": "bge-reranker-v2-m3",
}


def _server_info_response(payload: dict) -> MagicMock:
    response = MagicMock()
    response.json.return_value = payload
    response.raise_for_status = MagicMock()
    return response


@pytest.mark.asyncio
async def test_get_server_info_exposes_v076_features_and_search_defaults():
    """Every v0.76.0 flag and ``search_defaults`` survive parsing (#257)."""
    client = _make_initialized_client()
    payload = {
        "name": "Kagura Memory Cloud",
        "version": "0.76.0",
        "description": "Remote MCP Server + Web Management",
        "environment": "production",
        "search_defaults": _V076_SEARCH_DEFAULTS,
        "features": _V076_FEATURES,
    }

    with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = _server_info_response(payload)
        result = await client.get_server_info()

    assert set(ServerFeatures.model_fields) == set(_V076_FEATURES)
    assert result.features.model_extra == {}
    assert result.features.model_dump() == _V076_FEATURES
    assert result.search_defaults == _V076_SEARCH_DEFAULTS
    await client.close()


@pytest.mark.parametrize(
    ("sent", "expected"),
    [
        ({"version": "0.77.0", "terms_version": "2026-09"}, "2026-09"),
        ({"version": "0.77.0", "terms_version": None}, None),
        ({"version": "0.76.0"}, None),
    ],
    ids=["set", "null", "absent"],
)
@pytest.mark.asyncio
async def test_get_server_info_reads_terms_version(sent: dict, expected: str | None):
    """memory-cloud v0.77.0 sends ``terms_version`` (#1665), null when acceptance is off.

    A server before v0.77.0 omits the key. Both read as ``None``.
    """
    client = _make_initialized_client()
    payload = {"name": "Kagura Memory Cloud", **sent, "features": _V076_FEATURES}

    try:
        with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _server_info_response(payload)
            result = await client.get_server_info()
    finally:
        await client.close()

    assert result.terms_version == expected
    # A top-level key, not a feature flag.
    assert result.features.model_extra == {}


def test_server_features_keeps_unknown_future_flags():
    """A flag newer than the SDK is kept in ``model_extra``, not dropped."""
    info = ServerInfo.model_validate(
        {
            "name": "Kagura Memory Cloud",
            "version": "9.0.0",
            "features": {**_V076_FEATURES, "brand_new_flag": True},
        }
    )
    assert info.features.model_extra == {"brand_new_flag": True}
    assert info.features.beta_invites is True


def test_server_info_from_an_older_server_defaults_missing_fields():
    """Flags a server does not send read as ``False``; ``search_defaults`` as ``None``."""
    info = ServerInfo.model_validate(
        {"name": "Kagura Memory Cloud", "version": "0.53.0", "features": {"neural_memory": True}}
    )
    assert info.features.neural_memory is True
    assert info.features.reranking is False
    assert info.features.beta_invites is False
    assert info.search_defaults is None
    assert info.features.model_extra == {}


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
            assert "is below" not in caplog.text

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
async def test_check_server_version_non_semver(caplog):
    """check_server_version() reads a pre-release suffix instead of skipping the check."""
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
        import logging

        with caplog.at_level(logging.WARNING, logger="kagura_memory"):
            result = await client.check_server_version()
            assert result.version == "0.6.1-rc1"
            assert "Server version 0.6.1-rc1 is below" in caplog.text

    await client.close()


def test_min_server_version_tuple_is_parsed_from_the_constant():
    from kagura_memory.client import _MIN_SERVER_VERSION_TUPLE

    assert MIN_SERVER_VERSION == "0.17.1"
    assert _MIN_SERVER_VERSION_TUPLE == (0, 17, 1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("version", "warns"),
    [
        ("0.6.1-rc1", True),
        ("0.16.9-beta", True),
        ("v0.16.0", True),
        ("0.17.0-rc1", True),
        ("0.17.1-rc1", True),  # a pre-release of the minimum comes before it
        ("0.9.99", True),
        (MIN_SERVER_VERSION, False),
        (f"v{MIN_SERVER_VERSION}", False),
        ("0.17.1+build", False),
        ("0.17.2-rc1", False),
        ("0.76.0", False),
        # Unparseable: no verdict, so no warning (kagura doctor reports "info").
        ("0.17", False),
        ("main-abc123", False),
        ("", False),
    ],
)
async def test_check_server_version_verdicts(caplog, version, warns):
    """check_server_version() warns exactly when the version is below the minimum."""
    import logging

    client = _make_initialized_client()
    mock_response = MagicMock()
    mock_response.json.return_value = {"name": "Kagura Memory Cloud", "version": version}
    mock_response.raise_for_status = MagicMock()

    try:
        with (
            patch.object(client._client, "get", new_callable=AsyncMock, return_value=mock_response),
            caplog.at_level(logging.WARNING, logger="kagura_memory"),
        ):
            result = await client.check_server_version()
    finally:
        await client.close()

    assert result.version == version
    warnings = [r.getMessage() for r in caplog.records if "tested minimum" in r.getMessage()]
    if warns:
        assert warnings == [
            f"Server version {version} is below the SDK's tested minimum {MIN_SERVER_VERSION}. "
            "Some features may not work; older servers may silently ignore unknown parameters."
        ]
    else:
        assert warnings == []


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


@pytest.mark.parametrize("status", [404, 422])
@pytest.mark.parametrize("method", ["get_embedding_status", "list_memories"])
@pytest.mark.asyncio
async def test_rest_get_keeps_404_and_422_as_connection_errors_without_an_operation(method, status):
    """Only a caller that names an MCP operation (list_tags, #273) remaps 404/422."""
    client = _make_initialized_client()
    response = httpx.Response(
        status,
        json={"error": f"HTTP-{status}", "message": "Not here", "details": {}},
        request=httpx.Request("GET", "https://test.com/api/v1/x"),
    )
    args = ("ctx-1",) if method == "list_memories" else ()
    try:
        with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = response
            with pytest.raises(KaguraConnectionError) as exc:
                await getattr(client, method)(*args)
    finally:
        await client.close()

    assert not isinstance(exc.value, KaguraNotFoundError)
    assert str(exc.value) == f"HTTP {status}: Not here"


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
async def test_get_sleep_history_returns_every_run_when_degraded_runs_mixed_in():
    """One ``degraded`` run (server v0.43.0+, #1183) must not fail the whole listing."""
    client = _make_initialized_client()

    degraded = {**sleep_report_summary_dict("rid-2"), "status": "degraded", "llm_call_failures": 3}
    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {
            "status": "success",
            "reports": [
                sleep_report_summary_dict("rid-1"),
                degraded,
                sleep_report_summary_dict("rid-3"),
            ],
            "count": 3,
        }
        result = await client.get_sleep_history(context_id="ctx-1")

    assert [r.report_id for r in result] == ["rid-1", "rid-2", "rid-3"]
    assert [r.status for r in result] == ["completed", "degraded", "completed"]
    assert result[1].llm_call_failures == 3

    await client.close()


@pytest.mark.asyncio
async def test_get_sleep_report_degraded_run():
    """get_sleep_report() parses a degraded run, incl. ``merge_retention_result``."""
    client = _make_initialized_client()

    report = sleep_report_detail_dict(
        "rid-9", status="degraded", llm_call_failures=1, merge_retention_result={"purged": 2}
    )
    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {
            "status": "success",
            "report": report,
            "actions": [],
            "action_count": 0,
        }
        result = await client.get_sleep_report(context_id="ctx-1", report_id="rid-9")

    assert result.status == "degraded"
    assert result.llm_call_failures == 1
    assert result.merge_retention_result == {"purged": 2}

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


@pytest.mark.parametrize("with_tags", [None, [], ["  ", ""]])
@pytest.mark.asyncio
async def test_list_tags_stays_on_mcp_without_a_drill_down(with_tags):
    """No with_tags, an empty list, or only blank values: MCP list_tags, no with_tags.

    An empty drill-down is a no-op server-side (``tags @> '{}'``), and blank
    values are dropped before it is judged empty (#273).
    """
    client = _make_initialized_client()

    with (
        patch.object(client, "_call_tool", new_callable=AsyncMock) as mock,
        patch.object(client._client, "get", new_callable=AsyncMock) as mock_get,
    ):
        mock.return_value = _list_tags_envelope()
        await client.list_tags(context_id="ctx-1", with_tags=with_tags)
        assert mock.call_args[0] == (
            "list_tags",
            {"context_id": "ctx-1", "limit": 50, "min_count": 1, "sort": "count"},
        )
        mock_get.assert_not_called()

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


# ----------------------------------------------------------------------------
# list_tags with_tags drill-down over REST (#273)
#
# MCP list_tags has no with_tags before memory-cloud v0.77.0 (memory-cloud
# #1669) and silently returned the unfiltered vocabulary, so a drill-down goes
# to GET /api/v1/contexts/{id}/tags on every server; the route has had it
# since v0.17.2.
# ----------------------------------------------------------------------------

_TAGS_CTX = "c1"
_TAGS_PATH = f"/api/v1/contexts/{_TAGS_CTX}/tags"


def _rest_tags_body(context_id: str = _TAGS_CTX, **overrides) -> dict:
    """What the REST tags route returns: no ``status``, plus ``sample_summary``.

    No ``context_name`` either: the route sends it only from memory-cloud v0.77.0.
    """
    body = {
        "context_id": context_id,
        "tags": [
            {
                "tag": "when:2026-09",
                "count": 2,
                "sample_summary": None,
                "last_used_at": "2026-09-01T00:00:00Z",
            }
        ],
        "total": 1,
    }
    body.update(overrides)
    return body


class _TagsServer:
    """MCP ``list_tags`` and the REST tags route behind one ``httpx.MockTransport``."""

    def __init__(self, *, rest_body=None, rest_status: int = 200, mcp_result=None) -> None:
        self.rest_body = rest_body if rest_body is not None else _rest_tags_body()
        self.rest_status = rest_status
        self.mcp_result = (
            mcp_result
            if mcp_result is not None
            else _list_tags_envelope(context_id=_TAGS_CTX, context_name="demo")
        )
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.method == "GET":
            return httpx.Response(self.rest_status, json=self.rest_body)
        body = json.loads(request.content)
        if body["method"] == "initialize":
            result: dict = {"serverInfo": {}}
        else:
            result = {"content": [{"type": "text", "text": json.dumps(self.mcp_result)}]}
        return httpx.Response(
            200,
            json={"jsonrpc": "2.0", "id": body["id"], "result": result},
            headers={"mcp-session-id": "sess-1"},
        )

    def gets(self) -> list[httpx.Request]:
        return [r for r in self.requests if r.method == "GET"]

    def tool_calls(self) -> list[dict]:
        bodies = [json.loads(r.content) for r in self.requests if r.method == "POST"]
        return [b["params"] for b in bodies if b["method"] == "tools/call"]


def _tags_client(server: _TagsServer, mcp_url: str = "https://test.com/mcp") -> KaguraClient:
    client = KaguraClient(api_key="test-key", mcp_url=mcp_url)
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(server.handler),
        headers={"Authorization": "Bearer test-key"},
    )
    return client


@pytest.mark.asyncio
async def test_list_tags_with_tags_goes_to_the_rest_route_as_repeated_keys():
    """The drill-down is a GET with one trimmed with_tags key per tag; MCP never sees it."""
    server = _TagsServer()
    client = _tags_client(server)
    try:
        await client.list_tags(
            context_id=_TAGS_CTX,
            prefix="when:",
            with_tags=["client:acme.co.jp", " kind:invoice ", "  "],
        )
    finally:
        await client.close()

    rest = server.gets()[0]
    assert server.requests[0] is rest  # REST first, so its errors are what a caller sees
    assert f"{rest.url.scheme}://{rest.url.host}{rest.url.path}" == f"https://test.com{_TAGS_PATH}"
    # Repeated keys, never comma-joined: the server reads "a,b" as ONE tag.
    assert rest.url.params.get_list("with_tags") == ["client:acme.co.jp", "kind:invoice"]
    assert rest.url.params["limit"] == "50"
    assert rest.url.params["min_count"] == "1"
    assert rest.url.params["sort"] == "count"
    assert rest.url.params["prefix"] == "when:"
    assert rest.headers["authorization"] == "Bearer test-key"
    for call in server.tool_calls():
        assert "with_tags" not in call["arguments"]


@pytest.mark.asyncio
async def test_list_tags_with_tags_passes_limit_min_count_sort_and_omits_empty_prefix():
    server = _TagsServer()
    client = _tags_client(server)
    try:
        await client.list_tags(
            context_id=_TAGS_CTX, limit=7, min_count=3, sort="alpha", with_tags=["a"]
        )
    finally:
        await client.close()

    params = server.gets()[0].url.params
    assert params["limit"] == "7"
    assert params["min_count"] == "3"
    assert params["sort"] == "alpha"
    assert "prefix" not in params


@pytest.mark.asyncio
async def test_list_tags_with_tags_returns_the_mcp_shape_naming_the_context_via_list_tags():
    """Before v0.77.0 the route sends no context_name: one list_tags limit=1 call supplies it."""
    from kagura_memory import ListTagsResponse

    server = _TagsServer()
    client = _tags_client(server)
    try:
        result = await client.list_tags(context_id=_TAGS_CTX, with_tags=["client:acme"])
    finally:
        await client.close()

    assert isinstance(result, ListTagsResponse)
    assert result.context_id == _TAGS_CTX
    assert result.context_name == "demo"
    assert result.total == 1
    assert [(t.tag, t.count) for t in result.tags] == [("when:2026-09", 2)]
    assert result.tags[0].last_used_at == datetime(2026, 9, 1, tzinfo=UTC)
    assert server.tool_calls() == [
        {"name": "list_tags", "arguments": {"context_id": _TAGS_CTX, "limit": 1}}
    ]


@pytest.mark.asyncio
async def test_list_tags_with_tags_maps_a_missing_last_used_at_to_none():
    server = _TagsServer(rest_body=_rest_tags_body(tags=[{"tag": "a", "count": 1}]))
    client = _tags_client(server)
    try:
        result = await client.list_tags(context_id=_TAGS_CTX, with_tags=["b"])
    finally:
        await client.close()
    assert [(t.tag, t.count, t.last_used_at) for t in result.tags] == [("a", 1, None)]


@pytest.mark.asyncio
async def test_list_tags_with_tags_reuses_the_name_a_plain_call_returned():
    server = _TagsServer()
    client = _tags_client(server)
    try:
        await client.list_tags(context_id=_TAGS_CTX)
        result = await client.list_tags(context_id=_TAGS_CTX, with_tags=["client:acme"])
    finally:
        await client.close()

    assert result.context_name == "demo"
    assert len(server.tool_calls()) == 1  # the plain call only


@pytest.mark.asyncio
async def test_list_tags_with_tags_looks_a_name_up_once_per_client():
    server = _TagsServer()
    client = _tags_client(server)
    other = _tags_client(server)
    try:
        await client.list_tags(context_id=_TAGS_CTX, with_tags=["a"])
        await client.list_tags(context_id=_TAGS_CTX, with_tags=["b"])
        assert len(server.tool_calls()) == 1
        # The cache belongs to the client, not the process.
        await other.list_tags(context_id=_TAGS_CTX, with_tags=["a"])
        assert len(server.tool_calls()) == 2
    finally:
        await client.close()
        await other.close()


@pytest.mark.asyncio
async def test_list_tags_with_tags_keys_the_name_cache_on_the_canonical_id():
    """A caller may spell the UUID in upper case; both routes answer in lower case."""
    ctx = "0a1b2c3d-4e5f-4a6b-8c7d-9e0f1a2b3c4d"
    server = _TagsServer(
        rest_body=_rest_tags_body(context_id=ctx),
        mcp_result=_list_tags_envelope(context_id=ctx, context_name="demo"),
    )
    client = _tags_client(server)
    try:
        await client.list_tags(context_id=ctx.upper())
        result = await client.list_tags(context_id=ctx.upper(), with_tags=["a"])
    finally:
        await client.close()

    assert result.context_id == ctx
    assert result.context_name == "demo"
    assert len(server.tool_calls()) == 1


@pytest.mark.asyncio
async def test_list_tags_with_tags_uses_a_context_name_the_route_sends():
    """memory-cloud v0.77.0 (#1669) sends context_name on the route: no lookup is needed."""
    server = _TagsServer(rest_body=_rest_tags_body(context_name="from-rest"))
    client = _tags_client(server)
    try:
        result = await client.list_tags(context_id=_TAGS_CTX, with_tags=["a"])
    finally:
        await client.close()

    assert result.context_name == "from-rest"
    assert server.tool_calls() == []


# memory-cloud rewraps every HTTPException / RequestValidationError into
# {error, message, details} (api/main.py, #992); FastAPI's own {"detail": ...}
# is kept as the shape of other deployments and older servers.
_VALIDATION_ERRORS = [
    {"loc": ["path", "context_id"], "msg": "Input should be a valid UUID", "type": "uuid_parsing"}
]


@pytest.mark.parametrize(
    "body",
    [
        {"error": "HTTP-404", "message": "Context not found", "details": {}},
        {"detail": "Context not found"},
    ],
)
@pytest.mark.asyncio
async def test_list_tags_with_tags_raises_not_found_on_a_rest_404_and_looks_no_name_up(body):
    server = _TagsServer(rest_status=404, rest_body=body)
    client = _tags_client(server)
    try:
        with pytest.raises(KaguraNotFoundError) as exc:
            await client.list_tags(context_id=_TAGS_CTX, with_tags=["a"])
    finally:
        await client.close()

    assert str(exc.value) == "list_tags: Context not found"
    # Chained explicitly, like every other status the REST helper maps.
    assert isinstance(exc.value.__cause__, httpx.HTTPStatusError)
    assert len(server.requests) == 1


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            {"error": "HTTP-422", "message": "with_tags accepts at most 50 tags.", "details": {}},
            "list_tags failed (invalid_argument): with_tags accepts at most 50 tags.",
        ),
        (
            {
                "error": "VAL-001",
                "message": "Request validation failed",
                "details": {"errors": _VALIDATION_ERRORS},
            },
            "list_tags failed (invalid_argument): Request validation failed: "
            "path.context_id: Input should be a valid UUID",
        ),
        (
            {"detail": "with_tags accepts at most 50 tags."},
            "list_tags failed (invalid_argument): with_tags accepts at most 50 tags.",
        ),
        (
            {"detail": _VALIDATION_ERRORS},
            "list_tags failed (invalid_argument): path.context_id: Input should be a valid UUID",
        ),
    ],
)
@pytest.mark.asyncio
async def test_list_tags_with_tags_raises_kagura_error_on_a_rest_422(body, message):
    """A refused value is a KaguraError, as on MCP — not a KaguraConnectionError."""
    server = _TagsServer(rest_status=422, rest_body=body)
    client = _tags_client(server)
    try:
        with pytest.raises(KaguraError) as exc:
            await client.list_tags(context_id=_TAGS_CTX, with_tags=["a"])
    finally:
        await client.close()

    assert not isinstance(exc.value, KaguraConnectionError)
    assert str(exc.value) == message
    assert isinstance(exc.value.__cause__, httpx.HTTPStatusError)


@pytest.mark.parametrize(
    ("status", "error_class", "message"),
    [
        (404, KaguraNotFoundError, "list_tags: HTTP 404"),
        (422, KaguraError, "list_tags failed (invalid_argument): HTTP 422"),
    ],
)
@pytest.mark.asyncio
async def test_list_tags_with_tags_names_the_status_when_the_body_has_no_detail(
    status, error_class, message
):
    server = _TagsServer(rest_status=status, rest_body={"unexpected": "shape"})
    client = _tags_client(server)
    try:
        with pytest.raises(error_class) as exc:
            await client.list_tags(context_id=_TAGS_CTX, with_tags=["a"])
    finally:
        await client.close()
    assert str(exc.value) == message


@pytest.mark.parametrize(
    ("status", "error_class"),
    [(401, KaguraAuthError), (429, KaguraRateLimitError), (500, KaguraConnectionError)],
)
@pytest.mark.asyncio
async def test_list_tags_with_tags_keeps_the_standard_status_mapping(status, error_class):
    server = _TagsServer(rest_status=status, rest_body={"detail": "no"})
    client = _tags_client(server)
    try:
        with pytest.raises(error_class):
            await client.list_tags(context_id=_TAGS_CTX, with_tags=["a"])
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_list_tags_with_tags_raises_not_found_when_the_name_lookup_cannot_see_it():
    server = _TagsServer(
        mcp_result={
            "status": "error",
            "error": "context_not_found",
            "message": "Context not found or you don't have access to it.",
        }
    )
    client = _tags_client(server)
    try:
        with pytest.raises(KaguraNotFoundError, match="list_tags"):
            await client.list_tags(context_id=_TAGS_CTX, with_tags=["a"])
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_list_tags_with_tags_raises_response_error_when_the_lookup_has_no_name():
    from kagura_memory import KaguraResponseError

    server = _TagsServer(
        mcp_result={"status": "success", "context_id": _TAGS_CTX, "tags": [], "total": 0}
    )
    client = _tags_client(server)
    try:
        with pytest.raises(KaguraResponseError, match="context_name") as exc:
            await client.list_tags(context_id=_TAGS_CTX, with_tags=["a"])
    finally:
        await client.close()
    assert exc.value.operation == "list_tags"


@pytest.mark.parametrize(
    "body",
    [
        {"context_id": _TAGS_CTX, "total": 0},
        {"context_id": _TAGS_CTX, "tags": []},
        {"tags": [], "total": 0},
        {"context_id": _TAGS_CTX, "tags": [{"tag": "a"}], "total": 1},
        ["not", "an", "object"],
    ],
)
@pytest.mark.asyncio
async def test_list_tags_with_tags_raises_response_error_on_a_malformed_rest_body(body):
    """Drift is a KaguraResponseError naming list_tags, reported before any name lookup."""
    from kagura_memory import KaguraResponseError

    server = _TagsServer(rest_body=body)
    client = _tags_client(server)
    try:
        with pytest.raises(KaguraResponseError) as exc:
            await client.list_tags(context_id=_TAGS_CTX, with_tags=["a"])
    finally:
        await client.close()

    assert exc.value.operation == "list_tags"
    # Names the (unexported) model after the server's, never a private name.
    assert str(exc.value).startswith(
        "list_tags: unexpected server response for ContextTagsResponse ("
    )
    assert server.tool_calls() == []


@pytest.mark.parametrize(
    ("with_tags", "match"),
    [
        ([f"t{i}" for i in range(51)], r"with_tags accepts at most 50 tags, got 51"),
        (["ok", "x" * 201], r"each with_tags value must be at most 200 characters, got 201"),
    ],
)
@pytest.mark.asyncio
async def test_list_tags_rejects_an_oversized_drill_down_before_any_request(with_tags, match):
    server = _TagsServer()
    client = _tags_client(server)
    try:
        with pytest.raises(ValueError, match=match):
            await client.list_tags(context_id=_TAGS_CTX, with_tags=with_tags)
    finally:
        await client.close()
    assert server.requests == []


@pytest.mark.asyncio
async def test_list_tags_rejects_a_bare_string_with_tags():
    """A str would be iterated into one-character tags — a silently wrong drill-down."""
    server = _TagsServer()
    client = _tags_client(server)
    try:
        with pytest.raises(TypeError, match="with_tags must be a list"):
            await client.list_tags(context_id=_TAGS_CTX, with_tags="client:acme")  # type: ignore[arg-type]
    finally:
        await client.close()
    assert server.requests == []


@pytest.mark.parametrize(
    ("with_tags", "type_name"),
    [([1], "int"), (["ok", None], "NoneType"), (("ok", b"raw"), "bytes")],
)
@pytest.mark.asyncio
async def test_list_tags_rejects_a_non_str_with_tags_item(with_tags, type_name):
    """A TypeError naming the item's type, not an AttributeError from .strip()."""
    server = _TagsServer()
    client = _tags_client(server)
    try:
        with pytest.raises(TypeError, match=f"with_tags items must be str, got {type_name}$"):
            await client.list_tags(context_id=_TAGS_CTX, with_tags=with_tags)
    finally:
        await client.close()
    assert server.requests == []


@pytest.mark.asyncio
async def test_list_tags_accepts_any_iterable_of_str_with_tags():
    """A tuple or a one-shot generator is read once, then normalized like a list."""
    server = _TagsServer()
    client = _tags_client(server)
    try:
        await client.list_tags(context_id=_TAGS_CTX, with_tags=(t for t in [" a ", "b"]))  # type: ignore[arg-type]
    finally:
        await client.close()
    assert server.gets()[0].url.params.get_list("with_tags") == ["a", "b"]


@pytest.mark.asyncio
async def test_list_tags_drops_blank_values_before_the_50_tag_cap():
    server = _TagsServer()
    client = _tags_client(server)
    try:
        await client.list_tags(
            context_id=_TAGS_CTX, with_tags=[f"t{i}" for i in range(50)] + ["  "] * 10
        )
    finally:
        await client.close()
    assert len(server.gets()[0].url.params.get_list("with_tags")) == 50


@pytest.mark.asyncio
async def test_list_tags_with_tags_counts_the_200_cap_in_characters():
    """The server's len() counts code points, as Python's does — not UTF-8 bytes."""
    server = _TagsServer()
    client = _tags_client(server)
    try:
        await client.list_tags(context_id=_TAGS_CTX, with_tags=["🏷" * 200])
    finally:
        await client.close()
    assert server.gets()[0].url.params["with_tags"] == "🏷" * 200


@pytest.mark.asyncio
async def test_list_tags_with_tags_uses_the_rest_base_of_a_workspace_scoped_mcp_url():
    server = _TagsServer()
    client = _tags_client(server, mcp_url="https://test.com/mcp/w/ws-1?profile=core")
    try:
        await client.list_tags(context_id=_TAGS_CTX, with_tags=["a"])
    finally:
        await client.close()

    assert str(server.requests[0].url).startswith(f"https://test.com{_TAGS_PATH}?")
    # The name lookup is an MCP call and keeps the MCP URL as given.
    assert str(server.requests[1].url) == "https://test.com/mcp/w/ws-1?profile=core"


@pytest.mark.asyncio
async def test_list_tags_with_tags_encodes_the_context_id_into_one_path_segment():
    server = _TagsServer(rest_status=404, rest_body={"detail": "Not Found"})
    client = _tags_client(server)
    try:
        with pytest.raises(KaguraNotFoundError):
            await client.list_tags(context_id="a/../b", with_tags=["a"])
    finally:
        await client.close()
    assert server.requests[0].url.raw_path.startswith(b"/api/v1/contexts/a%2F..%2Fb/tags?")


# ============================================================================
# get_agent_bootstrap (#231, server v0.49.0+)
# ============================================================================

_BOOTSTRAP_ENVELOPE = bootstrap_envelope_dict()


@pytest.mark.asyncio
async def test_get_agent_bootstrap_minimal():
    """get_agent_bootstrap() sends agent_id only and parses the envelope."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = _BOOTSTRAP_ENVELOPE
        result = await client.get_agent_bootstrap("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
        name, args = mock.call_args[0][0], mock.call_args[0][1]
        assert name == "get_agent_bootstrap"
        assert args == {"agent_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"}

    assert isinstance(result, AgentBootstrapResponse)
    assert result.degraded is False
    assert result.agent.name == "ci-agent"
    assert result.agent.binding is not None and result.agent.binding.is_default is True
    # The context block reuses ContextDetail (byte-compatible with get_context_info).
    assert result.context is not None and result.context.name == "dev"
    # Component payloads stay dicts — shapes belong to the standalone tools.
    assert result.components["pinned"]["status"] == "ok"
    assert result.components["recall"] == {"status": "skipped", "reason": "no_query"}
    assert result.correlation is not None and result.correlation.session_id == "run-42"
    assert result.generated_at is not None

    await client.close()


@pytest.mark.asyncio
async def test_get_agent_bootstrap_full_args():
    """get_agent_bootstrap() forwards every optional argument."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = _BOOTSTRAP_ENVELOPE
        await client.get_agent_bootstrap(
            "agent-uuid",
            context_id="ctx-uuid",
            session_id="run-42",
            query="current task",
            recall_k=7,
            pinned_cap=50,
            upcoming_until="2026-08-01T00:00:00",
            include=["pinned", "recall"],
        )
        args = mock.call_args[0][1]
        assert args == {
            "agent_id": "agent-uuid",
            "context_id": "ctx-uuid",
            "session_id": "run-42",
            "query": "current task",
            "recall_k": 7,
            "pinned_cap": 50,
            "upcoming_until": "2026-08-01T00:00:00",
            "include": ["pinned", "recall"],
        }

    await client.close()


@pytest.mark.asyncio
async def test_get_agent_bootstrap_optional_args_not_sent_when_none():
    """Omitted optionals must not appear in the tool arguments (server defaults)."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = _BOOTSTRAP_ENVELOPE
        await client.get_agent_bootstrap("agent-uuid")
        args = mock.call_args[0][1]
        for key in (
            "context_id",
            "session_id",
            "query",
            "recall_k",
            "pinned_cap",
            "upcoming_until",
            "include",
        ):
            assert key not in args

    await client.close()


@pytest.mark.asyncio
async def test_get_agent_bootstrap_degraded_component():
    """A failed component parses fail-soft: degraded=True, others intact."""
    client = _make_initialized_client()
    envelope = {
        **_BOOTSTRAP_ENVELOPE,
        "degraded": True,
        "components": {
            "pinned": {"status": "ok", "memories": [], "total_available": 0},
            "state": {"status": "error", "error": "component_error"},
        },
    }

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = envelope
        result = await client.get_agent_bootstrap("agent-uuid")
        assert result.degraded is True
        assert result.components["state"] == {"status": "error", "error": "component_error"}
        assert result.components["pinned"]["status"] == "ok"

    await client.close()


@pytest.mark.asyncio
async def test_get_agent_bootstrap_agent_not_found_raises():
    """agent_not_found domain errors raise KaguraNotFoundError (uniform 404)."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {
            "status": "error",
            "error": "agent_not_found",
            "message": "Agent not found.",
        }
        with pytest.raises(KaguraNotFoundError, match="Agent not found"):
            await client.get_agent_bootstrap("agent-uuid")

    await client.close()


@pytest.mark.asyncio
async def test_get_agent_bootstrap_invalid_arguments_raises():
    """invalid_arguments domain errors raise the generic KaguraError."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {
            "status": "error",
            "error": "invalid_arguments",
            "message": "'session_id' allows only [A-Za-z0-9._-].",
        }
        with pytest.raises(KaguraError, match="invalid_arguments"):
            await client.get_agent_bootstrap("agent-uuid", session_id="bad session")

    await client.close()


# ============================================================================
# Agent registry + bindings (#235, server v0.49.0+)
# ============================================================================


@pytest.mark.asyncio
async def test_register_agent_minimal():
    """register_agent() sends name only and parses the agent row."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"status": "success", "agent": agent_dict()}
        agent = await client.register_agent("ci-agent")
        name, args = mock.call_args[0][0], mock.call_args[0][1]
        assert name == "register_agent"
        assert args == {"name": "ci-agent"}

    assert isinstance(agent, Agent)
    assert agent.name == "ci-agent"
    assert agent.status == "active" and agent.enforcement_mode == "enforce"

    await client.close()


@pytest.mark.asyncio
async def test_register_agent_full_args():
    """register_agent() forwards every optional metadata field."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"status": "success", "agent": agent_dict()}
        await client.register_agent(
            "ci-agent",
            description="CI runner",
            framework="claude-code",
            environment="production",
            version="1.2.3",
        )
        args = mock.call_args[0][1]
        assert args == {
            "name": "ci-agent",
            "description": "CI runner",
            "framework": "claude-code",
            "environment": "production",
            "version": "1.2.3",
        }

    await client.close()


@pytest.mark.asyncio
async def test_get_agent_parses_row():
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"status": "success", "agent": agent_dict()}
        agent = await client.get_agent("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
        assert mock.call_args[0][0] == "get_agent"
        assert mock.call_args[0][1] == {"agent_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"}

    assert agent.workspace_id == "11111111-2222-3333-4444-555555555555"

    await client.close()


@pytest.mark.asyncio
async def test_list_agents_parses_rows():
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {
            "status": "success",
            "agents": [agent_dict(), agent_dict(name="other")],
            "count": 2,
        }
        agents = await client.list_agents()
        assert mock.call_args[0][0] == "list_agents"
        assert mock.call_args[0][1] == {}

    assert [a.name for a in agents] == ["ci-agent", "other"]

    await client.close()


@pytest.mark.asyncio
async def test_update_agent_full_args():
    """update_agent() forwards lifecycle + metadata fields; omits None."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {
            "status": "success",
            "agent": agent_dict(status="suspended"),
            "changed": ["status"],
        }
        agent = await client.update_agent(
            "agent-uuid", status="suspended", enforcement_mode="shadow"
        )
        args = mock.call_args[0][1]
        assert args == {
            "agent_id": "agent-uuid",
            "status": "suspended",
            "enforcement_mode": "shadow",
        }
        for key in ("name", "description", "framework", "environment", "version"):
            assert key not in args

    assert agent.status == "suspended"

    await client.close()


@pytest.mark.asyncio
async def test_update_agent_rejects_empty_update():
    """No-op update_agent() fails fast locally instead of a server 400 (Copilot #239)."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        with pytest.raises(ValueError, match="at least one field"):
            await client.update_agent("agent-uuid")
        mock.assert_not_called()

    await client.close()


@pytest.mark.asyncio
async def test_update_agent_binding_rejects_empty_update():
    """No-op update_agent_binding() fails fast locally (Copilot #239)."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        with pytest.raises(ValueError, match="at least one of"):
            await client.update_agent_binding("agent-uuid", "binding-uuid")
        mock.assert_not_called()

    await client.close()


@pytest.mark.asyncio
async def test_delete_agent_returns_deleted():
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"status": "success", "deleted": True, "agent_id": "agent-uuid"}
        assert await client.delete_agent("agent-uuid") is True
        assert mock.call_args[0][0] == "delete_agent"

    await client.close()


@pytest.mark.asyncio
async def test_bind_agent_context_defaults_omitted():
    """bind_agent_context() sends only agent_id+context_id when defaults apply."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"status": "success", "binding": agent_binding_dict()}
        binding = await client.bind_agent_context("agent-uuid", "ctx-uuid")
        name, args = mock.call_args[0][0], mock.call_args[0][1]
        assert name == "bind_agent_context"
        assert args == {"agent_id": "agent-uuid", "context_id": "ctx-uuid"}

    assert isinstance(binding, AgentBinding)
    assert binding.write_policy == "deny" and binding.can_read is True

    await client.close()


@pytest.mark.asyncio
async def test_bind_agent_context_full_args():
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {
            "status": "success",
            "binding": agent_binding_dict(write_policy="direct", is_default=True),
        }
        binding = await client.bind_agent_context(
            "agent-uuid", "ctx-uuid", can_read=True, write_policy="direct", is_default=True
        )
        args = mock.call_args[0][1]
        assert args == {
            "agent_id": "agent-uuid",
            "context_id": "ctx-uuid",
            "can_read": True,
            "write_policy": "direct",
            "is_default": True,
        }

    assert binding.is_default is True

    await client.close()


@pytest.mark.asyncio
async def test_list_agent_bindings_parses_rows():
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {
            "status": "success",
            "bindings": [agent_binding_dict()],
            "count": 1,
        }
        bindings = await client.list_agent_bindings("agent-uuid")
        assert mock.call_args[0][1] == {"agent_id": "agent-uuid"}

    assert len(bindings) == 1
    assert bindings[0].context_id == "11111111-2222-3333-4444-555555555555"

    await client.close()


@pytest.mark.asyncio
async def test_update_agent_binding_sends_changes_only():
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {
            "status": "success",
            "binding": agent_binding_dict(can_read=False),
            "changed": ["can_read"],
        }
        binding = await client.update_agent_binding("agent-uuid", "binding-uuid", can_read=False)
        args = mock.call_args[0][1]
        assert args == {
            "agent_id": "agent-uuid",
            "binding_id": "binding-uuid",
            "can_read": False,
        }

    assert binding.can_read is False

    await client.close()


@pytest.mark.asyncio
async def test_unbind_agent_context_returns_deleted():
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"status": "success", "deleted": True, "binding_id": "binding-uuid"}
        assert await client.unbind_agent_context("agent-uuid", "binding-uuid") is True
        args = mock.call_args[0][1]
        assert args == {"agent_id": "agent-uuid", "binding_id": "binding-uuid"}

    await client.close()


@pytest.mark.asyncio
async def test_binding_not_found_raises_not_found():
    """binding_not_found domain errors map to KaguraNotFoundError."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {
            "status": "error",
            "error": "binding_not_found",
            "message": "Binding not found.",
        }
        with pytest.raises(KaguraNotFoundError, match="Binding not found"):
            await client.unbind_agent_context("agent-uuid", "binding-uuid")

    await client.close()


@pytest.mark.asyncio
async def test_agent_name_conflict_raises_kagura_error():
    """agent_name_conflict stays a generic KaguraError (not not-found)."""
    client = _make_initialized_client()

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {
            "status": "error",
            "error": "agent_name_conflict",
            "message": "An agent with this name already exists.",
        }
        with pytest.raises(KaguraError, match="agent_name_conflict"):
            await client.register_agent("ci-agent")

    await client.close()
