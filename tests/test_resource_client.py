"""Tests for ResourceClient."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from kagura_memory import (
    KaguraAuthError,
    KaguraConnectionError,
    KaguraNotFoundError,
    KaguraQuotaError,
    ResourceClient,
    ResourceEventRequest,
)

# ============================================================================
# HTTPS enforcement (C-3)
# ============================================================================


def test_rejects_http_url():
    """HTTP URLs (non-localhost) should raise ValueError."""
    with pytest.raises(ValueError, match="must use HTTPS"):
        ResourceClient(api_key="test", base_url="http://evil.com")


def test_allows_https_url():
    """HTTPS URLs should be accepted."""
    client = ResourceClient(api_key="test", base_url="https://memory.kagura-ai.com")
    assert client.base_url == "https://memory.kagura-ai.com"


def test_allows_localhost_http():
    """HTTP localhost should be allowed for development."""
    client = ResourceClient(api_key="test", base_url="http://localhost:8080")
    assert client.base_url == "http://localhost:8080"


def test_allows_127_http():
    """HTTP 127.0.0.1 should be allowed for development."""
    client = ResourceClient(api_key="test", base_url="http://127.0.0.1:8080")
    assert client.base_url == "http://127.0.0.1:8080"


# ============================================================================
# API key not stored (C-1)
# ============================================================================


def test_api_key_not_on_instance():
    """API key should not be accessible as instance attribute."""
    client = ResourceClient(api_key="secret-key", base_url="https://test.com")
    assert not hasattr(client, "api_key")


# ============================================================================
# from_mcp_url
# ============================================================================


def test_from_mcp_url_strips_mcp_suffix():
    """from_mcp_url should strip /mcp suffix."""
    client = ResourceClient.from_mcp_url(api_key="test", mcp_url="https://memory.kagura-ai.com/mcp")
    assert client.base_url == "https://memory.kagura-ai.com"


def test_from_mcp_url_strips_trailing_slash():
    """from_mcp_url should handle trailing slash."""
    client = ResourceClient.from_mcp_url(
        api_key="test", mcp_url="https://memory.kagura-ai.com/mcp/"
    )
    assert client.base_url == "https://memory.kagura-ai.com"


def test_from_mcp_url_no_mcp_suffix():
    """from_mcp_url should work even without /mcp suffix."""
    client = ResourceClient.from_mcp_url(api_key="test", mcp_url="https://custom.server.com")
    assert client.base_url == "https://custom.server.com"


def test_from_mcp_url_localhost():
    """from_mcp_url should allow localhost."""
    client = ResourceClient.from_mcp_url(api_key="test", mcp_url="http://localhost:8080/mcp")
    assert client.base_url == "http://localhost:8080"


def test_from_mcp_url_with_workspace_id():
    """from_mcp_url should strip /mcp/w/{workspace_id} path."""
    client = ResourceClient.from_mcp_url(
        api_key="test",
        mcp_url="http://localhost:8080/mcp/w/81dfe87f-29db-4569-b9a0-3cd308827e1e",
    )
    assert client.base_url == "http://localhost:8080"


# ============================================================================
# Token CRUD
# ============================================================================


def _mock_response(status_code: int = 200, json_data: dict | None = None) -> MagicMock:
    """Create a mock httpx.Response."""
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = json_data or {}
    response.raise_for_status = MagicMock()
    response.headers = {}
    return response


@pytest.mark.asyncio
async def test_create_token():
    """Test token creation returns ResourceTokenCreateResponse."""
    client = ResourceClient(api_key="test", base_url="https://test.com")

    response_data = {
        "id": 1,
        "resource_id": "products",
        "description": "Test token",
        "quota_events_per_hour": 1000,
        "created_by": "user1",
        "created_at": "2026-03-29T00:00:00Z",
        "last_used_at": None,
        "is_active": True,
        "status": "active",
        "token": "kagura_resource_abc123",
    }
    mock_resp = _mock_response(201, response_data)

    with patch.object(client._client, "request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = mock_resp

        result = await client.create_token(resource_id="products", description="Test token")

        assert result.id == 1
        assert result.resource_id == "products"
        assert result.token == "kagura_resource_abc123"
        mock_req.assert_called_once()
        call_args = mock_req.call_args
        assert call_args[0][0] == "POST"
        assert "/api/v1/resource-tokens" in call_args[0][1]

    await client.close()


@pytest.mark.asyncio
async def test_list_tokens():
    """Test token listing returns paginated response."""
    client = ResourceClient(api_key="test", base_url="https://test.com")

    response_data = {
        "tokens": [
            {
                "id": 1,
                "resource_id": "products",
                "quota_events_per_hour": 1000,
                "created_at": "2026-03-29T00:00:00Z",
                "is_active": True,
                "status": "active",
            }
        ],
        "total": 1,
        "limit": 50,
        "offset": 0,
    }
    mock_resp = _mock_response(200, response_data)

    with patch.object(client._client, "request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = mock_resp

        result = await client.list_tokens(resource_id="products")

        assert result.total == 1
        assert len(result.tokens) == 1
        assert result.tokens[0].resource_id == "products"

    await client.close()


@pytest.mark.asyncio
async def test_update_token():
    """Test token update."""
    client = ResourceClient(api_key="test", base_url="https://test.com")

    response_data = {
        "id": 1,
        "resource_id": "products",
        "description": "Updated",
        "quota_events_per_hour": 2000,
        "created_at": "2026-03-29T00:00:00Z",
        "is_active": True,
        "status": "active",
    }
    mock_resp = _mock_response(200, response_data)

    with patch.object(client._client, "request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = mock_resp

        result = await client.update_token(token_id=1, quota_events_per_hour=2000)

        assert result.quota_events_per_hour == 2000
        call_args = mock_req.call_args
        assert call_args[0][0] == "PATCH"
        assert "/api/v1/resource-tokens/1" in call_args[0][1]

    await client.close()


@pytest.mark.asyncio
async def test_revoke_token():
    """Test token revocation sends DELETE."""
    client = ResourceClient(api_key="test", base_url="https://test.com")

    mock_resp = _mock_response(204)

    with patch.object(client._client, "request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = mock_resp

        await client.revoke_token(token_id=42)

        call_args = mock_req.call_args
        assert call_args[0][0] == "DELETE"
        assert "/api/v1/resource-tokens/42" in call_args[0][1]

    await client.close()


# ============================================================================
# Event Ingestion
# ============================================================================


@pytest.mark.asyncio
async def test_ingest_event_sends_resource_api_key():
    """Ingest event should use X-Resource-API-Key header."""
    client = ResourceClient(api_key="test", base_url="https://test.com")

    response_data = {"status": "success", "event_id": 123, "queued": True}
    mock_resp = _mock_response(201, response_data)

    with patch.object(client._client, "request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = mock_resp

        event = ResourceEventRequest(
            op="upsert",
            doc_id="SKU-001",
            version=1,
            payload={"name": "Widget", "price": 9.99},
        )
        result = await client.ingest_event("products", "resource_key_123", event)

        assert result.event_id == 123
        call_kwargs = mock_req.call_args[1]
        assert call_kwargs["headers"] == {"X-Resource-API-Key": "resource_key_123"}

    await client.close()


@pytest.mark.asyncio
async def test_resource_api_key_not_stored():
    """resource_api_key should not be stored on the client instance."""
    client = ResourceClient(api_key="test", base_url="https://test.com")

    response_data = {"status": "success", "event_id": 1, "queued": True}
    mock_resp = _mock_response(201, response_data)

    with patch.object(client._client, "request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = mock_resp

        event = ResourceEventRequest(op="upsert", doc_id="doc1", version=1, payload={"x": 1})
        await client.ingest_event("res", "secret_resource_key", event)

    assert not hasattr(client, "resource_api_key")
    # Verify it's not in the default client headers
    assert "X-Resource-API-Key" not in client._client.headers

    await client.close()


@pytest.mark.asyncio
async def test_ingest_events_batch():
    """Batch ingestion should POST to events/batch endpoint."""
    client = ResourceClient(api_key="test", base_url="https://test.com")

    response_data = {
        "status": "success",
        "created_count": 2,
        "failed_count": 0,
        "event_ids": [1, 2],
        "errors": [],
    }
    mock_resp = _mock_response(201, response_data)

    with patch.object(client._client, "request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = mock_resp

        events = [
            ResourceEventRequest(op="upsert", doc_id="doc1", version=1, payload={"a": 1}),
            ResourceEventRequest(op="upsert", doc_id="doc2", version=1, payload={"b": 2}),
        ]
        result = await client.ingest_events("products", "key123", events)

        assert result.created_count == 2
        assert result.failed_count == 0
        call_args = mock_req.call_args
        assert "/events/batch" in call_args[0][1]

    await client.close()


# ============================================================================
# Resource Impact / Stats
# ============================================================================


@pytest.mark.asyncio
async def test_get_resource_impact():
    """get_resource_impact should GET /api/v1/resources/{id}/impact."""
    client = ResourceClient(api_key="test", base_url="https://test.com")

    response_data = {
        "resource_id": "products",
        "token_count": 3,
        "memory_count": 150,
        "current_schema_version": 2,
    }
    mock_resp = _mock_response(200, response_data)

    with patch.object(client._client, "request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = mock_resp

        result = await client.get_resource_impact("products")

        assert result.resource_id == "products"
        assert result.token_count == 3
        assert result.memory_count == 150
        assert result.current_schema_version == 2
        call_args = mock_req.call_args
        assert call_args[0][0] == "GET"
        assert "/api/v1/resources/products/impact" in call_args[0][1]

    await client.close()


# ============================================================================
# Resource Schema
# ============================================================================


@pytest.mark.asyncio
async def test_get_resource_schema():
    """get_resource_schema should GET /api/v1/resources/{id}/schema."""
    client = ResourceClient(api_key="test", base_url="https://test.com")

    response_data = {
        "resource_id": "products",
        "schema_version": 2,
        "field_definitions": [
            {
                "name": "product_name",
                "type": "text",
                "description": "Product name",
                "required": True,
            },
            {
                "name": "price",
                "type": "number",
                "description": "Price",
                "unit": "JPY",
            },
        ],
        "created_at": "2026-03-29T10:00:00",
    }
    mock_resp = _mock_response(200, response_data)

    with patch.object(client._client, "request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = mock_resp

        result = await client.get_resource_schema("products")

        assert result is not None
        assert result.resource_id == "products"
        assert result.schema_version == 2
        assert len(result.field_definitions) == 2
        assert result.field_definitions[0].name == "product_name"
        assert result.field_definitions[0].required is True
        assert result.field_definitions[1].unit == "JPY"
        call_args = mock_req.call_args
        assert call_args[0][0] == "GET"
        assert "/api/v1/resources/products/schema" in call_args[0][1]

    await client.close()


@pytest.mark.asyncio
async def test_get_resource_schema_with_version():
    """get_resource_schema should pass schema_version as query param."""
    client = ResourceClient(api_key="test", base_url="https://test.com")

    response_data = {
        "resource_id": "products",
        "schema_version": 1,
        "field_definitions": [],
        "created_at": "2026-03-29T10:00:00",
    }
    mock_resp = _mock_response(200, response_data)

    with patch.object(client._client, "request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = mock_resp

        result = await client.get_resource_schema("products", schema_version=1)

        assert result is not None
        assert result.schema_version == 1
        call_kwargs = mock_req.call_args[1]
        assert call_kwargs["params"] == {"schema_version": 1}

    await client.close()


@pytest.mark.asyncio
async def test_get_resource_schema_returns_none_on_404():
    """get_resource_schema should return None when schema is not registered."""
    client = ResourceClient(api_key="test", base_url="https://test.com")

    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.json.return_value = {"detail": "Not found"}
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "404", request=MagicMock(), response=mock_response
    )

    with patch.object(client._client, "request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = mock_response

        result = await client.get_resource_schema("no-schema")

        assert result is None

    await client.close()


# ============================================================================
# Error handling
# ============================================================================


@pytest.mark.asyncio
async def test_not_found_error_on_404():
    """404 response should raise KaguraNotFoundError."""
    client = ResourceClient(api_key="test", base_url="https://test.com")

    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.json.return_value = {"detail": "Resource not found"}
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "404", request=MagicMock(), response=mock_response
    )

    with patch.object(client._client, "request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = mock_response

        with pytest.raises(KaguraNotFoundError, match="Resource not found"):
            await client.list_tokens()

    await client.close()


@pytest.mark.asyncio
async def test_not_found_error_non_json_body():
    """404 with non-JSON body should raise KaguraNotFoundError with default message."""
    client = ResourceClient(api_key="test", base_url="https://test.com")

    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.json.side_effect = ValueError("not json")
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "404", request=MagicMock(), response=mock_response
    )

    with patch.object(client._client, "request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = mock_response

        with pytest.raises(KaguraNotFoundError, match="Not found"):
            await client.list_tokens()

    await client.close()


@pytest.mark.asyncio
async def test_auth_error_on_401():
    """401 response should raise KaguraAuthError."""
    client = ResourceClient(api_key="bad", base_url="https://test.com")

    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "401", request=MagicMock(), response=mock_response
    )

    with patch.object(client._client, "request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = mock_response

        with pytest.raises(KaguraAuthError, match="Authentication failed"):
            await client.list_tokens()

    await client.close()


@pytest.mark.asyncio
async def test_quota_error_on_429():
    """429 response should raise KaguraQuotaError."""
    client = ResourceClient(api_key="test", base_url="https://test.com")

    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.headers = {"Retry-After": "60"}
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "429", request=MagicMock(), response=mock_response
    )

    with patch.object(client._client, "request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = mock_response

        with pytest.raises(KaguraQuotaError, match="Quota exceeded") as exc_info:
            event = ResourceEventRequest(op="upsert", doc_id="d", version=1, payload={"x": 1})
            await client.ingest_event("res", "key", event)

        assert exc_info.value.retry_after == 60

    await client.close()


@pytest.mark.asyncio
async def test_connection_error_on_network_failure():
    """Network failure should raise KaguraConnectionError."""
    client = ResourceClient(api_key="test", base_url="https://test.com")

    with patch.object(client._client, "request", new_callable=AsyncMock) as mock_req:
        mock_req.side_effect = httpx.ConnectError("Connection refused")

        with pytest.raises(KaguraConnectionError, match="Connection failed"):
            await client.list_tokens()

    await client.close()


@pytest.mark.asyncio
async def test_connection_error_on_409():
    """409 Conflict should raise KaguraConnectionError with detail."""
    client = ResourceClient(api_key="test", base_url="https://test.com")

    mock_response = MagicMock()
    mock_response.status_code = 409
    mock_response.json.return_value = {"detail": "Duplicate version"}
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "409", request=MagicMock(), response=mock_response
    )

    with patch.object(client._client, "request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = mock_response

        with pytest.raises(KaguraConnectionError, match="Duplicate version"):
            event = ResourceEventRequest(op="upsert", doc_id="d", version=1, payload={"x": 1})
            await client.ingest_event("res", "key", event)

    await client.close()


@pytest.mark.asyncio
async def test_connection_error_non_json_body():
    """Non-JSON error response should still raise KaguraConnectionError."""
    client = ResourceClient(api_key="test", base_url="https://test.com")

    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.json.side_effect = ValueError("not json")
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "500", request=MagicMock(), response=mock_response
    )

    with patch.object(client._client, "request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = mock_response

        with pytest.raises(KaguraConnectionError, match="HTTP 500"):
            await client.list_tokens()

    await client.close()


# ============================================================================
# Context manager
# ============================================================================


# ============================================================================
# setup_resource
# ============================================================================


def _make_mcp_mock(server_response: dict) -> AsyncMock:
    """Emulate ``async with KaguraClient(...) as mcp`` with a stubbed setup_resource."""
    mock_mcp = AsyncMock()
    mock_mcp.setup_resource.return_value = server_response
    mock_mcp.__aenter__ = AsyncMock(return_value=mock_mcp)
    mock_mcp.__aexit__ = AsyncMock(return_value=None)
    return mock_mcp


@pytest.mark.asyncio
async def test_setup_resource():
    """All caller-provided args propagate to the atomic MCP tool; response is type-validated."""
    client = ResourceClient.from_mcp_url(api_key="test", mcp_url="http://localhost:8080/mcp")

    server_response = {
        "status": "success",
        "message": "Resource 'my-res' set up successfully.",
        "context_id": "ctx-uuid",
        "context_name": "my-res",
        "resource_id": "my-res",
        "token": "kagura_resource_abc",
        "token_id": 42,
        "warning": "Save this token — it will not be shown again.",
    }
    mock_mcp = _make_mcp_mock(server_response)

    with patch("kagura_memory.resource_client.KaguraClient", return_value=mock_mcp):
        result = await client.setup_resource(
            resource_id="my-res",
            context_name="my-res",
            summary="Test setup",
            description="Setup test token",
            quota_events_per_hour=2000,
        )

    assert result.token == "kagura_resource_abc"
    assert result.token_id == 42
    assert result.context_id == "ctx-uuid"
    assert result.resource_id == "my-res"
    assert result.warning == "Save this token — it will not be shown again."

    mock_mcp.setup_resource.assert_called_once_with(
        resource_id="my-res",
        name="my-res",
        summary="Test setup",
        description="Setup test token",
        quota_events_per_hour=2000,
    )

    await client.close()


@pytest.mark.asyncio
async def test_setup_resource_passes_none_context_name():
    """Omitting context_name must surface as name=None so the server applies its own default."""
    client = ResourceClient.from_mcp_url(api_key="test", mcp_url="http://localhost:8080/mcp")

    server_response = {
        "context_id": "ctx-uuid",
        "context_name": "auto-named",
        "resource_id": "auto-named",
        "token": "kagura_resource_xyz",
        "token_id": 7,
    }
    mock_mcp = _make_mcp_mock(server_response)

    with patch("kagura_memory.resource_client.KaguraClient", return_value=mock_mcp):
        result = await client.setup_resource(resource_id="auto-named")

    assert result.warning is None
    mock_mcp.setup_resource.assert_called_once_with(
        resource_id="auto-named",
        name=None,
        summary=None,
        description=None,
        quota_events_per_hour=1000,
    )

    await client.close()


@pytest.mark.asyncio
async def test_setup_resource_requires_mcp_url():
    """setup_resource() should raise RuntimeError without from_mcp_url."""
    client = ResourceClient(api_key="test", base_url="https://test.com")

    with pytest.raises(RuntimeError, match="MCP URL"):
        await client.setup_resource(resource_id="test")

    await client.close()


@pytest.mark.asyncio
async def test_setup_resource_validates_auth_header():
    """setup_resource() should raise KaguraAuthError if auth header is malformed."""
    client = ResourceClient.from_mcp_url(api_key="test", mcp_url="http://localhost:8080/mcp")
    # Corrupt the Authorization header
    client._client.headers["authorization"] = "BadFormat"

    with pytest.raises(KaguraAuthError, match="Authorization header"):
        await client.setup_resource(resource_id="test")

    await client.close()


def test_from_mcp_url_stores_mcp_url():
    """from_mcp_url should store mcp_url for setup_resource."""
    client = ResourceClient.from_mcp_url(api_key="test", mcp_url="http://localhost:8080/mcp/w/abc")
    assert client._mcp_url == "http://localhost:8080/mcp/w/abc"


# ============================================================================
# Context manager
# ============================================================================


@pytest.mark.asyncio
async def test_context_manager():
    """async with should work correctly."""
    async with ResourceClient(api_key="test", base_url="https://test.com") as client:
        assert isinstance(client, ResourceClient)
