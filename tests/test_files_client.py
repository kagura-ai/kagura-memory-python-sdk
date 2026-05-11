"""Tests for FilesClient."""

import base64
import hashlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from kagura_memory import (
    FileListResponse,
    FileObject,
    FilesClient,
    KaguraAuthError,
    KaguraConnectionError,
    KaguraIntegrityError,
    KaguraNotFoundError,
    KaguraQuotaError,
)

# ============================================================================
# Test fixtures and helpers
# ============================================================================

SAMPLE_BODY = b"hello kagura files"
SAMPLE_SHA256_HEX = hashlib.sha256(SAMPLE_BODY).hexdigest()
SAMPLE_SHA256_B64 = base64.b64encode(hashlib.sha256(SAMPLE_BODY).digest()).decode()
SAMPLE_CTX_ID = "00000000-0000-0000-0000-000000000001"
SAMPLE_FILE_ID = "10000000-0000-0000-0000-000000000002"


def _ok_response(
    status_code: int = 200,
    json_data: dict | list | None = None,
) -> MagicMock:
    """Build a MagicMock response with no error side-effect."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = {} if json_data is None else json_data
    resp.raise_for_status = MagicMock()
    resp.headers = {}
    return resp


def _error_response(status_code: int, json_data: dict | None = None) -> MagicMock:
    """Build a MagicMock response that raises HTTPStatusError on raise_for_status."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.headers = {}
    resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        f"{status_code}", request=MagicMock(), response=resp
    )
    return resp


def _file_object_dict(file_id: str = SAMPLE_FILE_ID, status: str = "uploaded") -> dict:
    """Build a FileObject-shaped dict matching server's FileObjectOut."""
    return {
        "id": file_id,
        "workspace_id": SAMPLE_CTX_ID,
        "filename": "hello.txt",
        "content_type": "text/plain",
        "size_bytes": len(SAMPLE_BODY),
        "sha256": SAMPLE_SHA256_HEX,
        "status": status,
        "created_at": "2026-05-11T00:00:00Z",
        "uploaded_at": "2026-05-11T00:00:01Z",
    }


def _reserve_response_dict(
    file_id: str = SAMPLE_FILE_ID,
    upload_url: str = "https://r2.example.com/bucket/key",
) -> dict:
    return {
        "file_id": file_id,
        "upload_url": upload_url,
        "expires_at": "2026-05-11T00:05:00Z",
    }


# ============================================================================
# Construction / HTTPS enforcement
# ============================================================================


def test_rejects_http_url():
    """HTTP URLs (non-localhost) should raise ValueError."""
    with pytest.raises(ValueError, match="must use HTTPS"):
        FilesClient(api_key="test", base_url="http://evil.com")


def test_allows_https_url():
    client = FilesClient(api_key="test", base_url="https://memory.kagura-ai.com")
    assert client.base_url == "https://memory.kagura-ai.com"


def test_allows_localhost_http():
    client = FilesClient(api_key="test", base_url="http://localhost:8080")
    assert client.base_url == "http://localhost:8080"


def test_strips_trailing_slash():
    client = FilesClient(api_key="test", base_url="https://example.com/")
    assert client.base_url == "https://example.com"


def test_from_mcp_url_strips_mcp_path():
    client = FilesClient.from_mcp_url(api_key="test", mcp_url="https://memory.kagura-ai.com/mcp")
    assert client.base_url == "https://memory.kagura-ai.com"


def test_from_mcp_url_strips_workspace_segment():
    client = FilesClient.from_mcp_url(
        api_key="test", mcp_url="https://memory.kagura-ai.com/mcp/w/abc"
    )
    assert client.base_url == "https://memory.kagura-ai.com"


def test_api_key_not_stored_as_attribute():
    """API key must live only in the httpx headers, never as an instance attribute."""
    client = FilesClient(api_key="kagura_secret", base_url="https://example.com")
    public_attrs = {k: v for k, v in vars(client).items() if not k.startswith("_")}
    for value in public_attrs.values():
        assert "kagura_secret" not in str(value)


def test_upload_client_has_no_authorization_header():
    """The R2-bound client must NOT carry the Bearer token by construction."""
    client = FilesClient(api_key="kagura_secret", base_url="https://example.com")
    assert "Authorization" not in client._upload_client.headers
    # Sanity: the API client still has it.
    assert client._client.headers.get("Authorization") == "Bearer kagura_secret"


# ============================================================================
# upload() — happy path
# ============================================================================


@pytest.mark.asyncio
async def test_upload_bytes_happy_path():
    """reserve → PUT → confirm; returns finalized FileObject."""
    client = FilesClient(api_key="test", base_url="https://example.com")

    reserve_resp = _ok_response(201, _reserve_response_dict())
    confirm_resp = _ok_response(200, _file_object_dict(status="uploaded"))
    put_resp = _ok_response(200, {})

    with (
        patch.object(client._client, "request", new_callable=AsyncMock) as mock_req,
        patch.object(client._upload_client, "put", new_callable=AsyncMock) as mock_put,
    ):
        mock_req.side_effect = [reserve_resp, confirm_resp]
        mock_put.return_value = put_resp

        result = await client.upload(
            context_id=SAMPLE_CTX_ID,
            source=SAMPLE_BODY,
            filename="hello.txt",
        )

    assert isinstance(result, FileObject)
    assert result.id == SAMPLE_FILE_ID
    assert result.status == "uploaded"

    # reserve call inspection
    reserve_call = mock_req.call_args_list[0]
    assert reserve_call[0][0] == "POST"
    assert reserve_call[0][1].endswith("/api/v1/files/reserve")
    body = reserve_call[1]["json"]
    assert body["workspace_id"] == SAMPLE_CTX_ID
    assert body["filename"] == "hello.txt"
    assert body["size_bytes"] == len(SAMPLE_BODY)
    assert body["sha256"] == SAMPLE_SHA256_HEX

    # confirm call inspection
    confirm_call = mock_req.call_args_list[1]
    assert confirm_call[0][0] == "POST"
    assert f"/api/v1/files/{SAMPLE_FILE_ID}/confirm" in confirm_call[0][1]
    assert confirm_call[1]["json"] == {"sha256": SAMPLE_SHA256_HEX}

    await client.close()


@pytest.mark.asyncio
async def test_upload_sends_base64_raw_digest_header():
    """x-amz-checksum-sha256 must be base64 of the raw 32-byte digest, NOT hex.

    This is the canonical footgun: base64-encoding the hex string yields a
    64-char value; base64-encoding the raw digest yields a 44-char value.
    R2 enforces the raw-digest form.
    """
    client = FilesClient(api_key="test", base_url="https://example.com")

    reserve_resp = _ok_response(201, _reserve_response_dict())
    confirm_resp = _ok_response(200, _file_object_dict())
    put_resp = _ok_response(200, {})

    with (
        patch.object(client._client, "request", new_callable=AsyncMock) as mock_req,
        patch.object(client._upload_client, "put", new_callable=AsyncMock) as mock_put,
    ):
        mock_req.side_effect = [reserve_resp, confirm_resp]
        mock_put.return_value = put_resp

        await client.upload(
            context_id=SAMPLE_CTX_ID,
            source=SAMPLE_BODY,
            filename="hello.txt",
        )

    # PUT was called with the correct base64-of-raw-digest header
    put_kwargs = mock_put.call_args.kwargs
    sent = put_kwargs["headers"]["x-amz-checksum-sha256"]
    assert sent == SAMPLE_SHA256_B64
    assert len(sent) == 44  # base64(32 raw bytes) — never 64 (which would be hex-encoded)
    # And NOT the wrong hex-then-base64 form
    wrong = base64.b64encode(SAMPLE_SHA256_HEX.encode()).decode()
    assert sent != wrong
    assert len(wrong) == 88  # sanity: the wrong form has different length

    await client.close()


@pytest.mark.asyncio
async def test_upload_from_path(tmp_path: Path):
    """Path source reads bytes from disk and uses path.name as filename."""
    client = FilesClient(api_key="test", base_url="https://example.com")

    p = tmp_path / "report.pdf"
    p.write_bytes(SAMPLE_BODY)

    reserve_resp = _ok_response(201, _reserve_response_dict())
    confirm_resp = _ok_response(200, _file_object_dict())
    put_resp = _ok_response(200, {})

    with (
        patch.object(client._client, "request", new_callable=AsyncMock) as mock_req,
        patch.object(client._upload_client, "put", new_callable=AsyncMock) as mock_put,
    ):
        mock_req.side_effect = [reserve_resp, confirm_resp]
        mock_put.return_value = put_resp

        await client.upload(context_id=SAMPLE_CTX_ID, source=p)

    body = mock_req.call_args_list[0][1]["json"]
    assert body["filename"] == "report.pdf"
    assert body["sha256"] == SAMPLE_SHA256_HEX
    # mimetypes guess for .pdf
    assert body["content_type"] == "application/pdf"

    await client.close()


@pytest.mark.asyncio
async def test_upload_explicit_content_type_overrides_sniff():
    """User-supplied content_type is passed through verbatim."""
    client = FilesClient(api_key="test", base_url="https://example.com")

    reserve_resp = _ok_response(201, _reserve_response_dict())
    confirm_resp = _ok_response(200, _file_object_dict())
    put_resp = _ok_response(200, {})

    with (
        patch.object(client._client, "request", new_callable=AsyncMock) as mock_req,
        patch.object(client._upload_client, "put", new_callable=AsyncMock) as mock_put,
    ):
        mock_req.side_effect = [reserve_resp, confirm_resp]
        mock_put.return_value = put_resp

        await client.upload(
            context_id=SAMPLE_CTX_ID,
            source=SAMPLE_BODY,
            filename="hello.txt",
            content_type="application/x-custom",
        )

    body = mock_req.call_args_list[0][1]["json"]
    assert body["content_type"] == "application/x-custom"

    await client.close()


@pytest.mark.asyncio
async def test_upload_content_type_defaults_to_octet_stream_for_bytes():
    """bytes input without filename hint → application/octet-stream."""
    client = FilesClient(api_key="test", base_url="https://example.com")

    reserve_resp = _ok_response(201, _reserve_response_dict())
    confirm_resp = _ok_response(200, _file_object_dict())
    put_resp = _ok_response(200, {})

    with (
        patch.object(client._client, "request", new_callable=AsyncMock) as mock_req,
        patch.object(client._upload_client, "put", new_callable=AsyncMock) as mock_put,
    ):
        mock_req.side_effect = [reserve_resp, confirm_resp]
        mock_put.return_value = put_resp

        await client.upload(
            context_id=SAMPLE_CTX_ID,
            source=SAMPLE_BODY,
            filename="payload",  # no extension → unguessable
        )

    body = mock_req.call_args_list[0][1]["json"]
    assert body["content_type"] == "application/octet-stream"

    await client.close()


@pytest.mark.asyncio
async def test_upload_path_not_found_raises(tmp_path: Path):
    """source=Path pointing at a missing file → FileNotFoundError, no HTTP call."""
    client = FilesClient(api_key="test", base_url="https://example.com")
    missing = tmp_path / "missing.bin"  # tmp_path guarantees this doesn't exist
    with (
        patch.object(client._client, "request", new_callable=AsyncMock) as mock_req,
        patch.object(client._upload_client, "put", new_callable=AsyncMock) as mock_put,
    ):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            await client.upload(context_id=SAMPLE_CTX_ID, source=missing)
    mock_req.assert_not_called()
    mock_put.assert_not_called()
    await client.close()


@pytest.mark.asyncio
async def test_upload_bytes_requires_filename():
    """source=bytes without filename → ValueError before any HTTP call."""
    client = FilesClient(api_key="test", base_url="https://example.com")

    with (
        patch.object(client._client, "request", new_callable=AsyncMock) as mock_req,
        patch.object(client._upload_client, "put", new_callable=AsyncMock) as mock_put,
    ):
        with pytest.raises(ValueError, match="filename is required"):
            await client.upload(context_id=SAMPLE_CTX_ID, source=SAMPLE_BODY)

    mock_req.assert_not_called()
    mock_put.assert_not_called()
    await client.close()


# ============================================================================
# upload() — error and dedup paths
# ============================================================================


@pytest.mark.asyncio
async def test_upload_409_dedup_returns_existing_file():
    """reserve 409 with existing_file payload → return FileObject, no PUT/confirm."""
    client = FilesClient(api_key="test", base_url="https://example.com")

    existing = _file_object_dict(file_id="existing-uuid", status="uploaded")
    dup_resp = _error_response(
        409,
        {"detail": "file with this sha256 already exists", "existing_file": existing},
    )

    with (
        patch.object(client._client, "request", new_callable=AsyncMock) as mock_req,
        patch.object(client._upload_client, "put", new_callable=AsyncMock) as mock_put,
    ):
        mock_req.return_value = dup_resp

        result = await client.upload(
            context_id=SAMPLE_CTX_ID,
            source=SAMPLE_BODY,
            filename="hello.txt",
        )

    assert result.id == "existing-uuid"
    assert result.status == "uploaded"
    # No PUT, no confirm — dedup short-circuited the flow.
    assert mock_req.call_count == 1
    mock_put.assert_not_called()
    await client.close()


@pytest.mark.asyncio
async def test_upload_409_without_existing_file_raises():
    """409 without an existing_file payload is just a normal connection error."""
    client = FilesClient(api_key="test", base_url="https://example.com")

    dup_resp = _error_response(409, {"detail": "conflict, but no payload"})

    with patch.object(client._client, "request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = dup_resp

        with pytest.raises(KaguraConnectionError, match="HTTP 409"):
            await client.upload(
                context_id=SAMPLE_CTX_ID,
                source=SAMPLE_BODY,
                filename="hello.txt",
            )

    await client.close()


@pytest.mark.asyncio
async def test_upload_r2_bad_digest_raises_integrity_error():
    """R2 returning 400 → KaguraIntegrityError (not a generic connection error)."""
    client = FilesClient(api_key="test", base_url="https://example.com")

    reserve_resp = _ok_response(201, _reserve_response_dict())
    r2_error = _error_response(400, {})

    with (
        patch.object(client._client, "request", new_callable=AsyncMock) as mock_req,
        patch.object(client._upload_client, "put", new_callable=AsyncMock) as mock_put,
    ):
        mock_req.return_value = reserve_resp
        mock_put.return_value = r2_error

        with pytest.raises(KaguraIntegrityError, match="BadDigest"):
            await client.upload(
                context_id=SAMPLE_CTX_ID,
                source=SAMPLE_BODY,
                filename="hello.txt",
            )

    await client.close()


@pytest.mark.asyncio
async def test_upload_r2_network_error_raises_connection_error():
    """Non-timeout httpx.RequestError on PUT → KaguraConnectionError."""
    client = FilesClient(api_key="test", base_url="https://example.com")

    reserve_resp = _ok_response(201, _reserve_response_dict())

    with (
        patch.object(client._client, "request", new_callable=AsyncMock) as mock_req,
        patch.object(client._upload_client, "put", new_callable=AsyncMock) as mock_put,
    ):
        mock_req.return_value = reserve_resp
        mock_put.side_effect = httpx.ConnectError("connection refused")

        with pytest.raises(KaguraConnectionError, match="Object store PUT failed"):
            await client.upload(
                context_id=SAMPLE_CTX_ID,
                source=SAMPLE_BODY,
                filename="hello.txt",
            )

    await client.close()


@pytest.mark.asyncio
async def test_upload_r2_5xx_raises_connection_error():
    """Non-400 R2 HTTP error (e.g. 500) → KaguraConnectionError, not IntegrityError."""
    client = FilesClient(api_key="test", base_url="https://example.com")

    reserve_resp = _ok_response(201, _reserve_response_dict())
    r2_500 = _error_response(500, {})

    with (
        patch.object(client._client, "request", new_callable=AsyncMock) as mock_req,
        patch.object(client._upload_client, "put", new_callable=AsyncMock) as mock_put,
    ):
        mock_req.return_value = reserve_resp
        mock_put.return_value = r2_500

        with pytest.raises(KaguraConnectionError, match="Object store PUT failed: HTTP 500"):
            await client.upload(
                context_id=SAMPLE_CTX_ID,
                source=SAMPLE_BODY,
                filename="hello.txt",
            )

    await client.close()


@pytest.mark.asyncio
async def test_upload_r2_timeout_raises_connection_error():
    """R2 PUT timeout → KaguraConnectionError with 'timed out' in message."""
    client = FilesClient(api_key="test", base_url="https://example.com")

    reserve_resp = _ok_response(201, _reserve_response_dict())

    with (
        patch.object(client._client, "request", new_callable=AsyncMock) as mock_req,
        patch.object(client._upload_client, "put", new_callable=AsyncMock) as mock_put,
    ):
        mock_req.return_value = reserve_resp
        mock_put.side_effect = httpx.TimeoutException("timeout")

        with pytest.raises(KaguraConnectionError, match="timed out"):
            await client.upload(
                context_id=SAMPLE_CTX_ID,
                source=SAMPLE_BODY,
                filename="hello.txt",
            )

    await client.close()


# ============================================================================
# Error mapping on _request (mirrors ResourceClient contract)
# ============================================================================


@pytest.mark.asyncio
async def test_request_401_raises_auth_error():
    client = FilesClient(api_key="test", base_url="https://example.com")
    err_resp = _error_response(401, {"detail": "bad key"})
    with patch.object(client._client, "request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = err_resp
        with pytest.raises(KaguraAuthError):
            await client.delete("some-file-id")
    await client.close()


@pytest.mark.asyncio
async def test_request_404_raises_not_found():
    client = FilesClient(api_key="test", base_url="https://example.com")
    err_resp = _error_response(404, {"detail": "file gone"})
    with patch.object(client._client, "request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = err_resp
        with pytest.raises(KaguraNotFoundError, match="file gone"):
            await client.download_url("some-file-id")
    await client.close()


@pytest.mark.asyncio
async def test_request_429_raises_quota_error_with_retry_after():
    client = FilesClient(api_key="test", base_url="https://example.com")
    err_resp = _error_response(429, {})
    err_resp.headers = {"Retry-After": "60"}
    with patch.object(client._client, "request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = err_resp
        with pytest.raises(KaguraQuotaError) as exc_info:
            await client.delete("some-file-id")
        assert exc_info.value.retry_after == 60
    await client.close()


@pytest.mark.asyncio
async def test_request_network_error_raises_connection_error():
    client = FilesClient(api_key="test", base_url="https://example.com")
    with patch.object(client._client, "request", new_callable=AsyncMock) as mock_req:
        mock_req.side_effect = httpx.ConnectError("refused")
        with pytest.raises(KaguraConnectionError, match="Connection failed"):
            await client.delete("some-file-id")
    await client.close()


# ============================================================================
# download_url / delete / list
# ============================================================================


@pytest.mark.asyncio
async def test_download_url_returns_string():
    client = FilesClient(api_key="test", base_url="https://example.com")
    resp = _ok_response(200, {"download_url": "https://r2.example.com/get/key?sig=..."})
    with patch.object(client._client, "request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = resp
        url = await client.download_url(SAMPLE_FILE_ID)
    assert url == "https://r2.example.com/get/key?sig=..."
    call = mock_req.call_args
    assert call[0][0] == "GET"
    assert f"/api/v1/files/{SAMPLE_FILE_ID}/download-url" in call[0][1]
    await client.close()


@pytest.mark.asyncio
async def test_delete_returns_none():
    client = FilesClient(api_key="test", base_url="https://example.com")
    resp = _ok_response(204, {})
    with patch.object(client._client, "request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = resp
        result = await client.delete(SAMPLE_FILE_ID)
    assert result is None
    call = mock_req.call_args
    assert call[0][0] == "DELETE"
    assert f"/api/v1/files/{SAMPLE_FILE_ID}" in call[0][1]
    await client.close()


@pytest.mark.asyncio
async def test_list_with_legacy_bare_array_response():
    """Server v0.15.x returns bare list[FileObjectOut]; SDK wraps it."""
    client = FilesClient(api_key="test", base_url="https://example.com")
    resp = _ok_response(200, [_file_object_dict(), _file_object_dict(file_id="another")])
    # Mocked response returns the list when .json() is called
    resp.json.return_value = [_file_object_dict(), _file_object_dict(file_id="another")]

    with patch.object(client._client, "request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = resp
        result = await client.list(context_id=SAMPLE_CTX_ID, limit=10)

    assert isinstance(result, FileListResponse)
    assert len(result.files) == 2
    assert result.next_cursor is None  # current server has no cursor

    call = mock_req.call_args
    assert call[0][0] == "GET"
    assert "/api/v1/files" in call[0][1]
    assert call[1]["params"]["workspace_id"] == SAMPLE_CTX_ID
    assert call[1]["params"]["limit"] == 10

    await client.close()


@pytest.mark.asyncio
async def test_list_forwards_cursor_when_provided():
    """When cursor is given, it is forwarded as a query param (forward-compat)."""
    client = FilesClient(api_key="test", base_url="https://example.com")
    resp = _ok_response(200, [])
    resp.json.return_value = []

    with patch.object(client._client, "request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = resp
        await client.list(context_id=SAMPLE_CTX_ID, limit=20, cursor="opaque-token")

    call = mock_req.call_args
    assert call[1]["params"]["cursor"] == "opaque-token"
    await client.close()


@pytest.mark.asyncio
async def test_list_with_future_dict_response():
    """A future server returning {files, next_cursor} is parsed natively."""
    client = FilesClient(api_key="test", base_url="https://example.com")
    resp = _ok_response(
        200,
        {"files": [_file_object_dict()], "next_cursor": "page-2-token"},
    )

    with patch.object(client._client, "request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = resp
        result = await client.list(context_id=SAMPLE_CTX_ID)

    assert len(result.files) == 1
    assert result.next_cursor == "page-2-token"
    await client.close()


# ============================================================================
# Lifecycle
# ============================================================================


# ============================================================================
# _extract_existing_file defensive branches (module-level helper)
# ============================================================================


def _make_dedup_error(response_mock: MagicMock) -> KaguraConnectionError:
    """Build a KaguraConnectionError chained from an HTTPStatusError, mirroring
    what `_request` raises for non-{401, 404, 429} HTTP errors."""
    status_err = httpx.HTTPStatusError(
        f"{response_mock.status_code}",
        request=MagicMock(),
        response=response_mock,
    )
    err = KaguraConnectionError(f"HTTP {response_mock.status_code}")
    err.__cause__ = status_err
    return err


def test_extract_existing_file_returns_none_when_cause_not_httpstatus():
    """KaguraConnectionError without an HTTPStatusError cause → None."""
    from kagura_memory.files_client import _extract_existing_file

    err = KaguraConnectionError("plain network failure")
    err.__cause__ = httpx.ConnectError("refused")
    assert _extract_existing_file(err) is None


def test_extract_existing_file_returns_none_for_non_409_status():
    """Cause is HTTPStatusError but status != 409 → None."""
    from kagura_memory.files_client import _extract_existing_file

    resp = _error_response(500)
    err = _make_dedup_error(resp)
    assert _extract_existing_file(err) is None


def test_extract_existing_file_returns_none_when_body_not_parseable():
    """409 body that fails to decode as JSON → None (defensive)."""
    from kagura_memory.files_client import _extract_existing_file

    resp = _error_response(409)
    resp.json.side_effect = ValueError("not json")
    err = _make_dedup_error(resp)
    assert _extract_existing_file(err) is None


def test_extract_existing_file_returns_none_when_body_is_not_dict():
    """409 body that's a bare list/string → None (defensive)."""
    from kagura_memory.files_client import _extract_existing_file

    resp = _error_response(409)
    resp.json.return_value = ["not", "a", "dict"]
    err = _make_dedup_error(resp)
    assert _extract_existing_file(err) is None


# ============================================================================
# extract_detail (_http.py) — non-dict body fallthrough
# ============================================================================


@pytest.mark.asyncio
async def test_request_500_with_non_dict_body_message():
    """5xx with a bare-string JSON body → HTTP 500 message (no ': detail' suffix)."""
    client = FilesClient(api_key="test", base_url="https://example.com")
    err_resp = _error_response(500)
    err_resp.json.return_value = "just a string, not a dict"
    with patch.object(client._client, "request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = err_resp
        with pytest.raises(KaguraConnectionError) as exc_info:
            await client.delete("some-id")
    # extract_detail returns "" for non-dict bodies; the message has no suffix.
    assert str(exc_info.value) == "HTTP 500"
    await client.close()


# ============================================================================
# Lifecycle
# ============================================================================


@pytest.mark.asyncio
async def test_close_closes_both_clients():
    client = FilesClient(api_key="test", base_url="https://example.com")
    with (
        patch.object(client._client, "aclose", new_callable=AsyncMock) as api_close,
        patch.object(client._upload_client, "aclose", new_callable=AsyncMock) as upload_close,
    ):
        await client.close()
    api_close.assert_awaited_once()
    upload_close.assert_awaited_once()


@pytest.mark.asyncio
async def test_async_context_manager_closes_on_exit():
    client = FilesClient(api_key="test", base_url="https://example.com")
    with (
        patch.object(client._client, "aclose", new_callable=AsyncMock) as api_close,
        patch.object(client._upload_client, "aclose", new_callable=AsyncMock) as upload_close,
    ):
        async with client:
            pass
    api_close.assert_awaited_once()
    upload_close.assert_awaited_once()
