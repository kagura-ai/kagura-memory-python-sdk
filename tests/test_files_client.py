"""Tests for FilesClient."""

import base64
import hashlib
from datetime import UTC, datetime, timedelta
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
from kagura_memory.auth.credentials import (
    CredentialsFile,
    KaguraOAuth,
    OAuthCredentials,
    get_shared_state,
    reset_state_cache,
    save_credentials_file,
)

# ============================================================================
# Test fixtures and helpers
# ============================================================================

SAMPLE_BODY = b"hello kagura files"
SAMPLE_SHA256_HEX = hashlib.sha256(SAMPLE_BODY).hexdigest()
SAMPLE_SHA256_B64 = base64.b64encode(hashlib.sha256(SAMPLE_BODY).digest()).decode()
SAMPLE_CTX_ID = "00000000-0000-0000-0000-000000000001"
SAMPLE_FILE_ID = "10000000-0000-0000-0000-000000000002"
SAMPLE_BINDING_CTX_ID = "00000000-0000-0000-0000-0000000000bb"


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
    upload_url: str | None = None,
    *,
    sign_checksum: bool = True,
) -> dict:
    """Build a reserve response with a SigV4-style presigned ``upload_url``.

    ``sign_checksum`` controls whether ``X-Amz-SignedHeaders`` lists
    ``x-amz-checksum-sha256`` — i.e. whether the server enabled R2 checksum
    binding. The SDK must only send the checksum header when the presign
    signed it, else R2 rejects the PUT with 403 SignatureDoesNotMatch (#226).
    """
    if upload_url is None:
        signed = "content-length;content-type;host"
        if sign_checksum:
            signed += ";x-amz-checksum-sha256"
        upload_url = (
            f"https://r2.example.com/bucket/key?X-Amz-SignedHeaders={signed}"
            "&X-Amz-Signature=deadbeef"
        )
    return {
        "file_id": file_id,
        "upload_url": upload_url,
        "expires_at": "2026-05-11T00:05:00Z",
    }


# ============================================================================
# FileObject model — optional context_id binding (#222, server v0.41.0)
# ============================================================================


def test_file_object_reads_context_id_and_ignores_unknown_fields():
    """FileObject exposes the new nullable context_id and tolerates unknown fields.

    The server's FileObjectOut gained a nullable context_id (the owning context
    for ACL). The SDK model must surface it AND stay lenient about other future
    server fields — a strict (extra='forbid') model would 500 the SDK the moment
    the server adds context_id. This pins both halves of the contract.
    """
    fo = FileObject.model_validate(
        {
            **_file_object_dict(),
            "context_id": SAMPLE_BINDING_CTX_ID,
            "some_future_field": "ignored",  # forward-compat: must not raise
        }
    )
    assert fo.context_id == SAMPLE_BINDING_CTX_ID


def test_file_object_context_id_defaults_none_when_absent():
    """A legacy (NULL-context) file has no context_id key → defaults to None."""
    fo = FileObject.model_validate(_file_object_dict())
    assert fo.context_id is None


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


def test_upload_client_has_no_auth_handler_static_path():
    """Defense in depth: ``_upload_client._auth`` must be ``None`` on the static path.

    The static path uses an ``Authorization`` header on ``_client``; a stray
    ``httpx.Auth`` on ``_upload_client`` would silently inject credentials into
    R2 PUT requests. Lock the invariant so a future refactor cannot regress it.
    """
    client = FilesClient(api_key="kagura_secret", base_url="https://example.com")
    assert client._upload_client.auth is None
    # Sanity: the static API client also has no httpx.Auth — it uses the header.
    assert client._client.auth is None


def test_init_requires_api_key_or_oauth():
    """Bare FilesClient() must refuse construction with a helpful pointer.

    Auto-resolving credentials at ``__init__`` would silently read
    ``~/.kagura/credentials.json`` from disk on every constructor call.
    Force the OAuth path through ``from_mcp_url`` so the disk read is
    explicit and discoverable in the call site.
    """
    with pytest.raises(ValueError, match="from_mcp_url"):
        FilesClient()  # type: ignore[call-overload]


# ============================================================================
# SDK-level UUID validation (FilesClient.upload / list early reject)
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_context_id",
    ["auto", "not-a-uuid", "", "12345"],
    ids=["auto-sentinel", "garbage", "empty", "short-digits"],
)
async def test_upload_rejects_non_uuid_context_id_locally(bad_context_id: str):
    """``FilesClient.upload`` must reject non-UUID context_id before the server round-trip.

    Issue #110 motivating case: passing the CLI's ``"auto"`` sentinel
    straight through previously surfaced as a generic ``HTTP 422``.
    Local validation turns it into a clear ``ValueError`` whose message
    points the caller at the OAuth profile's workspace_id.
    """
    client = FilesClient(api_key="test", base_url="https://example.com")
    with pytest.raises(ValueError, match="context_id must be a UUID"):
        await client.upload(context_id=bad_context_id, source=SAMPLE_BODY, filename="x.txt")
    await client.close()


@pytest.mark.asyncio
async def test_list_rejects_non_uuid_context_id_locally():
    """``FilesClient.list`` shares the early reject so neither verb hits the wire."""
    client = FilesClient(api_key="test", base_url="https://example.com")
    with pytest.raises(ValueError, match="context_id must be a UUID"):
        await client.list(context_id="auto")
    await client.close()


# ============================================================================
# OAuth resolution (FilesClient.from_mcp_url with profile / credentials.json)
# ============================================================================


def _make_oauth_creds(
    workspace_id: str = "00000000-0000-0000-0000-0000000000ff",
    expires_in_seconds: int = 3600,
) -> OAuthCredentials:
    """Build a usable OAuthCredentials fixture for FilesClient.from_mcp_url tests."""
    return OAuthCredentials(
        server="https://oauth.example.com",
        mcp_url="https://oauth.example.com/mcp",
        client_id="kagura-cli",
        access_token="atok-files-test",
        refresh_token="rtok-files-test",
        token_type="Bearer",
        expires_at=datetime.now(UTC) + timedelta(seconds=expires_in_seconds),
        scope="memory:read memory:write",
        workspace_id=workspace_id,
        workspace_name="files-test-ws",
        user_email="files@example.com",
        issued_at=datetime.now(UTC),
    )


@pytest.fixture
def _isolated_credentials(tmp_path: Path, monkeypatch):
    """Redirect credentials.json to tmp_path and clear all credential env vars.

    Without this isolation, the OAuth resolution path on the developer's
    machine would silently pick up the real ~/.kagura/credentials.json.
    """
    fake_path = tmp_path / "credentials.json"
    monkeypatch.setattr("kagura_memory.auth.credentials.DEFAULT_CREDENTIALS_PATH", fake_path)
    monkeypatch.delenv("KAGURA_API_KEY", raising=False)
    monkeypatch.delenv("KAGURA_PROFILE", raising=False)
    monkeypatch.delenv("KAGURA_MCP_URL", raising=False)
    monkeypatch.setattr("kagura_memory._auth.load_config", lambda: {"api_key": ""})
    reset_state_cache()
    yield fake_path
    reset_state_cache()


def test_from_mcp_url_static_path_unchanged(_isolated_credentials):
    """from_mcp_url(api_key=...) keeps the static-bearer behavior intact."""
    client = FilesClient.from_mcp_url(
        api_key="kagura_explicit",
        mcp_url="https://memory.kagura-ai.com/mcp",
    )
    assert client._client.headers.get("Authorization") == "Bearer kagura_explicit"
    assert client._client.auth is None
    assert "Authorization" not in client._upload_client.headers
    assert client._upload_client.auth is None


def test_from_mcp_url_resolves_oauth_profile(_isolated_credentials):
    """With api_key=None and a profile saved on disk, from_mcp_url uses KaguraOAuth."""
    cf = CredentialsFile()
    cf.set_profile("default", _make_oauth_creds())
    save_credentials_file(cf, _isolated_credentials)

    client = FilesClient.from_mcp_url(api_key=None, profile="default")

    # OAuth path: no static Authorization header; httpx.Auth installed instead.
    assert "Authorization" not in client._client.headers
    assert isinstance(client._client.auth, KaguraOAuth)
    # base_url is derived from the profile's stored mcp_url.
    assert client.base_url == "https://oauth.example.com"
    # The R2-bound client must remain auth-less even when OAuth is in play.
    assert client._upload_client.auth is None
    assert "Authorization" not in client._upload_client.headers


def test_from_mcp_url_missing_profile_raises_loud(_isolated_credentials):
    """Explicit profile arg with no matching credentials.json entry raises."""
    cf = CredentialsFile()
    cf.set_profile("default", _make_oauth_creds())
    save_credentials_file(cf, _isolated_credentials)
    with pytest.raises(KaguraAuthError, match="Profile 'nonexistent'"):
        FilesClient.from_mcp_url(api_key=None, profile="nonexistent")


@pytest.mark.asyncio
async def test_upload_client_drops_bearer_in_oauth_mode(_isolated_credentials):
    """CSO finding: OAuth-mode R2 PUT must never carry an Authorization header.

    The KaguraOAuth httpx.Auth subclass injects ``Authorization`` per-request
    on the client it is attached to. ``_upload_client`` is constructed without
    ``auth=`` so the OAuth handler is structurally unable to reach it — this
    test pins that invariant by inspecting the actual PUT call's headers.
    """
    cf = CredentialsFile()
    cf.set_profile("default", _make_oauth_creds())
    save_credentials_file(cf, _isolated_credentials)

    client = FilesClient.from_mcp_url(api_key=None, profile="default")

    reserve_resp = _ok_response(201, _reserve_response_dict())
    confirm_resp = _ok_response(200, _file_object_dict(status="uploaded"))
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
            filename="oauth.txt",
        )

    put_headers = mock_put.call_args.kwargs["headers"]
    assert "Authorization" not in put_headers, (
        f"R2 PUT must not carry Authorization header in OAuth mode; got: {put_headers}"
    )
    await client.close()


@pytest.mark.asyncio
async def test_oauth_refresh_fires_when_token_near_expiry(_isolated_credentials):
    """KaguraOAuth refresh fires through FilesClient just like through KaguraClient.

    Stores a credential expiring inside the refresh skew window (5 min default)
    and stubs the device-flow refresh path. Calling the auth handler's
    ``_maybe_refresh`` directly is sufficient — the production code reaches
    the same entry point on every request through ``async_auth_flow``.
    Bypassing the full httpx round-trip keeps the test focused on the
    refresh wiring rather than re-testing httpx's auth-flow machinery.
    """
    near_expiry = _make_oauth_creds(expires_in_seconds=10)  # well inside 5-min skew
    cf = CredentialsFile()
    cf.set_profile("default", near_expiry)
    save_credentials_file(cf, _isolated_credentials)
    reset_state_cache()

    rotated_token = MagicMock()
    rotated_token.access_token = "atok-rotated"
    rotated_token.refresh_token = None  # server omits → keep old refresh_token
    rotated_token.expires_at = datetime.now(UTC) + timedelta(hours=1)
    rotated_token.scope = "memory:read memory:write"

    with (
        patch(
            "kagura_memory.auth.device_flow.refresh_access_token",
            new_callable=AsyncMock,
            return_value=rotated_token,
        ) as mock_refresh,
        patch("kagura_memory.auth.device_flow.make_oauth_client") as mock_client_factory,
    ):
        mock_oauth_client = AsyncMock()
        mock_oauth_client.__aenter__ = AsyncMock(return_value=mock_oauth_client)
        mock_oauth_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_factory.return_value = mock_oauth_client

        client = FilesClient.from_mcp_url(api_key=None, profile="default")
        assert isinstance(client._client.auth, KaguraOAuth)
        await client._client.auth._maybe_refresh()

    mock_refresh.assert_awaited_once()
    state = get_shared_state(profile="default")
    assert state is not None
    assert state.credentials.access_token == "atok-rotated"
    await client.close()


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


# ============================================================================
# upload() — conditional checksum header (#226): only send it when the presign
# signed it, else R2 rejects the PUT with 403 SignatureDoesNotMatch
# ============================================================================


@pytest.mark.asyncio
async def test_upload_omits_checksum_header_when_presign_does_not_sign_it():
    """If X-Amz-SignedHeaders omits the checksum header, the SDK must NOT send it.

    Regression for the live 403: a server with R2 checksum binding OFF presigns
    without ``x-amz-checksum-sha256`` in SignedHeaders; sending it anyway makes
    the SigV4 signature mismatch and R2 returns 403 SignatureDoesNotMatch.
    """
    client = FilesClient(api_key="test", base_url="https://example.com")

    reserve_resp = _ok_response(201, _reserve_response_dict(sign_checksum=False))
    confirm_resp = _ok_response(200, _file_object_dict())
    put_resp = _ok_response(200, {})

    with (
        patch.object(client._client, "request", new_callable=AsyncMock) as mock_req,
        patch.object(client._upload_client, "put", new_callable=AsyncMock) as mock_put,
    ):
        mock_req.side_effect = [reserve_resp, confirm_resp]
        mock_put.return_value = put_resp

        await client.upload(context_id=SAMPLE_CTX_ID, source=SAMPLE_BODY, filename="hello.txt")

    put_headers = mock_put.call_args.kwargs["headers"]
    assert "x-amz-checksum-sha256" not in put_headers
    # Content-Type is a signed header and must still be sent.
    assert put_headers["Content-Type"] == "text/plain"
    await client.close()


@pytest.mark.asyncio
async def test_upload_sends_checksum_header_when_presign_signs_it():
    """When the presign DOES sign the checksum header, the SDK still sends it (binding on)."""
    client = FilesClient(api_key="test", base_url="https://example.com")

    reserve_resp = _ok_response(201, _reserve_response_dict(sign_checksum=True))
    confirm_resp = _ok_response(200, _file_object_dict())
    put_resp = _ok_response(200, {})

    with (
        patch.object(client._client, "request", new_callable=AsyncMock) as mock_req,
        patch.object(client._upload_client, "put", new_callable=AsyncMock) as mock_put,
    ):
        mock_req.side_effect = [reserve_resp, confirm_resp]
        mock_put.return_value = put_resp

        await client.upload(context_id=SAMPLE_CTX_ID, source=SAMPLE_BODY, filename="hello.txt")

    put_headers = mock_put.call_args.kwargs["headers"]
    assert put_headers["x-amz-checksum-sha256"] == SAMPLE_SHA256_B64
    await client.close()


def test_presign_signs_checksum_detects_signed_header():
    """`_presign_signs_checksum` reads X-Amz-SignedHeaders case-insensitively."""
    from kagura_memory.files_client import _presign_signs_checksum

    on = "https://r2/key?X-Amz-SignedHeaders=content-length;content-type;host;x-amz-checksum-sha256"
    off = "https://r2/key?X-Amz-SignedHeaders=content-length;content-type;host"
    assert _presign_signs_checksum(on) is True
    assert _presign_signs_checksum(off) is False
    assert _presign_signs_checksum("https://r2/key") is False  # no query at all
    # Case-insensitive: some SDKs upper-case the header token.
    assert (
        _presign_signs_checksum("https://r2/key?X-Amz-SignedHeaders=X-Amz-Checksum-Sha256") is True
    )


@pytest.mark.asyncio
async def test_upload_confirm_sends_workspace_id_query_param():
    """The confirm step must carry workspace_id as a query param (#226).

    memory-cloud v0.41.0 requires ``workspace_id`` on the query string of
    ``POST /api/v1/files/{id}/confirm``; without it the server returns 422
    (VAL-001, query.workspace_id Field required) and the upload never finalizes.
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

        await client.upload(context_id=SAMPLE_CTX_ID, source=SAMPLE_BODY, filename="hello.txt")

    confirm_call = mock_req.call_args_list[1]
    assert f"/api/v1/files/{SAMPLE_FILE_ID}/confirm" in confirm_call[0][1]
    assert confirm_call.kwargs["params"] == {"workspace_id": SAMPLE_CTX_ID}
    assert confirm_call.kwargs["json"] == {"sha256": SAMPLE_SHA256_HEX}
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
# upload() — optional context_id binding (#222, server v0.41.0)
# ============================================================================


@pytest.mark.asyncio
async def test_upload_forwards_binding_context_id_to_reserve():
    """binding_context_id rides the reserve body as the wire field `context_id`.

    The wire `context_id` is the server's owning-context ACL binding — distinct
    from the SDK's `context_id` param, which maps to `workspace_id`. Both travel
    on the same reserve body when a binding is requested.
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
            binding_context_id=SAMPLE_BINDING_CTX_ID,
        )

    reserve_body = mock_req.call_args_list[0][1]["json"]
    assert reserve_body["workspace_id"] == SAMPLE_CTX_ID
    assert reserve_body["context_id"] == SAMPLE_BINDING_CTX_ID
    await client.close()


@pytest.mark.asyncio
async def test_upload_without_binding_context_id_omits_context_id():
    """Legacy path: no binding → no `context_id` key in the reserve body (NULL-context)."""
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

        await client.upload(context_id=SAMPLE_CTX_ID, source=SAMPLE_BODY, filename="hello.txt")

    reserve_body = mock_req.call_args_list[0][1]["json"]
    assert "context_id" not in reserve_body
    await client.close()


@pytest.mark.asyncio
async def test_upload_422_binding_context_not_in_workspace_surfaces_detail():
    """A binding context outside the workspace → 422; the server detail must surface."""
    client = FilesClient(api_key="test", base_url="https://example.com")
    reserve_422 = _error_response(422, {"detail": "context does not belong to this workspace"})

    with patch.object(client._client, "request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = reserve_422
        with pytest.raises(KaguraConnectionError, match="does not belong to this workspace"):
            await client.upload(
                context_id=SAMPLE_CTX_ID,
                source=SAMPLE_BODY,
                filename="hello.txt",
                binding_context_id=SAMPLE_BINDING_CTX_ID,
            )
    await client.close()


@pytest.mark.asyncio
async def test_upload_403_binding_context_denied_surfaces_detail():
    """Write-denied on the binding context → 403; the server's reason must surface."""
    client = FilesClient(
        api_key="test",
        base_url="https://example.com",
        _auth_source="config",
        _workspace_id_hint="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    )
    reserve_403 = _error_response(403, {"detail": "context write denied"})

    with patch.object(client._client, "request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = reserve_403
        with pytest.raises(KaguraConnectionError) as exc_info:
            await client.upload(
                context_id=SAMPLE_CTX_ID,
                source=SAMPLE_BODY,
                filename="hello.txt",
                binding_context_id=SAMPLE_BINDING_CTX_ID,
            )
        msg = str(exc_info.value)
    await client.close()

    assert "context write denied" in msg


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
            await client.delete("some-file-id", context_id=SAMPLE_CTX_ID)
    await client.close()


@pytest.mark.asyncio
async def test_request_403_without_source_includes_sanitized_detail():
    """Direct ``FilesClient(api_key=...)`` (no resolver) → ``HTTP 403`` + server detail.

    SDK callers who construct ``FilesClient`` without going through
    ``_from_resolved_auth`` haven't told us their credential source, so
    we cannot produce the workspace-specific hint. But the server's
    ``detail`` field still flows through (after sanitization), so
    operators can see "forbidden" / scope-related reasons even on the
    bare path.
    """
    client = FilesClient(api_key="test-key", base_url="https://example.com")
    err_resp = _error_response(403, {"detail": "forbidden"})
    with patch.object(client._client, "request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = err_resp
        with pytest.raises(KaguraConnectionError) as exc_info:
            await client.delete("some-file-id", context_id=SAMPLE_CTX_ID)
        msg = str(exc_info.value)
    await client.close()

    assert msg == "HTTP 403: forbidden"


@pytest.mark.asyncio
async def test_request_403_delete_emits_workspace_hint():
    """delete now carries workspace_id (v0.41.0), so its 403 uses the workspace hint.

    Before v0.41.0 the file-id endpoints sent no workspace and got the generic
    heading; now delete / download-url require ``workspace_id`` on the query
    string, so a 403 surfaces the workspace-mismatch hint like upload / list.
    (The generic-heading branch is still covered by the
    ``_format_workspace_403_hint`` unit tests.)
    """
    client = FilesClient(
        api_key="test-key",
        base_url="https://example.com",
        _auth_source="config",
        _workspace_id_hint="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    )
    err_resp = _error_response(403, {"detail": "insufficient_scope"})
    with patch.object(client._client, "request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = err_resp
        with pytest.raises(KaguraConnectionError) as exc_info:
            await client.delete("some-file-id", context_id="11111111-2222-3333-4444-555555555555")
        msg = str(exc_info.value)
    await client.close()

    assert "HTTP 403" in msg
    assert "workspace not accessible" in msg
    # Source provenance, bound + requested workspace prefixes, and detail surface.
    assert ".kagura.json" in msg
    assert "aaaaaaaa" in msg
    assert "workspace requested: 11111111" in msg
    assert "insufficient_scope" in msg


@pytest.mark.asyncio
async def test_request_403_with_source_emits_workspace_hint():
    """OAuth-sourced client → 403 message names the source and workspace prefix.

    Same-source pairing (#115) attaches ``_auth_source="oauth"`` and
    the OAuth profile's ``workspace_id`` to the client. On 403 the SDK
    emits a multi-line hint citing the source label and both the
    bound and requested workspace prefixes.
    """
    client = FilesClient(
        api_key="test-key",
        base_url="https://example.com",
        _auth_source="oauth",
        _workspace_id_hint="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    )
    err_resp = _error_response(403, {"detail": "forbidden"})
    with patch.object(client._client, "request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = err_resp
        with pytest.raises(KaguraConnectionError) as exc_info:
            # files list sends workspace_id in query params.
            await client.list(context_id="11111111-2222-3333-4444-555555555555")
        msg = str(exc_info.value)
    await client.close()

    assert "HTTP 403" in msg
    assert "workspace not accessible" in msg
    assert "OAuth profile" in msg
    assert "aaaaaaaa" in msg
    assert "workspace requested: 11111111" in msg
    assert "--context-id" in msg


def test_sanitize_server_detail_drops_credential_markers():
    """A server detail string containing credential markers must be dropped.

    ``detail`` is operator-facing text the server controls. A future server
    bug echoing back the Bearer header or api_key value would otherwise
    flow straight through to the user. The sanitizer drops any detail
    that contains common credential-shaped markers.
    """
    from kagura_memory.files_client import _sanitize_server_detail

    # Safe details pass through unchanged.
    assert _sanitize_server_detail("forbidden") == "forbidden"
    assert _sanitize_server_detail("insufficient_scope") == "insufficient_scope"
    assert _sanitize_server_detail("account deactivated") == "account deactivated"
    # Empty / None → None.
    assert _sanitize_server_detail("") is None
    assert _sanitize_server_detail(None) is None
    # Credential markers → drop (case-insensitive).
    assert _sanitize_server_detail("Bearer rotk-leaked-value") is None
    assert _sanitize_server_detail("bearer xxx") is None
    assert _sanitize_server_detail("Authorization: Bearer xxx") is None
    assert _sanitize_server_detail("api_key=plaintext-bad") is None


def test_format_workspace_403_hint_handles_unknown_source_tag():
    """Defensive: an unexpected ``auth_source`` value never raises KeyError.

    ``_auth_source`` is set only by internal resolver code so this path
    shouldn't fire in practice — but 403 handling must not crash mid-error,
    or the real HTTP failure would be masked by a KeyError.
    """
    from kagura_memory.files_client import _format_workspace_403_hint

    msg = _format_workspace_403_hint(
        auth_source="unexpected-source-tag",  # type: ignore[arg-type]
        source_workspace_hint=None,
        requested_workspace=None,
    )
    assert "HTTP 403" in msg
    assert "unexpected-source-tag" in msg


def test_short_workspace_returns_none_for_empty():
    """`_short_workspace` returns the ``<none>`` sentinel for empty/None input."""
    from kagura_memory.files_client import _short_workspace

    assert _short_workspace(None) == "<none>"
    assert _short_workspace("") == "<none>"


def test_extract_requested_workspace_handles_no_workspace_path():
    """``_extract_requested_workspace`` returns None when no workspace_id is carried.

    Covers the file-id-based request path (download-url / delete / confirm)
    where the 403 hint must render without a "workspace requested:" line.
    """
    from kagura_memory.files_client import _extract_requested_workspace

    assert _extract_requested_workspace(None, None) is None
    assert _extract_requested_workspace({}, None) is None
    assert _extract_requested_workspace(None, {}) is None
    # Non-string workspace_id (defensive against malformed payloads) → None.
    assert _extract_requested_workspace({"workspace_id": None}, None) is None
    assert _extract_requested_workspace(None, {"workspace_id": 42}) is None
    # Happy paths: string workspace_id is recovered from either body or params.
    assert _extract_requested_workspace({"workspace_id": "ws-from-json"}, None) == "ws-from-json"
    assert (
        _extract_requested_workspace(None, {"workspace_id": "ws-from-params"}) == "ws-from-params"
    )


@pytest.mark.asyncio
async def test_request_403_config_source_emits_workspace_hint():
    """Static api_key from ``.kagura.json`` → 403 hint shows the bound workspace.

    After the /simplify refactor the CLI threads
    ``workspace_id_hint`` into ``FilesClient._from_resolved_auth`` for
    the ``_StaticAuth(source="config")`` path, so a config-source 403
    no longer surfaces ``workspace=<none>`` — it shows the workspace
    UUID prefix the api_key was bound to.
    """
    client = FilesClient(
        api_key="test-key",
        base_url="https://example.com",
        _auth_source="config",
        _workspace_id_hint="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    )
    err_resp = _error_response(403, {"detail": "forbidden"})
    with patch.object(client._client, "request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = err_resp
        with pytest.raises(KaguraConnectionError) as exc_info:
            await client.list(context_id="11111111-2222-3333-4444-555555555555")
        msg = str(exc_info.value)
    await client.close()

    assert ".kagura.json" in msg
    # Bound workspace prefix surfaces — not "<none>".
    assert "aaaaaaaa" in msg
    assert "workspace=<none>" not in msg
    # Request workspace is distinct from the bound one (the mismatch the
    # hint is meant to clarify).
    assert "workspace requested: 11111111" in msg


@pytest.mark.asyncio
async def test_request_403_hint_does_not_leak_api_key_or_bearer():
    """The 403 hint must not include the api_key value or Authorization header.

    python.md rule: never store api_keys as instance attributes. The
    hint is built from the source label and workspace UUIDs only —
    this test pins that promise so a future refactor cannot
    accidentally embed the secret in the operator-facing message.
    """
    secret = "kagura_super_secret_api_key_value"
    client = FilesClient(
        api_key=secret,
        base_url="https://example.com",
        _auth_source="config",
        _workspace_id_hint="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    )
    err_resp = _error_response(403, {"detail": "forbidden"})
    with patch.object(client._client, "request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = err_resp
        with pytest.raises(KaguraConnectionError) as exc_info:
            await client.list(context_id="11111111-2222-3333-4444-555555555555")
        msg = str(exc_info.value)
    await client.close()

    assert secret not in msg
    assert "Bearer" not in msg
    assert "api_key=" not in msg  # no key=value pair surfacing the value
    assert "Authorization" not in msg


@pytest.mark.asyncio
async def test_request_404_raises_not_found():
    client = FilesClient(api_key="test", base_url="https://example.com")
    err_resp = _error_response(404, {"detail": "file gone"})
    with patch.object(client._client, "request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = err_resp
        with pytest.raises(KaguraNotFoundError, match="file gone"):
            await client.download_url("some-file-id", context_id=SAMPLE_CTX_ID)
    await client.close()


@pytest.mark.asyncio
async def test_request_429_raises_quota_error_with_retry_after():
    client = FilesClient(api_key="test", base_url="https://example.com")
    err_resp = _error_response(429, {})
    err_resp.headers = {"Retry-After": "60"}
    with patch.object(client._client, "request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = err_resp
        with pytest.raises(KaguraQuotaError) as exc_info:
            await client.delete("some-file-id", context_id=SAMPLE_CTX_ID)
        assert exc_info.value.retry_after == 60
    await client.close()


@pytest.mark.asyncio
async def test_request_network_error_raises_connection_error():
    client = FilesClient(api_key="test", base_url="https://example.com")
    with patch.object(client._client, "request", new_callable=AsyncMock) as mock_req:
        mock_req.side_effect = httpx.ConnectError("refused")
        with pytest.raises(KaguraConnectionError, match="Connection failed"):
            await client.delete("some-file-id", context_id=SAMPLE_CTX_ID)
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
        url = await client.download_url(SAMPLE_FILE_ID, context_id=SAMPLE_CTX_ID)
    assert url == "https://r2.example.com/get/key?sig=..."
    call = mock_req.call_args
    assert call[0][0] == "GET"
    assert f"/api/v1/files/{SAMPLE_FILE_ID}/download-url" in call[0][1]
    # v0.41.0 requires workspace_id on the query string of file-id endpoints (#226).
    assert call.kwargs["params"] == {"workspace_id": SAMPLE_CTX_ID}
    await client.close()


@pytest.mark.asyncio
async def test_delete_returns_none():
    client = FilesClient(api_key="test", base_url="https://example.com")
    resp = _ok_response(204, {})
    with patch.object(client._client, "request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = resp
        result = await client.delete(SAMPLE_FILE_ID, context_id=SAMPLE_CTX_ID)
    assert result is None
    call = mock_req.call_args
    assert call[0][0] == "DELETE"
    assert f"/api/v1/files/{SAMPLE_FILE_ID}" in call[0][1]
    assert call.kwargs["params"] == {"workspace_id": SAMPLE_CTX_ID}
    await client.close()


@pytest.mark.asyncio
async def test_download_url_rejects_non_uuid_context_id_locally():
    """download_url validates context_id before the wire (parity with upload/list)."""
    client = FilesClient(api_key="test", base_url="https://example.com")
    with pytest.raises(ValueError, match="context_id must be a UUID"):
        await client.download_url("some-file-id", context_id="auto")
    await client.close()


@pytest.mark.asyncio
async def test_delete_rejects_non_uuid_context_id_locally():
    """delete validates context_id before the wire (parity with upload/list)."""
    client = FilesClient(api_key="test", base_url="https://example.com")
    with pytest.raises(ValueError, match="context_id must be a UUID"):
        await client.delete("some-file-id", context_id="auto")
    await client.close()


@pytest.mark.asyncio
async def test_list_canonicalizes_noncanonical_context_id_spelling():
    """Tolerated non-canonical UUID spellings ({braces}, dashless) are
    CANONICALIZED before the wire, not sent raw (#236) — a validate-only
    check let them through and they surfaced server-side as a misleading
    uniform 404."""
    client = FilesClient(api_key="test", base_url="https://example.com")
    resp = _ok_response(200, [])
    resp.json.return_value = []

    with patch.object(client._client, "request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = resp
        await client.list(context_id=SAMPLE_CTX_ID.replace("-", ""))
        await client.list(context_id="{" + SAMPLE_CTX_ID + "}")

    for call in mock_req.call_args_list:
        assert call[1]["params"]["workspace_id"] == SAMPLE_CTX_ID

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
            await client.delete("some-id", context_id=SAMPLE_CTX_ID)
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
