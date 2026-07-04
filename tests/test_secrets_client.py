"""Tests for kagura_memory.secrets.client.SecretClient (REST wire layer).

Schema mirrors the live memory-cloud OpenAPI (v0.39.0) for
``/api/v1/config/secrets``. HTTP is mocked at ``client._client.request``,
following the ResourceClient test convention.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from kagura_memory._auth import _OAuthAuth
from kagura_memory.exceptions import (
    KaguraAuthError,
    KaguraConnectionError,
    KaguraNotFoundError,
    KaguraSecretError,
)
from kagura_memory.secrets import crypto
from kagura_memory.secrets.client import SecretClient
from kagura_memory.secrets.models import (
    PubkeyResponse,
    SecretMetaResponse,
    SecretPutResponse,
    SecretValueResponse,
)


def _mock_response(status_code: int = 200, json_data=None) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = {} if json_data is None else json_data
    response.raise_for_status = MagicMock()
    response.headers = {}
    return response


def _error_response(status_code: int, detail: str = "boom") -> MagicMock:
    response = _mock_response(status_code, {"detail": detail})
    response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "error",
        request=httpx.Request("POST", "https://test.com/x"),
        response=response,
    )
    return response


def _pubkey_json(pubkey: str, *, status: str = "active", pk_id: str | None = None) -> dict:
    return {
        "id": pk_id or str(uuid.uuid4()),
        "identity_id": str(uuid.uuid4()),
        "pubkey": pubkey,
        "fingerprint": crypto.fingerprint(pubkey),
        "label": "laptop",
        "status": status,
        "created_at": "2026-06-29T00:00:00Z",
        "attested_at": None,
        "revoked_at": None,
    }


def _client() -> SecretClient:
    return SecretClient(api_key="test", base_url="https://test.com")


# --- construction parity ----------------------------------------------------


def test_rejects_http_url():
    with pytest.raises(ValueError, match="must use HTTPS"):
        SecretClient(api_key="t", base_url="http://evil.com")


def test_api_key_not_on_instance():
    client = _client()
    assert not hasattr(client, "api_key")


def test_from_mcp_url_strips_mcp_suffix():
    client = SecretClient.from_mcp_url(api_key="t", mcp_url="https://memory.kagura-ai.com/mcp")
    assert client.base_url == "https://memory.kagura-ai.com"


# --- pubkey endpoints -------------------------------------------------------


@pytest.mark.asyncio
async def test_register_pubkey():
    client = _client()
    _, recipient = crypto.generate_keypair()
    resp = _mock_response(201, _pubkey_json(recipient, status="pending"))
    with patch.object(client._client, "request", new_callable=AsyncMock) as req:
        req.return_value = resp
        result = await client.register_pubkey(recipient, label="laptop")
    assert isinstance(result, PubkeyResponse)
    assert result.pubkey == recipient
    method, url = req.call_args[0][0], req.call_args[0][1]
    assert method == "POST"
    assert url.endswith("/api/v1/config/secrets/pubkeys")
    assert req.call_args.kwargs["json"] == {"pubkey": recipient, "label": "laptop"}
    await client.close()


@pytest.mark.asyncio
async def test_list_pubkeys_returns_list():
    client = _client()
    _, r1 = crypto.generate_keypair()
    _, r2 = crypto.generate_keypair()
    resp = _mock_response(200, [_pubkey_json(r1), _pubkey_json(r2, status="pending")])
    with patch.object(client._client, "request", new_callable=AsyncMock) as req:
        req.return_value = resp
        result = await client.list_pubkeys()
    assert [p.pubkey for p in result] == [r1, r2]
    assert req.call_args[0][1].endswith("/api/v1/config/secrets/pubkeys")
    await client.close()


@pytest.mark.asyncio
async def test_list_my_pubkeys_hits_me_endpoint():
    client = _client()
    _, r1 = crypto.generate_keypair()
    resp = _mock_response(200, [_pubkey_json(r1)])
    with patch.object(client._client, "request", new_callable=AsyncMock) as req:
        req.return_value = resp
        await client.list_my_pubkeys()
    assert req.call_args[0][1].endswith("/api/v1/config/secrets/pubkeys/me")
    await client.close()


@pytest.mark.asyncio
async def test_approve_pubkey():
    client = _client()
    _, r1 = crypto.generate_keypair()
    pk_id = str(uuid.uuid4())
    resp = _mock_response(200, _pubkey_json(r1, pk_id=pk_id))
    with patch.object(client._client, "request", new_callable=AsyncMock) as req:
        req.return_value = resp
        result = await client.approve_pubkey(pk_id)
    assert result.id == pk_id
    method, url = req.call_args[0][0], req.call_args[0][1]
    assert method == "POST"
    assert url.endswith(f"/api/v1/config/secrets/pubkeys/{pk_id}/approve")
    await client.close()


@pytest.mark.asyncio
async def test_revoke_pubkey():
    client = _client()
    _, r1 = crypto.generate_keypair()
    pk_id = str(uuid.uuid4())
    resp = _mock_response(200, _pubkey_json(r1, status="revoked", pk_id=pk_id))
    with patch.object(client._client, "request", new_callable=AsyncMock) as req:
        req.return_value = resp
        await client.revoke_pubkey(pk_id)
    assert req.call_args[0][1].endswith(f"/api/v1/config/secrets/pubkeys/{pk_id}/revoke")
    await client.close()


# --- secret endpoints -------------------------------------------------------


@pytest.mark.asyncio
async def test_put_secret_sends_contract_body():
    client = _client()
    ids = [str(uuid.uuid4())]
    resp = _mock_response(
        201, {"name": "db", "version_number": 1, "status": "active", "rotation_needed": False}
    )
    with patch.object(client._client, "request", new_callable=AsyncMock) as req:
        req.return_value = resp
        result = await client.put_secret(
            name="db",
            ciphertext="-----BEGIN AGE ENCRYPTED FILE-----\nx\n-----END AGE ENCRYPTED FILE-----\n",
            recipients_snapshot=["fp1"],
            grant_pubkey_ids=ids,
        )
    assert isinstance(result, SecretPutResponse)
    assert result.version_number == 1
    body = req.call_args.kwargs["json"]
    assert body["name"] == "db"
    assert body["recipients_snapshot"] == ["fp1"]
    assert body["grant_pubkey_ids"] == ids
    assert req.call_args[0][1].endswith("/api/v1/config/secrets")
    await client.close()


def test_secret_meta_response_allows_null_timestamps():
    # The live server returns updated_at/created_at as null for some secrets
    # (caught by live E2E); the model must accept that, not 500 the SDK.
    m = SecretMetaResponse.model_validate(
        {
            "name": "db",
            "status": "active",
            "rotation_needed": False,
            "current_version": 1,
            "grant_count": 1,
            "created_at": None,
            "updated_at": None,
        }
    )
    assert m.created_at is None
    assert m.updated_at is None


@pytest.mark.asyncio
async def test_list_secrets_returns_list():
    client = _client()
    resp = _mock_response(
        200,
        [
            {
                "name": "db",
                "status": "active",
                "rotation_needed": False,
                "current_version": 2,
                "grant_count": 3,
                "created_at": "2026-06-29T00:00:00Z",
                "updated_at": "2026-06-29T00:00:00Z",
            }
        ],
    )
    with patch.object(client._client, "request", new_callable=AsyncMock) as req:
        req.return_value = resp
        result = await client.list_secrets()
    assert isinstance(result[0], SecretMetaResponse)
    assert result[0].grant_count == 3
    await client.close()


@pytest.mark.asyncio
async def test_fetch_secret_carries_name_in_body():
    client = _client()
    resp = _mock_response(
        200,
        {
            "name": "db",
            "version_number": 1,
            "alg": "age",
            "ciphertext": "armored-ct",
            "blob_ref": None,
            "recipients_snapshot": ["fp1"],
            "rotation_needed": False,
            "created_at": "2026-06-29T00:00:00Z",
        },
    )
    with patch.object(client._client, "request", new_callable=AsyncMock) as req:
        req.return_value = resp
        result = await client.fetch_secret("db")
    assert isinstance(result, SecretValueResponse)
    assert result.alg == "age"
    method, url = req.call_args[0][0], req.call_args[0][1]
    assert method == "POST"
    assert url.endswith("/api/v1/config/secrets/fetch")
    assert req.call_args.kwargs["json"] == {"name": "db"}
    await client.close()


@pytest.mark.asyncio
async def test_fetch_secret_pins_version():
    client = _client()
    resp = _mock_response(
        200,
        {
            "name": "db",
            "version_number": 5,
            "alg": "age",
            "ciphertext": "x",
            "blob_ref": None,
            "recipients_snapshot": [],
            "rotation_needed": False,
            "created_at": "2026-06-29T00:00:00Z",
        },
    )
    with patch.object(client._client, "request", new_callable=AsyncMock) as req:
        req.return_value = resp
        await client.fetch_secret("db", version_number=5)
    assert req.call_args.kwargs["json"] == {"name": "db", "version_number": 5}
    await client.close()


@pytest.mark.asyncio
async def test_revoke_grant():
    client = _client()
    pk_id = str(uuid.uuid4())
    resp = _mock_response(
        200,
        {
            "name": "db",
            "status": "active",
            "rotation_needed": True,
            "current_version": 2,
            "grant_count": 2,
            "created_at": "2026-06-29T00:00:00Z",
            "updated_at": "2026-06-29T00:00:00Z",
        },
    )
    with patch.object(client._client, "request", new_callable=AsyncMock) as req:
        req.return_value = resp
        result = await client.revoke_grant("db", pk_id)
    assert result.rotation_needed is True
    assert req.call_args.kwargs["json"] == {"name": "db", "recipient_pubkey_id": pk_id}
    assert req.call_args[0][1].endswith("/api/v1/config/secrets/revoke-grant")
    await client.close()


@pytest.mark.asyncio
async def test_delete_secret_success():
    client = _client()
    resp = _mock_response(204, None)
    with patch.object(client._client, "request", new_callable=AsyncMock) as req:
        req.return_value = resp
        result = await client.delete_secret("db")
    assert result is None
    method, url = req.call_args[0][0], req.call_args[0][1]
    assert method == "DELETE"
    assert url.endswith("/api/v1/config/secrets/db")
    await client.close()


@pytest.mark.asyncio
async def test_delete_secret_encodes_slash_name_segmentwise():
    # Slash-containing names must stay addressable: the '/' separators are
    # structural (the server uses a {name:path} converter), but each segment
    # is percent-encoded, so a space inside a segment becomes %20.
    client = _client()
    resp = _mock_response(204, None)
    with patch.object(client._client, "request", new_callable=AsyncMock) as req:
        req.return_value = resp
        await client.delete_secret("cloudflare/api token")
    url = req.call_args[0][1]
    assert url.endswith("/api/v1/config/secrets/cloudflare/api%20token")
    await client.close()


@pytest.mark.asyncio
async def test_delete_secret_missing_raises_not_found():
    client = _client()
    with patch.object(client._client, "request", new_callable=AsyncMock) as req:
        req.return_value = _error_response(404, "secret not found")
        with pytest.raises(KaguraNotFoundError):
            await client.delete_secret("nope")
    await client.close()


@pytest.mark.asyncio
async def test_delete_secret_non_owner_raises_access_denied():
    # The delete is owner-only: a non-owner (incl. a non-owner admin) gets 403.
    client = _client()
    with patch.object(client._client, "request", new_callable=AsyncMock) as req:
        req.return_value = _error_response(403, "owner only")
        with pytest.raises(KaguraConnectionError, match="(?i)access denied"):
            await client.delete_secret("db")
    await client.close()


@pytest.mark.asyncio
async def test_verify_audit():
    client = _client()
    resp = _mock_response(
        200, {"valid": True, "entries": 10, "head": "abc", "broken_at": None, "reason": None}
    )
    with patch.object(client._client, "request", new_callable=AsyncMock) as req:
        req.return_value = resp
        result = await client.verify_audit()
    assert result.valid is True
    assert result.entries == 10
    assert req.call_args[0][1].endswith("/api/v1/config/secrets/audit/verify")
    await client.close()


# --- high-level orchestration (HARD invariant #2: set-equality) -------------


@pytest.mark.asyncio
async def test_put_secret_for_recipients_roundtrips_and_matches_sets():
    client = _client()
    id1, r1 = crypto.generate_keypair()
    id2, r2 = crypto.generate_keypair()
    pk1 = PubkeyResponse.model_validate(_pubkey_json(r1))
    pk2 = PubkeyResponse.model_validate(_pubkey_json(r2))
    resp = _mock_response(
        201, {"name": "db", "version_number": 1, "status": "active", "rotation_needed": False}
    )
    with patch.object(client._client, "request", new_callable=AsyncMock) as req:
        req.return_value = resp
        await client.put_secret_for_recipients("db", b"hunter2", [pk1, pk2])
        body = req.call_args.kwargs["json"]

    # set-equality: snapshot fingerprints == fingerprints of granted pubkeys
    assert set(body["recipients_snapshot"]) == {pk1.fingerprint, pk2.fingerprint}
    assert set(body["grant_pubkey_ids"]) == {pk1.id, pk2.id}
    # the ciphertext actually decrypts for each recipient
    assert crypto.decrypt(body["ciphertext"], id1) == b"hunter2"
    assert crypto.decrypt(body["ciphertext"], id2) == b"hunter2"
    await client.close()


@pytest.mark.asyncio
async def test_put_secret_for_recipients_rejects_non_active():
    client = _client()
    _, r1 = crypto.generate_keypair()
    pending = PubkeyResponse.model_validate(_pubkey_json(r1, status="pending"))
    with pytest.raises(KaguraSecretError, match="active"):
        await client.put_secret_for_recipients("db", b"x", [pending])
    await client.close()


@pytest.mark.asyncio
async def test_put_secret_for_recipients_rejects_empty():
    client = _client()
    with pytest.raises(KaguraSecretError):
        await client.put_secret_for_recipients("db", b"x", [])
    await client.close()


@pytest.mark.asyncio
async def test_put_secret_for_recipients_detects_fingerprint_mismatch():
    """A pubkey whose advertised fingerprint != sha256(pubkey) is rejected."""
    client = _client()
    _, r1 = crypto.generate_keypair()
    bad = _pubkey_json(r1)
    bad["fingerprint"] = "0" * 64  # tampered / inconsistent listing
    pk = PubkeyResponse.model_validate(bad)
    with pytest.raises(KaguraSecretError, match="fingerprint"):
        await client.put_secret_for_recipients("db", b"x", [pk])
    await client.close()


# --- error mapping ----------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_unknown_secret_raises_not_found():
    client = _client()
    with patch.object(client._client, "request", new_callable=AsyncMock) as req:
        req.return_value = _error_response(404, "secret not found")
        with pytest.raises(KaguraNotFoundError):
            await client.fetch_secret("nope")
    await client.close()


@pytest.mark.asyncio
async def test_401_raises_auth_error():
    client = _client()
    with patch.object(client._client, "request", new_callable=AsyncMock) as req:
        req.return_value = _error_response(401)
        with pytest.raises(KaguraAuthError):
            await client.list_secrets()
    await client.close()


@pytest.mark.asyncio
async def test_403_maps_to_clear_access_denied():
    # The live server returns 403 (not 404) for a secret the caller can't read
    # — hide existence. Surface an actionable message, not a bare "HTTP 403".
    client = _client()
    with patch.object(client._client, "request", new_callable=AsyncMock) as req:
        req.return_value = _error_response(403, "")
        with pytest.raises(KaguraConnectionError, match="(?i)access denied"):
            await client.fetch_secret("foo")
    await client.close()


@pytest.mark.asyncio
async def test_403_appends_server_detail():
    client = _client()
    with patch.object(client._client, "request", new_callable=AsyncMock) as req:
        req.return_value = _error_response(403, "no grant for caller")
        with pytest.raises(KaguraConnectionError, match="no grant for caller"):
            await client.fetch_secret("foo")
    await client.close()


@pytest.mark.asyncio
async def test_generic_http_error_maps_to_connection_error():
    client = _client()
    with patch.object(client._client, "request", new_callable=AsyncMock) as req:
        req.return_value = _error_response(500, "internal error")
        with pytest.raises(KaguraConnectionError):
            await client.list_secrets()
    await client.close()


@pytest.mark.asyncio
async def test_request_error_maps_to_connection_error():
    client = _client()
    with patch.object(client._client, "request", new_callable=AsyncMock) as req:
        req.side_effect = httpx.ConnectError("network down")
        with pytest.raises(KaguraConnectionError):
            await client.list_secrets()
    await client.close()


# --- construction edges -----------------------------------------------------


def test_requires_credentials():
    with pytest.raises(ValueError, match="requires api_key"):
        SecretClient()


def test_rejects_http_base_url():
    with pytest.raises(ValueError, match="HTTPS"):
        SecretClient(api_key="t", base_url="http://insecure.example.com")


@pytest.mark.asyncio
async def test_oauth_construction_via_resolved_auth():
    resolved = _OAuthAuth(
        oauth=httpx.BasicAuth("u", "p"),  # stand-in httpx.Auth for the OAuth handler
        mcp_url="https://memory.kagura-ai.com/mcp",
        workspace_id="ws-uuid",
    )
    client = SecretClient._from_resolved_auth(resolved)
    assert client.base_url == "https://memory.kagura-ai.com"
    assert client._oauth is not None
    await client.close()


@pytest.mark.asyncio
async def test_async_context_manager_closes():
    async with SecretClient(api_key="t", base_url="https://test.com") as c:
        assert c.base_url == "https://test.com"


@pytest.mark.asyncio
async def test_429_stays_generic_connection_error():
    """#229 contract: the secret surface keeps its historical 429 mapping.

    Unlike the other REST clients (KaguraQuotaError), SecretClient has
    always rendered 429 through the generic branch — pin that so the
    shared-base refactor cannot silently change the exception type.
    """
    client = SecretClient(api_key="kagura_test")
    resp = httpx.Response(
        429,
        json={"detail": "slow down"},
        request=httpx.Request("GET", "https://x/api/v1/config/secrets"),
    )
    with patch.object(
        client._client,
        "request",
        AsyncMock(side_effect=httpx.HTTPStatusError("429", request=resp.request, response=resp)),
    ):
        with pytest.raises(KaguraConnectionError, match="HTTP 429: slow down"):
            await client.list_secrets()
    await client.close()
