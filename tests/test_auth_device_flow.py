"""Tests for kagura_memory.auth.device_flow (RFC 8628 async functions)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from kagura_memory.auth.device_flow import (
    DEFAULT_CLIENT_ID,
    authorize_device,
    make_oauth_client,
    poll_for_token,
    refresh_access_token,
    revoke_token,
)
from kagura_memory.exceptions import (
    KaguraAuthDeniedError,
    KaguraAuthError,
    KaguraAuthExpiredError,
    KaguraConnectionError,
)

SERVER = "https://test.example.com"


def _mock_response(status: int, body: dict | None = None, text: str = "") -> MagicMock:
    """Build a MagicMock that quacks like httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status
    resp.json = MagicMock(return_value=body or {})
    resp.text = text
    if status >= 400:
        resp.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError("err", request=MagicMock(), response=resp)
        )
    else:
        resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# make_oauth_client — no Bearer header (key isolation)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_make_oauth_client_has_no_authorization_header():
    async with make_oauth_client() as client:
        assert "Authorization" not in client.headers


# ---------------------------------------------------------------------------
# authorize_device
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_authorize_device_happy_path():
    client = MagicMock()
    client.post = AsyncMock(
        return_value=_mock_response(
            200,
            {
                "device_code": "dc-1",
                "user_code": "ABCD-1234",
                "verification_uri": "https://test.example.com/device",
                "verification_uri_complete": "https://test.example.com/device?user_code=ABCD-1234",
                "expires_in": 600,
                "interval": 5,
            },
        )
    )

    da = await authorize_device(client, SERVER, scope="memory:read")
    assert da.user_code == "ABCD-1234"
    assert da.device_code == "dc-1"
    assert da.interval == 5
    # The POST body uses application/x-www-form-urlencoded data.
    client.post.assert_called_once()
    posted = client.post.call_args
    assert posted.kwargs["data"]["client_id"] == DEFAULT_CLIENT_ID
    assert posted.kwargs["data"]["scope"] == "memory:read"


@pytest.mark.asyncio
async def test_authorize_device_http_error_wraps_as_auth_error():
    client = MagicMock()
    response = _mock_response(400, {"detail": "invalid_client"})
    client.post = AsyncMock(return_value=response)

    with pytest.raises(KaguraAuthError, match="Device authorization failed"):
        await authorize_device(client, SERVER, scope="memory:read")


@pytest.mark.asyncio
async def test_authorize_device_network_error_wraps_as_connection_error():
    client = MagicMock()
    client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))

    with pytest.raises(KaguraConnectionError, match="Could not reach"):
        await authorize_device(client, SERVER, scope="memory:read")


# ---------------------------------------------------------------------------
# poll_for_token
# ---------------------------------------------------------------------------


async def _no_sleep(_seconds: float) -> None:
    """No-op sleep stub for tests."""


@pytest.mark.asyncio
async def test_poll_for_token_happy_path():
    client = MagicMock()
    client.post = AsyncMock(
        return_value=_mock_response(
            200,
            {
                "access_token": "atok",
                "refresh_token": "rtok",
                "token_type": "Bearer",
                "expires_in": 3600,
                "scope": "memory:read",
                "user_email": "u@example.com",
                "workspace_id": "ws-1",
                "workspace_name": "ws",
            },
        )
    )

    token = await poll_for_token(
        client,
        SERVER,
        client_id=DEFAULT_CLIENT_ID,
        device_code="dc-1",
        interval=5,
        expires_at=datetime.now(UTC) + timedelta(minutes=10),
        sleep=_no_sleep,
    )
    assert token.access_token == "atok"
    assert token.refresh_token == "rtok"
    assert token.user_email == "u@example.com"
    assert token.expires_at > datetime.now(UTC)


@pytest.mark.asyncio
async def test_poll_for_token_authorization_pending_then_success():
    client = MagicMock()
    client.post = AsyncMock(
        side_effect=[
            _mock_response(400, {"error": "authorization_pending"}),
            _mock_response(400, {"error": "authorization_pending"}),
            _mock_response(
                200,
                {
                    "access_token": "atok",
                    "refresh_token": "rtok",
                    "token_type": "Bearer",
                    "expires_in": 3600,
                    "scope": "memory:read",
                },
            ),
        ]
    )
    token = await poll_for_token(
        client,
        SERVER,
        client_id=DEFAULT_CLIENT_ID,
        device_code="dc",
        interval=5,
        expires_at=datetime.now(UTC) + timedelta(minutes=10),
        sleep=_no_sleep,
    )
    assert token.access_token == "atok"
    assert client.post.await_count == 3


@pytest.mark.asyncio
async def test_poll_for_token_slow_down_increases_interval():
    """A ``slow_down`` response should add 5s to the polling interval."""
    intervals: list[float] = []

    async def record_sleep(seconds: float) -> None:
        intervals.append(seconds)

    client = MagicMock()
    client.post = AsyncMock(
        side_effect=[
            _mock_response(400, {"error": "slow_down"}),
            _mock_response(
                200,
                {
                    "access_token": "atok",
                    "refresh_token": "rtok",
                    "token_type": "Bearer",
                    "expires_in": 3600,
                    "scope": "memory:read",
                },
            ),
        ]
    )
    await poll_for_token(
        client,
        SERVER,
        client_id=DEFAULT_CLIENT_ID,
        device_code="dc",
        interval=5,
        expires_at=datetime.now(UTC) + timedelta(minutes=10),
        sleep=record_sleep,
    )
    # First poll sleeps the original interval (5), second poll sleeps +5 = 10.
    assert intervals == [5, 10]


@pytest.mark.asyncio
async def test_poll_for_token_access_denied_raises_denied_error():
    client = MagicMock()
    client.post = AsyncMock(return_value=_mock_response(400, {"error": "access_denied"}))

    with pytest.raises(KaguraAuthDeniedError, match="Authorization denied"):
        await poll_for_token(
            client,
            SERVER,
            client_id=DEFAULT_CLIENT_ID,
            device_code="dc",
            interval=5,
            expires_at=datetime.now(UTC) + timedelta(minutes=10),
            sleep=_no_sleep,
        )


@pytest.mark.asyncio
async def test_poll_for_token_expired_token_raises_expired_error():
    client = MagicMock()
    client.post = AsyncMock(return_value=_mock_response(400, {"error": "expired_token"}))

    with pytest.raises(KaguraAuthExpiredError):
        await poll_for_token(
            client,
            SERVER,
            client_id=DEFAULT_CLIENT_ID,
            device_code="dc",
            interval=5,
            expires_at=datetime.now(UTC) + timedelta(minutes=10),
            sleep=_no_sleep,
        )


@pytest.mark.asyncio
async def test_poll_for_token_expires_before_approval():
    """Past ``expires_at`` cuts off the loop and raises expired."""
    client = MagicMock()
    client.post = AsyncMock(return_value=_mock_response(400, {"error": "authorization_pending"}))

    with pytest.raises(KaguraAuthExpiredError):
        await poll_for_token(
            client,
            SERVER,
            client_id=DEFAULT_CLIENT_ID,
            device_code="dc",
            interval=5,
            expires_at=datetime.now(UTC) - timedelta(seconds=1),
            sleep=_no_sleep,
        )


@pytest.mark.asyncio
async def test_poll_for_token_network_error_during_poll():
    client = MagicMock()
    client.post = AsyncMock(side_effect=httpx.ConnectError("drop"))

    with pytest.raises(KaguraConnectionError, match="Lost connection"):
        await poll_for_token(
            client,
            SERVER,
            client_id=DEFAULT_CLIENT_ID,
            device_code="dc",
            interval=5,
            expires_at=datetime.now(UTC) + timedelta(minutes=10),
            sleep=_no_sleep,
        )


# ---------------------------------------------------------------------------
# refresh_access_token
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refresh_happy_path():
    client = MagicMock()
    client.post = AsyncMock(
        return_value=_mock_response(
            200,
            {
                "access_token": "atok-new",
                "refresh_token": "rtok-new",
                "token_type": "Bearer",
                "expires_in": 3600,
                "scope": "memory:read",
            },
        )
    )
    token = await refresh_access_token(
        client,
        SERVER,
        client_id=DEFAULT_CLIENT_ID,
        refresh_token="rtok-old",
    )
    assert token.access_token == "atok-new"
    assert token.refresh_token == "rtok-new"
    posted = client.post.call_args
    assert posted.kwargs["data"]["grant_type"] == "refresh_token"
    assert "scope" not in posted.kwargs["data"]


@pytest.mark.asyncio
async def test_refresh_with_scope_includes_scope_param():
    client = MagicMock()
    client.post = AsyncMock(
        return_value=_mock_response(
            200,
            {
                "access_token": "atok",
                "refresh_token": "rtok",
                "token_type": "Bearer",
                "expires_in": 3600,
                "scope": "memory:read memory:write",
            },
        )
    )
    token = await refresh_access_token(
        client,
        SERVER,
        client_id=DEFAULT_CLIENT_ID,
        refresh_token="rtok-old",
        scope="memory:read memory:write",
    )
    assert token.scope == "memory:read memory:write"
    posted = client.post.call_args
    assert posted.kwargs["data"]["scope"] == "memory:read memory:write"


@pytest.mark.asyncio
async def test_refresh_invalid_grant_raises_expired():
    client = MagicMock()
    client.post = AsyncMock(return_value=_mock_response(400, {"error": "invalid_grant"}))

    with pytest.raises(KaguraAuthExpiredError, match="Your login expired"):
        await refresh_access_token(
            client, SERVER, client_id=DEFAULT_CLIENT_ID, refresh_token="rtok-old"
        )


@pytest.mark.asyncio
async def test_refresh_insufficient_scope_raises_auth_error():
    """``insufficient_scope`` → generic KaguraAuthError so CLI can detect+retry."""
    client = MagicMock()
    client.post = AsyncMock(return_value=_mock_response(400, {"error": "insufficient_scope"}))

    with pytest.raises(KaguraAuthError, match="insufficient_scope"):
        await refresh_access_token(
            client,
            SERVER,
            client_id=DEFAULT_CLIENT_ID,
            refresh_token="rtok-old",
            scope="memory:write",
        )


# ---------------------------------------------------------------------------
# revoke_token (best-effort)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_revoke_token_returns_true_on_200():
    client = MagicMock()
    client.post = AsyncMock(return_value=_mock_response(200))
    assert await revoke_token(client, SERVER, token="atok") is True


@pytest.mark.asyncio
async def test_revoke_token_returns_false_on_5xx():
    client = MagicMock()
    client.post = AsyncMock(return_value=_mock_response(500))
    assert await revoke_token(client, SERVER, token="atok") is False


@pytest.mark.asyncio
async def test_revoke_token_returns_false_on_network_failure():
    """Revoke is best-effort — network failure must not raise."""
    client = MagicMock()
    client.post = AsyncMock(side_effect=httpx.ConnectError("drop"))
    assert await revoke_token(client, SERVER, token="atok") is False
