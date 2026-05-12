"""OAuth2 RFC 8628 device authorization grant — stateless async helpers.

Pure-function API: every entry point takes an :class:`httpx.AsyncClient`
(use :func:`make_oauth_client` to construct one) plus the relevant
parameters, and returns a dataclass. No CLI, no terminal IO, no global
state — that lives in :mod:`auth.cli` and :mod:`auth.credentials`.

The transport client is intentionally separate from
:class:`KaguraClient`'s own ``httpx.AsyncClient`` so the SDK's
``Authorization: Bearer`` header (for normal MCP calls) cannot leak
into ``/oauth2/*`` requests, which use ``client_id`` body parameter
authentication (RFC 8628 §3.1 ``token_endpoint_auth_method='none'``).
This mirrors :class:`FilesClient._upload_client`'s "isolate secrets to
a dedicated client" idiom.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from .._http import SDK_VERSION, extract_detail
from ..exceptions import (
    KaguraAuthDeniedError,
    KaguraAuthError,
    KaguraAuthExpiredError,
    KaguraConnectionError,
)

# OAuth2 endpoint paths under {server}
_PATH_DEVICE_AUTHORIZE = "/api/v1/oauth2/device/authorize"
_PATH_TOKEN = "/api/v1/oauth2/token"
_PATH_REVOKE = "/api/v1/oauth2/revoke"

# RFC 8628 §3.5 — "slow_down" requires the client to add 5 seconds.
_SLOW_DOWN_INCREMENT_SEC = 5

DEFAULT_CLIENT_ID = "kagura-cli"
"""The pre-registered public client ID seeded by memory-cloud #624."""

DEVICE_FLOW_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"
REFRESH_TOKEN_GRANT_TYPE = "refresh_token"


# ---------------------------------------------------------------------------
# Response dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DeviceAuthorizationResponse:
    """RFC 8628 §3.2 device authorization response."""

    device_code: str
    user_code: str
    verification_uri: str
    verification_uri_complete: str
    expires_in: int
    interval: int
    expires_at: datetime  # = now + expires_in


@dataclass
class TokenResponse:
    """RFC 8628 §3.5 / RFC 6749 §5.1 successful token response.

    ``expires_at`` is computed once at receipt time so a paused or
    suspended laptop never sees a negative TTL after wake.
    """

    access_token: str
    refresh_token: str
    token_type: str
    expires_at: datetime
    scope: str
    user_email: str = ""
    workspace_id: str = ""
    workspace_name: str = ""


# ---------------------------------------------------------------------------
# Transport client factory
# ---------------------------------------------------------------------------


def make_oauth_client(timeout: float = 30.0) -> httpx.AsyncClient:
    """Construct an unauthenticated ``httpx.AsyncClient`` for ``/oauth2/*``.

    No ``Authorization`` header is set — device-flow uses ``client_id``
    in the form body for client authentication, not a bearer token.
    """
    return httpx.AsyncClient(
        timeout=timeout,
        headers={"User-Agent": f"kagura-memory-sdk/{SDK_VERSION}"},
    )


# ---------------------------------------------------------------------------
# Public flow functions
# ---------------------------------------------------------------------------


async def authorize_device(
    client: httpx.AsyncClient,
    server: str,
    *,
    client_id: str = DEFAULT_CLIENT_ID,
    scope: str = "memory:read",
) -> DeviceAuthorizationResponse:
    """POST ``/api/v1/oauth2/device/authorize`` and parse the response."""
    url = f"{server.rstrip('/')}{_PATH_DEVICE_AUTHORIZE}"
    data = {"client_id": client_id, "scope": scope}

    try:
        response = await client.post(url, data=data)
        response.raise_for_status()
        body = response.json()
    except httpx.HTTPStatusError as e:
        detail = extract_detail(e.response) or e.response.text
        raise KaguraAuthError(
            f"Device authorization failed (HTTP {e.response.status_code}): {detail}\n"
            f"  Verify the server URL and that '{client_id}' is registered."
        ) from e
    except httpx.RequestError as e:
        raise KaguraConnectionError(f"Could not reach {url}: {e}") from e

    return DeviceAuthorizationResponse(
        device_code=body["device_code"],
        user_code=body["user_code"],
        verification_uri=body["verification_uri"],
        verification_uri_complete=body.get("verification_uri_complete", body["verification_uri"]),
        expires_in=int(body["expires_in"]),
        interval=int(body.get("interval", 5)),
        expires_at=datetime.now(UTC) + timedelta(seconds=int(body["expires_in"])),
    )


async def poll_for_token(
    client: httpx.AsyncClient,
    server: str,
    *,
    client_id: str,
    device_code: str,
    interval: int,
    expires_at: datetime,
    sleep: Any = asyncio.sleep,
) -> TokenResponse:
    """Poll ``/api/v1/oauth2/token`` until the user approves or denies.

    The ``sleep`` parameter is injectable so tests can supply a no-op
    or counter-based stub without waiting real seconds.

    Raises:
        KaguraAuthDeniedError: user clicked "Deny" at the consent screen
            (server returns ``access_denied``).
        KaguraAuthExpiredError: the ``device_code`` lifetime elapsed
            without approval (server returns ``expired_token``).
        KaguraAuthError: any other OAuth error or unexpected response.
        KaguraConnectionError: network failure during polling.
    """
    current_interval = interval

    while True:
        if datetime.now(UTC) >= expires_at:
            raise KaguraAuthExpiredError(
                "Device code expired before user approval. Run: kagura auth login",
                expires_at=expires_at,
            )

        await sleep(current_interval)

        try:
            response = await client.post(
                f"{server.rstrip('/')}{_PATH_TOKEN}",
                data={
                    "grant_type": DEVICE_FLOW_GRANT_TYPE,
                    "device_code": device_code,
                    "client_id": client_id,
                },
            )
        except httpx.RequestError as e:
            raise KaguraConnectionError(
                f"Lost connection while waiting for approval: {e}\n"
                f"  The login session may still be valid; re-run: kagura auth login"
            ) from e

        if response.status_code == 200:
            return _token_response_from_body(response.json())

        # RFC 8628 §3.5 — errors come as HTTP 4xx with JSON ``error`` field.
        body = _safe_json(response)
        error = body.get("error", "")

        if error == "authorization_pending":
            continue
        if error == "slow_down":
            current_interval += _SLOW_DOWN_INCREMENT_SEC
            continue
        if error == "access_denied":
            raise KaguraAuthDeniedError(
                "Authorization denied at the consent screen.\n"
                "  Re-run: kagura auth login\n"
                "  To use a different workspace, log in with that account "
                "in your browser first."
            )
        if error == "expired_token":
            raise KaguraAuthExpiredError(
                "Device code expired before user approval. Run: kagura auth login",
                expires_at=expires_at,
            )

        # Unknown error — surface what we got.
        description = body.get("error_description", "")
        raise KaguraAuthError(f"Token endpoint returned unexpected error '{error}': {description}")


async def refresh_access_token(
    client: httpx.AsyncClient,
    server: str,
    *,
    client_id: str,
    refresh_token: str,
    scope: str | None = None,
) -> TokenResponse:
    """POST ``/api/v1/oauth2/token`` with ``grant_type=refresh_token``.

    When ``scope`` is supplied, the server may reject the call with
    ``insufficient_scope`` / ``invalid_scope`` if the grant doesn't
    cover it — the CLI catches that and re-runs the full device flow
    for incremental consent.

    Raises:
        KaguraAuthExpiredError: refresh token is invalid or expired
            (server returns ``invalid_grant``).
        KaguraAuthError: any other OAuth error.
        KaguraConnectionError: network failure.
    """
    url = f"{server.rstrip('/')}{_PATH_TOKEN}"
    data: dict[str, str] = {
        "grant_type": REFRESH_TOKEN_GRANT_TYPE,
        "refresh_token": refresh_token,
        "client_id": client_id,
    }
    if scope is not None:
        data["scope"] = scope

    try:
        response = await client.post(url, data=data)
    except httpx.RequestError as e:
        raise KaguraConnectionError(f"Could not reach {url}: {e}") from e

    if response.status_code == 200:
        return _token_response_from_body(response.json())

    body = _safe_json(response)
    error = body.get("error", "")

    if error == "invalid_grant":
        raise KaguraAuthExpiredError(
            "Your login expired (refresh token is no longer valid).\n"
            "  Run: kagura auth login\n"
            "  Your server and workspace selection are preserved."
        )

    description = body.get("error_description", "")
    raise KaguraAuthError(f"Refresh failed: {error}{f' — {description}' if description else ''}")


async def revoke_token(
    client: httpx.AsyncClient,
    server: str,
    *,
    token: str,
    client_id: str = DEFAULT_CLIENT_ID,
) -> bool:
    """POST ``/api/v1/oauth2/revoke``. Best-effort — never raises.

    Returns ``True`` on success, ``False`` on any failure. The caller
    (``kagura auth logout``) deletes the local profile regardless of
    the return value, on the principle that local logout must succeed
    even when the server is unreachable.
    """
    url = f"{server.rstrip('/')}{_PATH_REVOKE}"
    try:
        response = await client.post(
            url,
            data={"token": token, "client_id": client_id},
        )
        return response.status_code in (200, 204)
    except httpx.RequestError:
        return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _token_response_from_body(body: dict[str, Any]) -> TokenResponse:
    """Build a :class:`TokenResponse` from the server's JSON body.

    ``expires_at`` is computed from ``expires_in`` at receipt time so
    laptop sleep / clock skew won't yield a negative TTL after wake.
    """
    expires_in = int(body.get("expires_in", 0))
    return TokenResponse(
        access_token=body["access_token"],
        refresh_token=body.get("refresh_token", ""),
        token_type=body.get("token_type", "Bearer"),
        expires_at=datetime.now(UTC) + timedelta(seconds=expires_in),
        scope=body.get("scope", ""),
        user_email=body.get("user_email", ""),
        workspace_id=body.get("workspace_id", ""),
        workspace_name=body.get("workspace_name", ""),
    )


def _safe_json(response: httpx.Response) -> dict[str, Any]:
    """Return ``response.json()`` as a dict, or ``{}`` if unparseable."""
    try:
        body = response.json()
    except (ValueError, UnicodeDecodeError):
        return {}
    return body if isinstance(body, dict) else {}
