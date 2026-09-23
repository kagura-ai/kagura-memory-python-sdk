"""OAuth2 RFC 8628 device authorization grant — stateless async helpers.

Pure-function API: every entry point takes an :class:`httpx.AsyncClient`
(use :func:`make_oauth_client` to construct one) plus the relevant
parameters, and returns a dataclass. No CLI, no terminal IO, no global
state — that lives in :mod:`auth.cli` and :mod:`auth.credentials`. The
``kagura auth login --invite`` helpers live here too: pure functions over
strings, plus one best-effort ``/system/info`` probe.

The transport client is intentionally separate from
:class:`KaguraClient`'s own ``httpx.AsyncClient`` so the SDK's
``Authorization: Bearer`` header (for normal MCP calls) cannot leak
into ``/oauth/*`` requests, which use ``client_id`` body parameter
authentication (RFC 8628 §3.1 ``token_endpoint_auth_method='none'``).
This mirrors :class:`FilesClient._upload_client`'s "isolate secrets to
a dedicated client" idiom.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from urllib.parse import quote, urlsplit

import httpx

from .._http import SDK_VERSION, _retry_after_seconds, extract_detail, validate_https_url
from ..exceptions import (
    KaguraAuthDeniedError,
    KaguraAuthError,
    KaguraAuthExpiredError,
    KaguraConnectionError,
    _exc_message,
)

# OAuth2 endpoint paths under {server}.
# The path prefix is /api/v1/oauth/ (NOT /oauth2/) per memory-cloud's
# actual mount point; the token endpoint requires the trailing slash.
_PATH_DEVICE_AUTHORIZE = "/api/v1/oauth/device/authorize"
_PATH_TOKEN = "/api/v1/oauth/token/"
_PATH_REVOKE = "/api/v1/oauth/revoke"
_PATH_SYSTEM_INFO = "/api/v1/system/info"

# RFC 8628 §3.5 — "slow_down" requires the client to add 5 seconds.
_SLOW_DOWN_INCREMENT_SEC = 5

# memory-cloud's per-IP device-flow window (memory-cloud#1667, v0.76.0): the
# wait to report when a 429 carries no usable Retry-After.
_DEVICE_RATE_LIMIT_RETRY_AFTER_SEC = 60

DEFAULT_CLIENT_ID = "kagura-cli"
"""The pre-registered public client ID seeded by memory-cloud #624."""

DEVICE_FLOW_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"
REFRESH_TOKEN_GRANT_TYPE = "refresh_token"

# memory-cloud's own invite-token shape (``BETA_INVITE_TOKEN_PATTERN`` in
# backend/src/services/beta_invite_service.py), minus the anchors: matched
# with ``fullmatch`` so a trailing newline cannot sneak past ``$``.
_INVITE_TOKEN_RE = re.compile(r"[A-Za-z0-9_-]{20,128}")
_INVITE_TOKEN_RULE = "20-128 characters from A-Z, a-z, 0-9, '_' and '-'"
_SEMVER_PREFIX_RE = re.compile(r"v?(\d+)\.(\d+)\.(\d+)")
_DEFAULT_PORTS = {"http": 80, "https": 443}

JOIN_RETURN_TO_MIN_SERVER_VERSION: tuple[int, int, int] = (0, 76, 0)
"""First memory-cloud release whose ``/join/<token>`` honours ``return_to``.

memory-cloud v0.76.0 ships the ``/join`` → ``/device`` hand-off
(memory-cloud#1655, merged as memory-cloud#1666). It came with no capability
flag, so the server version is the switch: from 0.76.0 :func:`invite_support`
answers ``"hand_off"``, and 0.75.x and older get the two-step fallback. The
link shape :func:`build_invite_link` builds is the one memory-cloud documents
in ``docs/deployment.md`` ("Invites and device or MCP sign-in").
"""

# The /system/info probe runs while the device code is already ticking, so a
# hung server must not eat into the user's approval window.
_SYSTEM_INFO_TIMEOUT_SEC = 5.0

InviteSupport = Literal["disabled", "hand_off", "two_step"]


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


@dataclass(frozen=True)
class InviteRef:
    """A parsed ``kagura auth login --invite`` value.

    The token is a bearer secret for one sign-up, so it and the link that
    carries it are kept out of ``repr`` (and therefore out of tracebacks).
    """

    token: str = field(repr=False)
    origin: str | None = None
    """``scheme://host[:port]`` of a full invite link; ``None`` for a bare token."""
    link: str | None = field(default=None, repr=False)
    """The invite link as given, minus query and fragment; ``None`` for a bare token."""


# ---------------------------------------------------------------------------
# Transport client factory
# ---------------------------------------------------------------------------


def make_oauth_client(timeout: float = 30.0) -> httpx.AsyncClient:
    """Construct an unauthenticated ``httpx.AsyncClient`` for ``/oauth/*`` and ``/system/info``.

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
    """POST ``/api/v1/oauth/device/authorize`` and parse the response.

    Raises:
        KaguraAuthError: the server refused the request. A 429 (memory-cloud
            v0.76.0+ limits this endpoint per client address) says how long
            to wait, from ``Retry-After``.
        KaguraConnectionError: network failure.
    """
    url = f"{server.rstrip('/')}{_PATH_DEVICE_AUTHORIZE}"
    # memory-cloud's device/authorize accepts JSON (DeviceAuthorizationRequest
    # pydantic model), unlike the /oauth/token/ + /oauth/revoke endpoints
    # which take application/x-www-form-urlencoded.
    body = {"client_id": client_id, "scope": scope}

    try:
        response = await client.post(url, json=body)
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            raise KaguraAuthError(_device_rate_limited_message(e.response)) from e
        detail = extract_detail(e.response) or e.response.text
        raise KaguraAuthError(
            f"Device authorization failed (HTTP {e.response.status_code}): {detail}\n"
            f"  Verify the server URL and that '{client_id}' is registered."
        ) from e
    except httpx.RequestError as e:
        raise KaguraConnectionError(f"Could not reach {url}: {_exc_message(e)}") from e

    body = _safe_json_object(response, "Device authorization")
    try:
        expires_in = int(body["expires_in"])
        return DeviceAuthorizationResponse(
            device_code=body["device_code"],
            user_code=body["user_code"],
            verification_uri=body["verification_uri"],
            verification_uri_complete=body.get(
                "verification_uri_complete", body["verification_uri"]
            ),
            expires_in=expires_in,
            interval=int(body.get("interval", 5)),
            expires_at=datetime.now(UTC) + timedelta(seconds=expires_in),
        )
    except (KeyError, TypeError, ValueError) as e:
        raise KaguraAuthError(
            f"Device authorization returned HTTP 200 but body is missing required fields: {e}. "
            f"Body keys: {sorted(body.keys())}"
        ) from e


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
    """Poll ``/api/v1/oauth/token/`` until the user approves or denies.

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
    first_poll = True

    while True:
        if datetime.now(UTC) >= expires_at:
            raise KaguraAuthExpiredError(
                "Device code expired before user approval. Run: kagura auth login",
                expires_at=expires_at,
            )

        # Skip the initial sleep so an immediate approval (or a fast
        # server-side error) doesn't wait one whole interval. Sleep
        # only between retries — after authorization_pending / slow_down.
        if first_poll:
            first_poll = False
        else:
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
                f"Lost connection while waiting for approval: {_exc_message(e)}\n"
                f"  The login session may still be valid; re-run: kagura auth login"
            ) from e

        if response.status_code == 200:
            return _token_response_from_response(response)

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

        # Unknown error — surface the HTTP status + raw response so the
        # operator can debug non-OAuth failures (HTML 5xx, proxy errors,
        # non-JSON bodies that make `error` come back empty).
        description = body.get("error_description", "")
        if error or description:
            raise KaguraAuthError(
                f"Token endpoint returned unexpected error '{error}': {description}"
            )
        detail = extract_detail(response) or response.text[:200]
        raise KaguraAuthError(
            f"Token endpoint returned HTTP {response.status_code} with no OAuth error code. "
            f"Body: {detail}"
        )


async def refresh_access_token(
    client: httpx.AsyncClient,
    server: str,
    *,
    client_id: str,
    refresh_token: str,
    scope: str | None = None,
) -> TokenResponse:
    """POST ``/api/v1/oauth/token/`` with ``grant_type=refresh_token``.

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
        raise KaguraConnectionError(f"Could not reach {url}: {_exc_message(e)}") from e

    if response.status_code == 200:
        return _token_response_from_response(response)

    body = _safe_json(response)
    error = body.get("error", "")

    if error == "invalid_grant":
        raise KaguraAuthExpiredError(
            "Your login expired (refresh token is no longer valid).\n"
            "  Run: kagura auth login\n"
            "  Your server and workspace selection are preserved."
        )

    description = body.get("error_description", "")
    if error:
        raise KaguraAuthError(
            f"Refresh failed: {error}{f' — {description}' if description else ''}"
        )
    # Non-OAuth failure (HTML 5xx, network proxy returning text/plain, etc.).
    detail = extract_detail(response) or response.text[:200]
    raise KaguraAuthError(
        f"Refresh failed: HTTP {response.status_code} with no OAuth error code. Body: {detail}"
    )


async def revoke_token(
    client: httpx.AsyncClient,
    server: str,
    *,
    token: str,
    client_id: str = DEFAULT_CLIENT_ID,
) -> bool:
    """POST ``/api/v1/oauth/revoke``. Best-effort — never raises.

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
# Invite sign-up (``kagura auth login --invite``)
# ---------------------------------------------------------------------------


def parse_invite(value: str) -> InviteRef:
    """Parse an invite link or a bare invite token, without any network call.

    A link is ``https://<host>[/<base path>]/join/<token>``: its query and
    fragment are ignored, and only the last two path segments must be
    ``join/<token>`` so a frontend served under a base path still parses.
    A bare token must match memory-cloud's pattern (20-128 of
    ``[A-Za-z0-9_-]``). Surrounding whitespace is ignored.

    Raises:
        ValueError: ``value`` is neither. The message never echoes the
            input, which may hold a (mistyped) token.
    """
    value = value.strip()
    if "://" not in value:
        if "/" in value:
            raise ValueError("an invite link must be a full https://<host>/join/<token> URL")
        _check_invite_token(value)
        return InviteRef(token=value)

    origin = _url_origin(value)
    if origin is None:
        raise ValueError("an invite link must be an https://<host>/join/<token> URL")
    validate_https_url(origin, label="An invite link")
    path = urlsplit(value).path.rstrip("/")
    segments = path.split("/")
    if len(segments) < 3 or segments[-2] != "join":
        raise ValueError("an invite link must end in /join/<token>")
    _check_invite_token(segments[-1])
    return InviteRef(token=segments[-1], origin=origin, link=f"{origin}{path}")


def invite_base_url(verification_uri: str) -> str | None:
    """Return the web app's base URL: ``verification_uri`` minus its ``/device``.

    memory-cloud builds ``verification_uri`` as ``{frontend_url}/device``, and
    it is the only web-app location the CLI learns — ``--server`` is the API
    origin, which can differ. ``None`` when the URI does not end in a
    ``/device`` segment or fails the same HTTPS check as ``--server``: the
    CLI then cannot place ``/join`` safely.
    """
    origin = _url_origin(verification_uri)
    path = urlsplit(verification_uri).path.rstrip("/")
    if origin is None or not path.endswith("/device"):
        return None
    base = f"{origin}{path.removesuffix('/device')}"
    try:
        validate_https_url(base)
    except ValueError:
        return None
    return base


def build_invite_link(
    verification_uri: str, verification_uri_complete: str, token: str
) -> str | None:
    """Build the single link that accepts an invite and lands on ``/device``.

    ``{base}/join/{token}?return_to=<path and query of verification_uri_complete>``,
    with ``base`` from :func:`invite_base_url`. ``return_to`` is a
    same-origin relative path (e.g. ``/device?user_code=ABCD-1234``),
    percent-encoded whole. memory-cloud v0.76.0 (memory-cloud#1666) reads it
    with ``URLSearchParams`` and keeps it only when its ``safeReturnTo``
    passes: a path starting with exactly one ``/``, with no backslash or C0
    control character. Named so the TypeScript CLI can mirror it as
    ``buildInviteLink`` (kagura-memory-typescript-sdk#44).

    Returns:
        The link, or ``None`` when ``/join`` cannot be placed or would drop
        ``return_to``: no ``base``, ``verification_uri_complete`` on a
        different origin, or a path ``/join`` would not accept.

    Raises:
        ValueError: ``token`` does not match the invite-token pattern.
    """
    _check_invite_token(token)
    base = invite_base_url(verification_uri)
    if base is None or _url_origin(verification_uri_complete) != _url_origin(verification_uri):
        return None
    parts = urlsplit(verification_uri_complete)
    return_to = f"{parts.path}?{parts.query}" if parts.query else parts.path
    if not _join_keeps_return_to(return_to):
        return None
    return f"{base}/join/{token}?return_to={quote(return_to, safe='')}"


def check_invite_origin(invite: InviteRef, verification_uri: str) -> None:
    """Refuse a full invite link whose origin is not the web app's.

    The web app's origin is ``verification_uri``'s — ``--server`` is the API
    origin, which can differ. The CLI never rewrites a link onto another
    host, so a mismatch means the user is logging in to the wrong server.
    A bare token carries no origin and always passes.

    Raises:
        ValueError: the origins differ. The message names both origins,
            never the token.
    """
    if invite.origin is None:
        return
    web_origin = _url_origin(verification_uri)
    if invite.origin != web_origin:
        raise ValueError(
            f"This invite is for a different server ({invite.origin}) than the one "
            f"you are logging in to ({web_origin or verification_uri})."
        )


async def fetch_system_info(
    client: httpx.AsyncClient,
    server: str,
    *,
    timeout: float = _SYSTEM_INFO_TIMEOUT_SEC,
) -> dict[str, Any] | None:
    """GET the public ``/api/v1/system/info`` and return the raw JSON object.

    Sent without credentials (there are none yet during login). The raw body
    is returned rather than :class:`~kagura_memory.models.ServerInfo`
    because ``ServerFeatures`` does not model ``beta_invites`` yet (swap to
    ``ServerInfo`` once #257 adds it). Best-effort: any HTTP error, timeout
    or non-object body yields ``None``.
    """
    try:
        response = await client.get(f"{server.rstrip('/')}{_PATH_SYSTEM_INFO}", timeout=timeout)
    except httpx.RequestError:
        return None
    if response.status_code != 200:
        return None
    return _safe_json(response) or None


def invite_support(system_info: dict[str, Any] | None) -> InviteSupport:
    """Decide how ``--invite`` is presented, from a raw ``/system/info`` body.

    A ``features`` object without ``beta_invites: true`` means invites are
    off, as memory-cloud's own web app reads it: the flag is default-off, and
    a server older than the flag (before 0.70.0) has no ``/join`` route.

    Returns:
        ``"disabled"`` when ``features`` is an object whose ``beta_invites``
        is not ``true`` (invites have no effect on this server);
        ``"hand_off"`` when the server version is at least
        :data:`JOIN_RETURN_TO_MIN_SERVER_VERSION` (memory-cloud v0.76.0), so
        one ``/join`` link carries the user on to ``/device``; otherwise
        ``"two_step"`` (no info or no ``features`` object, or an older or
        unparseable version).
    """
    if system_info is None:
        return "two_step"
    features = system_info.get("features")
    if isinstance(features, dict) and features.get("beta_invites") is not True:
        return "disabled"
    version = system_info.get("version")
    parsed = _parse_version_prefix(version) if isinstance(version, str) else None
    if parsed is not None and parsed >= JOIN_RETURN_TO_MIN_SERVER_VERSION:
        return "hand_off"
    return "two_step"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _device_rate_limited_message(response: httpx.Response) -> str:
    """Explain a 429 from ``device/authorize``, with the wait from ``Retry-After``.

    memory-cloud v0.76.0 (memory-cloud#1667) limits ``device/authorize`` per
    client address and answers 429 with ``Retry-After: 60`` and an RFC 6749
    ``error_description``, which is kept. A missing or non-numeric
    ``Retry-After`` reads as that same 60 s window.
    """
    retry_after = _retry_after_seconds(response)
    if retry_after is None:
        retry_after = _DEVICE_RATE_LIMIT_RETRY_AFTER_SEC
    message = (
        "Too many sign-in attempts from this address (HTTP 429). "
        f"Retry after {retry_after} seconds."
    )
    detail = extract_detail(response)
    return f"{message}\n  Server said: {detail}" if detail else message


def _check_invite_token(token: str) -> None:
    """Raise ``ValueError`` (without echoing ``token``) unless it is well-formed."""
    if not _INVITE_TOKEN_RE.fullmatch(token):
        raise ValueError(f"an invite token must be {_INVITE_TOKEN_RULE}")


def _join_keeps_return_to(path: str) -> bool:
    """Whether memory-cloud's ``/join`` keeps ``path`` as its ``return_to``.

    Mirrors the relative-path branch of the frontend's ``safeReturnTo``
    (memory-cloud v0.76.0): exactly one leading ``/``, no backslash and no
    C0 control character. ``/join`` drops any other value silently and sends
    the invitee to the dashboard, so the CLI prints the two steps instead.
    """
    return (
        path.startswith("/")
        and not path.startswith("//")
        and not any(ch == "\\" or ord(ch) <= 0x1F for ch in path)
    )


def _parse_version_prefix(version: str) -> tuple[int, int, int] | None:
    """``(major, minor, patch)`` from a ``v?MAJOR.MINOR.PATCH`` prefix; ``None`` otherwise.

    Same shape as ``doctor._parse_version_prefix`` so the two can merge.
    """
    match = _SEMVER_PREFIX_RE.match(version)
    if match is None:
        return None
    return (int(match[1]), int(match[2]), int(match[3]))


def _url_origin(url: str) -> str | None:
    """``scheme://host[:port]`` of an http(s) URL, lower-cased; ``None`` otherwise.

    The scheme's default port is dropped, as browsers do, so
    ``https://host:443`` and ``https://host`` are the same origin.
    """
    parts = urlsplit(url)
    try:
        host, port = parts.hostname, parts.port
    except ValueError:  # a non-numeric or out-of-range port
        return None
    if parts.scheme not in _DEFAULT_PORTS or not host:
        return None
    if ":" in host:  # IPv6 literal — hostname drops the brackets
        host = f"[{host}]"
    if port is None or port == _DEFAULT_PORTS[parts.scheme]:
        return f"{parts.scheme}://{host}"
    return f"{parts.scheme}://{host}:{port}"


def _safe_json_object(response: httpx.Response, endpoint: str) -> dict[str, Any]:
    """Parse a 200 response body as a JSON object.

    Converts malformed / non-dict success bodies into a
    :class:`KaguraAuthError` with HTTP status + truncated body, so a
    server that wedges and returns HTML / a JSON array / a scalar
    doesn't surface as an unhelpful ``ValueError`` traceback.
    """
    try:
        body = response.json()
    except (ValueError, UnicodeDecodeError) as e:
        detail = response.text[:200] if response.text else ""
        raise KaguraAuthError(
            f"{endpoint} returned HTTP {response.status_code} but body is not JSON: {e}. "
            f"Body: {detail}"
        ) from e
    if not isinstance(body, dict):
        raise KaguraAuthError(
            f"{endpoint} returned HTTP {response.status_code} but body is not a JSON object "
            f"(got {type(body).__name__})"
        )
    return body


def _token_response_from_response(response: httpx.Response) -> TokenResponse:
    """Build a :class:`TokenResponse` from a 200 ``/oauth/token/`` response.

    ``expires_at`` is computed from ``expires_in`` at receipt time so
    laptop sleep / clock skew won't yield a negative TTL after wake.
    Missing or invalid required fields surface as :class:`KaguraAuthError`
    rather than ``KeyError`` / ``ValueError``.
    """
    body = _safe_json_object(response, "Token endpoint")
    try:
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
    except (KeyError, TypeError, ValueError) as e:
        raise KaguraAuthError(
            f"Token endpoint returned HTTP 200 but body is missing required fields: {e}. "
            f"Body keys: {sorted(body.keys())}"
        ) from e


def _safe_json(response: httpx.Response) -> dict[str, Any]:
    """Return ``response.json()`` as a dict, or ``{}`` if unparseable."""
    try:
        body = response.json()
    except (ValueError, UnicodeDecodeError):
        return {}
    return body if isinstance(body, dict) else {}
