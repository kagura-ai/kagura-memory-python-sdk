"""Shared HTTP utilities for Kagura Memory SDK clients."""

from __future__ import annotations

import re
import uuid
from importlib.metadata import version as _pkg_version
from typing import Any, NoReturn, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from .exceptions import (
    KaguraAuthError,
    KaguraConnectionError,
    KaguraRateLimitError,
    KaguraResponseError,
    _exc_message,
)

SDK_VERSION: str = _pkg_version("kagura-memory")
"""Package version string, shared across client modules."""

_M = TypeVar("_M", bound=BaseModel)

_MAX_LISTED_RESPONSE_ERRORS = 3

_UPGRADE_HINT = "The server may be newer than this SDK; upgrading kagura-memory may help."


def base_url_from_mcp(mcp_url: str) -> str:
    """Derive REST API base URL from an MCP URL.

    Strips ``/mcp`` and everything after it (e.g. ``/mcp/w/{workspace}``).

    Args:
        mcp_url: MCP server URL (already stripped of trailing slash).

    Returns:
        Base URL suitable for REST API calls.
    """
    m = re.search(r"/mcp(?=/|$)", mcp_url)
    return mcp_url[: m.start()] if m else mcp_url


def extract_detail(response: httpx.Response) -> str:
    """Return a useful server-supplied error string from an httpx response.

    Handles six response shapes:

    - ``{"detail": "string"}`` — returned as-is (FastAPI HTTPException default).
    - ``{"detail": [{"loc": [...], "msg": "...", ...}, ...]}`` — FastAPI's
      validation-error format (RequestValidationError / Pydantic). Each entry
      is formatted as ``"<loc.path>: <msg>"`` and joined with ``"; "``. Without
      this, a 422 surfaces to the caller as a bare ``"HTTP 422"`` with no hint
      at which field failed validation.
    - ``{"error": "<CODE>", "message": "string", "details": {...}}`` — the
      memory-cloud canonical envelope (most errors render this way, not
      ``detail``). Returns ``message``; when ``details.errors`` carries a
      validation list it is appended so a 422 names the failing field instead
      of the generic "Request validation failed".
    - ``{"jsonrpc": "2.0", "error": {"code": int, "message": "string"}, ...}`` —
      the MCP transport's 4xx (see :func:`jsonrpc_error_body`). Returns
      ``error.message``, e.g. the session-expired 404's re-initialize hint.
    - ``{"error": "<code>", "error_description": "string"}`` — the OAuth-style
      body of the MCP transport's workspace-URL 400/403 (and its 401). Returns
      ``error_description``, e.g. "You are not a member of this workspace."
    - Anything else — non-JSON body, non-dict body, missing ``detail`` and
      ``message``, or values of unexpected type — returns an empty string so
      callers can fall back to ``response.text`` or just print the status.
    """
    try:
        body = response.json()
    except (ValueError, UnicodeDecodeError):
        return ""
    if not isinstance(body, dict):
        return ""
    detail = body.get("detail")
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list):
        return _format_validation_errors(detail)
    message = body.get("message")
    if isinstance(message, str) and message:
        details = body.get("details")
        if isinstance(details, dict):
            errors = details.get("errors")
            if isinstance(errors, list):
                formatted = _format_validation_errors(errors)
                if formatted:
                    return f"{message}: {formatted}"
        return message
    error = body.get("error")
    if isinstance(error, dict):
        rpc_message = error.get("message")
        if isinstance(rpc_message, str):
            return rpc_message
    description = body.get("error_description")
    if isinstance(description, str):
        return description
    return ""


def jsonrpc_error_body(response: httpx.Response) -> dict[str, Any] | None:
    """Return ``response``'s body when it is a JSON-RPC error envelope, else ``None``.

    memory-cloud's MCP transport answers a request it rejects before dispatch
    — a malformed envelope, an unsupported protocol version, an unknown
    session (see :func:`mcp_session_expired`) — with a 4xx whose body is a
    JSON-RPC ``error`` object (``-32603``, ``-32600``, ``-32601``, ``-32602``,
    ``-32022``; memory-cloud #1541/#1544), not the REST ``detail`` envelope.
    The OAuth-style 400/401/403 bodies' ``error`` is a string
    (``"invalid_token"``), so they are not matched.

    Args:
        response: Any MCP endpoint response.
    """
    try:
        body = response.json()
    except (ValueError, UnicodeDecodeError):
        return None
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        return body
    return None


def mcp_session_header(session_id: str | None) -> dict[str, str]:
    """Headers naming the MCP session ``session_id`` — none when there is no session yet."""
    return {"mcp-session-id": session_id} if session_id else {}


_JSONRPC_METHOD_NOT_FOUND = -32601


def mcp_session_expired(response: httpx.Response, session_id: str | None) -> bool:
    """Whether ``response`` says the MCP session a request carried is gone.

    MCP Streamable HTTP answers a request naming a session the server no
    longer holds with ``404``, and the client must then send a new
    ``initialize``. memory-cloud keeps legacy (``initialize``-handshake)
    sessions in process memory, drops them after an idle hour and on every
    restart, and has that ``404`` + ``-32603`` check; but as deployed
    (v0.75.0) its routes normalize ``/mcp`` to ``/mcp/``, which skips the
    check and re-adopts the unknown id as a new session. So this is the
    spec-mandated path, reached against a server that enforces it.

    The one ``404`` that is not about the session is the stateless 2026-07-28
    ``-32601`` Method-not-found reply (memory-cloud #1544): that path ignores
    the session id, so re-initializing would only mint an orphan session.

    Args:
        response: The response to the request.
        session_id: The ``mcp-session-id`` the request was sent with, if any.
    """
    if not session_id or response.status_code != 404:
        return False
    body = jsonrpc_error_body(response)
    return body is None or body["error"].get("code") != _JSONRPC_METHOD_NOT_FOUND


def sanitize_server_detail(detail: str | None) -> str | None:
    """Drop server-provided detail strings that contain credential markers.

    Server 403 ``detail`` payloads usually surface non-sensitive reasons
    (scope, deactivation, plan limit) that are valuable to operators —
    forwarding them helps debugging. But the detail field is operator-
    facing text the server controls; a future server bug echoing back
    the Bearer header or api_key would otherwise be passed straight to
    the user. Drop the detail entirely when it carries any marker that
    looks credential-shaped. Returns ``None`` when the detail is empty
    or unsafe to display.
    """
    if not detail:
        return None
    lowered = detail.lower()
    # ``api_key=`` covers ``api_key=<value>`` style; ``bearer`` catches
    # ``Bearer <token>`` echoes; ``authorization`` catches header reflections.
    if "bearer" in lowered or "authorization" in lowered or "api_key=" in lowered:
        return None
    return detail


def _retry_after_seconds(response: httpx.Response) -> int | None:
    """Parse a numeric ``Retry-After`` header (delta-seconds), else ``None``.

    Only the integer-seconds form is honored; an HTTP-date ``Retry-After`` (rare
    for rate limits) is treated as absent rather than mis-parsed.
    """
    raw = response.headers.get("Retry-After")
    if raw is None:
        return None
    raw = raw.strip()
    return int(raw) if raw.isdigit() else None


def raise_for_kagura_status(e: httpx.HTTPStatusError) -> NoReturn:
    """Translate an httpx ``HTTPStatusError`` into the matching Kagura error.

    Maps ``401`` → :class:`~kagura_memory.exceptions.KaguraAuthError`, ``429`` →
    :class:`~kagura_memory.exceptions.KaguraRateLimitError` (honoring a numeric
    ``Retry-After`` header), and every other status →
    :class:`~kagura_memory.exceptions.KaguraConnectionError`. The server-supplied
    ``detail`` (a FastAPI string or validation-error list) is appended when
    present — surfacing e.g. which field failed a 422 — otherwise the
    exception's own message is used so the status is never left bare. This
    function always raises.
    """
    response = e.response
    status = response.status_code
    if status == 401:
        raise KaguraAuthError("Authentication failed. Check your API key.") from e
    detail = extract_detail(response) or _exc_message(e)
    if status == 429:
        raise KaguraRateLimitError(
            f"Rate limit exceeded (HTTP 429): {detail}",
            retry_after=_retry_after_seconds(response),
        ) from e
    raise KaguraConnectionError(f"HTTP {status}: {detail}") from e


def parse_response(model: type[_M], data: Any, *, operation: str) -> _M:
    """Validate a server response payload into ``model`` (#250).

    Drift — a payload the model rejects, usually because the server is
    newer than the SDK — raises :class:`KaguraResponseError` naming
    ``operation`` and the failing fields instead of letting a raw
    ``pydantic.ValidationError`` escape the client. The message leaves
    payload values out (they can be secret ciphertext or key plaintext);
    the full error stays on ``__cause__``, whose own text does include
    them.

    Response side only: request models built from caller arguments must
    keep raising ``ValidationError``, which reports a caller mistake.

    Args:
        model: Pydantic model the payload should match.
        data: Decoded JSON payload. Pass a missing or ``null`` envelope
            key through as ``None`` so it fails here too.
        operation: Call being parsed, used as the message prefix and the
            exception's ``operation`` attribute.

    Raises:
        KaguraResponseError: If ``data`` does not validate against ``model``.
    """
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        errors = exc.errors(include_url=False, include_context=False, include_input=False)
        listed = _format_validation_errors(errors[:_MAX_LISTED_RESPONSE_ERRORS])
        if len(errors) > _MAX_LISTED_RESPONSE_ERRORS:
            listed += f" (+{len(errors) - _MAX_LISTED_RESPONSE_ERRORS} more)"
        problem = f"for {model.__name__} ({listed})"
        raise KaguraResponseError(
            f"{operation}: unexpected server response {problem}. {_UPGRADE_HINT}",
            operation=operation,
        ) from exc


def parse_response_list(model: type[_M], data: Any, *, operation: str) -> list[_M]:
    """Validate a JSON array of ``model`` rows, each as :func:`parse_response` does.

    ``data`` that is not a list — its envelope key was missing, ``null`` or
    another type — raises :class:`KaguraResponseError` instead of a
    ``TypeError`` on iteration.
    """
    if not isinstance(data, list):
        raise response_shape_error(
            operation, f"expected a list of {model.__name__}, got {type(data).__name__}"
        )
    return [parse_response(model, row, operation=operation) for row in data]


def response_shape_error(operation: str, problem: str) -> KaguraResponseError:
    """Build the :class:`KaguraResponseError` for a mis-shaped 2xx envelope.

    For drift caught before any model sees it (a list field that is not a
    list), so nothing is chained. ``problem`` describes the shape only,
    never payload values.
    """
    return KaguraResponseError(
        f"{operation}: unexpected server response ({problem}). {_UPGRADE_HINT}",
        operation=operation,
    )


def _format_validation_errors(errors: list[Any]) -> str:
    # Silent-skip malformed entries so a single bad entry doesn't blank the line.
    # ``loc`` is a list on the wire (FastAPI) and a tuple from pydantic's own
    # ``ValidationError.errors()`` (parse_response).
    parts: list[str] = []
    for entry in errors:
        if not isinstance(entry, dict):
            continue
        msg = entry.get("msg")
        if not isinstance(msg, str) or not msg:
            continue
        loc = entry.get("loc")
        if isinstance(loc, (list, tuple)) and loc:
            loc_path = ".".join(str(part) for part in loc)
            parts.append(f"{loc_path}: {msg}")
        else:
            parts.append(msg)
    return "; ".join(parts)


# Plain-HTTP is permitted only for genuine loopback hosts. The host token must be
# followed by a boundary — a port (``:\d+``), a path/query/fragment delimiter, or
# end-of-string — so a prefix-match attack like ``http://localhost.evil.com`` or a
# userinfo trick like ``http://localhost@evil.com`` cannot smuggle an external host
# past the check (#189). Scheme matching stays case-sensitive to preserve prior
# behavior; the trigger below is the lowercase ``http://`` literal.
_LOCALHOST_HTTP_RE = re.compile(r"^http://(?:localhost|127\.0\.0\.1|\[::1\])(?::\d+)?(?:[/?#]|$)")


def normalize_uuid(value: object, *, label: str) -> str:
    """Return the canonical UUID string, rejecting non-UUIDs before URL use.

    ``uuid.UUID`` tolerates non-canonical spellings (``{braces}``,
    ``urn:uuid:`` prefix, dashless 32-hex); interpolating the RAW input
    into a URL path would send those to the server and surface as a
    misleading uniform 404 — normalize instead of just validating.

    Args:
        value: Candidate UUID. Typed ``object`` because this is a runtime
            guard at the public-parameter trust boundary: non-string
            values from untyped callers are coerced via ``str()`` and
            rejected with the same uniform ValueError.
        label: Parameter name used in the error message.

    Raises:
        ValueError: If ``value`` is not a parseable UUID.
    """
    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label} must be a UUID, got {value!r}") from exc


def validate_lat_lon(lat: object, lon: object) -> None:
    """Reject coordinates the server would reject anyway (memory-cloud #1331).

    Shared by every geo surface — the ``recall_nearby`` query point and the
    CLI's ``--location`` write shorthand — so the WHERE axis has one definition
    of a valid coordinate. Unlike ``radius_m``, which the server clamps, the
    server *rejects* out-of-range lat/lon, so checking locally only saves a
    round-trip that could return nothing but a 422.

    Args:
        lat: Latitude, -90 to 90. Typed ``object`` for the same reason as
            :func:`normalize_uuid`: this is a runtime guard at the
            public-parameter trust boundary, so a non-numeric value from an
            untyped caller is rejected with a uniform ValueError rather than
            an opaque TypeError out of the comparison. A **string** coordinate
            is rejected rather than coerced — the server 422s string-typed
            numerics, so silently accepting ``"35.68"`` would only move the
            failure to the wire.
        lon: Longitude, -180 to 180. Same rules.

    Raises:
        ValueError: If either coordinate is non-numeric or outside its range.
    """
    for label, value, limit in (("lat", lat, 90), ("lon", lon, 180)):
        # bool is a subclass of int, but True is never a coordinate.
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(
                f"{label} must be a number, got {type(value).__name__} ({value!r}). "
                "The server rejects string-typed coordinates."
            )
        # NaN fails every comparison, so this form rejects it. Do not rewrite
        # it as `value < -limit or value > limit` — that looks equivalent but
        # evaluates False for NaN, letting it through.
        if not -limit <= value <= limit:
            raise ValueError(f"{label} must be between -{limit} and {limit}, got {value}")


def validate_https_url(url: str, *, label: str = "URL") -> None:
    """Enforce HTTPS except for localhost development.

    Args:
        url: URL to validate.
        label: Human-readable label for error messages.

    Raises:
        ValueError: If URL uses HTTP and is not a loopback host.
    """
    if url.startswith("http://") and not _LOCALHOST_HTTP_RE.match(url):
        raise ValueError(
            f"{label} must use HTTPS for security (got: {url}). "
            "HTTP is only allowed for localhost development."
        )
