"""Shared HTTP utilities for Kagura Memory SDK clients."""

from __future__ import annotations

import re
import uuid
from collections.abc import Collection, Mapping
from datetime import UTC, datetime, time, timedelta
from importlib.metadata import version as _pkg_version
from typing import Any, NoReturn, TypeVar
from urllib.parse import parse_qsl, unquote_plus, urlencode, urlsplit, urlunsplit

import httpx
from pydantic import BaseModel, ValidationError

from .exceptions import (
    KaguraAuthError,
    KaguraConnectionError,
    KaguraError,
    KaguraFeatureNotAvailableError,
    KaguraQuotaError,
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

    Drops the query and fragment (``/mcp?guardrails=off`` can arrive through
    ``KAGURA_MCP_URL`` or ``--server``), then strips ``/mcp`` and everything
    after it (e.g. ``/mcp/w/{workspace}``).

    Args:
        mcp_url: MCP server URL.

    Returns:
        Base URL suitable for REST API calls.
    """
    url = re.split(r"[?#]", mcp_url, maxsplit=1)[0].rstrip("/")
    m = re.search(r"/mcp(?=/|$)", url)
    return url[: m.start()] if m else url


def mcp_url_with_query(
    mcp_url: str, *, guardrails: str | None = None, tool_profile: str | None = None
) -> str:
    """Return ``mcp_url`` with memory-cloud's ``guardrails`` / ``profile`` query set.

    The rest of the existing query is kept verbatim. A key being set replaces
    every earlier value of it rather than appending, because the server reads
    only the first ``guardrails`` / ``profile`` it finds. ``None`` leaves that
    key as it is.

    Args:
        mcp_url: The MCP endpoint URL, with or without a query.
        guardrails: ``?guardrails=`` value (see :func:`normalize_guardrails`).
        tool_profile: ``?profile=`` value, the tool profile ``tools/list``
            serves (e.g. ``core``); the server rejects an unknown name.

    Returns:
        The URL to POST MCP requests to.
    """
    updates = {
        key: value
        for key, value in (("guardrails", guardrails), ("profile", tool_profile))
        if value is not None
    }
    if not updates:
        return mcp_url
    parts = urlsplit(mcp_url)
    kept = _query_without(parts.query, updates)
    return urlunsplit(parts._replace(query="&".join([*kept, urlencode(updates)])))


def _query_without(query: str, keys: Collection[str]) -> list[str]:
    """The non-empty ``&`` segments of ``query`` whose decoded name is not in ``keys``."""
    return [
        segment
        for segment in query.split("&")
        if segment and unquote_plus(segment.split("=", 1)[0]) not in keys
    ]


def mcp_url_without_query_param(mcp_url: str, key: str) -> str:
    """Return ``mcp_url`` with every ``key`` parameter dropped from its query.

    Names are compared decoded, as the server reads them (``guard%72ails`` is
    ``guardrails``); the rest of the query is kept verbatim.

    Args:
        mcp_url: The MCP endpoint URL.
        key: The query parameter to drop, e.g. ``guardrails``.

    Returns:
        The URL without any ``key`` segment and without a bare ``?`` when
        nothing else is left; ``mcp_url`` itself, as written, when it has no
        ``key`` parameter.
    """
    parts = urlsplit(mcp_url)
    kept = _query_without(parts.query, (key,))
    if len(kept) == sum(1 for segment in parts.query.split("&") if segment):
        return mcp_url
    return urlunsplit(parts._replace(query="&".join(kept)))


def mcp_url_has_tools_allowlist(mcp_url: str) -> bool:
    """True when ``mcp_url`` carries memory-cloud's ``?tools=`` allowlist.

    The server applies a ``tools`` allowlist instead of ``profile``, so a tool
    profile set on such a URL has no effect.

    Args:
        mcp_url: The MCP endpoint URL.

    Returns:
        Whether its query has a ``tools`` parameter.
    """
    query = urlsplit(mcp_url).query
    return any(key == "tools" for key, _ in parse_qsl(query, keep_blank_values=True))


def mcp_url_guardrails_off(mcp_url: str) -> bool:
    """True when ``mcp_url``'s first ``guardrails`` value is ``off`` (any case).

    The server reads only the first value, and ``off`` also drops the
    ``guardrails`` block from ``get_context_info``.

    Args:
        mcp_url: The MCP endpoint URL.

    Returns:
        Whether the URL turns both guardrail lanes off.
    """
    query = urlsplit(mcp_url).query
    values = [v for k, v in parse_qsl(query, keep_blank_values=True) if k == "guardrails"]
    return bool(values) and values[0].strip().lower() == "off"


def normalize_guardrails(value: str) -> str:
    """Validate a ``?guardrails=`` value: ``off`` or a context UUID.

    memory-cloud silently ignores any other value (it neither errors nor
    switches the lane), so a typo is rejected here instead.

    Args:
        value: ``off`` (any case) or a context UUID in any spelling
            ``uuid.UUID`` accepts.

    Returns:
        ``"off"`` or the canonical UUID string.

    Raises:
        ValueError: ``value`` is neither.
    """
    stripped = value.strip()
    if stripped.lower() == "off":
        return "off"
    try:
        return normalize_uuid(stripped, label="guardrails")
    except ValueError:
        raise ValueError(f"guardrails must be 'off' or a context UUID, got {value!r}") from None


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


# Gate refusals (#256): memory-cloud v0.75.0+ (#1644) stamps every plan or
# quota refusal with a ``gate`` descriptor — REST in ``details``, MCP as
# top-level envelope fields. Older servers send only the error code and the
# legacy fields, so the code is the fallback.
_FEATURE_GATES = frozenset({"plan", "allowlist", "deployment"})
_FEATURE_CODES = frozenset({"plan_required", "feature_not_available", "FEAT-001"})
_QUOTA_CODES = frozenset({"quota_exceeded", "QUOTA-001"})
# The daily MCP call cap every non-read-only tool checks in-band. It carries
# no ``gate`` and names its counts ``used_today`` / ``daily_limit``; the cap
# counts calls per UTC day.
_MCP_DAILY_CAP_CODE = "rate_limit_exceeded"
# Each names exactly one cap, so the code alone makes it a quota refusal.
_SINGLE_CAP_QUOTA_CODES = frozenset({"QUOTA-002", "CONNECTOR-001", _MCP_DAILY_CAP_CODE})
# Envelope keys that frame an MCP refusal rather than describe it.
_ENVELOPE_KEYS = frozenset({"status", "error", "message"})
# memory-cloud's frozen ``quota_type`` vocabulary (``QUOTA_TYPES``). A quota
# refusal naming one is a tier quota even from a server without ``gate``.
_TIER_QUOTA_TYPES = frozenset(
    {
        "contexts",
        "members",
        "workspace_limit_reached",
        "memories_per_day",
        "memory_analysis",
        "sleep_enabled_contexts",
        "storage_bytes",
        "agents",
        "resource_tokens",
        "connectors",
        "embedding_spend_daily",
        "embedding_spend_monthly",
        "api_mcp_daily",
        "api_rest_daily",
        "api_public_daily",
    }
)


def error_envelope(response: httpx.Response) -> tuple[str, str, dict[str, Any]] | None:
    """Return ``(error, message, details)`` of memory-cloud's canonical REST error body.

    ``None`` when the body is not ``{"error": <str>, ...}`` — non-JSON, the
    FastAPI ``{"detail": ...}`` shape, or the JSON-RPC ``error`` object. A
    missing ``message`` reads as ``""`` and missing ``details`` as ``{}``.
    """
    try:
        body = response.json()
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(body, dict) or not isinstance(body.get("error"), str):
        return None
    message = body.get("message")
    details = body.get("details")
    return (
        body["error"],
        message if isinstance(message, str) else "",
        details if isinstance(details, dict) else {},
    )


def gate_error(
    code: str,
    message: str,
    fields: Mapping[str, Any],
    *,
    retry_after: int | None = None,
) -> KaguraError | None:
    """Build the typed exception for a plan or quota refusal (#256).

    Branches on ``gate`` first (memory-cloud v0.75.0+): ``"quota"`` is a
    quota, ``"plan"`` / ``"allowlist"`` / ``"deployment"`` a feature gate —
    whatever the error code. Without a ``gate`` (an older server, or a kind
    this SDK does not know) the code decides: ``plan_required``,
    ``feature_not_available`` and ``FEAT-001`` are feature gates;
    ``quota_exceeded`` and ``QUOTA-001`` are quotas only when they name a
    known ``quota_type`` or a retry window, because the same code also
    covers limits no plan lifts, such as the 1 MB memory-size guard. MCP
    ``rate_limit_exceeded`` (the daily MCP call cap) is the
    ``api_mcp_daily`` quota, resetting at the next UTC midnight.

    Malformed detail fields are dropped rather than raised, so drift never
    hides the refusal itself.

    Args:
        code: The error code — MCP ``error`` or REST ``error``.
        message: Message for the exception.
        fields: Where the descriptor lives: the MCP envelope itself, or
            the REST ``details``.
        retry_after: The retry window the response named (REST
            ``Retry-After``). When ``None``, MCP's ``retry_after_seconds``
            (the resource events-per-hour ceiling) is read from ``fields``.

    Returns:
        :class:`KaguraQuotaError` or :class:`KaguraFeatureNotAvailableError`,
        each carrying every descriptor field as ``details``; a plain
        :class:`KaguraError` for a quota code that is neither a tier quota
        nor a retry window; ``None`` when ``code`` is not a plan or quota
        refusal.
    """
    details = {k: v for k, v in fields.items() if k not in _ENVELOPE_KEYS}
    if code == _MCP_DAILY_CAP_CODE:
        # Fill in the canonical names the envelope leaves implicit; whatever
        # the server does send wins.
        midnight = datetime.combine(datetime.now(UTC).date() + timedelta(days=1), time(), UTC)
        fields = {
            "quota_type": "api_mcp_daily",
            "current": fields.get("used_today"),
            "limit": fields.get("daily_limit"),
            "resets_at": midnight.isoformat(),
            **fields,
        }
    if retry_after is None:
        retry_after = _opt_int(fields.get("retry_after_seconds"))
    gate = _opt_str(fields.get("gate"))
    if gate == "quota":
        return _quota_error(message, fields, retry_after, details)
    if gate in _FEATURE_GATES or code in _FEATURE_CODES:
        return KaguraFeatureNotAvailableError(message, **_plan_fields(fields), details=details)
    if code in _SINGLE_CAP_QUOTA_CODES:
        return _quota_error(message, fields, retry_after, details)
    if code in _QUOTA_CODES:
        if _opt_str(fields.get("quota_type")) in _TIER_QUOTA_TYPES or retry_after is not None:
            return _quota_error(message, fields, retry_after, details)
        return KaguraError(message)
    return None


def _quota_error(
    message: str,
    fields: Mapping[str, Any],
    retry_after: int | None,
    details: dict[str, Any],
) -> KaguraQuotaError:
    return KaguraQuotaError(
        message,
        retry_after,
        quota_type=_opt_str(fields.get("quota_type")),
        limit=_opt_int(fields.get("limit")),
        current=_opt_int(fields.get("current")),
        used_today=_opt_int(fields.get("used_today")),
        resets_at=_opt_datetime(fields.get("resets_at")),
        **_plan_fields(fields),
        details=details,
    )


def _plan_fields(fields: Mapping[str, Any]) -> dict[str, str | None]:
    """The descriptor keys a quota and a feature gate share."""
    keys = ("gate", "feature", "required_plan", "required_plan_display", "current_plan")
    return {key: _opt_str(fields.get(key)) for key in keys}


def _opt_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _opt_int(value: object) -> int | None:
    # bool is a subclass of int, but True is never a count.
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _opt_datetime(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


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
# past the check (#189). Every URL parser reads the scheme and host without regard
# to case, so ``HTTP://evil.com`` is fetched over plain HTTP exactly like
# ``http://evil.com`` (#274); the match is ASCII-only, so no Unicode case fold
# (``ſ`` → ``s``) can spell ``localhost``. Any ``http:`` scheme counts, slashes or
# not: WHATWG parsers read ``http:/evil.com`` and ``http:evil.com`` as
# ``http://evil.com``, and a loopback URL written that way is refused.
_PLAIN_HTTP_RE = re.compile(r"^http:", re.IGNORECASE | re.ASCII)
_LOCALHOST_HTTP_RE = re.compile(
    r"^http://(?:localhost|127\.0\.0\.1|\[::1\])(?::\d+)?(?:[/?#]|$)", re.IGNORECASE | re.ASCII
)
# What URL parsers drop before they read a scheme: whitespace and C0 controls
# around the URL, and (WHATWG: Node, browsers, Rust's ``url`` crate, which the
# harnesses ``kagura setup`` writes for use) a tab or newline anywhere in it.
_URL_IGNORED_RE = re.compile(r"^[\x00-\x20\s]+|[\x00-\x20\s]+$|[\t\n\r]")


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


def validate_coordinate(label: str, value: object, limit: int) -> None:
    """Reject one coordinate the server would reject anyway.

    The single definition of a valid coordinate on the WHERE axis, shared by
    every geo surface — the ``recall_nearby`` query point and the CLI's
    ``--location`` write shorthand (both via :func:`validate_lat_lon`), and
    each ``list_memories`` bbox bound. Unlike ``radius_m``, which the server
    clamps, the server *rejects* out-of-range lat/lon (memory-cloud #1331,
    #1334), so checking locally only saves a round-trip that could return
    nothing but a 422.

    Args:
        label: Parameter name used in the error message.
        value: The coordinate. Typed ``object`` for the same reason as
            :func:`normalize_uuid`: this is a runtime guard at the
            public-parameter trust boundary, so a non-numeric value from an
            untyped caller is rejected with a uniform ValueError rather than
            an opaque TypeError out of the comparison. A **string** coordinate
            is rejected rather than coerced — the server 422s string-typed
            numerics, so silently accepting ``"35.68"`` would only move the
            failure to the wire.
        limit: The range is ``-limit`` to ``limit`` — 90 for a latitude,
            180 for a longitude.

    Raises:
        ValueError: If ``value`` is non-numeric or outside its range.
    """
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


def validate_lat_lon(lat: object, lon: object) -> None:
    """Reject a lat/lon point the server would reject anyway (memory-cloud #1331).

    Args:
        lat: Latitude, -90 to 90. See :func:`validate_coordinate` for the rules.
        lon: Longitude, -180 to 180. Same rules.

    Raises:
        ValueError: If either coordinate is non-numeric or outside its range.
    """
    validate_coordinate("lat", lat, 90)
    validate_coordinate("lon", lon, 180)


def normalize_url(url: str) -> str:
    """Return ``url`` as a URL parser reads it, before it reads the scheme.

    Surrounding whitespace and C0 controls go, and so does any tab or newline
    inside it. Nothing else changes: the scheme and host keep their case.

    Args:
        url: A URL as the user gave it.

    Returns:
        The URL :func:`validate_https_url` checks.
    """
    return _URL_IGNORED_RE.sub("", url)


def validate_https_url(url: str, *, label: str = "URL") -> None:
    """Enforce HTTPS except for localhost development.

    The URL is checked as a parser reads it (:func:`normalize_url`), the
    scheme and host in any case. ``" HTTP://evil.com"`` is plain HTTP to
    ``evil.com`` for httpx once a caller strips it, and for every harness
    that reads the URL from its config (#274).

    Args:
        url: URL to validate.
        label: Human-readable label for error messages.

    Raises:
        ValueError: If URL uses HTTP and is not a loopback host.
    """
    candidate = normalize_url(url)
    if _PLAIN_HTTP_RE.match(candidate) and not _LOCALHOST_HTTP_RE.match(candidate):
        raise ValueError(
            f"{label} must use HTTPS for security (got: {candidate}). "
            "HTTP is only allowed for localhost development."
        )
