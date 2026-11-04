"""Shared HTTP utilities for Kagura Memory SDK clients."""

from __future__ import annotations

import re
import uuid
from collections.abc import Collection, Mapping, Sequence
from datetime import UTC, datetime, time, timedelta
from importlib.metadata import version as _pkg_version
from typing import Any, NamedTuple, NoReturn, TypeVar
from urllib.parse import parse_qsl, quote, unquote_plus, urlencode, urlsplit, urlunsplit

import httpx
from pydantic import BaseModel, ValidationError

from .exceptions import (
    KaguraAuthError,
    KaguraCapacityLockedError,
    KaguraConnectionError,
    KaguraError,
    KaguraFeatureNotAvailableError,
    KaguraQuotaError,
    KaguraRateLimitError,
    KaguraResponseError,
    KaguraServiceUnavailableError,
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
    mcp_url: str,
    *,
    guardrails: str | None = None,
    tool_profile: str | None = None,
    add_tools: Sequence[str] | None = None,
    context: str | None = None,
) -> str:
    """Return ``mcp_url`` with memory-cloud's endpoint query set.

    The keys are ``guardrails``, ``profile``, ``add_tools`` and ``context``.
    The rest of the existing query is kept verbatim. A key being set replaces
    every earlier value of it rather than appending, because the server reads
    only the first value of each. ``None`` leaves that key as it is.

    Args:
        mcp_url: The MCP endpoint URL, with or without a query.
        guardrails: ``?guardrails=`` value (see :func:`normalize_guardrails`).
        tool_profile: ``?profile=`` value, the tool profile ``tools/list``
            serves (``core`` or ``full``); the server rejects an unknown name.
            Without one, memory-cloud 0.93.0+ lists ``core`` (memory-cloud#1849).
        add_tools: ``?add_tools=`` names, added to the listed profile
            (server v0.129.0+, memory-cloud#2354), normalized by
            :func:`normalize_add_tools`. A list that names no tool leaves the
            key as it is, like ``None``.
        context: ``?context=`` value, the connection's default context: a
            context UUID (server v0.114.0+, memory-cloud#2125).

    Returns:
        The URL to POST MCP requests to.

    Raises:
        TypeError: ``add_tools`` is a ``str``, or holds a non-``str`` item.
        ValueError: ``add_tools`` names more than 100 tools or one starting
            with ``-``, or ``context`` is not a UUID.
    """
    added = normalize_add_tools(add_tools) if add_tools is not None else ""
    default_context = normalize_uuid(context, label="context") if context is not None else None
    updates = {
        key: value
        for key, value in (
            ("guardrails", guardrails),
            ("profile", tool_profile),
            ("add_tools", added or None),
            ("context", default_context),
        )
        if value is not None
    }
    if not updates:
        return mcp_url
    parts = urlsplit(mcp_url)
    kept = _query_without(parts.query, updates)
    # A literal comma keeps an add_tools list readable in the stored entry; the
    # server splits the decoded value either way.
    query = urlencode(updates, safe=",")
    return urlunsplit(parts._replace(query="&".join([*kept, query])))


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
        Whether its query has a ``tools`` parameter, even a blank one.
    """
    return mcp_url_tool_view(mcp_url).tools is not None


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


#: The most names memory-cloud reads from one ``add_tools`` value (``MAX_TOOL_NAMES``).
MAX_ADD_TOOLS = 100


def normalize_add_tools(names: Sequence[str]) -> str:
    """Validate ``?add_tools=`` tool names and join them as the server reads them.

    memory-cloud (v0.129.0+, memory-cloud#2354) adds the names to the profile
    the URL lists. It trims them, compares them case-sensitively, ignores an
    unknown one, and reads at most 100. An item may itself be comma-separated
    (``"explore,list_edges"``, as the CLI flags take them); blank names are
    dropped and duplicates removed, keeping the order. A name starting with
    ``-`` is refused: no tool has one, and ``kagura-mcp --add-tools -x`` in a
    stored entry would stop the proxy from starting.

    Args:
        names: Tool names, e.g. ``["explore"]``.

    Returns:
        The names joined with ``,``; ``""`` when none is left.

    Raises:
        TypeError: ``names`` is a ``str`` (it would be read one character at
            a time), or an item is not a ``str``.
        ValueError: More than 100 names, which the server would cut off, or a
            name starting with ``-``.
    """
    if isinstance(names, str):
        raise TypeError("add_tools must be a list of tool names, not a str")
    found: list[str] = []
    for item in names:
        if not isinstance(item, str):
            raise TypeError(f"add_tools items must be str, got {type(item).__name__}")
        found.extend(name for part in item.split(",") if (name := part.strip()))
    unique = list(dict.fromkeys(found))
    if dashed := next((name for name in unique if name.startswith("-")), None):
        raise ValueError(f"add_tools names cannot start with '-': {dashed!r}")
    if len(unique) > MAX_ADD_TOOLS:
        raise ValueError(f"add_tools accepts at most {MAX_ADD_TOOLS} tool names, got {len(unique)}")
    return ",".join(unique)


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
    sessions in process memory and drops them after an idle hour and on
    every restart. It re-adopts such an unknown id for the authenticated
    caller (#1686), but since v0.82.0 (memory-cloud#1740) only one in the
    server-minted shape (``mcp-`` + 16 hex). It answers ``404`` + ``-32603``
    for any other unknown id, for an id ended with ``DELETE /mcp`` (remembered
    for the idle timeout; a restart forgets it) and for another user's or
    workspace's session. Those are the cases this path handles.

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

    Since memory-cloud v0.80.1 (#1719) no literal 403 message contains these
    markers (a server guard test checks), so the server's reason passes
    through. That includes the Web UI refusal "API keys and OAuth
    access tokens are not accepted on Web UI endpoints. Sign in with a
    browser session." Before v0.80.1 that sentence read "Bearer" and was
    dropped here, which left only the client's generic hint.
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
        "content_chunk_contexts",  # v0.113.0, memory-cloud#2103
        "llm_selector_contexts",  # v0.115.0, memory-cloud#2144
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
    """Build the typed exception for a plan, quota or capacity refusal (#256).

    Branches on ``gate`` first (memory-cloud v0.75.0+): ``"capacity"`` is
    the Free capacity lock (v0.99.0+, #1941), ``"quota"`` a quota,
    ``"plan"`` / ``"allowlist"`` / ``"deployment"`` a feature gate —
    whatever the error code. Without a ``gate`` (an older server, or a kind
    this SDK does not know) the code decides: MCP ``capacity_locked`` and
    REST ``CAPACITY-001`` are the capacity lock; ``plan_required``,
    ``feature_not_available`` and ``FEAT-001`` are feature gates;
    ``quota_exceeded`` and ``QUOTA-001`` are quotas only when they name a
    known ``quota_type`` or a retry window, because the same code also
    covers limits no plan lifts, such as the REST 1 MB memory-size guard
    (MCP answers that one ``payload_too_large`` since v0.108.0, #2003, which
    is no gate code). MCP ``rate_limit_exceeded`` (the daily MCP call cap) is
    the ``api_mcp_daily`` quota, resetting at the next UTC midnight.

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
        :class:`KaguraCapacityLockedError`, :class:`KaguraQuotaError` or
        :class:`KaguraFeatureNotAvailableError`, each carrying every
        descriptor field as ``details``; a plain :class:`KaguraError` for a
        quota code that is neither a tier quota nor a retry window; ``None``
        when ``code`` is not a plan, quota or capacity refusal. The error's
        ``code`` is ``code``.
    """
    typed = _gate_error(code, message, fields, retry_after=retry_after)
    if typed is not None:
        annotate_server_error(typed, code, fields)
    return typed


def _gate_error(
    code: str,
    message: str,
    fields: Mapping[str, Any],
    *,
    retry_after: int | None,
) -> KaguraError | None:
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
    if gate == _CAPACITY_GATE or code in _CAPACITY_CODES:
        return _capacity_error(message, fields, details)
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

    Maps ``401`` → :class:`~kagura_memory.exceptions.KaguraAuthError`, a ``403``
    ``insufficient_scope`` (an OAuth token without the scope the call needs,
    memory-cloud v0.79.0+, #1686) → :class:`KaguraAuthError` naming the scope,
    ``429`` → :class:`~kagura_memory.exceptions.KaguraRateLimitError`
    (honoring a numeric ``Retry-After`` header), ``503`` →
    :class:`~kagura_memory.exceptions.KaguraServiceUnavailableError` (its
    ``Retry-After`` and reason), and every other status →
    :class:`~kagura_memory.exceptions.KaguraConnectionError`. The server-supplied
    ``detail`` (a FastAPI string or validation-error list) is appended when
    present — surfacing e.g. which field failed a 422 — otherwise the
    exception's own message is used so the status is never left bare. The
    error code and fields the body carried (a JSON-RPC error's ``data``, the
    REST ``details``) are kept on the error's ``code`` / ``details``. Plan,
    quota and capacity refusals are not typed here (see :func:`gate_error`):
    the REST-backed ``KaguraClient`` methods it serves are listing and status
    reads, which memory-cloud keeps open under the Free capacity lock (#1941).
    This function always raises.
    """
    response = e.response
    status = response.status_code
    if status == 401:
        raise KaguraAuthError("Authentication failed. Check your API key.") from e
    if status == 403 and (scope_error := insufficient_scope_error(response)) is not None:
        raise scope_error from e
    detail = extract_detail(response) or _exc_message(e)
    if status == 429:
        err: KaguraError = KaguraRateLimitError(
            f"Rate limit exceeded (HTTP 429): {detail}",
            retry_after=_retry_after_seconds(response),
        )
    elif status == 503:
        err = service_unavailable_error(response, f"HTTP 503: {detail}")
    else:
        err = KaguraConnectionError(f"HTTP {status}: {detail}")
    if (fields := server_error_fields(response)) is not None:
        annotate_server_error(err, *fields)
    raise err from e


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


def path_segment(value: str, *, label: str) -> str:
    """``value`` percent-encoded as ONE segment of a REST path.

    Encoded whole, so a ``/``, ``?``, ``#`` or ``%`` stays inside the
    segment. ``.``, ``..`` and the empty string are refused instead: no
    encoding changes them, and URL resolution then drops or climbs the
    segment — ``remove_member(ws, "..")`` would send ``DELETE`` to
    ``/api/v1/workspaces/<ws>``, the workspace itself (#285).

    Raises:
        ValueError: If ``value`` is empty, ``.`` or ``..``.
    """
    if value in ("", ".", ".."):
        raise ValueError(
            f"{label} must not be {value!r}: as a URL path segment it would "
            "address a different endpoint"
        )
    return quote(value, safe="")


def public_id_segment(value: object, *, label: str) -> str:
    """A resource id (``int`` or opaque ``str``) as ONE REST path segment.

    memory-cloud v0.89.0 (#1008) replaces the integer ids of API keys,
    resource tokens and workspace invitations with opaque prefixed strings
    (``akey_…``, ``rtok_…``, ``winv_…``) and answers integer path ids with
    422; older servers still use integers. Both are accepted. A ``bool`` or
    ``float`` is refused instead of coerced — ``int(7.9)`` would silently
    target a DIFFERENT resource on a destructive endpoint — and a string is
    encoded by :func:`path_segment`, so a blank, ``.`` or ``..`` id is
    refused and a ``/`` cannot reach another endpoint.

    Raises:
        ValueError: If ``value`` is not an int or a non-blank string.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, str) and value.strip():
        return path_segment(value, label=label)
    raise ValueError(f"{label} must be an integer or a non-empty string id, got {value!r}")


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


def mcp_url_query_values(mcp_url: str) -> dict[str, str]:
    """The first value of each parameter in ``mcp_url``'s query, as memory-cloud reads it.

    The server reads only the first value of a repeated parameter. A blank
    value is kept: ``?profile=`` is present (and refused), not absent.

    Args:
        mcp_url: The MCP endpoint URL.

    Returns:
        Decoded parameter name → its first decoded value.
    """
    values: dict[str, str] = {}
    for key, value in parse_qsl(urlsplit(mcp_url).query, keep_blank_values=True):
        values.setdefault(key, value)
    return values


class McpToolView(NamedTuple):
    """The ``tools/list`` view an MCP URL's query selects (see :func:`mcp_url_tool_view`)."""

    #: The first ``profile`` value, trimmed; ``""`` for a blank one, which the
    #: server refuses; ``None`` without one (the server's default profile).
    profile: str | None
    #: The names in the first ``tools`` value; ``()`` when it names none, which
    #: the server refuses; ``None`` without one. When set, the server ignores
    #: ``profile`` and ``add_tools``.
    tools: tuple[str, ...] | None
    #: The names in the first ``add_tools`` value; ``()`` without one or when it
    #: names none (it then adds nothing).
    add_tools: tuple[str, ...]


def mcp_url_tool_view(mcp_url: str) -> McpToolView:
    """The ``profile`` / ``tools`` / ``add_tools`` in ``mcp_url``'s query, as the server reads them.

    The one reader of a URL's tool view (#311), for ``kagura setup``'s re-run
    notes and digest previews and for ``kagura doctor``. It follows the
    server's ``select_tool_definitions`` (memory-cloud v0.142.0): the first
    value of a repeated parameter wins, even a blank one
    (:func:`mcp_url_query_values`), ``profile`` is trimmed, and a name list
    is read as ``_read_names`` reads it — its first 100 comma-separated
    items, trimmed, blank names dropped, duplicates removed in order. Which
    parameter wins (``tools`` over the other two) is left to the caller.

    Args:
        mcp_url: The MCP endpoint URL.

    Returns:
        The view; every field empty for a URL without these parameters.
    """
    query = mcp_url_query_values(mcp_url)
    profile = query.get("profile")
    tools = query.get("tools")
    return McpToolView(
        profile=profile.strip() if profile is not None else None,
        tools=_tool_names(tools) if tools is not None else None,
        add_tools=_tool_names(query.get("add_tools", "")),
    )


def _tool_names(value: str) -> tuple[str, ...]:
    """The tool names memory-cloud reads from one ``tools`` / ``add_tools`` value."""
    parts = value.split(",", MAX_ADD_TOOLS)[:MAX_ADD_TOOLS]
    return tuple(dict.fromkeys(name for part in parts if (name := part.strip())))


# ---------------------------------------------------------------------------
# Server error fields, capacity lock, scope and 503 (memory-cloud v0.78.0+, #299)
# ---------------------------------------------------------------------------

# The Free capacity lock (memory-cloud v0.99.0+, #1941): REST ``403
# CAPACITY-001`` with ``details.gate == "capacity"``, MCP ``capacity_locked``.
_CAPACITY_GATE = "capacity"
_CAPACITY_CODES = frozenset({"capacity_locked", "CAPACITY-001"})
_INSUFFICIENT_SCOPE = "insufficient_scope"
# One item of a ``WWW-Authenticate`` value: a ``name=value`` parameter (the
# value a quoted string, RFC 9110 §11.2, or a bare token), else a bare token —
# an auth-scheme, or the token68 some schemes carry. One value can hold several
# challenges: httpx joins repeated headers with ", " (RFC 9110 §11.6.1).
_CHALLENGE_ITEM_RE = re.compile(
    r'([A-Za-z_][\w-]*)\s*=\s*(?:"((?:[^"\\]|\\.)*)"|([^\s,"]+))|([^\s,="]+=*)'
)
_QUOTED_PAIR_RE = re.compile(r"\\(.)")


def _capacity_error(
    message: str, fields: Mapping[str, Any], details: dict[str, Any]
) -> KaguraCapacityLockedError:
    counts = (
        "memory_count",
        "memory_limit",
        "over_memories",
        "used_bytes",
        "storage_limit_bytes",
        "over_bytes",
    )
    return KaguraCapacityLockedError(
        message,
        cleanup_url=_opt_str(fields.get("cleanup_url")),
        **{key: _opt_int(fields.get(key)) for key in counts},
        details=details,
    )


def annotate_server_error(err: KaguraError, code: str, fields: Mapping[str, Any]) -> KaguraError:
    """Keep the server's error ``code`` and its other fields on ``err``.

    Args:
        err: The error the SDK built for the refusal or failure.
        code: The server's error code (MCP ``error``, REST ``error``, OAuth
            ``error``, or a JSON-RPC ``data.error``).
        fields: What came with it — the MCP envelope, the REST ``details``
            or the JSON-RPC ``data``. ``status`` / ``error`` / ``message``
            frame the envelope and are left out.

    Returns:
        ``err``, whose ``code`` and ``details`` are set unless the error
        already had its own (a typed error builds ``details`` itself).
    """
    if err.code is None:
        err.code = code
    if not err.details:
        err.details = {k: v for k, v in fields.items() if k not in _ENVELOPE_KEYS}
    return err


def server_error_fields(response: httpx.Response) -> tuple[str, dict[str, Any]] | None:
    """``(code, fields)`` of an error body that names an error code, else ``None``.

    Reads the three shapes memory-cloud answers with: the REST envelope
    ``{"error": <code>, "message": ..., "details": {...}}`` (``fields`` is
    ``details``); a JSON-RPC error whose ``data`` carries ``error`` — the
    ``/mcp`` transport's refusals and server failures, with ``cause``,
    ``correlation_id``, ``help`` and ``retryable`` (memory-cloud v0.82.0+,
    #1742; ``fields`` is ``data``); and the OAuth-style ``{"error": <code>,
    "error_description": ...}`` (``fields`` is the body without ``error``).

    Args:
        response: A non-2xx response.
    """
    try:
        body = response.json()
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(body, dict):
        return None
    error = body.get("error")
    if isinstance(error, dict):
        data = error.get("data")
        if isinstance(data, dict) and isinstance(data.get("error"), str):
            return data["error"], data
        return None
    if not isinstance(error, str):
        return None
    details = body.get("details")
    if isinstance(details, dict):
        return error, details
    return error, {k: v for k, v in body.items() if k != "error"}


def _bearer_challenge(response: httpx.Response) -> dict[str, str]:
    """The parameters of ``response``'s ``WWW-Authenticate: Bearer`` challenge (RFC 6750 §3).

    The Bearer challenge may come after another one (``Basic realm="x",
    Bearer error=...``, e.g. a proxy's challenge joined with the server's).
    Names are lower-cased; ``{}`` when there is no Bearer challenge.
    """
    header = response.headers.get("WWW-Authenticate")
    if not isinstance(header, str):
        return {}
    params: dict[str, str] | None = None
    for match in _CHALLENGE_ITEM_RE.finditer(header):
        scheme = match.group(4)
        if scheme is not None:
            if params is not None:
                break  # the next challenge starts: the Bearer one is complete
            if scheme.lower() == "bearer":
                params = {}
        elif params is not None:
            quoted = match.group(2)
            params[match.group(1).lower()] = (
                _QUOTED_PAIR_RE.sub(r"\1", quoted) if quoted is not None else match.group(3)
            )
    return params or {}


def insufficient_scope_error(response: httpx.Response) -> KaguraAuthError | None:
    """The :class:`KaguraAuthError` for a 403 ``insufficient_scope``, else ``None``.

    memory-cloud v0.79.0+ (#1686) refuses a ``tools/call`` whose OAuth token
    lacks the tool's scope — ``memory:read`` for the read-only tools,
    ``recall`` and ``get_agent_bootstrap``, ``memory:write`` for every other
    tool — with HTTP 403, a ``WWW-Authenticate: Bearer
    error="insufficient_scope"`` challenge and a JSON-RPC error whose
    ``data`` names ``required_scope``. REST ``/api/v1/*`` sends the challenge
    with ``scope="<required>"``. ``data.required_scope`` wins: the ``/mcp``
    challenge's ``scope`` is the granted scopes plus the required one, the
    value to re-authorize with.

    Args:
        response: A 403 response.

    Returns:
        The error, with ``code`` ``"insufficient_scope"`` and
        ``details["required_scope"]`` when the scope is known; ``None`` for
        any other 403.
    """
    fields = server_error_fields(response)
    data = fields[1] if fields is not None and fields[0] == _INSUFFICIENT_SCOPE else None
    challenge = _bearer_challenge(response)
    if data is None and challenge.get("error") != _INSUFFICIENT_SCOPE:
        return None
    scope = _opt_str((data or {}).get("required_scope")) or challenge.get("scope") or None
    needs = f"the {scope} scope" if scope else "a scope"
    err = KaguraAuthError(
        f"The OAuth sign-in lacks {needs} this call needs (HTTP 403 insufficient_scope). "
        "Run `kagura auth login` to sign in again with it; a `--read-only` login grants "
        "memory:read only."
    )
    annotate_server_error(err, _INSUFFICIENT_SCOPE, data or {})
    if scope:
        err.details["required_scope"] = scope
    return err


def service_unavailable_error(
    response: httpx.Response, message: str
) -> KaguraServiceUnavailableError:
    """The :class:`KaguraServiceUnavailableError` for an HTTP 503 ``response``.

    ``retry_after`` is the ``Retry-After`` header, else a JSON-RPC
    ``data.retry_after_seconds``; ``reason`` is the REST ``details.reason``
    (``DB-002``, memory-cloud v0.104.0+), else the MCP ``cause``, else the
    error code (``/mcp``'s ``temporarily_unavailable``, v0.82.0+).

    Args:
        response: The 503 response.
        message: Message for the exception.
    """
    fields = server_error_fields(response)
    code, data = fields if fields is not None else (None, {})
    retry_after = _retry_after_seconds(response)
    if retry_after is None:
        retry_after = _opt_int(data.get("retry_after_seconds"))
    reason = _opt_str(data.get("reason")) or _opt_str(data.get("cause")) or code
    err = KaguraServiceUnavailableError(message, retry_after=retry_after, reason=reason)
    if code is not None:
        annotate_server_error(err, code, data)
    return err
