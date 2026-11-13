"""Low-level REST API client for Kagura Memory Cloud."""

import asyncio
import itertools
import json
import logging
import math
import re
import uuid
import warnings
from collections.abc import AsyncGenerator
from contextlib import aclosing
from datetime import UTC, date, datetime
from typing import Any, Literal, Self, TypeVar, get_args, overload
from urllib.parse import quote

import httpx
from pydantic import BaseModel as _BaseModel

from ._auth import _OAuthAuth, _resolve_auth, _StaticAuth
from ._http import (
    SDK_VERSION,
    ErrorBody,
    _opt_int,
    annotate_server_error,
    base_url_from_mcp,
    gate_error,
    mcp_session_expired,
    mcp_session_header,
    parse_response,
    parse_response_list,
    raise_for_kagura_status,
    validate_coordinate,
    validate_https_url,
    validate_lat_lon,
    version_conflict_error,
)
from ._version import meets_minimum, parse_version
from .exceptions import (
    KaguraConnectionError,
    KaguraError,
    KaguraNotFoundError,
    KaguraPartialRollbackError,
    KaguraPartialSessionRollbackError,
    KaguraResponseError,
    _exc_message,
)
from .models import (
    _BOOTSTRAP_MIN_CHARS,
    _MAX_CHARS_MAX,
    _MAX_CHARS_MIN,
    _OBSERVED_DATE,
    _RECALL_FULL_MAX,
    Agent,
    AgentBinding,
    AgentBootstrapComponentName,
    AgentBootstrapResponse,
    BootstrapComponentName,
    BootstrapFormat,
    BootstrapResponse,
    ChangesSinceResponse,
    ContextInfo,
    ContextTagsResponse,
    DedupeMode,
    DuplicatesResponse,
    Edge,
    EmbeddingModelsResponse,
    EmbeddingStatus,
    ForgetBatchResult,
    GuardrailSet,
    GuideResponse,
    ListMatchingResponse,
    ListOrderBy,
    ListTagsResponse,
    MeasurementAggregate,
    MeasurementPeriod,
    MeasurementResult,
    MeasurementSeries,
    MemoryListResponse,
    MemoryStatsResponse,
    RecallVerbosity,
    RememberBatchResult,
    RevisionsPage,
    RollbackResult,
    RollbackSummary,
    ServerInfo,
    SessionRollbackResult,
    SessionRollbackSummary,
    SleepReport,
    SleepReportDetail,
    ToolDescriptions,
    ToolListResponse,
    ToolTrigger,
    UsageInfo,
    _agent_update_payload,
    _binding_scope_payload,
    _bootstrap_payload,
    _details_with_tool_trigger,
    _validate_int_range,
    _validate_max_chars,
    _validate_or_none,
)

_T = TypeVar("_T", bound=_BaseModel)

_logger = logging.getLogger("kagura_memory")


MIN_SERVER_VERSION = "0.17.1"
"""Minimum memory-cloud server version this SDK was tested against.

This is the lowest server version where every parameter the SDK exposes
(remember/recall pass-through fields, resource APIs) is fully supported.
The check is opt-in: callers must explicitly invoke
:meth:`KaguraClient.check_server_version` to log an advisory warning when
the connected server is older. Plain ``KaguraClient`` instantiation and
tool calls never raise on version mismatch, and older servers may
silently ignore unknown parameters."""

_min_server_version = parse_version(MIN_SERVER_VERSION)
if _min_server_version is None:  # pragma: no cover - a malformed constant fails at import
    raise ValueError(f"MIN_SERVER_VERSION is not MAJOR.MINOR.PATCH: {MIN_SERVER_VERSION!r}")
_MIN_SERVER_VERSION_TUPLE: tuple[int, int, int] = _min_server_version

# Measurement-lane column caps (memory-cloud #1333: ``measurements.metric`` is
# VARCHAR(64), ``unit`` VARCHAR(32)). Checked locally only to spare a round-trip
# that could return nothing but a validation_error — the server stays the
# authority on every other rule (e.g. the 365-day series window).
_METRIC_MAX_LEN = 64
_UNIT_MAX_LEN = 32

# Shared by KaguraClient.setup_resource and ResourceClient.setup_resource (#273).
_SETUP_SUMMARY_DEPRECATED = (
    "setup_resource(summary=...) is deprecated and ignored: the server's setup_resource "
    "has no summary, so it is not sent. Set it afterwards with "
    "update_context(context_id, summary=...) on the returned context_id."
)


def _validate_metric(metric: object) -> None:
    """Reject a series name the server would reject (non-empty, <= 64 chars).

    Raises:
        ValueError: If ``metric`` is not a non-empty string within the cap.
    """
    if not isinstance(metric, str) or not metric:
        raise ValueError(f"metric must be a non-empty string, got {metric!r}")
    if len(metric) > _METRIC_MAX_LEN:
        raise ValueError(f"metric must be at most {_METRIC_MAX_LEN} characters, got {len(metric)}")


def _validate_measurement_value(value: object) -> None:
    """Reject a measurement value that is not a finite number.

    A string is rejected rather than coerced (the parameter is typed
    ``float``), and ``bool`` — an ``int`` subclass — is never a measurement.
    NaN / infinity would poison every aggregate of the series.

    Raises:
        ValueError: If ``value`` is not a finite ``int`` / ``float``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"value must be a number, got {type(value).__name__}")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        # An int beyond float range makes isfinite() raise instead of False.
        finite = False
    if not finite:
        raise ValueError("value must be finite (NaN and infinity are rejected)")


def _iso_arg(value: str | datetime) -> str:
    """Render a time argument for an MCP tool that takes ISO 8601 strings.

    A ``datetime`` is serialized with :meth:`datetime.isoformat` — naive means
    UTC to the server, aware keeps its offset (the server normalizes it). A
    string passes through untouched for the server to parse.
    """
    return value.isoformat() if isinstance(value, datetime) else value


# ``observed_at`` input forms memory-cloud accepts (utils/observed_at.py, #2330):
# extended ISO 8601 only, at most 40 characters, checked before the calendar.
_OBSERVED_AT_MAX_CHARS = 40
_OBSERVED_INPUT_DATETIME = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]+)?(Z|[+-][0-9]{2}:[0-9]{2})?"
)
_OBSERVED_AT_HINT = (
    "observed_at must be an ISO 8601 date (2024-03-07) or datetime (2024-03-07T10:00:00Z)"
)


def _observed_at_arg(value: object) -> str:
    """Render an ``observed_at`` argument and refuse a form the server refuses.

    A ``datetime`` becomes the server's stored form, ``YYYY-MM-DDTHH:MM:SSZ``:
    naive (no UTC offset) means UTC (as for
    :meth:`KaguraClient.record_measurement`), an aware one is converted, and
    fractional seconds are dropped as the server drops them. A ``date``
    becomes ``YYYY-MM-DD``. A string is sent as given once it passes the
    server's own checks: at most 40 characters as sent (the request model's
    cap), then, stripped, a date ``YYYY-MM-DD`` or ``YYYY-MM-DDTHH:MM:SS``
    with optional fractional seconds and an optional ``Z`` / ``±HH:MM``
    offset, a real calendar day and time, and an instant in UTC years
    1-9999 (memory-cloud v0.127.0, #2330; basic forms such as ``20240307``,
    a space separator or a time without seconds are refused there with
    ``validation_error``).

    Raises:
        ValueError: ``value`` is not such a string, ``date`` or ``datetime``.
    """
    if isinstance(value, datetime):
        try:
            moment = value if value.utcoffset() is None else value.astimezone(UTC)
        except OverflowError as e:
            # The offset moves the instant out of years 1-9999, as the server finds.
            raise ValueError(f"{_OBSERVED_AT_HINT}, got {value!r}") from e
        return (
            f"{moment.year:04d}-{moment.month:02d}-{moment.day:02d}"
            f"T{moment.hour:02d}:{moment.minute:02d}:{moment.second:02d}Z"
        )
    if isinstance(value, date):
        return value.isoformat()
    if not isinstance(value, str):
        raise ValueError(f"{_OBSERVED_AT_HINT}, got {type(value).__name__}")
    text = value.strip()
    # The cap is on the argument as sent (``max_length=40``), before the strip.
    if not text or len(value) > _OBSERVED_AT_MAX_CHARS:
        raise ValueError(f"{_OBSERVED_AT_HINT}, got {value!r}")
    try:
        if _OBSERVED_DATE.fullmatch(text):
            date.fromisoformat(text)
        elif _OBSERVED_INPUT_DATETIME.fullmatch(text):
            if text[11:13] == "24":
                # Python 3.14 reads 24:00:00 as the next midnight; the
                # server's parser (Python 3.11) refuses the hour.
                raise ValueError(text)
            parsed = datetime.fromisoformat(text)
            if parsed.tzinfo is not None:
                # OverflowError for an offset that leaves years 1-9999.
                parsed.astimezone(UTC)
        else:
            raise ValueError(text)
    except (ValueError, OverflowError) as e:
        raise ValueError(f"{_OBSERVED_AT_HINT}, got {value!r}") from e
    return value


def _check_observed_at_agrees(observed_at: str, details: dict[str, Any] | None, name: str) -> None:
    """Refuse an ``observed_at`` that ``details`` contradicts, as the server does.

    The server folds the argument into ``details.observed_at`` and compares
    the two values as sent (``MemoryService._fold_observed_at``), so a
    ``details_patch`` ``"observed_at": None`` — a key delete — disagrees too.

    Raises:
        ValueError: ``details`` holds an ``observed_at`` other than ``observed_at``.
    """
    if details is not None and "observed_at" in details and details["observed_at"] != observed_at:
        raise ValueError(
            f"{name}['observed_at'] and observed_at disagree; give one of them "
            "(the server answers validation_error)"
        )


class KaguraClient:
    """
    Low-level REST API client for Kagura Memory Cloud MCP tools.

    All methods may raise:
        KaguraAuthError: Authentication failed — also an OAuth token that
            lacks the scope a tool needs (HTTP 403 ``insufficient_scope``,
            memory-cloud v0.79.0+: ``memory:read`` for the read-only tools,
            ``recall`` and ``get_agent_bootstrap``, ``memory:write`` for the
            rest), whose message names the scope
        KaguraConnectionError: Connection to server failed; its subclass
            :class:`KaguraServiceUnavailableError` for an HTTP 503 (``/mcp``
            could not verify the credential or reach a dependency), with
            ``retry_after`` and ``reason``
        KaguraRateLimitError: Rate limit exceeded — any HTTP 429, including
            the daily MCP quota when the server's rate-limit middleware
            refuses the request before the tool runs

    MCP tool methods additionally translate the server's structured domain
    errors (``{"status": "error", ...}``) into exceptions rather than
    returning them as data (issue #180): a missing context, memory, report or
    rollback session raises :class:`KaguraNotFoundError`, a quota refusal (including the
    in-band daily MCP call cap, ``rate_limit_exceeded``)
    :class:`KaguraQuotaError`, a plan/feature gate
    :class:`KaguraFeatureNotAvailableError` and the Free capacity lock
    (``capacity_locked``, memory-cloud v0.99.0+)
    :class:`KaguraCapacityLockedError`, each carrying the server's detail
    fields (#256), a stale ``expected_version`` (``version_conflict``)
    :class:`KaguraVersionConflictError` (#302), and any other domain error
    raises :class:`KaguraError`.
    Every one keeps the envelope's ``error`` as ``code`` and its other
    fields (``help``, and on a server failure ``cause``, ``correlation_id``,
    ``retryable``, ``outcome``) in ``details`` — branch on ``code``, never
    on the message. Callers should use ``try/except`` rather than
    inspecting ``result["status"]``. On the tool methods that return a
    model, a success payload that does not match it (a server newer than
    the SDK) raises :class:`KaguraResponseError` naming the tool (#250).
    The REST-backed methods (``get_server_info``, ``check_server_version``,
    ``get_embedding_status``, ``get_memory_stats``, ``find_duplicates``,
    ``list_memories``, ``list_embedding_models``) do the same with
    ``operation="KaguraClient.<method>"`` (#254, #277); from them a network
    failure, a non-JSON 2xx body and a non-2xx status other than 401, 429
    and a 403 ``insufficient_scope`` raise :class:`KaguraConnectionError`
    (a 503 its subclass :class:`KaguraServiceUnavailableError`).
    """

    def __init__(
        self,
        api_key: str | None = None,
        mcp_url: str | None = None,
        timeout: float = 30.0,
        profile: str | None = None,
    ):
        """Initialize Kagura API client.

        Authentication resolution order when ``api_key`` is omitted:

        1. ``KAGURA_API_KEY`` env var (highest — CI / service accounts always win).
        2. The OAuth profile from ``~/.kagura/credentials.json``,
           selected by the ``profile`` argument or the ``KAGURA_PROFILE``
           env var, falling back to ``default_profile``.
        3. ``.kagura.json`` (cwd or ``~/``) plus its own env fallback
           (existing legacy behavior).

        Args:
            api_key: Explicit Kagura API key. When omitted, the
                resolution chain above runs.
            mcp_url: Explicit MCP URL. When omitted, derived from the
                resolved credential source (OAuth profile, env, or
                ``.kagura.json``). It may carry memory-cloud's endpoint
                query, which every MCP request sends. ``?profile=<name>`` /
                ``?tools=a,b`` (v0.73.0+) and ``?add_tools=a,b`` (v0.129.0+)
                choose what ``tools/list`` (:meth:`get_tool_definitions`)
                returns: a bare URL lists the ``core`` profile on
                memory-cloud 0.93.0+ (21 tools on 0.141.0, memory-cloud#1849)
                and ``?profile=full`` every tool (72). A profile is a view,
                not an authorization boundary — ``tools/call`` reaches an
                unlisted tool, so the methods here keep working on a
                ``core`` or ``full`` URL — except that :meth:`recall`'s
                ``explore_hints`` need ``explore`` listed
                (``?add_tools=explore`` or ``?profile=full``; v0.135.0,
                memory-cloud#2470). The ``?profile=google`` view and its
                alias ``?profile=microsoft`` (v0.158.0+, memory-cloud#2773 /
                #2778) are the exception: they list the ``core`` tools
                without ``describe_tools`` and ``guide``, which
                ``?add_tools=`` cannot bring back, and refuse both at
                ``tools/call``, so :meth:`describe_tools` and :meth:`guide`
                raise :class:`KaguraError` with ``code == "unknown_tool"``
                there. On those views :meth:`recall` makes no learning
                writes and :meth:`bootstrap` carries no service ``notices``
                (v0.161.0). ``?context=<uuid>`` (v0.114.0+) names a default
                context for tools called without ``context_id``,
                ``?guardrails=off`` (v0.74.0+) drops the ``guardrails`` block
                from :meth:`get_context_info` and :meth:`bootstrap`, and
                ``?notices=off`` (v0.161.0+) keeps service notices out of
                :meth:`bootstrap` (for a headless connection whose replies
                nobody reads). REST calls derive their base URL without the
                query (#258).
            timeout: Request timeout in seconds.
            profile: Named OAuth profile to load (overrides
                ``KAGURA_PROFILE`` and the credentials file's
                ``default_profile``).
        """
        self._init_from_auth(
            _resolve_auth(api_key=api_key, mcp_url=mcp_url, profile=profile), timeout
        )

    @classmethod
    def _from_resolved_auth(
        cls, resolved: _StaticAuth | _OAuthAuth, *, timeout: float = 30.0
    ) -> Self:
        """Construct from a pre-resolved auth — internal helper.

        For a caller that picks the credential itself, e.g. an OAuth profile
        that ``KAGURA_API_KEY`` must not outrank (``kagura setup``, #260).
        Mirrors the REST clients' ``_from_resolved_auth``.
        """
        client = cls.__new__(cls)
        client._init_from_auth(resolved, timeout)
        return client

    def _init_from_auth(self, resolved: _StaticAuth | _OAuthAuth, timeout: float) -> None:
        stripped_url = resolved.mcp_url.rstrip("/")
        validate_https_url(stripped_url, label="MCP URL")

        self.mcp_url = stripped_url
        self._base_url = base_url_from_mcp(stripped_url)
        self.timeout = timeout

        if isinstance(resolved, _StaticAuth):
            # Long-lived API key path: bake the bearer header once and
            # forget the value (compliant with python.md "Never store
            # API keys as instance attributes").
            self._client = httpx.AsyncClient(
                timeout=timeout,
                headers={"Authorization": f"Bearer {resolved.api_key}"},
            )
        else:
            # OAuth path: the KaguraOAuth httpx.Auth subclass injects a
            # fresh bearer header per request and triggers refresh when
            # the access_token is within REFRESH_SKEW_SEC of expiry.
            # Concurrent KaguraClient instances pointing at the same
            # credentials file share an asyncio.Lock through the
            # module-level cache, so only one refresh fires per cycle.
            self._client = httpx.AsyncClient(
                timeout=timeout,
                auth=resolved.oauth,
            )

        self._session_id: str | None = None
        # Makes the ``initialize`` handshake single-flight (see _initialize_session).
        self._session_lock = asyncio.Lock()
        self._request_id_counter = itertools.count(1)
        # Per-(client, context_id) cache for ingest steering: get_context_info
        # is fetched at most once per context for the client's lifetime. A
        # context_id present as a key with value None is the "fetched but
        # unusable" sentinel (empty info or fetch failure) — it suppresses
        # re-fetching on every section summarization. See
        # :meth:`_get_context_info_cached`.
        self._context_info_cache: dict[str, ContextInfo | None] = {}
        # Context id (lower case) → name, for the list_tags drill-down, whose
        # REST route sends no name before memory-cloud v0.77.0 (#273). Never
        # stale: no server API renames a context (update_context cannot
        # change ``name``).
        self._context_names: dict[str, str] = {}

    def _next_request_id(self) -> int:
        """Get next JSON-RPC request ID (concurrency-safe via itertools.count)."""
        return next(self._request_id_counter)

    async def _initialize_session(self) -> None:
        """Initialize MCP session if not already initialized.

        Single-flight: calls that find no session at the same time (on first
        use, or after all hitting one expired session) share one ``initialize``
        instead of each opening, and orphaning, a session of its own.
        """
        if self._session_id:
            return
        async with self._session_lock:
            if not self._session_id:  # nobody opened one while we waited
                await self._open_session()

    async def _open_session(self) -> None:
        """Run the ``initialize`` handshake and keep the session id it returns."""
        body = {
            "jsonrpc": "2.0",
            "id": self._next_request_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "kagura-memory-sdk", "version": SDK_VERSION},
            },
        }

        try:
            response = await self._client.post(self.mcp_url, json=body)
            response.raise_for_status()

            # Extract session ID from header
            self._session_id = response.headers.get("mcp-session-id")
            if not self._session_id:
                raise KaguraConnectionError("No session ID returned from server")

        except httpx.HTTPStatusError as e:
            raise_for_kagura_status(e)
        except httpx.RequestError as e:
            raise KaguraConnectionError(f"Connection failed: {_exc_message(e)}") from e

    async def _make_jsonrpc_request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """
        Make a JSON-RPC 2.0 request to MCP server (DRY: common logic).

        Args:
            method: JSON-RPC method name (e.g., "tools/call", "tools/list")
            params: Method parameters

        Returns:
            Result dict from JSON-RPC response

        Raises:
            KaguraAuthError: Authentication failed
            KaguraConnectionError: Connection to server failed
        """
        await self._initialize_session()

        body = {
            "jsonrpc": "2.0",
            "id": self._next_request_id(),
            "method": method,
            "params": params,
        }

        try:
            response = await self._post_in_session(body)
            response.raise_for_status()

            data = response.json() or {}
            if "error" in data:
                error = data["error"]
                raise KaguraConnectionError(f"MCP error: {error.get('message', error)}")

            return data.get("result", {})

        except httpx.HTTPStatusError as e:
            raise_for_kagura_status(e)
        except httpx.RequestError as e:
            raise KaguraConnectionError(f"Connection failed: {_exc_message(e)}") from e

    async def _post_in_session(self, body: dict[str, Any]) -> httpx.Response:
        """POST ``body`` in the MCP session, re-opening the session once if it expired.

        MCP Streamable HTTP answers a request naming a session the server no
        longer holds with ``404`` and requires a new ``initialize``; without
        this a long-lived client would fail every call from then on. The
        server rejects the request before dispatch, which makes the single
        retry safe even for a non-idempotent ``tools/call``. A second ``404``
        is returned for the caller's ``raise_for_status`` to report.
        (memory-cloud re-adopts a session id it minted itself after an idle
        cleanup or a restart, so this fires for an id it ended or never
        minted; see :func:`~kagura_memory._http.mcp_session_expired`.)

        Args:
            body: The JSON-RPC request.

        Returns:
            The response to the request, or to its one retry.
        """
        session_id = self._session_id
        response = await self._client.post(
            self.mcp_url, json=body, headers=mcp_session_header(session_id)
        )
        if not mcp_session_expired(response, session_id):
            return response
        # Forget the session only while it is still the stale one, so one that
        # a concurrent call already re-opened is kept; concurrent calls that
        # hit the same expired session then share one single-flight initialize.
        if self._session_id == session_id:
            self._session_id = None
        await self._initialize_session()
        return await self._client.post(
            self.mcp_url, json=body, headers=mcp_session_header(self._session_id)
        )

    async def _rest_get_json(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        mcp_tool: str | None = None,
    ) -> Any:
        """GET a REST endpoint and return its decoded JSON body.

        Args:
            path: URL path (appended to ``_base_url``).
            params: Optional query parameters. A list value is sent as one
                repeated key per item (``?k=a&k=b``), which is how FastAPI
                reads a ``list[str]`` query.
            mcp_tool: The MCP tool this call stands in for. When set, a
                ``404`` and a ``422`` raise what that tool's
                ``context_not_found`` and ``invalid_argument`` errors raise
                (:meth:`_raise_for_mcp_error`), so a method moved from MCP to
                REST keeps its exceptions (#273); the error's ``details`` are
                the REST body's (e.g. a 422's ``errors``).

        Raises:
            KaguraAuthError / KaguraRateLimitError / KaguraConnectionError: A
                non-2xx status (see :func:`raise_for_kagura_status`).
            KaguraNotFoundError / KaguraError: A ``404`` / ``422`` when
                ``mcp_tool`` is set.
            KaguraConnectionError: A network failure, or a 2xx body that is
                not JSON.
        """
        url = f"{self._base_url}{path}"
        try:
            response = await self._client.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            # One body for both paths: the MCP-style remap and the status mapping.
            body = ErrorBody(e.response)
            code = {404: "context_not_found", 422: "invalid_argument"}.get(e.response.status_code)
            if mcp_tool is not None and code is not None:
                message = body.detail or f"HTTP {e.response.status_code}"
                try:
                    self._raise_for_mcp_error(
                        {"status": "error", "error": code, "message": message}, mcp_tool
                    )
                except KaguraError as mapped:
                    # ``code`` stays the MCP tool's; the REST body's fields are
                    # what the server sent with it.
                    if body.fields is not None:
                        annotate_server_error(mapped, *body.fields)
                    # Chained explicitly, as raise_for_kagura_status chains it.
                    raise mapped from e
            raise_for_kagura_status(e, body)
        except httpx.RequestError as e:
            raise KaguraConnectionError(f"Connection failed: {_exc_message(e)}") from e
        except (ValueError, TypeError) as e:
            raise KaguraConnectionError(f"Invalid response format: {_exc_message(e)}") from e

    async def _rest_get(
        self,
        path: str,
        model: type[_T],
        params: dict[str, Any] | None = None,
        *,
        operation: str,
        mcp_tool: str | None = None,
    ) -> _T:
        """GET a REST endpoint and parse its body into ``model``.

        :meth:`_rest_get_json` for the transport, :func:`parse_response` for
        the body, so drift raises :class:`KaguraResponseError` naming
        ``operation`` (#277; until 0.41.x these methods raised
        ``KaguraConnectionError("Invalid response format")``, with pydantic's
        text and the payload values in it).

        Args:
            path: URL path (appended to ``_base_url``).
            model: Pydantic model class for response validation.
            params: Optional query parameters.
            operation: The exception's ``operation`` and its message prefix:
                ``KaguraClient.<method>``, or the MCP tool's name when the
                call stands in for one.
            mcp_tool: Passed on to :meth:`_rest_get_json`: only with it does
                a ``404`` / ``422`` become that tool's error. Without it a
                non-2xx status stays what :func:`raise_for_kagura_status`
                makes of it.

        Returns:
            Validated model instance.

        Raises:
            KaguraAuthError / KaguraRateLimitError / KaguraConnectionError /
            KaguraNotFoundError / KaguraError:
                As :meth:`_rest_get_json` raises them.
            KaguraResponseError: The 2xx body does not match ``model``.
        """
        data = await self._rest_get_json(path, params, mcp_tool=mcp_tool)
        return parse_response(model, data, operation=operation)

    async def _call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        Call MCP tool via JSON-RPC.

        Args:
            tool_name: Tool name (e.g., "remember", "recall")
            arguments: Tool arguments

        Returns:
            Tool result parsed from content[0].text

        Raises:
            KaguraAuthError: Authentication failed
            KaguraConnectionError: Connection failed
        """
        result = await self._make_jsonrpc_request(
            method="tools/call", params={"name": tool_name, "arguments": arguments}
        )

        # Parse MCP tool response format
        # Result format: {"content": [{"type": "text", "text": "{...}"}]}
        content = result.get("content", [])
        if content:
            try:
                text = content[0].get("text", "{}")
                return json.loads(text)
            except json.JSONDecodeError as e:
                raise KaguraConnectionError(f"Invalid response format: {_exc_message(e)}") from e

        return {}

    async def _call_tool_checked(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call an MCP tool and translate domain errors into exceptions (issue #180).

        Wraps :meth:`_call_tool` with :meth:`_raise_for_mcp_error` so a server
        ``{"status": "error", ...}`` response raises :class:`KaguraNotFoundError`
        / :class:`KaguraError` instead of being returned as data. This is the
        single chokepoint every tool method routes through; use it instead of
        calling :meth:`_call_tool` directly unless a method deliberately wants
        the raw error dict.

        Args:
            tool_name: MCP tool name; also used as the operation label in the
                raised exception's message.
            arguments: Tool arguments.

        Returns:
            The parsed tool result (only when the server reported success).
        """
        result = await self._call_tool(tool_name, arguments)
        self._raise_for_mcp_error(result, tool_name)
        return result

    async def remember(
        self,
        context_id: str,
        summary: str,
        content: str,
        type: str = "note",
        importance: float = 0.5,
        tags: list[str] | None = None,
        source_uri: str | None = None,
        linked_memory_ids: list[str] | None = None,
        linked_source_uris: list[str] | None = None,
        source_type: Literal["file", "url", "vault", "api", "manual"] | None = None,
        context_summary: str | None = None,
        details: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        delivery_mode: Literal["always", "on_recall", "on_trigger"] = "on_recall",
        supersedes: str | None = None,
        *,
        tool_trigger: ToolTrigger | dict[str, Any] | None = None,
        observed_at: str | date | datetime | None = None,
        tags_normalize: bool = False,
        dedupe: DedupeMode | None = None,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Call remember MCP tool.

        ``observed_at``, ``tags_normalize``, ``dedupe`` and ``verbose`` are
        keyword-only and sent only when set, so a server that predates one
        never sees it. From memory-cloud v0.82.0 (#1742) a server refuses an
        argument it does not declare (``KaguraError``, code
        ``invalid_argument``); one before v0.82.0 ignores it.

        Args:
            context_id: Context ID.
            summary: Memory summary (10-500 chars). Since memory-cloud
                v0.108.0 (#2003) a longer one is cut at a sentence end instead
                of refused, its full text kept at the head of ``content``; the
                reply then says so (see Returns).
            content: Memory content. Always sent: the server's ``remember``
                refuses a call without it (``missing_fields``). When the
                summary already holds the whole fact, put one short line
                naming the evidence or source here (memory-cloud#1936).
            type: Memory type. Server validates against its own vocabulary;
                the SDK passes through.
            importance: Importance score (0.0-1.0).
            tags: Optional tags: at most 50, each at most 100 characters —
                the caps memory-cloud v0.82.0+ enforces on a write
                (#1743), checked before any request.
            source_uri: Origin URI (e.g. ``file:///``, ``https://``,
                ``vault://``).
            linked_memory_ids: Existing memory UUIDs to declare as graph
                edges from this memory. Server creates ``declared_link``
                edges with ``weight=1.0`` atomically. Like
                ``linked_source_uris`` and ``supersedes``, it links only the
                caller's own memories and skips any other silently; in a
                private context a memory of an account linked to the
                caller's counts too (memory-cloud v0.90.0+, #1803).
                Identities are per ``user_id`` and never matched by email
                (v0.86.0, #1783): what one person writes with an API key and
                with an OAuth sign-in belongs to two creators unless the
                accounts are linked.
            linked_source_uris: Source URIs to resolve into linked memories.
                Unresolved URIs are silently skipped server-side.
            source_type: Origin classification. Closed enum per the MCP
                tool schema; pairs with ``source_uri`` for downstream filters.
            context_summary: Brief explanation (max 2000 chars) of why the
                memory exists and how to use it. Distinct from ``summary``,
                which is the search-target text.
            details: Structured details JSON. Use for additional metadata
                like code locations, parent/child links, or any caller-defined
                payload that the server should store as-is. Reserved keys the
                server validates: ``location``, ``trigger`` (``type="time"``)
                and ``tool_trigger`` (see ``tool_trigger`` below).
            context: Open-ended context metadata JSON. Less structured than
                ``details``; useful for free-form provenance hints.
            delivery_mode: When the memory is surfaced. ``"on_recall"``
                (default) leaves it to probabilistic :meth:`recall`.
                ``"always"`` pins it to ``scope="persistent"`` on write so it
                is returned by every :meth:`load_pinned` call (Goal / Guardrail
                / critical-policy memories). ``"on_trigger"`` is for
                trigger-delivered memories. The default is sent only when it
                differs from the server's ``server_default='on_recall'``, so
                an unset value stays forward-compatible. Keyword-only in
                practice — keep passing it by name. (Appended at the end of the
                signature so existing positional callers are unaffected.)
            supersedes: UUID of a memory this one replaces. The old memory is
                **shadowed, not deleted**: it drops out of default
                :meth:`recall` but stays reachable via
                ``recall(include_superseded=True)`` and :meth:`explore`, and
                deleting the edge restores it. Prefer this
                over :meth:`forget` + :meth:`remember` when storing a newer
                version of a fact — that pair destroys the history the
                supersede edge exists to preserve. Requires memory-cloud
                server v0.45.0+ (#1208). The old memory must be the caller's
                own (or, in a private context, a linked account's; #1803),
                else no edge is written and nothing is raised: a member
                cannot shadow another member's memory.
            tool_trigger: Mark the memory as a **tool guardrail** (memory-cloud
                v0.74.0+): a client hook injects its summary when a matching
                tool call happens, and :meth:`load_guardrails` serves it. Sent
                as ``details["tool_trigger"]``, merged with any other
                ``details`` keys; see :class:`~kagura_memory.models.ToolTrigger`
                for the fields. Needs context **editor** or above (else
                ``permission_denied``) and a user credential (an agent-bound
                key gets ``validation_error`` /
                ``tool_trigger_requires_user_credential``). The server
                validates the patterns and returns
                ``invalid details.tool_trigger: <code>: ...`` on a bad one.
                Keyword-only.
            observed_at: When the fact held or the event happened — never the
                write time (memory-cloud v0.127.0+, #2330). A ``YYYY-MM-DD``
                date or an extended ISO 8601 datetime string, or a ``date`` /
                ``datetime`` (a naive ``datetime`` means UTC). The server
                stores it as ``details.observed_at``: a date as written, a
                datetime in UTC as ``YYYY-MM-DDTHH:MM:SSZ``. :meth:`recall`
                filters on it (``observed_after`` / ``observed_before``) and
                its rows, :meth:`reference` and :meth:`list_memories` carry
                it. Checked locally like the server checks it. Keyword-only.
            tags_normalize: ``True`` stores each tag under the context's
                established spelling when the two differ only by case,
                separators, a simple plural or (v0.110.0+, #2028) hiragana /
                katakana or one word-final ``ー``, and reports each mapping in
                ``lint`` as ``{code: "tag_normalized", subject, replacement}``
                (memory-cloud v0.93.0+, #1853; tags whose numbers differ are
                never mapped, v0.95.0+, #1871). Sent only when ``True``.
                Keyword-only.
            dedupe: ``"suggest"`` (the server default), ``"check"`` or
                ``"off"`` (memory-cloud v0.93.0+, #1853; see
                :data:`~kagura_memory.models.DedupeMode`). With ``"check"`` a
                near-duplicate (similarity ≥ 0.85 to a live memory of the
                context) writes nothing and the reply is ``{status:
                "duplicate_candidate", candidate: {memory_id, summary,
                similarity, supersedable?}}`` — returned, not raised: store
                with ``supersedes=<id>``, update the existing memory, or
                repeat with ``"off"`` (``supersedable: False`` marks another
                member's memory, which ``supersedes`` cannot take). ``"off"``
                writes and opts this memory out of supersede suggestions for
                good (#2024). Keyword-only.
            verbose: ``True`` asks for ``persistence.detail``, the lifecycle
                prose the server sends only with ``verbose=true`` since
                memory-cloud v0.93.0 (#1851). Sent only when ``True``.
                Keyword-only.

        Returns:
            API response with ``memory_id``, ``scope``, ``version`` (the
            memory's version, :meth:`update_memory`'s ``expected_version``;
            memory-cloud v0.113.0+, #2043) and the context fields. Since
            memory-cloud v0.65.0 two optional keys may follow, each absent
            (never ``null``) when it does not apply: ``persistence`` —
            ``{scope, committed, promotes_via,
            consolidation_archive_min_age_days}``, how the scope the memory
            landed in is consolidated; since memory-cloud v0.93.0 (#1851) the
            ``detail`` prose comes only with the server's ``verbose=true``
            (``verbose=True`` here) — and ``lint`` — ``[{code, hint,
            subject?}]``, advisory hints about a write that will recall
            poorly (``summary_short``, ``summary_long``,
            ``summary_narrative``, ``no_tags``, ``tag_near_duplicate``; and
            ``tag_normalized`` with ``tags_normalize``). The memory is stored
            either way; act on a hint with :meth:`update_memory`. Since
            v0.108.0 a cut summary adds ``summary_truncated: true`` and
            ``summary_original_length`` (the ``summary_long`` hint is then
            left out). With ``dedupe="check"`` the reply may instead be the
            ``duplicate_candidate`` one described there, with nothing
            written: check ``result["status"]``.

            From memory-cloud v0.142.0 (#2615) a success may carry
            ``plan_headroom``: one entry per plan dimension at or above 80 %
            (``level`` ``"warning"``) or 95 % (``"critical"``) of its
            effective limit (add-on and referral bonuses included) —
            ``{dimension, level, used, limit, percentage, plan, remedies,
            help}``, plus ``resets_at`` (the next UTC midnight) on the daily
            dimensions; ``plan`` is left out when the server could not
            resolve it. ``dimension`` is ``memories``, ``memories_per_day``
            or ``mcp_calls_per_day``; ``remedies`` starts with ``cleanup``
            (the total cap) or ``wait`` (a daily cap), then ``upgrade`` when
            a higher tier raises that cap, or ``admin`` on a deployment
            without billing; ``help`` says what to do and what fails at
            100 %. Below 80 % the key is absent (never ``[]`` or ``null``),
            and writes show an entry at most once an hour per workspace,
            dimension and level, so a missing key does not mean the plan has
            room. The entries stay dicts here;
            :class:`~kagura_memory.models.PlanHeadroomNotice` is the typed
            form :meth:`bootstrap` and :meth:`remember_batch` return.

        Raises:
            ValueError: ``tool_trigger`` is given and ``details`` already has
                a ``"tool_trigger"`` key — pass the guardrail one way only;
                ``tags`` holds more than 50 tags or a tag longer than 100
                characters; ``observed_at`` is not a form the server accepts,
                or disagrees with ``details["observed_at"]`` (the server's
                ``validation_error``); or ``dedupe`` is not ``"suggest"`` /
                ``"check"`` / ``"off"``. Checked before any request.
            KaguraCapacityLockedError: The workspace is over its Free
                capacity (memory-cloud v0.99.0+); ``cleanup_url`` says where
                to clean up.
            KaguraFeatureNotAvailableError: ``feature == "contexts"``: the
                context has no writable slot because the plan dropped below
                the workspace's contexts (v0.100.0+); reads keep working.
            KaguraError: Server-side rejection, e.g. ``permission_denied`` or
                ``validation_error`` for a guardrail write, or ``code ==
                "payload_too_large"`` for a memory over 1 MB across summary,
                context_summary, content and details (v0.108.0+, with
                ``limit`` / ``actual`` in ``details``; older servers send an
                untyped ``quota_exceeded``), or ``code ==
                "dedupe_unavailable"`` when ``dedupe="check"`` could not embed
                the summary (nothing written). Match ``code``, not the
                message.
        """
        _validate_memory_tags(tags)
        details = _details_with_tool_trigger(details, tool_trigger)
        observed = None if observed_at is None else _observed_at_arg(observed_at)
        if observed is not None:
            _check_observed_at_agrees(observed, details, "details")
        if dedupe is not None and dedupe not in get_args(DedupeMode):
            raise ValueError(f"dedupe must be 'suggest', 'check' or 'off', got {dedupe!r}")
        arguments: dict[str, Any] = {
            "context_id": context_id,
            "summary": summary,
            "content": content,
            "type": type,
            "importance": importance,
        }
        if tags is not None:
            arguments["tags"] = tags
        if source_uri is not None:
            arguments["source_uri"] = source_uri
        if source_type is not None:
            arguments["source_type"] = source_type
        # Only send a non-default delivery_mode; the server applies
        # server_default='on_recall' so omitting it stays forward-compatible.
        if delivery_mode != "on_recall":
            arguments["delivery_mode"] = delivery_mode
        if context_summary is not None:
            arguments["context_summary"] = context_summary
        if details is not None:
            arguments["details"] = details
        if context is not None:
            arguments["context"] = context
        if linked_memory_ids is not None:
            arguments["linked_memory_ids"] = linked_memory_ids
        if linked_source_uris is not None:
            arguments["linked_source_uris"] = linked_source_uris
        if supersedes is not None:
            arguments["supersedes"] = supersedes
        if observed is not None:
            arguments["observed_at"] = observed
        if tags_normalize:
            arguments["tags_normalize"] = True
        if dedupe is not None:
            arguments["dedupe"] = dedupe
        if verbose:
            arguments["verbose"] = True

        return await self._call_tool_checked("remember", arguments)

    async def recall(
        self,
        context_id: str | None = None,
        query: str = "",
        k: int = 5,
        use_rerank: bool | None = None,
        filters: dict[str, Any] | None = None,
        search_mode: str | None = None,
        context_ids: list[str] | None = None,
        include_explore_hints: bool = False,
        include_superseded: bool = False,
        *,
        verbosity: RecallVerbosity | None = None,
        full: int | None = None,
        max_chars: int | None = None,
    ) -> dict[str, Any]:
        """
        Call recall MCP tool.

        Since memory-cloud v0.108.0 (#2002) a hit is **compact** unless you
        pass ``verbosity="full"``: a 160-character summary head and no
        ``tags`` / ``context_summary``. Since v0.114.0 (#2124) the top results
        also carry ``content`` unless you pass ``full=0``. ``verbosity``,
        ``full`` and ``max_chars`` are keyword-only and sent only when set.
        From v0.82.0 (#1742) a server refuses a tool argument it does not
        declare (``KaguraError``, code ``invalid_argument``), so ``verbosity``
        needs v0.108.0+ and ``full`` v0.114.0+ there; a server before v0.82.0
        ignores all three. This method never retries without them: the
        caller decides. (The ``kagura recall`` CLI does retry, once, without
        the ones the refusal lists in ``unknown_arguments``, so its
        SessionStart hook works on every server.)

        Args:
            context_id: Context ID for single-context search.
            query: Search query
            k: Number of results
            use_rerank: Cross-encoder reranking, tri-state since memory-cloud
                v0.69.0 (#1572). ``None`` (the default) omits the argument, so
                the server follows the context's search config
                (``search_config.use_rerank``, set via
                :meth:`update_search_config`). ``True`` requests reranking,
                which applies only when the context enables it (and the
                workspace plan and deployment allow it). ``False`` disables
                reranking for this call. With ``context_ids``, only the first
                listed context's config governs both ``None`` and ``True``.
                Servers before v0.69.0 treat an omitted value as ``False``, so
                ``None`` never reranks there.
            filters: Optional filters. Supported keys:
                - ``type``: memory type (e.g., ``"code"``)
                - ``tags``: list of tag strings (e.g., ``["python"]``)
                - ``tags_match``: ``"any"`` (default) or ``"all"`` for AND logic
                - ``tags_normalize``: ``True`` also matches tag spellings that
                  differ only by case, hyphen/underscore/space or a simple
                  plural (``"dev-environment"`` = ``"Dev_Environment"``;
                  memory-cloud v0.65.0+). Abbreviations never match — they
                  come back as ``tag_suggestions`` instead.
                - ``created_after`` / ``created_before``: ISO 8601 datetime
                - ``updated_after`` / ``updated_before``: ISO 8601 datetime
                - ``observed_after`` (inclusive) / ``observed_before``
                  (exclusive): when the fact held (``details.observed_at``,
                  set with :meth:`remember`'s ``observed_at``), an ISO 8601
                  date or datetime; a date means the start of that UTC day,
                  and a memory without ``observed_at`` never matches
                  (memory-cloud v0.127.0+, #2330)
                - ``trust_tier``: ``"trusted"`` EXCLUDES external/connector-ingested
                  memories from the results (opt-in; default recall returns them).
                  Pass it for behaviour-influencing reads where untrusted content
                  must not be treated as instructions (OWASP LLM01/LLM03). Trust is
                  **server-derived from server-stamped provenance** (memory-cloud
                  #887): passing ``source_type`` on :meth:`remember` records origin
                  but no longer establishes trust — the client is untrusted by
                  contract, so it cannot mark its own writes ``"trusted"``.
            search_mode: Search strategy — "hybrid" (default), "semantic", or "keyword"
            context_ids: Search across multiple contexts (2–20 IDs).
                When provided, ``context_id`` is not required.
            include_explore_hints: When True, ask for up to 3 graph discovery
                hints under ``explore_hints`` — seeds for a follow-up
                :meth:`explore` call. Since memory-cloud v0.135.0 (#2470) the
                server makes them only when the MCP URL lists the ``explore``
                tool, which the bare URL (the core tool profile) does not: add
                ``?add_tools=explore`` or ``?profile=full`` to ``mcp_url`` (or
                ``explore`` to a ``tools=`` / ``add_tools=`` list it already
                has). On any other URL the call still succeeds and returns
                ``explore_hints_omitted`` — a sentence naming that opt-in — in
                place of ``explore_hints``.
            include_superseded: When True, also return memories shadowed by a
                ``supersedes`` edge, annotated with ``superseded_by``. Default
                recall demotes them, so this is how you read the history that
                :meth:`remember` (``supersedes=...``) deliberately preserves —
                without it, superseding would be a one-way door. It also turns
                off ``folded`` (see Returns): every version comes back as its
                own row. Note this is a top-level argument, **not** a
                ``filters`` key. Requires memory-cloud server v0.45.0+ (#1208).
            verbosity: Result detail, ``"compact"`` or ``"full"``
                (memory-cloud v0.108.0+, #2002). ``None`` (the default) omits
                the argument, so the deployment default applies — compact
                unless ``MCP_RECALL_DEFAULT_VERBOSITY`` says otherwise. A
                compact hit is ``memory_id``, the first 160 characters of
                ``summary`` (``summary_cut: True`` when it was shortened),
                ``type``, ``importance``, ``score``, ``created_at``
                (v0.110.0+), ``updated_at``, and the content keys below.
                ``"full"`` adds ``context_summary``, ``scope``, ``tags``,
                ``version`` (v0.113.0+) and the uncut ``summary``. Both shapes
                carry ``observed_at`` (v0.127.0+), ``superseded_by``,
                ``contradicts``, ``supersede_candidate`` and ``folded`` when
                set.
            full: How many of the top results carry ``content``, 0-100
                (memory-cloud v0.114.0+, #2124). Each carries up to 1,500
                characters; a longer one adds ``content_truncated: True`` and
                ``content_next_offset`` (where the server's ``reference`` tool
                continues; :meth:`reference` returns the whole memory, bounded
                on memory-cloud 0.78.0+). ``0`` returns summaries only.
                ``None`` omits the argument, so the deployment default applies
                (``MCP_RECALL_DEFAULT_FULL``, 3 unless set; v0.115.0+, #2140)
                — and a defaulted ``full`` puts content on trusted,
                non-connector rows only, while an explicit value makes every
                row eligible. Never more than ``k``; lowered to fit
                ``max_chars``.
            max_chars: Response budget in characters, not tokens,
                10,000-100,000 (server default 20,000; memory-cloud v0.82.0+,
                #1743). Over budget the server takes ``content`` off the
                lowest-ranked rows first (``full_applied`` then counts the rows
                that kept it), then ``context_summary`` off every row
                (``context_summary_omitted: True``), then drops the
                lowest-ranked rows (``truncated: True``; ``count`` is the rows
                returned).

        Returns:
            API response with ``confidence`` first — ``{level, top_score,
            prominence, relative_margin, result_count, rationale}``, where
            ``level`` (``high`` / ``moderate`` / ``low`` / ``none``) says
            whether the hits are worth reading — then ``status``,
            ``results`` (ranked hits in the ``verbosity`` shape), ``count``
            and the context fields. The optional keys below are absent when
            they do not apply, so read them with ``.get()``:

            - ``related_tags`` (``[{tag, count}]``): omitted when empty
              since memory-cloud v0.93.0 (#1851).
            - ``degraded`` / ``degraded_reason`` (v0.66.0+): the semantic half
              of hybrid search was unavailable, so the results are
              keyword-only. An empty or weak result then means "search
              impaired", not "nothing stored" — retry later.
            - ``tag_suggestions`` (v0.65.0+): ``{requested_tag: ["stored-tag
              (count)", ...]}`` when a tag filter matched nothing and similar
              stored tags exist. Advisory — the filter was not widened.
            - ``explore_hints`` or ``explore_hints_omitted``: see
              ``include_explore_hints``. ``explore_hints`` may be an empty
              list.
            - ``full_applied`` (v0.114.0+), ``context_summary_omitted`` and
              ``truncated`` (v0.82.0+): see ``max_chars``.
            - ``selector`` (v0.115.0+, #2144): only on a context with the LLM
              selector on (``llm_selector_enabled``) — ``{applied: True,
              model, kept, dropped, dropped_memory_ids}``, or ``{applied:
              False, reason}`` with the unselected page. A row whose content
              the selector cut to stored passages carries ``spans``
              (``[{start, end}]`` offsets into its stored content).
            - Per result, ``folded`` (v0.114.0+, #2123): ``{count,
              memory_ids}`` — judge-verified older versions of the same fact,
              folded under this, the newest, row.
            - Per result, ``summary_cut``, the content keys, ``observed_at``,
              ``tags``, ``context_summary``, ``superseded_by``,
              ``contradicts`` and ``supersede_candidate``: absent when empty
              (since memory-cloud v0.73.0 for the annotations; older servers
              send ``null`` / ``[]``, so test them for truthiness).
            - ``supersede_candidate`` (a suggestion, never an edge):
              ``{memory_id, summary, similarity, detected_at, verified,
              acceptance}`` (``acceptance`` from memory-cloud v0.142.0).
              ``verified`` is ``True`` when the deployment's judge confirmed
              that this memory replaces the candidate (memory-cloud
              v0.109.0+, #2016, ``SUPERSEDE_VERIFY_MODE=llm``) and absent
              otherwise; a pair the judge read as two parallel facts is not
              suggested. Since v0.137.0 (#2256) an older memory never takes
              a newer one as its candidate. From memory-cloud v0.142.0
              (#2508) the candidate also carries the caller's
              ``acceptance`` — ``{allowed, reason}``, whether
              ``update_memory(accept_supersede_candidate=True)`` would go
              through — as :meth:`reference` has since v0.137.0 (#2427); see
              :meth:`reference` for the reasons. It is absent on an older
              server and where the server could not evaluate it.

        Since memory-cloud v0.78.0 (#1683) ``tools/list`` annotates ``recall``
        as not read-only and destructive: recall's Hebbian learning reweights
        and prunes graph edges. An MCP client that confirms destructive tools
        asks before each recall; this client does not.

        Raises:
            ValueError: If ``query`` is empty/whitespace; if neither
                ``context_id`` nor ``context_ids`` is provided; if
                ``context_ids`` has fewer than 2 or more than 20 IDs; if
                ``search_mode`` is not one of "hybrid"/"semantic"/"keyword";
                if ``verbosity`` is not ``"compact"`` / ``"full"``; or if
                ``full`` (0-100) or ``max_chars`` (10,000-100,000) is not an
                ``int`` in range. Checked before any request.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if context_ids is not None:
            if len(context_ids) < 2 or len(context_ids) > 20:
                raise ValueError(f"context_ids must contain 2–20 IDs, got {len(context_ids)}")
        elif context_id is None:
            raise ValueError("Either context_id or context_ids must be provided")
        arguments: dict[str, Any] = {
            "query": query,
            "k": k,
        }
        if context_ids is not None:
            arguments["context_ids"] = context_ids
        else:
            arguments["context_id"] = context_id
        # Send an explicit False: since server v0.69.0 an omitted use_rerank
        # follows the context config, so dropping False would still rerank.
        if use_rerank is not None:
            arguments["use_rerank"] = use_rerank
        if filters:
            arguments["filters"] = filters
        if search_mode:
            if search_mode not in ("hybrid", "semantic", "keyword"):
                raise ValueError(f"Invalid search_mode: {search_mode!r}")
            arguments["search_mode"] = search_mode
        if include_explore_hints:
            arguments["include_explore_hints"] = True
        if include_superseded:
            arguments["include_superseded"] = True
        if verbosity is not None:
            if verbosity not in get_args(RecallVerbosity):
                raise ValueError(f"verbosity must be 'compact' or 'full', got {verbosity!r}")
            arguments["verbosity"] = verbosity
        if full is not None:
            _validate_int_range("full", full, 0, _RECALL_FULL_MAX)
            arguments["full"] = full
        if max_chars is not None:
            _validate_max_chars(max_chars)
            arguments["max_chars"] = max_chars
        return await self._call_tool_checked("recall", arguments)

    async def recall_upcoming(
        self,
        context_id: str,
        *,
        from_: str | None = None,
        until: str | None = None,
        k: int = 20,
        include_details: bool = False,
    ) -> dict[str, Any]:
        """List Time Memories whose scheduled window overlaps a range, soonest first.

        Calls the ``recall_upcoming`` MCP tool. This is a **deterministic time
        query** over ``type="time"`` memories — not semantic search and with no
        Hebbian side-effects — so it is distinct from :meth:`recall`. Create a
        Time Memory with ``remember(type="time", details={"trigger": {...}})``.

        Args:
            context_id: Target context UUID.
            from_: Lower bound as naive ISO (e.g. ``"2026-06-01T00:00:00"``), or
                the literal ``"now"`` which the server resolves. Omit for no
                lower bound. (Named ``from_`` because ``from`` is a reserved
                word; it is sent to the server as ``"from"``.)
            until: Upper bound as naive ISO. Omit for an open-ended future window.
            k: Maximum results (default 20, server max 100).
            include_details: Return each item's full ``details`` object instead
                of its ``trigger``. Details can be large. Sent only when
                ``True``. Requires memory-cloud v0.73.0+ (#1599); an older
                server ignores it and always returns ``details``.

        Returns:
            API response with ``results`` — ``type="time"`` memories whose window
            overlaps the range, soonest first. Since memory-cloud v0.73.0 each
            item is ``{memory_id, summary, type, trigger}``, where ``trigger``
            is the memory's ``details.trigger``. With ``include_details=True``
            the item carries the full ``details`` object in place of
            ``trigger`` (``details.trigger`` is inside it). Since memory-cloud
            v0.82.0 (#1743) the details are held to the reply budget (about
            20,000 characters): from the last item backwards, an item that
            does not fit carries ``details_omitted: True`` and
            ``details_total_chars`` in place of ``details`` — read those
            with :meth:`reference`.
        """
        # `from` is a Python reserved word, so the public param is `from_` but
        # the MCP tool expects the key `"from"`.
        arguments: dict[str, Any] = {"context_id": context_id, "k": k}
        if from_ is not None:
            arguments["from"] = from_
        if until is not None:
            arguments["until"] = until
        if include_details:
            arguments["include_details"] = True
        return await self._call_tool_checked("recall_upcoming", arguments)

    async def recall_nearby(
        self,
        context_id: str,
        lat: float,
        lon: float,
        *,
        radius_m: float = 1000,
        k: int = 20,
        include_details: bool = False,
    ) -> dict[str, Any]:
        """List memories near a geographic point, nearest first with ``distance_m``.

        Calls the ``recall_nearby`` MCP tool. This is the WHERE-axis mirror of
        :meth:`recall_upcoming`: a **deterministic spatial query** over stored
        coordinates — not semantic search and with no Hebbian side-effects — so
        it is distinct from :meth:`recall`. Attach a location with
        ``remember(details={"location": {"lat": ..., "lon": ..., "label": ...}})``;
        any memory type can carry one.

        Requires memory-cloud server v0.53.0+ (#1331); older servers return an
        MCP "tool not found". See :meth:`update_memory` for the caveat about
        revising a memory that carries a location.

        Args:
            context_id: Target context UUID.
            lat: Query latitude, -90 to 90. Must be a JSON number — the server
                rejects string-typed numerics with a 422 by design.
            lon: Query longitude, -180 to 180. Same numeric requirement.
            radius_m: Search radius in meters (default 1000). The server clamps
                this to [1, 1000000] rather than rejecting out-of-range values.
            k: Maximum results (default 20, server max 100).
            include_details: Return each item's full ``details`` object in
                place of its ``location``. Details can be large. Sent only
                when ``True``. Requires memory-cloud v0.82.0+ (#1743); an
                older server ignores it and always returns ``details``.

        Returns:
            API response with ``results`` — memories whose stored location falls
            within the radius, nearest first. Since memory-cloud v0.82.0 each
            item is ``{memory_id, summary, type, location, distance_m}``, where
            ``location`` is the memory's ``details.location``; older servers
            send the whole ``details`` instead. With ``include_details=True``
            the item carries ``details`` in place of ``location``
            (``details.location`` is inside it), held to the reply budget
            (about 20,000 characters): from the last item backwards, an item
            that does not fit carries ``details_omitted: True`` and
            ``details_total_chars`` instead — read those with
            :meth:`reference`.

        Raises:
            ValueError: If ``lat`` or ``lon`` is not a number (strings are
                rejected, not coerced) or is outside its valid range.
        """
        validate_lat_lon(lat, lon)

        arguments: dict[str, Any] = {
            "context_id": context_id,
            "lat": lat,
            "lon": lon,
            "radius_m": radius_m,
            "k": k,
        }
        if include_details:
            arguments["include_details"] = True
        return await self._call_tool_checked("recall_nearby", arguments)

    async def record_measurement(
        self,
        context_id: str,
        metric: str,
        value: float,
        *,
        measured_at: str | datetime | None = None,
        unit: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> MeasurementResult:
        """Append one numeric observation to a metric's series (HOW-MUCH axis).

        Calls the ``record_measurement`` MCP tool. Measurements are a lane
        **separate from memories**: never embedded, never returned by
        :meth:`recall`, never merged or rewritten by Sleep consolidation. The
        lane is append-only — nothing is upserted, so recording the same point
        twice stores two rows, and there is no delete tool. Store raw numbers
        here (weight, revenue, reps) and prose such as "hit goal weight" with
        :meth:`remember`. Read a series back with :meth:`recall_series`.

        Requires memory-cloud server v0.54.0+ (#1333); older servers return
        an MCP "tool not found". Retention: from server v0.55.0 (#1355) an
        operator can set ``SLEEP_MEASUREMENT_RETENTION_DAYS`` > 0 (or a
        per-context config row; default 0 = keep forever), and Sleep then
        **hard-deletes** observations older than that window — ``kagura sleep
        rollback`` / :meth:`rollback_sleep_run` cannot restore them.

        Args:
            context_id: Target context UUID (the series is scoped to it).
            metric: Series name, e.g. ``"weight_kg"`` (1-64 chars). Reuse the
                exact name to extend a series.
            value: The observed value — a finite number. NaN, infinity,
                ``bool`` and strings are rejected locally.
            measured_at: Observation time as an ISO 8601 string or a
                ``datetime``. Naive means **UTC** to the server, not local
                time. Omit for "now"; pass it to backdate imports.
            unit: Optional display unit, e.g. ``"kg"`` (1-32 chars).
            details: Optional JSON metadata (device, source, notes).

        Returns:
            :class:`MeasurementResult` with the stored ``measurement_id``,
            ``metric``, ``measured_at``, ``value`` and ``unit``.

        Raises:
            ValueError: If ``metric`` is empty or over 64 characters, ``value``
                is not a finite number, or ``unit`` is empty or over 32
                characters.
            KaguraNotFoundError: Context not found (or a read-only agent
                binding forbids writes to it).
            KaguraResponseError: The success payload did not match
                :class:`MeasurementResult` (``operation="record_measurement"``).
            KaguraError: Other server-side error, e.g. a read-only viewer.
        """
        _validate_metric(metric)
        _validate_measurement_value(value)
        if unit is not None and (
            not isinstance(unit, str) or not unit or len(unit) > _UNIT_MAX_LEN
        ):
            raise ValueError(
                f"unit must be a non-empty string of at most {_UNIT_MAX_LEN} characters, "
                f"got {unit!r}"
            )

        arguments: dict[str, Any] = {"context_id": context_id, "metric": metric, "value": value}
        if measured_at is not None:
            arguments["measured_at"] = _iso_arg(measured_at)
        if unit is not None:
            arguments["unit"] = unit
        if details is not None:
            arguments["details"] = details
        result = await self._call_tool_checked("record_measurement", arguments)
        return parse_response(MeasurementResult, result, operation="record_measurement")

    async def recall_series(
        self,
        context_id: str,
        metric: str,
        *,
        period: MeasurementPeriod | None = None,
        agg: MeasurementAggregate | None = None,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
    ) -> MeasurementSeries:
        """Read one metric's series, bucketed by period and aggregated per bucket.

        Calls the ``recall_series`` MCP tool — a **deterministic query**, not
        search, over the lane written by :meth:`record_measurement`. Empty
        buckets are omitted. Buckets align to UTC boundaries, so a local day
        may span two.

        Requires memory-cloud server v0.54.0+ (#1333); older servers return
        an MCP "tool not found".

        Args:
            context_id: Target context UUID.
            metric: Series name as passed to :meth:`record_measurement`
                (1-64 chars).
            period: Bucket size — ``"day"``, ``"week"`` or ``"month"``. Omit
                for the server default (``"day"``).
            agg: Per-bucket aggregate — ``"avg"``, ``"min"``, ``"max"``,
                ``"sum"``, ``"count"`` or ``"last"`` (the most recent value in
                the bucket). Omit for the server default (``"avg"``).
            start: Window start, inclusive, as an ISO 8601 string or a
                ``datetime`` (naive = UTC). Omit for ``end`` minus 30 days.
            end: Window end, exclusive (naive = UTC). Omit for "now". The
                window may span at most 365 days — the server rejects wider
                windows with a ``validation_error``.

        Returns:
            :class:`MeasurementSeries` with ``series`` (one
            :class:`SeriesBucket` per non-empty bucket, oldest first) and
            ``count`` (the number of buckets).

        Raises:
            ValueError: If ``metric`` is empty or over 64 characters.
            KaguraNotFoundError: Context not found.
            KaguraResponseError: The success payload did not match
                :class:`MeasurementSeries` (``operation="recall_series"``).
            KaguraError: Other server-side error, e.g. an inverted or
                over-365-day window.
        """
        _validate_metric(metric)

        arguments: dict[str, Any] = {"context_id": context_id, "metric": metric}
        if period is not None:
            arguments["period"] = period
        if agg is not None:
            arguments["agg"] = agg
        if start is not None:
            arguments["start"] = _iso_arg(start)
        if end is not None:
            arguments["end"] = _iso_arg(end)
        result = await self._call_tool_checked("recall_series", arguments)
        return parse_response(MeasurementSeries, result, operation="recall_series")

    async def load_pinned(
        self,
        context_id: str,
        cap: int | None = None,
        *,
        max_chars: int | None = None,
    ) -> dict[str, Any]:
        """Deterministically load a context's pinned (``delivery_mode="always"``) memories.

        This is the **deterministic** counterpart to :meth:`recall`: it returns
        the complete, unranked pinned set on every call — no semantic search, no
        ranking, no rerank — so an agent's Goal / Guardrail / critical-policy
        memories load identically every turn. Pin a memory with
        :meth:`remember` (``delivery_mode="always"``) or :meth:`update_memory`
        (``delivery_mode="always"``); unpin with
        :meth:`update_memory` (``delivery_mode="on_recall"``).

        Results carry summary + context_summary only (Layer 1+2). Fetch full
        content with :meth:`reference` using a result's ``memory_id``.

        The set is **bounded, never silently dropped**: when more pinned
        memories exist than ``cap``, the response ``truncated`` flag is ``True``
        and ``total_available`` reports the real count. Callers loading a
        complete policy set should check ``truncated`` and re-call with a larger
        ``cap`` (up to the server maximum). Note that :meth:`list_memories` is
        not a substitute for paging the pinned set — its responses do not carry
        ``delivery_mode``/pinned state, so it cannot reconstruct the pinned
        subset; use this method with a sufficient ``cap`` instead.

        Args:
            context_id: Target context UUID.
            cap: Optional override for the maximum number returned (1-1000).
                Omit to use the server default (100).
            max_chars: Response budget in characters, not tokens,
                10,000-100,000 (server default 20,000). Over budget the server
                leaves ``context_summary`` out of every item first
                (``context_summary_omitted: True``), then cuts items from the
                end (``truncated: True``; ``total_available`` keeps the real
                count). Keyword-only, sent only when set. Requires
                memory-cloud v0.82.0+ (#1743); an older server ignores it.

        Returns:
            API response with ``memories`` (the pinned set — each item
            ``{memory_id, summary, context_summary, type, importance,
            delivery_mode}``), ``truncated`` (bool: more exist than ``cap``
            or ``max_chars`` allows), ``total_available`` (int), ``cap`` (the
            cap applied) and the context fields, plus
            ``context_summary_omitted`` when the budget forced it.

        Raises:
            ValueError: ``max_chars`` is not an ``int`` in range (before any
                request).

        Example:
            >>> pinned = await client.load_pinned(context_id=ctx)
            >>> if pinned["truncated"]:
            ...     pinned = await client.load_pinned(context_id=ctx, cap=1000)
            >>> for m in pinned["memories"]:
            ...     full = await client.reference(
            ...         context_id=ctx, memory_id=m["memory_id"]
            ...     )
        """
        arguments: dict[str, Any] = {"context_id": context_id}
        if cap is not None:
            arguments["cap"] = cap
        if max_chars is not None:
            _validate_max_chars(max_chars)
            arguments["max_chars"] = max_chars
        return await self._call_tool_checked("load_pinned", arguments)

    async def load_guardrails(
        self, context_id: str, *, cap: int | None = None, max_chars: int | None = None
    ) -> GuardrailSet:
        """Deterministically load a context's guardrail set for a client-side hook.

        Calls the ``load_guardrails`` MCP tool (memory-cloud v0.74.0+, #1619) —
        :meth:`load_pinned`'s twin: no search, no ranking, trusted-tier rows
        only. Two independently capped lanes: ``pinned``
        (``delivery_mode="always"``, bounded by the server's pinned cap) and
        ``tool_triggered`` (memories marked with ``details.tool_trigger`` —
        see :meth:`remember`'s ``tool_trigger``), bounded by ``cap``. The
        server returns the patterns as data and never runs them; matching
        is the hook's job.

        The set is never silently cut: check ``pinned_truncated`` and
        ``tool_triggered_truncated`` (``truncated`` is either one) and re-call
        with a larger ``cap`` if the tool-triggered lane is incomplete. A
        memory that is both pinned and tool-triggered appears in both lists.

        Requires memory-cloud v0.74.0+; older servers return an MCP
        "tool not found". The REST twin for API-key-only callers is
        :meth:`MemoryClient.load_guardrails
        <kagura_memory.memory_client.MemoryClient.load_guardrails>`.

        Args:
            context_id: Target context UUID.
            cap: Max tool-triggered memories returned (1-1000; the server
                clamps out-of-range values on this surface). Omit for the
                server default (50). Does not bound the pinned lane.
            max_chars: Response budget in characters, not tokens,
                10,000-100,000 (server default 20,000). Over budget the server
                leaves ``context_summary`` out of the pinned items first
                (``context_summary_omitted``), then cuts pinned and then
                tool-triggered items from the end, setting that lane's
                ``*_truncated``. Sent only when set. Requires memory-cloud
                v0.82.0+ (#1743); an older server ignores it. The REST twin
                takes no ``max_chars``.

        Returns:
            :class:`~kagura_memory.models.GuardrailSet`.

        Raises:
            ValueError: ``max_chars`` is not an ``int`` in range (before any
                request).
            KaguraNotFoundError: Context not found (uniform — nonexistent and
                not-yours are indistinguishable).
            KaguraResponseError: The response does not parse as a
                ``GuardrailSet`` — e.g. a truncation flag is missing, so the
                set cannot be trusted as complete.
            KaguraError: Other server-side error.
        """
        arguments: dict[str, Any] = {"context_id": context_id}
        if cap is not None:
            arguments["cap"] = cap
        if max_chars is not None:
            _validate_max_chars(max_chars)
            arguments["max_chars"] = max_chars
        result = await self._call_tool_checked("load_guardrails", arguments)
        return parse_response(GuardrailSet, result, operation="load_guardrails")

    async def feedback(
        self,
        context_id: str,
        memory_id: str,
        helpful: bool,
        *,
        query: str | None = None,
        note: str | None = None,
    ) -> dict[str, Any]:
        """Record whether a recalled memory was useful for a query.

        Calls the ``feedback`` MCP tool. This is an **append-only usefulness
        signal** that lets SDK consumers (ai-worker, agents) teach the substrate
        which :meth:`recall` results were on-target. Each call appends a new
        event, so repeated or contradicting signals are kept as a time series
        rather than overwriting.

        Feedback is a **separate lane from knowledge**: it is not embedded and is
        structurally excluded from :meth:`recall`, so rating a result never
        pollutes the search space. Anyone who can read the context may record it.

        Args:
            context_id: Context UUID (the recalled memory's context).
            memory_id: UUID of the recalled memory being rated.
            helpful: ``True`` if the memory was useful for the query,
                ``False`` if not.
            query: Optional recall query this feedback is about (max 1024 chars).
            note: Optional free-text note, e.g. why the result was wrong
                (max 2000 chars).

        Returns:
            API response acknowledging the recorded feedback event.

        Raises:
            KaguraNotFoundError: Context or memory not found.
            KaguraError: Other server-side error.

        Example:
            >>> hits = await client.recall(context_id=ctx, query="auth flow")
            >>> await client.feedback(
            ...     context_id=ctx,
            ...     memory_id=hits["results"][0]["memory_id"],
            ...     helpful=True,
            ...     query="auth flow",
            ... )
        """
        arguments: dict[str, Any] = {
            "context_id": context_id,
            "memory_id": memory_id,
            "helpful": helpful,
        }
        if query is not None:
            arguments["query"] = query
        if note is not None:
            arguments["note"] = note
        return await self._call_tool_checked("feedback", arguments)

    async def set_state(
        self,
        context_id: str,
        key: str,
        value: Any,
        *,
        ttl_seconds: int | None = None,
    ) -> dict[str, Any]:
        """Set ephemeral agent run-state at ``(context_id, key)``.

        Calls the ``set_state`` MCP tool. This is the write side of a
        **TTL-bounded key/value lane for autonomous-agent run state** (current
        task, step, scratch flags) — kept deliberately **separate from
        memories**: state is not embedded and is structurally excluded from
        :meth:`recall`, so transient run-state never pollutes the knowledge
        search space. Writing upserts the value for the key.

        Use this for transient run state, **not** durable knowledge — use
        :meth:`remember` for knowledge.

        Args:
            context_id: Target context UUID (state is scoped to this context).
            key: State key (max 255 chars). Re-using a key overwrites its value.
            value: Arbitrary JSON value to store (object, array, string, number,
                or boolean), at most 16,384 characters as compact JSON — the
                cap memory-cloud v0.82.0+ enforces (#1743), measured as the
                server measures it (``ensure_ascii=False``, no spaces) and
                checked before any request. Store large data as a memory and
                keep its id here.
            ttl_seconds: Optional TTL in seconds (server clamps to 2592000 =
                30 days). Omit for no expiry.

        Returns:
            API response acknowledging the upsert.

        Raises:
            ValueError: ``value`` is longer than 16,384 characters as
                compact JSON.
            KaguraNotFoundError: Context not found.
            KaguraError: Other server-side error.
        """
        _validate_state_value(value)
        arguments: dict[str, Any] = {
            "context_id": context_id,
            "key": key,
            "value": value,
        }
        if ttl_seconds is not None:
            arguments["ttl_seconds"] = ttl_seconds
        return await self._call_tool_checked("set_state", arguments)

    async def get_state(
        self,
        context_id: str,
        key: str | None = None,
        *,
        cursor: str | None = None,
        limit: int | None = None,
        max_chars: int | None = None,
    ) -> dict[str, Any]:
        """Read ephemeral agent run-state.

        Calls the ``get_state`` MCP tool — the read side of the session-state
        lane (see :meth:`set_state`). Supply ``key`` to read one value, or omit
        it to list the context's live entries. Expired entries are never
        returned. This lane is excluded from :meth:`recall` by design.

        Since memory-cloud v0.82.0 (#1743) the list is paged in key order:
        50 entries a page by default (``limit``) within a 20,000-character
        budget by default (``max_chars``). The reply then carries
        ``has_more`` and ``next_cursor``; pass ``next_cursor`` back as
        ``cursor`` for the next page. A value too large for the page is named
        in ``omitted_keys`` instead (read it by ``key``). Older servers return
        every entry and no paging fields, and ignore ``cursor``, ``limit``
        and ``max_chars``.

        Args:
            context_id: Target context UUID.
            key: Optional state key. Omit to list the live ``(key, value)``
                entries for the context.
            cursor: List mode only: the ``next_cursor`` of the previous page.
                Sent only when set (memory-cloud v0.82.0+). Keyword-only, like
                the two below.
            limit: List mode only: entries per page, 1-200 (server default
                50). Sent only when set and checked before any request.
                Requires memory-cloud v0.82.0+ (#1743).
            max_chars: List mode only: the page's budget in characters, not
                tokens, 10,000-100,000 (server default 20,000). Sent only when
                set and checked before any request. Requires memory-cloud
                v0.82.0+ (#1743).

        Returns:
            API response with the value for ``key``, or one page of live
            entries when ``key`` is omitted: ``states`` (key → value),
            ``count``, ``has_more``, ``next_cursor`` and, when a value was
            left out, ``omitted_keys``.

        Raises:
            ValueError: ``cursor``, ``limit`` or ``max_chars`` is given
                together with ``key``, or ``limit`` / ``max_chars`` is not an
                ``int`` in range.
            KaguraNotFoundError: Context not found.
            KaguraError: Other server-side error.
        """
        paging = {"cursor": cursor, "limit": limit, "max_chars": max_chars}
        sent = [name for name, value in paging.items() if value is not None]
        if key is not None and sent:
            raise ValueError(f"{', '.join(sent)}: list mode only; omit key to list the entries")
        if limit is not None:
            _validate_int_range("limit", limit, 1, _STATE_LIST_MAX_LIMIT)
        if max_chars is not None:
            _validate_max_chars(max_chars)
        arguments: dict[str, Any] = {"context_id": context_id}
        if key is not None:
            arguments["key"] = key
        arguments.update({name: paging[name] for name in sent})
        return await self._call_tool_checked("get_state", arguments)

    async def get_agent_bootstrap(
        self,
        agent_id: str,
        *,
        context_id: str | None = None,
        session_id: str | None = None,
        query: str | None = None,
        recall_k: int | None = None,
        pinned_cap: int | None = None,
        upcoming_until: str | None = None,
        include: list[AgentBootstrapComponentName] | None = None,
        max_chars: int | None = None,
    ) -> AgentBootstrapResponse:
        """Rehydrate an agent's cognitive state in one session-start call.

        Calls the ``get_agent_bootstrap`` MCP tool (server v0.49.0+,
        RFC-0002 P0-3). The server composes existing primitives — context
        guide + pinned memories (:meth:`load_pinned`) + a trusted-only
        :meth:`recall` (only when ``query`` is supplied) + upcoming time
        memories (:meth:`recall_upcoming`) + the agent-state lane
        (:meth:`get_state`) — with bounds, ordering, and trust filtering
        inherited from those standalone tools, not re-specified.

        Components are **fail-soft**: a failing component reports
        ``{"status": "error", ...}`` under ``components`` while the rest
        still return, with the top-level ``degraded`` flag set.
        Identity/authorization failures are total and raise instead.

        Requires memory-cloud v0.49.0+ — older servers return an MCP
        "tool not found" error. ``MIN_SERVER_VERSION`` is deliberately not
        bumped; only this method needs the newer server. The REST companion
        (``POST /api/v1/agents/{agent_id}/bootstrap``) is available via
        :class:`~kagura_memory.agents_client.AgentsClient` for
        API-key-only callers such as agent-bound member keys.

        Args:
            agent_id: Agent UUID from the registry (required).
            context_id: Target context UUID. Omit to use the agent's
                default binding.
            session_id: Opaque correlation id (max 128 chars,
                ``[A-Za-z0-9._-]``); echoed in the ``correlation`` block.
            query: Recall query (max 1024 chars). Supplying it enables the
                trusted-only recall component; omit to skip recall — the
                server never fabricates a query, so the component reports
                ``status="skipped"`` even when ``include`` names it.
            recall_k: Number of recall results; forwarded to recall's
                ``k`` validation.
            pinned_cap: Override for the pinned-set cap; clamped by
                ``load_pinned`` to [1, 1000]. Omit for the server default,
                20 since memory-cloud v0.82.0 (#1743; 100 before, and
                :meth:`load_pinned` still defaults to 100).
            upcoming_until: ISO upper bound for upcoming time memories
                (the lower bound is always now).
            include: Component selector — a subset of ``"pinned"``,
                ``"recall"``, ``"upcoming"``, ``"state"``, ``"policy"``.
                Omit for all components.
            max_chars: Whole-envelope budget in characters, not tokens,
                10,000-100,000 (server default 20,000). Over budget the
                server leaves ``context_summary`` out of every pinned, recall
                and upcoming item first (``context_summary_omitted``), then
                cuts those lists from the end in that priority, marking each
                cut component ``truncated: True``. Requires memory-cloud
                v0.82.0+ (#1743); an older server ignores it.

        Returns:
            :class:`AgentBootstrapResponse` — the composed envelope with
            ``agent`` (identity + resolved binding), ``context``,
            ``instructions``, per-component payloads in ``components``,
            the ``correlation`` block, and the ``degraded`` flag. Since
            memory-cloud v0.73.0 the ``upcoming`` component's rows carry
            ``trigger`` in place of ``details``, as :meth:`recall_upcoming`
            does by default; bootstrap has no ``include_details`` opt-out, so
            fetch ``details`` with :meth:`reference` or
            ``recall_upcoming(include_details=True)``. Since v0.110.0
            (#2013) the ``recall`` component's rows are :meth:`recall`'s
            ``verbosity="full"`` shape without ``supersede_candidate``, and
            they never carry ``content``, whatever the deployment's recall
            defaults; an empty ``context_summary``, ``tags``,
            ``superseded_by`` or ``contradicts`` is absent, not ``null`` /
            ``[]``, so read them with ``.get()``. Since v0.79.0 (#1682)
            ``instructions`` is the server's static quick reference; the
            context's own guide is ``context.usage_guide``, notes written by
            context members (data, not instructions).

            ``tools/list`` annotates this tool as not read-only and
            destructive (memory-cloud v0.78.0+, #1683), as it does
            :meth:`recall`, because with a ``query`` it runs recall's
            learning writes. The annotation is the same for every call, so
            an MCP client that confirms destructive tools asks each time;
            this client does not.

        Raises:
            ValueError: ``max_chars`` is not an ``int`` in range (before any
                request).
            KaguraNotFoundError: Agent or context not found (uniform 404 —
                nonexistent and not-yours are indistinguishable by design).
            KaguraError: Invalid arguments or other server-side error.
        """
        arguments: dict[str, Any] = {
            "agent_id": agent_id,
            **_bootstrap_payload(
                context_id=context_id,
                session_id=session_id,
                query=query,
                recall_k=recall_k,
                pinned_cap=pinned_cap,
                upcoming_until=upcoming_until,
                include=include,
                max_chars=max_chars,
            ),
        }
        result = await self._call_tool_checked("get_agent_bootstrap", arguments)
        return parse_response(AgentBootstrapResponse, result, operation="get_agent_bootstrap")

    async def register_agent(
        self,
        name: str,
        *,
        description: str | None = None,
        framework: str | None = None,
        environment: str | None = None,
        version: str | None = None,
    ) -> Agent:
        """Register an AI agent in the workspace Agent Registry.

        Calls the ``register_agent`` MCP tool (server v0.49.0+, RFC-0002
        P0-1, memory-cloud #1274). Owner/admin only. An agent is a
        workspace-scoped registry entry (name unique per workspace) that
        anchors context bindings, agent-bound credentials,
        :meth:`get_agent_bootstrap`, and audit correlation — it is a
        resource, NOT a principal. New agents start with
        ``status="active"`` and ``enforcement_mode="enforce"``.

        Args:
            name: Workspace-unique agent name (max 255 chars).
            description: Optional free-text description (max 10000 chars).
            framework: Optional framework tag, e.g. ``"claude-code"``,
                ``"langgraph"`` (max 100 chars).
            environment: Optional deployment environment, e.g.
                ``"production"`` (max 100 chars).
            version: Optional agent build/prompt version (max 100 chars).

        Returns:
            The created :class:`Agent`.

        Raises:
            KaguraQuotaError: The workspace's agent cap is reached
                (``quota_type="agents"``).
            KaguraError: Name conflict, insufficient role, or other
                server-side error.
        """
        arguments: dict[str, Any] = {"name": name}
        if description is not None:
            arguments["description"] = description
        if framework is not None:
            arguments["framework"] = framework
        if environment is not None:
            arguments["environment"] = environment
        if version is not None:
            arguments["version"] = version
        result = await self._call_tool_checked("register_agent", arguments)
        return parse_response(Agent, result.get("agent"), operation="register_agent")

    async def get_agent(self, agent_id: str) -> Agent:
        """Fetch one registered agent by id (owner/admin only).

        Args:
            agent_id: Agent UUID from :meth:`register_agent` /
                :meth:`list_agents`.

        Returns:
            The :class:`Agent`.

        Raises:
            KaguraNotFoundError: Agent not found (uniform 404).
        """
        result = await self._call_tool_checked("get_agent", {"agent_id": agent_id})
        return parse_response(Agent, result.get("agent"), operation="get_agent")

    async def list_agents(self) -> list[Agent]:
        """List the workspace's registered agents, newest first (owner/admin only).

        Since memory-cloud v0.82.0 (#1743) the tool returns 50 agents a page
        with ``has_more`` / ``next_cursor``; this method follows the cursor
        and returns every agent, at most 50 pages (an older server answers in
        one page). The cursor is an offset, so an agent registered or deleted
        between two calls shifts the rows: an agent an earlier page returned
        is dropped from a later one (each ``id`` once), but one deleted
        between calls can make the next page skip an agent. Each row's
        ``description`` is a 200-character preview
        there, with ``description_truncated`` set when it was cut —
        :meth:`get_agent` returns it whole.

        Returns:
            List of :class:`Agent` rows in the active workspace.

        Raises:
            KaguraResponseError: A row does not match :class:`Agent`, a page
                repeats a cursor or says ``has_more`` without one, or the
                list is not complete after 50 pages.
        """
        return await _collect_pages(self, "list_agents", Agent, "agents")

    async def update_agent(
        self,
        agent_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        framework: str | None = None,
        environment: str | None = None,
        version: str | None = None,
        status: Literal["active", "suspended", "retired"] | None = None,
        enforcement_mode: Literal["shadow", "enforce"] | None = None,
    ) -> Agent:
        """Update a registered agent, including lifecycle transitions.

        Calls the ``update_agent`` MCP tool (owner/admin only).
        ``status`` is the **fail-closed kill switch**: ``"suspended"`` /
        ``"retired"`` agents cause every key bound to them to be rejected
        at verify time. Setting ``enforcement_mode`` from ``"enforce"``
        to ``"shadow"`` is an audited privilege-widening event (bindings
        stop being enforced and are only logged).

        Set-only wrapper: omitted fields are left untouched. The server's
        null-clears-a-metadata-field semantics is not expressible through
        this wrapper — clear fields via the web UI or the raw API.

        Args:
            agent_id: Agent UUID to update.
            name: New workspace-unique name (max 255 chars).
            description: New description (max 10000 chars).
            framework: New framework tag (max 100 chars).
            environment: New environment (max 100 chars).
            version: New version (max 100 chars).
            status: Lifecycle state — ``"active"`` | ``"suspended"`` |
                ``"retired"``.
            enforcement_mode: Binding enforcement ramp — ``"shadow"`` |
                ``"enforce"``.

        Returns:
            The updated :class:`Agent`.

        Raises:
            ValueError: If no update field is provided (the call would be
                an empty no-op request).
            KaguraNotFoundError: Agent not found.
            KaguraError: Name conflict or other server-side error.
        """
        changes = _agent_update_payload(
            name=name,
            description=description,
            framework=framework,
            environment=environment,
            version=version,
            status=status,
            enforcement_mode=enforcement_mode,
        )
        if not changes:
            raise ValueError("update_agent requires at least one field to update")
        result = await self._call_tool_checked("update_agent", {"agent_id": agent_id, **changes})
        return parse_response(Agent, result.get("agent"), operation="update_agent")

    async def delete_agent(self, agent_id: str) -> bool:
        """Hard-delete an Agent Registry row (owner/admin only).

        Prefer ``update_agent(status="retired")`` for operational
        retirement — delete is permanent and cascades every API key bound
        to the agent (fail-closed).

        Args:
            agent_id: Agent UUID to delete.

        Returns:
            ``True`` once the server confirms deletion.

        Raises:
            KaguraNotFoundError: Agent not found.
        """
        result = await self._call_tool_checked("delete_agent", {"agent_id": agent_id})
        return bool(result.get("deleted", True))

    async def bind_agent_context(
        self,
        agent_id: str,
        context_id: str,
        *,
        can_read: bool | None = None,
        write_policy: Literal["deny", "direct"] | None = None,
        is_default: bool | None = None,
    ) -> AgentBinding:
        """Bind an agent to a context — purely subtractive scoping.

        Calls the ``bind_agent_context`` MCP tool (server v0.49.0+,
        RFC-0002 P0-2, memory-cloud #1275). Owner/admin only. The
        effective permission for an agent-bound request is the existing
        RBAC decision ∩ binding. Under ``enforcement_mode="enforce"``,
        contexts WITHOUT a binding row are denied for the agent
        (default-deny); under ``"shadow"``, violations are only logged.

        ``allowed_memory_types``/``allowed_source_types`` are reserved
        for memory-cloud #1286 and deliberately not exposed here (the
        server accepts only null until per-memory enforcement ships).

        Args:
            agent_id: Agent UUID.
            context_id: Context to bind (must belong to the agent's
                workspace).
            can_read: Whether the agent may read this context (server
                default: ``True``).
            write_policy: Write gate — ``"deny"`` (server default) or
                ``"direct"``. ``"staged"`` is reserved for a later phase.
            is_default: Mark as the agent's bootstrap default binding
                (max one per agent).

        Returns:
            The created :class:`AgentBinding`.

        Raises:
            KaguraNotFoundError: Agent or context not found.
            KaguraError: Duplicate binding or other server-side error.
        """
        arguments: dict[str, Any] = {
            "agent_id": agent_id,
            "context_id": context_id,
            **_binding_scope_payload(
                can_read=can_read, write_policy=write_policy, is_default=is_default
            ),
        }
        result = await self._call_tool_checked("bind_agent_context", arguments)
        return parse_response(AgentBinding, result.get("binding"), operation="bind_agent_context")

    async def list_agent_bindings(self, agent_id: str) -> list[AgentBinding]:
        """List an agent's context bindings (owner/admin only).

        Args:
            agent_id: Agent UUID.

        Returns:
            List of :class:`AgentBinding` rows.

        Raises:
            KaguraNotFoundError: Agent not found.
        """
        result = await self._call_tool_checked("list_agent_bindings", {"agent_id": agent_id})
        return parse_response_list(
            AgentBinding, result.get("bindings", []), operation="list_agent_bindings"
        )

    async def update_agent_binding(
        self,
        agent_id: str,
        binding_id: str,
        *,
        can_read: bool | None = None,
        write_policy: Literal["deny", "direct"] | None = None,
        is_default: bool | None = None,
    ) -> AgentBinding:
        """Update a binding's scoping fields (owner/admin only).

        ``context_id`` is immutable — :meth:`unbind_agent_context` and
        re-:meth:`bind_agent_context` to re-target. Changes are audited
        with old→new values.

        Args:
            agent_id: Agent UUID.
            binding_id: Binding UUID from :meth:`list_agent_bindings`.
            can_read: New read gate.
            write_policy: New write gate — ``"deny"`` | ``"direct"``.
            is_default: New bootstrap-default flag (max one per agent).

        Returns:
            The updated :class:`AgentBinding`.

        Raises:
            ValueError: If no scoping field is provided (the call would
                be an empty no-op request).
            KaguraNotFoundError: Agent or binding not found.
            KaguraError: Other server-side error.
        """
        changes = _binding_scope_payload(
            can_read=can_read, write_policy=write_policy, is_default=is_default
        )
        if not changes:
            raise ValueError(
                "update_agent_binding requires at least one of "
                "can_read, write_policy, or is_default"
            )
        result = await self._call_tool_checked(
            "update_agent_binding",
            {"agent_id": agent_id, "binding_id": binding_id, **changes},
        )
        return parse_response(AgentBinding, result.get("binding"), operation="update_agent_binding")

    async def unbind_agent_context(self, agent_id: str, binding_id: str) -> bool:
        """Delete a binding — the agent loses that context (owner/admin only).

        Under ``enforcement_mode="enforce"`` the agent's requests against
        the unbound context are denied afterwards (uniform
        ``context_not_found``).

        Args:
            agent_id: Agent UUID.
            binding_id: Binding UUID to delete.

        Returns:
            ``True`` once the server confirms deletion.

        Raises:
            KaguraNotFoundError: Agent or binding not found.
        """
        result = await self._call_tool_checked(
            "unbind_agent_context", {"agent_id": agent_id, "binding_id": binding_id}
        )
        return bool(result.get("deleted", True))

    async def list_contexts(
        self,
        *,
        name_contains: str | None = None,
        include_summary: bool = False,
        include_details: bool = False,
        include_stats: bool = False,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """List the contexts you can access, most recently used first.

        Calls the ``list_contexts`` MCP tool. Since memory-cloud v0.73.0
        (#1600) this is a slim name → id directory by default: each item is
        ``{id, name, is_private, is_locked, last_used_at}``. Summaries and the
        embedding model are opt-in. For one context's full details call
        :meth:`get_context_info`.

        ``name_contains``, ``include_summary`` and ``include_details`` need
        memory-cloud v0.73.0+. An older server ignores them: it returns every
        context, every item already carries ``summary`` and
        ``embedding_model``, and the envelope has no ``total`` (read
        ``result.get("total")`` or use ``len(result["contexts"])``). ``hint``
        needs v0.75.0+. ``include_stats`` works on any server.

        Since memory-cloud v0.82.0 (#1743) the directory is paged: 100
        contexts a page by default (20 with ``include_details`` and no
        ``name_contains``), or ``limit``, and a page also stops at the
        server's 20,000-character budget. That budget is fixed: unlike
        :meth:`get_state`, ``list_contexts`` declares no ``max_chars``, and a
        server from v0.82.0 refuses an argument its schema does not declare
        (``invalid_argument``, #1742). ``has_more`` says more contexts follow;
        pass ``next_cursor`` back as ``cursor`` for the next page, with the
        same ``name_contains`` (the cursor counts rows of the filtered list).

        Args:
            name_contains: Only contexts whose name or display name contains
                this text (case-insensitive, max 100 characters). Sent trimmed,
                as the server reads it, and omitted when blank or unset.
            include_summary: Add ``summary`` truncated to 300 characters; items
                that were cut also carry ``summary_truncated: true``.
            include_details: Add the full ``summary`` and ``embedding_model``
                (the pre-v0.73.0 item shape). Wins over ``include_summary``.
                Large on big workspaces, so combine it with ``name_contains``.
            include_stats: Add ``memory_count`` per context.
            cursor: The ``next_cursor`` of the previous page. Sent only when
                set (memory-cloud v0.82.0+).
            limit: Contexts per page, 1-200 (server default 100). With
                ``include_details`` and no ``name_contains`` the server allows
                at most 20 (and defaults to 20). A page can hold fewer: it
                also stops at 20,000 characters. Keyword-only, sent only when
                set, and checked before any request. Requires memory-cloud
                v0.82.0+ (#1743); an older server ignores it and returns every
                context.

        Returns:
            The envelope ``{status, contexts, count, total, limit, can_create,
            has_more, next_cursor, hint?}``:

            - ``contexts``: the items described above.
            - ``count``: contexts in the workspace. This is quota usage, so
              ``name_contains`` does not change it. With no current workspace
              it is the number of contexts the caller can see, still counted
              before ``name_contains``.
            - ``total``: contexts in this response (v0.73.0+). ``0`` on no
              match is still a success.
            - ``has_more`` / ``next_cursor``: whether more contexts follow
              this page, and the ``cursor`` that returns them (``None`` on the
              last page; v0.82.0+, absent before).
            - ``limit`` / ``can_create``: the plan's context maximum and whether
              another context fits. Absent when the caller has no current
              workspace; ``0`` / ``False`` when the server could not read the
              quota (:meth:`create_context` then skips its local pre-check
              and lets the server decide).
            - ``hint``: present only when the caller can see no context at all
              (memory-cloud v0.75.0+, #1658). With a workspace it says how to
              create a context or get access; without one it says to create or
              select a workspace in the web UI. A ``name_contains`` that
              matches nothing does not produce it, and neither does a failed
              access lookup (which also answers with an empty list).

            The envelope's ``limit`` is the plan's context maximum, not the
            page size argument of the same name.

        Raises:
            ValueError: ``limit`` is not an ``int`` in 1-200, or is above 20
                with ``include_details`` and no ``name_contains`` (the server
                refuses that page with ``validation_error``).
        """
        name_contains = _normalize_name_contains(name_contains)
        if limit is not None:
            _validate_int_range("limit", limit, 1, _LIST_CONTEXTS_MAX_LIMIT)
            if _details_page_over_cap(include_details, name_contains, limit):
                raise ValueError(
                    "include_details returns full summaries: pass name_contains or a limit "
                    f"of at most {_LIST_CONTEXTS_DETAILS_MAX_LIMIT}, got {limit}"
                )
        arguments: dict[str, Any] = {}
        if name_contains is not None:
            arguments["name_contains"] = name_contains
        if include_summary:
            arguments["include_summary"] = True
        if include_details:
            arguments["include_details"] = True
        if include_stats:
            arguments["include_stats"] = True
        if cursor is not None:
            arguments["cursor"] = cursor
        if limit is not None:
            arguments["limit"] = limit
        return await self._call_tool_checked("list_contexts", arguments)

    async def list_tags(
        self,
        context_id: str,
        limit: int = 50,
        min_count: int = 1,
        sort: Literal["count", "recent", "alpha"] = "count",
        prefix: str = "",
        with_tags: list[str] | None = None,
    ) -> ListTagsResponse:
        """List the tag vocabulary in a context with usage counts and recency.

        Call before :meth:`remember` to reuse existing tag spellings, or before
        :meth:`recall` with ``filters={"tags": [...]}`` to build accurate
        filters. This is the primary mitigation for tag drift (e.g. ``"auth"``
        vs ``"authentication"`` silently degrading recall precision).

        Server floors: this tool exists from memory-cloud v0.15.4+ (older
        servers expose ``list_contexts`` and ``recall`` but not ``list_tags``,
        and raise an MCP "tool not found"), and the ``with_tags`` drill-down
        needs v0.17.2+. As elsewhere in this client, ``MIN_SERVER_VERSION`` is
        not bumped to match — floors are tracked per surface.

        A ``with_tags`` drill-down is sent to the REST route
        ``GET /api/v1/contexts/{id}/tags`` rather than MCP, on every server:
        the MCP tool gained ``with_tags`` only in memory-cloud v0.77.0
        (memory-cloud#1669), and before that it silently returned the
        unfiltered vocabulary (#273). The result has the same shape. From
        v0.77.0 the route also sends ``context_name``, so a drill-down is one
        request there; against an older server the client looks the name up
        once per context with a ``list_tags(limit=1)`` MCP call and keeps it
        (a call without ``with_tags`` fills the same cache). The route takes
        API keys and OAuth profiles alike (OAuth needs the ``memory:read``
        scope, which every ``kagura auth login`` grant has).

        Since memory-cloud v0.82.0 (#1743) the MCP tool returns at most 200
        tags whatever the ``limit``, and sets ``has_more`` on the result when
        more matched. The tool takes no cursor: narrow with ``prefix`` or
        ``min_count``. The REST drill-down still returns up to ``limit``.

        Args:
            context_id: Context ID to list tags from.
            limit: Maximum tags to return (1-500, default 50; the server's
                own bound). Over MCP a reply holds at most 200 (see above).
            min_count: Minimum memory count per tag (1-10000, default 1).
            sort: Sort order — ``"count"`` (default), ``"recent"``, or ``"alpha"``.
            prefix: Case-insensitive prefix filter for autocomplete-style lookup.
                ``%`` and ``_`` are treated as literals server-side. Max 200 chars.
            with_tags: Multi-tag AND drill-down. Restricts the count to memories
                whose tags contain **all** of these values, and excludes the
                ``with_tags`` values themselves from the returned vocabulary.
                This is the primitive for faceted browsing — each level is one
                server-side aggregation, with no local index::

                    await client.list_tags(ctx, prefix="client:")
                    await client.list_tags(
                        ctx, prefix="when:", with_tags=["client:acme.co.jp"]
                    )
                    await client.recall(
                        ctx, query=..., filters={"tags": [...], "tags_match": "all"}
                    )

                Requires memory-cloud server v0.17.2+ (#830). Values are
                trimmed and blank ones dropped, as the server does; at most 50
                may remain, each at most 200 characters. An empty result is no
                drill-down and stays on MCP.

        Returns:
            :class:`ListTagsResponse` with ``context_id``, ``context_name``,
            ``tags`` (list of :class:`TagInfo`), ``total`` count and
            ``has_more``.

        Raises:
            ValueError: If ``limit``, ``min_count``, ``prefix`` or
                ``with_tags`` are out of range, before any request.
            TypeError: If ``with_tags`` is a ``str`` rather than a list, or
                holds an item that is not a ``str``.
            KaguraNotFoundError: Context not found or caller lacks access.
            KaguraError: Other server-side error.
            KaguraResponseError: A response that does not match
                :class:`ListTagsResponse` (``operation="list_tags"``).
        """
        if not 1 <= limit <= 500:
            raise ValueError(f"limit must be between 1 and 500, got {limit}")
        if not 1 <= min_count <= 10_000:
            raise ValueError(f"min_count must be between 1 and 10000, got {min_count}")
        if len(prefix) > 200:
            raise ValueError(f"prefix must be at most 200 characters, got {len(prefix)}")
        # A str would be iterated into one-character tags: a wrong drill-down
        # the server would happily run.
        if isinstance(with_tags, str):
            raise TypeError("with_tags must be a list of tags, not a str")
        tags = list(with_tags or ())
        for tag in tags:
            if not isinstance(tag, str):
                raise TypeError(f"with_tags items must be str, got {type(tag).__name__}")
        # Normalized as the server normalizes it, so the caps below judge the
        # list the server would. An empty drill-down matches everything
        # (``tags @> '{}'``), so it is the same as none.
        drill_down = [s for t in tags if (s := t.strip())]
        if len(drill_down) > 50:
            raise ValueError(f"with_tags accepts at most 50 tags, got {len(drill_down)}")
        longest = max(map(len, drill_down), default=0)
        if longest > 200:
            raise ValueError(f"each with_tags value must be at most 200 characters, got {longest}")

        query: dict[str, Any] = {"limit": limit, "min_count": min_count, "sort": sort}
        if prefix:
            query["prefix"] = prefix
        if drill_down:
            # The same query, over REST: MCP list_tags ignores with_tags
            # before memory-cloud v0.77.0 (#273). Send it over MCP, and drop
            # _list_tags_via_rest, ContextTagsResponse and _context_names,
            # once MIN_SERVER_VERSION >= 0.77.0.
            return await self._list_tags_via_rest(context_id, {**query, "with_tags": drill_down})
        result = await self._call_tool_checked("list_tags", {"context_id": context_id, **query})
        return self._remember_context_name(
            parse_response(ListTagsResponse, result, operation="list_tags")
        )

    async def _list_tags_via_rest(
        self, context_id: str, params: dict[str, Any]
    ) -> ListTagsResponse:
        """The ``list_tags`` drill-down over ``GET /api/v1/contexts/{id}/tags`` (#273).

        Reshaped to what the MCP tool returns. The route sends
        ``context_name`` from memory-cloud v0.77.0; for an older server
        :meth:`_context_name_for` supplies it.
        """
        # Quoted, so a caller's id cannot add segments to the request path.
        path = f"/api/v1/contexts/{quote(context_id, safe='')}/tags"
        body = await self._rest_get(
            path, ContextTagsResponse, params, operation="list_tags", mcp_tool="list_tags"
        )
        # Only after the REST call, so its error is the one a caller sees.
        context_name = body.context_name or await self._context_name_for(body.context_id)
        return self._remember_context_name(
            ListTagsResponse(
                context_id=body.context_id,
                context_name=context_name,
                tags=body.tags,
                total=body.total,
            )
        )

    async def _context_name_for(self, context_id: str) -> str:
        """A context's name, from the cache or from one ``list_tags`` MCP call.

        ``list_tags`` runs the same access check as the REST tags route (both
        call the server's ``ContextService.aggregate_tags``), is exempt from
        the MCP rate limit, and with ``limit=1`` carries one tag.
        """
        cached = self._context_names.get(context_id.lower())
        if cached is not None:
            return cached
        result = await self._call_tool_checked("list_tags", {"context_id": context_id, "limit": 1})
        response = parse_response(ListTagsResponse, result, operation="list_tags")
        return self._remember_context_name(response).context_name

    def _remember_context_name(self, response: ListTagsResponse) -> ListTagsResponse:
        """Cache the name a ``list_tags`` result carries, and return the result."""
        self._context_names[response.context_id.lower()] = response.context_name
        return response

    async def get_tool_definitions(self) -> list[dict[str, Any]]:
        """
        Call tools/list to get available MCP tools definitions.

        This retrieves the current tool specifications from the MCP server,
        including tool names, descriptions, and parameter schemas.

        The list is the view the MCP URL's query selects, not every tool: on
        memory-cloud 0.93.0+ a bare URL lists the ``core`` profile (21 tools
        on 0.141.0, memory-cloud#1849), ``?profile=full`` lists all of them
        (72), ``?add_tools=a,b`` adds names to the profile (v0.129.0+,
        memory-cloud#2354) and ``?tools=a,b`` is an allowlist. A profile is
        a view, not an authorization boundary: a tool left out of the list
        stays callable, so every method of this client works whatever it
        returns. Only :meth:`recall`'s ``explore_hints`` depend on it, and
        need ``explore`` listed (``?add_tools=explore`` or ``?profile=full``,
        v0.135.0+).

        Returns:
            List of tool definition dicts with name, description, and inputSchema

        Raises:
            KaguraAuthError: Authentication failed
            KaguraConnectionError: Connection to server failed
            KaguraResponseError: The result is not a ``tools/list`` result
                (e.g. ``"result": null``, or ``tools`` that is not a list of
                objects)
        """
        result = await self._make_jsonrpc_request(method="tools/list", params={})
        listing = parse_response(
            ToolListResponse, result, operation="KaguraClient.get_tool_definitions"
        )
        return listing.tools

    async def explore(
        self,
        context_id: str,
        memory_id: str,
        depth: int = 2,
        min_weight: float = 0.05,
    ) -> dict[str, Any]:
        """
        Call explore MCP tool (neural graph traversal).

        Args:
            context_id: Context ID
            memory_id: Seed memory ID to explore from
            depth: Maximum traversal depth (1-5)
            min_weight: Minimum edge weight threshold

        Returns:
            API response with related memories
        """
        arguments = {
            "context_id": context_id,
            "memory_id": memory_id,
            "depth": depth,
            "min_weight": min_weight,
        }
        return await self._call_tool_checked("explore", arguments)

    async def reference(
        self,
        context_id: str,
        memory_id: str,
        *,
        version: int | None = None,
        content_offset: int | None = None,
    ) -> dict[str, Any]:
        """
        Call reference MCP tool (get full memory details).

        ``version`` and ``content_offset`` are keyword-only and sent only when
        set. From memory-cloud v0.82.0 (#1742) a server refuses an argument it
        does not declare (``KaguraError``, code ``invalid_argument``); one
        before v0.82.0 ignores it.

        Since memory-cloud v0.78.0 (#1685) the reply is bounded (20,000
        characters by default): ``content`` that does not fit comes back
        sliced with ``content_truncated`` / ``content_next_offset``, and a
        ``details`` / ``context`` / links field that does not fit is left out
        with ``<field>_omitted`` and ``<field>_total_chars`` — never cut
        silently.

        Args:
            context_id: Context ID
            memory_id: Memory ID to retrieve full details for
            version: Read an earlier version of the memory, an ``int`` ≥ 1
                from its revision log (memory-cloud v0.113.0+, #2043). The
                reply then carries ``historical: True``, ``current_version``
                and, when the log could not rebuild some fields,
                ``unavailable_fields``; the links and ``supersede_candidate``,
                which describe the memory now, are left out. A version the
                memory never had is ``KaguraError`` with ``code ==
                "validation_error"`` and ``details["current_version"]``.
                Keyword-only.
            content_offset: Continue ``content`` from this character, an
                ``int`` ≥ 0 — the ``content_next_offset`` of a cut
                :meth:`recall` row or of the previous reference reply
                (memory-cloud v0.78.0+, #1685). With an offset the server
                returns that one field only (no ``details``, ``context`` or
                links). An offset past the end is ``KaguraError`` with ``code
                == "invalid_argument"``. Keyword-only.

        Returns:
            API response dict. Memory data is in ``result["memory"]``::

                result = await client.reference(ctx, mem_id)
                memory = result["memory"]
                print(memory["summary"], memory["content"])

            ``memory["version"]`` (memory-cloud v0.113.0+) is the version to
            pass to :meth:`update_memory` as ``expected_version``, and
            ``memory["observed_at"]`` (v0.127.0+, #2330) when the fact held,
            when it is set. A live ``supersede_candidate`` has
            :meth:`recall`'s keys (``verified`` included) plus
            ``acceptance`` — ``{allowed, reason}``, whether the caller may
            accept the suggestion (memory-cloud v0.137.0+, #2427; on every
            :meth:`recall` row that carries a candidate from v0.142.0,
            #2508). ``reason`` is set only when ``allowed`` is false:
            ``not_writable`` (no write access to the context; from v0.142.0
            (#2614) also a memory carrying ``details.tool_trigger`` when the
            caller is below context editor or not on a user credential — the
            guardrail gate the accept runs first), ``not_owner`` (both
            memories must be the caller's own, see :meth:`create_edge`),
            ``newer_candidate``, or ``edge_exists`` (another declared edge
            already links the pair; v0.138.0+, #2518). From v0.142.0
            (#2614) ``edge_exists`` reads only the edge kept by the
            carrier's author, the account the accept writes under, so a
            linked caller's own different edge no longer reports it.
            Accept it with ``update_memory(accept_supersede_candidate=True)``.

        A call without an offset counts as a read in the memory's access
        stats; a continuation counts only when it is the session's first read
        of the memory (v0.122.0+, #2139). REST ``POST /api/v1/memory/reference``
        also takes ``record_access: false`` to read without counting
        (v0.138.0+, #2518); the MCP tool has no such argument and this client
        does not wrap the REST route.

        Raises:
            ValueError: ``version`` is not an ``int`` ≥ 1, or ``content_offset``
                not an ``int`` ≥ 0 (``bool`` is neither, as on the server).
                Checked before any request.
        """
        arguments: dict[str, Any] = {
            "context_id": context_id,
            "memory_id": memory_id,
        }
        if version is not None:
            _validate_int_range("version", version, 1)
            arguments["version"] = version
        if content_offset is not None:
            _validate_int_range("content_offset", content_offset, 0)
            arguments["content_offset"] = content_offset
        return await self._call_tool_checked("reference", arguments)

    async def update_memory(
        self,
        context_id: str,
        memory_id: str | None = None,
        external_id: str | None = None,
        summary: str | None = None,
        content: str | None = None,
        type: str | None = None,
        importance: float | None = None,
        tags: list[str] | None = None,
        context_summary: str | None = None,
        delivery_mode: Literal["always", "on_recall", "on_trigger"] | None = None,
        details: dict[str, Any] | None = None,
        *,
        dismiss_supersede_candidate: bool = False,
        tool_trigger: ToolTrigger | dict[str, Any] | None = None,
        observed_at: str | date | datetime | None = None,
        add_tags: list[str] | None = None,
        remove_tags: list[str] | None = None,
        details_patch: dict[str, Any] | None = None,
        expected_version: int | None = None,
        accept_supersede_candidate: bool = False,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Update an existing memory in-place or upsert by external ID.

        Two modes (provide exactly one of memory_id or external_id):

        1. In-place update (memory_id): Modifies specific fields while
           preserving memory ID, graph edges, and creation timestamp.
        2. Upsert (external_id): Finds by external resource ID within context.
           If found, replaces. If not found, creates new.
           Requires summary, content, and type. In a private context an
           upsert from either of two linked accounts replaces the same memory
           (memory-cloud v0.90.0+, #1803); shared contexts are unchanged.

        The keyword-only arguments from ``observed_at`` on are sent only when
        set, so a server that predates one never sees it. From memory-cloud
        v0.82.0 (#1742) a server refuses an argument it does not declare
        (``KaguraError``, code ``invalid_argument``); one before v0.82.0
        ignores it — an ``add_tags`` / ``details_patch`` edit then changes
        nothing, and ``expected_version`` guards nothing.

        Args:
            context_id: Context UUID.
            memory_id: UUID of memory to update in-place.
            external_id: External resource ID for upsert lookup.
            summary: Updated summary (10-500 chars).
            content: Updated content.
            type: Updated memory type.
            importance: Updated importance (0.0-1.0).
            tags: Updated tags (replaces the list; ``[]`` clears it). At most
                50, each at most 100 characters, as for :meth:`remember`. To
                change only some tags, use ``add_tags`` / ``remove_tags``.
            context_summary: Updated context summary (max 2000 chars).
                ``""`` clears it; ``None`` leaves it unchanged.
            delivery_mode: Pin or unpin the memory. ``"always"`` pins it
                (deterministically loaded every turn via :meth:`load_pinned`,
                promoted to ``scope="persistent"``); ``"on_recall"`` unpins it
                (back to probabilistic :meth:`recall`; the memory stays
                persistent). Omit to leave the current delivery mode unchanged.
            details: Updated structured details JSON. **Replaces ``details``
                wholesale** — read the current value with :meth:`reference`
                and re-send every key you want to keep (notably ``location``,
                which otherwise drops off :meth:`recall_nearby`, and
                ``tool_trigger``, which otherwise unmarks a guardrail), or
                change only some keys with ``details_patch``. ``{}`` clears
                it; omit to leave details unchanged.
            dismiss_supersede_candidate: Reject this memory's current
                ``supersede_candidate`` (the older near-duplicate that
                :meth:`recall` / :meth:`reference` suggest it replaces), for two
                memories that are deliberately separate. Nothing is deleted or
                shadowed, and it can be the only change in the call. Requires
                ``memory_id``. Sent only when ``True``. Requires memory-cloud
                v0.65.0+ (#1504). An older server silently drops the flag: a
                dismissal-only call then succeeds as an empty in-place update
                that dismisses nothing and refreshes ``updated_at``. To accept
                the suggestion instead, pass ``accept_supersede_candidate``.
            tool_trigger: Mark (or re-mark) the memory as a tool guardrail —
                same contract and permissions as :meth:`remember`'s
                ``tool_trigger`` (memory-cloud v0.74.0+). It is merged into
                ``details_patch`` when that is given, which keeps the other
                stored keys; otherwise into ``details``, and because
                ``details`` is replaced wholesale, passing only
                ``tool_trigger`` sends ``details={"tool_trigger": ...}`` and
                drops every other key — pass the keys to keep in ``details``,
                or use ``details_patch``. To unmark, send ``details`` without
                the key (or with ``"tool_trigger": None``), or
                ``details_patch={"tool_trigger": None}``. Any edit of a memory
                that already carries a trigger needs context editor or above.
                Keyword-only.
            observed_at: When the fact held (memory-cloud v0.127.0+, #2330),
                in :meth:`remember`'s forms. Without ``details`` it changes
                only ``details.observed_at`` of the stored details (a details
                edit, so the version moves); with ``details`` or
                ``details_patch`` their own ``observed_at``, if they carry
                one, must equal it. Clear it by resending ``details`` without
                the key or with ``details_patch={"observed_at": None}``.
                Keyword-only.
            add_tags: Tags to append to the stored list, each once, in order,
                after ``remove_tags`` is applied (memory-cloud v0.140.0+,
                #2585). The 50 × 100-character caps of ``tags`` apply here,
                and the server applies the count cap again to the resulting
                list. In-place mode only, never with ``tags``. Keyword-only.
            remove_tags: Tags to remove from the stored list; one that is not
                there is ignored (v0.140.0+, #2585). Not capped: a stored tag
                longer than today's limit can still be removed. In-place mode
                only, never with ``tags``. Keyword-only.
            details_patch: A shallow merge into the stored details
                (v0.140.0+, #2585): each key replaces the stored key, and a
                ``None`` value **deletes** the key. Stored keys the patch does
                not name are kept, so ``location`` and ``tool_trigger``
                survive a status flip. The reserved keys keep their
                validation; a patch that leaves null details empty changes
                nothing. In-place mode only, never with ``details``.
                Keyword-only.
            expected_version: Apply the edit only while the memory is still
                at this version (an ``int`` ≥ 1, the ``version`` that
                :meth:`reference`, :meth:`remember` and this method return;
                memory-cloud v0.113.0+, #2043). Otherwise nothing is written
                and :class:`KaguraVersionConflictError` is raised. Omit for
                last writer wins. In-place mode only. Keyword-only.
            accept_supersede_candidate: Accept this memory's current
                ``supersede_candidate``: write the declared ``supersedes``
                edge from this memory to the candidate, which then leaves
                default :meth:`recall`, and clear the suggestion (memory-cloud
                v0.126.0+, #2308). A graph action only: needs ``memory_id``,
                takes no field edits in the same call and leaves ``version``
                and ``updated_at`` alone; ``expected_version`` may guard it.
                Both memories must be the caller's own (in a private context a
                linked account's count, #1803). On a memory carrying
                ``details.tool_trigger`` the accept first runs the guardrail
                gate (context editor or above, a user credential). Check
                ``supersede_candidate.acceptance`` first, on :meth:`reference`
                (v0.137.0+) or on the :meth:`recall` row (v0.142.0+, #2508):
                it says whether the accept would be refused, and from v0.142.0
                (#2614) reports that guardrail gate as the ``not_writable``
                reason. Sent only when ``True``. Keyword-only.
            verbose: ``True`` asks for ``persistence.detail`` (memory-cloud
                v0.93.0+, #1851). Sent only when ``True``. Keyword-only.

        Returns:
            API response with updated memory info (``memory_id``,
            ``operation``, ``re_embedded``, ``scope``, and ``version`` — the
            version after this call, unchanged when it changed no versioned
            field; memory-cloud v0.113.0+). Since memory-cloud v0.65.0 it may
            also carry ``persistence`` and ``lint`` as in :meth:`remember` —
            ``lint`` describes the memory after the update. When a dismissal
            applied, ``supersede_candidate_dismissed`` holds the rejected
            candidate's memory_id. It is the only confirmation that a
            dismissal happened: the key is absent both when there was no
            live suggestion and when the server predates v0.65.0. An accept
            returns ``superseded_memory_id``, the memory now shadowed.
            ``plan_headroom`` (memory-cloud v0.142.0+, #2615) comes as on
            :meth:`remember`.

        Raises:
            ValueError: Checked before any request, mirroring the server's
                ``validation_error``: neither or both of ``memory_id`` and
                ``external_id``; ``external_id`` with
                ``dismiss_supersede_candidate``, ``accept_supersede_candidate``,
                ``expected_version``, ``add_tags``, ``remove_tags`` or
                ``details_patch`` (an upsert replaces the memory under a new
                id); ``tags`` with ``add_tags`` / ``remove_tags``, or
                ``details`` with ``details_patch`` (a call replaces a field or
                edits it, not both); both supersede flags; the accept with any
                field edit (``tool_trigger`` and ``observed_at`` included);
                ``expected_version`` that is not an ``int`` ≥ 1;
                ``observed_at`` in a form the server refuses, or disagreeing
                with ``details`` / ``details_patch``; ``tool_trigger`` with a
                ``"tool_trigger"`` key where it would be merged; or ``tags`` /
                ``add_tags`` over 50 tags or with a tag over 100 characters.
            KaguraVersionConflictError: ``expected_version`` is not the
                memory's version; nothing was written. Its
                ``current_version`` is the version to resend with, after
                reading the memory again.
            KaguraFeatureNotAvailableError: ``feature == "contexts"``: the
                context has no writable slot because the plan dropped below
                the workspace's contexts (memory-cloud v0.100.0+).
            KaguraError: Server-side rejection, e.g. ``permission_denied`` for
                a guardrail edit below context editor; ``validation_error`` for
                an accept with no live suggestion, a candidate written after
                this memory or a pair that is not both the caller's own; or
                ``code == "edge_exists"`` when a different declared edge kept
                by this memory's author already links the pair (the accept
                writes under that author; ``details["existing_edge"]``;
                nothing written). This error code is not the ``acceptance``
                reason of the same name, which predicts it.
        """
        if not memory_id and not external_id:
            raise ValueError("Provide exactly one of memory_id or external_id")
        if memory_id and external_id:
            raise ValueError("Provide exactly one of memory_id or external_id")
        partial = [
            name
            for name, value in (
                ("add_tags", add_tags),
                ("remove_tags", remove_tags),
                ("details_patch", details_patch),
            )
            if value is not None
        ]
        if external_id:
            # The upsert replaces the memory under a new id: there is no
            # suggestion, version, stored list or stored details to act on.
            if dismiss_supersede_candidate or accept_supersede_candidate:
                flag = (
                    "dismiss_supersede_candidate"
                    if dismiss_supersede_candidate
                    else "accept_supersede_candidate"
                )
                raise ValueError(
                    f"{flag} requires memory_id (in-place mode); "
                    "an external_id upsert replaces the memory and its suggestion."
                )
            if expected_version is not None:
                raise ValueError(
                    "expected_version requires memory_id (in-place mode); an external_id "
                    "upsert replaces the memory under a new id."
                )
            if partial:
                raise ValueError(
                    f"{', '.join(partial)} require memory_id (in-place mode); "
                    "use tags / details with external_id."
                )
        if accept_supersede_candidate and dismiss_supersede_candidate:
            raise ValueError(
                "accept_supersede_candidate and dismiss_supersede_candidate cannot "
                "both be true: accept the suggestion or reject it."
            )
        if tags is not None and (add_tags is not None or remove_tags is not None):
            raise ValueError(
                "tags replaces the stored list and add_tags / remove_tags edit it; "
                "give one or the other."
            )
        if details is not None and details_patch is not None:
            raise ValueError(
                "details replaces the stored details and details_patch merges into them; "
                "give one or the other."
            )
        if expected_version is not None:
            _validate_int_range("expected_version", expected_version, 1)
        _validate_memory_tags(tags)
        _validate_memory_tags(add_tags, "add_tags")
        if details_patch is not None:
            details_patch = _details_with_tool_trigger(details_patch, tool_trigger)
        else:
            details = _details_with_tool_trigger(details, tool_trigger)
        observed = None if observed_at is None else _observed_at_arg(observed_at)
        if observed is not None:
            _check_observed_at_agrees(observed, details, "details")
            _check_observed_at_agrees(observed, details_patch, "details_patch")
        if accept_supersede_candidate:
            # A graph action: an edit in the same call would re-embed and
            # version the memory it marks as superseding (#2308).
            edits = [
                name
                for name, value in (
                    ("summary", summary),
                    ("content", content),
                    ("type", type),
                    ("importance", importance),
                    ("tags", tags),
                    ("context_summary", context_summary),
                    ("delivery_mode", delivery_mode),
                    ("details", details),
                    ("details_patch", details_patch),
                    ("add_tags", add_tags),
                    ("remove_tags", remove_tags),
                    ("observed_at", observed),
                )
                if value is not None
            ]
            if edits:
                raise ValueError(
                    "accept_supersede_candidate cannot be combined with field edits "
                    f"({', '.join(edits)}); send the edit in a separate update_memory call."
                )

        arguments: dict[str, Any] = {"context_id": context_id}
        if memory_id is not None:
            arguments["memory_id"] = memory_id
        if external_id is not None:
            arguments["external_id"] = external_id
        if summary is not None:
            arguments["summary"] = summary
        if content is not None:
            arguments["content"] = content
        if type is not None:
            arguments["type"] = type
        if importance is not None:
            arguments["importance"] = importance
        if tags is not None:
            arguments["tags"] = tags
        if context_summary is not None:
            arguments["context_summary"] = context_summary
        if delivery_mode is not None:
            arguments["delivery_mode"] = delivery_mode
        if details is not None:
            arguments["details"] = details
        if dismiss_supersede_candidate:
            arguments["dismiss_supersede_candidate"] = True
        if add_tags is not None:
            arguments["add_tags"] = add_tags
        if remove_tags is not None:
            arguments["remove_tags"] = remove_tags
        if details_patch is not None:
            arguments["details_patch"] = details_patch
        if observed is not None:
            arguments["observed_at"] = observed
        if expected_version is not None:
            arguments["expected_version"] = expected_version
        if accept_supersede_candidate:
            arguments["accept_supersede_candidate"] = True
        if verbose:
            arguments["verbose"] = True
        return await self._call_tool_checked("update_memory", arguments)

    @overload
    async def forget(
        self,
        context_id: str,
        memory_id: str | None = None,
        query: str | None = None,
        k: int = 10,
        *,
        memory_ids: None = None,
    ) -> dict[str, Any]: ...

    @overload
    async def forget(
        self,
        context_id: str,
        memory_id: None = None,
        query: None = None,
        k: None = None,
        *,
        memory_ids: list[str],
    ) -> ForgetBatchResult: ...

    async def forget(
        self,
        context_id: str,
        memory_id: str | None = None,
        query: str | None = None,
        k: int | None = None,
        *,
        memory_ids: list[str] | None = None,
    ) -> dict[str, Any] | ForgetBatchResult:
        """
        Call forget MCP tool (soft delete memories).

        Delete by specific memory_id, by a list of ids (``memory_ids``) or by
        search query. This is a soft
        delete: a deleted memory stays recoverable until the deployment's
        cleanup sweep purges it. The window is set per deployment
        (``CLEANUP_DELETED_MEMORIES_RETENTION_DAYS``, default 30 days, ``0``
        disables the sweep; memory-cloud v0.66.0+), and Sleep retention
        settings can purge sooner, so do not rely on a fixed recovery period.

        ``query`` mode is refused while recall is degraded (memory-cloud
        v0.66.0+): with the semantic half of hybrid search unavailable, the
        candidates would be a keyword-only set rather than the memories the
        query normally matches, so the server raises an error instead of
        deleting them. Retry once search is healthy, or delete by
        ``memory_id``.

        A target the caller may not delete is **silently skipped**, not an
        error. Since memory-cloud v0.74.0 that includes every tool guardrail
        (a memory carrying ``details.tool_trigger``) when the caller is below
        context editor or uses an agent-bound key: a ``memory_id`` delete
        then reports ``deleted_count: 0``, and a ``query`` sweep deletes its
        other matches without counting the guardrail. Check
        ``deleted_count`` rather than assuming success.

        ``memory_ids`` (memory-cloud v0.117.0+, #2127) forgets 1-50 distinct
        ids in one call, each exactly as ``memory_id`` would and committed on
        its own, and returns a :class:`~kagura_memory.models.ForgetBatchResult`
        with one row per id (``deleted`` / ``skipped`` / ``error``). The ids
        share one time budget (the server's ``forget`` timeout, v0.118.0+,
        #2191): an id it did not reach is an ``error`` row with ``error ==
        "not_attempted"`` — resend those. An older server refuses the
        argument (``invalid_argument``, v0.82.0+).

        Args:
            context_id: Context ID
            memory_id: UUID of specific memory to delete
            query: Search query to find and delete matching memories
            k: Number of memories to delete in query mode (default: 10).
                Query mode only: passed with ``memory_ids`` it raises
                ``ValueError``.
            memory_ids: 1-50 distinct memory UUIDs to delete in one call.
                Keyword-only; not combined with ``memory_id``, ``query`` or
                ``k``.

        Returns:
            For ``memory_id`` / ``query``: the API response dict
            (``deleted_count``, ``memory_ids``) — may be fewer than
            targeted, see above. Since memory-cloud v0.113.0 (#2044) it adds
            ``session_id`` when the request has a session (for
            :meth:`changes_since` / :meth:`rollback_session`). For
            ``memory_ids``: a :class:`~kagura_memory.models.ForgetBatchResult`.

        Raises:
            ValueError: If none of ``memory_id``, ``query`` and
                ``memory_ids`` is provided — ``forget`` always targets a
                specific memory or a search; an unqualified call would be an
                ambiguous no-op — or ``memory_ids`` is combined with another
                target or with ``k``, is empty, holds more than 50 ids, an id
                that is not a UUID, or the same id twice (all checked before
                any request).
            TypeError: ``memory_ids`` is a bare ``str``, not a list, or holds a
                non-``str``.
            KaguraError: Server-side rejection, e.g. a ``query`` delete while
                recall is degraded. With ``memory_ids``, every id failing is
                ``code == "batch_failed"`` with the rows in
                ``details["results"]``.
            KaguraResponseError: A ``memory_ids`` reply does not match
                :class:`~kagura_memory.models.ForgetBatchResult`
                (``operation="forget"``).
        """
        if memory_ids is not None:
            if memory_id or query:
                raise ValueError("Provide exactly one of memory_id, memory_ids or query")
            if k is not None:
                raise ValueError("k applies to query mode only, not to memory_ids")
            result = await self._call_tool_checked(
                "forget", {"context_id": context_id, "memory_ids": _checked_memory_ids(memory_ids)}
            )
            return parse_response(ForgetBatchResult, result, operation="forget")
        if not memory_id and not query:
            raise ValueError("Provide either memory_id or query")
        arguments: dict[str, Any] = {"context_id": context_id}
        if memory_id:
            arguments["memory_id"] = memory_id
        if query:
            arguments["query"] = query
            arguments["k"] = _FORGET_DEFAULT_K if k is None else k
        return await self._call_tool_checked("forget", arguments)

    async def bootstrap(
        self,
        context_id: str,
        *,
        since: str | datetime | None = None,
        include: list[BootstrapComponentName] | None = None,
        max_chars: int | None = None,
        format: BootstrapFormat | None = None,
    ) -> BootstrapResponse:
        """Open an interactive session on one context in one call.

        Calls the ``bootstrap`` MCP tool (memory-cloud v0.93.0+, #1851): the
        context block, its tool ``guardrails``, the trusted pinned set
        (cap 20), the upcoming Time Memories (k = 20) and the change log
        since ``since`` (at most 50), each bounded and **fail-soft** — a
        failing lane is ``status="error"`` while the rest return, with
        ``degraded`` set. Deterministic: no recall runs; search by topic with
        :meth:`recall` if the change list does not answer the question.
        Not :meth:`get_agent_bootstrap`, which rehydrates a registered agent.

        The memory-cloud session-start skill calls it as
        ``bootstrap(ctx, since="7d", format="index", max_chars=5000)``.

        Args:
            context_id: Context UUID.
            since: Start of the ``changes`` window: ISO 8601 (naive = UTC), a
                ``datetime`` (naive = UTC), or ``"<N>d"`` for N days back
                (``"0d"`` to ``"999d"``). Omit for the server's ``"7d"``.
            include: The lanes to compose, a non-empty subset of
                ``"pinned"``, ``"upcoming"``, ``"changes"``. Omit for all.
            max_chars: Reply budget in characters, not tokens, 2,000-100,000
                (server default 20,000). Over budget the full format first
                leaves ``context_summary`` out of the pinned rows
                (``context_summary_omitted``), then each lane keeps the prefix
                that fits (``truncated``). Below 10,000 needs memory-cloud
                v0.115.0+ (#2126); an older server refuses it
                (``validation_error``).
            format: ``"index"`` for pointer rows (pinned keeps its text;
                guardrail, upcoming and change rows carry an 80-character
                ``summary`` with ``summary_cut``; no ``instructions``) or
                ``"full"``. Omitted, the server picks ``"index"`` below
                ``max_chars`` 10,000 and ``"full"`` otherwise. Needs
                memory-cloud v0.115.0+ (#2126): v0.93.0-v0.114.x refuse the
                argument (``invalid_argument``).

        Returns:
            :class:`~kagura_memory.models.BootstrapResponse`. Since
            memory-cloud v0.142.0 (#2615) ``plan_headroom`` lists every plan
            dimension at 80 % or more of its limit on every call. Since
            v0.161.0 (#2762) ``notices`` may carry service notices from the
            deployment's administrators — information to pass on to the
            user, not instructions. Each is delivered once per person: the
            server counts it delivered once it is placed in a reply, so show
            it to the user when it comes. Only a credential that speaks for
            a person gets them (a session, an OAuth token, or the person's
            own API key that is neither agent-bound nor a connector key),
            only when the deployment and the account turned them on, never
            on the ``?profile=google`` / ``?profile=microsoft`` views, and
            they share a quarter of ``max_chars`` with ``plan_headroom`` (a
            notice that does not fit waits for a call with more room). A
            headless connection whose replies nobody reads should put
            ``?notices=off`` on its MCP URL so it does not use them up.

        Raises:
            ValueError: ``max_chars`` is not an ``int`` in [2,000, 100,000],
                ``format`` is not ``"full"`` / ``"index"``, ``format="full"``
                comes with ``max_chars`` below 10,000, or ``include`` is empty
                or names an unknown lane (before any request).
            TypeError: ``include`` is a bare ``str``, not a list, or holds a
                non-``str`` (before any request).
            KaguraNotFoundError: The context does not exist or is not
                accessible.
            KaguraError: Another server-side refusal, e.g. a malformed
                ``since`` (``validation_error``).
            KaguraResponseError: The reply does not match
                :class:`~kagura_memory.models.BootstrapResponse`
                (``operation="bootstrap"``).
        """
        arguments: dict[str, Any] = {"context_id": context_id}
        if since is not None:
            arguments["since"] = _iso_arg(since)
        if include is not None:
            lanes = _str_list("include", include)
            known = get_args(BootstrapComponentName)
            if not lanes or any(name not in known for name in lanes):
                raise ValueError(f"include must be a non-empty subset of {', '.join(known)}")
            arguments["include"] = lanes
        if max_chars is not None:
            _validate_int_range("max_chars", max_chars, _BOOTSTRAP_MIN_CHARS, _MAX_CHARS_MAX)
            arguments["max_chars"] = max_chars
        if format is not None:
            if format not in get_args(BootstrapFormat):
                raise ValueError(f"format must be 'full' or 'index', got {format!r}")
            if format == "full" and max_chars is not None and max_chars < _MAX_CHARS_MIN:
                raise ValueError(
                    f'format="full" needs max_chars of at least {_MAX_CHARS_MIN}; '
                    'pass format="index" (or omit format) under it'
                )
            arguments["format"] = format
        result = await self._call_tool_checked("bootstrap", arguments)
        return parse_response(BootstrapResponse, result, operation="bootstrap")

    async def changes_since(
        self,
        context_id: str,
        since: str | datetime,
        *,
        until: str | datetime | None = None,
        kinds: list[str] | None = None,
        cursor: str | None = None,
        limit: int | None = None,
        session_id: str | None = None,
        by_me: bool | None = None,
        max_chars: int | None = None,
    ) -> ChangesSinceResponse:
        """List what changed in a context since a time, oldest first.

        Calls the ``changes_since`` MCP tool (memory-cloud v0.93.0+, #1852): a
        deterministic memory-level change log — ``created``, ``updated``
        (Sleep maintenance included), ``superseded`` (with
        ``superseded_by``) and ``forgotten`` events — keyset-paged. A soft
        delete logs ``forgotten`` only (v0.98.0+, #1924), and every event of
        a memory that is forgotten now has no ``summary`` (v0.96.0+, #1876).
        No ranking, no embedding, no Hebbian write. In a private context only
        the caller's own memories (and a linked account's) are listed.

        Args:
            context_id: Context UUID.
            since: Window start, inclusive: ISO 8601 or a ``datetime`` (naive
                = UTC).
            until: Window end, exclusive (later than ``since``). Omit for now.
            kinds: Kinds to include, a non-empty subset of ``created``,
                ``updated``, ``superseded``, ``forgotten`` — or, with
                ``session_id`` / ``by_me``, of ``created``, ``updated``,
                ``forgotten``, ``restored``. Omit for all.
            cursor: ``next_cursor`` of the previous page.
            limit: Changes per page (server default 50, clamped to 1-100).
            session_id: Only that session's writes (a ``session_id`` a write
                returned; memory-cloud v0.113.0+, #2044). Another principal's
                session lists nothing, unless the caller owns the context.
            by_me: ``True`` lists only the caller's own writes, in any session
                (memory-cloud v0.114.0+, #2121) — what
                ``rollback_session(since=...)`` would address.
            max_chars: Reply budget in characters, 10,000-100,000 (server
                default 20,000); a cut page sets ``truncated``.

        Returns:
            :class:`~kagura_memory.models.ChangesSinceResponse`; call again
            with ``cursor=result.next_cursor`` while ``result.has_more``.

        Raises:
            ValueError: ``max_chars`` is out of range (before any request).
            TypeError: ``kinds`` is a bare ``str``, not a list, or holds a
                non-``str`` (before any request).
            KaguraNotFoundError: The context does not exist or is not
                accessible.
            KaguraError: Another server-side refusal (``validation_error``:
                a malformed time, ``until`` not after ``since``, a bad
                ``kinds`` / ``session_id`` / ``cursor``). Each argument after
                ``cursor``/``limit`` is refused as ``invalid_argument`` by a
                server older than its floor.
            KaguraResponseError: The reply does not match
                :class:`~kagura_memory.models.ChangesSinceResponse`
                (``operation="changes_since"``).
        """
        arguments: dict[str, Any] = {"context_id": context_id, "since": _iso_arg(since)}
        if until is not None:
            arguments["until"] = _iso_arg(until)
        if kinds is not None:
            arguments["kinds"] = _str_list("kinds", kinds)
        if cursor is not None:
            arguments["cursor"] = cursor
        if limit is not None:
            arguments["limit"] = limit
        if session_id is not None:
            arguments["session_id"] = session_id
        if by_me is not None:
            arguments["by_me"] = by_me
        if max_chars is not None:
            _validate_max_chars(max_chars)
            arguments["max_chars"] = max_chars
        result = await self._call_tool_checked("changes_since", arguments)
        return parse_response(ChangesSinceResponse, result, operation="changes_since")

    async def list_matching(
        self,
        context_id: str,
        *,
        filters: dict[str, Any] | None = None,
        order_by: ListOrderBy | None = None,
        direction: Literal["asc", "desc"] | None = None,
        cursor: str | None = None,
        limit: int | None = None,
        include_details: bool = False,
        max_chars: int | None = None,
    ) -> ListMatchingResponse:
        """List every live memory of a context that matches exact filters.

        Calls the ``list`` MCP tool (memory-cloud v0.93.0+, #1852): no
        ranking, a stable order and a ``total``, e.g. the open backlog
        ``filters={"type": ["tech-debt"], "tags": ["status:open"]}`` or
        ``{"type": "task", "details.status": "open"}``. Use :meth:`recall`
        for meaning.

        :meth:`list_memories` is a different call: REST
        ``GET /api/v1/memory/list``, with a summary substring (``q``), a
        trigger window, a bounding box and the cross-context "my memories"
        view. This method is MCP and takes ``recall``'s exact filter
        vocabulary plus ``details.<key>`` equality, orders by
        ``updated_at`` / ``created_at`` / ``importance`` / ``observed_at``
        and pages with a cursor. In a private context both list only the
        caller's own memories.

        Args:
            context_id: Context UUID.
            filters: Exact-match filters, ANDed: ``type``, ``scope``,
                ``source_type``, ``delivery_mode``, ``tags`` (+ ``tags_match``
                ``any`` | ``all``, ``tags_normalize``), ``importance``
                ``{gte|lte|gt|lt}``, ``created_after`` / ``created_before`` /
                ``updated_after`` / ``updated_before`` (ISO 8601),
                ``observed_after`` / ``observed_before`` (v0.127.0+),
                ``source_uri_prefix``, ``trust_tier="trusted"``, and
                ``details.<key>`` — a scalar matched with its stored JSON type
                (``2`` is not ``2.0`` or ``"2"``). The server refuses an
                unknown filter (``validation_error``).
            order_by: ``"updated_at"`` (server default; a never-edited memory
                sorts by ``created_at``), ``"created_at"``, ``"importance"``,
                ``"observed_at"`` (memory-cloud v0.127.0+, #2330) or
                ``"event_start"`` (v0.152.0+, #2724); the last two put
                memories without one last. The id breaks ties.
            direction: ``"asc"`` or ``"desc"`` (server default).
            cursor: ``next_cursor`` of the previous page.
            limit: Memories per page (server default 50, clamped to 1-100).
            include_details: ``True`` adds each memory's ``details`` (dropped
                first when the reply passes ``max_chars``:
                ``details_omitted``). Sent only when ``True``.
            max_chars: Reply budget in characters, 10,000-100,000 (server
                default 20,000); a cut page sets ``truncated``.

        Returns:
            :class:`~kagura_memory.models.ListMatchingResponse`; call again
            with ``cursor=result.next_cursor`` while ``result.has_more``.

        Raises:
            ValueError: ``max_chars`` is out of range (before any request).
            KaguraNotFoundError: The context does not exist or is not
                accessible.
            KaguraError: Another server-side refusal (``validation_error``:
                an unknown filter, ``order_by`` or ``direction``).
            KaguraResponseError: The reply does not match
                :class:`~kagura_memory.models.ListMatchingResponse`
                (``operation="list"``).
        """
        arguments: dict[str, Any] = {"context_id": context_id}
        if filters is not None:
            arguments["filters"] = filters
        if order_by is not None:
            arguments["order_by"] = order_by
        if direction is not None:
            arguments["direction"] = direction
        if cursor is not None:
            arguments["cursor"] = cursor
        if limit is not None:
            arguments["limit"] = limit
        if include_details:
            arguments["include_details"] = True
        if max_chars is not None:
            _validate_max_chars(max_chars)
            arguments["max_chars"] = max_chars
        result = await self._call_tool_checked("list", arguments)
        return parse_response(ListMatchingResponse, result, operation="list")

    async def remember_batch(
        self,
        context_id: str,
        items: list[dict[str, Any]],
        *,
        atomic: bool | None = None,
        dedupe: DedupeMode | None = None,
        tags_normalize: bool | None = None,
        verbose: bool | None = None,
    ) -> RememberBatchResult:
        """Store up to 50 memories in one call.

        Calls the ``remember_batch`` MCP tool (memory-cloud v0.93.0+, #1853).
        Each item is validated and limited like a single :meth:`remember`.
        Without ``atomic`` each item is written and reported on its own; with
        ``atomic=True`` the batch is one transaction — a failing item, or a
        ``dedupe="check"`` candidate, rolls everything back and the other
        items come back ``skipped``.

        Args:
            context_id: Context UUID; it applies to every item.
            items: 1-50 dicts of the server's ``remember`` arguments, by
                their wire names: ``summary``, ``content`` and ``type`` are
                required, plus any of ``importance``, ``tags``,
                ``source_uri``, ``source_type``, ``context_summary``,
                ``details`` (a tool guardrail is
                ``details={"tool_trigger": {...}}``), ``observed_at``
                (memory-cloud v0.127.0+), ``context``, ``delivery_mode``,
                ``linked_memory_ids``, ``linked_source_uris``,
                ``supersedes`` — no SDK defaults are added. An item may
                repeat the batch's ``context_id`` but name no other. An
                undeclared key fails that item (``invalid_argument``; a
                whole atomic batch is refused). Items are sent as given,
                ``dedupe`` / ``tags_normalize`` / ``verbose`` included, and
                the server decides on them (memory-cloud v0.156.0+, #2753):
                an item may repeat the value set here; when one is not set
                here and every otherwise accepted item carrying it agrees on
                one valid value, that value applies to the whole batch and
                the reply's ``lint`` names it (``batch_level_argument``); an
                item whose value differs from the one set here is refused,
                and when items disagree with each other (or agree on an
                invalid value) every item carrying the key is refused. The
                server compares JSON type and value after ``remember``'s
                string coercion (``"true"`` equals ``True``, ``1`` does not).
                A server before v0.156.0 refuses every item carrying one of
                the three. Set them here instead.
            atomic: ``True``: all or nothing. Server default ``False``.
            dedupe: As ``remember``'s, applied per item: ``"suggest"``
                (server default), ``"check"`` (a near-duplicate is returned as
                a ``duplicate_candidate`` row instead of written) or
                ``"off"``.
            tags_normalize: ``True`` stores each item's tags under the
                context's established spelling (reported in ``lint``).
            verbose: ``True`` keeps ``persistence.detail`` on each row.

        Returns:
            :class:`~kagura_memory.models.RememberBatchResult` with one row
            per item. Check ``failed`` / ``candidates`` / ``skipped`` and
            each row's ``status``; on ``committed_after_timeout`` every item
            is stored — do not resend. ``lint`` (v0.156.0+) names the
            batch-level keys the server took from the items.

        Raises:
            ValueError: ``items`` is empty or holds more than 50 items, an
                item carries a ``context_id`` other than the batch's, or an
                item's ``tags`` break the 50-tag / 100-character cap (all
                before any request). A batch-level key on an item is the
                server's to accept or refuse: a refused item is an ``error``
                row with ``invalid_argument`` (before v0.156.0 every such
                item), and with ``atomic=True`` it refuses the whole batch
                (``batch_refused`` below).
            TypeError: ``items`` is not a list, or an item is not a dict.
            KaguraQuotaError: An atomic batch did not fit the daily quota
                (``quota_exceeded``; nothing was written, a smaller batch may
                fit).
            KaguraCapacityLockedError: The workspace is over its Free
                capacity.
            KaguraError: Nothing was written. ``code == "batch_failed"``
                (every item failed) or ``"batch_refused"`` (an atomic batch
                with an invalid item) carries the per-item rows in
                ``details["results"]``; a ``dedupe_unavailable`` item (the
                duplicate check is down) is such a row — retry it or send
                ``dedupe="off"``. ``"timeout"`` (an atomic batch rolled back
                at its time limit) carries no rows: send fewer items, or drop
                ``atomic``.
            KaguraResponseError: The reply does not match
                :class:`~kagura_memory.models.RememberBatchResult`
                (``operation="remember_batch"``).
        """
        arguments: dict[str, Any] = {
            "context_id": context_id,
            "items": _checked_batch_items(items, context_id),
        }
        if atomic is not None:
            arguments["atomic"] = atomic
        if dedupe is not None:
            arguments["dedupe"] = dedupe
        if tags_normalize is not None:
            arguments["tags_normalize"] = tags_normalize
        if verbose is not None:
            arguments["verbose"] = verbose
        result = await self._call_tool_checked("remember_batch", arguments)
        return parse_response(RememberBatchResult, result, operation="remember_batch")

    async def describe_tools(
        self,
        *,
        query: str | None = None,
        names: list[str] | None = None,
    ) -> ToolDescriptions:
        """Find the tools this MCP URL's ``tools/list`` leaves out.

        Calls the ``describe_tools`` MCP tool (memory-cloud v0.93.0+, #1849).
        A profile only shapes ``tools/list``: this client can call every
        tool, but a model behind a mainstream MCP client can call listed
        tools only, so the reply says which URL query lists more.

        Args:
            query: Case-insensitive substring narrowing the hidden-tool list
                by name, title or summary.
            names: Up to 20 tool names whose complete definitions to return
                instead of the list.

        Returns:
            :class:`~kagura_memory.models.ToolDescriptions`.

        Raises:
            ValueError: ``names`` holds more than 20 names (before any
                request).
            TypeError: ``names`` is a bare ``str`` or holds a non-``str``.
            KaguraResponseError: The reply does not match
                :class:`~kagura_memory.models.ToolDescriptions`
                (``operation="describe_tools"``).
        """
        arguments: dict[str, Any] = {}
        if query is not None:
            arguments["query"] = query
        if names is not None:
            arguments["names"] = _checked_names("names", names, _DESCRIBE_MAX_NAMES)
        result = await self._call_tool_checked("describe_tools", arguments)
        return parse_response(ToolDescriptions, result, operation="describe_tools")

    async def guide(self, topics: list[str] | None = None) -> GuideResponse:
        """Read sections of the server's tool manual.

        Calls the ``guide`` MCP tool (memory-cloud v0.93.0+, #1850): the
        walkthroughs the short tool descriptions leave out — how to read a
        reply, when to pick one tool over another, the write rules. A caller
        error's ``help`` often names a topic (``'Manual: guide(["<tool>"])'``).
        Static text, nothing caller-specific.

        Args:
            topics: Up to 20 tool names (``"recall"``) or ``"<tool>.<section>"``
                topics. Omit for the server's ``["index"]``, which lists them
                all.

        Returns:
            :class:`~kagura_memory.models.GuideResponse`; unknown names are in
            ``unknown``, not raised.

        Raises:
            ValueError: ``topics`` holds more than 20 names (before any
                request).
            TypeError: ``topics`` is a bare ``str`` or holds a non-``str``.
            KaguraResponseError: The reply does not match
                :class:`~kagura_memory.models.GuideResponse`
                (``operation="guide"``).
        """
        arguments: dict[str, Any] = {}
        if topics is not None:
            arguments["topics"] = _checked_names("topics", topics, _GUIDE_MAX_TOPICS)
        result = await self._call_tool_checked("guide", arguments)
        return parse_response(GuideResponse, result, operation="guide")

    async def list_revisions(
        self,
        context_id: str,
        memory_id: str,
        *,
        limit: int | None = None,
        cursor: str | None = None,
        max_chars: int | None = None,
    ) -> RevisionsPage:
        """Read one memory's revision log, newest first.

        Calls the ``list_revisions`` MCP tool (memory-cloud v0.113.0+,
        #2043): one row per ``created`` / ``updated`` (with the fields it
        changed) / ``forgotten`` / ``restored`` event, each with the version
        after it. Same permission as :meth:`reference`; a forgotten memory's
        log is unreadable (``memory_not_found``).

        Args:
            context_id: Context UUID.
            memory_id: Memory UUID.
            limit: Events per page (server default 20, clamped to 1-100).
            cursor: ``next_cursor`` of the previous page.
            max_chars: Reply budget in characters, 10,000-100,000 (server
                default 20,000); a cut page sets ``truncated``.

        Returns:
            :class:`~kagura_memory.models.RevisionsPage`.

        Raises:
            ValueError: ``max_chars`` is out of range (before any request).
            KaguraNotFoundError: The memory or context does not exist, is not
                accessible, or the memory is forgotten.
            KaguraResponseError: The reply does not match
                :class:`~kagura_memory.models.RevisionsPage`
                (``operation="list_revisions"``).
        """
        arguments: dict[str, Any] = {"context_id": context_id, "memory_id": memory_id}
        if limit is not None:
            arguments["limit"] = limit
        if cursor is not None:
            arguments["cursor"] = cursor
        if max_chars is not None:
            _validate_max_chars(max_chars)
            arguments["max_chars"] = max_chars
        result = await self._call_tool_checked("list_revisions", arguments)
        return parse_response(RevisionsPage, result, operation="list_revisions")

    async def rollback_session(
        self,
        context_id: str,
        *,
        session_id: str | None = None,
        since: str | datetime | None = None,
        until: str | datetime | None = None,
        dry_run: bool = True,
    ) -> SessionRollbackResult:
        """Plan or undo what one session — or the caller in a window — wrote.

        Calls the ``rollback_session`` MCP tool (memory-cloud v0.113.0+,
        #2044). **Destructive with** ``dry_run=False``: per memory, one the
        session created is forgotten, one it edited gets the values from
        before its first edit back (a new version), one it forgot is
        restored and re-embedded. A memory someone else changed since, Sleep
        merged, a resource row or one the caller may not write
        (``not_permitted``, v0.134.0+, #2432) is skipped with a cause, never
        half-reverted. Only the caller's own writes (same user or a linked
        account, same agent credential); a context owner may name any
        session. Needs the context editor role. A call handles at most 500
        events (100 restores): ``status == "in_progress"`` with
        ``remaining > 0`` means call again with the same arguments. Once
        complete, further calls are refused (``already_rolled_back``) until
        new writes land.

        Args:
            context_id: Context UUID.
            session_id: The session whose writes to undo — the
                ``session_id`` a ``remember`` / ``update_memory`` / ``forget``
                / :meth:`remember_batch` reply returned.
            since: Instead of ``session_id``: undo the caller's own writes
                since then (ISO 8601 or ``datetime``, naive = UTC; memory-cloud
                v0.114.0+, #2121). Preview them with
                ``changes_since(since=..., by_me=True)``.
            until: Window end, exclusive, at most 7 days after ``since``.
                Omit for now, or for the end of the caller's unfinished
                rollback of a window from the same ``since``.
            dry_run: ``True`` (default) only plans and changes nothing;
                ``False`` rolls back. Always sent, so the plan-only default
                never depends on the server's.

        Returns:
            :class:`~kagura_memory.models.SessionRollbackResult`.

        Raises:
            ValueError: Neither or both of ``session_id`` and ``since`` are
                given, or ``until`` comes without ``since`` (before any
                request).
            KaguraNotFoundError: ``session_not_found`` (no session with this
                id the caller may roll back in this context), or the context
                is not accessible.
            KaguraPartialSessionRollbackError: Some items failed this call;
                every other item was handled and saved. ``session_summary``
                lists the failures in ``errors``; ``totals``, ``rollback_id``
                and ``remaining`` come with it. Call again with the same
                arguments to retry the failed items.
            KaguraError: ``no_writes_in_window``, ``already_rolled_back``,
                ``permission_denied`` (below editor) or ``validation_error``
                (e.g. a window wider than 7 days), in ``code``.
            KaguraResponseError: The reply does not match
                :class:`~kagura_memory.models.SessionRollbackResult`
                (``operation="rollback_session"``).
        """
        if (session_id is None) == (since is None):
            raise ValueError("Provide exactly one of session_id or since")
        if until is not None and since is None:
            raise ValueError("until needs since")
        arguments: dict[str, Any] = {"context_id": context_id, "dry_run": dry_run}
        if session_id is not None:
            arguments["session_id"] = session_id
        if since is not None:
            arguments["since"] = _iso_arg(since)
        if until is not None:
            arguments["until"] = _iso_arg(until)
        result = await self._call_tool("rollback_session", arguments)
        if result.get("status") == "error" and result.get("error") == "partial_rollback":
            raise _partial_session_rollback_error(result)
        self._raise_for_mcp_error(result, "rollback_session")
        return parse_response(SessionRollbackResult, result, operation="rollback_session")

    async def create_context(
        self,
        name: str,
        display_name: str | None = None,
        description: str | None = None,
        summary: str | None = None,
        usage_guide: str | None = None,
        resource_id: str | None = None,
        is_private: bool = True,
        embedding_model: str | None = None,
    ) -> dict[str, Any]:
        """Create a new context in the current workspace.

        Args:
            name: Context name (lowercase alphanumeric + hyphen/underscore).
            display_name: Human-readable display name.
            description: Context description.
            summary: LLM-oriented summary (200-500 chars).
            usage_guide: LLM-oriented memory usage guidelines.
            resource_id: Deprecated and not sent (#273): the server's
                ``create_context`` does not read it (memory-cloud through
                v0.76.0), so it was always dropped. Passing it emits a
                :class:`DeprecationWarning`. Set it afterwards with
                :meth:`update_context` (owner only), or create a resource
                context with :meth:`setup_resource`.
            is_private: Privacy flag (default: True).
            embedding_model: Embedding model for this context. It is fixed at
                creation — no SDK call changes it — but since memory-cloud
                v0.66.0 an operator can migrate a context to another model,
                so read the current one from
                ``get_context_info().context.embedding_model`` rather than
                caching it. :meth:`list_embedding_models` lists the models the
                deployment offers; any other is refused with
                ``invalid_embedding_model``.

        Returns:
            Created context dict with id, name, and metadata.

        Raises:
            KaguraQuotaError: Context limit reached for this workspace
                (``quota_type="contexts"``). The SDK checks ``list_contexts``
                first, so this usually carries ``current`` / ``limit`` but
                no ``required_plan``.
            KaguraFeatureNotAvailableError: The plan does not allow a
                shared context (``is_private=False``; server v0.75.0+).
        """
        if resource_id is not None:
            warnings.warn(
                "create_context(resource_id=...) is deprecated and ignored: the server's "
                "create_context does not read resource_id, so it is not sent. Set it "
                "afterwards with update_context(context_id, resource_id=...), or use "
                "setup_resource().",
                DeprecationWarning,
                stacklevel=2,
            )
        # Pre-check quota. count / limit / can_create do not depend on the
        # page, so one row is enough (an older server ignores limit).
        contexts = await self.list_contexts(limit=1)
        count = contexts.get("count")
        limit = contexts.get("limit")
        # ``limit: 0`` with ``can_create: false`` is how list_contexts reports
        # a failed quota lookup (every plan allows at least one context), so
        # the server's own check decides then (#256).
        if not contexts.get("can_create", True) and limit != 0:
            from .exceptions import KaguraQuotaError

            # Use ``.get`` then coerce None → "?" so a quota response that is
            # missing count/limit OR carries them as null (both forms of server
            # schema drift) still yields a clean message, never KeyError or a
            # literal "None" (issue #183). A real 0 is preserved as "0".
            raise KaguraQuotaError(
                f"Context limit reached ({'?' if count is None else count}/"
                f"{'?' if limit is None else limit}). "
                "Delete unused contexts or upgrade your plan.",
                quota_type="contexts",
                current=_opt_int(count),
                limit=_opt_int(limit),
            )

        arguments: dict[str, Any] = {"name": name, "is_private": is_private}
        if display_name is not None:
            arguments["display_name"] = display_name
        if description is not None:
            arguments["description"] = description
        if summary is not None:
            arguments["summary"] = summary
        if usage_guide is not None:
            arguments["usage_guide"] = usage_guide
        if embedding_model is not None:
            arguments["embedding_model"] = embedding_model
        return await self._call_tool_checked("create_context", arguments)

    async def delete_context(self, context_id: str) -> dict[str, Any]:
        """Soft-delete a context and all its memories.

        Owner only. Since memory-cloud v0.101.0 (#1960) the context's files
        are soft-deleted with it and their bytes leave the storage quota at
        once; an operator restore within 6 days brings them back unless the
        same content was uploaded again meanwhile. (The server's ``DELETE
        /api/v1/contexts/{id}/files``, which clears a context's files without
        deleting it, takes a browser session only and is not wrapped.) Since
        v0.126.0 a workspace viewer cannot delete a context, not even a
        private one it created.

        Args:
            context_id: Context UUID to delete.

        Returns:
            API response with deletion confirmation.

        Raises:
            KaguraNotFoundError: The context does not exist or the caller
                cannot reach it.
            KaguraError: ``permission_denied`` when the caller is not the
                owner; ``validation_error`` (memory-cloud v0.82.0+, #1742)
                for a business-rule refusal such as the workspace's default
                context, with nothing deleted.
        """
        return await self._call_tool_checked("delete_context", {"context_id": context_id})

    async def update_context(
        self,
        context_id: str,
        display_name: str | None = None,
        description: str | None = None,
        summary: str | None = None,
        usage_guide: str | None = None,
        resource_id: str | None = None,
        is_public: bool | None = None,
        is_locked: bool | None = None,
    ) -> dict[str, Any]:
        """Update an existing context's settings.

        Args:
            context_id: Context UUID to update.
            display_name: Updated human-readable display name.
            description: Updated context description.
            summary: Updated LLM-oriented summary (max 500 chars).
            usage_guide: Updated LLM-oriented usage guidelines (max 2000 chars).
            resource_id: Updated resource identifier for external data ingestion:
                lowercase letters, digits, ``_`` and ``-``, unique on the
                server (a slug another workspace's context uses conflicts
                too; the message says so since memory-cloud v0.129.0, #2284).
                Changing it revokes the active tokens the caller minted for
                the old slug; tokens other owners minted stay active.
            is_public: Updated public visibility (required for resource tokens).
                Making a context public is plan-gated (``plan_required``).
                Since memory-cloud v0.68.0 (#1551) the gate is the plan's
                ``public_contexts`` feature (XL only by default); earlier
                servers gated it on plans with shared contexts. A context
                that is already public keeps serving. Making one private is
                refused with ``cannot_make_private`` while it has a
                ``resource_id``.
            is_locked: Lock/unlock context. Locked contexts cannot be deleted.

        Returns:
            Updated context dict.

        Raises:
            KaguraFeatureNotAvailableError: ``is_public=True`` on a plan
                without the ``public_contexts`` feature.
        """
        arguments: dict[str, Any] = {"context_id": context_id}
        if display_name is not None:
            arguments["display_name"] = display_name
        if description is not None:
            arguments["description"] = description
        if summary is not None:
            arguments["summary"] = summary
        if usage_guide is not None:
            arguments["usage_guide"] = usage_guide
        if resource_id is not None:
            arguments["resource_id"] = resource_id
        if is_public is not None:
            arguments["is_public"] = is_public
        if is_locked is not None:
            arguments["is_locked"] = is_locked
        return await self._call_tool_checked("update_context", arguments)

    async def setup_resource(
        self,
        resource_id: str,
        name: str | None = None,
        summary: str | None = None,
        description: str | None = None,
        quota_events_per_hour: int = 1000,
    ) -> dict[str, Any]:
        """Atomically create Context + Resource entity + ingestion token.

        Wraps the v0.14 server-side ``setup_resource`` MCP tool, which performs
        Context creation, Resource entity binding, and token issuance in a
        single transaction. On failure, no orphan rows are left on the server.

        Args:
            resource_id: Resource identifier for data ingestion: lowercase
                letters, digits, ``_`` and ``-``, at most 255 characters. It
                is unique on the server, so a slug another workspace's
                context uses conflicts too (``resource_id_conflict``; the
                message says so since memory-cloud v0.129.0, #2284).
            name: Context name, which the server requires. Defaults to
                ``resource_id``, which always matches the server's
                context-name pattern. A ``name`` of its own is needed when the
                id is longer than the 100-character name limit, or when the
                workspace already has a context of that name: the server then
                refuses with ``validation_error`` ("Context '<name>' already
                exists in this workspace.").
            summary: Deprecated and not sent (#273): the server's
                ``setup_resource`` has no summary (memory-cloud through
                v0.76.0), so it was always dropped. Passing it emits a
                :class:`DeprecationWarning`. Set it afterwards with
                :meth:`update_context` on the returned ``context_id`` (owner
                only).
            description: Token description.
            quota_events_per_hour: Token quota (1-10000).

        Returns:
            Server response dict with keys: ``context_id``, ``context_name``,
            ``resource_id``, ``token`` (plaintext, shown once), ``token_id``,
            ``warning``. Server may include additional fields (e.g. ``status``,
            ``message``) which callers can ignore.

        Raises:
            KaguraFeatureNotAvailableError: The plan lacks the ``resources``
                feature.
            KaguraQuotaError: The workspace's context cap or active-token
                cap is reached (``quota_type`` ``"contexts"`` /
                ``"resource_tokens"``).

        Note:
            Idempotency for repeated calls with the same ``resource_id`` is
            not guaranteed by this SDK; server-side behavior may evolve.

            Creating a resource — and with it a public context and a token —
            is plan-gated (``plan_required``). Since memory-cloud v0.68.0
            (#1551) the gate is the plan's ``resources`` feature (XL only by
            default); earlier servers gated it on plans with shared contexts
            and resource tokens. Existing resources keep serving.
        """
        if summary is not None:
            warnings.warn(_SETUP_SUMMARY_DEPRECATED, DeprecationWarning, stacklevel=2)
        arguments: dict[str, Any] = {
            "resource_id": resource_id,
            # Required by the server: without it every call was refused with
            # missing_fields (#273).
            "name": resource_id if name is None else name,
            "quota_events_per_hour": quota_events_per_hour,
        }
        if description is not None:
            arguments["description"] = description
        return await self._call_tool_checked("setup_resource", arguments)

    async def merge_contexts(
        self,
        source_id: str,
        target_id: str,
        delete_source: bool = False,
    ) -> dict[str, Any]:
        """
        Merge memories from one context into another.

        Copies every live memory from the source context into the target —
        copied, not moved: the source keeps its memories unless
        ``delete_source`` is set. Both contexts must use the same embedding
        model and belong to the same workspace, and the caller needs owner
        access to both.

        Args:
            source_id: Context ID to copy memories from.
            target_id: Context ID to copy memories into.
            delete_source: If True, soft-delete the source context after merge.
                Refused for the workspace's default context and for a locked
                context. Since memory-cloud v0.65.0 the server checks the
                default context before copying anything, and fails the merge
                (keeping the source) if not every live memory reached the
                target. Since v0.101.0 (#1960) the source's files then move
                to the target instead of staying behind, hidden, in the
                deleted source.

        Returns:
            API response with ``merged``, ``pending_embedding``,
            ``source_id`` / ``target_id`` (context UUIDs) and
            ``delete_source``. Since memory-cloud v0.65.0 ``merged`` counts
            memory **rows** copied, including rows not embedded yet;
            ``pending_embedding`` says how many of them are not searchable
            until the server embeds them in the target. Older servers counted
            only embedded memories and send no ``pending_embedding``.

        Raises:
            ValueError: If source_id and target_id are the same.
            KaguraError: Server-side rejection, e.g. ``delete_source`` on the
                default context.
        """
        if source_id == target_id:
            raise ValueError("source_id and target_id must be different")
        arguments: dict[str, Any] = {
            "source_context_id": source_id,
            "target_context_id": target_id,
        }
        if delete_source:
            arguments["delete_source"] = True
        return await self._call_tool_checked("merge_contexts", arguments)

    async def list_edges(
        self,
        context_id: str,
        memory_id: str,
        min_weight: float = 0.0,
        edge_types: list[str] | None = None,
        limit: int | None = None,
    ) -> list[Edge]:
        """List neural memory edges connected to a memory.

        Returns both outgoing and incoming edges, deduplicated, heaviest
        first.

        Since memory-cloud v0.82.0 (#1743) each direction is cut at ``limit``
        edges (50 unless set, at most 200). The tool takes no cursor: when a
        direction held more, its ``outgoing_has_more`` / ``incoming_has_more``
        flag is set, which this list cannot carry. When the cut came from the
        server's default (``limit=None``) or its 200 clamp, not from the
        ``limit`` passed, the client logs a warning on the ``kagura_memory``
        logger. Raise ``limit`` to get more of them, or raise ``min_weight`` /
        narrow ``edge_types`` so the ones that matter fit.

        Args:
            context_id: Context UUID containing the memory.
            memory_id: Memory UUID whose edges to list.
            min_weight: Minimum edge weight (0.0-3.0). Edges below this are
                filtered. Always sent: since memory-cloud v0.117.0 (#2127) the
                server's own default is 0.05, which hides weak automatic
                edges, while this method keeps sending ``0.0`` (every edge),
                as on every earlier server.
            edge_types: Restrict to these edge types (e.g. ``["related_to"]``). ``None``
                returns all types.
            limit: Maximum edges per direction (the server clamps it to
                1-200). **The server applies this to outgoing AND incoming
                queries independently**, so the practical maximum returned is
                ``2 * limit`` minus dedup overlap. ``None`` sends nothing: the
                server's default, 50 per direction since v0.82.0 (no limit
                before).

        Returns:
            List of :class:`Edge` instances ordered as the server returns them.

        Raises:
            KaguraNotFoundError: Context or memory not found.
            KaguraError: Other server-side error.
        """
        arguments: dict[str, Any] = {
            "context_id": context_id,
            "memory_id": memory_id,
            "min_weight": min_weight,
        }
        if edge_types is not None:
            arguments["edge_types"] = edge_types
        if limit is not None:
            arguments["limit"] = limit
        result = await self._call_tool_checked("list_edges", arguments)
        edges = parse_response_list(Edge, result.get("edges", []), operation="list_edges")
        cut = [d for d in ("outgoing", "incoming") if result.get(f"{d}_has_more") is True]
        # A cut at the caller's own limit is what they asked for: warn only when
        # the server's default or its clamp made it.
        if cut and (limit is None or limit > _LIST_EDGES_MAX_LIMIT):
            _logger.warning(
                "list_edges returned %d edges of memory %s, but more %s edges exist past "
                "the per-direction limit: raise limit (max 200), or narrow with "
                "min_weight / edge_types",
                len(edges),
                memory_id,
                " and ".join(cut),
            )
        return edges

    async def create_edge(
        self,
        context_id: str,
        source_id: str,
        target_id: str,
        edge_type: str = "related_to",
        weight: float = 0.5,
        confidence: float = 1.0,
    ) -> Edge:
        """Create or upsert a neural memory edge from ``source_id`` to ``target_id``.

        The server uses ``(user_id, source_id, target_id)`` as a unique key, so if an
        edge already exists for the same pair the server applies **max-weight UPSERT**
        semantics: the existing edge's weight is replaced only when the new weight is
        higher, and the previous ``edge_type`` may be overwritten. This is **not** a
        pure INSERT — callers expecting INSERT-or-fail semantics should call
        :meth:`list_edges` first.

        Args:
            context_id: Context UUID containing both endpoints.
            source_id: Source memory UUID.
            target_id: Target memory UUID. Must differ from ``source_id`` (self-loops
                are rejected client-side and server-side).
            edge_type: Edge type label. Server validates against its current
                ``VALID_EDGE_TYPES`` set; ``"related_to"`` is the standard default for
                manual links.
            weight: Edge weight in [0.0, 3.0]. Default 0.5 is a sensible mid-range
                value for manual edges.
            confidence: Edge confidence in [0.0, 1.0].

        Returns:
            The created :class:`Edge`.

        Raises:
            ValueError: If ``source_id == target_id``.
            KaguraNotFoundError: Context or memory not found.
            KaguraError: Server-side validation error (e.g. weight out of range,
                self-loop accepted past the client preflight, edge type rejected).
                Since memory-cloud v0.132.0 (#2313) ``edge_type="supersedes"``
                needs both memories to be the caller's own (a linked account's
                counts only in a private context), else ``permission_denied``:
                a shared-context editor cannot shadow another member's memory.
        """
        if source_id == target_id:
            raise ValueError(
                "source_id and target_id must be different (self-loops are not allowed)"
            )
        arguments: dict[str, Any] = {
            "context_id": context_id,
            "source_id": source_id,
            "target_id": target_id,
            "edge_type": edge_type,
            "weight": weight,
            "confidence": confidence,
        }
        result = await self._call_tool_checked("create_edge", arguments)
        return parse_response(Edge, result.get("edge", result), operation="create_edge")

    async def update_edge(
        self,
        context_id: str,
        source_id: str,
        target_id: str,
        weight: float | None = None,
        edge_type: str | None = None,
    ) -> Edge:
        """Update an existing edge's weight and/or edge type.

        The edge is identified by the ``(source_id, target_id)`` pair (the server's
        DB unique constraint covers ``(user_id, src, dst)``). Pass ``None`` for
        either ``weight`` or ``edge_type`` to leave that field unchanged.

        Args:
            context_id: Context UUID containing both endpoints.
            source_id: Source memory UUID.
            target_id: Target memory UUID.
            weight: New edge weight in [0.0, 3.0]. ``None`` keeps the existing value.
            edge_type: New edge type label. ``None`` keeps the existing value.

        Returns:
            The updated :class:`Edge`.

        Raises:
            KaguraNotFoundError: Context not found.
            KaguraError: Edge not found or other server-side error. Since
                memory-cloud v0.132.0 (#2313) an edge that is or becomes
                ``supersedes`` needs both memories to be the caller's own,
                else ``permission_denied``; retyping one away from
                ``supersedes`` is allowed.
        """
        arguments: dict[str, Any] = {
            "context_id": context_id,
            "source_id": source_id,
            "target_id": target_id,
        }
        if weight is not None:
            arguments["weight"] = weight
        if edge_type is not None:
            arguments["edge_type"] = edge_type
        result = await self._call_tool_checked("update_edge", arguments)
        return parse_response(Edge, result.get("edge", result), operation="update_edge")

    async def delete_edge(
        self,
        context_id: str,
        source_id: str,
        target_id: str,
    ) -> bool:
        """Delete the edge between ``source_id`` and ``target_id``.

        Args:
            context_id: Context UUID containing both endpoints.
            source_id: Source memory UUID.
            target_id: Target memory UUID.

        Returns:
            ``True`` once the server confirms deletion succeeded.

        Raises:
            KaguraNotFoundError: Context not found.
            KaguraError: Edge not found or other server-side error.
        """
        arguments: dict[str, Any] = {
            "context_id": context_id,
            "source_id": source_id,
            "target_id": target_id,
        }
        result = await self._call_tool_checked("delete_edge", arguments)
        # The server confirms a delete with {"status": "success"} and NO
        # "deleted" key; a missing edge comes back as a status=="error" response
        # that _call_tool_checked already raised above. So reaching here means
        # deletion was confirmed — the default True is load-bearing and must not
        # be "fixed" to False (that would report failure on every real delete).
        # The .get() still honors an explicit "deleted" flag should the server
        # contract ever add one. (Verified against memory-cloud edge.py.)
        return bool(result.get("deleted", True))

    async def get_usage(self) -> UsageInfo:
        """Get workspace usage and quota limits.

        Since memory-cloud v0.100.0 (#1946) ``memories.used`` counts the live
        rows of this workspace, the figure the memory quota and the capacity
        lock read; the quota gate counted across workspaces before.

        Returns:
            UsageInfo with plan, memories, contexts, members, and MCP call limits.
        """
        result = await self._call_tool_checked("get_usage", {})
        return parse_response(UsageInfo, result, operation="get_usage")

    async def get_context_info(
        self,
        context_id: str,
        include_details: bool = True,
    ) -> ContextInfo:
        """Get context information, usage guidelines, and search config.

        Args:
            context_id: Context UUID.
            include_details: Include memory count breakdown (default: True).

        Returns:
            ContextInfo with context metadata, search_config, stats, and
            instructions — plus, on memory-cloud v0.74.0+, the ``guardrails``
            block (the context's tool guardrails for hookless clients; see
            :class:`~kagura_memory.models.ContextInfo` for its three states).
        """
        arguments: dict[str, Any] = {
            "context_id": context_id,
            "include_details": include_details,
        }
        result = await self._call_tool_checked("get_context_info", arguments)
        return parse_response(ContextInfo, result, operation="get_context_info")

    async def _get_context_info_cached(self, context_id: str) -> ContextInfo | None:
        """Best-effort, cached :meth:`get_context_info` for ingest steering.

        Internal: the only caller is the ingest pipeline (see
        :meth:`FileIngestor._resolve_steering`). It lives on the client — not
        the ingestor — so the fetch-once cache is keyed per
        ``(client, context_id)`` and shared across ingestors that reuse one
        client, but it is deliberately kept off the public API surface.

        After the first successful (or failed) fetch the result is cached for
        this client's lifetime — including ``None`` on failure — so repeated
        callers (e.g. per-section ingest summarization, which is the hot path
        this exists for) never re-fetch. On any error the failure is swallowed
        and ``None`` is cached and returned: steering is purely additive and
        best-effort, so a missing or unreachable context must never crash
        ingestion.

        Within a single ingest the fetch happens exactly once: the ingestor
        resolves steering before fanning out the per-section calls. The only
        case that can fetch more than once is two *concurrent* ingests on the
        same client racing on the same not-yet-cached ``context_id`` (both miss
        the cache before either stores) — that simply repeats an idempotent
        read and converges, so no lock is used.

        Note: the cache is not invalidated mid-session. A concurrent
        ``update_context`` is not reflected until a fresh client is created
        (an intentional v1 design seam).

        Args:
            context_id: Context UUID.

        Returns:
            The :class:`ContextInfo`, or ``None`` if it could not be fetched.
        """
        if context_id in self._context_info_cache:
            return self._context_info_cache[context_id]
        try:
            info: ContextInfo | None = await self.get_context_info(context_id)
        except Exception as e:  # noqa: BLE001
            # Best-effort contract: ingest must never crash because steering
            # could not be fetched. Catch broadly — KaguraError (including the
            # KaguraResponseError a malformed/changed server payload raises)
            # and anything unexpected — log a warning, and cache None so we
            # degrade to steering=None without re-fetching. (python.md
            # permits a broad catch that logs.)
            _logger.warning(
                "get_context_info failed for context %s; ingest steering disabled: %s",
                context_id,
                e,
            )
            info = None
        self._context_info_cache[context_id] = info
        return info

    async def update_search_config(
        self,
        context_id: str,
        semantic_weight: float | None = None,
        bm25_weight: float | None = None,
        fetch_factor: int | None = None,
        use_rerank: bool | None = None,
        reranker_provider: str | None = None,
        reranker_model: str | None = None,
        *,
        content_chunks_enabled: bool | None = None,
        llm_selector_enabled: bool | None = None,
    ) -> dict[str, Any]:
        """Update hybrid search configuration for a context.

        Weights must sum to 1.0 (±0.01). The weights, ``fetch_factor`` and
        the reranker settings need context editor or above;
        ``content_chunks_enabled`` and ``llm_selector_enabled`` need the
        context owner or a workspace owner/admin (an editor gets
        ``permission_denied`` with ``details["required_role"] == "owner"``).
        New contexts start from the deployment's defaults, which
        :meth:`get_server_info` reports as ``search_defaults`` (memory-cloud
        v0.69.0+).

        Args:
            context_id: Context UUID.
            semantic_weight: Semantic search weight (0.0-1.0, server default 0.6).
            bm25_weight: BM25 keyword search weight (0.0-1.0, server default 0.4).
            fetch_factor: Candidate fetch multiplier (1-10, server default 3).
            use_rerank: Enable AI reranking. Since server v0.69.0 this is also
                what a :meth:`recall` that omits ``use_rerank`` follows.
            reranker_provider: Reranker provider: ``"voyage"`` or
                ``"cohere"`` (each needs its provider's API key), or
                ``"self_hosted"`` (memory-cloud v0.42.0+) — the deployment's
                keyless local reranker, an OpenAI-compatible backend such as
                Ollama or vLLM.
            reranker_model: Reranker model name. May be omitted for
                ``"self_hosted"`` (memory-cloud v0.69.0+).
            content_chunks_enabled: Embed long ``content`` (1,500+
                characters) as chunk vectors, so dense retrieval sees the
                body, not only the summary (memory-cloud v0.110.0+, #2015).
                **On, memory content — not only summaries — is sent to the
                embedding and rerank providers.** Turning it on queues the
                context's existing long memories for chunking (each costs
                embeddings; v0.112.0+), turning it off deletes the chunks, and
                sending the stored value is a no-op. Turning it on is refused
                on a LanceDB deployment (``invalid_search_config``, v0.112.0+,
                #2072) and, where the deployment gates it by plan
                (``CONTENT_CHUNKS_REQUIRE_PLAN``, v0.113.0+, #2103), without
                the plan's ``content_chunks`` feature or past its
                ``max_content_chunk_contexts``. Sent when not ``None`` —
                ``False`` turns it off. Keyword-only.
            llm_selector_enabled: Let an MCP :meth:`recall` send the
                context's summaries and attached content to the deployment's
                judge model, which keeps the results that answer the query and
                cuts their content to verbatim passages (``selector`` /
                ``spans`` in the reply; memory-cloud v0.115.0+, #2144). Same
                owner rule; where the deployment gates it by plan
                (``LLM_SELECTOR_REQUIRE_PLAN``), turning it on needs the
                ``llm_selector`` feature and a free slot under
                ``max_llm_selector_contexts``. Sent when not ``None``.
                Keyword-only.

        Returns:
            Current search config after update, under ``config``. Besides the
            arguments it carries ``reinforce_*`` and ``routing_mode``, and the
            stored ``content_chunks_enabled`` (v0.110.0+) and
            ``llm_selector_enabled`` (v0.115.0+) each with its ``*_effective``
            — whether the stored flag is acted on (store, plan and cap). Read
            the stored flag, never the effective one, before writing it back:
            echoing an effective ``False`` would turn the setting off.

        Raises:
            KaguraFeatureNotAvailableError: The plan lacks ``content_chunks`` /
                ``llm_selector`` (where the deployment gates it by plan).
            KaguraQuotaError: The workspace is at its cap — ``quota_type``
                ``content_chunk_contexts`` or ``llm_selector_contexts``.
            KaguraError: ``code == "permission_denied"`` (an editor changing an
                owner-only flag, or a viewer), ``"invalid_search_config"`` (the
                weights do not sum to 1.0, a provider / model mismatch, or
                content chunks on LanceDB; the REST twin answers ``422
                VAL-001``), or ``"context_not_found"``.
        """
        arguments: dict[str, Any] = {"context_id": context_id}
        if semantic_weight is not None:
            arguments["semantic_weight"] = semantic_weight
        if bm25_weight is not None:
            arguments["bm25_weight"] = bm25_weight
        if fetch_factor is not None:
            arguments["fetch_factor"] = fetch_factor
        if use_rerank is not None:
            arguments["use_rerank"] = use_rerank
        if reranker_provider is not None:
            arguments["reranker_provider"] = reranker_provider
        if reranker_model is not None:
            arguments["reranker_model"] = reranker_model
        if content_chunks_enabled is not None:
            arguments["content_chunks_enabled"] = content_chunks_enabled
        if llm_selector_enabled is not None:
            arguments["llm_selector_enabled"] = llm_selector_enabled
        return await self._call_tool_checked("update_search_config", arguments)

    async def get_server_info(self) -> ServerInfo:
        """Get server name, version, environment, and feature flags.

        Calls ``GET /api/v1/system/info``.

        Returns:
            ServerInfo with the version string, the deployment's feature
            flags (:class:`~kagura_memory.models.ServerFeatures` — flags newer
            than this SDK land in ``features.model_extra``),
            ``search_defaults`` (memory-cloud v0.69.0+) — the reranker
            settings new contexts start with — and ``terms_version``
            (v0.77.0+): the terms-of-service version, ``None`` when the
            deployment does not record acceptance. Accepting the terms is a
            web sign-in step that never gates API or MCP calls.

        Raises:
            KaguraAuthError: HTTP 401.
            KaguraRateLimitError: HTTP 429.
            KaguraConnectionError: Network failure, non-JSON body or any
                other non-2xx status.
            KaguraResponseError: The 2xx body does not match
                :class:`ServerInfo`; ``operation`` is
                ``"KaguraClient.get_server_info"`` (#277 — SDKs before
                0.42.0 raised ``KaguraConnectionError`` here).
        """
        return await self._rest_get(
            "/api/v1/system/info", ServerInfo, operation="KaguraClient.get_server_info"
        )

    async def check_server_version(self) -> ServerInfo:
        """Check the connected server's version against the SDK's tested minimum.

        Advisory only — calls ``get_server_info()`` and logs a warning
        via :mod:`logging` when the server version is below
        :data:`MIN_SERVER_VERSION`. An old version does not raise. Older
        servers may silently ignore unknown parameters.

        A ``v`` prefix, build metadata and pre-release suffixes are read,
        so ``"v0.16.0"`` and ``"0.17.1-rc1"`` (a pre-release of the minimum)
        both warn. A version with no ``MAJOR.MINOR.PATCH`` at its start, such
        as ``"0.17"`` or ``"main-abc123"``, cannot be compared and does not
        warn; ``kagura doctor`` reports it as ``info``.

        Returns:
            ServerInfo from the server.

        Raises:
            KaguraAuthError: HTTP 401.
            KaguraRateLimitError: HTTP 429.
            KaguraConnectionError: Network failure, non-JSON body or any
                other non-2xx status.
            KaguraResponseError: The 2xx body does not match
                :class:`ServerInfo`; ``operation`` is
                ``"KaguraClient.get_server_info"``, the call this makes
                (#277 — SDKs before 0.42.0 raised ``KaguraConnectionError``
                here, which ``kagura doctor`` reported as unreachable).
        """
        info = await self.get_server_info()
        if meets_minimum(info.version, _MIN_SERVER_VERSION_TUPLE) is False:
            _logger.warning(
                "Server version %s is below the SDK's tested minimum %s. "
                "Some features may not work; older servers may silently "
                "ignore unknown parameters.",
                info.version,
                MIN_SERVER_VERSION,
            )
        return info

    async def get_embedding_status(self) -> EmbeddingStatus:
        """Get embedding queue status for the workspace.

        Calls ``GET /api/v1/workspace/embedding-status``. Since memory-cloud
        v0.65.0 the counts and ``failed_memories`` cover only the contexts
        the caller can see: the workspace owner sees every context, other
        members see shared contexts and the private ones they created.

        Returns:
            EmbeddingStatus with total, by_status breakdown, and failed memories.

        Raises:
            KaguraAuthError: HTTP 401.
            KaguraRateLimitError: HTTP 429.
            KaguraConnectionError: Network failure, non-JSON body or any
                other non-2xx status.
            KaguraResponseError: The 2xx body does not match
                :class:`EmbeddingStatus`; ``operation`` is
                ``"KaguraClient.get_embedding_status"`` (#277 — SDKs before
                0.42.0 raised ``KaguraConnectionError`` here).
        """
        return await self._rest_get(
            "/api/v1/workspace/embedding-status",
            EmbeddingStatus,
            operation="KaguraClient.get_embedding_status",
        )

    async def get_memory_stats(
        self,
        context_id: str,
        sort_by: str = "use_count",
        sort_order: Literal["asc", "desc"] = "desc",
        limit: int = 50,
        offset: int = 0,
    ) -> MemoryStatsResponse:
        """Get per-memory usage statistics for a context.

        Calls ``GET /api/v1/contexts/{context_id}/memory-stats``.

        Args:
            context_id: Context UUID.
            sort_by: Sort field (default: "use_count").
            sort_order: Sort order — "asc" or "desc" (default: "desc").
            limit: Maximum results (1-200, default: 50).
            offset: Pagination offset (default: 0).

        Returns:
            MemoryStatsResponse with per-memory stats and pagination info.

        Raises:
            KaguraAuthError: HTTP 401.
            KaguraRateLimitError: HTTP 429.
            KaguraConnectionError: Network failure, non-JSON body or any
                other non-2xx status (an unknown or inaccessible
                ``context_id`` is an HTTP 404).
            KaguraResponseError: The 2xx body does not match
                :class:`MemoryStatsResponse`; ``operation`` is
                ``"KaguraClient.get_memory_stats"`` (#277 — SDKs before
                0.42.0 raised ``KaguraConnectionError`` here).
        """
        params = {
            "sort_by": sort_by,
            "sort_order": sort_order,
            "limit": limit,
            "offset": offset,
        }
        return await self._rest_get(
            f"/api/v1/contexts/{context_id}/memory-stats",
            MemoryStatsResponse,
            params=params,
            operation="KaguraClient.get_memory_stats",
        )

    async def find_duplicates(
        self,
        context_id: str,
        threshold: float = 0.90,
        limit: int = 50,
    ) -> DuplicatesResponse:
        """Find duplicate memory pairs in a context.

        Calls ``GET /api/v1/contexts/{context_id}/duplicates``.

        Args:
            context_id: Context UUID.
            threshold: Similarity threshold (0.5-1.0, default: 0.90).
            limit: Maximum pairs (1-200, default: 50).

        Returns:
            DuplicatesResponse with duplicate pairs and similarity scores.

        Raises:
            KaguraAuthError: HTTP 401.
            KaguraRateLimitError: HTTP 429.
            KaguraConnectionError: Network failure, non-JSON body or any
                other non-2xx status (an unknown or inaccessible
                ``context_id`` is an HTTP 404).
            KaguraResponseError: The 2xx body does not match
                :class:`DuplicatesResponse`; ``operation`` is
                ``"KaguraClient.find_duplicates"`` (#277 — SDKs before
                0.42.0 raised ``KaguraConnectionError`` here).
        """
        params = {"threshold": threshold, "limit": limit}
        return await self._rest_get(
            f"/api/v1/contexts/{context_id}/duplicates",
            DuplicatesResponse,
            params=params,
            operation="KaguraClient.find_duplicates",
        )

    async def list_memories(
        self,
        context_id: str | None = None,
        q: str | None = None,
        scope: Literal["working", "persistent"] | None = None,
        type: str | None = None,
        limit: int = 50,
        offset: int = 0,
        trigger_from: str | None = None,
        trigger_until: str | None = None,
        order_by: Literal["created_at", "trigger_from"] | None = None,
        *,
        lat_min: float | None = None,
        lat_max: float | None = None,
        lon_min: float | None = None,
        lon_max: float | None = None,
    ) -> MemoryListResponse:
        """List memories with optional substring, facet, time-window and bbox filters.

        Ordering is newest-first by default; pass ``order_by="trigger_from"`` to
        sort Time Memories soonest-scheduled first.

        Calls ``GET /api/v1/memory/list``. Without ``context_id`` this returns
        the caller's own memories across all contexts; with ``context_id`` it
        returns every memory in a shared context, or only the caller's own in a
        private one (the server enforces this scoping).

        Args:
            context_id: Optional context UUID to scope results to one context.
                Omit for the caller's cross-context "my memories" view.
            q: Optional case-insensitive substring filter on memory summaries.
                Surrounding whitespace is stripped and whitespace-only values
                are treated as ``None`` (no filter), mirroring the server and
                avoiding a wasted request. Matching targets ``summary`` only —
                ``content`` and ``context_summary`` are deliberately not
                searched (memory-cloud #580); use :meth:`recall` for semantic
                or full-text search.
            scope: Filter by scope — ``"working"`` or ``"persistent"``.
            type: Filter by memory type. Server validates against its own
                vocabulary; the SDK passes through.
            limit: Maximum results (server accepts 1-500, default: 50).
            offset: Pagination offset (default: 0).
            trigger_from: Lower bound (naive ISO) of a time window. Restricts
                results to ``type="time"`` memories whose trigger window
                **overlaps** ``[trigger_from, trigger_until]`` — i.e. a memory
                may start before ``trigger_from`` and still match as long as its
                window overlaps the range (not a "starts at or after" filter).
                Omit for no lower bound.
            trigger_until: Upper bound (naive ISO) of the time window (same
                overlap semantics as ``trigger_from``). Omit for an open-ended
                window. For a "what's upcoming" view, prefer
                :meth:`recall_upcoming`, which is purpose-built for that query.
            order_by: Sort order — ``"created_at"`` (default, newest-first) or
                ``"trigger_from"`` (soonest scheduled first). Omit to use the
                server default.
            lat_min: WHERE-axis bounding box — lower latitude bound in degrees
                (-90 to 90). Keyword-only, like the other three bounds. Bounds
                may be one-sided, and **any** bound restricts results to
                memories with a complete ``details.location``. Requires
                memory-cloud server v0.54.0+ (#1334); older servers ignore the
                bbox silently and return an unfiltered page.
            lat_max: Upper latitude bound (-90 to 90).
            lon_min: Lower longitude bound (-180 to 180). ``lon_min > lon_max``
                selects the antimeridian-crossing box (``lon >= lon_min OR
                lon <= lon_max``) rather than an empty one.
            lon_max: Upper longitude bound (-180 to 180).

        Returns:
            :class:`MemoryListResponse` with ``memories`` (ordered per
            ``order_by``; newest-first by default; each carrying ``location``
            when it has one, and ``observed_at`` — when the fact held — on
            memory-cloud v0.127.0+, #2330), ``total`` (matching rows across
            all pages), and ``has_more``. This REST route has no
            ``observed_at`` filter or sort (memory-cloud v0.142.0 accepts
            ``order_by`` ``created_at`` / ``trigger_from`` only): filter with
            :meth:`recall`'s ``observed_after`` / ``observed_before``, or the
            MCP ``list`` tool, which this client does not wrap.

        Raises:
            ValueError: If a bbox bound is non-numeric or out of range — checked
                locally, as :meth:`recall_nearby` checks its point, since the
                server would only 422 it.
            KaguraAuthError: Authentication failed.
            KaguraConnectionError: Network failure, non-JSON body or non-2xx
                response — e.g. a ``context_id`` that does not exist or is not
                accessible surfaces as ``HTTP 404``.
            KaguraResponseError: The 2xx body does not match
                :class:`MemoryListResponse` (a server newer than the SDK), with
                ``operation="KaguraClient.list_memories"``. SDKs before 0.40.0
                raised ``KaguraConnectionError`` here.
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if context_id is not None:
            params["context_id"] = context_id
        # Normalize like the server/frontend: strip and drop whitespace-only so
        # an empty search box doesn't pin results to summaries containing spaces.
        q_normalized = (q or "").strip()
        if q_normalized:
            params["q"] = q_normalized
        if scope is not None:
            params["scope"] = scope
        if type is not None:
            params["type"] = type
        if trigger_from is not None:
            params["trigger_from"] = trigger_from
        if trigger_until is not None:
            params["trigger_until"] = trigger_until
        if order_by is not None:
            params["order_by"] = order_by
        # `is not None`, not truthiness: 0.0 (equator / prime meridian) is a bound.
        # Each bound is range-checked on its own; the pair ordering is not, since
        # lon_min > lon_max is the valid antimeridian box.
        for key, bound, max_abs in (
            ("lat_min", lat_min, 90),
            ("lat_max", lat_max, 90),
            ("lon_min", lon_min, 180),
            ("lon_max", lon_max, 180),
        ):
            if bound is not None:
                validate_coordinate(key, bound, max_abs)
                params[key] = bound
        return await self._rest_get(
            "/api/v1/memory/list",
            MemoryListResponse,
            params=params,
            operation="KaguraClient.list_memories",
        )

    @staticmethod
    def _raise_for_mcp_error(result: dict[str, Any], operation: str) -> None:
        """Translate an MCP tool's structured error response to an SDK exception.

        The server's MCP tools return ``{"status": "error", "error": <code>,
        "message": <str>, ...}`` for domain errors that the JSON-RPC transport
        layer cannot represent (e.g. ``report_not_found``). HTTP-level
        errors (401, 5xx) are already handled by ``_make_jsonrpc_request``.

        A plan or quota refusal raises :class:`KaguraQuotaError` or
        :class:`KaguraFeatureNotAvailableError` carrying the envelope's gate
        fields, the Free capacity lock (``capacity_locked``, memory-cloud
        v0.99.0+) :class:`KaguraCapacityLockedError`, and
        ``partial_rollback`` raises :class:`KaguraPartialRollbackError`
        (#256). ``version_conflict`` (``update_memory`` with a stale
        ``expected_version``, v0.113.0+) raises
        :class:`KaguraVersionConflictError` (#302). Every other code raises
        :class:`KaguraError` — e.g. ``payload_too_large`` (a memory over
        1 MB, v0.108.0+, with ``limit`` / ``actual``).

        Every error keeps the envelope: ``code`` is its ``error`` and
        ``details`` everything but ``status`` / ``error`` / ``message`` —
        ``help`` (v0.78.0+) and, on a server failure, ``cause``,
        ``correlation_id``, ``retryable`` and for a write ``outcome``
        (v0.82.0+), which :class:`KaguraError` exposes as properties.
        """
        if result.get("status") != "error":
            return
        code = result.get("error", "unknown")
        message = result.get("message", "Unknown error")
        text = f"{operation} failed ({code}): {message}"
        err: KaguraError
        drift: KaguraResponseError | None = None
        if code in (
            "report_not_found",
            "context_not_found",
            "memory_not_found",
            "agent_not_found",
            "binding_not_found",
            "session_not_found",
        ):
            err = KaguraNotFoundError(f"{operation}: {message}")
        elif code == "partial_rollback":
            report_id = result.get("report_id")
            report_id = report_id if isinstance(report_id, str) else None
            try:
                summary = parse_response(
                    RollbackSummary, result.get("rollback_summary"), operation=operation
                )
                err = KaguraPartialRollbackError(text, report_id=report_id, summary=summary)
            except KaguraResponseError as e:
                # The partial reversal is already committed; a summary the
                # SDK cannot read must not hide that (#256).
                drift = e
                err = KaguraPartialRollbackError(
                    f"{text} (rollback_summary could not be read)", report_id=report_id
                )
        else:
            err = (
                version_conflict_error(str(code), text, result)
                or gate_error(str(code), text, result)
                or KaguraError(text)
            )
        annotate_server_error(err, str(code), result)
        if drift is not None:
            raise err from drift
        raise err

    async def get_sleep_history(
        self,
        context_id: str,
        limit: int = 10,
        *,
        source: Literal["sleep", "analysis", "all"] | None = None,
    ) -> list[SleepReport]:
        """List recent Sleep Maintenance runs for a context.

        Each run's ``status`` is one of :data:`SleepRunStatus` —
        ``running``, ``completed``, ``degraded``, ``failed``, ``cancelled``
        or ``rolled_back`` — or a newer value passed through as-is.
        ``degraded`` means the run finished, but some judge-LLM calls
        failed (server v0.43.0+; ``llm_call_failures`` gives the count) or,
        since server v0.46.0, a phase failed; its changes were applied and
        can be rolled back. ``failed`` means the run errored — every
        judge-LLM call failed or the run raised — or a later rollback of it
        only partly succeeded. These summaries do not carry the reason:
        :meth:`get_sleep_report` returns it as ``error_message``
        (e.g. ``phase_failure: <phases>`` on a degraded run).

        Since memory-cloud v0.138.0 (#2219) the tool lists Sleep runs only
        (``source="sleep"``) unless ``source`` says otherwise, and each row
        carries ``source``. Older servers have no filter, may list the rows
        Memory Analysis writes for its LLM spend among the Sleep runs, and
        send no ``source``.

        Args:
            context_id: Context UUID.
            limit: Maximum number of runs to return (server clamps to 1-50,
                default: 10).
            source: ``"sleep"`` (Sleep runs, the server's default),
                ``"analysis"`` (Memory Analysis spend rows) or ``"all"``.
                Sent only when set (memory-cloud v0.138.0+). Keyword-only.

        Returns:
            List of ``SleepReport`` summaries ordered by ``started_at``
            descending (newest first).

        Raises:
            KaguraNotFoundError: Context not found.
            KaguraResponseError: A run did not match the SDK's model.
            KaguraError: Other server-side error.
        """
        arguments: dict[str, Any] = {"context_id": context_id, "limit": limit}
        if source is not None:
            arguments["source"] = source
        result = await self._call_tool_checked("get_sleep_history", arguments)
        return parse_response_list(
            SleepReport, result.get("reports"), operation="get_sleep_history"
        )

    async def get_sleep_report(
        self,
        context_id: str,
        report_id: str,
        *,
        actions_cursor: str | None = None,
        actions_limit: int | None = None,
        max_chars: int | None = None,
    ) -> SleepReportDetail:
        """Get a detailed Sleep Maintenance report including audit log.

        ``context_id`` is never sent: besides ``report_id``, only the paging
        arguments below go out, each only when set. No memory-cloud release
        declares or reads a ``context_id`` on this tool, and since v0.82.0
        (memory-cloud#1742) the server refuses an argument the tool's schema
        does not declare, so SDKs up to 0.42.1, which sent one, failed every
        call with ``invalid_argument`` (#296). Since memory-cloud v0.126.0
        (memory-cloud#2315) the server authorizes the call against the
        report's own context: read access there is enough.

        Since memory-cloud v0.82.0 (#1743) ``actions`` is one page of the
        run's actions in id order, 50 by default, within a 20,000-character
        budget: ``action_count`` is the run's total, and while
        ``actions_has_more`` is ``True`` pass ``actions_next_cursor`` back as
        ``actions_cursor`` for the next page. An action too large for its
        page comes with ``details_omitted`` instead of ``details``. Older
        servers send every action at once. An analysis report
        (``source="analysis"``) carries a ``note`` (v0.140.0+, #2547): it
        records Memory Analysis spend and has no Sleep actions.

        Args:
            context_id: Unused; kept for backwards compatibility. The server
                scopes the call by ``report_id``.
            report_id: Sleep report UUID.
            actions_cursor: The ``actions_next_cursor`` of the previous page.
                Keyword-only, like the two below; all three need memory-cloud
                v0.82.0+.
            actions_limit: Actions per page (the server clamps it to 1-200,
                default 50).
            max_chars: Response budget in characters, not tokens (10,000 to
                100,000, default 20,000; the server refuses other values with
                ``validation_error``).

        Returns:
            ``SleepReportDetail`` with per-phase results and one page of the
            per-action audit log.

        Raises:
            KaguraNotFoundError: Report not found or not owned by caller, or
                (memory-cloud v0.126.0+) in a context the caller cannot
                read: the server answers ``report_not_found`` for each.
            KaguraResponseError: The report did not match the SDK's model.
            KaguraError: Other server-side error.
        """
        arguments: dict[str, Any] = {"report_id": report_id}
        if actions_cursor is not None:
            arguments["actions_cursor"] = actions_cursor
        if actions_limit is not None:
            arguments["actions_limit"] = actions_limit
        if max_chars is not None:
            arguments["max_chars"] = max_chars
        result = await self._call_tool_checked("get_sleep_report", arguments)
        # The MCP tool wraps the report fields under a "report" key;
        # flatten so SleepReportDetail (a SleepReport subclass) validates
        # naturally without forcing callers through an extra ``.report.``
        # accessor. A missing or non-object "report" is passed through
        # unflattened so it fails validation like any other drift.
        report = result.get("report")
        if isinstance(report, dict):
            report = {
                **report,
                **{key: result[key] for key in _SLEEP_REPORT_ENVELOPE_KEYS if key in result},
            }
        return parse_response(SleepReportDetail, report, operation="get_sleep_report")

    async def rollback_sleep_run(
        self,
        context_id: str,
        report_id: str,
    ) -> RollbackResult:
        """Reverse the effects of a completed (or degraded) Sleep Maintenance run.

        Reverses edge creation, memory merges, importance updates, scope
        promotions, and archives. Only ``completed`` and ``degraded`` runs
        can be rolled back. The server processes actions in reverse
        order with per-step commits — a 5xx or partial failure means SOME
        actions may have been reversed before the error surfaced.

        Only ``report_id`` is sent, as for :meth:`get_sleep_report` (#296).
        Since memory-cloud v0.126.0 (memory-cloud#2315) the server
        authorizes the call against the report's own context: it needs the
        editor or owner role there, and a workspace viewer cannot roll back.

        Args:
            context_id: Unused; kept for backwards compatibility. The server
                scopes the call by ``report_id``.
            report_id: Sleep report UUID.

        Returns:
            ``RollbackResult`` with per-category counts of reversed actions.
            Since memory-cloud v0.112.0 ``chunks_requeued`` /
            ``points_requeued`` count restored memories whose chunks / vector
            were queued for a rebuild, and since v0.153.0
            ``related_indexes_reverted`` the related-index entries undone
            (see :class:`RollbackSummary`).

        Raises:
            KaguraNotFoundError: Report not found or not owned by caller, or
                (memory-cloud v0.126.0+) in a context the caller cannot
                read: the server answers ``report_not_found`` for each.
            KaguraPartialRollbackError: Some undo steps failed. The rest
                stay reversed; ``.summary`` counts them and lists the
                failures in ``.summary.errors``.
            KaguraError: Other server-side error. The exception message
                includes the server-side error code for triage, e.g.
                ``permission_denied`` when the caller can read the report's
                context but not edit it (v0.126.0+), or ``analysis_report``
                for a Memory Analysis report (``source="analysis"``), which
                records LLM spend and has no Sleep actions to reverse
                (v0.140.0+, memory-cloud#2547).
        """
        result = await self._call_tool_checked("rollback_sleep_run", {"report_id": report_id})
        return parse_response(RollbackResult, result, operation="rollback_sleep_run")

    async def list_embedding_models(self) -> EmbeddingModelsResponse:
        """List available embedding models.

        Calls ``GET /api/v1/system/embedding/models`` to retrieve
        server-supported embedding models with provider info and availability.
        Since memory-cloud v0.66.0 the list holds only the models the
        deployment's allowlist offers — the ones :meth:`create_context`
        accepts.

        Returns:
            EmbeddingModelsResponse with models list and default_model.

        Raises:
            KaguraAuthError: HTTP 401.
            KaguraRateLimitError: HTTP 429.
            KaguraConnectionError: Network failure, non-JSON body or any
                other non-2xx status.
            KaguraResponseError: The 2xx body does not match
                :class:`EmbeddingModelsResponse`; ``operation`` is
                ``"KaguraClient.list_embedding_models"`` (#277 — SDKs before
                0.42.0 raised ``KaguraConnectionError`` here).
        """
        return await self._rest_get(
            "/api/v1/system/embedding/models",
            EmbeddingModelsResponse,
            operation="KaguraClient.list_embedding_models",
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> "KaguraClient":
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: Any,
    ) -> None:
        """Async context manager exit."""
        await self.close()


# ---------------------------------------------------------------------------
# memory-cloud v0.82.0 bounds (#1743): write caps and cursor paging (#299)
# ---------------------------------------------------------------------------

# remember / update_memory tag caps (server ``MEMORY_TAGS_MAX_COUNT`` /
# ``MEMORY_TAG_MAX_CHARS``), enforced on new writes only.
_MEMORY_TAGS_MAX_COUNT = 50
_MEMORY_TAG_MAX_CHARS = 100
# set_state value cap (server ``STATE_VALUE_MAX_CHARS``), in characters of
# compact JSON.
_STATE_VALUE_MAX_CHARS = 16_384
# Pages a plain-list method follows before giving up on an unfinished list.
_MAX_AUTO_PAGES = 50
# list_edges per-direction clamp (server ``LIST_EDGES_MAX_LIMIT``).
_LIST_EDGES_MAX_LIMIT = 200
# list_contexts page sizes (server ``_LIST_CONTEXTS_MAX_LIMIT``, and
# ``_LIST_CONTEXTS_DETAILS_MAX_LIMIT`` for include_details without
# name_contains). Out of range the server clamps the first and refuses the second.
_LIST_CONTEXTS_MAX_LIMIT = 200
_LIST_CONTEXTS_DETAILS_MAX_LIMIT = 20
# get_state list-mode page size (server ``STATE_LIST_MAX_LIMIT``, clamped there).
_STATE_LIST_MAX_LIMIT = 200
# get_sleep_report envelope keys merged into the flattened report.
_SLEEP_REPORT_ENVELOPE_KEYS = (
    "actions",
    "action_count",
    "actions_has_more",
    "actions_next_cursor",
    "note",
)


def _validate_memory_tags(tags: list[str] | None, name: str = "tags") -> None:
    """Reject a ``tags`` list the server would refuse on a write.

    Measured as the server measures it: the list as given (no trimming or
    deduplication) and each tag's length in characters. ``update_memory``'s
    ``add_tags`` takes the same caps (memory-cloud v0.140.0, #2585).

    Args:
        tags: The list, or ``None`` (nothing to check).
        name: The argument's name, for the message.

    Raises:
        ValueError: More than 50 tags, or a tag longer than 100 characters.
    """
    if tags is None:
        return
    if len(tags) > _MEMORY_TAGS_MAX_COUNT:
        raise ValueError(f"{name} accepts at most {_MEMORY_TAGS_MAX_COUNT} tags, got {len(tags)}")
    longest = max((len(tag) for tag in tags if isinstance(tag, str)), default=0)
    if longest > _MEMORY_TAG_MAX_CHARS:
        raise ValueError(
            f"each tag must be at most {_MEMORY_TAG_MAX_CHARS} characters, got {longest}"
        )


def _validate_state_value(value: Any) -> None:
    """Reject a ``set_state`` value longer than the server's cap.

    The length is the server's measure: compact JSON (``ensure_ascii=False``,
    no spaces), with a value JSON cannot encode measured as its ``str``, so
    the check never raises for a reason of its own.

    Raises:
        ValueError: The value is longer than 16,384 characters as compact JSON.
    """
    # Mirrors the server's ``utils.response_budget.json_chars`` exactly.
    size = len(json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str))
    if size > _STATE_VALUE_MAX_CHARS:
        raise ValueError(
            f"value must be at most {_STATE_VALUE_MAX_CHARS} characters as compact JSON, "
            f"got {size}; store large data as a memory and keep its id in the state"
        )


def _normalize_name_contains(name_contains: Any) -> Any:
    """``list_contexts``' ``name_contains`` as the server reads it.

    A string is trimmed and a blank one is no filter (``None``). Any other
    value is returned as is, for the server to refuse.
    """
    if isinstance(name_contains, str):
        return name_contains.strip() or None
    return name_contains


def _details_page_over_cap(include_details: bool, name_contains: Any, limit: int | None) -> bool:
    """Whether memory-cloud refuses this ``list_contexts`` page (0.82.0+, #1743).

    With ``include_details`` and no filter (a non-blank string
    ``name_contains``) a page holds at most 20 contexts, and a larger
    ``limit`` is refused with ``validation_error``, not clamped.
    """
    filtered = isinstance(_normalize_name_contains(name_contains), str)
    return (
        bool(include_details)
        and not filtered
        and limit is not None
        and limit > _LIST_CONTEXTS_DETAILS_MAX_LIMIT
    )


async def _collect_pages(client: KaguraClient, tool: str, model: type[_T], key: str) -> list[_T]:
    """Every row of a cursor-paged MCP list tool (memory-cloud v0.82.0+, #1743).

    The first call sends no paging argument, so an older server, which pages
    nothing and sends no ``has_more``, answers in one call. Each later call
    sends only ``cursor``, the ``next_cursor`` of the page before.

    The cursor is an offset, so a row added or removed between two calls
    shifts the rows after it. A row whose ``id`` an earlier page already
    returned is dropped (the first occurrence is kept). A row removed
    between calls can still make the next page skip the row after it.

    Args:
        client: The client whose MCP session runs the calls.
        tool: The MCP list tool; also the error's ``operation``.
        model: The row model.
        key: The envelope key that holds a page's rows.

    Returns:
        The rows of every page, in order, each ``id`` once.

    Raises:
        KaguraResponseError: A row does not match ``model``; ``has_more`` came
            without a usable ``next_cursor``, or with one an earlier page
            already sent; or the list is not complete after 50 pages.
    """
    rows: list[_T] = []
    returned_ids: set[Any] = set()
    async with aclosing(_iter_pages(client, tool, {})) as pages:
        async for result in pages:
            page = parse_response_list(model, result.get(key, []), operation=tool)
            rows.extend(row for row in page if getattr(row, "id", None) not in returned_ids)
            returned_ids.update(getattr(row, "id", None) for row in page)
            returned_ids.discard(None)
    return rows


async def _iter_pages(
    client: KaguraClient, tool: str, arguments: dict[str, Any]
) -> AsyncGenerator[dict[str, Any], None]:
    """Each page of a cursor-paged MCP list tool, in order (memory-cloud v0.82.0+, #1743).

    The first call sends ``arguments`` as given, so an older server, which
    pages nothing and sends no ``has_more``, answers in one call. Each later
    call sends ``arguments`` plus ``cursor``, the ``next_cursor`` of the page
    before. The next page is fetched only when the caller asks for it, so a
    caller that stops early sends no further request; wrap the iterator in
    :func:`contextlib.aclosing` when it may stop early.

    Args:
        client: The client whose MCP session runs the calls.
        tool: The MCP list tool; also the error's ``operation``.
        arguments: The tool arguments every page is sent with.

    Yields:
        Each page's result envelope, unparsed.

    Raises:
        KaguraResponseError: ``has_more`` came without a usable
            ``next_cursor``, or with one an earlier page already sent; or
            the list is not complete after 50 pages.
    """
    seen: set[str] = set()
    page_arguments = dict(arguments)
    for _ in range(_MAX_AUTO_PAGES):
        result = await client._call_tool_checked(tool, page_arguments)
        yield result
        cursor = _next_page_cursor(result, tool, seen)
        if cursor is None:
            return
        page_arguments = {**arguments, "cursor": cursor}
    raise _unfinished_list_error(tool)


def _next_page_cursor(result: dict[str, Any], tool: str, seen: set[str]) -> str | None:
    """The ``cursor`` for the page after ``result``, or None when it is the last.

    A page is the last unless it says ``has_more: true`` (an older server,
    which pages nothing, never does). The cursor returned is added to
    ``seen``, the cursors earlier pages of the same list sent.

    Args:
        result: One page of a cursor-paged MCP list tool (memory-cloud
            v0.82.0+, #1743).
        tool: The MCP list tool; also the error's ``operation``.
        seen: The cursors earlier pages sent; updated in place.

    Raises:
        KaguraResponseError: ``has_more`` came without a usable
            ``next_cursor``, or with one an earlier page already sent.
    """
    if result.get("has_more") is not True:
        return None
    cursor = result.get("next_cursor")
    if not isinstance(cursor, str) or not cursor:
        raise KaguraResponseError(
            f"{tool}: unexpected server response (has_more without a next_cursor)",
            operation=tool,
        )
    if cursor in seen:
        raise KaguraResponseError(
            f"{tool}: unexpected server response (a page repeated an earlier next_cursor)",
            operation=tool,
        )
    seen.add(cursor)
    return cursor


def _unfinished_list_error(tool: str) -> KaguraResponseError:
    """The error for a paged list still unfinished after ``_MAX_AUTO_PAGES`` pages."""
    return KaguraResponseError(
        f"{tool}: the list did not end within {_MAX_AUTO_PAGES} pages", operation=tool
    )


# ---------------------------------------------------------------------------
# Session tools (#301): local checks mirroring the server's own bounds
# ---------------------------------------------------------------------------

# remember_batch items (server ``MAX_BATCH_ITEMS``). The batch-level keys an
# item may carry (server ``_BATCH_LEVEL``) are not checked here: memory-cloud
# v0.156.0+ (#2753) decides on them with a type-strict rule after remember's
# string coercion, which a local copy would drift from.
_BATCH_MAX_ITEMS = 50
# forget(memory_ids=) ids per call (server ``FORGET_MAX_IDS``).
_FORGET_MAX_IDS = 50
# forget(query=) matches deleted when ``k`` is unset (the server's default too).
_FORGET_DEFAULT_K = 10
# describe_tools names / guide topics per call (server ``MAX_NAMES`` / ``MAX_TOPICS``).
_DESCRIBE_MAX_NAMES = 20
_GUIDE_MAX_TOPICS = 20


def _checked_batch_items(items: Any, context_id: str) -> list[dict[str, Any]]:
    """Reject a ``remember_batch`` ``items`` list before any request.

    Per-item rules the server reports per item (a missing ``summary``, an
    unknown key, a batch-level ``dedupe`` / ``tags_normalize`` / ``verbose``
    it does not accept) stay the server's. Refused here: an item
    ``context_id`` other than the batch's (the server accepts one that
    repeats it, compared as a string), and the tag cap that
    :meth:`KaguraClient.remember` also checks.

    Raises:
        TypeError: ``items`` is not a list, or an item is not a dict.
        ValueError: Not 1-50 items, an item carries another context's
            ``context_id``, or an item's ``tags`` list breaks the 50-tag /
            100-character cap.
    """
    if not isinstance(items, list):
        raise TypeError(f"items must be a list of dicts, got {type(items).__name__}")
    if not 1 <= len(items) <= _BATCH_MAX_ITEMS:
        raise ValueError(f"items must hold 1 to {_BATCH_MAX_ITEMS} items, got {len(items)}")
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise TypeError(f"items[{index}] must be a dict, got {type(item).__name__}")
        if "context_id" in item and item["context_id"] != context_id:
            raise ValueError(
                f"items[{index}] carries another context_id: the batch's context_id "
                "applies to every item"
            )
        tags = item.get("tags")
        if isinstance(tags, list):
            try:
                _validate_memory_tags(tags)
            except ValueError as e:
                raise ValueError(f"items[{index}]: {e}") from None
    return items


def _partial_session_rollback_error(result: dict[str, Any]) -> KaguraPartialSessionRollbackError:
    """The error for ``rollback_session``'s ``partial_rollback`` envelope.

    Not :meth:`KaguraClient._raise_for_mcp_error`'s ``partial_rollback``:
    that one reads a Sleep run's ``rollback_summary``, which this envelope
    lacks (``session_rollback.py`` sends ``summary`` and ``totals``). The
    partial rollback is already committed, so a field the SDK cannot read is
    ``None`` rather than a second error that would hide it.
    """
    totals = result.get("totals")
    err = KaguraPartialSessionRollbackError(
        f"rollback_session failed (partial_rollback): {result.get('message', 'Unknown error')}",
        session_summary=_validate_or_none(SessionRollbackSummary, result.get("summary")),
        totals=totals if isinstance(totals, dict) else None,
        rollback_id=_opt_int(result.get("rollback_id")),
        remaining=_opt_int(result.get("remaining")),
    )
    annotate_server_error(err, "partial_rollback", result)
    return err


def _str_list(name: str, values: Any) -> list[str]:
    """A list-of-str argument, copied, or the ``TypeError`` naming it.

    A bare ``str`` is refused rather than iterated: ``list("created")``
    would send its characters as the list.

    Raises:
        TypeError: ``values`` is a ``str`` or not a list, or an entry is not
            a ``str``.
    """
    if isinstance(values, str) or not isinstance(values, list):
        raise TypeError(f"{name} must be a list of strings, not {type(values).__name__}")
    for index, value in enumerate(values):
        if not isinstance(value, str):
            raise TypeError(f"{name}[{index}] must be a str, got {type(value).__name__}")
    return list(values)


def _checked_memory_ids(memory_ids: Any) -> list[str]:
    """Reject a ``forget(memory_ids=)`` list the server would refuse.

    Distinct the way the server compares them: as UUIDs, so two spellings
    of one id (case) are the same id.

    Raises:
        TypeError: A bare ``str``, a non-list, or a non-``str`` id.
        ValueError: Not 1-50 ids, an id that is not a UUID, or a repeat.
    """
    ids = _str_list("memory_ids", memory_ids)
    if not 1 <= len(ids) <= _FORGET_MAX_IDS:
        raise ValueError(f"memory_ids must hold 1 to {_FORGET_MAX_IDS} ids, got {len(ids)}")
    seen: set[uuid.UUID] = set()
    for index, value in enumerate(ids):
        try:
            parsed = uuid.UUID(value)
        except ValueError:
            raise ValueError(f"memory_ids[{index}] is not a memory UUID") from None
        if parsed in seen:
            raise ValueError(f"memory_ids lists {parsed} more than once")
        seen.add(parsed)
    return ids


def _checked_names(name: str, values: Any, limit: int) -> list[str]:
    """Reject a list-of-names argument the server would refuse.

    Raises:
        TypeError: A bare ``str``, a non-list, or a non-``str`` entry.
        ValueError: More than ``limit`` entries.
    """
    names = _str_list(name, values)
    if len(names) > limit:
        raise ValueError(f"{name} holds at most {limit} names, got {len(names)}")
    return names
