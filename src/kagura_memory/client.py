"""Low-level REST API client for Kagura Memory Cloud."""

import asyncio
import itertools
import json
import logging
import math
import warnings
from datetime import datetime
from typing import Any, Literal, Self, TypeVar
from urllib.parse import quote

import httpx
from pydantic import BaseModel as _BaseModel

from ._auth import _OAuthAuth, _resolve_auth, _StaticAuth
from ._http import (
    SDK_VERSION,
    _opt_int,
    base_url_from_mcp,
    extract_detail,
    gate_error,
    mcp_session_expired,
    mcp_session_header,
    parse_response,
    parse_response_list,
    raise_for_kagura_status,
    validate_coordinate,
    validate_https_url,
    validate_lat_lon,
)
from ._version import meets_minimum, parse_version
from .exceptions import (
    KaguraConnectionError,
    KaguraError,
    KaguraNotFoundError,
    KaguraPartialRollbackError,
    KaguraResponseError,
    _exc_message,
)
from .models import (
    Agent,
    AgentBinding,
    AgentBootstrapComponentName,
    AgentBootstrapResponse,
    ContextInfo,
    ContextTagsResponse,
    DuplicatesResponse,
    Edge,
    EmbeddingModelsResponse,
    EmbeddingStatus,
    GuardrailSet,
    ListTagsResponse,
    MeasurementAggregate,
    MeasurementPeriod,
    MeasurementResult,
    MeasurementSeries,
    MemoryListResponse,
    MemoryStatsResponse,
    RollbackResult,
    RollbackSummary,
    ServerInfo,
    SleepReport,
    SleepReportDetail,
    ToolTrigger,
    UsageInfo,
    _agent_update_payload,
    _binding_scope_payload,
    _bootstrap_payload,
    _details_with_tool_trigger,
)

_T = TypeVar("_T", bound=_BaseModel)


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


class KaguraClient:
    """
    Low-level REST API client for Kagura Memory Cloud MCP tools.

    All methods may raise:
        KaguraAuthError: Authentication failed
        KaguraConnectionError: Connection to server failed
        KaguraRateLimitError: Rate limit exceeded — any HTTP 429, including
            the daily MCP quota when the server's rate-limit middleware
            refuses the request before the tool runs

    MCP tool methods additionally translate the server's structured domain
    errors (``{"status": "error", ...}``) into exceptions rather than
    returning them as data (issue #180): a missing context/memory/report
    raises :class:`KaguraNotFoundError`, a quota refusal (including the
    in-band daily MCP call cap, ``rate_limit_exceeded``)
    :class:`KaguraQuotaError` and a plan/feature gate
    :class:`KaguraFeatureNotAvailableError`, each carrying the server's
    detail fields (#256), and any other domain error raises
    :class:`KaguraError`. Callers should use ``try/except`` rather than
    inspecting ``result["status"]``. On the tool methods that return a
    model, a success payload that does not match it (a server newer than
    the SDK) raises :class:`KaguraResponseError` naming the tool (#250).
    :meth:`list_memories` does the same with
    ``operation="KaguraClient.list_memories"``. The other REST-backed methods
    (``get_server_info``, ``check_server_version``, ``get_embedding_status``,
    ``get_memory_stats``, ``find_duplicates``, ``list_embedding_models``)
    still raise :class:`KaguraConnectionError` ("Invalid response format") on
    drift; catch :class:`KaguraError` to cover both.
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
                query, which every MCP request sends: ``?profile=<name>`` /
                ``?tools=a,b`` (v0.73.0+) only narrow what ``tools/list``
                returns — ``tools/call`` never reads them, so every method
                here keeps working — and ``?guardrails=off`` (v0.74.0+)
                drops the ``guardrails`` block from :meth:`get_context_info`.
                REST calls derive their base URL without the query (#258).
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
        is returned for the caller's ``raise_for_status`` to report. (As
        deployed, memory-cloud v0.75.0 skips that session check and re-adopts
        an unknown session id instead, so against it this never fires.)

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
        operation: str | None = None,
    ) -> Any:
        """GET a REST endpoint and return its decoded JSON body.

        Args:
            path: URL path (appended to ``_base_url``).
            params: Optional query parameters. A list value is sent as one
                repeated key per item (``?k=a&k=b``), which is how FastAPI
                reads a ``list[str]`` query.
            operation: The MCP tool this call stands in for. When set, a
                ``404`` and a ``422`` raise what that tool's
                ``context_not_found`` and ``invalid_argument`` errors raise
                (:meth:`_raise_for_mcp_error`), so a method moved from MCP to
                REST keeps its exceptions (#273).

        Raises:
            KaguraAuthError / KaguraRateLimitError / KaguraConnectionError: A
                non-2xx status (see :func:`raise_for_kagura_status`).
            KaguraNotFoundError / KaguraError: A ``404`` / ``422`` when
                ``operation`` is set.
            KaguraConnectionError: A network failure, or a 2xx body that is
                not JSON.
        """
        url = f"{self._base_url}{path}"
        try:
            response = await self._client.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            code = {404: "context_not_found", 422: "invalid_argument"}.get(e.response.status_code)
            if operation is not None and code is not None:
                message = extract_detail(e.response) or f"HTTP {e.response.status_code}"
                try:
                    self._raise_for_mcp_error(
                        {"status": "error", "error": code, "message": message}, operation
                    )
                except KaguraError as mapped:
                    # Chained explicitly, as raise_for_kagura_status chains it.
                    raise mapped from e
            raise_for_kagura_status(e)
        except httpx.RequestError as e:
            raise KaguraConnectionError(f"Connection failed: {_exc_message(e)}") from e
        except (ValueError, TypeError) as e:
            raise KaguraConnectionError(f"Invalid response format: {_exc_message(e)}") from e

    async def _rest_get(
        self,
        path: str,
        model: type[_T],
        params: dict[str, Any] | None = None,
    ) -> _T:
        """GET a REST endpoint and parse into a Pydantic model.

        Drift raises :class:`KaguraConnectionError` ("Invalid response
        format"), not :class:`KaguraResponseError` — the contract #250 kept
        for these methods (``kagura doctor`` catches it on
        ``check_server_version``). A new REST method should instead call
        :meth:`_rest_get_json` and :func:`parse_response`, as
        :meth:`list_memories` does.

        Args:
            path: URL path (appended to ``_base_url``).
            model: Pydantic model class for response validation.
            params: Optional query parameters.

        Returns:
            Validated model instance.
        """
        data = await self._rest_get_json(path, params)
        try:
            return model.model_validate(data)
        except (ValueError, TypeError) as e:
            raise KaguraConnectionError(f"Invalid response format: {_exc_message(e)}") from e

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
    ) -> dict[str, Any]:
        """Call remember MCP tool.

        Args:
            context_id: Context ID.
            summary: Memory summary (10-500 chars).
            content: Memory content.
            type: Memory type. Server validates against its own vocabulary;
                the SDK passes through.
            importance: Importance score (0.0-1.0).
            tags: Optional tags.
            source_uri: Origin URI (e.g. ``file:///``, ``https://``,
                ``vault://``).
            linked_memory_ids: Existing memory UUIDs to declare as graph
                edges from this memory. Server creates ``declared_link``
                edges with ``weight=1.0`` atomically.
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
                server v0.45.0+ (#1208).
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

        Returns:
            API response with ``memory_id``, ``scope`` and the context fields.
            Since memory-cloud v0.65.0 two optional keys may follow, each
            absent (never ``null``) when it does not apply:
            ``persistence`` — ``{scope, committed, promotes_via,
            consolidation_archive_min_age_days, detail}``, how the scope the
            memory landed in is consolidated — and ``lint`` —
            ``[{code, hint, subject?}]``, advisory hints about a write that
            will recall poorly (``summary_short``, ``summary_long``,
            ``summary_narrative``, ``no_tags``, ``tag_near_duplicate``). The
            memory is stored either way; act on a hint with
            :meth:`update_memory`.

        Raises:
            ValueError: ``tool_trigger`` is given and ``details`` already has
                a ``"tool_trigger"`` key — pass the guardrail one way only.
            KaguraError: Server-side rejection, e.g. ``permission_denied`` or
                ``validation_error`` for a guardrail write.
        """
        details = _details_with_tool_trigger(details, tool_trigger)
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
    ) -> dict[str, Any]:
        """
        Call recall MCP tool.

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
            include_explore_hints: When True, the server includes up to 3
                graph discovery hints in the response under the
                ``explore_hints`` key — useful as seeds for a follow-up
                :meth:`explore` call.
            include_superseded: When True, also return memories shadowed by a
                ``supersedes`` edge, annotated with ``superseded_by``. Default
                recall demotes them, so this is how you read the history that
                :meth:`remember` (``supersedes=...``) deliberately preserves —
                without it, superseding would be a one-way door. Note this is a
                top-level argument, **not** a ``filters`` key. Requires
                memory-cloud server v0.45.0+ (#1208).

        Returns:
            API response with ``results`` (ranked summaries), ``count``,
            ``related_tags`` (``[{tag, count}]``), ``confidence`` and the
            context fields. The optional keys below are absent when they do
            not apply, so read them with ``.get()``:

            - ``degraded`` / ``degraded_reason`` (v0.66.0+): the semantic half
              of hybrid search was unavailable, so the results are
              keyword-only. An empty or weak result then means "search
              impaired", not "nothing stored" — retry later.
            - ``tag_suggestions`` (v0.65.0+): ``{requested_tag: ["stored-tag
              (count)", ...]}`` when a tag filter matched nothing and similar
              stored tags exist. Advisory — the filter was not widened.
            - ``explore_hints``: present whenever ``include_explore_hints``
              is set, possibly as an empty list.
            - Per result, ``context_summary``, ``superseded_by``,
              ``contradicts`` and ``supersede_candidate``: absent when empty
              since memory-cloud v0.73.0; older servers send ``null`` /
              ``[]``, so test them for truthiness.

        Raises:
            ValueError: If ``query`` is empty/whitespace; if neither
                ``context_id`` nor ``context_ids`` is provided; if
                ``context_ids`` has fewer than 2 or more than 20 IDs; or if
                ``search_mode`` is not one of "hybrid"/"semantic"/"keyword".
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
            ``trigger`` (``details.trigger`` is inside it).
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

        Returns:
            API response with ``results`` — memories whose stored location falls
            within the radius, nearest first, each carrying ``distance_m``.

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
                Omit to use the server default.

        Returns:
            API response with ``memories`` (the pinned set — each item
            ``{memory_id, summary, context_summary, type, importance,
            delivery_mode}``), ``truncated`` (bool), ``total_available``
            (int), ``cap`` (the cap applied) and the context fields.

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
        return await self._call_tool_checked("load_pinned", arguments)

    async def load_guardrails(self, context_id: str, *, cap: int | None = None) -> GuardrailSet:
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

        Returns:
            :class:`~kagura_memory.models.GuardrailSet`.

        Raises:
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
                or boolean).
            ttl_seconds: Optional TTL in seconds (server clamps to 2592000 =
                30 days). Omit for no expiry.

        Returns:
            API response acknowledging the upsert.

        Raises:
            KaguraNotFoundError: Context not found.
            KaguraError: Other server-side error.
        """
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
    ) -> dict[str, Any]:
        """Read ephemeral agent run-state.

        Calls the ``get_state`` MCP tool — the read side of the session-state
        lane (see :meth:`set_state`). Supply ``key`` to read one value, or omit
        it to list all live keys for the context. Expired entries are never
        returned. This lane is excluded from :meth:`recall` by design.

        Args:
            context_id: Target context UUID.
            key: Optional state key. Omit to list all live ``(key, value)``
                entries for the context.

        Returns:
            API response with the value for ``key``, or all live entries when
            ``key`` is omitted.

        Raises:
            KaguraNotFoundError: Context not found.
            KaguraError: Other server-side error.
        """
        arguments: dict[str, Any] = {"context_id": context_id}
        if key is not None:
            arguments["key"] = key
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
                ``load_pinned`` to [1, 1000].
            upcoming_until: ISO upper bound for upcoming time memories
                (the lower bound is always now).
            include: Component selector — a subset of ``"pinned"``,
                ``"recall"``, ``"upcoming"``, ``"state"``, ``"policy"``.
                Omit for all components.

        Returns:
            :class:`AgentBootstrapResponse` — the composed envelope with
            ``agent`` (identity + resolved binding), ``context``,
            ``instructions``, per-component payloads in ``components``,
            the ``correlation`` block, and the ``degraded`` flag. Since
            memory-cloud v0.73.0 the ``upcoming`` component's rows carry
            ``trigger`` in place of ``details``, as :meth:`recall_upcoming`
            does by default; bootstrap has no ``include_details`` opt-out, so
            fetch ``details`` with :meth:`reference` or
            ``recall_upcoming(include_details=True)``.

        Raises:
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

        Returns:
            List of :class:`Agent` rows in the active workspace.
        """
        result = await self._call_tool_checked("list_agents", {})
        return parse_response_list(Agent, result.get("agents", []), operation="list_agents")

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

        Args:
            name_contains: Only contexts whose name or display name contains
                this text (case-insensitive, max 100 characters). Omitted when
                empty or unset.
            include_summary: Add ``summary`` truncated to 300 characters; items
                that were cut also carry ``summary_truncated: true``.
            include_details: Add the full ``summary`` and ``embedding_model``
                (the pre-v0.73.0 item shape). Wins over ``include_summary``.
                Large on big workspaces, so combine it with ``name_contains``.
            include_stats: Add ``memory_count`` per context.

        Returns:
            The envelope ``{status, contexts, count, total, limit, can_create,
            hint?}``:

            - ``contexts``: the items described above.
            - ``count``: contexts in the workspace. This is quota usage, so
              ``name_contains`` does not change it. With no current workspace
              it is the number of contexts the caller can see, still counted
              before ``name_contains``.
            - ``total``: contexts in this response (v0.73.0+). ``0`` on no
              match is still a success.
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
        """
        arguments: dict[str, Any] = {}
        if name_contains:
            arguments["name_contains"] = name_contains
        if include_summary:
            arguments["include_summary"] = True
        if include_details:
            arguments["include_details"] = True
        if include_stats:
            arguments["include_stats"] = True
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

        Args:
            context_id: Context ID to list tags from.
            limit: Maximum tags to return (1-500, default 50).
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
            ``tags`` (list of :class:`TagInfo`), and ``total`` count.

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
        data = await self._rest_get_json(path, params, operation="list_tags")
        body = parse_response(ContextTagsResponse, data, operation="list_tags")
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

        Returns:
            List of tool definition dicts with name, description, and inputSchema

        Raises:
            KaguraAuthError: Authentication failed
            KaguraConnectionError: Connection to server failed
        """
        result = await self._make_jsonrpc_request(method="tools/list", params={})
        return result.get("tools", [])

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
    ) -> dict[str, Any]:
        """
        Call reference MCP tool (get full memory details).

        Args:
            context_id: Context ID
            memory_id: Memory ID to retrieve full details for

        Returns:
            API response dict. Memory data is in ``result["memory"]``::

                result = await client.reference(ctx, mem_id)
                memory = result["memory"]
                print(memory["summary"], memory["content"])
        """
        arguments = {
            "context_id": context_id,
            "memory_id": memory_id,
        }
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
    ) -> dict[str, Any]:
        """Update an existing memory in-place or upsert by external ID.

        Two modes (provide exactly one of memory_id or external_id):

        1. In-place update (memory_id): Modifies specific fields while
           preserving memory ID, graph edges, and creation timestamp.
        2. Upsert (external_id): Finds by external resource ID within context.
           If found, replaces. If not found, creates new.
           Requires summary, content, and type.

        Args:
            context_id: Context UUID.
            memory_id: UUID of memory to update in-place.
            external_id: External resource ID for upsert lookup.
            summary: Updated summary (10-500 chars).
            content: Updated content.
            type: Updated memory type.
            importance: Updated importance (0.0-1.0).
            tags: Updated tags (replaces the list; ``[]`` clears it).
            context_summary: Updated context summary (max 2000 chars).
                ``""`` clears it; ``None`` leaves it unchanged.
            delivery_mode: Pin or unpin the memory. ``"always"`` pins it
                (deterministically loaded every turn via :meth:`load_pinned`,
                promoted to ``scope="persistent"``); ``"on_recall"`` unpins it
                (back to probabilistic :meth:`recall`; the memory stays
                persistent). Omit to leave the current delivery mode unchanged.
            details: Updated structured details JSON. **Replaces ``details``
                wholesale** — the server does not deep-merge, so read the
                current value with :meth:`reference` and re-send every key you
                want to keep (notably ``location``, which otherwise drops off
                :meth:`recall_nearby`, and ``tool_trigger``, which otherwise
                unmarks a guardrail). ``{}`` clears it; omit to leave details
                unchanged.
            dismiss_supersede_candidate: Reject this memory's current
                ``supersede_candidate`` (the older near-duplicate that
                :meth:`recall` / :meth:`reference` suggest it replaces), for two
                memories that are deliberately separate. Nothing is deleted or
                shadowed, and it can be the only change in the call. Requires
                ``memory_id``. Sent only when ``True``. Requires memory-cloud
                v0.65.0+ (#1504). An older server silently drops the flag: a
                dismissal-only call then succeeds as an empty in-place update
                that dismisses nothing and refreshes ``updated_at``. To accept
                the suggestion instead, create a ``supersedes`` edge.
            tool_trigger: Mark (or re-mark) the memory as a tool guardrail —
                sent as ``details["tool_trigger"]``, merged with ``details``;
                same contract and permissions as :meth:`remember`'s
                ``tool_trigger`` (memory-cloud v0.74.0+). Because ``details``
                is replaced wholesale, passing only ``tool_trigger`` sends
                ``details={"tool_trigger": ...}`` and drops every other key —
                pass the keys to keep in ``details`` alongside it. To unmark,
                send ``details`` without the key (or with
                ``"tool_trigger": None``). Any edit of a memory that already
                carries a trigger needs context editor or above. Keyword-only.

        Returns:
            API response with updated memory info (``memory_id``,
            ``operation``, ``re_embedded``, ``scope``). Since memory-cloud
            v0.65.0 it may also carry ``persistence`` and ``lint`` as in
            :meth:`remember` — ``lint`` describes the memory after the update.
            When a dismissal applied,
            ``supersede_candidate_dismissed`` holds the rejected candidate's
            memory_id. It is the only confirmation that a dismissal happened:
            the key is absent both when there was no live suggestion and when
            the server predates v0.65.0.

        Raises:
            ValueError: If neither or both of ``memory_id`` and ``external_id``
                are given; if ``dismiss_supersede_candidate`` is combined
                with ``external_id`` (the server rejects that pair: an upsert
                replaces the memory and its suggestion); or if
                ``tool_trigger`` is given together with a
                ``details["tool_trigger"]`` key.
            KaguraError: Server-side rejection, e.g. ``permission_denied`` for
                a guardrail edit below context editor.
        """
        if not memory_id and not external_id:
            raise ValueError("Provide exactly one of memory_id or external_id")
        if memory_id and external_id:
            raise ValueError("Provide exactly one of memory_id or external_id")
        if dismiss_supersede_candidate and external_id:
            raise ValueError(
                "dismiss_supersede_candidate requires memory_id (in-place mode); "
                "an external_id upsert replaces the memory and its suggestion."
            )
        details = _details_with_tool_trigger(details, tool_trigger)

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
        return await self._call_tool_checked("update_memory", arguments)

    async def forget(
        self,
        context_id: str,
        memory_id: str | None = None,
        query: str | None = None,
        k: int = 10,
    ) -> dict[str, Any]:
        """
        Call forget MCP tool (soft delete memories).

        Delete by specific memory_id or by search query. This is a soft
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

        Args:
            context_id: Context ID
            memory_id: UUID of specific memory to delete
            query: Search query to find and delete matching memories
            k: Number of memories to delete in query mode (default: 10)

        Returns:
            API response with deletion results (``deleted_count``,
            ``memory_ids``) — may be fewer than targeted, see above.

        Raises:
            ValueError: If neither ``memory_id`` nor ``query`` is provided —
                ``forget`` always targets a specific memory or a search; an
                unqualified call would be an ambiguous no-op.
            KaguraError: Server-side rejection, e.g. a ``query`` delete while
                recall is degraded.
        """
        if not memory_id and not query:
            raise ValueError("Provide either memory_id or query")
        arguments: dict[str, Any] = {"context_id": context_id}
        if memory_id:
            arguments["memory_id"] = memory_id
        if query:
            arguments["query"] = query
            arguments["k"] = k
        return await self._call_tool_checked("forget", arguments)

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
        # Pre-check quota
        contexts = await self.list_contexts()
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

        Args:
            context_id: Context UUID to delete.

        Returns:
            API response with deletion confirmation.
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
            resource_id: Updated resource identifier for external data ingestion.
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
            resource_id: Resource identifier for data ingestion.
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
                target.

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

        Returns both outgoing and incoming edges, deduplicated.

        Args:
            context_id: Context UUID containing the memory.
            memory_id: Memory UUID whose edges to list.
            min_weight: Minimum edge weight (0.0-3.0). Edges below this are filtered.
            edge_types: Restrict to these edge types (e.g. ``["related_to"]``). ``None``
                returns all types.
            limit: Maximum edges per direction. **The server applies this to outgoing
                AND incoming queries independently**, so the practical maximum returned
                is ``2 * limit`` minus dedup overlap. ``None`` means no limit.

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
        return parse_response_list(Edge, result.get("edges", []), operation="list_edges")

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
            KaguraError: Edge not found or other server-side error.
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
            logging.getLogger("kagura_memory").warning(
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
    ) -> dict[str, Any]:
        """Update hybrid search configuration for a context.

        Weights must sum to 1.0 (±0.01). Requires owner or editor permission.
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

        Returns:
            Current search config after update.
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
        """
        return await self._rest_get("/api/v1/system/info", ServerInfo)

    async def check_server_version(self) -> ServerInfo:
        """Check the connected server's version against the SDK's tested minimum.

        Advisory only — calls ``get_server_info()`` and logs a warning
        via :mod:`logging` when the server version is below
        :data:`MIN_SERVER_VERSION`. Does not raise. Older servers may
        silently ignore unknown parameters.

        A ``v`` prefix, build metadata and pre-release suffixes are read,
        so ``"v0.16.0"`` and ``"0.17.1-rc1"`` (a pre-release of the minimum)
        both warn. A version with no ``MAJOR.MINOR.PATCH`` at its start, such
        as ``"0.17"`` or ``"main-abc123"``, cannot be compared and does not
        warn; ``kagura doctor`` reports it as ``info``.

        Returns:
            ServerInfo from the server.
        """
        info = await self.get_server_info()
        if meets_minimum(info.version, _MIN_SERVER_VERSION_TUPLE) is False:
            logging.getLogger("kagura_memory").warning(
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
        """
        return await self._rest_get("/api/v1/workspace/embedding-status", EmbeddingStatus)

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
        """
        params = {
            "sort_by": sort_by,
            "sort_order": sort_order,
            "limit": limit,
            "offset": offset,
        }
        return await self._rest_get(
            f"/api/v1/contexts/{context_id}/memory-stats", MemoryStatsResponse, params=params
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
        """
        params = {"threshold": threshold, "limit": limit}
        return await self._rest_get(
            f"/api/v1/contexts/{context_id}/duplicates", DuplicatesResponse, params=params
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
            when it has one), ``total`` (matching rows across all pages), and
            ``has_more``.

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
                raised ``KaguraConnectionError`` here, as the other
                REST-backed methods still do.
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
        data = await self._rest_get_json("/api/v1/memory/list", params)
        return parse_response(MemoryListResponse, data, operation="KaguraClient.list_memories")

    @staticmethod
    def _raise_for_mcp_error(result: dict[str, Any], operation: str) -> None:
        """Translate an MCP tool's structured error response to an SDK exception.

        The server's MCP tools return ``{"status": "error", "error": <code>,
        "message": <str>, ...}`` for domain errors that the JSON-RPC transport
        layer cannot represent (e.g. ``report_not_found``). HTTP-level
        errors (401, 5xx) are already handled by ``_make_jsonrpc_request``.

        A plan or quota refusal raises :class:`KaguraQuotaError` or
        :class:`KaguraFeatureNotAvailableError` carrying the envelope's gate
        fields, and ``partial_rollback`` raises
        :class:`KaguraPartialRollbackError` (#256). Every other code raises
        :class:`KaguraError`.
        """
        if result.get("status") != "error":
            return
        code = result.get("error", "unknown")
        message = result.get("message", "Unknown error")
        if code in (
            "report_not_found",
            "context_not_found",
            "memory_not_found",
            "agent_not_found",
            "binding_not_found",
        ):
            raise KaguraNotFoundError(f"{operation}: {message}")
        text = f"{operation} failed ({code}): {message}"
        if code == "partial_rollback":
            report_id = result.get("report_id")
            report_id = report_id if isinstance(report_id, str) else None
            try:
                summary = parse_response(
                    RollbackSummary, result.get("rollback_summary"), operation=operation
                )
            except KaguraResponseError as drift:
                # The partial reversal is already committed; a summary the
                # SDK cannot read must not hide that (#256).
                raise KaguraPartialRollbackError(
                    f"{text} (rollback_summary could not be read)", report_id=report_id
                ) from drift
            raise KaguraPartialRollbackError(text, report_id=report_id, summary=summary)
        raise gate_error(str(code), text, result) or KaguraError(text)

    async def get_sleep_history(
        self,
        context_id: str,
        limit: int = 10,
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

        Args:
            context_id: Context UUID.
            limit: Maximum number of runs to return (server clamps to 1-50,
                default: 10).

        Returns:
            List of ``SleepReport`` summaries ordered by ``started_at``
            descending (newest first).

        Raises:
            KaguraNotFoundError: Context not found.
            KaguraResponseError: A run did not match the SDK's model.
            KaguraError: Other server-side error.
        """
        result = await self._call_tool_checked(
            "get_sleep_history",
            {"context_id": context_id, "limit": limit},
        )
        return parse_response_list(
            SleepReport, result.get("reports"), operation="get_sleep_history"
        )

    async def get_sleep_report(
        self,
        context_id: str,
        report_id: str,
    ) -> SleepReportDetail:
        """Get a detailed Sleep Maintenance report including audit log.

        Args:
            context_id: Context UUID. Used for permission scoping; the server
                also verifies the report belongs to the caller.
            report_id: Sleep report UUID.

        Returns:
            ``SleepReportDetail`` with per-phase results and the per-action
            audit log.

        Raises:
            KaguraNotFoundError: Report not found or not owned by caller.
            KaguraResponseError: The report did not match the SDK's model.
            KaguraError: Other server-side error.
        """
        result = await self._call_tool_checked(
            "get_sleep_report",
            {"context_id": context_id, "report_id": report_id},
        )
        # The MCP tool wraps the report fields under a "report" key;
        # flatten so SleepReportDetail (a SleepReport subclass) validates
        # naturally without forcing callers through an extra ``.report.``
        # accessor. A missing or non-object "report" is passed through
        # unflattened so it fails validation like any other drift.
        report = result.get("report")
        if isinstance(report, dict):
            report = {
                **report,
                **{key: result[key] for key in ("actions", "action_count") if key in result},
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

        Args:
            context_id: Context UUID.
            report_id: Sleep report UUID.

        Returns:
            ``RollbackResult`` with per-category counts of reversed actions.

        Raises:
            KaguraNotFoundError: Report not found or not owned by caller.
            KaguraPartialRollbackError: Some undo steps failed. The rest
                stay reversed; ``.summary`` counts them and lists the
                failures in ``.summary.errors``.
            KaguraError: Other server-side error. The exception message
                includes the server-side error code for triage.
        """
        result = await self._call_tool_checked(
            "rollback_sleep_run",
            {"context_id": context_id, "report_id": report_id},
        )
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
        """
        return await self._rest_get("/api/v1/system/embedding/models", EmbeddingModelsResponse)

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
