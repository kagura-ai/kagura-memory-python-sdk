"""Low-level REST API client for Kagura Memory Cloud."""

import itertools
import json
import logging
from typing import Any, Literal, TypeVar

import httpx
from pydantic import BaseModel as _BaseModel

from ._auth import _resolve_auth, _StaticAuth
from ._http import SDK_VERSION, base_url_from_mcp, validate_https_url
from .exceptions import (
    KaguraAuthError,
    KaguraConnectionError,
    KaguraError,
    KaguraNotFoundError,
    _exc_message,
)
from .models import (
    ContextInfo,
    DuplicatesResponse,
    Edge,
    EmbeddingModelsResponse,
    EmbeddingStatus,
    ListTagsResponse,
    MemoryListResponse,
    MemoryStatsResponse,
    RollbackResult,
    ServerInfo,
    SleepReport,
    SleepReportDetail,
    UsageInfo,
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

_MIN_SERVER_VERSION_TUPLE = tuple(int(x) for x in MIN_SERVER_VERSION.split(".")[:3])


class KaguraClient:
    """
    Low-level REST API client for Kagura Memory Cloud MCP tools.

    All methods may raise:
        KaguraAuthError: Authentication failed
        KaguraConnectionError: Connection to server failed
        KaguraRateLimitError: Rate limit exceeded
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
                ``.kagura.json``).
            timeout: Request timeout in seconds.
            profile: Named OAuth profile to load (overrides
                ``KAGURA_PROFILE`` and the credentials file's
                ``default_profile``).
        """
        resolved = _resolve_auth(api_key=api_key, mcp_url=mcp_url, profile=profile)

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
        self._request_id_counter = itertools.count(1)
        # Per-(client, context_id) cache for ingest steering: get_context_info
        # is fetched at most once per context for the client's lifetime. A
        # context_id present as a key with value None is the "fetched but
        # unusable" sentinel (empty info or fetch failure) — it suppresses
        # re-fetching on every section summarization. See
        # :meth:`_get_context_info_cached`.
        self._context_info_cache: dict[str, ContextInfo | None] = {}

    def _next_request_id(self) -> int:
        """Get next JSON-RPC request ID (concurrency-safe via itertools.count)."""
        return next(self._request_id_counter)

    async def _initialize_session(self) -> None:
        """Initialize MCP session if not already initialized."""
        if self._session_id:
            return

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
            if e.response.status_code == 401:
                raise KaguraAuthError("Authentication failed. Check your API key.") from e
            raise KaguraConnectionError(f"HTTP {e.response.status_code}: {_exc_message(e)}") from e
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

        headers = {"mcp-session-id": self._session_id} if self._session_id else {}

        try:
            response = await self._client.post(self.mcp_url, json=body, headers=headers)
            response.raise_for_status()

            data = response.json() or {}
            if "error" in data:
                error = data["error"]
                raise KaguraConnectionError(f"MCP error: {error.get('message', error)}")

            return data.get("result", {})

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise KaguraAuthError("Authentication failed") from e
            raise KaguraConnectionError(f"HTTP {e.response.status_code}") from e
        except httpx.RequestError as e:
            raise KaguraConnectionError(f"Connection failed: {_exc_message(e)}") from e

    async def _rest_get(
        self,
        path: str,
        model: type[_T],
        params: dict[str, Any] | None = None,
    ) -> _T:
        """GET a REST endpoint and parse into a Pydantic model.

        Args:
            path: URL path (appended to ``_base_url``).
            model: Pydantic model class for response validation.
            params: Optional query parameters.

        Returns:
            Validated model instance.
        """
        url = f"{self._base_url}{path}"
        try:
            response = await self._client.get(url, params=params)
            response.raise_for_status()
            return model.model_validate(response.json())
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise KaguraAuthError("Authentication failed. Check your API key.") from e
            raise KaguraConnectionError(f"HTTP {e.response.status_code}") from e
        except httpx.RequestError as e:
            raise KaguraConnectionError(f"Connection failed: {_exc_message(e)}") from e
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
        delivery_mode: Literal["always", "on_recall", "on_trigger"] = "on_recall",
        context_summary: str | None = None,
        details: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
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
            delivery_mode: When the memory is surfaced. ``"on_recall"``
                (default) leaves it to probabilistic :meth:`recall`.
                ``"always"`` pins it to ``scope="persistent"`` on write so it
                is returned by every :meth:`load_pinned` call (Goal / Guardrail
                / critical-policy memories). ``"on_trigger"`` is for
                trigger-delivered memories. The default is sent only when it
                differs from the server's ``server_default='on_recall'``, so
                an unset value stays forward-compatible.
            context_summary: Brief explanation (max 2000 chars) of why the
                memory exists and how to use it. Distinct from ``summary``,
                which is the search-target text.
            details: Structured details JSON. Use for additional metadata
                like code locations, parent/child links, or any caller-defined
                payload that the server should store as-is.
            context: Open-ended context metadata JSON. Less structured than
                ``details``; useful for free-form provenance hints.

        Returns:
            API response with ``memory_id``.
        """
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

        return await self._call_tool("remember", arguments)

    async def recall(
        self,
        context_id: str | None = None,
        query: str = "",
        k: int = 5,
        use_rerank: bool = False,
        filters: dict[str, Any] | None = None,
        search_mode: str | None = None,
        context_ids: list[str] | None = None,
        include_explore_hints: bool = False,
    ) -> dict[str, Any]:
        """
        Call recall MCP tool.

        Args:
            context_id: Context ID for single-context search.
            query: Search query
            k: Number of results
            use_rerank: Enable AI reranking for higher quality results
            filters: Optional filters. Supported keys:
                - ``type``: memory type (e.g., ``"code"``)
                - ``tags``: list of tag strings (e.g., ``["python"]``)
                - ``tags_match``: ``"any"`` (default) or ``"all"`` for AND logic
                - ``created_after`` / ``created_before``: ISO 8601 datetime
                - ``updated_after`` / ``updated_before``: ISO 8601 datetime
            search_mode: Search strategy — "hybrid" (default), "semantic", or "keyword"
            context_ids: Search across multiple contexts (2–20 IDs).
                When provided, ``context_id`` is not required.
            include_explore_hints: When True, the server includes up to 3
                graph discovery hints in the response under the
                ``explore_hints`` key — useful as seeds for a follow-up
                :meth:`explore` call.

        Returns:
            API response with results list

        Raises:
            ValueError: If neither ``context_id`` nor ``context_ids`` is provided,
                or ``context_ids`` has fewer than 2 or more than 20 IDs.
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
        if use_rerank:
            arguments["use_rerank"] = True
        if filters:
            arguments["filters"] = filters
        if search_mode:
            if search_mode not in ("hybrid", "semantic", "keyword"):
                raise ValueError(f"Invalid search_mode: {search_mode!r}")
            arguments["search_mode"] = search_mode
        if include_explore_hints:
            arguments["include_explore_hints"] = True
        return await self._call_tool("recall", arguments)

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
        complete policy set should check ``truncated`` and raise ``cap`` (or
        page via :meth:`list_memories`) rather than assume completeness.

        Args:
            context_id: Target context UUID.
            cap: Optional override for the maximum number returned (1-1000).
                Omit to use the server default.

        Returns:
            API response with ``results`` (the pinned set), ``truncated``
            (bool), and ``total_available`` (int).

        Example:
            >>> pinned = await client.load_pinned(context_id=ctx)
            >>> if pinned["truncated"]:
            ...     pinned = await client.load_pinned(context_id=ctx, cap=1000)
            >>> for m in pinned["results"]:
            ...     full = await client.reference(
            ...         context_id=ctx, memory_id=m["memory_id"]
            ...     )
        """
        arguments: dict[str, Any] = {"context_id": context_id}
        if cap is not None:
            arguments["cap"] = cap
        return await self._call_tool("load_pinned", arguments)

    async def list_contexts(self) -> dict[str, Any]:
        """
        Call list_contexts MCP tool.

        Returns:
            API response with available contexts
        """
        return await self._call_tool("list_contexts", {})

    async def list_tags(
        self,
        context_id: str,
        limit: int = 50,
        min_count: int = 1,
        sort: Literal["count", "recent", "alpha"] = "count",
        prefix: str = "",
    ) -> ListTagsResponse:
        """List the tag vocabulary in a context with usage counts and recency.

        Call before :meth:`remember` to reuse existing tag spellings, or before
        :meth:`recall` with ``filters={"tags": [...]}`` to build accurate
        filters. This is the primary mitigation for tag drift (e.g. ``"auth"``
        vs ``"authentication"`` silently degrading recall precision).

        Requires memory-cloud server v0.15.4+ — older servers expose
        ``list_contexts`` and ``recall`` but not ``list_tags``, and will
        return an MCP "tool not found" error. ``MIN_SERVER_VERSION`` is
        deliberately not bumped because the rest of the SDK still works
        against v0.15.1+; only this method needs the newer server.

        Args:
            context_id: Context ID to list tags from.
            limit: Maximum tags to return (1-500, default 50).
            min_count: Minimum memory count per tag (1-10000, default 1).
            sort: Sort order — ``"count"`` (default), ``"recent"``, or ``"alpha"``.
            prefix: Case-insensitive prefix filter for autocomplete-style lookup.
                ``%`` and ``_`` are treated as literals server-side. Max 200 chars.

        Returns:
            :class:`ListTagsResponse` with ``context_id``, ``context_name``,
            ``tags`` (list of :class:`TagInfo`), and ``total`` count.

        Raises:
            ValueError: If ``limit``, ``min_count``, or ``prefix`` are out of range.
            KaguraNotFoundError: Context not found or caller lacks access.
            KaguraError: Other server-side error.
        """
        if not 1 <= limit <= 500:
            raise ValueError(f"limit must be between 1 and 500, got {limit}")
        if not 1 <= min_count <= 10_000:
            raise ValueError(f"min_count must be between 1 and 10000, got {min_count}")
        if len(prefix) > 200:
            raise ValueError(f"prefix must be at most 200 characters, got {len(prefix)}")

        arguments: dict[str, Any] = {
            "context_id": context_id,
            "limit": limit,
            "min_count": min_count,
            "sort": sort,
        }
        if prefix:
            arguments["prefix"] = prefix
        result = await self._call_tool("list_tags", arguments)
        self._raise_for_mcp_error(result, "list_tags")
        return ListTagsResponse.model_validate(result)

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
        return await self._call_tool("explore", arguments)

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
        return await self._call_tool("reference", arguments)

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
            tags: Updated tags.
            context_summary: Updated context summary (max 2000 chars).
            delivery_mode: Pin or unpin the memory. ``"always"`` pins it
                (deterministically loaded every turn via :meth:`load_pinned`,
                promoted to ``scope="persistent"``); ``"on_recall"`` unpins it
                (back to probabilistic :meth:`recall`; the memory stays
                persistent). Omit to leave the current delivery mode unchanged.

        Returns:
            API response with updated memory info.
        """
        if not memory_id and not external_id:
            raise ValueError("Provide exactly one of memory_id or external_id")
        if memory_id and external_id:
            raise ValueError("Provide exactly one of memory_id or external_id")

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
        return await self._call_tool("update_memory", arguments)

    async def forget(
        self,
        context_id: str,
        memory_id: str | None = None,
        query: str | None = None,
        k: int = 10,
    ) -> dict[str, Any]:
        """
        Call forget MCP tool (soft delete memories).

        Delete by specific memory_id or by search query.
        Soft delete with 30-day retention.

        Args:
            context_id: Context ID
            memory_id: UUID of specific memory to delete
            query: Search query to find and delete matching memories
            k: Number of memories to delete in query mode (default: 10)

        Returns:
            API response with deletion results
        """
        arguments: dict[str, Any] = {"context_id": context_id}
        if memory_id:
            arguments["memory_id"] = memory_id
        if query:
            arguments["query"] = query
            arguments["k"] = k
        return await self._call_tool("forget", arguments)

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
            resource_id: Resource identifier for external data ingestion.
            is_private: Privacy flag (default: True).
            embedding_model: Embedding model for this context (immutable after creation).
                Use ``list_embedding_models()`` to discover available options.

        Returns:
            Created context dict with id, name, and metadata.

        Raises:
            KaguraQuotaError: Context limit reached for this workspace.
        """
        # Pre-check quota
        contexts = await self.list_contexts()
        if not contexts.get("can_create", True):
            from .exceptions import KaguraQuotaError

            raise KaguraQuotaError(
                f"Context limit reached ({contexts['count']}/{contexts['limit']}). "
                "Delete unused contexts or upgrade your plan."
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
        if resource_id is not None:
            arguments["resource_id"] = resource_id
        if embedding_model is not None:
            arguments["embedding_model"] = embedding_model
        return await self._call_tool("create_context", arguments)

    async def delete_context(self, context_id: str) -> dict[str, Any]:
        """Soft-delete a context and all its memories.

        Args:
            context_id: Context UUID to delete.

        Returns:
            API response with deletion confirmation.
        """
        return await self._call_tool("delete_context", {"context_id": context_id})

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
            is_locked: Lock/unlock context. Locked contexts cannot be deleted.

        Returns:
            Updated context dict.
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
        return await self._call_tool("update_context", arguments)

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
            name: Context name (defaults to ``resource_id`` server-side).
            summary: Context summary.
            description: Token description.
            quota_events_per_hour: Token quota (1-10000).

        Returns:
            Server response dict with keys: ``context_id``, ``context_name``,
            ``resource_id``, ``token`` (plaintext, shown once), ``token_id``,
            ``warning``. Server may include additional fields (e.g. ``status``,
            ``message``) which callers can ignore.

        Note:
            Idempotency for repeated calls with the same ``resource_id`` is
            not guaranteed by this SDK; server-side behavior may evolve.
        """
        arguments: dict[str, Any] = {
            "resource_id": resource_id,
            "quota_events_per_hour": quota_events_per_hour,
        }
        if name is not None:
            arguments["name"] = name
        if summary is not None:
            arguments["summary"] = summary
        if description is not None:
            arguments["description"] = description
        return await self._call_tool("setup_resource", arguments)

    async def merge_contexts(
        self,
        source_id: str,
        target_id: str,
        delete_source: bool = False,
    ) -> dict[str, Any]:
        """
        Merge memories from one context into another.

        Copies all memories from the source context to the target context.
        Both contexts must use the same embedding model and belong to the
        same workspace.

        Args:
            source_id: Context ID to copy memories from.
            target_id: Context ID to copy memories into.
            delete_source: If True, soft-delete the source context after merge.

        Returns:
            API response with merge results (e.g., merged count).

        Raises:
            ValueError: If source_id and target_id are the same.
        """
        if source_id == target_id:
            raise ValueError("source_id and target_id must be different")
        arguments: dict[str, Any] = {
            "source_id": source_id,
            "target_id": target_id,
        }
        if delete_source:
            arguments["delete_source"] = True
        return await self._call_tool("merge_contexts", arguments)

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
        result = await self._call_tool("list_edges", arguments)
        self._raise_for_mcp_error(result, "list_edges")
        return [Edge.model_validate(e) for e in result.get("edges", [])]

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
        result = await self._call_tool("create_edge", arguments)
        self._raise_for_mcp_error(result, "create_edge")
        return Edge.model_validate(result.get("edge", result))

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
        result = await self._call_tool("update_edge", arguments)
        self._raise_for_mcp_error(result, "update_edge")
        return Edge.model_validate(result.get("edge", result))

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
        result = await self._call_tool("delete_edge", arguments)
        self._raise_for_mcp_error(result, "delete_edge")
        return bool(result.get("deleted", True))

    async def get_usage(self) -> UsageInfo:
        """Get workspace usage and quota limits.

        Returns:
            UsageInfo with plan, memories, contexts, members, and MCP call limits.
        """
        result = await self._call_tool("get_usage", {})
        return UsageInfo.model_validate(result)

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
            ContextInfo with context metadata, search_config, stats, and instructions.
        """
        arguments: dict[str, Any] = {
            "context_id": context_id,
            "include_details": include_details,
        }
        result = await self._call_tool("get_context_info", arguments)
        return ContextInfo.model_validate(result)

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
            # could not be fetched. Catch broadly — not just KaguraError but
            # also a pydantic ValidationError from get_context_info's
            # model_validate on a malformed/changed server payload — log a
            # warning, and cache None so we degrade to steering=None without
            # re-fetching. (python.md permits a broad catch that logs.)
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

        Args:
            context_id: Context UUID.
            semantic_weight: Semantic search weight (0.0-1.0, server default 0.6).
            bm25_weight: BM25 keyword search weight (0.0-1.0, server default 0.4).
            fetch_factor: Candidate fetch multiplier (1-10, server default 3).
            use_rerank: Enable AI reranking.
            reranker_provider: Reranker provider ("voyage", "cohere", or "ollama").
            reranker_model: Reranker model name.

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
        return await self._call_tool("update_search_config", arguments)

    async def get_server_info(self) -> ServerInfo:
        """Get server name, version, environment, and feature flags.

        Calls ``GET /api/v1/system/info``.

        Returns:
            ServerInfo with version string and feature flags.
        """
        return await self._rest_get("/api/v1/system/info", ServerInfo)

    async def check_server_version(self) -> ServerInfo:
        """Check the connected server's version against the SDK's tested minimum.

        Advisory only — calls ``get_server_info()`` and logs a warning
        via :mod:`logging` when the server version is below
        :data:`MIN_SERVER_VERSION`. Does not raise. Older servers may
        silently ignore unknown parameters.

        Returns:
            ServerInfo from the server.
        """
        info = await self.get_server_info()
        try:
            server_ver = tuple(int(x) for x in info.version.split(".")[:3])
        except (ValueError, IndexError):
            return info
        if server_ver < _MIN_SERVER_VERSION_TUPLE:
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

        Calls ``GET /api/v1/workspace/embedding-status``.

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
    ) -> MemoryListResponse:
        """List memories newest-first, with optional substring and facet filters.

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

        Returns:
            :class:`MemoryListResponse` with ``memories`` (newest-first),
            ``total`` (matching rows across all pages), and ``has_more``.

        Raises:
            KaguraAuthError: Authentication failed.
            KaguraConnectionError: Network failure or non-2xx response — e.g. a
                ``context_id`` that does not exist or is not accessible
                surfaces as ``HTTP 404``.
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
        return await self._rest_get("/api/v1/memory/list", MemoryListResponse, params=params)

    @staticmethod
    def _raise_for_mcp_error(result: dict[str, Any], operation: str) -> None:
        """Translate an MCP tool's structured error response to an SDK exception.

        The server's MCP tools return ``{"status": "error", "error": <code>,
        "message": <str>, ...}`` for domain errors that the JSON-RPC transport
        layer cannot represent (e.g. ``report_not_found``). HTTP-level
        errors (401, 5xx) are already handled by ``_make_jsonrpc_request``.
        """
        if result.get("status") != "error":
            return
        code = result.get("error", "unknown")
        message = result.get("message", "Unknown error")
        if code in ("report_not_found", "context_not_found", "memory_not_found"):
            raise KaguraNotFoundError(f"{operation}: {message}")
        raise KaguraError(f"{operation} failed ({code}): {message}")

    async def get_sleep_history(
        self,
        context_id: str,
        limit: int = 10,
    ) -> list[SleepReport]:
        """List recent Sleep Maintenance runs for a context.

        Args:
            context_id: Context UUID.
            limit: Maximum number of runs to return (server clamps to 1-50,
                default: 10).

        Returns:
            List of ``SleepReport`` summaries ordered by ``started_at``
            descending (newest first).

        Raises:
            KaguraNotFoundError: Context not found.
            KaguraError: Other server-side error.
        """
        result = await self._call_tool(
            "get_sleep_history",
            {"context_id": context_id, "limit": limit},
        )
        self._raise_for_mcp_error(result, "get_sleep_history")
        return [SleepReport.model_validate(r) for r in result["reports"]]

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
            KaguraError: Other server-side error.
        """
        result = await self._call_tool(
            "get_sleep_report",
            {"context_id": context_id, "report_id": report_id},
        )
        self._raise_for_mcp_error(result, "get_sleep_report")
        # The MCP tool wraps the report fields under a "report" key;
        # flatten so SleepReportDetail (a SleepReport subclass) validates
        # naturally without forcing callers through an extra ``.report.``
        # accessor.
        return SleepReportDetail.model_validate(
            {
                **result["report"],
                "actions": result["actions"],
                "action_count": result["action_count"],
            }
        )

    async def rollback_sleep_run(
        self,
        context_id: str,
        report_id: str,
    ) -> RollbackResult:
        """Reverse the effects of a completed Sleep Maintenance run.

        Reverses edge creation, memory merges, importance updates, scope
        promotions, and archives. The server processes actions in reverse
        order with per-step commits — a 5xx or partial failure means SOME
        actions may have been reversed before the error surfaced.

        Args:
            context_id: Context UUID.
            report_id: Sleep report UUID.

        Returns:
            ``RollbackResult`` with per-category counts of reversed actions.

        Raises:
            KaguraNotFoundError: Report not found or not owned by caller.
            KaguraError: Partial rollback (some actions failed) or other
                server-side error. The exception message includes the
                server-side error code for triage.
        """
        result = await self._call_tool(
            "rollback_sleep_run",
            {"context_id": context_id, "report_id": report_id},
        )
        self._raise_for_mcp_error(result, "rollback_sleep_run")
        return RollbackResult.model_validate(result)

    async def list_embedding_models(self) -> EmbeddingModelsResponse:
        """List available embedding models.

        Calls ``GET /api/v1/system/embedding/models`` to retrieve
        server-supported embedding models with provider info and availability.

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
