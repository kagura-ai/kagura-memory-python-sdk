"""Low-level REST API client for Kagura Memory Cloud."""

import itertools
import json
from typing import Any

import httpx

from ._http import SDK_VERSION, base_url_from_mcp, validate_https_url
from .exceptions import KaguraAuthError, KaguraConnectionError
from .models import EmbeddingModelsResponse


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
        api_key: str,
        mcp_url: str = "https://memory.kagura-ai.com/mcp",
        timeout: float = 30.0,
    ):
        """
        Initialize Kagura API client.

        Args:
            api_key: Kagura API key
            mcp_url: MCP server URL
            timeout: Request timeout in seconds
        """
        stripped_url = mcp_url.rstrip("/")
        validate_https_url(stripped_url, label="MCP URL")

        self.mcp_url = stripped_url
        self._base_url = base_url_from_mcp(stripped_url)
        self.timeout = timeout
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        self._session_id: str | None = None
        self._request_id_counter = itertools.count(1)

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
            raise KaguraConnectionError(f"HTTP {e.response.status_code}: {e}") from e
        except httpx.RequestError as e:
            raise KaguraConnectionError(f"Connection failed: {e}") from e

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
            raise KaguraConnectionError(f"Connection failed: {e}") from e

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
                raise KaguraConnectionError(f"Invalid response format: {e}") from e

        return {}

    async def remember(
        self,
        context_id: str,
        summary: str,
        content: str,
        type: str = "note",
        importance: float = 0.5,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Call remember MCP tool.

        Args:
            context_id: Context ID
            summary: Memory summary
            content: Memory content
            type: Memory type
            importance: Importance score (0.0-1.0)
            tags: Optional tags

        Returns:
            API response with memory_id
        """
        arguments = {
            "context_id": context_id,
            "summary": summary,
            "content": content,
            "type": type,
            "importance": importance,
        }
        if tags:
            arguments["tags"] = tags

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

        Returns:
            API response with results list

        Raises:
            ValueError: If neither ``context_id`` nor ``context_ids`` is provided,
                or ``context_ids`` has fewer than 2 or more than 20 IDs.
        """
        if not query:
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
        return await self._call_tool("recall", arguments)

    async def list_contexts(self) -> dict[str, Any]:
        """
        Call list_contexts MCP tool.

        Returns:
            API response with available contexts
        """
        return await self._call_tool("list_contexts", {})

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
            reranker_provider: Reranker provider ("voyage" or "cohere").
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

    async def list_embedding_models(self) -> EmbeddingModelsResponse:
        """List available embedding models.

        Calls ``GET /api/v1/system/embedding/models`` to retrieve
        server-supported embedding models with provider info and availability.

        Returns:
            EmbeddingModelsResponse with models list and default_model.

        Raises:
            KaguraAuthError: Authentication failed.
            KaguraConnectionError: Connection to server failed.
        """
        url = f"{self._base_url}/api/v1/system/embedding/models"
        try:
            response = await self._client.get(url)
            response.raise_for_status()
            return EmbeddingModelsResponse.model_validate(response.json())
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise KaguraAuthError("Authentication failed. Check your API key.") from e
            raise KaguraConnectionError(f"HTTP {e.response.status_code}") from e
        except httpx.RequestError as e:
            raise KaguraConnectionError(f"Connection failed: {e}") from e
        except (ValueError, TypeError) as e:
            raise KaguraConnectionError(f"Invalid response format: {e}") from e

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
