"""Low-level REST API client for Kagura Memory Cloud."""

import itertools
import json
from typing import Any

import httpx

from importlib.metadata import version as _pkg_version

from .exceptions import KaguraAuthError, KaguraConnectionError

_VERSION = _pkg_version("kagura-memory")


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
        # C-3: Enforce HTTPS except for localhost development
        stripped_url = mcp_url.rstrip("/")
        if stripped_url.startswith("http://") and not any(
            stripped_url.startswith(f"http://{h}") for h in ("localhost", "127.0.0.1", "[::1]")
        ):
            raise ValueError(
                f"MCP URL must use HTTPS for security (got: {stripped_url}). "
                "HTTP is only allowed for localhost development."
            )

        # C-1: Don't store api_key as instance attribute — only in httpx headers
        self.mcp_url = stripped_url
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
                "clientInfo": {"name": "kagura-memory-sdk", "version": _VERSION},
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

            data = response.json()
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
        if content and len(content) > 0:
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
        context_id: str,
        query: str,
        k: int = 5,
        use_rerank: bool = False,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Call recall MCP tool.

        Args:
            context_id: Context ID
            query: Search query
            k: Number of results
            use_rerank: Enable AI reranking for higher quality results
            filters: Optional filters (e.g., {"type": "code"}, {"tags": ["python"]})

        Returns:
            API response with results list
        """
        arguments: dict[str, Any] = {
            "context_id": context_id,
            "query": query,
            "k": k,
        }
        if use_rerank:
            arguments["use_rerank"] = True
        if filters:
            arguments["filters"] = filters
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
            API response with complete memory data
        """
        arguments = {
            "context_id": context_id,
            "memory_id": memory_id,
        }
        return await self._call_tool("reference", arguments)

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

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> "KaguraClient":
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit."""
        await self.close()
