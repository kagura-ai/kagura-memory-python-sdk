"""REST API client for Kagura Memory Cloud Resource Tokens."""

from __future__ import annotations

from typing import Any, Literal

import httpx

from ._http import SDK_VERSION, validate_https_url
from .client import KaguraClient
from .exceptions import (
    KaguraAuthError,
    KaguraConnectionError,
    KaguraNotFoundError,
    KaguraQuotaError,
)
from .models import (
    PaginatedResourceTokensResponse,
    ResourceEventBatchRequest,
    ResourceEventBatchResponse,
    ResourceEventRequest,
    ResourceEventResponse,
    ResourceImpactResponse,
    ResourceSchemaResponse,
    ResourceTokenCreate,
    ResourceTokenCreateResponse,
    ResourceTokenResponse,
    ResourceTokenUpdate,
)


class ResourceClient:
    """REST API client for Kagura Memory Cloud Resource Tokens.

    Handles two authentication modes:
    - Token CRUD: ``Authorization: Bearer <api_key>`` (set once in constructor)
    - Event ingestion: ``X-Resource-API-Key`` (passed per-call, never stored)

    All methods may raise:
        KaguraAuthError: Authentication failed (401)
        KaguraConnectionError: Connection or HTTP error
        KaguraQuotaError: Quota exceeded (429)
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://memory.kagura-ai.com",
        timeout: float = 30.0,
    ) -> None:
        """Initialize ResourceClient.

        Args:
            api_key: Kagura API key (Bearer token for CRUD operations).
            base_url: REST API base URL (without path).
            timeout: Request timeout in seconds.
        """
        stripped_url = base_url.rstrip("/")
        validate_https_url(stripped_url, label="Base URL")

        self.base_url = stripped_url
        self._mcp_url: str | None = None  # Set by from_mcp_url for setup_resource
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "User-Agent": f"kagura-memory-sdk/{SDK_VERSION}",
            },
        )

    @classmethod
    def from_mcp_url(
        cls,
        api_key: str,
        mcp_url: str = "https://memory.kagura-ai.com/mcp",
        timeout: float = 30.0,
    ) -> ResourceClient:
        """Create ResourceClient by deriving base URL from MCP URL.

        Strips ``/mcp`` and everything after it to get the REST API base URL.
        Handles both ``/mcp`` and ``/mcp/w/{workspace_id}`` formats.

        Args:
            api_key: Kagura API key.
            mcp_url: MCP server URL (e.g. ``https://memory.kagura-ai.com/mcp``).
            timeout: Request timeout in seconds.
        """
        url = mcp_url.rstrip("/")
        mcp_idx = url.find("/mcp")
        base_url = url[:mcp_idx] if mcp_idx != -1 else url
        instance = cls(api_key=api_key, base_url=base_url, timeout=timeout)
        instance._mcp_url = mcp_url.rstrip("/")
        return instance

    async def _request(
        self,
        method: Literal["GET", "POST", "PATCH", "DELETE"],
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        """Make an HTTP request with standard error handling.

        Args:
            method: HTTP method (GET, POST, PATCH, DELETE).
            path: API path (e.g. ``/api/v1/resource-tokens``).
            json: Request body.
            params: Query parameters.
            extra_headers: Per-request headers (e.g. X-Resource-API-Key).

        Returns:
            httpx.Response object.

        Raises:
            KaguraAuthError: On 401 responses.
            KaguraQuotaError: On 429 responses.
            KaguraConnectionError: On other HTTP/network errors.
        """
        url = f"{self.base_url}{path}"
        try:
            response = await self._client.request(
                method,
                url,
                json=json,
                params=params,
                headers=extra_headers,
            )
            response.raise_for_status()
            return response

        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status == 401:
                raise KaguraAuthError("Authentication failed. Check your API key.") from e
            if status == 404:
                detail = ""
                try:
                    body = e.response.json()
                    detail = body.get("detail", "") if isinstance(body, dict) else ""
                except (ValueError, UnicodeDecodeError):
                    pass
                raise KaguraNotFoundError(detail or "Not found") from e
            if status == 429:
                retry_after = e.response.headers.get("Retry-After")
                raise KaguraQuotaError(
                    "Quota exceeded. Try again later.",
                    retry_after=int(retry_after) if retry_after else None,
                ) from e
            # Try to extract detail from JSON response
            detail = ""
            try:
                body = e.response.json()
                detail = body.get("detail", "") if isinstance(body, dict) else ""
            except (ValueError, UnicodeDecodeError):
                pass
            msg = f"HTTP {status}"
            if detail:
                msg = f"{msg}: {detail}"
            raise KaguraConnectionError(msg) from e
        except httpx.RequestError as e:
            raise KaguraConnectionError(f"Connection failed: {e}") from e

    # -------------------------------------------------------------------
    # Token CRUD (Bearer auth)
    # -------------------------------------------------------------------

    async def create_token(
        self,
        resource_id: str,
        description: str | None = None,
        quota_events_per_hour: int = 1000,
    ) -> ResourceTokenCreateResponse:
        """Create a new resource token.

        Args:
            resource_id: Resource identifier this token is scoped to.
            description: Human-readable description.
            quota_events_per_hour: Event ingestion quota per hour (1-10000).

        Returns:
            Created token including plaintext token (shown only once).
        """
        body = ResourceTokenCreate(
            resource_id=resource_id,
            description=description,
            quota_events_per_hour=quota_events_per_hour,
        ).model_dump(exclude_none=True)

        response = await self._request("POST", "/api/v1/resource-tokens", json=body)
        return ResourceTokenCreateResponse.model_validate(response.json())

    async def list_tokens(
        self,
        resource_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> PaginatedResourceTokensResponse:
        """List resource tokens with optional filtering.

        Args:
            resource_id: Filter by resource ID.
            limit: Number of tokens per page (1-100).
            offset: Starting offset for pagination.

        Returns:
            Paginated list of resource tokens.
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if resource_id is not None:
            params["resource_id"] = resource_id

        response = await self._request("GET", "/api/v1/resource-tokens", params=params)
        return PaginatedResourceTokensResponse.model_validate(response.json())

    async def update_token(
        self,
        token_id: int,
        description: str | None = None,
        quota_events_per_hour: int | None = None,
    ) -> ResourceTokenResponse:
        """Update a resource token's metadata.

        Args:
            token_id: Token database ID.
            description: Updated description.
            quota_events_per_hour: Updated quota (1-10000).

        Returns:
            Updated token metadata.
        """
        body = ResourceTokenUpdate(
            description=description,
            quota_events_per_hour=quota_events_per_hour,
        ).model_dump(exclude_none=True)

        response = await self._request("PATCH", f"/api/v1/resource-tokens/{token_id}", json=body)
        return ResourceTokenResponse.model_validate(response.json())

    async def revoke_token(self, token_id: int) -> None:
        """Revoke (soft-delete) a resource token.

        Args:
            token_id: Token database ID.
        """
        await self._request("DELETE", f"/api/v1/resource-tokens/{token_id}")

    # -------------------------------------------------------------------
    # Setup helper
    # -------------------------------------------------------------------

    async def setup_resource(
        self,
        resource_id: str,
        context_name: str | None = None,
        summary: str | None = None,
        description: str | None = None,
        quota_events_per_hour: int = 1000,
    ) -> ResourceTokenCreateResponse:
        """Set up a resource for data ingestion in one call.

        Creates a public context, sets its resource_id, and creates a
        resource token. Requires the client to be created via ``from_mcp_url()``.

        Args:
            resource_id: Resource identifier for data ingestion.
            context_name: Context name (defaults to resource_id).
            summary: Context summary.
            description: Token description.
            quota_events_per_hour: Token quota (1-10000).

        Returns:
            ResourceTokenCreateResponse with token (shown only once).

        Raises:
            RuntimeError: If client was not created via ``from_mcp_url()``.
        """
        if not self._mcp_url:
            raise RuntimeError(
                "setup_resource() requires MCP URL. "
                "Create client via ResourceClient.from_mcp_url()."
            )

        # Extract API key from httpx Authorization header
        auth = self._client.headers.get("authorization", "")
        if not auth.startswith("Bearer "):
            raise ValueError("Authorization header missing or invalid")
        api_key = auth[7:]

        name = context_name or resource_id

        async with KaguraClient(api_key=api_key, mcp_url=self._mcp_url) as mcp:
            ctx = await mcp.create_context(name=name, summary=summary, is_private=False)
            ctx_id = ctx["context_id"]
            await mcp.update_context(context_id=ctx_id, resource_id=resource_id)

        return await self.create_token(
            resource_id=resource_id,
            description=description,
            quota_events_per_hour=quota_events_per_hour,
        )

    # -------------------------------------------------------------------
    # Resource Stats (Bearer auth)
    # -------------------------------------------------------------------

    async def get_resource_impact(
        self,
        resource_id: str,
    ) -> ResourceImpactResponse:
        """Get impact statistics for a resource.

        Args:
            resource_id: Resource identifier.

        Returns:
            Resource impact stats (token_count, memory_count, schema version).
        """
        response = await self._request("GET", f"/api/v1/resources/{resource_id}/impact")
        return ResourceImpactResponse.model_validate(response.json())

    async def get_resource_schema(
        self,
        resource_id: str,
        schema_version: int | None = None,
    ) -> ResourceSchemaResponse | None:
        """Get field definitions for a resource.

        Args:
            resource_id: Resource identifier.
            schema_version: Specific schema version to retrieve.
                Omit for latest version.

        Returns:
            Resource schema with field definitions, or None if no schema
            is registered for the resource.
        """
        params: dict[str, Any] | None = None
        if schema_version is not None:
            params = {"schema_version": schema_version}
        try:
            response = await self._request(
                "GET", f"/api/v1/resources/{resource_id}/schema", params=params
            )
        except KaguraNotFoundError:
            return None
        return ResourceSchemaResponse.model_validate(response.json())

    # -------------------------------------------------------------------
    # Event Ingestion (X-Resource-API-Key auth)
    # -------------------------------------------------------------------

    async def ingest_event(
        self,
        resource_id: str,
        resource_api_key: str,
        event: ResourceEventRequest,
    ) -> ResourceEventResponse:
        """Ingest a single resource event.

        Args:
            resource_id: Resource identifier.
            resource_api_key: Resource API key (X-Resource-API-Key header).
            event: Event data to ingest.

        Returns:
            Ingestion result with event_id.
        """
        response = await self._request(
            "POST",
            f"/api/v1/resources/{resource_id}/events",
            json=event.model_dump(exclude_none=True),
            extra_headers={"X-Resource-API-Key": resource_api_key},
        )
        return ResourceEventResponse.model_validate(response.json())

    async def ingest_events(
        self,
        resource_id: str,
        resource_api_key: str,
        events: list[ResourceEventRequest],
    ) -> ResourceEventBatchResponse:
        """Ingest a batch of resource events (max 100).

        Args:
            resource_id: Resource identifier.
            resource_api_key: Resource API key (X-Resource-API-Key header).
            events: List of events to ingest (1-100).

        Returns:
            Batch ingestion result with created/failed counts.
        """
        batch = ResourceEventBatchRequest(events=events)
        response = await self._request(
            "POST",
            f"/api/v1/resources/{resource_id}/events/batch",
            json=batch.model_dump(exclude_none=True),
            extra_headers={"X-Resource-API-Key": resource_api_key},
        )
        return ResourceEventBatchResponse.model_validate(response.json())

    # -------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> ResourceClient:
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
