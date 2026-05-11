"""REST API client for Kagura Memory Cloud Resource Tokens."""

from __future__ import annotations

from typing import Any, Literal

import httpx

from ._http import SDK_VERSION, base_url_from_mcp, extract_detail, validate_https_url
from .client import KaguraClient
from .exceptions import (
    KaguraAuthError,
    KaguraConnectionError,
    KaguraNotFoundError,
    KaguraQuotaError,
)
from .models import (
    IndexerStatusResponse,
    PaginatedResourceTokensResponse,
    ResourceEventBatchRequest,
    ResourceEventBatchResponse,
    ResourceEventRequest,
    ResourceEventResponse,
    ResourceImpactResponse,
    ResourceListResponse,
    ResourceSchemaResponse,
    ResourceSetupResponse,
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
        KaguraNotFoundError: Resource not found (404)
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
        base_url = base_url_from_mcp(url)
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
                raise KaguraNotFoundError(extract_detail(e.response) or "Not found") from e
            if status == 429:
                retry_after = e.response.headers.get("Retry-After")
                raise KaguraQuotaError(
                    "Quota exceeded. Try again later.",
                    retry_after=int(retry_after) if retry_after else None,
                ) from e
            detail = extract_detail(e.response)
            msg = f"HTTP {status}: {detail}" if detail else f"HTTP {status}"
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
    ) -> ResourceSetupResponse:
        """Atomically set up a resource for data ingestion (server v0.14+).

        Calls the server-side atomic ``setup_resource`` MCP tool which creates
        Context + Resource entity + token in a single transaction. On failure,
        no orphan Context rows are left on the server. Requires the client to
        be created via ``from_mcp_url()``.

        Args:
            resource_id: Resource identifier for data ingestion.
            context_name: Context name (defaults to resource_id).
            summary: Context summary.
            description: Token description.
            quota_events_per_hour: Token quota (1-10000).

        Returns:
            ResourceSetupResponse with plaintext token (shown only once).

        Raises:
            RuntimeError: If client was not created via ``from_mcp_url()``.
            KaguraAuthError: If the stored Authorization header is missing or malformed.

        Note:
            Idempotency for repeated calls with the same ``resource_id`` is
            not guaranteed; server-side behavior may evolve. Avoid retrying
            setup for an existing ``resource_id`` without first verifying state.
        """
        if not self._mcp_url:
            raise RuntimeError(
                "setup_resource() requires MCP URL. "
                "Create client via ResourceClient.from_mcp_url()."
            )

        auth = self._client.headers.get("authorization", "")
        if not auth.startswith("Bearer "):
            raise KaguraAuthError("Authorization header missing or invalid")
        api_key = auth[7:]

        async with KaguraClient(api_key=api_key, mcp_url=self._mcp_url) as mcp:
            response = await mcp.setup_resource(
                resource_id=resource_id,
                name=context_name,
                summary=summary,
                description=description,
                quota_events_per_hour=quota_events_per_hour,
            )
        return ResourceSetupResponse.model_validate(response)

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

    async def list_resources(self) -> ResourceListResponse:
        """List all resources in the caller's workspace (server v0.14+).

        Returns workspace-scoped resources with aggregated stats
        (token_count, memory_count, current_schema_version, ...). The
        endpoint is currently non-paginated; the server caps a workspace
        at well under 50 resources by design.

        Workspace owners only — non-owners receive 403.

        Returns:
            ResourceListResponse with ``resources`` and ``total`` count.
        """
        response = await self._request("GET", "/api/v1/resources")
        return ResourceListResponse.model_validate(response.json())

    async def get_indexer_status(self, resource_id: str) -> IndexerStatusResponse:
        """Get indexer state and recent ingest events for a resource.

        Args:
            resource_id: Resource identifier slug.

        Returns:
            IndexerStatusResponse. ``state`` is ``None`` when the indexer
            has never run for this resource — this is a normal 200 response,
            distinct from a 404. ``recent_events`` is server-capped at 5.

        Raises:
            KaguraNotFoundError: Resource slug does not exist in the caller's
                workspace (404; cross-workspace probe protection).
        """
        response = await self._request("GET", f"/api/v1/resources/{resource_id}/indexer-status")
        return IndexerStatusResponse.model_validate(response.json())

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
