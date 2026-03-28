"""REST API client for Kagura Memory Cloud Resource Tokens."""

from __future__ import annotations

from importlib.metadata import version as _pkg_version
from typing import Any

import httpx

from .exceptions import (
    KaguraAuthError,
    KaguraConnectionError,
    KaguraQuotaError,
)
from .models import (
    PaginatedResourceTokensResponse,
    ResourceEventBatchRequest,
    ResourceEventBatchResponse,
    ResourceEventRequest,
    ResourceEventResponse,
    ResourceTokenCreateResponse,
    ResourceTokenResponse,
)

_VERSION = _pkg_version("kagura-memory")


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
        self._validate_url(stripped_url)

        # C-1: Don't store api_key as instance attribute — only in httpx headers
        self.base_url = stripped_url
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "User-Agent": f"kagura-memory-sdk/{_VERSION}",
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

        Strips ``/mcp`` suffix to get the REST API base URL.

        Args:
            api_key: Kagura API key.
            mcp_url: MCP server URL (e.g. ``https://memory.kagura-ai.com/mcp``).
            timeout: Request timeout in seconds.
        """
        base_url = mcp_url.rstrip("/").removesuffix("/mcp")
        return cls(api_key=api_key, base_url=base_url, timeout=timeout)

    @staticmethod
    def _validate_url(url: str) -> None:
        """C-3: Enforce HTTPS except for localhost development."""
        if url.startswith("http://") and not any(
            url.startswith(f"http://{h}") for h in ("localhost", "127.0.0.1", "[::1]")
        ):
            raise ValueError(
                f"Base URL must use HTTPS for security (got: {url}). "
                "HTTP is only allowed for localhost development."
            )

    async def _request(
        self,
        method: str,
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
                detail = body.get("detail", "")
            except Exception:
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
        body: dict[str, Any] = {"resource_id": resource_id}
        if description is not None:
            body["description"] = description
        if quota_events_per_hour != 1000:
            body["quota_events_per_hour"] = quota_events_per_hour

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
        body: dict[str, Any] = {}
        if description is not None:
            body["description"] = description
        if quota_events_per_hour is not None:
            body["quota_events_per_hour"] = quota_events_per_hour

        response = await self._request(
            "PATCH", f"/api/v1/resource-tokens/{token_id}", json=body
        )
        return ResourceTokenResponse.model_validate(response.json())

    async def revoke_token(self, token_id: int) -> None:
        """Revoke (soft-delete) a resource token.

        Args:
            token_id: Token database ID.
        """
        await self._request("DELETE", f"/api/v1/resource-tokens/{token_id}")

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
