"""REST API client for Kagura Memory Cloud Resource Tokens."""

from __future__ import annotations

from typing import Any, Literal

import httpx

from ._auth import _AuthSource, _OAuthAuth, _resolve_auth, _StaticAuth
from ._http import SDK_VERSION, base_url_from_mcp, extract_detail, validate_https_url
from .auth.credentials import KaguraOAuth
from .client import KaguraClient
from .exceptions import (
    KaguraAuthError,
    KaguraConnectionError,
    KaguraNotFoundError,
    KaguraQuotaError,
)
from .logger import VerboseLogger, normalize_logger
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

# Message surfaced when ``setup_resource`` is called on an OAuth-resolved
# client. Defined at module scope so the CLI test can assert against the
# stable constant instead of pinning a substring of the rendered output.
#
# The hint MUST point at a credential source that outranks OAuth in the
# resolver chain (``env > OAuth profile > .kagura.json``). Suggesting
# ``.kagura.json`` is misleading — it ranks BELOW OAuth and would still
# be skipped while the OAuth profile is present.
_SETUP_OAUTH_NOT_SUPPORTED_MSG = (
    "setup_resource() is not yet available in OAuth mode. "
    "To run it, switch to a credential source that outranks OAuth: "
    "set KAGURA_API_KEY (env wins over OAuth) and retry, or run "
    "`kagura auth logout` first to remove the OAuth profile so "
    ".kagura.json is consulted. The CRUD/ingest endpoints continue "
    "to work in OAuth mode."
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
        api_key: str | None = None,
        base_url: str = "https://memory.kagura-ai.com",
        timeout: float = 30.0,
        *,
        _oauth: KaguraOAuth | None = None,
        _auth_source: _AuthSource | None = None,
    ) -> None:
        """Initialize ResourceClient with a static API key.

        For OAuth profile resolution (the auto chain env → ``~/.kagura/
        credentials.json`` → ``.kagura.json``), use :meth:`from_mcp_url`
        which runs the resolver and selects the right transport.

        Args:
            api_key: Kagura API key (Bearer token). Required unless
                ``_oauth`` is supplied (see ``from_mcp_url``).
            base_url: REST API base URL (without path).
            timeout: Request timeout in seconds.
            _oauth: Private. ``from_mcp_url`` passes a ``KaguraOAuth``
                instance for OAuth profile authentication. Public
                callers should not set this.
            _auth_source: Private. Provenance tag describing which
                ``_resolve_auth`` branch produced the credentials.
                Stored for parity with :class:`FilesClient` so future
                surfaces (e.g. 403 hints) can reuse the convention.

        Raises:
            ValueError: If neither ``api_key`` nor ``_oauth`` is given.
                The OAuth path is intentionally not auto-resolved here
                so that bare ``ResourceClient()`` does not silently read
                ``~/.kagura/credentials.json`` — that's the
                ``from_mcp_url`` factory's job.
        """
        if api_key is None and _oauth is None:
            raise ValueError(
                "ResourceClient requires api_key, or use ResourceClient.from_mcp_url(...) "
                "to resolve credentials from environment, OAuth profile, or .kagura.json."
            )

        stripped_url = base_url.rstrip("/")
        validate_https_url(stripped_url, label="Base URL")

        self.base_url = stripped_url
        self._mcp_url: str | None = None  # Set by from_mcp_url for setup_resource
        # ``_oauth`` is stored because :meth:`setup_resource` branches on it
        # to refuse OAuth mode (constructing a KaguraClient from a static
        # api_key scraped out of the Authorization header would not work
        # when the header is absent in the OAuth path).
        self._oauth: KaguraOAuth | None = _oauth
        self._auth_source: _AuthSource | None = _auth_source
        if _oauth is not None:
            # OAuth path: KaguraOAuth injects a fresh access_token per request
            # and coordinates refresh via the process-wide credentials lock.
            self._client = httpx.AsyncClient(
                timeout=timeout,
                headers={"User-Agent": f"kagura-memory-sdk/{SDK_VERSION}"},
                auth=_oauth,
            )
        else:
            # Static path: bake the bearer header once. ``api_key`` is not
            # stored as an instance attribute (per python.md "Never store
            # API keys as instance attributes").
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
        api_key: str | None = None,
        mcp_url: str | None = None,
        timeout: float = 30.0,
        *,
        profile: str | None = None,
    ) -> ResourceClient:
        """Create ResourceClient by resolving credentials from the SDK chain.

        Runs :func:`_resolve_auth` with the same precedence chain as
        :class:`KaguraClient` and :class:`FilesClient`: explicit
        ``api_key`` > ``KAGURA_API_KEY`` env > OAuth profile from
        ``~/.kagura/credentials.json`` > ``.kagura.json``. The chosen
        credential source picks the transport: a static API key bakes
        the Bearer header once; an OAuth profile installs a
        ``KaguraOAuth`` httpx.Auth handler with automatic refresh.

        The REST ``base_url`` is derived from the resolved ``mcp_url``
        (strips ``/mcp`` and any ``/mcp/w/{workspace_id}`` suffix).

        Args:
            api_key: Explicit Kagura API key. Skips the resolution chain.
            mcp_url: Explicit MCP URL. When omitted, the OAuth profile's
                stored ``mcp_url`` is used (or the default).
            timeout: Request timeout in seconds.
            profile: Named OAuth profile to load (overrides
                ``KAGURA_PROFILE`` env and the credentials file's
                ``default_profile``).
        """
        resolved = _resolve_auth(api_key=api_key, mcp_url=mcp_url, profile=profile)
        return cls._from_resolved_auth(resolved, timeout=timeout)

    @classmethod
    def _from_resolved_auth(
        cls,
        resolved: _StaticAuth | _OAuthAuth,
        *,
        timeout: float = 30.0,
    ) -> ResourceClient:
        """Construct from a pre-resolved auth — internal CLI helper.

        Shared by :meth:`from_mcp_url` (SDK entry) and the CLI. Mirrors
        :meth:`FilesClient._from_resolved_auth` so the two REST clients
        share one construction shape; downstream code (including the
        CLI's ``_run_resource_command``) can treat them symmetrically.

        Stores ``mcp_url`` on the instance so :meth:`setup_resource` can
        reach the MCP endpoint — the resolved auth always carries one,
        whether sourced from the OAuth profile or the priority-4 config.
        """
        base_url = base_url_from_mcp(resolved.mcp_url.rstrip("/"))
        if isinstance(resolved, _StaticAuth):
            instance = cls(
                api_key=resolved.api_key,
                base_url=base_url,
                timeout=timeout,
                _auth_source=resolved.source,
            )
        else:
            instance = cls(
                base_url=base_url,
                timeout=timeout,
                _oauth=resolved.oauth,
                _auth_source="oauth",
            )
        instance._mcp_url = resolved.mcp_url.rstrip("/")
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
                hint = (
                    "Re-run `kagura auth login` or inspect ~/.kagura/credentials.json."
                    if self._oauth is not None
                    else "Check your API key."
                )
                raise KaguraAuthError(f"Authentication failed. {hint}") from e
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
            NotImplementedError: If the client was constructed via the
                OAuth resolution path. ``setup_resource`` currently
                requires a static api_key because the underlying MCP
                client construction does not yet accept the OAuth
                httpx.Auth instance. Track follow-up work to lift this.
            KaguraAuthError: If the stored Authorization header is
                missing or malformed in the static path.

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

        if self._oauth is not None:
            # ``KaguraClient`` does not yet accept the resolved
            # ``KaguraOAuth`` httpx.Auth, so the underlying MCP call
            # in this method cannot be authenticated without a static
            # api_key. Surface the workaround instead of failing with
            # a header-scraping error a few lines down.
            raise NotImplementedError(_SETUP_OAUTH_NOT_SUPPORTED_MSG)

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
        logger: VerboseLogger | None = None,
    ) -> ResourceEventBatchResponse:
        """Ingest a batch of resource events (max 100).

        Args:
            resource_id: Resource identifier.
            resource_api_key: Resource API key (X-Resource-API-Key header).
            events: List of events to ingest (1-100).
            logger: Optional :class:`VerboseLogger` for progress events.
                Honors the terminal-event contract — an unhandled
                exception still emits ``kind=error`` with partial
                progress in ``detail`` before propagating.

        Returns:
            Batch ingestion result with created/failed counts.
        """
        log = normalize_logger(logger)
        log.action(
            "Ingesting events batch",
            f"{len(events)} event(s) for resource {resource_id}",
            stage="ingest_events",
        )
        try:
            batch = ResourceEventBatchRequest(events=events)
            response = await self._request(
                "POST",
                f"/api/v1/resources/{resource_id}/events/batch",
                json=batch.model_dump(exclude_none=True),
                extra_headers={"X-Resource-API-Key": resource_api_key},
            )
            result = ResourceEventBatchResponse.model_validate(response.json())
        except BaseException as e:
            log.error(
                f"Batch ingest failed: {e}",
                stage="complete",
                detail={"events_attempted": len(events), "resource_id": resource_id},
            )
            raise
        log.success(
            "Batch ingested",
            stage="complete",
            detail={
                "created": result.created_count,
                "failed": result.failed_count,
            },
        )
        return result

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
