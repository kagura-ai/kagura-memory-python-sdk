"""REST API client for Kagura Memory Cloud Resource Tokens."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from ._auth import _AuthSource, _OAuthAuth, _StaticAuth
from ._rest_base import KaguraRestClient
from .auth.credentials import KaguraOAuth
from .client import KaguraClient
from .exceptions import (
    KaguraAuthError,
    KaguraNotFoundError,
)
from .logger import VerboseLogger, normalize_logger
from .models import (
    IndexerStatusResponse,
    PaginatedResourceTokensResponse,
    ResourceEventBatchRequest,
    ResourceEventBatchResponse,
    ResourceEventRequest,
    ResourceEventResponse,
    ResourceEventsListResponse,
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
    "set KAGURA_API_KEY (and KAGURA_MCP_URL too if your api_key is "
    "not for the default cloud server — the env branch uses "
    "KAGURA_MCP_URL or the default URL, not the OAuth profile's "
    "stored mcp_url) and retry, or remove the active OAuth profile "
    "(e.g. `kagura auth logout`, or `kagura auth logout --profile <name>` "
    "for named profiles selected via KAGURA_PROFILE) so .kagura.json "
    "is consulted. The CRUD/ingest endpoints continue to work in "
    "OAuth mode."
)


class ResourceClient(KaguraRestClient):
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
        _workspace_id_hint: str | None = None,
    ) -> None:
        """Initialize ResourceClient (see :class:`KaguraRestClient`).

        Adds ``_mcp_url`` (None until :meth:`_from_resolved_auth` stamps
        it) — :meth:`setup_resource` needs the ORIGINAL MCP URL, which the
        REST ``base_url`` no longer carries.
        """
        super().__init__(
            api_key,
            base_url,
            timeout,
            _oauth=_oauth,
            _auth_source=_auth_source,
            _workspace_id_hint=_workspace_id_hint,
        )
        self._mcp_url: str | None = None

    @classmethod
    def _from_resolved_auth(
        cls,
        resolved: _StaticAuth | _OAuthAuth,
        *,
        timeout: float = 30.0,
        workspace_id_hint: str | None = None,
    ) -> ResourceClient:
        """Construct from a pre-resolved auth, stamping ``_mcp_url``.

        ``setup_resource()`` needs the ORIGINAL MCP URL to build its
        MCP session — the resolved auth always carries one, whether
        sourced from the OAuth profile or the priority-4 config.
        """
        instance = super()._from_resolved_auth(
            resolved, timeout=timeout, workspace_id_hint=workspace_id_hint
        )
        instance._mcp_url = resolved.mcp_url.rstrip("/")
        return instance

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

    async def list_resource_events(
        self,
        resource_id: str,
        *,
        limit: int = 50,
        cursor: str | None = None,
        op: Literal["upsert", "delete"] | None = None,
        doc_id: str | None = None,
        version: int | None = None,
        since: datetime | None = None,
    ) -> ResourceEventsListResponse:
        """List ingested events for a resource (server v0.15+).

        Mirrors ``GET /api/v1/resources/{resource_id}/events``. Results are
        cursor-paginated: pass the returned ``next_cursor`` back as
        ``cursor`` to fetch the next page (``next_cursor`` is ``None`` on
        the last page). Note this method uses ``limit``/``cursor``
        pagination, unlike :meth:`list_tokens` which uses ``limit``/``offset``.

        Uses Bearer auth (workspace read), not the ``X-Resource-API-Key``
        ingestion credential.

        Args:
            resource_id: Resource identifier slug.
            limit: Maximum events per page (1-100).
            cursor: Opaque pagination cursor from a prior ``next_cursor``.
            op: Filter by operation (``upsert`` or ``delete``).
            doc_id: Filter by document ID.
            version: Filter by document version.
            since: Return only events with ``created_at`` at or after this
                time (inclusive). Serialized to ISO 8601; a tz-naive value
                is assumed to be UTC.

        Returns:
            A page of :class:`ResourceEventRecord` plus ``next_cursor``. An
            unknown ``resource_id`` returns an empty page (``events=[]``,
            ``next_cursor=None``) rather than a 404 — observed against
            memory-cloud production.

        Raises:
            KaguraNotFoundError: The server returned 404 for this resource
                (cross-workspace probe protection). This endpoint currently
                returns an empty page for an unknown slug, so this path
                applies only if the server does respond with 404.
        """
        params: dict[str, Any] = {"limit": limit}
        if cursor is not None:
            params["cursor"] = cursor
        if op is not None:
            params["op"] = op
        if doc_id is not None:
            params["doc_id"] = doc_id
        if version is not None:
            params["version"] = version
        if since is not None:
            if since.tzinfo is None:
                since = since.replace(tzinfo=UTC)
            params["since"] = since.isoformat()

        response = await self._request(
            "GET", f"/api/v1/resources/{resource_id}/events", params=params
        )
        return ResourceEventsListResponse.model_validate(response.json())

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
