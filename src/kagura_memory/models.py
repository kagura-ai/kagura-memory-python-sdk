"""Pydantic models for Kagura Memory SDK."""

import re
from datetime import datetime
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

_M = TypeVar("_M", bound=BaseModel)

# ---------------------------------------------------------------------------
# Embedding model metadata
# ---------------------------------------------------------------------------


class EmbeddingModel(BaseModel):
    """An embedding model available on the server."""

    name: str
    dimensions: int
    provider: str
    available: bool


class EmbeddingModelsResponse(BaseModel):
    """Response from the embedding models endpoint."""

    models: list[EmbeddingModel]
    default_model: str


# ---------------------------------------------------------------------------
# Server info & usage models (v0.6.1)
# ---------------------------------------------------------------------------


class ServerFeatures(BaseModel):
    """Deployment feature flags from the ``features`` block of ``/api/v1/system/info``.

    The flags describe what the deployment offers, not what the caller's plan
    allows. Every flag defaults to ``False``, which also stands for "not
    reported": a server older than a flag omits it. Flags newer than this SDK
    are kept, not dropped — read them from ``model_extra``. The typed flags are
    the ones memory-cloud v0.141.0 sends.
    """

    model_config = ConfigDict(extra="allow")

    neural_memory: bool = False
    research_tools: bool = False
    # The web UI shows the Plan page (memory-cloud #1145; off by default for OSS).
    plan_page: bool = False
    # Workspace BYOK keys and the cost dashboard are available (#1167; on by default).
    byok: bool = False
    # The web UI shows money: cost page, analysis cost (#1571; on by default).
    cost_display: bool = False
    # Connectors run on the managed worker, no per-connector LLM needed (#1426).
    managed_connectors: bool = False
    # Memory Analysis can run without a workspace OpenAI key (#1569).
    managed_llm: bool = False
    # The referral program is available (#1470).
    referrals: bool = False
    # Closed-beta invite links are available (#1581); ``/beta-invites`` 404s when off.
    beta_invites: bool = False
    # Reranking is enabled and its default provider can run (#1572, v0.69.0+).
    reranking: bool = False
    # Content chunk vectors are a plan feature with a per-plan context cap
    # (CONTENT_CHUNKS_REQUIRE_PLAN; #2103, v0.113.0+; off by default).
    content_chunks_require_plan: bool = False
    # The recall LLM selector is a plan feature with a per-plan context cap
    # (LLM_SELECTOR_REQUIRE_PLAN; #2144, v0.115.0+; off by default).
    llm_selector_require_plan: bool = False


class ServerInfo(BaseModel):
    """Server information from /api/v1/system/info.

    ``search_defaults`` (memory-cloud v0.69.0+, #1572) holds the reranker
    settings a new context starts with — ``use_rerank``,
    ``reranker_provider`` and ``reranker_model`` (names only, never URLs or
    keys). It is ``None`` on older servers. From v0.112.0 (#2015) it also
    carries ``content_chunks_enabled``, the deployment's
    ``DEFAULT_CONTENT_CHUNKS_ENABLED``: whether a **new** context embeds its
    memories' content as chunks, not only their summaries (existing contexts
    keep their own setting). It stays a plain mapping, so read it with
    ``search_defaults.get("content_chunks_enabled")``.

    ``terms_version`` (memory-cloud v0.77.0+, #1665) is the current
    terms-of-service version. It is ``None`` when the deployment does not
    record acceptance, or on an older server. Acceptance is a web sign-in
    step and never gates API or MCP calls.
    """

    name: str
    version: str
    description: str | None = None
    environment: str | None = None
    search_defaults: dict[str, Any] | None = None
    terms_version: str | None = None
    features: ServerFeatures = Field(default_factory=ServerFeatures)


class UsageQuota(BaseModel):
    """Usage vs limit for a single resource category."""

    used: int
    limit: int
    percentage: float | None = None


class UsageQuotaLimitOnly(BaseModel):
    """Quota with limit only (no usage counter)."""

    limit: int


class CapacityLock(BaseModel):
    """How far a Free workspace is over its capacity (memory-cloud v0.99.0+, #1941).

    While the workspace is locked, search and saving raise
    :class:`~kagura_memory.exceptions.KaguraCapacityLockedError`; listing,
    deleting, export and usage keep working. ``over_*`` is 0 on an axis that
    is within its limit. Every field is optional so a reply that leaves one
    out still parses.

    Attributes:
        memory_count: Live memories in the workspace.
        memory_limit: The effective memory limit.
        over_memories: Memories to remove to unlock.
        used_bytes: File bytes the owner can still delete.
        storage_limit_bytes: The effective storage limit in bytes.
        over_bytes: File bytes to remove to unlock.
        cleanup_url: The web page to clean up or re-subscribe.
        help: The server's next step: how much to remove and what still works
            (MCP ``get_usage``).
        purge_at: When the workspace's data is deleted if it is still over
            capacity, in UTC (REST ``GET /api/v1/workspace/usage/current``
            only). ``None`` until the daily sweep has seen it and while the
            deployment cannot purge; it may move later (v0.126.0, #1971).
    """

    memory_count: int | None = None
    memory_limit: int | None = None
    over_memories: int | None = None
    used_bytes: int | None = None
    storage_limit_bytes: int | None = None
    over_bytes: int | None = None
    cleanup_url: str | None = None
    help: str | None = None
    purge_at: datetime | None = None


class UsageInfo(BaseModel):
    """Workspace usage and quota information.

    ``memories.used`` counts the live rows of this workspace since
    memory-cloud v0.100.0 (#1946). ``capacity_lock`` is ``None`` unless the
    workspace is over its Free capacity (memory-cloud v0.99.0+), and always
    on older servers.
    """

    plan: str
    memories: UsageQuota
    contexts: UsageQuota
    members: UsageQuota
    mcp_calls_per_day: UsageQuotaLimitOnly
    capacity_lock: CapacityLock | None = None


class SearchConfig(BaseModel):
    """Hybrid search configuration for a context.

    The content-chunk and LLM-selector fields are ``None`` when the reply did
    not carry them — a server older than the field, or a surface that does
    not send it. At memory-cloud v0.142.0 :meth:`KaguraClient.get_context_info`
    sends ``llm_selector_enabled`` / ``llm_selector_effective`` only;
    :meth:`KaguraClient.update_search_config` returns both ``*_enabled`` /
    ``*_effective`` pairs in its ``config``; the REST ``GET
    /api/v1/contexts/{id}/search-config`` route (not wrapped) carries every
    field. Fields newer than this SDK are ignored.

    Attributes:
        content_chunks_enabled: The stored opt-in (memory-cloud v0.110.0+,
            #2015): long ``content`` (1,500+ characters) is embedded as chunk
            vectors, so memory content — not only summaries — goes to the
            embedding and rerank providers.
        content_chunks_effective: Whether the stored flag is acted on: a
            vector store that keeps chunks (not LanceDB) and, where the
            deployment gates chunks by plan, a plan slot (v0.113.0+, #2103).
        content_chunks_limit: The plan's ``max_content_chunk_contexts``;
            ``None`` when the deployment does not gate by plan.
        content_chunks_in_use: Contexts of the workspace with the flag on.
        content_chunks_disabled_reason: Why the toggle cannot be turned on,
            or the chunks are inactive: ``unsupported``, ``plan``, ``cap`` or
            ``permission`` (a plain ``str``, so a newer reason passes).
        content_chunks_can_change: Whether the caller may change the flag.
        content_chunks_supported: Whether the vector store keeps chunk
            vectors (``False`` on LanceDB).
        content_chunks_backfill_pending: Long memories still waiting for
            their chunks (v0.112.0+); 0 while the flag is off.
        embedding_provider: Where content goes with chunks on (``openai`` /
            ``self_hosted``).
        llm_selector_enabled: The stored recall LLM-selector opt-in
            (v0.115.0+, #2144): an MCP :meth:`KaguraClient.recall` may send
            the context's summaries and attached content to the deployment's
            judge model.
        llm_selector_effective: Whether recall acts on it (the plan, where
            the deployment gates it).
        llm_selector_limit: The plan's ``max_llm_selector_contexts``;
            ``None`` when not gated by plan.
        llm_selector_in_use: Contexts of the workspace with the selector on.
        llm_selector_disabled_reason: ``plan``, ``cap`` or ``permission``.
        llm_selector_can_change: Whether the caller may change it.
    """

    model_config = ConfigDict(extra="ignore")

    semantic_weight: float = 0.6
    bm25_weight: float = 0.4
    fetch_factor: int = 3
    use_rerank: bool = False
    reranker_provider: str | None = None
    reranker_model: str | None = None
    content_chunks_enabled: bool | None = None
    content_chunks_effective: bool | None = None
    content_chunks_limit: int | None = None
    content_chunks_in_use: int | None = None
    content_chunks_disabled_reason: str | None = None
    content_chunks_can_change: bool | None = None
    content_chunks_supported: bool | None = None
    content_chunks_backfill_pending: int | None = None
    embedding_provider: str | None = None
    llm_selector_enabled: bool | None = None
    llm_selector_effective: bool | None = None
    llm_selector_limit: int | None = None
    llm_selector_in_use: int | None = None
    llm_selector_disabled_reason: str | None = None
    llm_selector_can_change: bool | None = None


class ContextDetail(BaseModel):
    """Context metadata returned by get_context_info."""

    id: str
    name: str
    display_name: str | None = None
    summary: str | None = None
    usage_guide: str | None = None
    is_private: bool = True
    is_locked: bool = False
    embedding_model: str | None = None
    embedding_dimensions: int | None = None
    search_config: SearchConfig = Field(default_factory=SearchConfig)


class WorkspaceInfo(BaseModel):
    """Workspace metadata in context info response."""

    id: str
    name: str
    description: str | None = None


class ContextStats(BaseModel):
    """Memory statistics for a context.

    ``details`` comes with :meth:`KaguraClient.get_context_info` unless it
    is called with ``include_details=False``: ``by_type`` (memory count per
    type), ``by_importance`` and ``recent_7days``. Memory ``type`` is free
    text, so since memory-cloud v0.82.1 (#1750) ``by_type`` keeps the 20
    largest types and folds the rest into ``"other"`` (a stored type named
    ``other`` absorbs them; the counts still sum to the total). A folded
    reply adds ``by_type_truncated: true`` and ``by_type_total_types`` (the
    number of distinct types stored); both keys are absent when nothing was
    folded. ``details`` stays a plain mapping.
    """

    total_memories: int
    working_memories: int = 0
    persistent_memories: int = 0
    details: dict[str, Any] | None = None


def _validate_or_none(model: type[_M], value: Any) -> _M | None:
    """Parse an optional, fail-open server block, degrading to ``None``.

    For blocks the server itself serves fail-open (a guardrail block, a
    guardrail's ``tool_trigger``): a shape the SDK cannot read is "not
    usable", never a failed read of the enclosing response.
    """
    if value is None or isinstance(value, model):
        return value
    try:
        return model.model_validate(value)
    except ValidationError:
        return None


class ContextGuardrailItem(BaseModel):
    """One entry of :class:`ContextGuardrails` — L1 only, summary cut server-side."""

    model_config = ConfigDict(extra="ignore")

    memory_id: str
    summary: str
    importance: float | None = None
    authored_by_caller: bool | None = None
    source_type: str | None = None


class ContextGuardrails(BaseModel):
    """The ``guardrails`` block of ``get_context_info`` (memory-cloud v0.74.0+, #1621).

    The context's trusted-only, binding-filtered **tool-triggered** set for
    MCP clients without tool hooks: at most 10 entries with summaries cut to
    300 characters, and ≤ 4,000 characters of JSON in total. ``truncated``
    reports that entries were left out; ``total_available`` is the full set
    size. ``tool_triggered_version`` changes whenever the served
    tool-triggered set changes (the same value as the digest header and the
    export block's begin marker); it is NOT ``GuardrailSet.version``, which
    also covers the pinned list. The full set, with patterns, is
    :meth:`KaguraClient.load_guardrails`.

    ``items``, ``total_available`` and ``truncated`` are required (the
    server always sends them): a block without them reads as ``None`` —
    unusable — never as a complete set.

    ``provenance`` (memory-cloud v0.79.0+, #1682) labels the items as notes
    that context editors stored — data, not operator instructions
    (``"Notes written by context editors (facts, not operator
    instructions)."``). ``None`` on older servers.
    """

    model_config = ConfigDict(extra="ignore")

    items: list[ContextGuardrailItem]
    total_available: int
    truncated: bool
    tool_triggered_version: str | None = None
    provenance: str | None = None


class ContextInfo(BaseModel):
    """Full response from get_context_info.

    ``guardrails`` (memory-cloud v0.74.0+) has three states: the key is
    **absent** when the MCP endpoint URL carries ``?guardrails=off`` (a client
    whose hooks already deliver guardrails), it is ``null`` when the server's
    read failed, and otherwise it is a :class:`ContextGuardrails`. Both of the
    first two read as ``None``; tell them apart with
    ``"guardrails" in info.model_fields_set``. A block the SDK cannot parse
    also reads as ``None`` rather than failing the call.
    """

    status: str = "success"
    context: ContextDetail
    workspace: WorkspaceInfo | None = None
    stats: ContextStats | None = None
    instructions: str | None = None
    guardrails: ContextGuardrails | None = None

    @field_validator("guardrails", mode="before")
    @classmethod
    def _unreadable_guardrails_is_none(cls, value: Any) -> ContextGuardrails | None:
        return _validate_or_none(ContextGuardrails, value)


# ---------------------------------------------------------------------------
# Embedding status models (v0.6.1)
# ---------------------------------------------------------------------------


class FailedMemoryInfo(BaseModel):
    """Info about a memory with failed embedding.

    ``updated_at`` is the memory's last edit (``None`` when it was never
    edited), not the failure time. ``embedding_attempted_at`` (memory-cloud
    v0.95.0+, #1868) is when the embedding last failed; the server lists the
    failures latest first by it. It is ``None`` on older servers and for a
    row that failed before the server kept that clock. Since v0.120.0
    (#2095) a run whose claim expired on every retry ends ``failed`` with
    ``embedding_error`` starting ``claim_expired:``; an admin retry resets
    it.
    """

    id: str
    summary: str
    embedding_error: str | None = None
    created_at: datetime
    updated_at: datetime | None = None
    embedding_attempted_at: datetime | None = None


class EmbeddingStatus(BaseModel):
    """Embedding queue status for the workspace."""

    total: int
    by_status: dict[str, int]
    failed_memories: list[FailedMemoryInfo]


# ---------------------------------------------------------------------------
# Memory stats models (v0.6.1)
# ---------------------------------------------------------------------------


class MemoryStatItem(BaseModel):
    """Per-memory usage statistics.

    ``access_count`` is a lower bound since memory-cloud v0.105.0 (#1985): a
    statistics write skips a row another transaction holds instead of
    waiting, so a read under contention can go uncounted.
    """

    id: str
    summary: str
    type: str
    importance: float
    scope: str
    use_count: int
    access_count: int
    last_used_at: datetime | None = None
    embedding_status: str
    created_at: datetime


class MemoryStatsResponse(BaseModel):
    """Response from memory-stats endpoint."""

    memories: list[MemoryStatItem]
    total: int
    sort_by: str
    sort_order: str


# ---------------------------------------------------------------------------
# Memory list (SDK issue #143; server origin memory-cloud #580)
# ---------------------------------------------------------------------------


class MemoryListItemLocation(BaseModel):
    """WHERE-axis coordinates of a list item (server v0.54.0+, memory-cloud #1334).

    Sourced from the server's generated ``location_lat`` / ``location_lon``
    columns (i.e. ``details.location``). The optional ``label`` is not part of
    the list projection — read it from ``details`` via ``reference()``.
    """

    model_config = ConfigDict(extra="ignore")

    lat: float
    lon: float


# The two forms memory-cloud stores ``details.observed_at`` in and lists it as
# (utils/observed_at.py ``DATE_RE`` / ``DATETIME_RE``, #2330).
_OBSERVED_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_OBSERVED_DATETIME = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")


class MemoryListItem(BaseModel):
    """A single memory row in a paginated ``list_memories`` response.

    Mirrors the server's ``MemoryListItem`` wire shape. ``created_at`` /
    ``updated_at`` arrive as ISO 8601 strings (``Z``-tagged) and are parsed
    into ``datetime``, consistent with the other list models in this module.
    ``location`` is ``None`` for memories without a complete location, and
    always ``None`` against servers older than v0.54.0 (which omit it).

    ``observed_at`` (memory-cloud v0.127.0+, #2330) is when the fact held —
    ``details.observed_at`` — as an aware UTC ``datetime``, or ``None`` when
    the memory has none (and always on older servers). The server keeps the
    writer's precision: a date-only value (``2024-03-07``) reads as midnight
    UTC of that day, the instant the server itself sorts and filters it by;
    :meth:`KaguraClient.reference` returns the stored text. A value in that
    shape that names no real day or time (``2024-02-30``) reads as ``None``:
    the server sends one for a memory whose details carried it before
    v0.127.0 checked them, and it must not fail the whole page.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    summary: str
    type: str
    scope: str
    importance: float
    created_at: datetime
    updated_at: datetime
    location: MemoryListItemLocation | None = None
    observed_at: datetime | None = None

    @field_validator("observed_at", mode="before")
    @classmethod
    def _date_is_start_of_utc_day(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        # A date would otherwise parse as a naive midnight next to aware datetimes.
        if _OBSERVED_DATE.fullmatch(value):
            value = f"{value}T00:00:00Z"
        if _OBSERVED_DATETIME.fullmatch(value):
            try:
                if value[11:13] == "24":
                    # Not a time on every Python (3.14 alone reads the next midnight).
                    raise ValueError(value)
                return datetime.fromisoformat(value)
            except ValueError:
                # The server's stored shape (utils/observed_at.py) is checked by
                # regex only, so a legacy "2024-02-30" reaches the list as is.
                return None
        # Any other text is not a value the server sends: drift, as for any field.
        return value


class MemoryListResponse(BaseModel):
    """Paginated response from ``list_memories`` (``GET /api/v1/memory/list``)."""

    model_config = ConfigDict(extra="ignore")

    memories: list[MemoryListItem] = Field(default_factory=list)
    total: int
    has_more: bool


# ---------------------------------------------------------------------------
# Measurement lane — HOW-MUCH axis (#254, server v0.54.0+, memory-cloud #1333)
# ---------------------------------------------------------------------------


MeasurementPeriod = Literal["day", "week", "month"]
"""Valid ``recall_series`` bucket sizes (request side only).

Mirrors the server's ``VALID_PERIODS``; the response echoes the period as a
plain ``str`` so a server that grows the set does not break parsing."""

MeasurementAggregate = Literal["avg", "min", "max", "sum", "count", "last"]
"""Valid ``recall_series`` per-bucket aggregates (request side only).

``"last"`` is the most recent value in the bucket by ``measured_at``. Mirrors
the server's ``VALID_AGGS``; echoed back as a plain ``str``."""


class MeasurementResult(BaseModel):
    """One observation as stored by ``record_measurement``.

    Measurements are a lane separate from memories: never embedded, never
    returned by ``recall()``, never merged or rewritten by Sleep consolidation
    (an operator-set retention window can purge old rows; see
    ``KaguraClient.record_measurement``). ``measured_at`` is the
    observation time the server stored (``Z``-tagged UTC; "now" when the call
    omitted it). ``value`` is exact ``NUMERIC`` at rest and arrives as a float.
    """

    model_config = ConfigDict(extra="ignore")

    status: str = "success"
    measurement_id: str
    metric: str
    measured_at: datetime
    value: float
    unit: str | None = None


class SeriesBucket(BaseModel):
    """One non-empty time bucket of a ``recall_series`` result.

    ``bucket`` is the UTC start of the period (buckets align to UTC
    boundaries), ``value`` the aggregate over it, and ``count`` the number of
    observations it holds. Empty buckets are omitted server-side.
    """

    model_config = ConfigDict(extra="ignore")

    bucket: datetime
    value: float
    count: int


class MeasurementSeries(BaseModel):
    """A metric's series bucketed by ``period`` and aggregated by ``agg``.

    ``count`` is the number of buckets in ``series`` (not observations — sum
    ``SeriesBucket.count`` for that). ``period`` / ``agg`` echo what the
    server applied, including its defaults, and are typed ``str`` for
    forward compatibility.
    """

    model_config = ConfigDict(extra="ignore")

    status: str = "success"
    metric: str
    period: str
    agg: str
    series: list[SeriesBucket] = Field(default_factory=list)
    count: int


# ---------------------------------------------------------------------------
# Duplicate detection models (v0.6.1)
# ---------------------------------------------------------------------------


class DuplicateMemoryInfo(BaseModel):
    """Memory info for duplicate pair display."""

    id: str
    summary: str
    type: str
    created_at: datetime


class DuplicatePair(BaseModel):
    """A pair of similar memories."""

    memory_a: DuplicateMemoryInfo
    memory_b: DuplicateMemoryInfo
    similarity: float


class DuplicatesResponse(BaseModel):
    """Response from duplicate detection endpoint."""

    pairs: list[DuplicatePair]
    total_pairs: int
    threshold: float
    memories_scanned: int


# ---------------------------------------------------------------------------
# Resource Token models
# ---------------------------------------------------------------------------


class ResourceTokenCreate(BaseModel):
    """Request model for creating a resource token."""

    resource_id: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    quota_events_per_hour: int = Field(default=1000, ge=1, le=10000)


class ResourceTokenUpdate(BaseModel):
    """Request model for updating a resource token."""

    description: str | None = None
    quota_events_per_hour: int | None = Field(default=None, ge=1, le=10000)


class ResourceTokenResponse(BaseModel):
    """Resource token metadata (no plaintext token).

    ``id`` is an int before server v0.89.0 and an opaque ``rtok_…`` string
    from v0.89.0 on (memory-cloud#1008).
    """

    id: int | str
    resource_id: str
    description: str | None = None
    quota_events_per_hour: int
    created_by: str | None = None
    created_at: datetime
    last_used_at: datetime | None = None
    is_active: bool
    status: Literal["active", "revoked"]


class ResourceTokenCreateResponse(ResourceTokenResponse):
    """Resource token creation response (includes plaintext token, shown once)."""

    token: str


class PaginatedResourceTokensResponse(BaseModel):
    """Paginated list of resource tokens."""

    tokens: list[ResourceTokenResponse]
    total: int
    limit: int
    offset: int


class ResourceSetupResponse(BaseModel):
    """Atomic resource setup response (server v0.14+).

    Returned by ``ResourceClient.setup_resource()`` and
    ``KaguraClient.setup_resource()``, which create a Context, Resource entity,
    and ingestion token in a single transaction. The plaintext ``token`` is
    shown only once — save it immediately.
    """

    context_id: str
    context_name: str
    resource_id: str
    token: str
    token_id: int | str  # opaque ``rtok_…`` string from server v0.89.0
    warning: str | None = None


class ResourceEventRequest(BaseModel):
    """Request model for resource event ingestion."""

    op: Literal["upsert", "delete"]
    doc_id: str = Field(..., min_length=1, max_length=255)
    version: int | None = Field(default=None, ge=1)
    payload: dict[str, Any] | None = None
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=255)
    event_metadata: dict[str, Any] = Field(default_factory=dict)
    importance: float | None = Field(default=None, ge=0.0, le=1.0)


class ResourceEventResponse(BaseModel):
    """Response from single event ingestion."""

    status: str = "success"
    event_id: int
    queued: bool = True
    estimated_indexing_time_seconds: int | None = None


class ResourceEventBatchRequest(BaseModel):
    """Request model for batch event ingestion."""

    events: list[ResourceEventRequest] = Field(..., min_length=1, max_length=100)


class ResourceEventBatchResponse(BaseModel):
    """Response from batch event ingestion.

    Each ``errors`` row is ``{index, doc_id, error}``: the event's position
    in the batch, its ``doc_id`` and a message (e.g. ``"Duplicate version
    3"``, ``"Payload too large: … bytes"``). The REST batch route keeps this
    historic shape (checked against v0.141.0); the ``cause`` / ``constraint`` /
    ``correlation_id`` / ``help`` fields memory-cloud v0.82.0 (#1742) added
    to failed events are on the MCP ``ingest_events`` tool only. Rows stay
    plain mappings.
    """

    status: str = "success"
    created_count: int
    failed_count: int = 0
    event_ids: list[int] = Field(default_factory=list)
    errors: list[dict[str, Any]] = Field(default_factory=list)


class ResourceEventRecord(BaseModel):
    """A single ingested event row returned by ``list_resource_events``.

    Mirrors the server's ``ResourceEventRecord`` for
    ``GET /api/v1/resources/{resource_id}/events``. This is the full
    read shape — distinct from :class:`ResourceEventItem`, the 5-field
    minimal row embedded in :class:`IndexerStatusResponse.recent_events`.
    """

    id: int
    op: Literal["upsert", "delete"]
    doc_id: str
    version: int | None = None
    idempotency_key: str | None = None
    importance: float | None = None
    created_at: datetime | None = None
    payload: dict[str, Any] | None = None
    event_metadata: dict[str, Any] = Field(default_factory=dict)
    payload_bytes: int | None = None
    payload_truncated: bool = False


class ResourceEventsListResponse(BaseModel):
    """Paginated resource events response.

    Mirrors the server's ``ResourceEventsResponse``: a page of events
    plus an opaque ``next_cursor`` (``None`` on the last page). Pass the
    cursor back to :meth:`ResourceClient.list_resource_events` to page
    forward. Unlike the ``limit``/``offset`` token list, events use
    ``limit``/``cursor`` pagination.
    """

    events: list[ResourceEventRecord] = Field(default_factory=list)
    next_cursor: str | None = None


class ResourceImpactResponse(BaseModel):
    """Resource impact statistics per resource_id."""

    resource_id: str
    token_count: int
    memory_count: int
    current_schema_version: int | None = None


class FieldDefinition(BaseModel):
    """Field metadata definition within a resource schema."""

    name: str
    type: Literal["text", "number", "boolean", "date", "array", "object"]
    description: str
    classification: Literal["public", "internal", "pii", "confidential"] = "public"
    index_hint: str = ""
    unit: str | None = None
    enum_values: list[str] | None = None
    example: str | None = None
    required: bool = False


class ResourceSchemaResponse(BaseModel):
    """Resource schema with field definitions (schema registry)."""

    resource_id: str
    schema_version: int
    field_definitions: list[FieldDefinition]
    created_at: datetime


# ============================================================================
# Resource list (workspace-scoped, server v0.14+)
# ============================================================================


class ResourceListItem(BaseModel):
    """Single resource entry in the workspace resource list.

    ``context_id`` and ``context_name`` are ``None`` for a resource whose
    contexts have all been deleted while it still holds active tokens
    (server v0.94.0+, kagura-ai/memory-cloud#1863): the server lists such a
    row only so those tokens stay reachable, and revoking the resource's
    last token removes the row. ``context_display_name`` is ``None`` then
    too, and whenever the context has no display name.
    """

    resource_id: str
    context_id: str | None = None
    context_name: str | None = None
    context_display_name: str | None = None
    token_count: int
    memory_count: int
    current_schema_version: int | None = None
    created_at: datetime
    updated_at: datetime


class ResourceListResponse(BaseModel):
    """Workspace resource list response (non-paginated; server caps at < 50)."""

    resources: list[ResourceListItem]
    total: int


# ============================================================================
# Indexer status (server v0.14+)
# ============================================================================


IndexerJobStatus = Literal["idle", "queued", "running", "failed"]
"""Known indexer job statuses. Mirrors the server-side CHECK constraint on
``indexer_state.job_status``.

Documents the known values only: :attr:`IndexerState.job_status` is typed
``str`` so a status a newer server adds does not break
``get_indexer_status`` (#250)."""


IndexerSkippedReason = Literal[
    "no_pending_events",
    "schema_not_found",
    "context_not_found",
    "empty_valid_points",
    "resource_entity_missing",
    "memories_per_day_exceeded",
    "plan_suspended",
    "memory_limit_exceeded",
]
"""Known reasons the indexer records under ``metrics.skipped_reason`` when a
run was skipped. ``memories_per_day_exceeded`` (server v0.68.0+,
memory-cloud #1549) means the batch was deferred until the workspace's daily
memory quota resets (UTC). ``plan_suspended`` (the plan no longer carries
this ingest) and ``memory_limit_exceeded`` (the workspace memory limit is
reached) arrived in server v0.99.0 (memory-cloud #1939).

Documents the known values only: :attr:`IndexerStateMetrics.skipped_reason`
is typed ``str`` (#250). The server nulls reasons *it* does not know, but a
reason newer than this SDK arrives verbatim — and must stay a string, since
``None`` on this field means the last run was not skipped."""


class IndexerStateMetrics(BaseModel):
    """Per-run indexer metrics, flattened from the server JSONB column.

    ``skipped_reason`` is set only when the last run was skipped; see
    :data:`IndexerSkippedReason` for the known values.

    ``failed_events`` (memory-cloud v0.135.0+, #2254) counts the events that
    failed and were neither run again nor cleared by an operator, capped at
    20. An event below ``last_offset`` is not retried: an operator rewinds
    the indexer to replay it. Since v0.139.0 (#2477) the count stays until
    an operator clears or rewinds it (v0.135.0-v0.138.x emptied it on the
    next clean batch). ``0`` on older servers.
    """

    applied_upserts: int = 0
    applied_deletes: int = 0
    errors: int = 0
    failed_events: int = 0
    skipped_reason: str | None = None


class IndexerState(BaseModel):
    """Indexer state snapshot for one resource.

    ``job_status`` is a ``str`` for forward compatibility; see
    :data:`IndexerJobStatus` for the known values.
    """

    job_status: str
    last_run_at: datetime | None = None
    next_run_at: datetime | None = None
    active_version: int
    last_offset: int
    lag_seconds: float | None = None
    metrics: IndexerStateMetrics


class ResourceEventItem(BaseModel):
    """Single row in the indexer's recent ingest events list."""

    id: int
    op: Literal["upsert", "delete"]
    doc_id: str
    version: int | None = None
    created_at: datetime | None = None


class IndexerStatusResponse(BaseModel):
    """Response body for ``GET /api/v1/resources/{resource_id}/indexer-status``.

    ``state`` is ``None`` when the indexer has never run for this resource
    (the endpoint still returns 200 in that case). A 404 means the resource
    slug does not exist in the caller's workspace.
    """

    resource_id: str
    state: IndexerState | None = None
    recent_events: list[ResourceEventItem] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Sleep Maintenance (issue #85)
# ---------------------------------------------------------------------------


SleepRunStatus = Literal["running", "completed", "degraded", "failed", "cancelled", "rolled_back"]
"""Known Sleep run statuses. Mirrors the server's ``valid_sleep_report_status``
CHECK constraint; ``degraded`` arrived in memory-cloud v0.43.0 (#1183).

Documents the known values only: ``SleepReport.status`` and
``RollbackResult.status`` are typed ``str`` so a status a newer server adds
does not break ``get_sleep_history`` (#250)."""


SleepReportSource = Literal["sleep", "analysis"]
"""Known values of ``SleepReport.source`` (memory-cloud v0.138.0+, #2219).

``sleep`` is a Sleep Maintenance run; ``analysis`` is the row a Memory
Analysis run writes to record its LLM spend: no Sleep phase ran under it,
it has no Sleep actions and ``rollback_sleep_run`` refuses it with
``analysis_report`` (v0.140.0+, #2547).
Documents the known values only: the field is typed ``str``."""


class SleepReport(BaseModel):
    """Summary of a Sleep Maintenance run, returned by ``get_sleep_history``.

    ``status`` is a ``str`` for forward compatibility; see
    :data:`SleepRunStatus` for the known values. ``degraded`` means the run
    finished but some judge-LLM calls failed (server v0.43.0+) or a phase
    failed (v0.46.0+) — its merges/promotions still happened, so it stays
    rollbackable.
    ``llm_call_failures`` is the judge-LLM failure count behind a
    ``degraded``/``failed`` grade; ``None`` from servers older than v0.43.0.
    ``llm_calls_made`` counts successful calls only since memory-cloud
    v0.112.0 (#2060), so it reads lower than an older server's count for
    the same traffic.
    ``source`` (memory-cloud v0.138.0+, #2219) is ``"sleep"`` or
    ``"analysis"`` (see :data:`SleepReportSource`); ``None`` on older
    servers. Typed ``str`` for forward compatibility.
    """

    report_id: str
    context_id: str | None = None
    status: str
    started_at: datetime | None = None
    completed_at: datetime | None = None
    memories_processed: int
    edges_created: int
    memories_merged: int
    memories_promoted: int
    llm_calls_made: int
    llm_tokens_used: int
    llm_call_failures: int | None = None
    source: str | None = None


class SleepAction(BaseModel):
    """One audit log entry from a Sleep Maintenance run.

    ``action_type`` and ``phase`` are free-form strings — the server may add
    new types over time. Known ``action_type`` values include ``create_edge``,
    ``merge``, ``update_importance``, ``promote``, ``archive``, and ``flag``.
    ``details`` is a generic dict whose shape depends on ``action_type``.

    An action too large for its page (memory-cloud v0.82.0+, #1743) comes
    without ``details``: ``details`` is ``None``, ``details_omitted`` is
    ``True`` and ``details_total_chars`` gives the size it would have had.
    """

    id: str
    phase: str
    action_type: str
    memory_id: str | None = None
    target_id: str | None = None
    details: dict[str, Any] | None = None
    created_at: datetime | None = None
    details_omitted: bool = False
    details_total_chars: int | None = None


class SleepReportDetail(SleepReport):
    """Full Sleep Maintenance report with audit log, returned by ``get_sleep_report``.

    Extends ``SleepReport`` with per-phase result blobs and the per-action
    audit log. Fields ending in ``_result`` are server-side phase outputs
    kept as raw dicts because their shape evolves with the maintenance
    pipeline.

    Since memory-cloud v0.82.0 (#1743) ``actions`` is one page of the run's
    actions in id order: ``action_count`` is the run's total,
    ``actions_has_more`` says more actions follow, and
    ``actions_next_cursor`` is the ``actions_cursor`` for the next page
    (``None`` on the last page). Older servers send every action and
    neither paging field. ``note`` (v0.140.0+, #2547) is set on an analysis
    report (``source="analysis"``): Memory Analysis spend with no Sleep
    actions, so there is nothing to roll back.
    """

    memories_flagged: int
    embedding_calls_made: int
    error_message: str | None = None
    edge_discovery_result: dict[str, Any] | None = None
    dedup_result: dict[str, Any] | None = None
    merge_retention_result: dict[str, Any] | None = None
    importance_result: dict[str, Any] | None = None
    consolidation_result: dict[str, Any] | None = None
    reindex_result: dict[str, Any] | None = None
    actions: list[SleepAction] = Field(default_factory=list)
    action_count: int
    actions_has_more: bool = False
    actions_next_cursor: str | None = None
    note: str | None = None


class RollbackSummary(BaseModel):
    """Per-category counts of actions reversed by ``rollback_sleep_run``.

    ``merges_unreversible`` (server v0.61.0+) counts recorded merges the
    rollback could not reverse because a later write changed the edge.
    Each one is also an ``errors`` entry, so a non-zero count means the
    server answered ``partial_rollback``: read it from
    :class:`~kagura_memory.exceptions.KaguraPartialRollbackError`'s
    ``summary`` — on :class:`RollbackResult` it is always 0.
    ``importance_kept`` / ``promotions_kept`` (server v0.66.0+) are
    by-design skips, not errors: the memory was pinned, forgotten or
    deleted since the run, so its recorded change is left standing.
    ``chunks_requeued`` (#2061) / ``points_requeued`` (#2065) (server
    v0.112.0+) count restored memories whose content chunks / vector could
    not be written during the rollback and were queued for the embedding
    sweep to rebuild — the restore itself went through, so they are not
    errors either.
    Failed undo steps are listed in ``errors``.
    """

    edges_deleted: int = 0
    merges_reversed: int = 0
    merges_unreversible: int = 0
    importance_restored: int = 0
    promotions_reversed: int = 0
    importance_kept: int = 0
    promotions_kept: int = 0
    archives_restored: int = 0
    chunks_requeued: int = 0
    points_requeued: int = 0
    errors: list[str] = Field(default_factory=list)


class RollbackResult(BaseModel):
    """Result of ``rollback_sleep_run`` on a successful (no-error) run.

    ``status`` is ``"rolled_back"`` today; typed ``str`` like
    ``SleepReport.status``. A rollback with failed steps raises
    :class:`~kagura_memory.exceptions.KaguraPartialRollbackError` instead.
    """

    report_id: str
    status: str
    rollback_summary: RollbackSummary


# ---------------------------------------------------------------------------
# Edge model
# ---------------------------------------------------------------------------


class Edge(BaseModel):
    """A neural memory edge between two memories.

    Represents a directed link from ``source_id`` to ``target_id`` with a
    semantic ``edge_type`` and a ``weight``/``confidence`` pair. Edges are
    created either by users (manual curation) or by server-side processes
    (Sleep Maintenance, k-NN seeding, declared links, tag co-occurrence).

    Note:
        ``edge_type`` is intentionally typed as ``str`` (not a ``Literal``)
        because the server's ``VALID_EDGE_TYPES`` set is open-ended and grows
        with new auto-discovery processes (currently 7 values:
        ``neural_association``, ``related_to``, ``depends_on``,
        ``learned_from``, ``semantic_similarity``, ``declared_link``,
        ``tag_cooccurrence``). The server is the authority on validation.
    """

    model_config = ConfigDict(extra="ignore")

    source_id: str
    target_id: str
    edge_type: str
    weight: float = Field(ge=0.0, le=3.0)
    confidence: float = Field(ge=0.0, le=1.0)
    created_at: datetime | None = None
    last_updated: datetime | None = None


# ---------------------------------------------------------------------------
# Tag vocabulary (server v0.15.4+, SDK issue #620; server origin #614)
# ---------------------------------------------------------------------------


class TagInfo(BaseModel):
    """A tag with its usage count and last-used timestamp.

    Mirrors the wire shape of the server's ``RelatedTagItem`` as emitted by
    the ``list_tags`` MCP tool. ``sample_summary`` from the server-side
    pydantic model is intentionally omitted because no MCP tool sends it:
    ``list_tags`` never populated it, and ``recall``'s ``related_tags``
    carries only ``{tag, count}`` since memory-cloud v0.73.0 (#1599). The
    schema is otherwise aligned so callers can unify their tag-info type
    between the two surfaces.
    """

    model_config = ConfigDict(extra="ignore")

    tag: str
    count: int
    last_used_at: datetime | None = None


class ListTagsResponse(BaseModel):
    """Response from ``list_tags``: tag vocabulary for a context.

    ``has_more`` (memory-cloud v0.82.0+, #1743) is ``True`` when more tags
    matched than this reply holds: the MCP tool returns at most 200 tags
    whatever the ``limit``. There is no cursor — narrow with ``prefix`` or
    ``min_count``. Always ``False`` on a ``with_tags`` drill-down (the REST
    route returns up to ``limit`` tags and reports nothing) and on older
    servers.
    """

    model_config = ConfigDict(extra="ignore")

    context_id: str
    context_name: str
    tags: list[TagInfo] = Field(default_factory=list)
    total: int
    has_more: bool = False


class ContextTagsResponse(BaseModel):
    """Internal response from ``GET /api/v1/contexts/{id}/tags`` (#273).

    ``list_tags`` reshapes it into :class:`ListTagsResponse`. Not exported;
    it carries the server model's name, which a drift error reports.
    Unlike the MCP tool, the route sends no ``context_name`` before
    memory-cloud v0.77.0 (memory-cloud#1669), and it adds a
    ``sample_summary`` per tag, which :class:`TagInfo` drops.
    """

    model_config = ConfigDict(extra="ignore")

    context_id: str
    context_name: str | None = None
    tags: list[TagInfo]
    total: int


# ---------------------------------------------------------------------------
# File objects (server v0.15.1+)
# ---------------------------------------------------------------------------


class FileObject(BaseModel):
    """File metadata returned by upload / list / dedup operations.

    Mirrors the server's ``FileObjectOut``. The ``workspace_id`` field
    name is preserved on the wire; SDK public methods accept the same
    value as ``context_id`` for vocabulary consistency with the rest of
    the SDK.

    ``status`` is typed as ``str`` (not a ``Literal``) because the server
    may add new lifecycle states over time. Known values today:
    ``reserved``, ``uploaded``, ``confirmed``.

    ``context_id`` is the owning context a file is bound to for access
    control (server v0.41.0+). It is ``None`` for legacy/workspace-scoped
    files that were uploaded with no binding context — those stay fully
    listable and accessible to the workspace.

    ``extra="ignore"`` is explicit (matching the other wire-shape response
    models): a strict model would 500 the SDK the moment the server adds a
    field, so forward-compat tolerance is a deliberate contract, not an
    accident of the pydantic default.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    workspace_id: str
    filename: str
    content_type: str
    size_bytes: int
    sha256: str
    status: str
    created_at: datetime
    uploaded_at: datetime | None = None
    context_id: str | None = None


class FileReserveResponse(BaseModel):
    """Internal response from ``POST /api/v1/files/reserve``."""

    file_id: str
    upload_url: str
    expires_at: datetime


class FileDownloadUrlResponse(BaseModel):
    """Internal response from ``GET /api/v1/files/{file_id}/download-url``."""

    download_url: str


class FileListResponse(BaseModel):
    """Paginated list of files.

    ``next_cursor`` is forward-compatible — the current server
    (memory-cloud v0.15.x) returns at most ``limit`` items with no
    cursor field; SDK preserves the field as ``None`` so a future
    server bump can populate it without breaking callers.
    """

    files: list[FileObject]
    next_cursor: str | None = None


# =============================================================================
# File ingestion (Issue #80)
# =============================================================================
# These models are returned by the file-ingestion module
# (``kagura_memory.ingest``) and exposed to CLI users via ``kagura ingest``.
# Heavy parsing dependencies (pymupdf, pillow) are loaded lazily inside the
# ingest subpackage; importing this models module never triggers them.


class CostBreakdown(BaseModel):
    """Cost and token usage for one ingest operation.

    Used both for ``--dry-run`` cost estimation (``is_estimate=True``, no
    network egress to LLM providers) and for the final cost reported by an
    actual ingestion. Token counts are computed via ``litellm.token_counter``
    when possible; ``None`` indicates the counter could not estimate that
    field.
    """

    model_config = ConfigDict(extra="ignore")

    is_estimate: bool = False
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    vision_tokens: int | None = None
    est_usd: float | None = None
    text_provider: str | None = None
    vision_provider: str | None = None


class IngestErrorRecord(BaseModel):
    """A single per-step failure during ingestion.

    Best-effort ingestion collects these in ``IngestResult.errors`` instead
    of aborting. The Exception class for fatal orchestration failures is
    :class:`kagura_memory.exceptions.KaguraIngestError` — these records
    represent recoverable per-section issues.
    """

    model_config = ConfigDict(extra="ignore")

    step: Literal[
        "fetch",
        "extract",
        "chunk",
        "summarize",
        "vision",
        "remember",
        "archive",
    ]
    section_index: int | None = None
    message: str
    exception_type: str | None = None


class IngestResult(BaseModel):
    """Result of a single ``kagura ingest`` invocation.

    Best-effort semantics: a non-empty ``errors`` list does NOT mean the
    overall ingestion failed — partial successes (e.g. 4 of 5 sections
    written) still return a populated result with the error recorded.
    Callers should check ``IngestResult.success`` (or equivalently
    ``overview_id is not None``) to confirm the overview memory was
    created; downstream sections are guaranteed to reference an existing
    overview when present.
    """

    model_config = ConfigDict(extra="ignore")

    is_dry_run: bool = False
    source_uri: str
    source_type: Literal["file", "url"]
    overview_id: str | None = None
    section_ids: list[str] = Field(default_factory=list)
    estimated_section_count: int | None = None
    """Number of sections detected during ``--dry-run`` extraction.

    Populated only by :meth:`FileIngestor.estimate_cost` (the dry-run
    path) where no memories are written and ``section_ids`` is empty.
    The Rich renderer reads this for the dry-run "Sections:" row.
    ``None`` on actual ingest runs — use ``len(section_ids)`` then.
    """
    skipped_images: int = 0
    archived_file_id: str | None = None
    """``FileObject.id`` when the source was archived to R2, else ``None``.

    ``None`` covers three cases: archival was opt-out (``archive_original=False``),
    no ``FilesClient`` was supplied, or the upload failed (also visible as an
    ``errors[*].step == "archive"`` record).
    """
    cost: CostBreakdown
    warnings: list[str] = Field(default_factory=list)
    errors: list[IngestErrorRecord] = Field(default_factory=list)

    @property
    def success(self) -> bool:
        """True iff the overview memory was created.

        Best-effort design: per-section ``errors`` do NOT flip this to
        ``False``. Callers wanting a stricter "completed cleanly" check
        should combine ``success`` with ``not errors``.
        """
        return self.overview_id is not None


# ---------------------------------------------------------------------------
# Workspace member / invitation management (#225, server v0.42.0+)
# ---------------------------------------------------------------------------


class WorkspaceMember(BaseModel):
    """A workspace member row (#225).

    ``extra="ignore"`` — the server may add fields; ignore them instead of
    breaking deserialization (#222 lesson). The list endpoint populates
    the display/audit fields (``user_name``, ``user_email``,
    ``last_login_at``, ``allowed_context_ids``, ``credentials_status``);
    add/set-role responses carry the minimal ``user_id``/``role``/
    ``joined_at`` shape and leave the rest ``None``.
    ``credentials_status`` stays an untyped mapping — its inner shape is
    server-owned display metadata (key counts / visibility booleans) that
    the SDK forwards without interpreting.

    ``seat_paused`` (memory-cloud v0.100.0+, #1943) is ``True`` for a member
    past the seats the workspace's plan carries: every workspace route
    refuses them (``403 FEAT-001``, ``feature: members``) until the plan
    carries enough seats or a member is removed. The owner always has a
    seat. ``False`` on older servers.
    """

    model_config = ConfigDict(extra="ignore")

    user_id: str
    role: str
    user_name: str | None = None
    user_email: str | None = None
    joined_at: datetime | None = None
    last_login_at: datetime | None = None
    allowed_context_ids: list[str] | None = None
    credentials_status: dict[str, Any] | None = None
    seat_paused: bool = False


class WorkspaceInvitation(BaseModel):
    """A workspace invitation (#225). Non-strict like :class:`WorkspaceMember`.

    Server shape (``WorkspaceInvitationResponse``): ``id`` is an integer
    before server v0.89.0 and an opaque ``winv_…`` string from v0.89.0 on
    (memory-cloud#1008); there is no ``status`` field — pending is derived from
    ``is_accepted``/``is_expired``. ``token``/``invitation_url`` are bearer
    join-credentials: the server nulls them on programmatic LIST responses,
    so they are optional here and the CLI prints them only on create.
    """

    model_config = ConfigDict(extra="ignore")

    id: int | str
    email: str | None = None
    role: str
    token: str | None = None
    invitation_url: str | None = None
    is_accepted: bool = False
    is_expired: bool = False
    created_at: datetime | None = None
    expires_at: datetime | None = None
    allowed_context_ids: list[str] | None = None


class MemberAPIKey(BaseModel):
    """A member API key row (#201, server v0.42.0+). Non-strict.

    Server shape (``MemberAPIKeyResponse``): ``id`` is an integer before
    server v0.89.0 and an opaque ``akey_…`` string from v0.89.0 on
    (memory-cloud#1008); the plaintext field is named ``plaintext_key`` — non-null ONLY in the
    mint 201 response. Owner-provisioned keys are force-hidden at
    creation, so no later call ever returns the plaintext.

    ``default_context_id`` / ``default_context_name`` (memory-cloud
    v0.118.0+, #2188) are the key's MCP default context; the name is
    ``None`` when none is set or the caller cannot read that context.
    ``agent_id`` (v0.118.0+) is set when the key is bound to an agent (such
    a key carries no default context: the agent's default binding applies).
    ``connector_id`` (v0.129.0+, #2299) is set when the key is a connector's
    shared key; ``None`` for an ordinary key. All four are ``None`` on older
    servers.
    """

    model_config = ConfigDict(extra="ignore")

    id: int | str
    name: str
    key_prefix: str
    plaintext_key: str | None = None
    is_visible: bool = False
    visibility_expires_at: datetime | None = None
    created_at: datetime | None = None
    last_used_at: datetime | None = None
    revoked_at: datetime | None = None
    expires_at: datetime | None = None
    bound_context_id: str | None = None
    default_context_id: str | None = None
    default_context_name: str | None = None
    agent_id: str | None = None
    connector_id: str | None = None


# ---------------------------------------------------------------------------
# Agent bootstrap (#231, server v0.49.0+, RFC-0002 P0-3)
# ---------------------------------------------------------------------------


AgentBootstrapComponentName = Literal["pinned", "recall", "upcoming", "state", "policy"]
"""Valid values for ``get_agent_bootstrap``'s ``include`` component selector.

Mirrors the server's closed component set; the server rejects unknown
names with ``invalid_arguments``."""


def _bootstrap_payload(
    *,
    context_id: str | None,
    session_id: str | None,
    query: str | None,
    recall_k: int | None,
    pinned_cap: int | None,
    upcoming_until: str | None,
    include: list[AgentBootstrapComponentName] | None,
    max_chars: int | None,
) -> dict[str, Any]:
    """Build the omit-when-None bootstrap payload shared by both surfaces.

    ``KaguraClient.get_agent_bootstrap`` (MCP tool arguments) and
    ``AgentsClient.bootstrap`` (REST JSON body) carry the same eight
    optional keys; a single builder keeps the two surfaces in lockstep —
    ``agent_id`` stays transport-specific (MCP argument vs URL path).
    Lives beside the bootstrap models because it IS the request half of
    this wire contract.

    Raises:
        ValueError: ``max_chars`` is not an integer in [10,000, 100,000].
    """
    payload: dict[str, Any] = {}
    if context_id is not None:
        payload["context_id"] = context_id
    if session_id is not None:
        payload["session_id"] = session_id
    if query is not None:
        payload["query"] = query
    if recall_k is not None:
        payload["recall_k"] = recall_k
    if pinned_cap is not None:
        payload["pinned_cap"] = pinned_cap
    if upcoming_until is not None:
        payload["upcoming_until"] = upcoming_until
    if include is not None:
        payload["include"] = include
    if max_chars is not None:
        _validate_max_chars(max_chars)
        payload["max_chars"] = max_chars
    return payload


class AgentBootstrapBinding(BaseModel):
    """The context binding a bootstrap resolved (``agent.binding``).

    ``is_default`` is ``True`` when the context came from the agent's
    default binding (no explicit ``context_id`` was passed).
    """

    model_config = ConfigDict(extra="ignore")

    context_id: str
    is_default: bool = False


class AgentBootstrapAgent(BaseModel):
    """Agent identity block in the bootstrap envelope."""

    model_config = ConfigDict(extra="ignore")

    agent_id: str
    name: str
    binding: AgentBootstrapBinding | None = None


class AgentBootstrapCorrelation(BaseModel):
    """Correlation block (RFC-0002 P0-4) — session/run/trace identifiers.

    ``session_id`` echoes the bootstrap argument when given, else the
    server's baggage-derived session id. ``run_id``/``trace_id``/``span_id``
    are populated from the per-request correlation context when present.
    """

    model_config = ConfigDict(extra="ignore")

    agent_id: str | None = None
    session_id: str | None = None
    run_id: str | None = None
    trace_id: str | None = None
    span_id: str | None = None


class AgentBootstrapResponse(BaseModel):
    """Composed envelope from ``get_agent_bootstrap`` (server v0.49.0+).

    One session-start call that rehydrates an agent's cognitive state by
    composing existing primitives. Each entry in ``components`` is
    **fail-soft** and carries its own ``status``: ``"ok"`` (payload
    inherited from the standalone tool — ``load_pinned``, trusted-only
    ``recall``, ``recall_upcoming``, ``get_state``), ``"skipped"`` (e.g.
    recall without a ``query``, with a ``reason``), or ``"error"`` (that
    component failed; the rest still return and the top-level ``degraded``
    flag is set). Component payloads stay dicts because their shapes belong
    to the standalone tools and evolve with them: ``pinned`` lists its rows
    under ``memories``, and since memory-cloud v0.73.0 ``upcoming`` rows
    carry ``trigger`` in place of ``details``, with no opt-out.

    ``context`` reuses :class:`ContextDetail` — the server emits the block
    byte-compatible with ``get_context_info`` (``search_config`` is not
    included in bootstrap and keeps its client-side default). Since
    memory-cloud v0.79.0 (#1682) ``instructions`` is the server's static
    quick reference and the context's guide is ``context.usage_guide``.

    ``context_summary_omitted`` is ``True`` when the reply did not fit its
    ``max_chars`` budget and the server left ``context_summary`` out of every
    pinned / recall / upcoming item (memory-cloud v0.82.0+, #1743); a list
    it then had to cut carries ``truncated: True`` in its component.
    """

    model_config = ConfigDict(extra="ignore")

    status: str = "success"
    degraded: bool = False
    agent: AgentBootstrapAgent
    context: ContextDetail | None = None
    instructions: str | None = None
    components: dict[str, dict[str, Any]] = Field(default_factory=dict)
    correlation: AgentBootstrapCorrelation | None = None
    generated_at: datetime | None = None
    context_summary_omitted: bool = False


# ---------------------------------------------------------------------------
# Agent registry + bindings (#235, server v0.49.0+, RFC-0002 P0-1/P0-2)
# ---------------------------------------------------------------------------


def _agent_update_payload(
    *,
    name: str | None,
    description: str | None,
    framework: str | None,
    environment: str | None,
    version: str | None,
    status: str | None,
    enforcement_mode: str | None,
) -> dict[str, Any]:
    """Build the set-only agent-update payload shared by both surfaces.

    ``KaguraClient.update_agent`` (MCP arguments) and
    ``AgentsClient.update_agent`` (REST PATCH body) carry the same seven
    optional fields; one builder keeps them in lockstep (the
    ``_bootstrap_payload`` precedent). ``agent_id`` stays
    transport-specific (MCP argument vs URL path).
    """
    payload: dict[str, Any] = {}
    if name is not None:
        payload["name"] = name
    if description is not None:
        payload["description"] = description
    if framework is not None:
        payload["framework"] = framework
    if environment is not None:
        payload["environment"] = environment
    if version is not None:
        payload["version"] = version
    if status is not None:
        payload["status"] = status
    if enforcement_mode is not None:
        payload["enforcement_mode"] = enforcement_mode
    return payload


def _binding_scope_payload(
    *,
    can_read: bool | None,
    write_policy: str | None,
    is_default: bool | None,
) -> dict[str, Any]:
    """Build the omit-when-None binding scope trio shared by all four sites.

    ``bind_agent_context``/``update_agent_binding`` (MCP) and
    ``bind_context``/``update_binding`` (REST) all carry the same three
    optional scoping fields. When memory-cloud #1286 ships the reserved
    ``allowed_memory_types``/``allowed_source_types`` filters, this is
    the ONE place to add them.
    """
    payload: dict[str, Any] = {}
    if can_read is not None:
        payload["can_read"] = can_read
    if write_policy is not None:
        payload["write_policy"] = write_policy
    if is_default is not None:
        payload["is_default"] = is_default
    return payload


AgentStatus = Literal["active", "suspended", "retired"]
"""Known values of ``Agent.status``; documents them only (the field is ``str``).

``suspended`` is also set by the server, not only by
``update_agent``: since memory-cloud v0.139.0 (#2373 / #2349) removing a
workspace member deletes their agent-bound keys and suspends an active
agent that had no other live key."""


class Agent(BaseModel):
    """A workspace-scoped Agent Registry row (memory-cloud #1274).

    An agent is a registry entry that anchors context bindings,
    agent-bound credentials, bootstrap, and audit correlation — it is a
    resource, NOT a principal (it never authenticates by itself).

    ``status`` (``active`` | ``suspended`` | ``retired``, see
    :data:`AgentStatus`) is the fail-closed kill switch: suspended/retired
    agents cause every key bound to them to be rejected at verify time.
    ``enforcement_mode`` (``shadow`` | ``enforce``) is the binding
    enforcement ramp. Both are typed ``str`` (not ``Literal``) for forward
    compatibility — the server is the authority; request-side params use
    the closed enums.

    In a ``list_agents`` row (memory-cloud v0.82.0+, #1743) ``description``
    is a 200-character preview, with ``description_truncated`` set when it
    was cut; :meth:`KaguraClient.get_agent` returns it whole.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    workspace_id: str
    name: str
    owner_user_id: str
    status: str
    enforcement_mode: str
    description: str | None = None
    framework: str | None = None
    environment: str | None = None
    version: str | None = None
    last_seen_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    description_truncated: bool = False


class AgentBinding(BaseModel):
    """An agent→context binding row (memory-cloud #1275).

    Bindings are **purely subtractive scoping**: the effective permission
    for an agent-bound request is the existing RBAC decision ∩ binding —
    ``can_read`` gates reads, ``write_policy`` (``deny`` | ``direct``)
    gates writes. Under ``enforcement_mode="enforce"`` contexts WITHOUT a
    binding row are denied for the agent (default-deny); under
    ``"shadow"`` violations are only logged. ``is_default`` marks the
    agent's bootstrap default binding (max one per agent).

    ``allowed_memory_types`` / ``allowed_source_types`` are reserved for
    memory-cloud #1286 (per-memory enforcement) and arrive as ``None``
    today.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    agent_id: str
    context_id: str
    can_read: bool
    write_policy: str
    is_default: bool
    allowed_memory_types: list[str] | None = None
    allowed_source_types: list[str] | None = None
    created_by: str
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# Tool guardrails (#253, server v0.74.0+, memory-cloud #1619/#1621)
# ---------------------------------------------------------------------------


class ToolTrigger(BaseModel):
    """``details.tool_trigger`` — marks a memory as a tool guardrail.

    A client-side hook injects the memory's summary into the model's context
    when a tool call matches: ``tool`` is a regex **fully** matched against
    the tool name (``"Bash|PowerShell"``, ``"Edit|Write"``,
    ``"mcp__.*__remember"``); ``match`` is an optional regex **searched** in
    the call's subject (the command, file path or argument JSON for
    ``on="pre"``; the error or result text for ``on="result"``). ``on`` is
    ``"pre"`` (default) or ``"result"``; ``action`` is ``"inform"`` (default)
    or ``"block"`` (deny the call, with the summary as the reason).

    For ``on="result"`` on a Kagura tool, match the error envelope's
    ``error`` code (``"error": "capacity_locked"``, ``payload_too_large``,
    ``insufficient_scope`` …), never its message: since memory-cloud
    v0.78.0 (#1684) messages are worded for people and may change, and a
    server failure's message is fixed text that names no cause (the
    envelope's ``cause`` / ``correlation_id`` do).

    The server is the validator and the SDK does not re-implement it: both
    patterns must fit a safe-regex subset shared by Python and JavaScript
    (``tool`` ≤ 128 characters, ``match`` ≤ 200), and ``action="block"``
    needs ``on="pre"`` plus a ``match`` naming at least one literal that
    cannot match the empty string. A rejected trigger raises
    :class:`~kagura_memory.exceptions.KaguraError` whose message carries a
    stable code (``invalid details.tool_trigger: <code>: ...``). Writing one
    needs context **editor** or above and a user credential (an agent-bound
    key is refused with ``tool_trigger_requires_user_credential``).

    ``on`` / ``action`` are typed ``str`` and unknown keys are kept (in
    ``model_extra``) so a read never breaks on a value or key a newer server
    adds — a hook skips a trigger whose ``on`` / ``action`` it does not know.
    On write an unknown key is forwarded like any other, so a misspelt one
    (``mtach=...``) is refused by the server with ``tool_trigger_unknown_key``
    rather than dropped into a broader guardrail than intended.
    """

    model_config = ConfigDict(extra="allow")

    tool: str
    on: str = "pre"
    match: str | None = None
    action: str = "inform"


def _details_with_tool_trigger(
    details: dict[str, Any] | None,
    tool_trigger: ToolTrigger | dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Merge a ``tool_trigger=`` argument into the ``details`` payload.

    Shared by ``remember`` and ``update_memory``. Supplying the trigger both
    ways is rejected rather than silently resolved (the ``--location`` vs
    ``details["location"]`` rule), so neither value is quietly dropped. A
    :class:`ToolTrigger` is dumped without ``None`` fields — the server
    rejects ``"match": null``; a dict is forwarded verbatim for the server to
    validate. The caller's ``details`` dict is never mutated.

    Raises:
        ValueError: ``tool_trigger`` is given and ``details`` already has a
            ``"tool_trigger"`` key.
    """
    if tool_trigger is None:
        return details
    if details is not None and "tool_trigger" in details:
        raise ValueError(
            "Pass the guardrail either as tool_trigger=... or as details['tool_trigger'], not both"
        )
    trigger = (
        tool_trigger.model_dump(exclude_none=True)
        if isinstance(tool_trigger, ToolTrigger)
        else tool_trigger
    )
    return {**(details or {}), "tool_trigger": trigger}


class GuardrailItem(BaseModel):
    """One entry of a :class:`GuardrailSet` — the same shape on both lanes.

    ``summary`` (L1) is the text a hook injects; ``context_summary`` (L2) is
    present on pinned items only. ``tool_trigger`` is the normalized trigger
    on tool-triggered items and ``None`` on pinned ones; a legacy value the
    SDK cannot read as a trigger also reads as ``None`` (skip it). Provenance:
    ``source_type`` and ``authored_by_caller`` (``None`` = unknown) let a
    client label a guardrail someone else wrote.
    """

    model_config = ConfigDict(extra="ignore")

    memory_id: str
    summary: str
    context_summary: str | None = None
    type: str | None = None
    importance: float
    delivery_mode: str | None = None
    tool_trigger: ToolTrigger | None = None
    source_type: str | None = None
    authored_by_caller: bool | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @field_validator("tool_trigger", mode="before")
    @classmethod
    def _unreadable_trigger_is_none(cls, value: Any) -> ToolTrigger | None:
        return _validate_or_none(ToolTrigger, value)


GUARDRAIL_FORMAT = 1
"""The newest ``GuardrailSet.format`` this SDK reads (memory-cloud ``GUARDRAIL_FORMAT``)."""


class GuardrailSet(BaseModel):
    """Response of ``load_guardrails`` — MCP tool and ``POST /api/v1/memory/guardrails``.

    Two independently capped lanes, each ordered ``importance DESC,
    created_at ASC, id ASC`` (keep the order; do not re-sort): ``pinned``
    (``delivery_mode="always"``, bounded by the server's ``pinned_cap``) and
    ``tool_triggered`` (memories carrying ``details.tool_trigger``, bounded by
    ``cap``). Both are trusted-tier only — connector-ingested memories are
    never served. A memory that is both pinned and tool-triggered appears in
    both lists; dedupe by ``memory_id`` before injecting.

    **Never a silent truncation**: ``pinned_truncated`` /
    ``tool_triggered_truncated`` say which lane is incomplete (``truncated``
    is either-lane, ``total_available`` the sum of the per-lane totals).
    These fields are required, so a response without them fails to parse
    (``load_guardrails`` raises :class:`~kagura_memory.exceptions.KaguraResponseError`)
    instead of reading as complete. ``version`` is an opaque hash of the
    served set (equal means unchanged). ``context_id`` / ``context_name`` are
    filled on the MCP surface only. ``context_summary_omitted`` is ``True``
    when an MCP reply passed its ``max_chars`` budget and the server left
    ``context_summary`` out of the pinned items (memory-cloud v0.82.0+,
    #1743); items it then had to cut set the lane's ``*_truncated`` flag.

    ``format`` is the shared cache/payload format version. It bumps only when
    an existing field changes meaning or is removed, so a consumer that sees
    a ``format`` greater than :data:`GUARDRAIL_FORMAT` must treat the set as
    absent (fail-open) rather than act on fields it would misread.
    """

    model_config = ConfigDict(extra="ignore")

    status: str = "success"
    format: int
    version: str
    pinned: list[GuardrailItem]
    tool_triggered: list[GuardrailItem]
    total_available: int
    truncated: bool
    cap: int
    pinned_cap: int
    pinned_total_available: int
    pinned_truncated: bool
    tool_triggered_total_available: int
    tool_triggered_truncated: bool
    context_id: str | None = None
    context_name: str | None = None
    context_summary_omitted: bool = False


GuardrailDigestTarget = Literal["export", "instructions"]
"""Valid ``target`` values for ``GET /api/v1/memory/guardrails/digest``."""


class GuardrailDigest(BaseModel):
    """A rendered guardrail digest (``GET /api/v1/memory/guardrails/digest``).

    ``target="export"``: ``text`` is the ``AGENTS.md`` block (``text/markdown``)
    between a ``<!-- kagura-memory:guardrails begin context=<uuid>
    tool_triggered_version=<hash> -->`` line and an end marker line, or ``""``
    when the context has no tool guardrails (nothing to write — remove an
    earlier block). ``target="instructions"``: ``text`` is the exact MCP
    server ``instructions`` string this credential would receive.

    ``tool_triggered_version`` is the ``X-Kagura-Guardrails-Tool-Triggered-Version``
    response header — compare it with a stored value to detect a change
    without parsing ``text``.
    """

    model_config = ConfigDict(extra="ignore")

    context_id: str
    target: str
    text: str
    tool_triggered_version: str | None = None
    content_type: str | None = None


# ---------------------------------------------------------------------------
# Recall shape and response budgets (#298, memory-cloud #1743 / #2002 / #2124)
# ---------------------------------------------------------------------------


RecallVerbosity = Literal["compact", "full"]
"""Valid ``recall`` ``verbosity`` values (memory-cloud v0.108.0+, #2002; request side only).

``"compact"`` is the server default; ``"full"`` adds ``tags``,
``context_summary``, ``scope``, ``version`` and the uncut ``summary``."""

DedupeMode = Literal["suggest", "check", "off"]
"""Valid ``remember`` ``dedupe`` values (memory-cloud v0.93.0+, #1853; request side only).

``"suggest"`` is the server default: write, and flag an older near-duplicate
later as ``supersede_candidate``. ``"check"`` writes nothing when a live
memory of the context is a near-duplicate and replies ``status:
"duplicate_candidate"`` instead. ``"off"`` writes and never suggests for this
memory, permanently (#2024)."""

_MAX_CHARS_MIN = 10_000
_MAX_CHARS_MAX = 100_000
# recall's ``full``: how many top results carry content (memory-cloud
# RECALL_MAX_FULL, recall's own ``k`` limit).
_RECALL_FULL_MAX = 100


def _validate_int_range(name: str, value: object, lo: int, hi: int | None = None) -> None:
    """Reject a request integer the server would refuse, before any request.

    ``bool`` is not an integer here, as on the server (``type(x) is int``).

    Args:
        name: The argument's name, for the message.
        value: The value the caller passed.
        lo: The smallest accepted value.
        hi: The largest accepted value; ``None`` leaves the range open above.

    Raises:
        ValueError: If ``value`` is not an ``int`` in [``lo``, ``hi``].
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < lo
        or (hi is not None and value > hi)
    ):
        bounds = f"of at least {lo}" if hi is None else f"between {lo} and {hi}"
        raise ValueError(f"{name} must be an integer {bounds}, got {value!r}")


def _validate_max_chars(max_chars: object) -> None:
    """Reject a response budget the server would refuse (memory-cloud #1743).

    ``recall``, ``load_pinned``, ``load_guardrails``, ``get_agent_bootstrap``
    (MCP, and the REST bootstrap) and ``get_state`` in list mode take
    ``max_chars`` — characters of the reply, not tokens — as an integer in
    [10,000, 100,000].

    Raises:
        ValueError: If ``max_chars`` is not an ``int`` in that range.
    """
    _validate_int_range("max_chars", max_chars, _MAX_CHARS_MIN, _MAX_CHARS_MAX)
