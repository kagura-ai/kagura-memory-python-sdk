"""Pydantic models for Kagura Memory SDK."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Message(BaseModel):
    """A message in a conversation session."""

    role: Literal["user", "assistant", "system"]
    content: str


class Artifact(BaseModel):
    """An artifact attached to a session (code, document, etc.)."""

    type: Literal["code", "document", "error", "config"]
    content: str
    source: str | None = None
    language: str | None = None


class Session(BaseModel):
    """A conversation session with messages and optional artifacts."""

    messages: list[Message]
    artifacts: list[Artifact] = Field(default_factory=list)


class MemoryInfo(BaseModel):
    """Information about a remembered memory."""

    memory_id: str
    summary: str


class Memory(BaseModel):
    """A recalled memory with relevance score."""

    memory_id: str
    summary: str
    score: float


class LLMUsage(BaseModel):
    """LLM token usage statistics."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str


class MemoryToStore(BaseModel):
    """A memory to be stored, as determined by LLM analysis."""

    summary: str = Field(..., min_length=1, max_length=250)
    content: str
    type: Literal["code", "note", "decision", "bug-fix", "feature", "learning"] = "note"
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)


class RecallQuery(BaseModel):
    """A recall query, as determined by LLM analysis."""

    query: str
    reason: str = ""
    filters: dict[str, Any] | None = None


class AnalysisResult(BaseModel):
    """Internal model for LLM analysis results."""

    should_remember: bool
    memories_to_store: list[MemoryToStore] = Field(default_factory=list)
    should_recall: bool
    recall_queries: list[RecallQuery] = Field(default_factory=list)
    llm_usage: LLMUsage | None = None


class ExploredMemory(BaseModel):
    """Memory discovered through explore operation."""

    memory_id: str
    summary: str
    activation: float = 0.0
    hop: int = 0


class ProcessResult(BaseModel):
    """Result of processing a session."""

    remembered: list[MemoryInfo] = Field(default_factory=list)
    recalled: list[Memory] = Field(default_factory=list)
    explored: list[ExploredMemory] = Field(default_factory=list)
    context_used: str
    actions: list[str] = Field(default_factory=list)
    llm_usage: LLMUsage | None = None


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
    """Feature flags reported by the server."""

    neural_memory: bool = False
    research_tools: bool = False


class ServerInfo(BaseModel):
    """Server information from /api/v1/system/info."""

    name: str
    version: str
    description: str | None = None
    environment: str | None = None
    features: ServerFeatures = Field(default_factory=ServerFeatures)


class UsageQuota(BaseModel):
    """Usage vs limit for a single resource category."""

    used: int
    limit: int
    percentage: float | None = None


class UsageQuotaLimitOnly(BaseModel):
    """Quota with limit only (no usage counter)."""

    limit: int


class UsageInfo(BaseModel):
    """Workspace usage and quota information."""

    plan: str
    memories: UsageQuota
    contexts: UsageQuota
    members: UsageQuota
    mcp_calls_per_day: UsageQuotaLimitOnly


class SearchConfig(BaseModel):
    """Hybrid search configuration for a context."""

    semantic_weight: float = 0.6
    bm25_weight: float = 0.4
    fetch_factor: int = 3
    use_rerank: bool = False
    reranker_provider: str | None = None
    reranker_model: str | None = None


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
    """Memory statistics for a context."""

    total_memories: int
    working_memories: int = 0
    persistent_memories: int = 0
    details: dict[str, Any] | None = None


class ContextInfo(BaseModel):
    """Full response from get_context_info."""

    status: str = "success"
    context: ContextDetail
    workspace: WorkspaceInfo | None = None
    stats: ContextStats | None = None
    instructions: str | None = None


# ---------------------------------------------------------------------------
# Embedding status models (v0.6.1)
# ---------------------------------------------------------------------------


class FailedMemoryInfo(BaseModel):
    """Info about a memory with failed embedding."""

    id: str
    summary: str
    embedding_error: str | None = None
    created_at: datetime
    updated_at: datetime | None = None


class EmbeddingStatus(BaseModel):
    """Embedding queue status for the workspace."""

    total: int
    by_status: dict[str, int]
    failed_memories: list[FailedMemoryInfo]


# ---------------------------------------------------------------------------
# Memory stats models (v0.6.1)
# ---------------------------------------------------------------------------


class MemoryStatItem(BaseModel):
    """Per-memory usage statistics."""

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


class MemoryListItem(BaseModel):
    """A single memory row in a paginated ``list_memories`` response.

    Mirrors the server's ``MemoryListItem`` wire shape. ``created_at`` /
    ``updated_at`` arrive as ISO 8601 strings (``Z``-tagged) and are parsed
    into ``datetime``, consistent with the other list models in this module.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    summary: str
    type: str
    scope: str
    importance: float
    created_at: datetime
    updated_at: datetime


class MemoryListResponse(BaseModel):
    """Paginated response from ``list_memories`` (``GET /api/v1/memory/list``)."""

    model_config = ConfigDict(extra="ignore")

    memories: list[MemoryListItem] = Field(default_factory=list)
    total: int
    has_more: bool


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
    """Resource token metadata (no plaintext token)."""

    id: int
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
    token_id: int
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
    """Response from batch event ingestion."""

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
    """Single resource entry in the workspace resource list."""

    resource_id: str
    context_id: str
    context_name: str
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
"""Indexer job status. Mirrors the server-side CHECK constraint on
``indexer_state.job_status``."""


IndexerSkippedReason = Literal[
    "no_pending_events",
    "schema_not_found",
    "context_not_found",
    "empty_valid_points",
    "resource_entity_missing",
]
"""Reasons the indexer may record under ``metrics.skipped_reason`` when a run
was skipped. Server degrades unknown values to ``None`` on the wire."""


class IndexerStateMetrics(BaseModel):
    """Per-run indexer metrics, flattened from the server JSONB column."""

    applied_upserts: int = 0
    applied_deletes: int = 0
    errors: int = 0
    skipped_reason: IndexerSkippedReason | None = None


class IndexerState(BaseModel):
    """Indexer state snapshot for one resource."""

    job_status: IndexerJobStatus
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


SleepRunStatus = Literal["running", "completed", "failed", "cancelled", "rolled_back"]


class SleepReport(BaseModel):
    """Summary of a Sleep Maintenance run, returned by ``get_sleep_history``."""

    report_id: str
    context_id: str | None = None
    status: SleepRunStatus
    started_at: datetime | None = None
    completed_at: datetime | None = None
    memories_processed: int
    edges_created: int
    memories_merged: int
    memories_promoted: int
    llm_calls_made: int
    llm_tokens_used: int


class SleepAction(BaseModel):
    """One audit log entry from a Sleep Maintenance run.

    ``action_type`` and ``phase`` are free-form strings — the server may add
    new types over time. Known ``action_type`` values include ``create_edge``,
    ``merge``, ``update_importance``, ``promote``, ``archive``, and ``flag``.
    ``details`` is a generic dict whose shape depends on ``action_type``.
    """

    id: str
    phase: str
    action_type: str
    memory_id: str | None = None
    target_id: str | None = None
    details: dict[str, Any] | None = None
    created_at: datetime | None = None


class SleepReportDetail(SleepReport):
    """Full Sleep Maintenance report with audit log, returned by ``get_sleep_report``.

    Extends ``SleepReport`` with per-phase result blobs and the per-action
    audit log. Fields ending in ``_result`` are server-side phase outputs
    kept as raw dicts because their shape evolves with the maintenance
    pipeline.
    """

    memories_flagged: int
    embedding_calls_made: int
    error_message: str | None = None
    edge_discovery_result: dict[str, Any] | None = None
    dedup_result: dict[str, Any] | None = None
    importance_result: dict[str, Any] | None = None
    consolidation_result: dict[str, Any] | None = None
    reindex_result: dict[str, Any] | None = None
    actions: list[SleepAction] = Field(default_factory=list)
    action_count: int


class RollbackSummary(BaseModel):
    """Per-category counts of actions reversed by ``rollback_sleep_run``."""

    edges_deleted: int = 0
    merges_reversed: int = 0
    importance_restored: int = 0
    promotions_reversed: int = 0
    archives_restored: int = 0
    errors: list[str] = Field(default_factory=list)


class RollbackResult(BaseModel):
    """Result of ``rollback_sleep_run`` on a successful (no-error) run."""

    report_id: str
    status: SleepRunStatus
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
    pydantic model is intentionally omitted because ``list_tags`` does not
    populate it (only ``recall.related_tags`` does). The schema is otherwise
    aligned so callers can unify their tag-info type between the two
    surfaces.
    """

    model_config = ConfigDict(extra="ignore")

    tag: str
    count: int
    last_used_at: datetime | None = None


class ListTagsResponse(BaseModel):
    """Response from ``list_tags``: tag vocabulary for a context."""

    model_config = ConfigDict(extra="ignore")

    context_id: str
    context_name: str
    tags: list[TagInfo] = Field(default_factory=list)
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
