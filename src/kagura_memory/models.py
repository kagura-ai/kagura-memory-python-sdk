"""Pydantic models for Kagura Memory SDK."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


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


class ResourceImpactResponse(BaseModel):
    """Resource impact statistics per resource_id."""

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
