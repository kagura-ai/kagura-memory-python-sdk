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


class AnalysisResult(BaseModel):
    """Internal model for LLM analysis results."""

    should_remember: bool
    memories_to_store: list[dict] = Field(default_factory=list)
    should_recall: bool
    recall_queries: list[dict] = Field(default_factory=list)
    llm_usage: "LLMUsage | None" = None


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
