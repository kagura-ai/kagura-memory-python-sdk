"""Kagura Memory SDK - AI-driven memory management for Kagura Memory Cloud."""

from .agent import KaguraAgent
from .client import KaguraClient
from .exceptions import (
    KaguraAuthError,
    KaguraConnectionError,
    KaguraContextError,
    KaguraError,
    KaguraLLMError,
    KaguraNotFoundError,
    KaguraQuotaError,
    KaguraRateLimitError,
)
from .models import (
    Artifact,
    ExploredMemory,
    FieldDefinition,
    LLMUsage,
    Memory,
    MemoryInfo,
    MemoryToStore,
    Message,
    PaginatedResourceTokensResponse,
    ProcessResult,
    RecallQuery,
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
    Session,
)
from .resource_client import ResourceClient

__version__ = "0.6.0"

__all__ = [
    # Core classes
    "KaguraAgent",
    "KaguraClient",
    "ResourceClient",
    # Models
    "Session",
    "Message",
    "Artifact",
    "ProcessResult",
    "Memory",
    "MemoryInfo",
    "MemoryToStore",
    "RecallQuery",
    "ExploredMemory",
    "LLMUsage",
    # Resource Token models
    "ResourceTokenCreate",
    "ResourceTokenUpdate",
    "ResourceTokenResponse",
    "ResourceTokenCreateResponse",
    "PaginatedResourceTokensResponse",
    "ResourceEventRequest",
    "ResourceEventResponse",
    "ResourceEventBatchRequest",
    "ResourceEventBatchResponse",
    "ResourceImpactResponse",
    "FieldDefinition",
    "ResourceSchemaResponse",
    # Exceptions
    "KaguraError",
    "KaguraAuthError",
    "KaguraConnectionError",
    "KaguraRateLimitError",
    "KaguraNotFoundError",
    "KaguraQuotaError",
    "KaguraLLMError",
    "KaguraContextError",
]
