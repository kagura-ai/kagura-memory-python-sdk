"""Kagura Memory SDK - AI-driven memory management for Kagura Memory Cloud."""

from .agent import KaguraAgent
from .client import KaguraClient
from .exceptions import (
    KaguraAuthError,
    KaguraConnectionError,
    KaguraContextError,
    KaguraError,
    KaguraLLMError,
    KaguraQuotaError,
    KaguraRateLimitError,
)
from .models import (
    Artifact,
    ExploredMemory,
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
    ResourceTokenCreate,
    ResourceTokenCreateResponse,
    ResourceTokenResponse,
    ResourceTokenUpdate,
    Session,
)
from .resource_client import ResourceClient

__version__ = "0.4.2"

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
    # Exceptions
    "KaguraError",
    "KaguraAuthError",
    "KaguraConnectionError",
    "KaguraRateLimitError",
    "KaguraQuotaError",
    "KaguraLLMError",
    "KaguraContextError",
]
