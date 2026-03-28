"""Kagura Memory SDK - AI-driven memory management for Kagura Memory Cloud."""

from .agent import KaguraAgent
from .client import KaguraClient
from .exceptions import (
    KaguraAuthError,
    KaguraConnectionError,
    KaguraContextError,
    KaguraError,
    KaguraLLMError,
    KaguraRateLimitError,
)
from .models import (
    Artifact,
    ExploredMemory,
    LLMUsage,
    Memory,
    MemoryInfo,
    Message,
    ProcessResult,
    Session,
)

__version__ = "0.2.3"

__all__ = [
    # Core classes
    "KaguraAgent",
    "KaguraClient",
    # Models
    "Session",
    "Message",
    "Artifact",
    "ProcessResult",
    "Memory",
    "MemoryInfo",
    "ExploredMemory",
    "LLMUsage",
    # Exceptions
    "KaguraError",
    "KaguraAuthError",
    "KaguraConnectionError",
    "KaguraRateLimitError",
    "KaguraLLMError",
    "KaguraContextError",
]
