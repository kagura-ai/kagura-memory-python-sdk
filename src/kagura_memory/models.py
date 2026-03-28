"""Pydantic models for Kagura Memory SDK."""

from typing import Literal

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
