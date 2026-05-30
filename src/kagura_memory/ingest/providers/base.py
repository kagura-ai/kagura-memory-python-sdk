"""Provider Protocol — text + vision LLM interface for ingestion."""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable


@runtime_checkable
class Provider(Protocol):
    """LLM provider abstraction for the ingestion pipeline.

    A provider exposes three concrete async operations and one local
    estimator. Implementations route through ``litellm`` so backend
    selection (Claude / Gemini / Ollama) is just a model-name change.

    Class attributes:
        name: Short identifier (``"claude"``, ``"gemini"``, ``"ollama"``).
        default_text_model: ``litellm`` model string for text summarization.
        default_vision_model: ``litellm`` model string for vision; ``None``
            if the provider has no vision capability.
    """

    name: ClassVar[str]
    default_text_model: ClassVar[str]
    default_vision_model: ClassVar[str | None]

    async def summarize(self, text: str, *, max_tokens: int, steering: str | None = None) -> str:
        """Produce a short summary of ``text``.

        Args:
            text: Section body to summarize.
            max_tokens: Approximate output ceiling.
            steering: Optional trusted domain-context string (owner-authored
                workspace configuration, never document content). When
                present, it is appended to the fixed system prompt as a
                clearly-demarcated, non-overriding block to focus terminology
                — it never replaces the fixed task. ``None`` (default) is the
                pre-steering behavior. See ``_litellm.py`` (§8.3) for why the
                document body always stays data in the user role.

        Returns:
            Plain-text summary; never None or empty.
        """
        ...

    async def summarize_overview(
        self, section_summaries: list[str], *, max_tokens: int, steering: str | None = None
    ) -> str:
        """Produce a document-level overview from per-section summaries.

        Args:
            section_summaries: Already-summarized section texts in document
                order.
            max_tokens: Approximate output ceiling.
            steering: Optional trusted domain-context string; same semantics
                as :meth:`summarize`.
        """
        ...

    async def describe_image(self, image_bytes: bytes, mime: str) -> str:
        """Extract visible text and describe layout for one image.

        Args:
            image_bytes: Raw image bytes (already preprocessed/resized
                upstream — providers do not re-resize).
            mime: MIME type, e.g. ``"image/jpeg"``.

        Returns:
            Plain-text description suitable for storing as a section memory.

        Raises:
            kagura_memory.exceptions.KaguraLLMError: If this provider has
                no vision capability or the call failed.
        """
        ...

    def count_tokens(self, text: str, *, for_vision: bool = False) -> int:
        """Local-only token count for ``text`` under this provider's model.

        Used by ``--dry-run`` cost estimation. MUST NOT make any network
        calls. Falls back to a heuristic when ``litellm`` cannot tokenize
        for the configured model.
        """
        ...
