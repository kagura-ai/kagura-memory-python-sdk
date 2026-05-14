"""Extractor Protocol — the structural-extraction interface."""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable

from .._types import ExtractedContent


@runtime_checkable
class Extractor(Protocol):
    """Convert raw bytes into a structured :class:`ExtractedContent`.

    Extractors are pure parsers — no LLM calls, no network. They run
    synchronously inside ``asyncio.to_thread`` because the underlying
    libraries (e.g. PyMuPDF) are CPU-bound and blocking.

    Class attributes:
        supports: Frozen set of MIME types this extractor accepts.
    """

    supports: ClassVar[frozenset[str]]

    def extract(self, source: bytes, source_uri: str | None = None) -> ExtractedContent:
        """Parse ``source`` into structural sections and embedded images.

        Args:
            source: Raw bytes of the document.
            source_uri: Origin URI for diagnostic messages. Optional.

        Returns:
            Populated :class:`ExtractedContent`. ``sections`` may be empty
            (e.g. a flat document with no detectable structure); the
            chunker will fall back to token-window splitting in that case.
        """
        ...
