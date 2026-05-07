"""Internal dataclasses for the file ingestion pipeline.

These types are NOT part of the public SDK surface — they flow between the
fetcher, extractor, chunker, provider, and orchestrator only. Public result
types (:class:`IngestResult`, :class:`CostBreakdown`,
:class:`IngestErrorRecord`) live in :mod:`kagura_memory.models`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExtractedSection:
    """One structural section produced by an :class:`Extractor`.

    Attributes:
        heading: Section heading text (e.g. PDF outline entry). May be
            ``None`` for documents without structural metadata.
        body_text: Plain-text body of the section.
        page_range: ``(start_page, end_page)`` 1-indexed for PDFs, or
            ``None`` for non-paginated formats.
        depth: Heading level (1 = top-level, 2 = subsection, ...). Defaults
            to 1.
        anchor: Stable identifier for the section (e.g. PDF outline title
            or page anchor). May be the heading text.
    """

    heading: str | None
    body_text: str
    page_range: tuple[int, int] | None = None
    depth: int = 1
    anchor: str | None = None


@dataclass
class ExtractedImage:
    """One image extracted from a document or fetched directly.

    Attributes:
        bytes_: Raw image bytes (preprocessing happens later in the
            provider/ingestor stack).
        mime: MIME type (e.g. ``"image/jpeg"``, ``"image/png"``).
        page: 1-indexed source page (for embedded images), or ``None`` for
            standalone image files.
        anchor: Optional human-readable label (e.g. ``"figure 3"``).
    """

    bytes_: bytes
    mime: str
    page: int | None = None
    anchor: str | None = None


@dataclass
class ExtractedContent:
    """Full output of one :class:`Extractor` invocation.

    Sections are returned in document order. Images are returned in the
    order discovered. Either may be empty.
    """

    title: str | None
    sections: list[ExtractedSection] = field(default_factory=list)
    images: list[ExtractedImage] = field(default_factory=list)
    page_count: int | None = None


@dataclass
class Chunk:
    """A summarization-ready chunk produced by the chunker.

    A chunk corresponds 1:1 to an output section memory. The chunker may
    split a long :class:`ExtractedSection` into multiple chunks; conversely
    it may pass through a single section as one chunk.
    """

    text: str
    heading: str | None
    page_range: tuple[int, int] | None
    depth: int
    anchor: str | None
    section_index: int  # 0-based index in the final output section list
