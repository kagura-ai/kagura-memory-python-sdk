"""Structural extractors for the file ingestion pipeline.

An :class:`Extractor` consumes raw bytes (e.g. from
:class:`kagura_memory.ingest.fetcher.Fetcher`) and produces an
:class:`~kagura_memory.ingest._types.ExtractedContent` describing the
document's structural breakdown. Extractors do NOT call LLMs — that
happens later, in :class:`~kagura_memory.ingest.providers.base.Provider`.

Phase 1 ships :class:`~kagura_memory.ingest.extractors.pdf.PdfExtractor`
only. The MIME-type → extractor registry lives in this package's
:func:`get_extractor` helper.
"""

from __future__ import annotations

from .base import Extractor


def get_extractor(mime: str) -> Extractor:
    """Return the extractor registered for ``mime``.

    Raises:
        ValueError: If no extractor handles the given MIME type. Callers
            should treat this as a terminal "format unsupported" error.
    """
    mime = mime.lower()
    if mime == "application/pdf":
        from .pdf import PdfExtractor

        return PdfExtractor()
    raise ValueError(f"no extractor registered for MIME {mime!r}")


__all__ = ["Extractor", "get_extractor"]
