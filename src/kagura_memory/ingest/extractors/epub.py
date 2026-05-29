"""EPUB extractor — reuses PyMuPDF (no new dependency).

PyMuPDF already opens EPUB files and exposes the same paginated, TOC-aware
API as PDF, so :class:`EpubExtractor` delegates to
:meth:`PdfExtractor._extract_from_doc`. The ``[ingest-epub]`` extra is an
alias of ``[ingest-pdf]`` — installing either provides ``pymupdf``.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ...exceptions import KaguraIngestError
from .._types import ExtractedContent
from .pdf import _load_pymupdf, extract_pymupdf_doc


class EpubExtractor:
    """Structural EPUB extractor backed by PyMuPDF.

    Sections come from the EPUB navigation/TOC when present; otherwise the
    document falls back to one section covering all reflowed pages. Shares
    the page/text caps defined in :mod:`.pdf`.
    """

    supports: ClassVar[frozenset[str]] = frozenset({"application/epub+zip"})

    def extract(self, source: bytes, source_uri: str | None = None) -> ExtractedContent:
        pymupdf = _load_pymupdf()
        doc: Any
        try:
            doc = pymupdf.open(stream=source, filetype="epub")
        except Exception as e:  # noqa: BLE001 - re-raised as our domain error
            raise KaguraIngestError(f"failed to open EPUB: {e}") from e

        try:
            return extract_pymupdf_doc(doc, source_uri, label="EPUB")
        finally:
            doc.close()
