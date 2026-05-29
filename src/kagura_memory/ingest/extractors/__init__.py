"""Structural extractors for the file ingestion pipeline.

An :class:`Extractor` consumes raw bytes (e.g. from
:class:`kagura_memory.ingest.fetcher.Fetcher`) and produces an
:class:`~kagura_memory.ingest._types.ExtractedContent` describing the
document's structural breakdown. Extractors do NOT call LLMs — that
happens later, in :class:`~kagura_memory.ingest.providers.base.Provider`.

The MIME-type → extractor mapping lives in the :data:`_REGISTRY` table
below. Each entry names the submodule and class to import lazily, so the
heavy parser dependency (``pymupdf``, ``python-docx``, …) is only loaded
when a document of that type is actually ingested. :func:`get_extractor`
resolves a MIME to an instance; :func:`supported_mimes` enumerates every
registered MIME without importing any extractor module.
"""

from __future__ import annotations

from importlib import import_module

from .base import Extractor

# Canonical MIME constants for the Office Open XML container formats and
# EPUB. Defined once here so the registry, the ingestor's dispatch
# heuristics, and the per-extractor ``supports`` sets agree.
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
EPUB_MIME = "application/epub+zip"

# MIME → (submodule, class name). The submodule is imported lazily on first
# use so optional parser deps stay opt-in. Keep this table in sync with each
# extractor class's ``supports`` set — ``test_extractor_registry`` guards the
# invariant.
_REGISTRY: dict[str, tuple[str, str]] = {
    "application/pdf": ("pdf", "PdfExtractor"),
    "text/plain": ("text", "TextExtractor"),
    "text/markdown": ("text", "TextExtractor"),
    "text/html": ("html", "HtmlExtractor"),
    DOCX_MIME: ("docx", "DocxExtractor"),
    XLSX_MIME: ("xlsx", "XlsxExtractor"),
    PPTX_MIME: ("pptx", "PptxExtractor"),
    EPUB_MIME: ("epub", "EpubExtractor"),
}


def supported_mimes() -> frozenset[str]:
    """Return every MIME type with a registered extractor.

    Pure metadata — does not import any extractor module, so it is safe to
    call for building "supported types" error messages even when the heavy
    parser deps are not installed.
    """
    return frozenset(_REGISTRY)


def get_extractor(mime: str) -> Extractor:
    """Return the extractor registered for ``mime``.

    The extractor's submodule is imported lazily here, so a missing optional
    dependency surfaces as a :class:`KaguraIngestError` from the extractor's
    own loader (pointing at the right extras), not as an ``ImportError`` at
    package import time.

    Accepts a raw HTTP Content-Type: parameters (e.g. ``; charset=utf-8``) and
    surrounding whitespace are stripped before the registry lookup, while the
    original value is preserved in the error message.

    Raises:
        ValueError: If no extractor handles the given MIME type. Callers
            should treat this as a terminal "format unsupported" error.
    """
    normalized = mime.split(";", 1)[0].strip().lower()
    entry = _REGISTRY.get(normalized)
    if entry is None:
        supported = ", ".join(sorted(supported_mimes()))
        raise ValueError(f"no extractor registered for MIME {mime!r}; supported types: {supported}")
    module_name, class_name = entry
    module = import_module(f".{module_name}", __package__)
    extractor_cls = getattr(module, class_name)
    return extractor_cls()  # type: ignore[no-any-return]


__all__ = [
    "DOCX_MIME",
    "EPUB_MIME",
    "Extractor",
    "PPTX_MIME",
    "XLSX_MIME",
    "get_extractor",
    "supported_mimes",
]
