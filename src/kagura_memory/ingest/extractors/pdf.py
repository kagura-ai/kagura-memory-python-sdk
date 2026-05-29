"""PDF extractor backed by PyMuPDF (``pymupdf>=1.24``).

Lazily imports ``pymupdf`` on first use; the import error message points
the user at the ``[ingest-pdf]`` extra. The page/TOC extraction logic lives
in the module-level :func:`extract_pymupdf_doc`, which is format-agnostic
(it operates on an already-open PyMuPDF ``Document``) and is reused by the
EPUB extractor.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ...exceptions import KaguraIngestError
from .._types import ExtractedContent, ExtractedSection
from ._util import MAX_TOTAL_TEXT_CHARS as _MAX_TOTAL_TEXT_CHARS
from ._util import filename_title

# Hard caps to defuse decompression bombs and pathological inputs. These are
# tuned to align with the design doc §8.2; tighter than fetcher.max_bytes
# because PDF can re-expand a 100 MB compressed file into multiple GB of
# decoded pages.
_MAX_PAGES = 10_000


def _load_pymupdf() -> Any:
    try:
        import pymupdf  # type: ignore[import-not-found]
    except ImportError as e:
        raise KaguraIngestError(
            "pymupdf is not installed. Install with: pip install 'kagura-memory[ingest-pdf]'"
        ) from e
    return pymupdf


def extract_pymupdf_doc(
    doc: Any, source_uri: str | None, *, label: str = "PDF"
) -> ExtractedContent:
    """Turn an open PyMuPDF ``Document`` into structured content.

    Shared by :class:`PdfExtractor` and the EPUB extractor (PyMuPDF opens
    both). ``label`` only customizes the error messages (e.g. "EPUB has no
    pages"). The caller owns the ``doc`` lifecycle (open and close).
    """
    page_count = doc.page_count
    if page_count <= 0:
        raise KaguraIngestError(f"{label} has no pages")
    if page_count > _MAX_PAGES:
        raise KaguraIngestError(f"{label} page count {page_count} exceeds limit {_MAX_PAGES}")

    title = _doc_title(doc, source_uri)
    toc = doc.get_toc(simple=True)  # list of [depth, title, page1based]

    # Pre-extract per-page text once (so multi-section spans don't re-render).
    page_texts: list[str] = []
    total_chars = 0
    for page_idx in range(page_count):
        try:
            text = doc[page_idx].get_text("text")
        except Exception as e:  # noqa: BLE001
            raise KaguraIngestError(f"failed to extract text from page {page_idx + 1}: {e}") from e
        total_chars += len(text)
        if total_chars > _MAX_TOTAL_TEXT_CHARS:
            raise KaguraIngestError(
                f"{label} total text exceeds {_MAX_TOTAL_TEXT_CHARS} chars (decompression bomb?)"
            )
        page_texts.append(text)

    sections = _sections_from_toc(toc, page_texts) if toc else _fallback_sections(page_texts)
    return ExtractedContent(
        title=title,
        sections=sections,
        images=[],  # no per-page image extraction yet
        page_count=page_count,
    )


def _doc_title(doc: Any, source_uri: str | None) -> str | None:
    meta = getattr(doc, "metadata", None) or {}
    title = (meta.get("title") or "").strip()
    if title:
        return title
    return filename_title(source_uri)


def _sections_from_toc(toc: list[list[Any]], page_texts: list[str]) -> list[ExtractedSection]:
    sections: list[ExtractedSection] = []
    for i, entry in enumerate(toc):
        # Each toc entry is [depth, title, page_number_1based].
        depth, heading, page_1based = entry[0], entry[1], entry[2]
        start_page = max(1, int(page_1based))
        # End just before the next entry's start page. Last entry runs to EOD.
        if i + 1 < len(toc):
            end_page = max(start_page, int(toc[i + 1][2]) - 1)
        else:
            end_page = len(page_texts)

        body_lines: list[str] = []
        for p in range(start_page, end_page + 1):
            if 1 <= p <= len(page_texts):
                body_lines.append(page_texts[p - 1])
        body = "\n".join(body_lines).strip()
        if not body:
            continue
        sections.append(
            ExtractedSection(
                heading=str(heading) if heading else None,
                body_text=body,
                page_range=(start_page, end_page),
                depth=int(depth) if depth else 1,
                anchor=str(heading) if heading else None,
            )
        )
    return sections


def _fallback_sections(page_texts: list[str]) -> list[ExtractedSection]:
    # No TOC: emit one section covering the whole document. The chunker
    # will token-window-split it.
    body = "\n".join(page_texts).strip()
    if not body:
        return []
    return [
        ExtractedSection(
            heading=None,
            body_text=body,
            page_range=(1, len(page_texts)),
            depth=1,
            anchor=None,
        )
    ]


class PdfExtractor:
    """Structural PDF extractor.

    Sections are derived from the PDF outline (table of contents) when
    present. When no outline is available, the document is returned as a
    single section per page with ``heading=None``; the chunker may then
    apply token-window splitting.
    """

    supports: ClassVar[frozenset[str]] = frozenset({"application/pdf"})

    def extract(self, source: bytes, source_uri: str | None = None) -> ExtractedContent:
        pymupdf = _load_pymupdf()
        try:
            doc = pymupdf.open(stream=source, filetype="pdf")
        except Exception as e:  # noqa: BLE001 - re-raised as our domain error
            raise KaguraIngestError(f"failed to open PDF: {e}") from e

        try:
            return extract_pymupdf_doc(doc, source_uri, label="PDF")
        finally:
            doc.close()
