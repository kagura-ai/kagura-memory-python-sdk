"""Tests for ingest.extractors.pdf — PdfExtractor error paths and TOC logic.

The happy-path PDF extraction is covered indirectly by
``tests/test_ingest_ingestor.py``. This module focuses on the error
paths and TOC-derived sectioning that are otherwise hard to reach.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("pymupdf", reason="pymupdf not installed — install [ingest-pdf] extras")

from kagura_memory.exceptions import KaguraIngestError  # noqa: E402
from kagura_memory.ingest.extractors.pdf import PdfExtractor  # noqa: E402


def _make_mock_doc(
    *,
    page_count: int,
    page_texts: list[str] | None = None,
    metadata: dict[str, str] | None = None,
    toc: list[list] | None = None,
    raise_on_page: int | None = None,
) -> MagicMock:
    """Build a MagicMock that quacks like a pymupdf Document."""
    doc = MagicMock()
    doc.page_count = page_count
    doc.metadata = metadata if metadata is not None else {}
    doc.get_toc.return_value = toc or []

    pages = []
    for i in range(page_count):
        page = MagicMock()
        if raise_on_page is not None and i == raise_on_page:
            page.get_text.side_effect = RuntimeError(f"corrupt page {i}")
        else:
            text = page_texts[i] if page_texts and i < len(page_texts) else f"page {i + 1} text"
            page.get_text.return_value = text
        pages.append(page)
    doc.__getitem__.side_effect = lambda i: pages[i]
    doc.close = MagicMock()
    return doc


def test_pdf_load_pymupdf_missing_raises_install_hint() -> None:
    """If pymupdf import fails, surface the installable extras name."""
    extractor = PdfExtractor()

    def fake_import(name: str, *args, **kwargs):
        if name == "pymupdf":
            raise ImportError("No module named 'pymupdf'")
        return __import__(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        with pytest.raises(KaguraIngestError, match="ingest-pdf"):
            extractor.extract(b"%PDF-1.4 fake")


def test_pdf_open_failure_becomes_kagura_ingest_error() -> None:
    """pymupdf.open() raising is wrapped as KaguraIngestError."""
    extractor = PdfExtractor()
    fake_module = MagicMock()
    fake_module.open.side_effect = RuntimeError("not a PDF")

    with patch.object(
        __import__("kagura_memory.ingest.extractors.pdf", fromlist=["_load_pymupdf"]),
        "_load_pymupdf",
        return_value=fake_module,
    ):
        with pytest.raises(KaguraIngestError, match="failed to open PDF"):
            extractor.extract(b"not a pdf")


def test_pdf_zero_pages_raises() -> None:
    extractor = PdfExtractor()
    fake_module = MagicMock()
    fake_module.open.return_value = _make_mock_doc(page_count=0)

    with patch("kagura_memory.ingest.extractors.pdf._load_pymupdf", return_value=fake_module):
        with pytest.raises(KaguraIngestError, match="no pages"):
            extractor.extract(b"%PDF")


def test_pdf_excess_pages_raises() -> None:
    """A PDF claiming > 10_000 pages is rejected (decompression-bomb defense)."""
    extractor = PdfExtractor()
    fake_module = MagicMock()
    fake_module.open.return_value = _make_mock_doc(page_count=10_001)

    with patch("kagura_memory.ingest.extractors.pdf._load_pymupdf", return_value=fake_module):
        with pytest.raises(KaguraIngestError, match="exceeds limit"):
            extractor.extract(b"%PDF")


def test_pdf_per_page_extract_failure_surfaces() -> None:
    extractor = PdfExtractor()
    fake_module = MagicMock()
    fake_module.open.return_value = _make_mock_doc(
        page_count=3, page_texts=["page 1", "page 2", "page 3"], raise_on_page=1
    )

    with patch("kagura_memory.ingest.extractors.pdf._load_pymupdf", return_value=fake_module):
        with pytest.raises(KaguraIngestError, match="page 2"):
            extractor.extract(b"%PDF")


def test_pdf_total_text_cap_triggers_decomp_bomb_error() -> None:
    """Total text > 50M chars raises a decompression-bomb error."""
    extractor = PdfExtractor()
    # 2 pages × 30M chars > 50M cap.
    huge_page = "x" * 30_000_000
    fake_module = MagicMock()
    fake_module.open.return_value = _make_mock_doc(page_count=2, page_texts=[huge_page, huge_page])

    with patch("kagura_memory.ingest.extractors.pdf._load_pymupdf", return_value=fake_module):
        with pytest.raises(KaguraIngestError, match="decompression bomb"):
            extractor.extract(b"%PDF")


def test_pdf_title_from_metadata_takes_precedence() -> None:
    extractor = PdfExtractor()
    fake_module = MagicMock()
    fake_module.open.return_value = _make_mock_doc(
        page_count=1,
        page_texts=["page 1 body"],
        metadata={"title": "From Metadata"},
    )

    with patch("kagura_memory.ingest.extractors.pdf._load_pymupdf", return_value=fake_module):
        content = extractor.extract(b"%PDF", source_uri="file:///x/y/fallback.pdf")

    assert content.title == "From Metadata"


def test_pdf_title_falls_back_to_source_uri_basename() -> None:
    extractor = PdfExtractor()
    fake_module = MagicMock()
    fake_module.open.return_value = _make_mock_doc(
        page_count=1, page_texts=["page 1 body"], metadata={}
    )

    with patch("kagura_memory.ingest.extractors.pdf._load_pymupdf", return_value=fake_module):
        content = extractor.extract(b"%PDF", source_uri="file:///x/y/report.pdf")

    assert content.title == "report.pdf"


def test_pdf_toc_produces_section_per_entry() -> None:
    extractor = PdfExtractor()
    fake_module = MagicMock()
    toc = [
        [1, "Chapter 1", 1],
        [1, "Chapter 2", 3],
    ]
    fake_module.open.return_value = _make_mock_doc(
        page_count=4,
        page_texts=["chapter 1 page 1", "chapter 1 page 2", "chapter 2 page 1", "chapter 2 page 2"],
        toc=toc,
    )

    with patch("kagura_memory.ingest.extractors.pdf._load_pymupdf", return_value=fake_module):
        content = extractor.extract(b"%PDF")

    assert len(content.sections) == 2
    assert content.sections[0].heading == "Chapter 1"
    assert content.sections[1].heading == "Chapter 2"
    assert content.sections[0].page_range == (1, 2)
    assert content.sections[1].page_range == (3, 4)


def test_pdf_empty_section_body_is_skipped() -> None:
    """TOC entries spanning blank pages don't emit empty sections."""
    extractor = PdfExtractor()
    fake_module = MagicMock()
    toc = [[1, "Intro", 1], [1, "Real Content", 2]]
    fake_module.open.return_value = _make_mock_doc(
        page_count=2,
        page_texts=["   ", "real content"],  # page 1 whitespace-only
        toc=toc,
    )

    with patch("kagura_memory.ingest.extractors.pdf._load_pymupdf", return_value=fake_module):
        content = extractor.extract(b"%PDF")

    # Only the non-empty section survives.
    assert len(content.sections) == 1
    assert content.sections[0].heading == "Real Content"


def test_pdf_fallback_empty_document_yields_no_sections() -> None:
    """A PDF with no TOC and only whitespace pages produces no sections."""
    extractor = PdfExtractor()
    fake_module = MagicMock()
    fake_module.open.return_value = _make_mock_doc(page_count=2, page_texts=["", "   \n  "], toc=[])

    with patch("kagura_memory.ingest.extractors.pdf._load_pymupdf", return_value=fake_module):
        content = extractor.extract(b"%PDF")

    assert content.sections == []
    assert content.page_count == 2
