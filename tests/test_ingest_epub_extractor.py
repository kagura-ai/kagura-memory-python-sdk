"""Tests for ingest.extractors.epub — EpubExtractor (reuses PyMuPDF).

These mock the PyMuPDF loader, so they run without ``pymupdf`` installed:
EpubExtractor delegates to the shared ``extract_pymupdf_doc()`` helper with
``label="EPUB"``, which is exercised here via a mock document.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from kagura_memory.exceptions import KaguraIngestError
from kagura_memory.ingest.extractors.epub import EpubExtractor


def _make_mock_doc(*, page_count: int, page_texts: list[str], toc: list[list] | None = None):
    doc = MagicMock()
    doc.page_count = page_count
    doc.metadata = {}
    doc.get_toc.return_value = toc or []
    pages = []
    for i in range(page_count):
        page = MagicMock()
        page.get_text.return_value = page_texts[i] if i < len(page_texts) else ""
        pages.append(page)
    doc.__getitem__.side_effect = lambda i: pages[i]
    doc.close = MagicMock()
    return doc


def test_epub_opens_with_epub_filetype_and_extracts_sections() -> None:
    fake_module = MagicMock()
    fake_module.open.return_value = _make_mock_doc(
        page_count=2,
        page_texts=["chapter one text", "chapter two text"],
        toc=[[1, "Chapter One", 1], [1, "Chapter Two", 2]],
    )

    with patch("kagura_memory.ingest.extractors.epub._load_pymupdf", return_value=fake_module):
        content = EpubExtractor().extract(b"PK\x03\x04 fake epub", source_uri="file:///x/book.epub")

    # Opened as EPUB, not PDF.
    _, kwargs = fake_module.open.call_args
    assert kwargs["filetype"] == "epub"
    assert [s.heading for s in content.sections] == ["Chapter One", "Chapter Two"]
    assert content.page_count == 2


def test_epub_without_toc_falls_back_to_single_section() -> None:
    # Structureless EPUB (no navigation/TOC) → one fallback section over all
    # pages, via the shared extract_pymupdf_doc fallback path.
    fake_module = MagicMock()
    fake_module.open.return_value = _make_mock_doc(
        page_count=2, page_texts=["chapter text one", "chapter text two"], toc=[]
    )
    with patch("kagura_memory.ingest.extractors.epub._load_pymupdf", return_value=fake_module):
        content = EpubExtractor().extract(b"PK\x03\x04 fake", source_uri="file:///x/book.epub")
    assert len(content.sections) == 1
    assert content.sections[0].heading is None
    assert "chapter text one" in content.sections[0].body_text
    assert "chapter text two" in content.sections[0].body_text


def test_epub_open_failure_becomes_ingest_error() -> None:
    fake_module = MagicMock()
    fake_module.open.side_effect = RuntimeError("not an epub")
    with patch("kagura_memory.ingest.extractors.epub._load_pymupdf", return_value=fake_module):
        with pytest.raises(KaguraIngestError, match="failed to open EPUB"):
            EpubExtractor().extract(b"garbage")


def test_epub_empty_document_uses_epub_label_in_error() -> None:
    fake_module = MagicMock()
    fake_module.open.return_value = _make_mock_doc(page_count=0, page_texts=[])
    with patch("kagura_memory.ingest.extractors.epub._load_pymupdf", return_value=fake_module):
        with pytest.raises(KaguraIngestError, match="EPUB has no pages"):
            EpubExtractor().extract(b"x")


def test_missing_pymupdf_raises_install_hint() -> None:
    def fake_import(name: str, *args, **kwargs):
        if name == "pymupdf":
            raise ImportError("No module named 'pymupdf'")
        return __import__(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        with pytest.raises(KaguraIngestError, match="ingest-pdf"):
            EpubExtractor().extract(b"x")


def test_supports_declares_epub_mime() -> None:
    assert EpubExtractor.supports == frozenset({"application/epub+zip"})
