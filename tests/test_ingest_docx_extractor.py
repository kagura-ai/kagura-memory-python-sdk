"""Tests for ingest.extractors.docx — DocxExtractor (python-docx)."""

from __future__ import annotations

import io
from unittest.mock import patch

import pytest

from kagura_memory.exceptions import KaguraIngestError
from kagura_memory.ingest.extractors.docx import DocxExtractor

docx = pytest.importorskip(
    "docx", reason="python-docx not installed — install [ingest-docx] extras"
)


def _make_docx(parts: list[tuple[str, str]]) -> bytes:
    """Build a .docx from (style, text) pairs and return its bytes."""
    document = docx.Document()
    for style, text in parts:
        if style.startswith("Heading") or style == "Title":
            document.add_paragraph(text, style=style)
        else:
            document.add_paragraph(text)
    buf = io.BytesIO()
    document.save(buf)
    return buf.getvalue()


def test_heading_styles_become_sections() -> None:
    data = _make_docx(
        [
            ("Heading 1", "Chapter One"),
            ("Normal", "first body"),
            ("Heading 2", "Sub Section"),
            ("Normal", "second body"),
        ]
    )
    content = DocxExtractor().extract(data, source_uri="file:///x/report.docx")

    headings = [s.heading for s in content.sections]
    assert headings == ["Chapter One", "Sub Section"]
    assert content.sections[0].depth == 1
    assert content.sections[1].depth == 2
    assert "first body" in content.sections[0].body_text
    assert content.sections[0].page_range is None


def test_no_headings_yields_single_fallback_section() -> None:
    data = _make_docx([("Normal", "plain para one"), ("Normal", "plain para two")])
    content = DocxExtractor().extract(data)
    assert len(content.sections) == 1
    assert content.sections[0].heading is None
    assert "plain para one" in content.sections[0].body_text


def test_open_failure_becomes_ingest_error() -> None:
    with pytest.raises(KaguraIngestError, match="failed to open DOCX"):
        DocxExtractor().extract(b"not a docx at all")


def test_missing_python_docx_raises_install_hint() -> None:
    def fake_import(name: str, *args, **kwargs):
        if name == "docx":
            raise ImportError("No module named 'docx'")
        return __import__(name, *args, **kwargs)

    # _load_docx uses a plain `import docx`, which routes through builtins.__import__.
    with patch("builtins.__import__", side_effect=fake_import):
        with pytest.raises(KaguraIngestError, match="ingest-docx"):
            DocxExtractor().extract(b"x")


def test_paragraph_cap_triggers_decomp_bomb_error() -> None:
    data = _make_docx([("Normal", "body")])
    with patch("kagura_memory.ingest.extractors.docx._MAX_PARAGRAPHS", 0):
        with pytest.raises(KaguraIngestError, match="paragraph count"):
            DocxExtractor().extract(data)


def test_total_text_cap_triggers_decomp_bomb_error() -> None:
    # A doc within the paragraph-count cap but over the total-text cap must
    # still be rejected (the anti-zip-bomb control for many short paragraphs).
    data = _make_docx([("Normal", "some body text"), ("Normal", "more text")])
    with patch("kagura_memory.ingest.extractors.docx._MAX_TOTAL_TEXT_CHARS", 5):
        with pytest.raises(KaguraIngestError, match="total text exceeds"):
            DocxExtractor().extract(data)
