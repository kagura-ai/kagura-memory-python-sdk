"""Tests for ingest.extractors.pptx — PptxExtractor (python-pptx)."""

from __future__ import annotations

import io
from unittest.mock import patch

import pytest

from kagura_memory.exceptions import KaguraIngestError
from kagura_memory.ingest.extractors.pptx import PptxExtractor

pptx = pytest.importorskip(
    "pptx", reason="python-pptx not installed — install [ingest-pptx] extras"
)


def _make_pptx(slides: list[tuple[str | None, list[str]]]) -> bytes:
    prs = pptx.Presentation()
    # Layout 1 is "Title and Content" — has a title + body placeholder.
    layout = prs.slide_layouts[1]
    for title, bullets in slides:
        slide = prs.slides.add_slide(layout)
        if title is not None:
            slide.shapes.title.text = title
        body = slide.placeholders[1].text_frame
        body.text = bullets[0] if bullets else ""
        for extra in bullets[1:]:
            body.add_paragraph().text = extra
    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()


def test_one_section_per_slide_title_is_heading() -> None:
    data = _make_pptx(
        [
            ("Intro", ["point a", "point b"]),
            ("Details", ["more text"]),
        ]
    )
    content = PptxExtractor().extract(data, source_uri="file:///x/deck.pptx")

    assert [s.heading for s in content.sections] == ["Intro", "Details"]
    assert "point a" in content.sections[0].body_text
    assert "point b" in content.sections[0].body_text
    assert content.sections[0].page_range is None
    assert content.title == "Intro"


def test_untitled_slide_falls_back_to_slide_number() -> None:
    data = _make_pptx([(None, ["only body"])])
    content = PptxExtractor().extract(data)
    assert content.sections[0].heading == "Slide 1"
    assert "only body" in content.sections[0].body_text


def test_open_failure_becomes_ingest_error() -> None:
    with pytest.raises(KaguraIngestError, match="failed to open PPTX"):
        PptxExtractor().extract(b"not a pptx")


def test_slide_count_cap_raises() -> None:
    data = _make_pptx([("A", ["x"]), ("B", ["y"])])
    with patch("kagura_memory.ingest.extractors.pptx._MAX_SLIDES", 1):
        with pytest.raises(KaguraIngestError, match="slide count"):
            PptxExtractor().extract(data)


def test_missing_python_pptx_raises_install_hint() -> None:
    def fake_import(name: str, *args, **kwargs):
        if name == "pptx":
            raise ImportError("No module named 'pptx'")
        return __import__(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        with pytest.raises(KaguraIngestError, match="ingest-pptx"):
            PptxExtractor().extract(b"x")
