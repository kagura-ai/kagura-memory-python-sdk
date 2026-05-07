"""Tests for the PdfExtractor against the static fixture PDF."""

from __future__ import annotations

from pathlib import Path

import pytest

from kagura_memory.exceptions import KaguraIngestError
from kagura_memory.ingest.extractors import get_extractor
from kagura_memory.ingest.extractors.pdf import PdfExtractor

FIXTURE = Path(__file__).parent / "fixtures" / "sample.pdf"


pytestmark = pytest.mark.skipif(not FIXTURE.exists(), reason="sample.pdf fixture missing")


def test_registry_returns_pdf_extractor() -> None:
    extractor = get_extractor("application/pdf")
    assert isinstance(extractor, PdfExtractor)


def test_registry_rejects_unknown_mime() -> None:
    with pytest.raises(ValueError, match="no extractor"):
        get_extractor("application/x-unknown")


def test_extract_finds_toc_sections() -> None:
    extractor = PdfExtractor()
    body = FIXTURE.read_bytes()
    content = extractor.extract(body)
    # The fixture has 3 TOC entries (Section 1/2/3) — extractor should
    # surface them as 3 sections.
    assert len(content.sections) == 3
    headings = [s.heading for s in content.sections]
    assert "Section 1: Introduction" in headings
    assert "Section 2: Methodology" in headings
    assert "Section 3: Conclusion" in headings


def test_extract_includes_page_count() -> None:
    content = PdfExtractor().extract(FIXTURE.read_bytes())
    assert content.page_count == 3


def test_extract_includes_title_from_metadata() -> None:
    content = PdfExtractor().extract(FIXTURE.read_bytes())
    assert content.title == "Sample Document"


def test_extract_section_body_contains_expected_text() -> None:
    content = PdfExtractor().extract(FIXTURE.read_bytes())
    by_heading = {s.heading: s for s in content.sections}
    assert "introduction" in by_heading["Section 1: Introduction"].body_text.lower()
    assert "methodology" in by_heading["Section 2: Methodology"].body_text.lower()


def test_corrupt_pdf_raises_ingest_error() -> None:
    extractor = PdfExtractor()
    with pytest.raises(KaguraIngestError, match="failed to open PDF"):
        extractor.extract(b"not a pdf at all")
