"""Tests for ingest.extractors.text — TextExtractor (stdlib, no deps)."""

from __future__ import annotations

import pytest

from kagura_memory.exceptions import KaguraIngestError
from kagura_memory.ingest.extractors.text import (
    _MAX_TOTAL_TEXT_CHARS,
    TextExtractor,
)


def test_markdown_atx_headings_become_sections() -> None:
    md = b"# Title\n\nintro body\n\n## Sub A\n\nbody a\n\n## Sub B\n\nbody b\n"
    content = TextExtractor().extract(md, source_uri="file:///x/doc.md")

    headings = [s.heading for s in content.sections]
    assert headings == ["Title", "Sub A", "Sub B"]
    assert content.sections[0].depth == 1
    assert content.sections[1].depth == 2
    assert content.sections[0].body_text == "intro body"
    assert content.sections[0].page_range is None
    assert content.title == "Title"


def test_leading_body_before_first_heading_is_kept() -> None:
    md = b"preamble line\n\n# First\n\nbody\n"
    content = TextExtractor().extract(md)

    assert content.sections[0].heading is None
    assert content.sections[0].body_text == "preamble line"
    assert content.sections[1].heading == "First"


def test_plain_text_with_no_headings_yields_single_fallback_section() -> None:
    txt = b"just some plain text\nwith two lines\n"
    content = TextExtractor().extract(txt, source_uri="file:///x/notes.txt")

    assert len(content.sections) == 1
    assert content.sections[0].heading is None
    assert "plain text" in content.sections[0].body_text
    # No depth-1 heading → title falls back to the basename.
    assert content.title == "notes.txt"


def test_hash_inside_fenced_code_block_is_not_a_heading() -> None:
    md = b"# Real Heading\n\n```\n# not a heading\ncode line\n```\n\ntail\n"
    content = TextExtractor().extract(md)

    headings = [s.heading for s in content.sections]
    assert headings == ["Real Heading"]
    assert "# not a heading" in content.sections[0].body_text


def test_heading_text_ending_in_hash_is_preserved() -> None:
    # CommonMark: a closing '#' run must be preceded by whitespace, so "C#" is
    # heading text, not an ATX closer.
    content = TextExtractor().extract(b"# C#\n\nbody\n")
    assert content.sections[0].heading == "C#"
    assert content.title == "C#"


def test_atx_closing_hash_run_is_stripped() -> None:
    content = TextExtractor().extract(b"## Heading ##\n\nbody\n")
    assert content.sections[0].heading == "Heading"


def test_mismatched_fence_marker_does_not_close_block() -> None:
    # A ```-opened block containing a ~~~ line stays open; the '#' line inside
    # must not be treated as a heading.
    md = b"# Top\n\n```\n~~~\n# still code\n```\n\ntail\n"
    content = TextExtractor().extract(md)
    headings = [s.heading for s in content.sections]
    assert headings == ["Top"]
    assert "# still code" in content.sections[0].body_text


def test_oversized_text_raises_ingest_error() -> None:
    huge = b"x" * (_MAX_TOTAL_TEXT_CHARS + 1)
    with pytest.raises(KaguraIngestError, match="decompression bomb"):
        TextExtractor().extract(huge)


def test_empty_document_yields_no_sections() -> None:
    content = TextExtractor().extract(b"   \n\n  \n")
    assert content.sections == []
    assert content.page_count is None


def test_title_fallback_strips_query_string() -> None:
    # No headings → title falls back to the URI basename, with ?query stripped.
    content = TextExtractor().extract(b"plain body", source_uri="https://x/notes.txt?token=abc")
    assert content.title == "notes.txt"


def test_supports_declares_text_mimes() -> None:
    assert TextExtractor.supports == frozenset({"text/plain", "text/markdown"})
