"""Tests for the structural-first chunker."""

from __future__ import annotations

from kagura_memory.ingest._types import ExtractedContent, ExtractedSection
from kagura_memory.ingest.chunker import chunk


def _section(text: str, heading: str | None = "S", depth: int = 1) -> ExtractedSection:
    return ExtractedSection(heading=heading, body_text=text, depth=depth, anchor=heading)


def test_empty_content_returns_no_chunks() -> None:
    assert chunk(ExtractedContent(title=None, sections=[])) == []


def test_single_short_section_passes_through() -> None:
    content = ExtractedContent(
        title="doc",
        sections=[_section("short body")],
    )
    chunks = chunk(content, max_tokens=1000)
    assert len(chunks) == 1
    assert chunks[0].text == "short body"
    assert chunks[0].heading == "S"
    assert chunks[0].section_index == 0


def test_multi_sections_preserve_order_and_index() -> None:
    content = ExtractedContent(
        title="doc",
        sections=[
            _section("first body", heading="A"),
            _section("second body", heading="B"),
            _section("third body", heading="C"),
        ],
    )
    chunks = chunk(content, max_tokens=1000)
    assert [c.section_index for c in chunks] == [0, 1, 2]
    assert [c.heading for c in chunks] == ["A", "B", "C"]


def test_long_section_splits_with_overlap() -> None:
    long_text = "abcdef " * 5000  # ~30000 chars → many tokens
    content = ExtractedContent(title="doc", sections=[_section(long_text, heading="L")])
    chunks = chunk(content, max_tokens=200, overlap_tokens=20)
    # Heading should appear only on the first sub-chunk.
    assert chunks[0].heading == "L"
    assert len(chunks) > 1
    for c in chunks[1:]:
        assert c.heading is None
        assert c.anchor is None
    # All chunks share the same source section_index? No — by spec,
    # section_index is the index in the OUTPUT chunk list. Confirm.
    assert [c.section_index for c in chunks] == list(range(len(chunks)))


def test_no_sections_returns_empty() -> None:
    # An ExtractedContent with empty sections is the chunker's
    # "no-structure" signal; the orchestrator may treat it differently.
    content = ExtractedContent(title="doc", sections=[])
    assert chunk(content) == []


def test_whitespace_only_body_skipped() -> None:
    content = ExtractedContent(
        title="doc",
        sections=[_section("    \n\n  \t  ", heading="empty")],
    )
    assert chunk(content) == []
