"""Tests for the shared SectionBuilder used by heading-driven extractors (#204).

SectionBuilder centralizes the "flush on heading, accumulate body, flush at end"
state machine that the text/Markdown, HTML, and DOCX extractors share. These
tests pin the flush rule and the per-format body-join seam; the extractor-level
behavior is covered (unchanged) by the existing test_ingest_*_extractor suites.
"""

from __future__ import annotations

import re

from kagura_memory.ingest.extractors._util import SectionBuilder


def test_empty_builder_yields_no_sections() -> None:
    assert SectionBuilder().finish() == []


def test_body_only_no_heading_yields_one_leading_section() -> None:
    b = SectionBuilder()
    b.add_body("hello")
    b.add_body("world")
    sections = b.finish()
    assert len(sections) == 1
    s = sections[0]
    assert s.heading is None
    assert s.depth == 1  # leading section defaults to depth 1
    assert s.anchor is None
    assert s.page_range is None
    assert s.body_text == "hello\nworld"


def test_heading_with_empty_body_is_emitted() -> None:
    """flush rule: a non-empty heading emits even with no body."""
    b = SectionBuilder()
    b.add_heading("Intro", 2)
    sections = b.finish()
    assert len(sections) == 1
    assert sections[0].heading == "Intro"
    assert sections[0].body_text == ""
    assert sections[0].depth == 2
    assert sections[0].anchor == "Intro"


def test_neither_body_nor_heading_yields_no_section() -> None:
    """flush rule: an empty body AND a None heading emits nothing."""
    b = SectionBuilder()
    b.add_heading(None, 1)  # heading explicitly None, no body
    assert b.finish() == []


def test_leading_body_then_heading_yields_two_sections() -> None:
    b = SectionBuilder()
    b.add_body("preamble")
    b.add_heading("Section 1", 1)
    b.add_body("content")
    sections = b.finish()
    assert [(s.heading, s.body_text, s.depth) for s in sections] == [
        (None, "preamble", 1),
        ("Section 1", "content", 1),
    ]


def test_heading_before_any_body_has_no_spurious_leading_section() -> None:
    """A heading as the very first item must not emit an empty leading section."""
    b = SectionBuilder()
    b.add_heading("Top", 1)
    b.add_body("body")
    sections = b.finish()
    assert len(sections) == 1
    assert sections[0].heading == "Top"
    assert sections[0].body_text == "body"


def test_depth_and_anchor_track_each_heading() -> None:
    b = SectionBuilder()
    b.add_heading("H1", 1)
    b.add_body("a")
    b.add_heading("H3", 3)
    b.add_body("b")
    sections = b.finish()
    assert [(s.depth, s.anchor) for s in sections] == [(1, "H1"), (3, "H3")]


def test_default_join_is_newline_join_then_strip() -> None:
    b = SectionBuilder()
    b.add_body("  line1  ")
    b.add_body("line2")
    # newline-join then outer strip (inner whitespace preserved)
    assert b.finish()[0].body_text == "line1  \nline2"


def test_custom_join_is_applied() -> None:
    """HTML passes a whitespace-collapsing join; the builder must honor it."""
    ws = re.compile(r"\s+")
    b = SectionBuilder(join=lambda parts: ws.sub(" ", " ".join(parts)).strip())
    b.add_body("a\n   b")
    b.add_body("  c ")
    assert b.finish()[0].body_text == "a b c"


def test_finish_is_terminal_snapshot() -> None:
    """finish() flushes the current section; calling it returns the built list."""
    b = SectionBuilder()
    b.add_heading("Only", 1)
    b.add_body("x")
    first = b.finish()
    assert len(first) == 1
    assert first[0].heading == "Only"
