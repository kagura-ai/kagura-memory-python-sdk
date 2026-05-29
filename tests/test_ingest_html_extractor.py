"""Tests for ingest.extractors.html — HtmlExtractor (beautifulsoup4)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from kagura_memory.exceptions import KaguraIngestError
from kagura_memory.ingest.extractors.html import _MAX_INPUT_BYTES, HtmlExtractor

pytest.importorskip("bs4", reason="beautifulsoup4 not installed — install [ingest-html] extras")


def test_headings_become_sections() -> None:
    html = b"""
    <html><head><title>Doc Title</title></head>
    <body>
      <p>intro</p>
      <h1>Chapter 1</h1>
      <p>body one</p>
      <h2>Section 1.1</h2>
      <p>body two</p>
    </body></html>
    """
    content = HtmlExtractor().extract(html, source_uri="https://x/page.html")

    assert content.title == "Doc Title"
    headings = [s.heading for s in content.sections]
    assert headings == [None, "Chapter 1", "Section 1.1"]
    assert content.sections[0].body_text == "intro"
    assert "body one" in content.sections[1].body_text
    assert content.sections[2].depth == 2
    assert content.sections[1].page_range is None


def test_script_and_style_content_dropped() -> None:
    html = (
        b"<html><body><h1>H</h1><script>var x=1;</script>"
        b"<style>.a{}</style><p>real</p></body></html>"
    )
    content = HtmlExtractor().extract(html)
    body = content.sections[0].body_text
    assert "real" in body
    assert "var x" not in body
    assert ".a{" not in body


def test_no_headings_yields_single_fallback_section() -> None:
    html = b"<html><body><p>only paragraph text here</p></body></html>"
    content = HtmlExtractor().extract(html)
    assert len(content.sections) == 1
    assert content.sections[0].heading is None
    assert "only paragraph text" in content.sections[0].body_text


def test_title_with_inline_child_tags_is_extracted() -> None:
    # <title> containing an inline tag → .string is None; get_text must still
    # recover the title rather than falling through to <h1>/filename.
    html = b"<html><head><title>Hello <b>World</b></title></head><body><h1>H1</h1></body></html>"
    assert HtmlExtractor().extract(html).title == "Hello World"


def test_title_falls_back_to_first_h1_then_uri() -> None:
    html = b"<html><body><h1>The H1</h1><p>b</p></body></html>"
    assert HtmlExtractor().extract(html).title == "The H1"

    no_heading = b"<html><body><p>x</p></body></html>"
    assert (
        HtmlExtractor().extract(no_heading, source_uri="file:///a/b/page.htm").title == "page.htm"
    )


def test_oversized_input_raises() -> None:
    huge = b"<html><body>" + b"x" * (_MAX_INPUT_BYTES + 1)
    with pytest.raises(KaguraIngestError, match="exceeds"):
        HtmlExtractor().extract(huge)


def test_missing_bs4_raises_install_hint() -> None:
    def fake_import(name: str, *args, **kwargs):
        if name == "bs4":
            raise ImportError("No module named 'bs4'")
        return __import__(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        with pytest.raises(KaguraIngestError, match="ingest-html"):
            HtmlExtractor().extract(b"<html></html>")
