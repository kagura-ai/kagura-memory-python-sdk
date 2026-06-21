"""HTML extractor backed by ``beautifulsoup4`` (stdlib ``html.parser``).

Lazily imports ``bs4`` on first use; the import error message points the
user at the ``[ingest-html]`` extra. Sections are derived from ``<h1>``..
``<h6>`` headings; ``<script>``/``<style>`` content is dropped.
"""

from __future__ import annotations

import re
from typing import Any, ClassVar

from ...exceptions import KaguraIngestError
from .._types import ExtractedContent, ExtractedSection
from ._util import MAX_TOTAL_TEXT_CHARS as _MAX_TOTAL_TEXT_CHARS
from ._util import SectionBuilder, filename_title

# Cap on input bytes (HTML is not compressed, but a hostile page can still be
# arbitrarily large). Derived from the shared text ceiling so the safety cap
# has a single source of truth — markup bytes are an upper bound on the text
# the parser will surface.
_MAX_INPUT_BYTES = _MAX_TOTAL_TEXT_CHARS
_HEADING_NAMES = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_HEADING_RE = re.compile(r"^h[1-6]$")
_WS = re.compile(r"\s+")


def _load_bs4() -> Any:
    try:
        from bs4 import BeautifulSoup  # type: ignore[import-untyped]
    except ImportError as e:
        raise KaguraIngestError(
            "beautifulsoup4 is not installed. "
            "Install with: pip install 'kagura-memory[ingest-html]'"
        ) from e
    return BeautifulSoup


class HtmlExtractor:
    """Structural HTML extractor.

    Each ``<h1>``..``<h6>`` opens a new section; text between headings (in
    document order) is the section body. Content before the first heading
    becomes a leading ``heading=None`` section. A document with no headings
    yields a single fallback section. ``page_range`` is ``None``.
    """

    supports: ClassVar[frozenset[str]] = frozenset({"text/html"})

    def extract(self, source: bytes, source_uri: str | None = None) -> ExtractedContent:
        if len(source) > _MAX_INPUT_BYTES:
            raise KaguraIngestError(
                f"HTML document exceeds {_MAX_INPUT_BYTES} bytes (got {len(source)})"
            )
        beautiful_soup = _load_bs4()
        markup = source.decode("utf-8", errors="replace")
        soup = beautiful_soup(markup, "html.parser")

        for tag in soup(["script", "style", "noscript", "template"]):
            tag.decompose()

        title = self._title(soup, source_uri)
        sections = self._split_sections(soup)
        return ExtractedContent(title=title, sections=sections, images=[], page_count=None)

    @staticmethod
    def _split_sections(soup: Any) -> list[ExtractedSection]:
        from bs4 import NavigableString, Tag  # type: ignore[import-untyped]

        # HTML collapses runs of whitespace (including newlines) to single
        # spaces, unlike the default newline-join used by text/DOCX.
        builder = SectionBuilder(join=lambda parts: _WS.sub(" ", " ".join(parts)).strip())

        # Walk the <body> only so <head>/<title> text never leaks into the
        # leading section. Fragments without a <body> fall back to the root.
        root = soup.body or soup
        for node in root.descendants:
            if isinstance(node, Tag) and node.name in _HEADING_NAMES:
                builder.add_heading(node.get_text(" ", strip=True) or None, int(node.name[1]))
            elif isinstance(node, NavigableString):
                # Skip strings that belong to a heading (already captured) so
                # heading text is not duplicated into the following body.
                if node.find_parent(_HEADING_RE) is not None:
                    continue
                builder.add_body(str(node))
        return builder.finish()

    @staticmethod
    def _title(soup: Any, source_uri: str | None) -> str | None:
        if soup.title:
            # get_text (not .string) so a <title> containing inline tags or
            # comments — where .string is None — still yields the title text.
            title = soup.title.get_text(" ", strip=True)
            if title:
                return title
        first_h1 = soup.find("h1")
        if first_h1:
            heading = first_h1.get_text(" ", strip=True)
            if heading:
                return heading
        return filename_title(source_uri)
