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
from ._util import filename_title

# Cap on input bytes (HTML is not compressed, but a hostile page can still
# be arbitrarily large). Mirrors the spirit of pdf's text cap.
_MAX_INPUT_BYTES = 50_000_000  # ~50 MB of markup
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

        sections: list[ExtractedSection] = []
        heading: str | None = None
        depth = 1
        body: list[str] = []

        def flush() -> None:
            text = _WS.sub(" ", " ".join(body)).strip()
            if text or heading:
                sections.append(
                    ExtractedSection(
                        heading=heading,
                        body_text=text,
                        page_range=None,
                        depth=depth,
                        anchor=heading,
                    )
                )

        # Walk the <body> only so <head>/<title> text never leaks into the
        # leading section. Fragments without a <body> fall back to the root.
        root = soup.body or soup
        for node in root.descendants:
            if isinstance(node, Tag) and node.name in _HEADING_NAMES:
                flush()
                heading = node.get_text(" ", strip=True) or None
                depth = int(node.name[1])
                body = []
            elif isinstance(node, NavigableString):
                # Skip strings that belong to a heading (already captured) so
                # heading text is not duplicated into the following body.
                if node.find_parent(_HEADING_RE) is not None:
                    continue
                body.append(str(node))
        flush()
        return sections

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
