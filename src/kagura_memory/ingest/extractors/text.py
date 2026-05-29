"""Plain-text and Markdown extractor — stdlib only, no optional deps.

Handles ``text/plain`` and ``text/markdown``. Markdown ATX headings
(``#`` .. ``######``) become section boundaries; plain text with no
headings collapses to a single fallback section. Fenced code blocks
(```` ``` ```` / ``~~~``) are skipped during heading detection so a ``#``
comment inside a code sample is not mistaken for a heading.
"""

from __future__ import annotations

import re
from typing import ClassVar

from ...exceptions import KaguraIngestError
from .._types import ExtractedContent, ExtractedSection

# Cap on decoded character count (mirrors pdf._MAX_TOTAL_TEXT_CHARS). Text
# files are not compressed, but a multi-GB log still needs a ceiling.
_MAX_TOTAL_TEXT_CHARS = 50_000_000  # ~50 MB of text

_ATX_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```+|~~~+)")


class TextExtractor:
    """Structural extractor for plain text and Markdown.

    Sections are derived from Markdown ATX headings. Content before the
    first heading (or the whole document, when there are no headings) is
    emitted as a leading section with ``heading=None`` so the chunker can
    token-window-split it. ``page_range`` is ``None`` — text is not
    paginated.
    """

    supports: ClassVar[frozenset[str]] = frozenset({"text/plain", "text/markdown"})

    def extract(self, source: bytes, source_uri: str | None = None) -> ExtractedContent:
        text = source.decode("utf-8", errors="replace")
        if len(text) > _MAX_TOTAL_TEXT_CHARS:
            raise KaguraIngestError(
                f"text document exceeds {_MAX_TOTAL_TEXT_CHARS} chars "
                f"(got {len(text)}; decompression bomb?)"
            )

        sections = self._split_sections(text)
        title = self._title(sections, source_uri)
        return ExtractedContent(title=title, sections=sections, images=[], page_count=None)

    @staticmethod
    def _split_sections(text: str) -> list[ExtractedSection]:
        sections: list[ExtractedSection] = []
        heading: str | None = None
        depth = 1
        body: list[str] = []
        in_fence = False

        def flush() -> None:
            joined = "\n".join(body).strip()
            if joined or heading:
                sections.append(
                    ExtractedSection(
                        heading=heading,
                        body_text=joined,
                        page_range=None,
                        depth=depth,
                        anchor=heading,
                    )
                )

        for line in text.splitlines():
            if _FENCE.match(line):
                in_fence = not in_fence
                body.append(line)
                continue
            m = None if in_fence else _ATX_HEADING.match(line)
            if m:
                flush()
                heading = m.group(2).strip() or None
                depth = len(m.group(1))
                body = []
            else:
                body.append(line)
        flush()
        return sections

    @staticmethod
    def _title(sections: list[ExtractedSection], source_uri: str | None) -> str | None:
        for section in sections:
            if section.depth == 1 and section.heading:
                return section.heading
        if source_uri:
            return source_uri.rsplit("/", 1)[-1] or source_uri
        return None
