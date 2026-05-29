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
from ._util import MAX_TOTAL_TEXT_CHARS as _MAX_TOTAL_TEXT_CHARS
from ._util import filename_title

# A decoded char is at most one source byte for UTF-8 (errors="replace" emits
# one U+FFFD per bad byte), so capping the raw byte length also bounds the
# decoded string and stops a multi-GB payload from being materialized first.
_MAX_INPUT_BYTES = _MAX_TOTAL_TEXT_CHARS

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
        # Check the byte length BEFORE decoding so a huge payload is rejected
        # without first allocating an equally huge str.
        if len(source) > _MAX_INPUT_BYTES:
            raise KaguraIngestError(
                f"text document exceeds {_MAX_INPUT_BYTES} bytes "
                f"(got {len(source)}; decompression bomb?)"
            )
        text = source.decode("utf-8", errors="replace")

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
        return filename_title(source_uri)
