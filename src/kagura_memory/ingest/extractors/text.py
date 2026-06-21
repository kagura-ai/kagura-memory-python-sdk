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
from ._util import SectionBuilder, filename_title

# Every decoded character costs at least one UTF-8 byte (and errors="replace"
# emits exactly one U+FFFD per undecodable byte), so the decoded length never
# exceeds len(source). Capping the raw byte length therefore bounds the decoded
# string and stops a multi-GB payload from being materialized first.
_MAX_INPUT_BYTES = _MAX_TOTAL_TEXT_CHARS

# ATX heading: 1-6 leading '#', a space, the text, then an OPTIONAL closing '#'
# run that — per CommonMark — must be preceded by whitespace to count as a
# closer. Without the leading `\s+`, a heading whose text ends in '#' (e.g.
# "# C#") would lose that character.
_ATX_HEADING = re.compile(r"^(#{1,6})\s+(.*?)(?:\s+#+)?\s*$")
# Fenced code block delimiter: a run of >=3 backticks or tildes, capturing the
# marker run (group 1) and any trailing info string (group 2).
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})\s*(.*)$")


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
        builder = SectionBuilder()  # default join: "\n".join + strip
        fence_marker: str | None = None  # the opening run (e.g. "```") while inside a fence

        for line in text.splitlines():
            fence = _FENCE.match(line)
            if fence_marker is None:
                if fence:
                    fence_marker = fence.group(1)
                    builder.add_body(line)
                    continue
            else:
                # Inside a fence: only a run of the SAME char, at least as long
                # as the opener and with no info string, closes it (CommonMark).
                if fence:
                    marker, info = fence.group(1), fence.group(2)
                    if (
                        marker[0] == fence_marker[0]
                        and len(marker) >= len(fence_marker)
                        and not info.strip()
                    ):
                        fence_marker = None
                builder.add_body(line)
                continue
            m = _ATX_HEADING.match(line)
            if m:
                builder.add_heading(m.group(2).strip() or None, len(m.group(1)))
            else:
                builder.add_body(line)
        return builder.finish()

    @staticmethod
    def _title(sections: list[ExtractedSection], source_uri: str | None) -> str | None:
        for section in sections:
            if section.depth == 1 and section.heading:
                return section.heading
        return filename_title(source_uri)
