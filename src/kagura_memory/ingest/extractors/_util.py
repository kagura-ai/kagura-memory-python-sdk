"""Shared helpers for structural extractors.

Keeps the decompression-bomb ceiling, the source-URI title fallback, and the
heading-driven section state machine in one place so every extractor enforces
the same safety policy, derives titles identically, and applies one flush rule.
"""

from __future__ import annotations

from collections.abc import Callable
from urllib.parse import urlsplit

from .._types import ExtractedSection

# Decompression-bomb ceiling on *decoded* text, shared by all extractors. A
# single source of truth so tuning this security limit touches one line.
MAX_TOTAL_TEXT_CHARS = 50_000_000  # ~50 MB of decoded text


def _default_join(parts: list[str]) -> str:
    """Newline-join body fragments and strip the outer whitespace.

    The default body normalizer for line/paragraph formats (text, Markdown,
    DOCX). HTML injects a whitespace-collapsing join instead.
    """
    return "\n".join(parts).strip()


class SectionBuilder:
    """Accumulate heading-delimited sections for heading-driven extractors.

    Centralizes the "flush on heading, accumulate body, flush at end" state
    machine shared by the text/Markdown, HTML, and DOCX extractors — together
    with the single flush rule (**emit a section only when its joined body text
    or its heading is non-empty**) and the :class:`ExtractedSection`
    construction. The per-format body join/normalize is injected via ``join``
    (default: newline-join then strip; HTML passes a whitespace-collapsing
    join).

    Content before the first heading is emitted as a leading section with
    ``heading=None`` and ``depth=1``; ``page_range`` is always ``None`` because
    these formats are not paginated. ``anchor`` mirrors the heading text.

    Typical use::

        builder = SectionBuilder()
        for item in source:
            if is_heading(item):
                builder.add_heading(heading_text(item), heading_depth(item))
            else:
                builder.add_body(body_text(item))
        return builder.finish()
    """

    def __init__(self, join: Callable[[list[str]], str] = _default_join) -> None:
        self._join = join
        self._sections: list[ExtractedSection] = []
        self._heading: str | None = None
        self._depth = 1
        self._body: list[str] = []

    def add_heading(self, text: str | None, depth: int) -> None:
        """Flush the current section, then begin a new one at ``text``/``depth``."""
        self._flush()
        self._heading = text
        self._depth = depth
        self._body = []

    def add_body(self, text: str) -> None:
        """Append a raw body fragment to the current section (joined on flush)."""
        self._body.append(text)

    def finish(self) -> list[ExtractedSection]:
        """Flush the final section and return all accumulated sections."""
        self._flush()
        return self._sections

    def _flush(self) -> None:
        body_text = self._join(self._body)
        if body_text or self._heading:
            self._sections.append(
                ExtractedSection(
                    heading=self._heading,
                    body_text=body_text,
                    page_range=None,
                    depth=self._depth,
                    anchor=self._heading,
                )
            )


def filename_title(source_uri: str | None) -> str | None:
    """Best-effort document title from a source URI's basename.

    Strips query/fragment (so ``report.pdf?token=...`` → ``report.pdf``),
    matching the upload-path filename derivation used elsewhere.
    """
    if not source_uri:
        return None
    path = urlsplit(source_uri).path or source_uri
    return path.rsplit("/", 1)[-1] or source_uri
