"""DOCX extractor backed by ``python-docx``.

Lazily imports the ``docx`` package on first use; the import error message
points the user at the ``[ingest-docx]`` extra. Sections are derived from
paragraph heading styles (``Heading 1`` .. ``Heading 9``).
"""

from __future__ import annotations

import io
import re
from typing import Any, ClassVar

from ...exceptions import KaguraIngestError
from .._types import ExtractedContent, ExtractedSection
from ._util import MAX_TOTAL_TEXT_CHARS as _MAX_TOTAL_TEXT_CHARS
from ._util import filename_title

# DOCX is a ZIP container — guard against zip bombs by capping the number of
# paragraphs walked and the total decoded text length.
_MAX_PARAGRAPHS = 500_000
_HEADING_STYLE = re.compile(r"^heading\s+(\d+)$", re.IGNORECASE)


def _load_docx() -> Any:
    try:
        # ``python-docx`` installs as the top-level ``docx`` package. This is
        # an absolute import (PEP 328), so it resolves to the dependency, not
        # to this ``extractors/docx.py`` module.
        import docx  # type: ignore[import-untyped]
    except ImportError as e:
        raise KaguraIngestError(
            "python-docx is not installed. Install with: pip install 'kagura-memory[ingest-docx]'"
        ) from e
    return docx


class DocxExtractor:
    """Structural DOCX extractor.

    Each ``Heading N`` paragraph opens a new section at depth ``N``; the
    following body paragraphs form the section text. Content before the
    first heading becomes a leading ``heading=None`` section; a document
    with no headings yields a single fallback section. ``page_range`` is
    ``None`` — Word documents are not paginated at the XML level.
    """

    supports: ClassVar[frozenset[str]] = frozenset(
        {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
    )

    def extract(self, source: bytes, source_uri: str | None = None) -> ExtractedContent:
        docx = _load_docx()
        try:
            document = docx.Document(io.BytesIO(source))
        except Exception as e:  # noqa: BLE001 - re-raised as our domain error
            raise KaguraIngestError(f"failed to open DOCX: {e}") from e

        sections = self._split_sections(document)
        title = self._title(document, sections, source_uri)
        return ExtractedContent(title=title, sections=sections, images=[], page_count=None)

    @classmethod
    def _split_sections(cls, document: Any) -> list[ExtractedSection]:
        sections: list[ExtractedSection] = []
        heading: str | None = None
        depth = 1
        body: list[str] = []
        total_chars = 0
        para_count = 0

        def flush() -> None:
            text = "\n".join(body).strip()
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

        for para in document.paragraphs:
            para_count += 1
            if para_count > _MAX_PARAGRAPHS:
                raise KaguraIngestError(
                    f"DOCX paragraph count exceeds limit {_MAX_PARAGRAPHS} (decompression bomb?)"
                )
            text = para.text or ""
            total_chars += len(text)
            if total_chars > _MAX_TOTAL_TEXT_CHARS:
                raise KaguraIngestError(
                    f"DOCX total text exceeds {_MAX_TOTAL_TEXT_CHARS} chars (decompression bomb?)"
                )
            level = cls._heading_level(para)
            if level is not None:
                flush()
                heading = text.strip() or None
                depth = level
                body = []
            else:
                body.append(text)
        flush()
        return sections

    @staticmethod
    def _heading_level(para: Any) -> int | None:
        style = getattr(para, "style", None)
        name = getattr(style, "name", None)
        if not name:
            return None
        if name.strip().lower() == "title":
            return 1
        m = _HEADING_STYLE.match(name.strip())
        if m:
            return min(9, max(1, int(m.group(1))))
        return None

    @staticmethod
    def _title(
        document: Any, sections: list[ExtractedSection], source_uri: str | None
    ) -> str | None:
        try:
            core_title = (document.core_properties.title or "").strip()
        except Exception:  # noqa: BLE001 - core_properties is best-effort metadata
            core_title = ""
        if core_title:
            return core_title
        for section in sections:
            if section.heading:
                return section.heading
        return filename_title(source_uri)
