"""XLSX extractor backed by ``openpyxl``.

``openpyxl`` is the light, purpose-built choice (``pandas`` would drag in
numpy for no benefit on a pure-parse path). Lazily imported; the error
message points at the ``[ingest-xlsx]`` extra. One section per worksheet;
rows are serialized as a Markdown table.
"""

from __future__ import annotations

import io
from typing import Any, ClassVar

from ...exceptions import KaguraIngestError
from .._types import ExtractedContent, ExtractedSection
from ._util import MAX_TOTAL_TEXT_CHARS as _MAX_TOTAL_TEXT_CHARS
from ._util import filename_title

# XLSX is a ZIP container — cap sheets, rows-per-sheet, and total serialized
# text to defuse decompression bombs and pathological grids.
_MAX_SHEETS = 1_000
_MAX_ROWS_PER_SHEET = 100_000


def _load_openpyxl() -> Any:
    try:
        import openpyxl  # type: ignore[import-untyped]
    except ImportError as e:
        raise KaguraIngestError(
            "openpyxl is not installed. Install with: pip install 'kagura-memory[ingest-xlsx]'"
        ) from e
    return openpyxl


class XlsxExtractor:
    """Structural XLSX extractor.

    Each worksheet becomes one section whose heading is the sheet name and
    whose body is a Markdown table of the populated cells. ``page_range`` is
    ``None`` — spreadsheets are not paginated.
    """

    supports: ClassVar[frozenset[str]] = frozenset(
        {"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}
    )

    def extract(self, source: bytes, source_uri: str | None = None) -> ExtractedContent:
        openpyxl = _load_openpyxl()
        try:
            workbook = openpyxl.load_workbook(io.BytesIO(source), read_only=True, data_only=True)
        except Exception as e:  # noqa: BLE001 - re-raised as our domain error
            raise KaguraIngestError(f"failed to open XLSX: {e}") from e

        try:
            sheets = workbook.worksheets
            if len(sheets) > _MAX_SHEETS:
                raise KaguraIngestError(
                    f"XLSX sheet count {len(sheets)} exceeds limit {_MAX_SHEETS}"
                )
            sections = self._sections(sheets)
        finally:
            workbook.close()

        title = self._title(source_uri)
        return ExtractedContent(title=title, sections=sections, images=[], page_count=None)

    @classmethod
    def _sections(cls, sheets: list[Any]) -> list[ExtractedSection]:
        sections: list[ExtractedSection] = []
        total_chars = 0
        for sheet in sheets:
            table, total_chars = cls._sheet_to_markdown(sheet, total_chars)
            if not table.strip():
                continue
            sections.append(
                ExtractedSection(
                    heading=str(sheet.title),
                    body_text=table,
                    page_range=None,
                    depth=1,
                    anchor=str(sheet.title),
                )
            )
        return sections

    @staticmethod
    def _sheet_to_markdown(sheet: Any, total_chars: int) -> tuple[str, int]:
        """Serialize a sheet to a Markdown table, accumulating ``total_chars``.

        The document-wide character budget is checked per row (not per sheet)
        so a runaway sheet is rejected before its full table is materialized.
        Returns the table text and the updated running total.
        """
        rows: list[list[str]] = []
        for row_idx, row in enumerate(sheet.iter_rows(values_only=True)):
            if row_idx >= _MAX_ROWS_PER_SHEET:
                raise KaguraIngestError(
                    f"XLSX sheet {sheet.title!r} exceeds {_MAX_ROWS_PER_SHEET} rows "
                    "(decompression bomb?)"
                )
            cells = [
                "" if v is None else str(v).replace("|", r"\|").replace("\n", " ") for v in row
            ]
            # Drop trailing empty cells so a sparse sheet does not emit a wall
            # of empty columns.
            while cells and cells[-1] == "":
                cells.pop()
            if cells:
                total_chars += sum(len(c) for c in cells)
                if total_chars > _MAX_TOTAL_TEXT_CHARS:
                    raise KaguraIngestError(
                        f"XLSX total text exceeds {_MAX_TOTAL_TEXT_CHARS} chars "
                        "(decompression bomb?)"
                    )
                rows.append(cells)
        if not rows:
            return "", total_chars

        width = max(len(r) for r in rows)
        rows = [r + [""] * (width - len(r)) for r in rows]
        header = rows[0]
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * width) + " |",
        ]
        for r in rows[1:]:
            lines.append("| " + " | ".join(r) + " |")
        return "\n".join(lines), total_chars

    @staticmethod
    def _title(source_uri: str | None) -> str | None:
        return filename_title(source_uri)
