"""Tests for ingest.extractors.xlsx — XlsxExtractor (openpyxl)."""

from __future__ import annotations

import io
from unittest.mock import patch

import pytest

from kagura_memory.exceptions import KaguraIngestError
from kagura_memory.ingest.extractors.xlsx import XlsxExtractor

openpyxl = pytest.importorskip(
    "openpyxl", reason="openpyxl not installed — install [ingest-xlsx] extras"
)


def _make_xlsx(sheets: dict[str, list[list]]) -> bytes:
    wb = openpyxl.Workbook()
    # Remove the default sheet so only our named sheets remain.
    wb.remove(wb.active)
    for name, rows in sheets.items():
        ws = wb.create_sheet(title=name)
        for row in rows:
            ws.append(row)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def test_one_section_per_sheet_with_markdown_table() -> None:
    data = _make_xlsx(
        {
            "People": [["name", "age"], ["alice", 30], ["bob", 25]],
            "Empty": [],
        }
    )
    content = XlsxExtractor().extract(data, source_uri="file:///x/book.xlsx")

    # Empty sheet produces no section.
    assert [s.heading for s in content.sections] == ["People"]
    table = content.sections[0].body_text
    assert "| name | age |" in table
    assert "| --- | --- |" in table
    assert "| alice | 30 |" in table
    assert content.sections[0].page_range is None
    assert content.title == "book.xlsx"


def test_pipe_and_newline_in_cells_are_escaped() -> None:
    # Two rows so escaping is exercised in a body data row, not only the header.
    data = _make_xlsx({"S": [["h1", "h2"], ["a|b", "c\nd"]]})
    table = XlsxExtractor().extract(data).sections[0].body_text
    assert r"a\|b" in table
    assert "c d" in table  # newline replaced by space
    assert "\n" in table.split("c d")[0]  # the data row is a distinct table line


def test_open_failure_becomes_ingest_error() -> None:
    with pytest.raises(KaguraIngestError, match="failed to open XLSX"):
        XlsxExtractor().extract(b"not a real xlsx")


def test_sheet_count_cap_raises() -> None:
    data = _make_xlsx({"A": [[1]], "B": [[2]]})
    with patch("kagura_memory.ingest.extractors.xlsx._MAX_SHEETS", 1):
        with pytest.raises(KaguraIngestError, match="sheet count"):
            XlsxExtractor().extract(data)


def test_row_cap_raises() -> None:
    data = _make_xlsx({"S": [["a"], ["b"], ["c"]]})
    with patch("kagura_memory.ingest.extractors.xlsx._MAX_ROWS_PER_SHEET", 1):
        with pytest.raises(KaguraIngestError, match="rows"):
            XlsxExtractor().extract(data)


def test_total_text_cap_triggers_decomp_bomb_error() -> None:
    data = _make_xlsx({"S": [["aaaa", "bbbb"], ["cccc", "dddd"]]})
    with patch("kagura_memory.ingest.extractors.xlsx._MAX_TOTAL_TEXT_CHARS", 5):
        with pytest.raises(KaguraIngestError, match="total text exceeds"):
            XlsxExtractor().extract(data)


def test_table_cell_cap_rejects_wide_sparse_table() -> None:
    # One wide row forces every row to pad to that width; the cell cap must
    # reject it before the padding allocation, even though cell text is small.
    wide = ["x"] + [None] * 8 + ["y"]  # 10 columns, mostly empty
    data = _make_xlsx({"S": [wide, wide, wide]})
    with patch("kagura_memory.ingest.extractors.xlsx._MAX_TABLE_CELLS", 5):
        with pytest.raises(KaguraIngestError, match="exceeds .* cells"):
            XlsxExtractor().extract(data)


def test_missing_openpyxl_raises_install_hint() -> None:
    def fake_import(name: str, *args, **kwargs):
        if name == "openpyxl":
            raise ImportError("No module named 'openpyxl'")
        return __import__(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        with pytest.raises(KaguraIngestError, match="ingest-xlsx"):
            XlsxExtractor().extract(b"x")
