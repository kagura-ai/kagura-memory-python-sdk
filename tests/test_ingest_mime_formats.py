"""MIME / format inference for the non-PDF extractor formats (#144).

Distinct from ``test_ingest_mime.py`` (which ``importorskip``s pymupdf):
these exercise the pure detection helpers, which have no optional-dep
dependency now that the registry resolves extractors lazily.
"""

from __future__ import annotations

import pytest

from kagura_memory.exceptions import KaguraIngestError
from kagura_memory.ingest.extractors import DOCX_MIME, EPUB_MIME, PPTX_MIME, XLSX_MIME
from kagura_memory.ingest.fetcher import FetchResult
from kagura_memory.ingest.ingestor import _infer_format, _infer_mime


def _fetch(uri: str, content_type: str = "", body: bytes = b"") -> FetchResult:
    return FetchResult(
        body=body,
        content_type=content_type,
        source_uri=uri,
        source_type="url" if "://" in uri else "file",
        final_url=uri,
        bytes_read=len(body),
    )


@pytest.mark.parametrize(
    ("suffix", "expected"),
    [
        (".txt", "text/plain"),
        (".md", "text/markdown"),
        (".markdown", "text/markdown"),
        (".html", "text/html"),
        (".htm", "text/html"),
        (".docx", DOCX_MIME),
        (".xlsx", XLSX_MIME),
        (".pptx", PPTX_MIME),
        (".epub", EPUB_MIME),
    ],
)
def test_infer_mime_by_suffix(suffix: str, expected: str) -> None:
    assert _infer_mime(_fetch(f"file:///x/doc{suffix}")) == expected


def test_infer_mime_content_type_strips_charset_param() -> None:
    assert _infer_mime(_fetch("x", "text/html; charset=utf-8")) == "text/html"


def test_infer_mime_html_magic_bytes_without_suffix_or_ct() -> None:
    assert _infer_mime(_fetch("file:///tmp/page", "", b"<!DOCTYPE html><html>")) == "text/html"
    assert _infer_mime(_fetch("file:///tmp/page2", "", b"  <html><body>")) == "text/html"


def test_infer_mime_suffix_beats_query_string() -> None:
    assert _infer_mime(_fetch("https://x/report.docx?token=abc")) == DOCX_MIME


def test_infer_mime_unknown_lists_supported_types() -> None:
    with pytest.raises(KaguraIngestError, match="could not determine MIME") as exc:
        _infer_mime(_fetch("file:///tmp/unknown.bin", "", b"\x00\x00\x00"))
    msg = str(exc.value)
    assert "text/markdown" in msg
    assert DOCX_MIME in msg


def test_pdf_magic_bytes_win_over_wrong_content_type() -> None:
    """A real PDF mislabeled with a registered Content-Type still routes to PDF.

    Regression guard: pre-#144 a `.pdf`/%PDF- file served as text/html went to
    PdfExtractor; the new registry must not let the wrong Content-Type win.
    """
    fetched = _fetch("https://x/report", "text/html", b"%PDF-1.7\n...")
    assert _infer_mime(fetched) == "application/pdf"


def test_html_magic_bytes_tolerate_utf8_bom() -> None:
    fetched = _fetch("file:///tmp/page", "", b"\xef\xbb\xbf<!DOCTYPE html><html>")
    assert _infer_mime(fetched) == "text/html"


def test_infer_format_short_labels() -> None:
    assert _infer_format(_fetch("file:///x/a.md")) == "markdown"
    assert _infer_format(_fetch("file:///x/a.docx")) == "docx"
    assert _infer_format(_fetch("file:///x/a.epub")) == "epub"
    assert _infer_format(_fetch("x", "text/html")) == "html"
    # Image content-type still maps to the "image" label.
    assert _infer_format(_fetch("x", "image/png")) == "image"
