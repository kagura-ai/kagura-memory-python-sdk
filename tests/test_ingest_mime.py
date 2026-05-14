"""Tests for the MIME / format inference helpers in ingest.ingestor.

These exercise the `_uri_path_lower`, `_infer_format`, and `_infer_mime`
helpers — corner cases like query strings, fragments, and missing
Content-Type that were not covered by the orchestrator-level tests.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pymupdf", reason="pymupdf not installed — install [ingest-pdf] extras")

from kagura_memory.exceptions import KaguraIngestError  # noqa: E402
from kagura_memory.ingest.fetcher import FetchResult  # noqa: E402
from kagura_memory.ingest.ingestor import (  # noqa: E402
    _infer_format,
    _infer_mime,
    _uri_path_lower,
)


def _fetch(uri: str, content_type: str = "", body: bytes = b"") -> FetchResult:
    return FetchResult(
        body=body,
        content_type=content_type,
        source_uri=uri,
        source_type="url" if "://" in uri else "file",
        final_url=uri,
        bytes_read=len(body),
    )


def test_uri_path_lower_strips_query_and_fragment() -> None:
    assert _uri_path_lower("https://example.com/Report.PDF?token=abc#frag") == "/report.pdf"


def test_uri_path_lower_for_bare_path() -> None:
    assert _uri_path_lower("/tmp/My.Report.PDF") == "/tmp/my.report.pdf"


def test_infer_mime_uses_content_type_when_present() -> None:
    assert _infer_mime(_fetch("anything", "application/pdf")) == "application/pdf"


def test_infer_mime_uses_extension_with_query_string() -> None:
    """The pre-fix bug: ``?token=...`` blocked extension detection."""
    fetched = _fetch("https://example.com/report.pdf?token=abc", "")
    assert _infer_mime(fetched) == "application/pdf"


def test_infer_mime_uses_extension_with_fragment() -> None:
    fetched = _fetch("https://example.com/report.pdf#section-1", "")
    assert _infer_mime(fetched) == "application/pdf"


def test_infer_mime_magic_byte_sniff_for_no_extension_no_ct() -> None:
    """Local PDFs with no extension and no Content-Type still sniff."""
    fetched = _fetch("file:///tmp/anonymous", "", b"%PDF-1.4\n")
    assert _infer_mime(fetched) == "application/pdf"


def test_infer_mime_unsupported_raises() -> None:
    fetched = _fetch("file:///tmp/unknown.bin", "", b"\x00\x00\x00\x00\x00")
    with pytest.raises(KaguraIngestError, match="could not determine MIME"):
        _infer_mime(fetched)


def test_infer_format_pdf_with_query_string() -> None:
    assert _infer_format(_fetch("https://example.com/r.pdf?x=1", "")) == "pdf"


def test_infer_format_image_from_content_type() -> None:
    assert _infer_format(_fetch("ignored", "image/png")) == "image"
