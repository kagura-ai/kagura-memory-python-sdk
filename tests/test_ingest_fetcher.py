"""Tests for the SSRF-hardened ingest Fetcher."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from kagura_memory.exceptions import KaguraFetchError
from kagura_memory.ingest.fetcher import Fetcher


def _mock_response(
    *,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
    chunks: list[bytes] | None = None,
) -> MagicMock:
    """Build a MagicMock that mimics httpx.Response enough for Fetcher."""
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.headers = headers or {}
    response.reason_phrase = "OK" if status_code < 400 else "Error"

    # raise_for_status: mimics HTTPStatusError on 4xx/5xx.
    if status_code >= 400:
        err = httpx.HTTPStatusError(f"HTTP {status_code}", request=MagicMock(), response=response)

        def _raise() -> None:
            raise err

        response.raise_for_status = _raise
    else:
        response.raise_for_status = lambda: None

    body_chunks = chunks if chunks is not None else [b""]

    async def _aiter_bytes() -> Any:
        for c in body_chunks:
            yield c

    response.aiter_bytes = _aiter_bytes
    response.aclose = AsyncMock()
    return response


@pytest.mark.asyncio
async def test_rejects_non_http_scheme() -> None:
    async with Fetcher() as fetcher:
        with pytest.raises(KaguraFetchError, match="unsupported scheme"):
            await fetcher.fetch("ftp://example.com/file")


@pytest.mark.asyncio
async def test_rejects_url_credentials() -> None:
    async with Fetcher() as fetcher:
        with patch("socket.getaddrinfo") as mock_getaddrinfo:
            mock_getaddrinfo.return_value = [(0, 0, 0, "", ("8.8.8.8", 0))]
            with pytest.raises(KaguraFetchError, match="credentials"):
                await fetcher.fetch("https://user:pass@example.com/x")


@pytest.mark.asyncio
async def test_rejects_http_when_not_allowed() -> None:
    async with Fetcher() as fetcher:
        with pytest.raises(KaguraFetchError, match="http://"):
            await fetcher.fetch("http://example.com/")


@pytest.mark.asyncio
async def test_rejects_oversized_url() -> None:
    long_url = "https://example.com/" + "x" * 9000
    async with Fetcher() as fetcher:
        with pytest.raises(KaguraFetchError, match="exceeds"):
            await fetcher.fetch(long_url)


@pytest.mark.asyncio
async def test_rejects_blocked_ip_resolution() -> None:
    async with Fetcher() as fetcher:
        with patch("socket.getaddrinfo") as mock_getaddrinfo:
            # Hostname resolves to a private IP — the SSRF guard must reject.
            mock_getaddrinfo.return_value = [(0, 0, 0, "", ("10.0.0.1", 0))]
            with pytest.raises(KaguraFetchError, match="blocked IP"):
                await fetcher.fetch("https://internal.example.com/")


@pytest.mark.asyncio
async def test_rejects_imds_address() -> None:
    """The cloud-metadata IP is the canonical SSRF target."""
    async with Fetcher() as fetcher:
        with patch("socket.getaddrinfo") as mock_getaddrinfo:
            mock_getaddrinfo.return_value = [(0, 0, 0, "", ("169.254.169.254", 0))]
            with pytest.raises(KaguraFetchError, match="blocked IP"):
                await fetcher.fetch("https://aws-metadata.example.com/")


@pytest.mark.asyncio
async def test_happy_path_streams_body() -> None:
    async with Fetcher() as fetcher:
        with patch("socket.getaddrinfo") as mock_getaddrinfo:
            mock_getaddrinfo.return_value = [(0, 0, 0, "", ("93.184.216.34", 0))]
            mock_response = _mock_response(
                headers={"Content-Type": "application/pdf; charset=binary"},
                chunks=[b"%PDF-1.4 ", b"abc", b"def"],
            )
            with patch.object(fetcher._client, "send", new=AsyncMock(return_value=mock_response)):
                result = await fetcher.fetch("https://example.com/doc.pdf")
            assert result.body == b"%PDF-1.4 abcdef"
            assert result.content_type == "application/pdf"
            assert result.source_uri == "https://example.com/doc.pdf"
            assert result.source_type == "url"


@pytest.mark.asyncio
async def test_body_cap_aborts_stream() -> None:
    async with Fetcher(max_bytes=10) as fetcher:
        with patch("socket.getaddrinfo") as mock_getaddrinfo:
            mock_getaddrinfo.return_value = [(0, 0, 0, "", ("93.184.216.34", 0))]
            mock_response = _mock_response(chunks=[b"x" * 5, b"y" * 6])
            with patch.object(fetcher._client, "send", new=AsyncMock(return_value=mock_response)):
                with pytest.raises(KaguraFetchError, match="exceeded max_bytes"):
                    await fetcher.fetch("https://example.com/large")


@pytest.mark.asyncio
async def test_content_length_header_rejects_oversized() -> None:
    async with Fetcher(max_bytes=100) as fetcher:
        with patch("socket.getaddrinfo") as mock_getaddrinfo:
            mock_getaddrinfo.return_value = [(0, 0, 0, "", ("93.184.216.34", 0))]
            mock_response = _mock_response(headers={"Content-Length": "999"}, chunks=[b""])
            with patch.object(fetcher._client, "send", new=AsyncMock(return_value=mock_response)):
                with pytest.raises(KaguraFetchError, match="Content-Length"):
                    await fetcher.fetch("https://example.com/large")


@pytest.mark.asyncio
async def test_redirect_to_blocked_ip_rejected() -> None:
    """A 301 to an internal hostname must be re-validated and rejected."""
    async with Fetcher() as fetcher:
        # First call resolves to public IP and returns a redirect to a new host.
        # Second call (the redirect target) resolves to a blocked IP.
        getaddrinfo_calls: list[str] = []

        def fake_getaddrinfo(host: str, *args: Any, **kwargs: Any) -> Any:
            getaddrinfo_calls.append(host)
            if host == "redirect.example.com":
                return [(0, 0, 0, "", ("10.0.0.1", 0))]
            return [(0, 0, 0, "", ("93.184.216.34", 0))]

        with patch("socket.getaddrinfo", side_effect=fake_getaddrinfo):
            mock_response = _mock_response(
                status_code=301,
                headers={"Location": "https://redirect.example.com/internal"},
            )
            with patch.object(fetcher._client, "send", new=AsyncMock(return_value=mock_response)):
                with pytest.raises(KaguraFetchError, match="blocked IP"):
                    await fetcher.fetch("https://example.com/start")
        assert "example.com" in getaddrinfo_calls
        assert "redirect.example.com" in getaddrinfo_calls


@pytest.mark.asyncio
async def test_local_file_path(tmp_path: Any) -> None:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 minimal content")
    async with Fetcher() as fetcher:
        result = await fetcher.fetch(str(pdf))
        assert result.body.startswith(b"%PDF-")
        assert result.source_type == "file"
        assert result.source_uri.startswith("file://")
        assert str(pdf) in result.source_uri


@pytest.mark.asyncio
async def test_local_file_size_cap(tmp_path: Any) -> None:
    big = tmp_path / "big.bin"
    big.write_bytes(b"x" * 1000)
    async with Fetcher(max_bytes=100) as fetcher:
        with pytest.raises(KaguraFetchError, match="exceeds max_bytes"):
            await fetcher.fetch(str(big))


@pytest.mark.asyncio
async def test_local_file_blocked_system_path() -> None:
    async with Fetcher() as fetcher:
        with pytest.raises(KaguraFetchError):
            # /etc/passwd may not exist as a regular file in some sandboxes,
            # but the path-resolve will still reject before stat. Either
            # outcome is acceptable — the test asserts the rejection.
            await fetcher.fetch("/etc/passwd")
