"""Additional fetcher error-path tests for branches not covered by the main suite.

The main `tests/test_ingest_fetcher.py` covers the happy path and a few key
error branches. This file fills in the remaining error paths: empty source,
non-hostname URLs, redirect chains (no Location header, redirect to unsupported
scheme, redirect to http when http disabled), DNS resolution failures, HTTP
build/send errors, raise_for_status error paths, non-integer Content-Length,
local file errors (resolve, stat, S_ISREG check, read), and the
hostname-resolves-to-no-IPs corner case.
"""

from __future__ import annotations

import socket
import stat as stat_module
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from kagura_memory.exceptions import KaguraFetchError
from kagura_memory.ingest.fetcher import Fetcher


@pytest.mark.asyncio
async def test_fetch_empty_source_raises() -> None:
    async with Fetcher() as fetcher:
        with pytest.raises(KaguraFetchError, match="non-empty string"):
            await fetcher.fetch("")


@pytest.mark.asyncio
async def test_fetch_url_without_hostname_raises() -> None:
    async with Fetcher() as fetcher:
        with pytest.raises(KaguraFetchError, match="no hostname"):
            await fetcher.fetch("https:///path-without-host")


@pytest.mark.asyncio
async def test_fetch_url_dns_resolution_failure_raises() -> None:
    async with Fetcher() as fetcher:
        with patch("socket.getaddrinfo", side_effect=socket.gaierror("name unknown")):
            with pytest.raises(KaguraFetchError, match="hostname resolution failed"):
                await fetcher.fetch("https://does-not-resolve.example/x.pdf")


@pytest.mark.asyncio
async def test_fetch_url_resolution_returns_no_ips_raises() -> None:
    """An empty getaddrinfo result must abort fetch."""
    async with Fetcher() as fetcher:
        with patch("socket.getaddrinfo", return_value=[]):
            with pytest.raises(KaguraFetchError, match="no IP addresses"):
                await fetcher.fetch("https://example.com/x.pdf")


@pytest.mark.asyncio
async def test_fetch_url_network_error_wraps_as_kagura_error() -> None:
    async with Fetcher() as fetcher:
        with patch(
            "socket.getaddrinfo",
            return_value=[(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("8.8.8.8", 0))],
        ):
            with patch.object(
                fetcher._client,
                "send",
                side_effect=httpx.ConnectError("conn refused"),
            ):
                with pytest.raises(KaguraFetchError, match="network error"):
                    await fetcher.fetch("https://example.com/x.pdf")


@pytest.mark.asyncio
async def test_fetch_url_raise_for_status_failure() -> None:
    """An HTTP 5xx surfaces as KaguraFetchError with the status / reason."""
    async with Fetcher() as fetcher:
        response = MagicMock(spec=httpx.Response)
        response.status_code = 503
        response.reason_phrase = "Service Unavailable"
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "503", request=MagicMock(), response=response
        )
        response.aclose = AsyncMock()

        with patch(
            "socket.getaddrinfo",
            return_value=[(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("8.8.8.8", 0))],
        ):
            with patch.object(fetcher._client, "send", return_value=response):
                with pytest.raises(KaguraFetchError, match="HTTP 503"):
                    await fetcher.fetch("https://example.com/x.pdf")


@pytest.mark.asyncio
async def test_fetch_url_redirect_without_location_raises() -> None:
    async with Fetcher() as fetcher:
        response = MagicMock(spec=httpx.Response)
        response.status_code = 302
        response.headers = {}  # no Location
        response.aclose = AsyncMock()

        with patch(
            "socket.getaddrinfo",
            return_value=[(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("8.8.8.8", 0))],
        ):
            with patch.object(fetcher._client, "send", return_value=response):
                with pytest.raises(KaguraFetchError, match="no Location header"):
                    await fetcher.fetch("https://example.com/x.pdf")


@pytest.mark.asyncio
async def test_fetch_url_redirect_to_unsupported_scheme_raises() -> None:
    async with Fetcher() as fetcher:
        response = MagicMock(spec=httpx.Response)
        response.status_code = 302
        response.headers = {"Location": "ftp://elsewhere/file"}
        response.aclose = AsyncMock()

        with patch(
            "socket.getaddrinfo",
            return_value=[(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("8.8.8.8", 0))],
        ):
            with patch.object(fetcher._client, "send", return_value=response):
                with pytest.raises(KaguraFetchError, match="unsupported scheme"):
                    await fetcher.fetch("https://example.com/x.pdf")


@pytest.mark.asyncio
async def test_fetch_url_redirect_to_credentialed_url_raises() -> None:
    """Redirect targets must reject embedded credentials like the initial URL."""
    async with Fetcher() as fetcher:
        response = MagicMock(spec=httpx.Response)
        response.status_code = 302
        response.headers = {"Location": "https://attacker:secret@evil.example/file.pdf"}
        response.aclose = AsyncMock()

        with patch(
            "socket.getaddrinfo",
            return_value=[(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("8.8.8.8", 0))],
        ):
            with patch.object(fetcher._client, "send", return_value=response):
                with pytest.raises(KaguraFetchError, match="embedded credentials"):
                    await fetcher.fetch("https://example.com/x.pdf")


@pytest.mark.asyncio
async def test_fetch_url_redirect_to_http_when_disabled_raises() -> None:
    async with Fetcher(allow_http=False) as fetcher:
        response = MagicMock(spec=httpx.Response)
        response.status_code = 302
        response.headers = {"Location": "http://insecure/file"}
        response.aclose = AsyncMock()

        with patch(
            "socket.getaddrinfo",
            return_value=[(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("8.8.8.8", 0))],
        ):
            with patch.object(fetcher._client, "send", return_value=response):
                with pytest.raises(KaguraFetchError, match="http://.*disabled"):
                    await fetcher.fetch("https://example.com/x.pdf")


@pytest.mark.asyncio
async def test_fetch_url_too_many_redirects_raises() -> None:
    """Exhausting the redirect budget surfaces 'too many redirects'."""

    async def fake_send(request, stream=False):
        response = MagicMock(spec=httpx.Response)
        response.status_code = 302
        response.headers = {"Location": "https://example.com/next"}
        response.aclose = AsyncMock()
        return response

    async with Fetcher(max_redirects=2) as fetcher:
        with patch(
            "socket.getaddrinfo",
            return_value=[(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("8.8.8.8", 0))],
        ):
            with patch.object(fetcher._client, "send", side_effect=fake_send):
                with pytest.raises(KaguraFetchError, match="too many redirects"):
                    await fetcher.fetch("https://example.com/x.pdf")


@pytest.mark.asyncio
async def test_fetch_url_non_integer_content_length_is_ignored() -> None:
    """A garbage Content-Length should not crash — fall through to streaming cap."""

    async def fake_iter(self):
        yield b"hello"

    async with Fetcher(max_bytes=1024) as fetcher:
        response = MagicMock(spec=httpx.Response)
        response.status_code = 200
        response.headers = {"Content-Length": "not-an-int", "Content-Type": "text/plain"}
        response.raise_for_status = MagicMock()
        response.aiter_bytes = fake_iter.__get__(response, type(response))
        response.aclose = AsyncMock()

        with patch(
            "socket.getaddrinfo",
            return_value=[(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("8.8.8.8", 0))],
        ):
            with patch.object(fetcher._client, "send", return_value=response):
                result = await fetcher.fetch("https://example.com/x.txt")

        assert result.body == b"hello"
        assert result.content_type == "text/plain"


# ---------------------------------------------------------------------------
# Local file path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_local_resolve_oserror_raises() -> None:
    """Path.resolve() raising OSError (not FileNotFoundError) is wrapped."""
    async with Fetcher() as fetcher:
        with patch.object(Path, "resolve", side_effect=OSError("permission denied")):
            with pytest.raises(KaguraFetchError, match="file resolve failed"):
                await fetcher.fetch("/some/path")


@pytest.mark.asyncio
async def test_fetch_local_stat_oserror_raises(tmp_path: Path) -> None:
    target = tmp_path / "data.pdf"
    target.write_bytes(b"hi")
    async with Fetcher() as fetcher:
        with patch.object(Path, "stat", side_effect=OSError("EIO")):
            with pytest.raises(KaguraFetchError, match="stat failed"):
                await fetcher.fetch(str(target))


@pytest.mark.asyncio
async def test_fetch_local_not_regular_file_raises(tmp_path: Path) -> None:
    """A directory (or symlink to non-file) must be rejected."""
    target = tmp_path / "data.pdf"
    target.write_bytes(b"hi")

    async with Fetcher() as fetcher:
        # Force stat to claim it's a directory.
        fake_stat = MagicMock()
        fake_stat.st_mode = stat_module.S_IFDIR | 0o755
        fake_stat.st_size = 0
        with patch.object(Path, "stat", return_value=fake_stat):
            with pytest.raises(KaguraFetchError, match="not a regular file"):
                await fetcher.fetch(str(target))


@pytest.mark.asyncio
async def test_fetch_local_read_oserror_raises(tmp_path: Path) -> None:
    target = tmp_path / "data.pdf"
    target.write_bytes(b"hi")

    async with Fetcher() as fetcher:
        with patch.object(Path, "read_bytes", side_effect=OSError("EIO read")):
            with pytest.raises(KaguraFetchError, match="file read failed"):
                await fetcher.fetch(str(target))
