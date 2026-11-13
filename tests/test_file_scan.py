"""Upload malware scanning (memory-cloud v0.160.0+, #565; SDK #337).

Every reply below is shaped like memory-cloud v0.161.0's: ``FileObjectOut``
with ``scan_status`` (``backend/src/api/routes/files.py``), and the
download-url refusals ``FileScanPendingError`` (409 ``SCAN-001``,
``Retry-After: 30``, ``details {file_id, scan_status: "pending"}``) and
``FileBlockedError`` (409 ``SCAN-002``, ``details {file_id, scan_status}``)
from ``backend/src/utils/exceptions.py`` as ``memory_cloud_exception_handler``
renders them (``{error, message, details}``, the exception's
``response_headers``). Requests go through a real ``httpx.MockTransport``, so
headers and bodies are read as the SDK reads them from a server.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from click.testing import CliRunner

import kagura_memory
from kagura_memory import (
    FileListResponse,
    FileObject,
    FilesClient,
    KaguraConnectionError,
    KaguraFileBlockedError,
    KaguraFileScanPendingError,
    KaguraNotFoundError,
)
from kagura_memory.cli import main

CTX = "00000000-0000-0000-0000-000000000001"
FILE_ID = "10000000-0000-0000-0000-000000000002"
BODY = b"hello kagura files"

_PENDING_MESSAGE = "This file is still being checked for malware. Try again shortly."
_BLOCKED_MESSAGE = "This file cannot be downloaded: the malware scan blocked it."


def _file(scan_status: str | None = "clean", **extra: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": FILE_ID,
        "workspace_id": CTX,
        "context_id": None,
        "filename": "hello.txt",
        "content_type": "text/plain",
        "size_bytes": len(BODY),
        "sha256": "0" * 64,
        "status": "uploaded",
        "created_at": "2026-11-14T00:00:00Z",
        "uploaded_at": "2026-11-14T00:00:01Z",
        **extra,
    }
    if scan_status is not None:
        row["scan_status"] = scan_status
    return row


def _pending(*, retry_after: str | None = "30") -> httpx.Response:
    headers = {} if retry_after is None else {"Retry-After": retry_after}
    return httpx.Response(
        409,
        headers=headers,
        json={
            "error": "SCAN-001",
            "message": _PENDING_MESSAGE,
            "details": {"file_id": FILE_ID, "scan_status": "pending"},
        },
    )


def _blocked(scan_status: str) -> httpx.Response:
    return httpx.Response(
        409,
        json={
            "error": "SCAN-002",
            "message": _BLOCKED_MESSAGE,
            "details": {"file_id": FILE_ID, "scan_status": scan_status},
        },
    )


def _client(handler: Any) -> FilesClient:
    client = FilesClient(api_key="test", base_url="https://example.com")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client._upload_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


async def _download(response: httpx.Response) -> Any:
    client = _client(lambda request: response)
    try:
        return await client.download_url(FILE_ID, context_id=CTX)
    finally:
        await client.close()


# ============================================================================
# FileObject.scan_status
# ============================================================================


def test_file_object_reads_scan_status_and_takes_an_unknown_value():
    assert FileObject.model_validate(_file("pending")).scan_status == "pending"
    # Plain str: a value a newer server adds still parses.
    assert FileObject.model_validate(_file("quarantined")).scan_status == "quarantined"


def test_file_object_without_scan_status_reads_none():
    """A server before v0.160.0 sends no ``scan_status``."""
    assert FileObject.model_validate(_file(None)).scan_status is None


@pytest.mark.asyncio
async def test_upload_confirm_keeps_scan_status():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/files/reserve":
            return httpx.Response(
                201,
                json={
                    "file_id": FILE_ID,
                    "upload_url": "https://r2.example.com/bucket/key?X-Amz-Signature=x",
                    "expires_at": "2026-11-14T00:05:00Z",
                },
            )
        if request.url.host == "r2.example.com":
            return httpx.Response(200)
        assert request.url.path == f"/api/v1/files/{FILE_ID}/confirm"
        return httpx.Response(200, json=_file("pending"))

    client = _client(handler)
    try:
        result = await client.upload(context_id=CTX, source=BODY, filename="hello.txt")
    finally:
        await client.close()
    assert result.scan_status == "pending"


@pytest.mark.asyncio
async def test_list_keeps_scan_status():
    rows = [_file("clean"), _file("unscanned", id="another"), _file(None, id="old")]
    client = _client(lambda request: httpx.Response(200, json=rows))
    try:
        result = await client.list(context_id=CTX)
    finally:
        await client.close()
    assert [f.scan_status for f in result.files] == ["clean", "unscanned", None]


@pytest.mark.asyncio
async def test_upload_dedup_hit_keeps_a_pending_scan_status():
    """The dedup ``409 RES-002`` hands back a ``pending`` file (never a blocked one)."""
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(
            409,
            json={
                "error": "RES-002",
                "message": f"file with sha256=... already exists in workspace; reuse "
                f"file_id={FILE_ID}",
                "details": {},
                "existing_file": _file("pending"),
            },
        )

    client = _client(handler)
    try:
        result = await client.upload(context_id=CTX, source=BODY, filename="hello.txt")
    finally:
        await client.close()
    assert result.id == FILE_ID and result.scan_status == "pending"
    assert len(sent) == 1  # no PUT, no confirm


@pytest.mark.asyncio
async def test_upload_dedup_naming_no_file_is_the_plain_conflict_as_before():
    """A blocked duplicate is not handed out: the 409 stays the generic error."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            409,
            json={
                "error": "RES-002",
                "message": "file with sha256=... already exists in workspace; an earlier "
                "upload of this file was blocked by the malware scan",
                "details": {},
            },
        )

    client = _client(handler)
    try:
        with pytest.raises(KaguraConnectionError, match="HTTP 409") as info:
            await client.upload(context_id=CTX, source=BODY, filename="hello.txt")
    finally:
        await client.close()
    assert type(info.value) is KaguraConnectionError
    assert info.value.code == "RES-002"


# ============================================================================
# download_url refusals
# ============================================================================


@pytest.mark.asyncio
async def test_download_url_scan_pending_is_typed_with_retry_after():
    with pytest.raises(KaguraFileScanPendingError) as info:
        await _download(_pending())
    err = info.value
    assert isinstance(err, KaguraConnectionError)  # existing handlers still catch it
    assert err.code == "SCAN-001"
    assert err.retry_after == 30
    assert err.file_id == FILE_ID and err.scan_status == "pending"
    assert err.details == {"file_id": FILE_ID, "scan_status": "pending"}
    assert err.retryable is True
    assert "HTTP 409" in str(err) and "checked for malware" in str(err)


@pytest.mark.asyncio
@pytest.mark.parametrize("header", [None, "soon", "Thu, 20 Nov 2026 01:00:00 GMT"])
async def test_download_url_scan_pending_without_a_usable_retry_after(header):
    with pytest.raises(KaguraFileScanPendingError) as info:
        await _download(_pending(retry_after=header))
    assert info.value.retry_after is None
    assert info.value.code == "SCAN-001"


@pytest.mark.asyncio
@pytest.mark.parametrize("scan_status", ["infected", "error", "quarantined"])
async def test_download_url_blocked_is_typed_and_not_retryable(scan_status):
    """The server's gate is an allow-list: any value but clean/unscanned/pending blocks."""
    with pytest.raises(KaguraFileBlockedError) as info:
        await _download(_blocked(scan_status))
    err = info.value
    assert isinstance(err, KaguraConnectionError)
    assert not isinstance(err, KaguraFileScanPendingError)
    assert err.code == "SCAN-002"
    assert err.file_id == FILE_ID and err.scan_status == scan_status
    assert err.retryable is False
    assert not hasattr(err, "retry_after")


@pytest.mark.asyncio
async def test_download_url_404_stays_not_found():
    """The scan gate runs after the access checks: an unreadable file is still 404."""
    response = httpx.Response(
        404, json={"error": "RES-001", "message": "File not found", "details": {}}
    )
    with pytest.raises(KaguraNotFoundError):
        await _download(response)


@pytest.mark.asyncio
async def test_a_scan_code_on_another_status_is_not_typed():
    response = httpx.Response(
        400, json={"error": "SCAN-001", "message": "odd", "details": {"file_id": FILE_ID}}
    )
    with pytest.raises(KaguraConnectionError) as info:
        await _download(response)
    assert type(info.value) is KaguraConnectionError


def test_scan_errors_are_exported():
    assert "KaguraFileScanPendingError" in kagura_memory.__all__
    assert "KaguraFileBlockedError" in kagura_memory.__all__


# ============================================================================
# kagura files download-url
# ============================================================================


def _invoke_files(*args: str, **methods: Any) -> Any:
    """Run ``kagura files <args>`` with a mocked FilesClient whose methods are ``methods``."""
    client = MagicMock()
    for name, mock in methods.items():
        setattr(client, name, mock)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    with (
        patch(
            "kagura_memory.cli.load_config",
            return_value={"api_key": "key", "mcp_url": "https://test.com/mcp"},
        ),
        patch("kagura_memory.cli.FilesClient") as cls,
    ):
        cls._from_resolved_auth.return_value = client
        return CliRunner().invoke(main, ["files", *args])


def _invoke_download(error: Exception) -> Any:
    return _invoke_files(
        "download-url", FILE_ID, "-c", CTX, download_url=AsyncMock(side_effect=error)
    )


def test_cli_files_list_prints_scan_status():
    page = FileListResponse(files=[FileObject.model_validate(_file("pending"))])
    result = _invoke_files("list", "-c", CTX, list=AsyncMock(return_value=page))
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["files"][0]["scan_status"] == "pending"


def _raised(response: httpx.Response) -> Exception:
    """The error ``download_url`` raises for ``response`` (the CLI runs its own loop)."""
    try:
        asyncio.run(_download(response))
    except Exception as e:  # noqa: BLE001 - the error under test
        return e
    raise AssertionError("download_url did not raise")


def test_cli_download_url_scan_pending_says_how_long_to_wait():
    result = _invoke_download(_raised(_pending()))
    assert result.exit_code == 1
    assert "checked for malware" in result.output
    assert "wait 30 s" in result.output and "run the command again" in result.output


def test_cli_download_url_scan_pending_without_retry_after():
    result = _invoke_download(_raised(_pending(retry_after=None)))
    assert result.exit_code == 1
    assert "wait a moment" in result.output


def test_cli_download_url_blocked_says_it_cannot_be_downloaded():
    result = _invoke_download(_raised(_blocked("infected")))
    assert result.exit_code == 1
    assert "blocked by the malware scan" in result.output
    assert "scan_status: infected" in result.output
    assert "run the command again" not in result.output
    assert FILE_ID in result.output
