"""REST client for Kagura Memory Cloud file uploads.

Drives the 3-step file upload flow against memory-cloud:

1. ``POST /api/v1/files/reserve`` — server returns a presigned PUT URL.
2. ``PUT`` to R2 with body bytes and the ``x-amz-checksum-sha256``
   header (base64 of the raw sha256 digest). Required when the
   backend has ``R2_CHECKSUM_BINDING_ENABLED=true`` (memory-cloud
   v0.15.1+).
3. ``POST /api/v1/files/{file_id}/confirm`` — finalize the row.

Plus ``download_url()``, ``delete()`` and ``list()`` for additional
file operations.

The R2 PUT is sent via a separate :class:`httpx.AsyncClient` so the
SDK's ``Authorization: Bearer`` header cannot leak to the object
store, regardless of what other methods are added to this class.
"""

from __future__ import annotations

import base64
import hashlib
import mimetypes
from pathlib import Path
from typing import Any, Literal

import httpx

from ._http import SDK_VERSION, base_url_from_mcp, extract_detail, validate_https_url
from .exceptions import (
    KaguraAuthError,
    KaguraConnectionError,
    KaguraIntegrityError,
    KaguraNotFoundError,
    KaguraQuotaError,
)
from .models import (
    FileDownloadUrlResponse,
    FileListResponse,
    FileObject,
    FileReserveResponse,
)


class FilesClient:
    """REST API client for Kagura Memory Cloud file objects.

    Authentication: ``Authorization: Bearer <api_key>`` (set once in
    constructor). The R2 PUT step is unauthenticated by design — the
    presigned URL carries its own short-lived SigV4 signature.

    All methods may raise:
        KaguraAuthError: Authentication failed (401)
        KaguraNotFoundError: File not found (404)
        KaguraIntegrityError: Body sha256 did not match the presigned PUT
            binding (R2 returned 400 BadDigest)
        KaguraConnectionError: Connection or other HTTP error
        KaguraQuotaError: Quota exceeded (429)
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://memory.kagura-ai.com",
        timeout: float = 30.0,
        upload_timeout: float = 300.0,
    ) -> None:
        """Initialize FilesClient.

        Args:
            api_key: Kagura API key (Bearer token for API operations).
            base_url: REST API base URL (without path).
            timeout: API request timeout in seconds.
            upload_timeout: R2 PUT timeout in seconds (defaults to 5
                minutes to accommodate large files on slow networks).
        """
        stripped_url = base_url.rstrip("/")
        validate_https_url(stripped_url, label="Base URL")

        self.base_url = stripped_url
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "User-Agent": f"kagura-memory-sdk/{SDK_VERSION}",
            },
        )
        # Separate client for outbound PUT to the object store. Carries
        # no Bearer header so a future contributor cannot accidentally
        # leak the API key by adding another method that reuses it.
        self._upload_client = httpx.AsyncClient(
            timeout=upload_timeout,
            headers={"User-Agent": f"kagura-memory-sdk/{SDK_VERSION}"},
        )

    @classmethod
    def from_mcp_url(
        cls,
        api_key: str,
        mcp_url: str = "https://memory.kagura-ai.com/mcp",
        timeout: float = 30.0,
        upload_timeout: float = 300.0,
    ) -> FilesClient:
        """Create FilesClient by deriving the REST base URL from an MCP URL.

        Strips ``/mcp`` and everything after it. Handles both
        ``/mcp`` and ``/mcp/w/{workspace_id}`` formats.
        """
        url = mcp_url.rstrip("/")
        base_url = base_url_from_mcp(url)
        return cls(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            upload_timeout=upload_timeout,
        )

    # -------------------------------------------------------------------
    # Internal HTTP helpers
    # -------------------------------------------------------------------

    async def _request(
        self,
        method: Literal["GET", "POST", "DELETE"],
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> httpx.Response:
        """Make an authenticated request with standard error mapping."""
        url = f"{self.base_url}{path}"
        try:
            response = await self._client.request(method, url, json=json, params=params)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status == 401:
                raise KaguraAuthError("Authentication failed. Check your API key.") from e
            if status == 404:
                raise KaguraNotFoundError(extract_detail(e.response) or "Not found") from e
            if status == 429:
                retry_after = e.response.headers.get("Retry-After")
                raise KaguraQuotaError(
                    "Quota exceeded. Try again later.",
                    retry_after=int(retry_after) if retry_after else None,
                ) from e
            detail = extract_detail(e.response)
            msg = f"HTTP {status}: {detail}" if detail else f"HTTP {status}"
            raise KaguraConnectionError(msg) from e
        except httpx.RequestError as e:
            raise KaguraConnectionError(f"Connection failed: {e}") from e

    async def _put_to_object_store(
        self,
        upload_url: str,
        body: bytes,
        sha256_hex: str,
        content_type: str,
    ) -> None:
        """PUT ``body`` to the presigned R2 URL with checksum binding.

        Sends ``x-amz-checksum-sha256`` as base64 of the raw 32-byte
        digest (NOT base64 of the hex string). When the backend has
        ``R2_CHECKSUM_BINDING_ENABLED=true`` and the body's actual
        sha256 differs from this header, R2 returns 400 BadDigest and
        we raise :class:`KaguraIntegrityError`.

        Note: any other 400 response from R2 (e.g. a malformed
        presigned URL or unexpected ``Content-Length`` mismatch) is
        also surfaced as :class:`KaguraIntegrityError`. BadDigest is
        the documented and observed common cause; parsing the
        S3-XML error body for finer discrimination is out of scope
        for v0.14.0.
        """
        checksum_b64 = base64.b64encode(bytes.fromhex(sha256_hex)).decode()
        try:
            response = await self._upload_client.put(
                upload_url,
                content=body,
                headers={
                    "Content-Type": content_type,
                    "x-amz-checksum-sha256": checksum_b64,
                },
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                raise KaguraIntegrityError(
                    "Object store rejected upload with HTTP 400 — most "
                    "commonly R2 BadDigest (body sha256 did not match the "
                    "presigned PUT binding). Verify that the SDK is "
                    "computing sha256 over the exact bytes being PUT."
                ) from e
            raise KaguraConnectionError(
                f"Object store PUT failed: HTTP {e.response.status_code}"
            ) from e
        except httpx.TimeoutException as e:
            raise KaguraConnectionError(f"Object store PUT timed out: {e}") from e
        except httpx.RequestError as e:
            raise KaguraConnectionError(f"Object store PUT failed: {e}") from e

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    async def upload(
        self,
        *,
        context_id: str,
        source: Path | bytes,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> FileObject:
        """Upload a file to Kagura Memory Cloud.

        Drives the full 3-step flow: reserve → presigned PUT →
        confirm. Returns the finalized :class:`FileObject` on success.

        Args:
            context_id: Target context (workspace) UUID. The SDK uses
                ``context_id`` consistently across clients; this is
                mapped to the backend's ``workspace_id`` field on the
                wire.
            source: File contents. ``Path`` reads the file fully
                into memory; ``bytes`` uses the buffer directly.
                Chunked / streaming PUT is out of scope for v0.14.0
                (the server caps file size at 100 MiB by default).
            filename: Required when ``source`` is ``bytes`` (the
                server requires a filename). Defaults to ``path.name``
                when ``source`` is a ``Path``.
            content_type: MIME type. When omitted, falls back to
                :func:`mimetypes.guess_type` and then to
                ``application/octet-stream``.

        Returns:
            FileObject with the finalized file metadata. When the file
            already exists with the same sha256 in the workspace the
            server returns 409 and the SDK surfaces the existing
            FileObject (dedup happy-path) rather than raising.

        Raises:
            ValueError: If ``source`` is ``bytes`` and ``filename`` is
                not provided.
            KaguraIntegrityError: If R2 rejected the body sha256
                binding.
            KaguraAuthError, KaguraNotFoundError, KaguraQuotaError,
            KaguraConnectionError: see class docstring.
        """
        resolved_filename, body, sha256_hex = _prepare_source(source, filename)
        size_bytes = len(body)
        resolved_content_type = _resolve_content_type(content_type, resolved_filename)

        reserve_body = {
            "workspace_id": context_id,
            "filename": resolved_filename,
            "content_type": resolved_content_type,
            "size_bytes": size_bytes,
            "sha256": sha256_hex,
        }

        try:
            reserve_resp = await self._request("POST", "/api/v1/files/reserve", json=reserve_body)
        except KaguraConnectionError as e:
            # 409 dedup happy-path: server reports the existing file
            # in the error detail; surface it as a FileObject so the
            # caller does not have to special-case duplicates.
            existing = _extract_existing_file(e)
            if existing is not None:
                return existing
            raise

        reserve = FileReserveResponse.model_validate(reserve_resp.json())

        await self._put_to_object_store(
            upload_url=reserve.upload_url,
            body=body,
            sha256_hex=sha256_hex,
            content_type=resolved_content_type,
        )

        confirm_resp = await self._request(
            "POST",
            f"/api/v1/files/{reserve.file_id}/confirm",
            json={"sha256": sha256_hex},
        )
        return FileObject.model_validate(confirm_resp.json())

    async def download_url(self, file_id: str) -> str:
        """Return a short-lived presigned GET URL for ``file_id``."""
        response = await self._request("GET", f"/api/v1/files/{file_id}/download-url")
        return FileDownloadUrlResponse.model_validate(response.json()).download_url

    async def delete(self, file_id: str) -> None:
        """Soft-delete a file by id (server hard-deletes after retention)."""
        await self._request("DELETE", f"/api/v1/files/{file_id}")

    async def list(
        self,
        *,
        context_id: str,
        limit: int = 50,
        cursor: str | None = None,
    ) -> FileListResponse:
        """List uploaded files in a workspace, newest first.

        Args:
            context_id: Workspace UUID to list files for.
            limit: Maximum number of files to return (1-500, default
                50). Server-side cap is 500.
            cursor: Forward-compatible pagination cursor. The current
                server (memory-cloud v0.15.x) ignores this field and
                always returns at most ``limit`` items; a future
                server version is expected to populate
                :attr:`FileListResponse.next_cursor`.

        Returns:
            FileListResponse with the page of files and an optional
            ``next_cursor`` (always ``None`` against the current
            server).
        """
        params: dict[str, Any] = {"workspace_id": context_id, "limit": limit}
        if cursor is not None:
            params["cursor"] = cursor
        response = await self._request("GET", "/api/v1/files", params=params)
        raw = response.json()
        # Current server returns a bare ``list[FileObjectOut]``;
        # future server versions may return ``{files, next_cursor}``.
        if isinstance(raw, list):
            return FileListResponse(
                files=[FileObject.model_validate(item) for item in raw],
                next_cursor=None,
            )
        return FileListResponse.model_validate(raw)

    # -------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------

    async def close(self) -> None:
        """Close both HTTP clients.

        Uses try/finally so the second client is still closed if the
        first raises during teardown — otherwise we would leak the
        upload client's connection pool on a flaky API client close.
        """
        try:
            await self._client.aclose()
        finally:
            await self._upload_client.aclose()

    async def __aenter__(self) -> FilesClient:
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: Any,
    ) -> None:
        await self.close()


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _extract_existing_file(error: KaguraConnectionError) -> FileObject | None:
    """Pull an ``existing_file`` payload out of a 409 dedup response.

    The 409 path comes through :meth:`_request` as a
    :class:`KaguraConnectionError` chained from the underlying
    :class:`httpx.HTTPStatusError`. We re-inspect the response body
    (not just the formatted message) to recover the structured
    ``existing_file`` field the server attaches.
    """
    cause = error.__cause__
    if not isinstance(cause, httpx.HTTPStatusError):
        return None
    if cause.response.status_code != 409:
        return None
    try:
        body = cause.response.json()
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(body, dict):
        return None
    existing = body.get("existing_file")
    if not isinstance(existing, dict):
        return None
    return FileObject.model_validate(existing)


def _resolve_content_type(content_type: str | None, filename: str) -> str:
    """Pick a content_type — explicit > mimetypes guess > octet-stream."""
    if content_type:
        return content_type
    guessed, _ = mimetypes.guess_type(filename)
    return guessed or "application/octet-stream"


def _prepare_source(
    source: Path | bytes,
    filename: str | None,
) -> tuple[str, bytes, str]:
    """Resolve filename, load bytes, compute sha256.

    Returns ``(filename, body, sha256_hex)``. The body is fully
    materialized in memory — the PUT step needs all bytes and the
    server caps file size at 100 MiB by default. Chunked PUT is out
    of scope for v0.14.0.
    """
    if isinstance(source, Path):
        if not source.is_file():
            raise FileNotFoundError(f"source path does not exist: {source}")
        resolved_filename = filename or source.name
        body = source.read_bytes()
        return resolved_filename, body, hashlib.sha256(body).hexdigest()

    if not filename:
        raise ValueError(
            "filename is required when source is bytes (the server requires a non-empty filename)."
        )
    return filename, source, hashlib.sha256(source).hexdigest()
