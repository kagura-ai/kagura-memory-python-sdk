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

import asyncio
import base64
import hashlib
import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx

from ._auth import (
    _SOURCE_LABEL,
    _AuthSource,
    _OAuthAuth,
    _resolve_auth,
    _StaticAuth,
)
from ._http import (
    SDK_VERSION,
    base_url_from_mcp,
    extract_detail,
    normalize_uuid,
    sanitize_server_detail,
)
from ._rest_base import KaguraRestClient
from .auth.credentials import KaguraOAuth
from .exceptions import (
    KaguraConnectionError,
    KaguraError,
    KaguraIntegrityError,
    _exc_message,
)
from .logger import VerboseLogger, normalize_logger
from .models import (
    FileDownloadUrlResponse,
    FileListResponse,
    FileObject,
    FileReserveResponse,
)


class FilesClient(KaguraRestClient):
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
        api_key: str | None = None,
        base_url: str = "https://memory.kagura-ai.com",
        timeout: float = 30.0,
        upload_timeout: float = 300.0,
        *,
        _oauth: KaguraOAuth | None = None,
        _auth_source: _AuthSource | None = None,
        _workspace_id_hint: str | None = None,
    ) -> None:
        """Initialize FilesClient (see :class:`KaguraRestClient`).

        Adds the R2 upload leg on top of the shared spine:

        Args:
            api_key: Kagura API key (Bearer token). Required unless
                ``_oauth`` is supplied (see ``from_mcp_url``).
            base_url: REST API base URL (without path).
            timeout: API request timeout in seconds.
            upload_timeout: R2 PUT timeout in seconds (defaults to 5
                minutes to accommodate large files on slow networks).
            _oauth: Private. OAuth handler from ``from_mcp_url``.
            _auth_source: Private. Credential provenance for the 403
                cross-source hint (issue #115).
            _workspace_id_hint: Private. Source-bound workspace UUID for
                the 403 hint — never sent on the wire.

        Raises:
            ValueError: If neither ``api_key`` nor ``_oauth`` is given.
        """
        super().__init__(
            api_key,
            base_url,
            timeout,
            _oauth=_oauth,
            _auth_source=_auth_source,
            _workspace_id_hint=_workspace_id_hint,
        )
        # Defense in depth: never pass auth= or an Authorization header here.
        # Bearer / OAuth credentials must not reach R2 — the presigned PUT
        # carries its own short-lived SigV4 signature. A separate httpx client
        # makes the no-auth invariant structural, not just a runtime convention.
        self._upload_client = httpx.AsyncClient(
            timeout=upload_timeout,
            headers={"User-Agent": f"kagura-memory-sdk/{SDK_VERSION}"},
        )

    @classmethod
    def from_mcp_url(
        cls,
        api_key: str | None = None,
        mcp_url: str | None = None,
        timeout: float = 30.0,
        upload_timeout: float = 300.0,
        *,
        profile: str | None = None,
    ) -> FilesClient:
        """Create FilesClient by resolving credentials from the SDK chain.

        Runs :func:`_resolve_auth` with the same precedence chain as
        :class:`KaguraClient`: explicit ``api_key`` > ``KAGURA_API_KEY``
        env > OAuth profile from ``~/.kagura/credentials.json`` >
        ``.kagura.json``. The chosen credential source picks the
        transport: a static API key bakes the Bearer header once; an
        OAuth profile installs a ``KaguraOAuth`` httpx.Auth handler
        with automatic refresh.

        The REST ``base_url`` is derived from the resolved ``mcp_url``
        (strips ``/mcp`` and any ``/mcp/w/{workspace_id}`` suffix).

        Args:
            api_key: Explicit Kagura API key. Skips the resolution chain.
            mcp_url: Explicit MCP URL. When omitted, the OAuth profile's
                stored ``mcp_url`` is used (or
                ``https://memory.kagura-ai.com/mcp`` as a final default).
            timeout: API request timeout in seconds.
            upload_timeout: R2 PUT timeout in seconds.
            profile: Named OAuth profile to load (overrides
                ``KAGURA_PROFILE`` env and the credentials file's
                ``default_profile``).
        """
        resolved = _resolve_auth(api_key=api_key, mcp_url=mcp_url, profile=profile)
        return cls._from_resolved_auth(resolved, timeout=timeout, upload_timeout=upload_timeout)

    @classmethod
    def _from_resolved_auth(
        cls,
        resolved: _StaticAuth | _OAuthAuth,
        *,
        timeout: float = 30.0,
        upload_timeout: float = 300.0,
        workspace_id_hint: str | None = None,
    ) -> FilesClient:
        """Construct from a pre-resolved auth — internal CLI helper.

        Shared by :meth:`from_mcp_url` (SDK entry) and the CLI
        (``cli._build_files_client``). The CLI resolves once so api_key
        and workspace_id can be paired from the same source (#115);
        threading the resolved auth through here keeps construction in
        one place and lets the 403 hint carry the source provenance.

        ``workspace_id_hint`` lets the CLI thread the workspace bound
        to a static api_key (from ``.kagura.json``'s ``context_id``)
        into the client so the 403 hint can show it. For the OAuth
        path the resolver already carries ``workspace_id`` on the
        :class:`_OAuthAuth` result, so the hint is set from there
        unconditionally — passing ``workspace_id_hint`` is ignored.
        """
        base_url = base_url_from_mcp(resolved.mcp_url.rstrip("/"))
        if isinstance(resolved, _StaticAuth):
            return cls(
                api_key=resolved.api_key,
                base_url=base_url,
                timeout=timeout,
                upload_timeout=upload_timeout,
                _auth_source=resolved.source,
                _workspace_id_hint=workspace_id_hint,
            )
        return cls(
            base_url=base_url,
            timeout=timeout,
            upload_timeout=upload_timeout,
            _oauth=resolved.oauth,
            _auth_source="oauth",
            _workspace_id_hint=resolved.workspace_id,
        )

    # -------------------------------------------------------------------
    # Internal HTTP helpers
    # -------------------------------------------------------------------

    def _error_403(
        self,
        e: httpx.HTTPStatusError,
        *,
        request_json: dict[str, Any] | None,
        request_params: dict[str, Any] | None,
    ) -> KaguraError:
        # Issue #115: a workspace-mismatch 403 is the operator's most common
        # 403 cause (api_key bound to workspace A, request targets workspace
        # B). Surface the credential source and requested workspace prefix so
        # the cause is visible without leaking the api_key value or Bearer
        # header. As of memory-cloud v0.41.0 every file endpoint puts
        # workspace_id on the wire, so the workspace-mismatch heading covers
        # them all; the generic heading remains a defensive fallback.
        return KaguraConnectionError(
            _format_workspace_403_hint(
                auth_source=self._auth_source,
                source_workspace_hint=self._workspace_id_hint,
                requested_workspace=_extract_requested_workspace(request_json, request_params),
                server_detail=extract_detail(e.response),
            )
        )

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

        The checksum header is sent **only when the presign signed it**
        (``X-Amz-SignedHeaders`` contains ``x-amz-checksum-sha256``). A
        server with R2 checksum binding OFF presigns without it, and
        sending it anyway makes the SigV4 signature mismatch → R2 rejects
        with 403 SignatureDoesNotMatch (#226). The sha256 is re-sent at
        the confirm step regardless, so server-side verification still
        holds when binding is off.
        """
        headers = {"Content-Type": content_type}
        if _presign_signs_checksum(upload_url):
            headers["x-amz-checksum-sha256"] = base64.b64encode(bytes.fromhex(sha256_hex)).decode()
        try:
            response = await self._upload_client.put(upload_url, content=body, headers=headers)
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
            raise KaguraConnectionError(f"Object store PUT timed out: {_exc_message(e)}") from e
        except httpx.RequestError as e:
            raise KaguraConnectionError(f"Object store PUT failed: {_exc_message(e)}") from e

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
        binding_context_id: str | None = None,
        logger: VerboseLogger | None = None,
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
                **Memory contract**: peak memory is bounded by the
                file size. R2 presigned PUT is single-PUT (chunked /
                streaming PUT is out of scope for v0.14.0), so the
                body has to be resident at upload time regardless of
                how it's hashed. The server caps file size at 100 MiB
                by default; callers uploading anywhere near that bound
                should size their runtime accordingly. The server
                rejects oversize uploads at the boundary — the SDK
                does not enforce a client-side cap so that a future
                server-side cap bump is non-breaking.
            filename: Required when ``source`` is ``bytes`` (the
                server requires a filename). Defaults to ``path.name``
                when ``source`` is a ``Path``.
            content_type: MIME type. When omitted, falls back to
                :func:`mimetypes.guess_type` and then to
                ``application/octet-stream``.
            binding_context_id: Optional owning context to bind the file
                to for access control (server v0.41.0+). It is sent as the
                wire ``context_id`` field — **distinct** from the
                ``context_id`` parameter above, which maps to
                ``workspace_id``. The caller must have write (EDITOR+)
                access to this context and it must belong to the upload's
                workspace, else the server returns 403 / 422 respectively.
                When omitted the file is NULL-context (workspace-scoped,
                legacy behaviour) and stays fully listable.

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
        log = normalize_logger(logger)
        reserved_file_id: str | None = None
        uploaded = False
        # ``confirm_started`` flips True when the confirm HTTP request is
        # dispatched; ``confirmed`` flips True only after the server response
        # is fully parsed into a FileObject. The two flags together let an AI
        # consumer reading the terminal-error detail tell apart "confirm
        # never ran — safe to redo from scratch" vs "confirm dispatched but
        # we don't know if the server processed it — must verify before
        # retrying".
        confirm_started = False
        confirmed = False
        try:
            # Pre-flight validators and source preparation can raise too
            # (ValueError from _normalize_context_id, FileNotFoundError /
            # ValueError from _prepare_source). They live INSIDE the try so
            # those failures also get a terminal kind=error event before
            # propagating — matching the contract documented in the logger
            # module's "terminal-event" docstring.
            context_id = _normalize_context_id(context_id)
            resolved_filename, body, sha256_hex = await _prepare_source(source, filename)
            size_bytes = len(body)
            resolved_content_type = _resolve_content_type(content_type, resolved_filename)

            log.action(
                "Reserving upload",
                f"{resolved_filename} ({size_bytes} bytes)",
                stage="reserve",
            )
            reserve_body: dict[str, Any] = {
                "workspace_id": context_id,
                "filename": resolved_filename,
                "content_type": resolved_content_type,
                "size_bytes": size_bytes,
                "sha256": sha256_hex,
            }
            # Only send context_id when a binding was requested — omitting it
            # keeps the upload NULL-context (workspace-scoped, legacy behaviour),
            # which is exactly what the server expects for an unbound file.
            if binding_context_id is not None:
                reserve_body["context_id"] = binding_context_id

            try:
                reserve_resp = await self._request(
                    "POST", "/api/v1/files/reserve", json=reserve_body
                )
            except KaguraConnectionError as e:
                # 409 dedup happy-path: server reports the existing file
                # in the error detail; surface it as a FileObject so the
                # caller does not have to special-case duplicates.
                existing = _extract_existing_file(e)
                if existing is not None:
                    log.success(
                        "Dedup hit — existing file returned",
                        stage="complete",
                        detail={"file_id": existing.id, "deduped": True},
                    )
                    return existing
                raise

            reserve = FileReserveResponse.model_validate(reserve_resp.json())
            reserved_file_id = reserve.file_id
            log.action("Uploading to object store", stage="upload")
            await self._put_to_object_store(
                upload_url=reserve.upload_url,
                body=body,
                sha256_hex=sha256_hex,
                content_type=resolved_content_type,
            )
            uploaded = True

            log.action("Confirming upload", stage="confirm")
            confirm_started = True
            # workspace_id is required on the confirm query string (memory-cloud
            # v0.41.0); omitting it returns 422 VAL-001 and the upload never
            # finalizes. context_id is the SDK's name for the workspace on the wire.
            confirm_resp = await self._request(
                "POST",
                f"/api/v1/files/{reserve.file_id}/confirm",
                params={"workspace_id": context_id},
                json={"sha256": sha256_hex},
            )
            result = FileObject.model_validate(confirm_resp.json())
            confirmed = True
            log.success(
                "Upload complete",
                stage="complete",
                detail={"file_id": result.id, "size_bytes": size_bytes},
            )
            return result
        except BaseException as e:
            # Terminal-event guarantee: include partial state so a recovering
            # AI consumer can decide whether to retry confirm vs re-upload.
            # ``confirm_started`` AND NOT ``confirmed`` is the ambiguous case
            # (the server may have processed the confirm, but we lost the
            # response) — consumers should verify file state before retrying
            # rather than re-uploading.
            log.error(
                f"Upload failed: {e}",
                stage="complete",
                detail={
                    "reserved_file_id": reserved_file_id,
                    "uploaded": uploaded,
                    "confirm_started": confirm_started,
                    "confirmed": confirmed,
                },
            )
            raise

    async def download_url(self, file_id: str, *, context_id: str) -> str:
        """Return a short-lived presigned GET URL for ``file_id``.

        Args:
            file_id: The file object id.
            context_id: The owning workspace UUID. **Required** — memory-cloud
                v0.41.0 scopes file-id lookups to the workspace and returns 422
                without ``workspace_id`` on the query. Sent as ``workspace_id``
                (the SDK's consistent ``context_id`` → ``workspace_id`` wire
                mapping). A file bound to a private context the caller can't
                read is reported as 404 (existence-hiding); a workspace mismatch
                (wrong ``context_id``) surfaces as a 403 with an actionable hint.
        """
        context_id = _normalize_context_id(context_id)
        response = await self._request(
            "GET",
            f"/api/v1/files/{file_id}/download-url",
            params={"workspace_id": context_id},
        )
        return FileDownloadUrlResponse.model_validate(response.json()).download_url

    async def delete(self, file_id: str, *, context_id: str) -> None:
        """Soft-delete a file by id (server hard-deletes after retention).

        Args:
            file_id: The file object id.
            context_id: The owning workspace UUID. **Required** (memory-cloud
                v0.41.0); sent as ``workspace_id`` on the query. Omitting it
                returns 422.
        """
        context_id = _normalize_context_id(context_id)
        await self._request(
            "DELETE",
            f"/api/v1/files/{file_id}",
            params={"workspace_id": context_id},
        )

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
        context_id = _normalize_context_id(context_id)
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
        """Close both HTTP clients (API + unauthenticated R2 leg)."""
        try:
            await super().close()
        finally:
            await self._upload_client.aclose()


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _short_workspace(uuid_str: str | None) -> str:
    """Truncate a UUID to its 8-char prefix for log-friendly display.

    Workspace UUIDs are not secret, but the full UUID adds noise to
    error messages — a prefix is enough to compare against the source's
    workspace at a glance. Returns ``"<none>"`` for falsy input.
    """
    if not uuid_str:
        return "<none>"
    return f"{uuid_str[:8]}…"


def _extract_requested_workspace(
    json: dict[str, Any] | None, params: dict[str, Any] | None
) -> str | None:
    """Recover the ``workspace_id`` the failing request was targeting.

    File endpoints carry workspace_id in either the JSON body (reserve) or the
    query string (confirm / download-url / delete / list, all of which put it
    on the query as of memory-cloud v0.41.0). Returns ``None`` only as a
    defensive fallback when neither the body nor the params carries a
    ``workspace_id`` — no current file request hits that path.
    """
    if json is not None:
        ws = json.get("workspace_id")
        if isinstance(ws, str):
            return ws
    if params is not None:
        ws = params.get("workspace_id")
        if isinstance(ws, str):
            return ws
    return None


# _sanitize_server_detail moved to _http.sanitize_server_detail so
# WorkspaceClient shares the same credential-marker scrubbing; the alias
# keeps this module's call sites and test imports stable.
_sanitize_server_detail = sanitize_server_detail


def _format_workspace_403_hint(
    *,
    auth_source: _AuthSource | None,
    source_workspace_hint: str | None,
    requested_workspace: str | None,
    server_detail: str | None = None,
) -> str:
    """Compose the actionable 403 message for issue #115.

    Always emits a structured hint when ``auth_source`` is set; falls
    back to the bare ``"HTTP 403"`` for legacy callers (no source
    threaded through construction) so the message stays compatible
    with existing CLI error-string assertions until they migrate.

    The hint shape adapts to the failing request:

    - When ``requested_workspace`` is set (upload / list endpoints), use
      workspace-mismatch language and the ``--context-id`` recovery hint.
    - When ``requested_workspace`` is ``None`` (file-id-based endpoints
      like download-url / delete), use a generic "access denied"
      heading and omit the ``--context-id`` hint — those commands do
      not accept ``--context-id`` and the recovery path is different.

    ``server_detail`` is appended (when non-empty and free of
    credential-shaped markers) so scope / deactivation / plan-limit
    reasons surfaced by the server are not lost.

    The hint never includes the api_key value or
    ``Authorization: Bearer ...`` header — only the source label and
    the UUID prefixes (per python.md "Never store API keys as instance
    attributes": the api_key is not on the client either).
    """
    if auth_source is None:
        # Even the bare path can carry a sanitized server detail.
        safe_detail = _sanitize_server_detail(server_detail)
        return f"HTTP 403: {safe_detail}" if safe_detail else "HTTP 403"
    # Use .get() with the raw value as fallback so an unexpected source tag
    # (shouldn't happen — _auth_source is set by internal resolver code only)
    # can never raise KeyError mid-403-handling and mask the real HTTP error.
    source_label = _SOURCE_LABEL.get(auth_source, str(auth_source))
    heading = (
        "HTTP 403 — workspace not accessible with current credentials."
        if requested_workspace
        else "HTTP 403 — access denied with current credentials."
    )
    lines = [
        heading,
        f"  api_key source: {source_label} (workspace={_short_workspace(source_workspace_hint)})",
    ]
    if requested_workspace:
        lines.append(f"  workspace requested: {_short_workspace(requested_workspace)}")
        lines.append("  Hint: --context-id may not match the workspace bound to your api_key.")
    safe_detail = _sanitize_server_detail(server_detail)
    if safe_detail:
        lines.append(f"  server detail: {safe_detail}")
    return "\n".join(lines)


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


def _normalize_context_id(context_id: str) -> str:
    """Canonicalize ``context_id`` before URL/query interpolation.

    The server's ``/api/v1/files/*`` endpoints validate ``workspace_id``
    server-side and return 422, but the round-trip masks a class of
    common errors — e.g. passing the CLI sentinel ``"auto"`` straight
    through to the SDK without resolution. Raising locally turns that
    into a clear ``ValueError`` instead of a generic HTTP status.

    Delegates to the shared :func:`~kagura_memory._http.normalize_uuid`
    so tolerated non-canonical spellings (``{braces}``, ``urn:uuid:``
    prefix, dashless 32-hex) are CANONICALIZED rather than sent raw —
    raw spellings pass a validate-only check but surface server-side as
    a misleading uniform 404 (#236).

    Args:
        context_id: The workspace UUID to canonicalize.

    Returns:
        The canonical UUID string.

    Raises:
        ValueError: If ``context_id`` is not a parseable UUID. The
            message points the caller at the OAuth profile's
            ``workspace_id`` and ``kagura context list``, which are the
            two ways the SDK expects ``context_id`` to be sourced.
    """
    try:
        return normalize_uuid(context_id, label="context_id")
    except ValueError as e:
        raise ValueError(
            f"context_id must be a UUID; got {context_id!r}. "
            "Use the OAuth profile's workspace_id, a UUID from "
            "`kagura context list`, or run `kagura auth login` first."
        ) from e


def _presign_signs_checksum(upload_url: str) -> bool:
    """True if the presigned URL signed the sha256 checksum header.

    Reads ``X-Amz-SignedHeaders`` from the presigned PUT URL's query and reports
    whether it lists ``x-amz-checksum-sha256`` (case-insensitive). The SDK must
    send the checksum header only when the server bound it into the presign (R2
    checksum binding on). Sending it against a presign that did not sign it
    invalidates the SigV4 signature and R2 rejects the PUT with 403
    SignatureDoesNotMatch (#226).

    Args:
        upload_url: The presigned R2 PUT URL returned by ``/files/reserve``.

    Returns:
        ``True`` when ``x-amz-checksum-sha256`` is in the signed-headers set,
        ``False`` otherwise (including when the URL has no query string).
    """
    signed = parse_qs(urlparse(upload_url).query).get("X-Amz-SignedHeaders", [""])[0]
    return "x-amz-checksum-sha256" in signed.lower()


def _resolve_content_type(content_type: str | None, filename: str) -> str:
    """Pick a content_type — explicit > mimetypes guess > octet-stream."""
    if content_type:
        return content_type
    guessed, _ = mimetypes.guess_type(filename)
    return guessed or "application/octet-stream"


def _read_and_hash(path: Path) -> tuple[bytes, str]:
    """Synchronously read a file and compute its sha256 — runs in a thread.

    Reads the full file into memory by design. R2 presigned PUT is a
    single-PUT operation (chunked / streaming PUT is out of scope for
    v0.14.0 per the issue body's "Out of scope" section), so the body
    is needed in memory at upload time regardless of how it's hashed.
    Peak memory is bounded by the server's 100 MiB file size cap; a
    streaming hash + second file read for PUT would double disk I/O
    without lowering peak memory. Revisit when chunked PUT lands in
    a future SDK release.
    """
    body = path.read_bytes()
    return body, hashlib.sha256(body).hexdigest()


async def _prepare_source(
    source: Path | bytes,
    filename: str | None,
) -> tuple[str, bytes, str]:
    """Resolve filename, load bytes, compute sha256.

    Returns ``(filename, body, sha256_hex)``. The body is fully
    materialized in memory — the PUT step needs all bytes and the
    server caps file size at 100 MiB by default. Chunked PUT is out
    of scope for v0.14.0.

    For ``Path`` sources the disk read + hash runs in ``asyncio.to_thread``
    so that ``upload()`` does not block the event loop while waiting on
    file I/O — important when several concurrent uploads share a runtime.
    ``bytes`` input stays on the loop because the buffer is already
    resident in memory.
    """
    if isinstance(source, Path):
        if not source.is_file():
            raise FileNotFoundError(f"source path does not exist: {source}")
        resolved_filename = filename or source.name
        body, sha256_hex = await asyncio.to_thread(_read_and_hash, source)
        return resolved_filename, body, sha256_hex

    if not filename:
        raise ValueError(
            "filename is required when source is bytes (the server requires a non-empty filename)."
        )
    return filename, source, hashlib.sha256(source).hexdigest()
