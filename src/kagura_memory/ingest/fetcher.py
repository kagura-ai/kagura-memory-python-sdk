"""SSRF-hardened URL/file fetcher for the file ingestion pipeline.

The :class:`Fetcher` resolves a hostname pre-flight, validates every
returned IP against an in-process denylist, then issues a stream-bounded
HTTP request via httpx with manual redirect handling. Local file paths
are read via ``asyncio.to_thread`` after a default-deny check on
sensitive system directories.

The fetcher's exception class is
:class:`kagura_memory.exceptions.KaguraFetchError`. Any HTTP- or
filesystem-level error surfaces as that single class with the offending
URL/path attached as ``.url``.
"""

from __future__ import annotations

import asyncio
import socket
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import httpx

from ..exceptions import KaguraFetchError
from ._safety import is_blocked_ip, is_blocked_system_path

# --- Configuration constants -------------------------------------------------

DEFAULT_MAX_BYTES: int = 100 * 1024 * 1024  # 100 MB
DEFAULT_CONNECT_TIMEOUT: float = 10.0
DEFAULT_READ_TIMEOUT: float = 60.0
DEFAULT_MAX_REDIRECTS: int = 3
MAX_URL_LENGTH: int = 8192
_ALLOWED_SCHEMES: frozenset[str] = frozenset({"http", "https"})


@dataclass
class FetchResult:
    """Result of a single :meth:`Fetcher.fetch` call.

    Attributes:
        body: The fetched payload, capped at ``max_bytes``.
        content_type: ``Content-Type`` from the final response (after
            redirects). Lowercased; the parameter portion (e.g.
            ``"; charset=utf-8"``) is stripped. May be ``""`` if the
            server omitted the header or for local files where MIME is
            not detected here.
        source_uri: Canonical URI for the fetched resource.
            ``"https://..."`` for URLs, ``"file:///..."`` for local files.
        source_type: ``"url"`` or ``"file"``.
        final_url: The URL after all redirects (URL fetches only). For
            local file fetches, equals ``source_uri``.
        bytes_read: Total bytes read; equals ``len(body)`` unless the
            stream was truncated at ``max_bytes`` (in which case the
            fetch raises :class:`KaguraFetchError` and no result is
            returned, so this attribute always equals ``len(body)`` in
            practice — kept for symmetry).
    """

    body: bytes
    content_type: str
    source_uri: str
    source_type: Literal["url", "file"]
    final_url: str
    bytes_read: int


class Fetcher:
    """SSRF-hardened URL and local-file fetcher.

    Use as an async context manager::

        async with Fetcher() as fetcher:
            result = await fetcher.fetch("https://example.com/doc.pdf")

    All limits are constructor-tunable; defaults were derived from the
    Phase 1 ingest design review (issue #80 / PR #109) — see the commit
    history for the rationale behind each value.
    """

    def __init__(
        self,
        max_bytes: int = DEFAULT_MAX_BYTES,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        read_timeout: float = DEFAULT_READ_TIMEOUT,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
        allow_http: bool = False,
        allow_system_paths: bool = False,
    ):
        self.max_bytes = max_bytes
        self.max_redirects = max_redirects
        self.allow_http = allow_http
        self.allow_system_paths = allow_system_paths

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=connect_timeout, read=read_timeout, write=30.0, pool=10.0
            ),
            follow_redirects=False,  # we handle redirects manually for SSRF re-check
            headers={"User-Agent": "kagura-memory-ingest/0.1"},
            # Pinning rewrites the URL host to the validated IP (#188), which
            # collapses httpcore's pool key (scheme, host, port) onto the IP.
            # Disable keepalive reuse so two different hostnames sharing an IP
            # never reuse a connection opened with the other's SNI/cert. The
            # fetcher streams one response per request and closes it anyway, so
            # there is nothing to gain from pooling.
            limits=httpx.Limits(max_keepalive_connections=0),
        )

    async def __aenter__(self) -> Fetcher:
        return self

    async def __aexit__(self, _exc_type, _exc_val, _exc_tb) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    # --- Public API ----------------------------------------------------------

    async def fetch(self, source: str) -> FetchResult:
        """Fetch a URL or local file path safely.

        Args:
            source: Either a URL (``http://`` / ``https://``) or a
                filesystem path (absolute or relative). Schemes other than
                ``http``/``https`` are rejected.

        Returns:
            A populated :class:`FetchResult`.

        Raises:
            KaguraFetchError: For any safety violation (blocked IP, blocked
                path, redirect-loop, body cap exceeded, network error, etc.).
        """
        if not source or not isinstance(source, str):
            raise KaguraFetchError("source must be a non-empty string", url=source or None)

        # URL form takes precedence: anything starting with a scheme like
        # http://, https://, ftp://, file://, ... is parsed as a URL.
        parsed = urlsplit(source)
        if parsed.scheme:
            if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
                raise KaguraFetchError(
                    f"unsupported scheme {parsed.scheme!r}; only http(s) is allowed",
                    url=source,
                )
            return await self._fetch_url(source)

        # Otherwise treat as local file path.
        return await self._fetch_local(source)

    # --- URL path ------------------------------------------------------------

    async def _fetch_url(self, url: str) -> FetchResult:
        if len(url) > MAX_URL_LENGTH:
            raise KaguraFetchError(f"URL exceeds {MAX_URL_LENGTH} chars (got {len(url)})", url=url)

        parsed = urlsplit(url)
        if parsed.scheme.lower() == "http" and not self.allow_http:
            raise KaguraFetchError(
                "http:// is disabled; use https:// or pass allow_http=True", url=url
            )
        if parsed.username or parsed.password:
            raise KaguraFetchError(
                "URL credentials (user:pass@host) are not allowed; pass auth via headers",
                url=url,
            )
        if not parsed.hostname:
            raise KaguraFetchError("URL has no hostname", url=url)

        connect_ips = await self._validate_hostname(parsed.hostname, url)

        current_url = url
        for _ in range(self.max_redirects + 1):
            response = await self._stream_request(current_url, connect_ips)
            if response.status_code in (301, 302, 303, 307, 308):
                # ALL redirect error paths must close the streamed response
                # before raising — otherwise repeated malformed redirects
                # leak connections. The successful redirect path already
                # closes via the explicit aclose() before the loop continues.
                try:
                    location = response.headers.get("Location")
                    if not location:
                        raise KaguraFetchError(
                            f"redirect status {response.status_code} with no Location header",
                            url=current_url,
                        )
                    # Resolve relative redirects against current_url.
                    next_url = str(httpx.URL(current_url).join(location))
                    next_parsed = urlsplit(next_url)
                    if next_parsed.scheme.lower() not in _ALLOWED_SCHEMES:
                        raise KaguraFetchError(
                            f"redirect to unsupported scheme {next_parsed.scheme!r}",
                            url=next_url,
                        )
                    if next_parsed.scheme.lower() == "http" and not self.allow_http:
                        raise KaguraFetchError(
                            "redirect to http:// is disabled; use https://",
                            url=next_url,
                        )
                    if not next_parsed.hostname:
                        raise KaguraFetchError("redirect URL has no hostname", url=next_url)
                    # Reject embedded credentials on redirect targets too — the
                    # initial URL passes this check, but a malicious server can
                    # bounce us through `https://user:pass@host`.
                    if next_parsed.username or next_parsed.password:
                        raise KaguraFetchError(
                            "redirect URL has embedded credentials (user:pass@host)",
                            url=next_url,
                        )
                    connect_ips = await self._validate_hostname(next_parsed.hostname, next_url)
                finally:
                    await response.aclose()
                current_url = next_url
                continue

            # Non-redirect: stream body and return.
            return await self._read_body(response, original_url=url, final_url=current_url)

        raise KaguraFetchError(f"too many redirects (max {self.max_redirects})", url=url)

    async def _validate_hostname(self, hostname: str, url: str) -> list[str]:
        """Resolve ``hostname``, reject blocked IPs, and return the safe set.

        Every resolved IP is checked against the denylist (a single blocked
        address rejects the whole hostname). The returned IPs are then used as
        the connection targets in :meth:`_stream_request`, so httpx connects to
        exactly the addresses we validated rather than re-resolving the hostname
        at connect time — which is what closes the DNS-rebinding window (#188).
        The full validated set is returned (resolution order preserved) so the
        caller can fall back across addresses, e.g. IPv6→IPv4 on a host whose
        IPv6 path is broken, without ever connecting to an unvalidated IP.
        """
        try:
            addrs = await asyncio.to_thread(
                socket.getaddrinfo, hostname, None, 0, socket.SOCK_STREAM
            )
        except socket.gaierror as e:
            raise KaguraFetchError(f"hostname resolution failed: {e}", url=url) from e
        # info[4] is the sockaddr; first element is the IP string for both
        # AF_INET and AF_INET6. Coerce explicitly because pyright reads the
        # tuple as ``str | int``. Dedup while preserving resolution order so the
        # connect order is deterministic (we validate every address first).
        ips: list[str] = list(dict.fromkeys(str(info[4][0]) for info in addrs))
        if not ips:
            raise KaguraFetchError(f"no IP addresses for {hostname!r}", url=url)
        blocked = sorted({ip for ip in ips if is_blocked_ip(ip)})
        if blocked:
            raise KaguraFetchError(
                f"hostname {hostname!r} resolved to blocked IP(s): {', '.join(blocked)}",
                url=url,
            )
        return ips

    async def _stream_request(self, url: str, connect_ips: list[str]) -> httpx.Response:
        """Issue a streaming GET pinned to one of the pre-validated ``connect_ips``.

        The request URL's host is rewritten to a pre-validated IP so httpx
        connects to exactly that address with no connect-time re-resolution
        (#188). The original ``Host`` header is preserved for virtual-host
        routing, and the TLS SNI + certificate hostname verification use the
        real hostname via the ``sni_hostname`` request extension — so HTTPS
        certificates are still validated against the hostname, not the IP. The
        extension is inert for plain ``http``. Caller closes the response.

        Addresses are tried in resolution order; a *connection-level* failure
        (e.g. a host whose IPv6 address is unreachable) falls back to the next
        validated address, restoring the multi-address robustness that pinning
        a single IP would otherwise drop. Every candidate has already passed the
        denylist, so fallback never connects to an unvalidated IP. Non-connect
        HTTP errors are not retried.
        """
        parsed = httpx.URL(url)
        # ``netloc`` strips userinfo and renders host[:port] with correct IPv6
        # bracketing, so the Host header always carries the real hostname.
        host_header = parsed.netloc.decode("ascii")
        last_exc: httpx.HTTPError | None = None
        for connect_ip in connect_ips:
            pinned_url = parsed.copy_with(host=connect_ip)
            try:
                request = self._client.build_request(
                    "GET",
                    pinned_url,
                    headers={"Host": host_header},
                    extensions={"sni_hostname": parsed.host},
                )
                return await self._client.send(request, stream=True)
            except (httpx.ConnectError, httpx.ConnectTimeout) as e:
                last_exc = e  # try the next validated address
                continue
            except httpx.HTTPError as e:
                raise KaguraFetchError(f"network error: {e}", url=url) from e
        raise KaguraFetchError(f"network error: {last_exc}", url=url) from last_exc

    async def _read_body(
        self,
        response: httpx.Response,
        original_url: str,
        final_url: str,
    ) -> FetchResult:
        try:
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise KaguraFetchError(
                    f"HTTP {response.status_code}: {e.response.reason_phrase}",
                    url=final_url,
                ) from e

            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    advertised = int(content_length)
                    if advertised > self.max_bytes:
                        raise KaguraFetchError(
                            f"Content-Length {advertised} exceeds max_bytes {self.max_bytes}",
                            url=final_url,
                        )
                except ValueError:
                    pass  # Non-integer Content-Length: ignore and rely on stream cap.

            buf = bytearray()
            async for chunk in response.aiter_bytes():
                buf.extend(chunk)
                if len(buf) > self.max_bytes:
                    raise KaguraFetchError(
                        f"body exceeded max_bytes {self.max_bytes} during stream",
                        url=final_url,
                    )

            content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        finally:
            await response.aclose()

        return FetchResult(
            body=bytes(buf),
            content_type=content_type,
            source_uri=original_url,
            source_type="url",
            final_url=final_url,
            bytes_read=len(buf),
        )

    # --- Local file path -----------------------------------------------------

    async def _fetch_local(self, path_str: str) -> FetchResult:
        path = Path(path_str)
        try:
            resolved = path.resolve(strict=True)
        except FileNotFoundError as e:
            raise KaguraFetchError(f"file not found: {path}", url=path_str) from e
        except OSError as e:
            raise KaguraFetchError(f"file resolve failed: {e}", url=path_str) from e

        if not self.allow_system_paths and is_blocked_system_path(resolved):
            raise KaguraFetchError(
                f"path {resolved} is in a default-blocked system directory; "
                "pass allow_system_paths=True to override",
                url=path_str,
            )

        try:
            st = resolved.stat()
        except OSError as e:
            raise KaguraFetchError(f"stat failed: {e}", url=path_str) from e
        if not stat.S_ISREG(st.st_mode):
            raise KaguraFetchError(f"not a regular file: {resolved}", url=path_str)
        if st.st_size > self.max_bytes:
            raise KaguraFetchError(
                f"file size {st.st_size} exceeds max_bytes {self.max_bytes}", url=path_str
            )

        try:
            body = await asyncio.to_thread(resolved.read_bytes)
        except OSError as e:
            raise KaguraFetchError(f"file read failed: {e}", url=path_str) from e

        return FetchResult(
            body=body,
            content_type="",  # local files: caller does magic-byte detection
            source_uri=resolved.as_uri(),
            source_type="file",
            final_url=resolved.as_uri(),
            bytes_read=len(body),
        )
