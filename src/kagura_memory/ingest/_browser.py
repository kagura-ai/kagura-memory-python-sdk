"""Browser-rendered URL fetch for JS-heavy / SPA pages (issue #145).

An OPT-IN counterpart to :class:`kagura_memory.ingest.fetcher.Fetcher`.
When the ingest pipeline is asked to render a URL, :class:`BrowserFetcher`
drives a headless Chromium (via Playwright's async API), waits for the page
to settle, captures the rendered DOM HTML, and returns it as a
:class:`FetchResult` with ``content_type="text/html"`` so it flows through
the existing HTML extractor → chunker → provider pipeline unchanged.

Playwright is a heavy optional dependency: it is lazily imported inside
:func:`_load_async_playwright` and surfaces as a :class:`KaguraIngestError`
pointing at the ``[ingest-browser]`` extra (plus the required
``playwright install chromium`` step) when missing.

Security
--------
Chromium performs its own DNS resolution, redirect following, and
sub-resource (XHR/fetch/image/script) loading entirely outside the Python
process, which would bypass a one-shot pre-flight SSRF check. To close that
gap, every request the browser makes is intercepted via ``page.route`` and
its host is re-resolved against the same RFC1918/loopback/link-local/IMDS
denylist used by :class:`Fetcher` (see
:func:`kagura_memory.ingest._safety.is_blocked_ip`). Requests resolving to a
blocked IP — including redirect targets and sub-resources — are aborted.
http(s) sub-resources are also gated by ``allow_http``.
"""

from __future__ import annotations

import asyncio
import socket
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from ..exceptions import KaguraFetchError, KaguraIngestError
from ._safety import is_blocked_ip
from .fetcher import (
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_MAX_BYTES,
    DEFAULT_READ_TIMEOUT,
    MAX_URL_LENGTH,
    FetchResult,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from playwright.async_api import Route  # type: ignore[import-not-found]

_ALLOWED_SCHEMES: frozenset[str] = frozenset({"http", "https"})
_DEFAULT_WAIT_UNTIL: str = "networkidle"


def _load_async_playwright() -> Any:
    """Lazily import Playwright's ``async_playwright`` factory.

    This is the single import seam the browser fetcher depends on; tests
    monkeypatch it to inject a fake Playwright chain so no real Chromium is
    required.

    Returns:
        The ``playwright.async_api.async_playwright`` callable.

    Raises:
        KaguraIngestError: If Playwright is not installed. The message
            points the user at the ``[ingest-browser]`` extra and the
            required ``playwright install chromium`` browser download.
    """
    try:
        from playwright.async_api import async_playwright  # type: ignore[import-not-found]
    except ImportError as e:
        raise KaguraIngestError(
            "playwright is not installed. Install with: "
            "pip install 'kagura-memory[ingest-browser]' then run: "
            "playwright install chromium"
        ) from e
    return async_playwright  # pragma: no cover - requires playwright installed


class BrowserFetcher:
    """Headless-Chromium fetcher for JS-rendered pages.

    Use as an async context manager::

        async with BrowserFetcher() as fetcher:
            result = await fetcher.fetch("https://spa.example.com/")

    The Playwright browser lifecycle is lazy: Chromium is launched on the
    first :meth:`fetch` call (or eagerly via :meth:`__aenter__` /
    :meth:`_ensure_browser`) and torn down in :meth:`close`.

    The returned :class:`FetchResult` always has ``content_type="text/html"``
    so the rendered DOM routes to the HTML extractor.

    Args:
        max_bytes: Maximum size of the *rendered* HTML (UTF-8 encoded).
            Exceeding this raises :class:`KaguraFetchError`. Note this caps
            the serialized DOM only; the total weight of sub-resources
            (images, scripts) fetched during rendering is not separately
            metered in this version.
        connect_timeout: Reserved for parity with :class:`Fetcher`; the
            browser uses ``read_timeout`` as the navigation deadline.
        read_timeout: Navigation timeout (seconds). With
            ``wait_until="networkidle"`` a page that never goes idle would
            otherwise hang forever, so this deadline is the backstop — on
            expiry the fetch fails with :class:`KaguraFetchError`.
        allow_http: When False (default), ``http://`` navigation targets and
            ``http://`` sub-resource requests are rejected/aborted.
        wait_until: Playwright ``goto`` wait condition. Default
            ``"networkidle"`` waits until the network is quiet, which is the
            best heuristic for single-page apps that hydrate after load; the
            navigation timeout above bounds the worst case.
    """

    def __init__(
        self,
        max_bytes: int = DEFAULT_MAX_BYTES,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        read_timeout: float = DEFAULT_READ_TIMEOUT,
        allow_http: bool = False,
        wait_until: str = _DEFAULT_WAIT_UNTIL,
    ):
        self.max_bytes = max_bytes
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.allow_http = allow_http
        self.wait_until = wait_until
        # Navigation timeout in milliseconds (Playwright uses ms).
        self.nav_timeout_ms = int(read_timeout * 1000)

        self._playwright: Any = None
        self._browser: Any = None
        # Per-fetch DNS cache for the route handler so a page with many
        # sub-resources to the same host resolves each host once.
        self._resolve_cache: dict[str, bool] = {}

    async def __aenter__(self) -> BrowserFetcher:
        await self._ensure_browser()
        return self

    async def __aexit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        await self.close()

    async def _ensure_browser(self) -> None:
        """Launch Chromium on first use (idempotent)."""
        if self._browser is not None:
            return
        async_playwright = _load_async_playwright()
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=True)

    async def close(self) -> None:
        """Tear down the browser and Playwright driver (idempotent)."""
        if self._browser is not None:
            try:
                await self._browser.close()
            finally:
                self._browser = None
        if self._playwright is not None:
            try:
                await self._playwright.stop()
            finally:
                self._playwright = None

    # --- Public API ----------------------------------------------------------

    async def fetch(self, url: str) -> FetchResult:
        """Render ``url`` in headless Chromium and return its HTML.

        Args:
            url: An ``http`` / ``https`` URL. Local file paths and other
                schemes are rejected — browser rendering only applies to
                URLs.

        Returns:
            A :class:`FetchResult` with ``content_type="text/html"`` whose
            ``body`` is the rendered DOM serialized as UTF-8.

        Raises:
            KaguraFetchError: For any URL-safety violation (bad scheme,
                http when disallowed, embedded credentials, missing
                hostname, oversize URL), an initial host resolving to a
                blocked IP, navigation timeout/error, or rendered HTML
                exceeding ``max_bytes``.
            KaguraIngestError: If Playwright is not installed.
        """
        self._validate_url(url)
        parsed = urlsplit(url)
        hostname = parsed.hostname
        assert hostname is not None  # guaranteed by _validate_url
        await self._reject_blocked_host(hostname, url)

        await self._ensure_browser()
        self._resolve_cache = {}

        context = await self._browser.new_context()
        try:
            # new_page() is inside the try so a failure here still closes the
            # freshly-created context (otherwise it would leak a browser context).
            page = await context.new_page()
            await page.route("**/*", self._route_handler)
            try:
                response = await page.goto(
                    url, wait_until=self.wait_until, timeout=self.nav_timeout_ms
                )
            except KaguraFetchError:
                raise
            except Exception as e:  # noqa: BLE001 - normalized to a domain error below
                raise KaguraFetchError(f"browser navigation failed: {e}", url=url) from e

            # Reject rendered error pages (4xx/5xx) the way Fetcher does via
            # raise_for_status(). A None response (about:blank, data: URLs)
            # carries no status and is left to flow through.
            if response is not None and isinstance(response.status, int) and response.status >= 400:
                raise KaguraFetchError(
                    f"browser navigation returned HTTP {response.status}", url=url
                )

            html = await page.content()
            body = html.encode("utf-8")
            if len(body) > self.max_bytes:
                raise KaguraFetchError(
                    f"rendered HTML {len(body)} bytes exceeds max_bytes {self.max_bytes}",
                    url=url,
                )
            final_url = page.url or url
        finally:
            await context.close()

        return FetchResult(
            body=body,
            content_type="text/html",
            source_uri=url,
            source_type="url",
            final_url=final_url,
            bytes_read=len(body),
        )

    # --- Internals -----------------------------------------------------------

    def _validate_url(self, url: str) -> None:
        """Mirror :class:`Fetcher`'s URL safety checks (raises on violation)."""
        if not url or not isinstance(url, str):
            raise KaguraFetchError("url must be a non-empty string", url=url or None)
        if len(url) > MAX_URL_LENGTH:
            raise KaguraFetchError(f"URL exceeds {MAX_URL_LENGTH} chars (got {len(url)})", url=url)
        parsed = urlsplit(url)
        scheme = parsed.scheme.lower()
        if scheme not in _ALLOWED_SCHEMES:
            raise KaguraFetchError(
                f"unsupported scheme {parsed.scheme!r}; only http(s) is allowed", url=url
            )
        if scheme == "http" and not self.allow_http:
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

    async def _reject_blocked_host(self, hostname: str, url: str) -> None:
        """Resolve ``hostname`` and raise if any IP is in the denylist."""
        if await self._host_is_blocked(hostname):
            raise KaguraFetchError(f"hostname {hostname!r} resolved to a blocked IP", url=url)

    async def _host_is_blocked(self, hostname: str) -> bool:
        """True iff ``hostname`` resolves to any blocked IP (cached per fetch).

        Any resolution failure is treated as blocked (fail-closed): besides
        :class:`socket.gaierror`, :func:`socket.getaddrinfo` can raise
        :class:`UnicodeError` for a malformed IDN host or other
        :class:`OSError` subclasses. All of these are conservatively treated
        as "blocked", matching the posture of :func:`is_blocked_ip` for
        malformed addresses.
        """
        cached = self._resolve_cache.get(hostname)
        if cached is not None:
            return cached
        try:
            addrs = await asyncio.to_thread(
                socket.getaddrinfo, hostname, None, 0, socket.SOCK_STREAM
            )
        except (socket.gaierror, UnicodeError, OSError):
            self._resolve_cache[hostname] = True
            return True
        ips = {str(info[4][0]) for info in addrs}
        blocked = not ips or any(is_blocked_ip(ip) for ip in ips)
        self._resolve_cache[hostname] = blocked
        return blocked

    async def _route_handler(self, route: Route) -> None:
        """Intercept every browser request and SSRF-gate it.

        Aborts requests whose host resolves to a blocked IP (covering
        redirects, XHR/fetch, and sub-resources to internal addresses) and
        ``http://`` requests when ``allow_http`` is False; otherwise lets the
        request continue.

        A route must be resolved exactly once. Any unexpected exception while
        deciding is treated fail-closed: the request is aborted rather than
        left dangling (which would hang navigation until the timeout). If
        resolving the route itself raises — e.g. the context is closing — that
        is swallowed, since the route can no longer be acted upon.
        """
        try:
            request_url = route.request.url
            parsed = urlsplit(request_url)
            scheme = parsed.scheme.lower()
            # Non-network schemes (data:, blob:, about:) are harmless for SSRF.
            if scheme not in _ALLOWED_SCHEMES:
                await route.continue_()
                return
            if scheme == "http" and not self.allow_http:
                await route.abort()
                return
            hostname = parsed.hostname
            if not hostname or await self._host_is_blocked(hostname):
                await route.abort()
                return
            await route.continue_()
        except Exception:  # noqa: BLE001 - fail closed, never let a route dangle
            # Any unexpected error -> abort (fail-closed). Swallow a secondary
            # error from abort() itself (e.g. the context is tearing down) so
            # the route handler never propagates into Playwright's callback.
            try:
                await route.abort()
            except Exception:  # noqa: BLE001 - route already resolved / closing
                pass


__all__ = ["BrowserFetcher"]
