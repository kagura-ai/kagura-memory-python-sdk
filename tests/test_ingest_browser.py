"""Tests for the opt-in browser-rendered URL fetch (issue #145).

Playwright is fully mocked here — no real Chromium is launched. A fake
``async_playwright`` chain is injected via the
``kagura_memory.ingest._browser._load_async_playwright`` seam.

One integration test (``test_browser_render_real_chromium``) is gated behind
the ``browser`` marker AND ``importorskip`` so it is default-skipped in CI
and only runs manually with a real ``playwright install``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from kagura_memory.exceptions import KaguraFetchError, KaguraIngestError
from kagura_memory.ingest import _browser
from kagura_memory.ingest._browser import BrowserFetcher


class _FakeRequest:
    def __init__(self, url: str) -> None:
        self.url = url


class _FakeRoute:
    """Minimal Playwright ``Route`` stand-in for the route handler."""

    def __init__(self, url: str) -> None:
        self.request = _FakeRequest(url)
        self.abort = AsyncMock()
        self.continue_ = AsyncMock()


def _make_fake_playwright(
    *,
    html: str = "<html><body><h1>Hi</h1></body></html>",
    final_url: str = "https://example.com/",
    goto_side_effect: Exception | None = None,
) -> tuple[Any, MagicMock, MagicMock]:
    """Build a fake ``async_playwright`` factory and its page/context mocks.

    Returns ``(factory, page, context)`` so tests can assert on the page's
    ``route``/``goto`` calls.
    """
    page = MagicMock()
    page.route = AsyncMock()
    page.goto = AsyncMock(side_effect=goto_side_effect)
    page.content = AsyncMock(return_value=html)
    page.url = final_url

    context = MagicMock()
    context.new_page = AsyncMock(return_value=page)
    context.close = AsyncMock()

    browser = MagicMock()
    browser.new_context = AsyncMock(return_value=context)
    browser.close = AsyncMock()

    chromium = MagicMock()
    chromium.launch = AsyncMock(return_value=browser)

    pw = MagicMock()
    pw.chromium = chromium
    pw.stop = AsyncMock()

    manager = MagicMock()
    manager.start = AsyncMock(return_value=pw)

    def factory() -> MagicMock:
        return manager

    return factory, page, context


def _patch_unblocked_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every hostname resolve to a public, non-blocked IP."""

    def fake_getaddrinfo(host: str, *args: Any, **kwargs: Any) -> Any:
        return [(2, 1, 6, "", ("93.184.216.34", 0))]

    monkeypatch.setattr(_browser.socket, "getaddrinfo", fake_getaddrinfo)


def _patch_blocked_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every hostname resolve to a blocked (loopback) IP."""

    def fake_getaddrinfo(host: str, *args: Any, **kwargs: Any) -> Any:
        return [(2, 1, 6, "", ("127.0.0.1", 0))]

    monkeypatch.setattr(_browser.socket, "getaddrinfo", fake_getaddrinfo)


# --- Happy path --------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path_returns_rendered_html(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_unblocked_resolver(monkeypatch)
    html = "<html><body><article>rendered</article></body></html>"
    factory, page, _ctx = _make_fake_playwright(html=html, final_url="https://example.com/final")
    monkeypatch.setattr(_browser, "_load_async_playwright", lambda: factory)

    async with BrowserFetcher() as fetcher:
        result = await fetcher.fetch("https://example.com/")

    assert result.content_type == "text/html"
    assert result.body == html.encode("utf-8")
    assert result.source_type == "url"
    assert result.source_uri == "https://example.com/"
    assert result.final_url == "https://example.com/final"
    assert result.bytes_read == len(html.encode("utf-8"))
    page.goto.assert_awaited_once()
    page.route.assert_awaited_once()


# --- SSRF guards -------------------------------------------------------------


@pytest.mark.asyncio
async def test_initial_blocked_ip_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_blocked_resolver(monkeypatch)
    factory, _page, _ctx = _make_fake_playwright()
    monkeypatch.setattr(_browser, "_load_async_playwright", lambda: factory)

    fetcher = BrowserFetcher()
    with pytest.raises(KaguraFetchError, match="blocked IP"):
        await fetcher.fetch("https://internal.example.com/")
    await fetcher.close()


@pytest.mark.asyncio
async def test_route_handler_aborts_blocked_host(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_blocked_resolver(monkeypatch)
    fetcher = BrowserFetcher()
    route = _FakeRoute("https://metadata.internal/latest")

    await fetcher._route_handler(route)  # type: ignore[arg-type]

    route.abort.assert_awaited_once()
    route.continue_.assert_not_awaited()


@pytest.mark.asyncio
async def test_route_handler_continues_public_host(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_unblocked_resolver(monkeypatch)
    fetcher = BrowserFetcher()
    route = _FakeRoute("https://cdn.example.com/app.js")

    await fetcher._route_handler(route)  # type: ignore[arg-type]

    route.continue_.assert_awaited_once()
    route.abort.assert_not_awaited()


@pytest.mark.asyncio
async def test_route_handler_aborts_http_when_disallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_unblocked_resolver(monkeypatch)
    fetcher = BrowserFetcher(allow_http=False)
    route = _FakeRoute("http://cdn.example.com/app.js")

    await fetcher._route_handler(route)  # type: ignore[arg-type]

    route.abort.assert_awaited_once()
    route.continue_.assert_not_awaited()


@pytest.mark.asyncio
async def test_route_handler_allows_data_scheme(monkeypatch: pytest.MonkeyPatch) -> None:
    fetcher = BrowserFetcher()
    route = _FakeRoute("data:text/html,<p>x</p>")

    await fetcher._route_handler(route)  # type: ignore[arg-type]

    route.continue_.assert_awaited_once()


# --- Missing dependency ------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_playwright_raises_ingest_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # _load_async_playwright raising ImportError → fetch() surfaces a
    # KaguraIngestError pointing at the extra (tested here by simulating the
    # ImportError branch directly).
    def boom() -> Any:
        raise KaguraIngestError(
            "playwright is not installed. Install with: "
            "pip install 'kagura-memory[ingest-browser]' then run: "
            "playwright install chromium"
        )

    _patch_unblocked_resolver(monkeypatch)
    monkeypatch.setattr(_browser, "_load_async_playwright", boom)

    fetcher = BrowserFetcher()
    with pytest.raises(KaguraIngestError, match=r"ingest-browser"):
        await fetcher.fetch("https://example.com/")


@pytest.mark.asyncio
async def test_load_async_playwright_import_error_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("playwright"):
            raise ImportError("nope")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(KaguraIngestError) as exc:
        _browser._load_async_playwright()
    assert "playwright install chromium" in str(exc.value)


# --- Oversize ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_oversize_rendered_html_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_unblocked_resolver(monkeypatch)
    big_html = "<html>" + ("x" * 5000) + "</html>"
    factory, _page, _ctx = _make_fake_playwright(html=big_html)
    monkeypatch.setattr(_browser, "_load_async_playwright", lambda: factory)

    fetcher = BrowserFetcher(max_bytes=1000)
    with pytest.raises(KaguraFetchError, match="exceeds max_bytes"):
        await fetcher.fetch("https://example.com/")
    await fetcher.close()


# --- Navigation errors -------------------------------------------------------


@pytest.mark.asyncio
async def test_navigation_timeout_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_unblocked_resolver(monkeypatch)

    class _PlaywrightTimeoutError(Exception):
        pass

    factory, _page, _ctx = _make_fake_playwright(
        goto_side_effect=_PlaywrightTimeoutError("Timeout 60000ms exceeded")
    )
    monkeypatch.setattr(_browser, "_load_async_playwright", lambda: factory)

    fetcher = BrowserFetcher()
    with pytest.raises(KaguraFetchError, match="navigation failed"):
        await fetcher.fetch("https://example.com/")
    await fetcher.close()


# --- URL validation ----------------------------------------------------------


@pytest.mark.asyncio
async def test_rejects_non_http_scheme() -> None:
    fetcher = BrowserFetcher()
    with pytest.raises(KaguraFetchError, match="unsupported scheme"):
        await fetcher.fetch("ftp://example.com/x")
    await fetcher.close()


@pytest.mark.asyncio
async def test_rejects_http_when_not_allowed() -> None:
    fetcher = BrowserFetcher(allow_http=False)
    with pytest.raises(KaguraFetchError, match="http:// is disabled"):
        await fetcher.fetch("http://example.com/")
    await fetcher.close()


@pytest.mark.asyncio
async def test_rejects_embedded_credentials() -> None:
    fetcher = BrowserFetcher()
    with pytest.raises(KaguraFetchError, match="credentials"):
        await fetcher.fetch("https://user:pass@example.com/")
    await fetcher.close()


@pytest.mark.asyncio
async def test_rejects_oversized_url() -> None:
    fetcher = BrowserFetcher()
    long_url = "https://example.com/" + ("a" * 9000)
    with pytest.raises(KaguraFetchError, match="exceeds"):
        await fetcher.fetch(long_url)
    await fetcher.close()


# --- Orchestrator missing-dependency path ------------------------------------


@pytest.mark.asyncio
async def test_ingest_render_without_playwright_yields_fetch_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FileIngestor.ingest(render=True) without playwright → IngestResult.

    The missing optional dependency must surface as a ``step="fetch"`` error
    record on the result, never an uncaught exception.
    """
    from kagura_memory.ingest.ingestor import FileIngestor

    # Force the BrowserFetcher's lazy loader to raise as if playwright is
    # absent, regardless of whether it happens to be installed.
    def boom() -> Any:
        raise KaguraIngestError(
            "playwright is not installed. Install with: pip install 'kagura-memory[ingest-browser]'"
        )

    monkeypatch.setattr(_browser, "_load_async_playwright", boom)

    fake_client = MagicMock()
    fake_text = MagicMock()
    fake_text.name = "claude"
    ingestor = FileIngestor(
        client=fake_client,
        text_provider=fake_text,
        vision_provider=None,
        vision_provider_name=None,
    )

    result = await ingestor.ingest(
        "https://spa.example.com/",
        context_id="ctx-1",
        render=True,
    )

    assert not result.success
    assert any(e.step == "fetch" for e in result.errors)
    assert any("playwright" in e.message.lower() for e in result.errors)


# --- Integration (default-skipped) -------------------------------------------


@pytest.mark.browser
@pytest.mark.asyncio
async def test_browser_render_real_chromium() -> None:
    """End-to-end render against a real Chromium (manual only).

    Default-skipped: requires `playwright install chromium` plus network
    access to example.com. Run with ``pytest -m browser``.
    """
    pytest.importorskip("playwright", reason="playwright not installed")

    async with BrowserFetcher() as fetcher:
        result = await fetcher.fetch("https://example.com/")

    assert result.content_type == "text/html"
    assert b"<html" in result.body.lower() or b"<!doctype" in result.body.lower()
