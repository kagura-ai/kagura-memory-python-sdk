"""Tests for the YouTube transcript source resolver (issue #146).

The ``youtube-transcript-api`` client and the oEmbed ``httpx`` call are
mocked throughout so the unit suite needs no network access. One
integration test (``@pytest.mark.youtube``) hits a real video and is
default-skipped.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# The module under test references youtube_transcript_api's exception classes
# via the lazy loader (and the test helpers below import them directly), so the
# whole module must skip cleanly on a bare install without the optional
# [ingest-youtube] extra — otherwise collection breaks at import time.
pytest.importorskip("youtube_transcript_api")

from kagura_memory.exceptions import KaguraFetchError, KaguraIngestError  # noqa: E402
from kagura_memory.ingest import _youtube  # noqa: E402
from kagura_memory.ingest._youtube import (  # noqa: E402
    extract_video_id,
    fetch_youtube,
    is_youtube_url,
)

# --- URL parsing -------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://m.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("http://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ?si=abc123", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        # watch?v with a playlist param still resolves the single video.
        (
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PL12345",
            "dQw4w9WgXcQ",
        ),
        # Pure playlist / channel URLs have no single video id.
        ("https://www.youtube.com/playlist?list=PL12345", None),
        ("https://www.youtube.com/channel/UC12345", None),
        ("https://www.youtube.com/@somehandle", None),
        # Non-YouTube hosts.
        ("https://example.com/watch?v=dQw4w9WgXcQ", None),
        ("https://vimeo.com/123456", None),
    ],
)
def test_extract_video_id(url: str, expected: str | None) -> None:
    assert extract_video_id(url) == expected


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.youtube.com/watch?v=abc", True),
        ("https://youtube.com/watch?v=abc", True),
        ("https://m.youtube.com/watch?v=abc", True),
        ("https://youtu.be/abc", True),
        ("https://www.youtube.com/playlist?list=PL1", True),
        ("https://example.com/watch?v=abc", False),
        ("https://vimeo.com/123", False),
        ("not a url", False),
        ("", False),
    ],
)
def test_is_youtube_url(url: str, expected: bool) -> None:
    assert is_youtube_url(url) is expected


# --- Mock helpers ------------------------------------------------------------


def _snippet(text: str, start: float, duration: float) -> SimpleNamespace:
    return SimpleNamespace(text=text, start=start, duration=duration)


def _fake_transcript(snippets: list[SimpleNamespace], *, language_code: str = "en"):
    """Build a fake ``Transcript`` whose ``.fetch()`` yields ``snippets``."""
    transcript = MagicMock()
    transcript.language_code = language_code
    transcript.is_generated = False
    transcript.fetch.return_value = snippets
    return transcript


def _fake_api(transcript) -> MagicMock:
    """Build a fake ``YouTubeTranscriptApi`` whose ``.list()`` finds ``transcript``."""
    transcript_list = MagicMock()
    transcript_list.find_manually_created_transcript.return_value = transcript
    transcript_list.find_generated_transcript.side_effect = AssertionError(
        "manual transcript should be preferred"
    )
    transcript_list.__iter__.return_value = iter([transcript])

    api = MagicMock()
    api.list.return_value = transcript_list
    return api


def _patch_api(api: MagicMock):
    """Patch the lazy loader so it returns the real module but with a fake client.

    The implementation references the library's exception classes off the
    loaded module (e.g. ``module.TranscriptsDisabled``), so the fake must carry
    the real exception classes — only ``YouTubeTranscriptApi`` is swapped for a
    constructor that returns the mocked ``api``.
    """
    import youtube_transcript_api as real_module

    fake_module = SimpleNamespace(
        **{
            name: getattr(real_module, name)
            for name in dir(real_module)
            if not name.startswith("_")
        }
    )
    fake_module.YouTubeTranscriptApi = MagicMock(return_value=api)
    return patch.object(_youtube, "_load_youtube_transcript_api", return_value=fake_module)


def _patch_oembed(
    title: str = "Rick Astley - Never Gonna Give You Up", author: str = "Rick Astley"
):
    """Patch the oEmbed fetch to return a title/author."""
    return patch.object(
        _youtube,
        "_fetch_oembed",
        new=AsyncMock(return_value=(title, author)),
    )


# --- Happy path --------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_youtube_happy_path() -> None:
    snippets = [
        _snippet("Hello there", 0.0, 3.0),
        _snippet("welcome to the show", 3.0, 4.0),
        _snippet("this is later content", 75.0, 5.0),
    ]
    api = _fake_api(_fake_transcript(snippets))
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    with _patch_api(api), _patch_oembed():
        result = await fetch_youtube(url, max_bytes=10_000_000, read_timeout=10.0)

    assert result.content_type == "text/markdown"
    assert result.source_type == "url"
    assert result.source_uri == url
    body = result.body.decode("utf-8")
    # Title flows in as an H1 so TextExtractor promotes it to the title.
    assert body.startswith("# Rick Astley - Never Gonna Give You Up")
    assert "Rick Astley" in body  # author in the metadata line
    # Time-window section headings.
    assert "## [00:00]" in body
    assert "## [01:15]" in body
    assert "Hello there" in body
    assert "this is later content" in body
    assert result.bytes_read == len(result.body)


@pytest.mark.asyncio
async def test_fetch_youtube_falls_back_to_generated_transcript() -> None:
    snippets = [_snippet("auto caption", 0.0, 2.0)]
    generated = _fake_transcript(snippets)
    generated.is_generated = True

    transcript_list = MagicMock()
    transcript_list.find_manually_created_transcript.side_effect = _no_transcript_found()
    transcript_list.find_generated_transcript.return_value = generated

    api = MagicMock()
    api.list.return_value = transcript_list

    with _patch_api(api), _patch_oembed():
        result = await fetch_youtube(
            "https://youtu.be/dQw4w9WgXcQ", max_bytes=10_000_000, read_timeout=10.0
        )

    assert b"auto caption" in result.body


# --- Failure modes -----------------------------------------------------------


def _no_transcript_found() -> Exception:
    from youtube_transcript_api import NoTranscriptFound

    # NoTranscriptFound's __init__ signature varies; build a bare instance.
    exc = NoTranscriptFound.__new__(NoTranscriptFound)
    Exception.__init__(exc, "no transcript")
    return exc


@pytest.mark.asyncio
async def test_fetch_youtube_transcripts_disabled() -> None:
    from youtube_transcript_api import TranscriptsDisabled

    exc = TranscriptsDisabled.__new__(TranscriptsDisabled)
    Exception.__init__(exc, "disabled")

    api = MagicMock()
    api.list.side_effect = exc

    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    with _patch_api(api), _patch_oembed():
        with pytest.raises(KaguraFetchError) as ei:
            await fetch_youtube(url, max_bytes=10_000_000, read_timeout=10.0)
    msg = str(ei.value).lower()
    assert "caption" in msg or "transcript" in msg
    # The error carries the ORIGINAL url, not the bare video id, so the
    # derived IngestResult.source_uri surfaces the caller's URL.
    assert ei.value.url == url


@pytest.mark.asyncio
async def test_fetch_youtube_no_transcript_found_carries_original_url() -> None:
    """NoTranscriptFound surfaces the original URL, not the bare video id."""
    transcript_list = MagicMock()
    transcript_list.find_manually_created_transcript.side_effect = _no_transcript_found()
    transcript_list.find_generated_transcript.side_effect = _no_transcript_found()
    transcript_list.__iter__.return_value = iter([])

    api = MagicMock()
    api.list.return_value = transcript_list

    url = "https://youtu.be/dQw4w9WgXcQ"
    with _patch_api(api), _patch_oembed():
        with pytest.raises(KaguraFetchError) as ei:
            await fetch_youtube(url, max_bytes=10_000_000, read_timeout=10.0)
    assert ei.value.url == url


@pytest.mark.asyncio
async def test_fetch_youtube_age_restricted_carries_original_url() -> None:
    from youtube_transcript_api import AgeRestricted

    exc = AgeRestricted.__new__(AgeRestricted)
    Exception.__init__(exc, "age restricted")

    api = MagicMock()
    api.list.side_effect = exc

    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    with _patch_api(api), _patch_oembed():
        with pytest.raises(KaguraFetchError) as ei:
            await fetch_youtube(url, max_bytes=10_000_000, read_timeout=10.0)
    assert "age-restricted" in str(ei.value).lower()
    assert ei.value.url == url


@pytest.mark.asyncio
async def test_fetch_youtube_could_not_retrieve_carries_original_url() -> None:
    from youtube_transcript_api import CouldNotRetrieveTranscript

    # CouldNotRetrieveTranscript.__str__ formats a watch URL from .video_id, so
    # set it on the bare instance (its __init__ signature varies by version).
    exc = CouldNotRetrieveTranscript.__new__(CouldNotRetrieveTranscript)
    exc.video_id = "dQw4w9WgXcQ"
    Exception.__init__(exc, "ip blocked")

    api = MagicMock()
    api.list.side_effect = exc

    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    with _patch_api(api), _patch_oembed():
        with pytest.raises(KaguraFetchError) as ei:
            await fetch_youtube(url, max_bytes=10_000_000, read_timeout=10.0)
    assert ei.value.url == url


@pytest.mark.asyncio
async def test_fetch_youtube_oversize_body_raises() -> None:
    """A transcript whose body exceeds max_bytes is a hard failure."""
    snippets = [_snippet("word " * 50, 0.0, 2.0)]
    api = _fake_api(_fake_transcript(snippets))
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    with _patch_api(api), _patch_oembed():
        with pytest.raises(KaguraFetchError) as ei:
            # max_bytes far below the assembled markdown body forces the cap.
            await fetch_youtube(url, max_bytes=10, read_timeout=10.0)
    msg = str(ei.value).lower()
    assert "max_bytes" in msg
    assert ei.value.url == url


@pytest.mark.asyncio
async def test_fetch_youtube_transcript_timeout_raises() -> None:
    """A blocking transcript fetch that exceeds read_timeout raises KaguraFetchError."""
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    async def _never(awaitable: Any, *args: Any, **kwargs: Any) -> Any:
        # Close the to_thread coroutine we were handed so it isn't left
        # un-awaited (avoids a RuntimeWarning), then simulate the timeout.
        close = getattr(awaitable, "close", None)
        if close is not None:
            close()
        raise TimeoutError

    # Patch asyncio.wait_for to simulate the to_thread call timing out, so the
    # test stays fast and deterministic (no real hang).
    with patch.object(_youtube.asyncio, "wait_for", new=_never), _patch_oembed():
        with pytest.raises(KaguraFetchError) as ei:
            await fetch_youtube(url, max_bytes=10_000_000, read_timeout=5.0)
    msg = str(ei.value).lower()
    assert "timed out" in msg
    assert "5.0" in str(ei.value)
    assert ei.value.url == url


@pytest.mark.asyncio
async def test_fetch_youtube_video_unavailable() -> None:
    from youtube_transcript_api import VideoUnavailable

    exc = VideoUnavailable.__new__(VideoUnavailable)
    Exception.__init__(exc, "unavailable")

    api = MagicMock()
    api.list.side_effect = exc

    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    with _patch_api(api), _patch_oembed():
        with pytest.raises(KaguraFetchError) as ei:
            await fetch_youtube(url, max_bytes=10_000_000, read_timeout=10.0)
    assert ei.value.url == url


@pytest.mark.asyncio
async def test_fetch_youtube_playlist_rejected() -> None:
    with pytest.raises(KaguraFetchError) as ei:
        await fetch_youtube(
            "https://www.youtube.com/playlist?list=PL12345",
            max_bytes=10_000_000,
            read_timeout=10.0,
        )
    assert "playlist" in str(ei.value).lower() or "single video" in str(ei.value).lower()


@pytest.mark.asyncio
async def test_fetch_youtube_missing_dependency() -> None:
    with patch.object(
        _youtube,
        "_load_youtube_transcript_api",
        side_effect=KaguraIngestError(
            "youtube-transcript-api is not installed. Install with: "
            "pip install 'kagura-memory[ingest-youtube]'"
        ),
    ):
        with pytest.raises(KaguraIngestError) as ei:
            await fetch_youtube(
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                max_bytes=10_000_000,
                read_timeout=10.0,
            )
    assert "ingest-youtube" in str(ei.value)


@pytest.mark.asyncio
async def test_fetch_youtube_oembed_failure_degrades_gracefully() -> None:
    """An oEmbed failure must NOT fail the ingest — title falls back to id."""
    snippets = [_snippet("content here", 0.0, 2.0)]
    api = _fake_api(_fake_transcript(snippets))

    with (
        _patch_api(api),
        patch.object(_youtube, "_fetch_oembed", new=AsyncMock(return_value=(None, None))),
    ):
        result = await fetch_youtube(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            max_bytes=10_000_000,
            read_timeout=10.0,
        )

    body = result.body.decode("utf-8")
    # Title falls back to the video id.
    assert body.startswith("# dQw4w9WgXcQ")
    assert b"content here" in result.body


# --- Real loader (no dep installed → KaguraIngestError) ----------------------


def test_load_youtube_transcript_api_success() -> None:
    """The real loader returns the module when the dep is installed."""
    import youtube_transcript_api as real_module

    assert _youtube._load_youtube_transcript_api() is real_module


def test_is_youtube_url_handles_unparseable() -> None:
    """A URL whose parsing raises ValueError returns False, not an error."""
    # An unterminated IPv6 literal makes urlsplit raise ValueError; the helper
    # must swallow it and report "not a YouTube URL".
    assert is_youtube_url("https://[invalid") is False


def test_extract_video_id_handles_unparseable() -> None:
    """A URL whose parsing raises ValueError returns None, not an error."""
    assert extract_video_id("https://[invalid") is None


def test_load_youtube_transcript_api_importerror(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("youtube_transcript_api"):
            raise ImportError("simulated missing dep")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(KaguraIngestError) as ei:
        _youtube._load_youtube_transcript_api()
    assert "ingest-youtube" in str(ei.value)


# --- Internal helpers --------------------------------------------------------


def test_format_timestamp_hours() -> None:
    """Past an hour, the timestamp gains an ``h:mm:ss`` field."""
    assert _youtube._format_timestamp(3725.0) == "1:02:05"
    assert _youtube._format_timestamp(59.0) == "00:59"


def test_segment_windows_skips_empty_snippets() -> None:
    """Blank/whitespace-only snippets are dropped before windowing."""
    snippets = [
        _snippet("   ", 0.0, 1.0),
        _snippet("real text", 1.0, 1.0),
        _snippet("", 2.0, 1.0),
    ]
    windows = _youtube._segment_windows(snippets)
    assert windows == [(1.0, "real text")]


def test_segment_windows_closes_before_overrunning_char_cap() -> None:
    """A new window opens when the next snippet WOULD push past _WINDOW_CHARS.

    The close check looks ahead at the snippet about to be added, so no window
    overruns _WINDOW_CHARS by a whole snippet.
    """
    from kagura_memory.ingest._youtube import _WINDOW_CHARS

    half = "a" * (_WINDOW_CHARS // 2 + 50)  # two of these would exceed the cap
    snippets = [
        _snippet(half, 0.0, 1.0),
        _snippet(half, 1.0, 1.0),  # adding this would overrun -> opens window 2
    ]
    windows = _youtube._segment_windows(snippets)
    assert len(windows) == 2
    # Each window holds exactly one snippet, so neither overruns the cap.
    assert len(windows[0][1]) <= _WINDOW_CHARS
    assert len(windows[1][1]) <= _WINDOW_CHARS
    assert windows[1][0] == 1.0  # window 2 anchored at the 2nd snippet's start


def test_segment_windows_oversized_single_snippet_forms_own_window() -> None:
    """A single snippet larger than _WINDOW_CHARS still forms one window (no loop)."""
    from kagura_memory.ingest._youtube import _WINDOW_CHARS

    giant = "z" * (_WINDOW_CHARS * 3)
    windows = _youtube._segment_windows([_snippet(giant, 0.0, 1.0)])
    assert windows == [(0.0, giant)]


@pytest.mark.asyncio
async def test_resolve_transcript_first_available_fallback() -> None:
    """When no manual/generated transcript matches, the first available wins."""
    snippets = [_snippet("fallback caption", 0.0, 2.0)]
    fallback = _fake_transcript(snippets, language_code="de")

    transcript_list = MagicMock()
    transcript_list.find_manually_created_transcript.side_effect = _no_transcript_found()
    transcript_list.find_generated_transcript.side_effect = _no_transcript_found()
    transcript_list.__iter__.return_value = iter([fallback])

    api = MagicMock()
    api.list.return_value = transcript_list

    with _patch_api(api), _patch_oembed():
        result = await fetch_youtube(
            "https://youtu.be/dQw4w9WgXcQ", max_bytes=10_000_000, read_timeout=10.0
        )
    assert b"fallback caption" in result.body


@pytest.mark.asyncio
async def test_fetch_youtube_truncates_when_over_text_cap() -> None:
    """A transcript over _MAX_TOTAL_TEXT_CHARS bytes is truncated with a marker."""
    from kagura_memory.ingest._youtube import _MAX_TOTAL_TEXT_CHARS

    # One huge snippet whose text alone blows past the char/byte cap.
    big = "x" * (_MAX_TOTAL_TEXT_CHARS + 5000)
    api = _fake_api(_fake_transcript([_snippet(big, 0.0, 2.0)]))

    with _patch_api(api), _patch_oembed():
        # max_bytes above the text cap so the _build_markdown truncation path
        # (not the max_bytes hard cap) is the one exercised here.
        result = await fetch_youtube(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            max_bytes=_MAX_TOTAL_TEXT_CHARS + 1_000_000,
            read_timeout=10.0,
        )
    body = result.body.decode("utf-8")
    assert "[transcript truncated: exceeded length cap]" in body
    assert len(result.body) <= _MAX_TOTAL_TEXT_CHARS


@pytest.mark.asyncio
async def test_fetch_oembed_degrades_on_http_error() -> None:
    """A non-200 / network error from oEmbed yields (None, None), never raises."""
    import httpx

    class _FailClient:
        async def __aenter__(self) -> _FailClient:
            return self

        async def __aexit__(self, *exc: Any) -> None:
            return None

        async def get(self, *args: Any, **kwargs: Any) -> Any:
            raise httpx.ConnectError("boom")

    with patch.object(httpx, "AsyncClient", return_value=_FailClient()):
        title, author = await _youtube._fetch_oembed(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ", read_timeout=10.0
        )
    assert title is None
    assert author is None


@pytest.mark.asyncio
async def test_fetch_oembed_returns_title_and_author() -> None:
    """A 200 oEmbed response yields the parsed title/author."""
    import httpx

    class _OkResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"title": "My Video", "author_name": "Some Channel"}

    class _OkClient:
        async def __aenter__(self) -> _OkClient:
            return self

        async def __aexit__(self, *exc: Any) -> None:
            return None

        async def get(self, *args: Any, **kwargs: Any) -> _OkResponse:
            return _OkResponse()

    with patch.object(httpx, "AsyncClient", return_value=_OkClient()):
        title, author = await _youtube._fetch_oembed(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ", read_timeout=10.0
        )
    assert title == "My Video"
    assert author == "Some Channel"


@pytest.mark.asyncio
async def test_fetch_oembed_threads_connect_timeout() -> None:
    """The caller's connect_timeout reaches the httpx.Timeout connect field."""
    import httpx

    captured: dict[str, Any] = {}

    class _FailClient:
        async def __aenter__(self) -> _FailClient:
            return self

        async def __aexit__(self, *exc: Any) -> None:
            return None

        async def get(self, *args: Any, **kwargs: Any) -> Any:
            raise httpx.ConnectError("boom")

    def _make(*args: Any, **kwargs: Any) -> Any:
        captured["timeout"] = kwargs.get("timeout")
        return _FailClient()

    with patch.object(httpx, "AsyncClient", new=_make):
        await _youtube._fetch_oembed(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            connect_timeout=3.5,
            read_timeout=10.0,
        )
    timeout = captured["timeout"]
    assert timeout.connect == 3.5
    assert timeout.read == 10.0


# --- Integration (default-skipped) ------------------------------------------


@pytest.mark.youtube
@pytest.mark.asyncio
async def test_fetch_youtube_real_video() -> None:
    pytest.importorskip("youtube_transcript_api")
    # "Me at the zoo" — the first YouTube video, stable and captioned.
    url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
    result = await fetch_youtube(url, max_bytes=10_000_000, read_timeout=30.0)
    assert result.content_type == "text/markdown"
    assert result.body
    assert result.body.decode("utf-8").startswith("# ")
