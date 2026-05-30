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

from kagura_memory.exceptions import KaguraFetchError, KaguraIngestError
from kagura_memory.ingest import _youtube
from kagura_memory.ingest._youtube import (
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

    with _patch_api(api), _patch_oembed():
        with pytest.raises(KaguraFetchError) as ei:
            await fetch_youtube(
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                max_bytes=10_000_000,
                read_timeout=10.0,
            )
    msg = str(ei.value).lower()
    assert "caption" in msg or "transcript" in msg


@pytest.mark.asyncio
async def test_fetch_youtube_video_unavailable() -> None:
    from youtube_transcript_api import VideoUnavailable

    exc = VideoUnavailable.__new__(VideoUnavailable)
    Exception.__init__(exc, "unavailable")

    api = MagicMock()
    api.list.side_effect = exc

    with _patch_api(api), _patch_oembed():
        with pytest.raises(KaguraFetchError):
            await fetch_youtube(
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                max_bytes=10_000_000,
                read_timeout=10.0,
            )


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
    def _raise() -> Any:
        raise ImportError("no module")

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
