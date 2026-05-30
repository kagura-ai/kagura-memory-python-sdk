"""YouTube transcript source resolver for the ingestion pipeline (issue #146).

A YouTube video's ingestible content is its *transcript* (captions), not the
media stream. This module is a **Fetcher-layer source resolver**: it resolves a
single-video YouTube URL to a timestamped transcript, formats it as Markdown
(an H1 title + ``## [mm:ss]`` time-window sections), and returns a
:class:`~kagura_memory.ingest.fetcher.FetchResult` with
``content_type="text/markdown"``. That flows unchanged through the existing
``TextExtractor`` (which splits on ``##`` ATX headings and promotes the first
H1 to the document title) → chunker → provider pipeline, so no orchestrator
structural change is needed for sections or title.

Dependencies (opt-in ``[ingest-youtube]`` extra):

* ``youtube-transcript-api`` (>=1.0) — captions retrieval, no API key.
* YouTube **oEmbed** (``https://www.youtube.com/oembed``) — title/channel
  metadata, no API key. oEmbed is best-effort: a failure degrades the title to
  the video id rather than failing the whole ingest.

Chapters are deferred to a follow-up (they require yt-dlp / InnerTube). v1 uses
time-window segmentation of the transcript instead.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx

from ..exceptions import KaguraFetchError, KaguraIngestError
from .extractors._util import MAX_TOTAL_TEXT_CHARS as _MAX_TOTAL_TEXT_CHARS
from .fetcher import FetchResult

# Hosts that identify a YouTube URL.
_YOUTUBE_HOSTS: frozenset[str] = frozenset(
    {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}
)

# A YouTube video id is exactly 11 chars of [A-Za-z0-9_-].
_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

# Default caption languages, in descending priority.
_DEFAULT_LANGUAGES: tuple[str, ...] = ("en",)

# Fallback connect timeout (seconds) when the caller does not pass one, e.g.
# unit tests that call fetch_youtube/_fetch_oembed directly.
_DEFAULT_CONNECT_TIMEOUT: float = 10.0

# Time-window segmentation: group caption snippets into windows of roughly this
# duration OR this many characters, whichever boundary is hit first. Each window
# becomes one ``## [mm:ss]`` Markdown section.
_WINDOW_SECONDS: float = 60.0
_WINDOW_CHARS: int = 1500

# oEmbed endpoint for title/channel metadata (no API key).
_OEMBED_URL = "https://www.youtube.com/oembed"


def _normalize_host(host: str | None) -> str:
    """Lowercase a hostname for comparison against :data:`_YOUTUBE_HOSTS`.

    The host set already lists both the apex and ``www.`` forms, so no prefix
    stripping is needed here.
    """
    return (host or "").lower()


def is_youtube_url(url: str) -> bool:
    """Return ``True`` if ``url`` points at a YouTube host.

    Detection is host-based (``youtube.com``, ``www.youtube.com``,
    ``m.youtube.com``, ``youtu.be``); it does not require a resolvable single
    video id, so playlist and channel URLs also return ``True`` here (they are
    rejected later by :func:`fetch_youtube`).

    Args:
        url: The candidate URL.

    Returns:
        Whether the URL's host is a known YouTube host.
    """
    try:
        host = _normalize_host(urlsplit(url).hostname)
    except ValueError:
        return False
    return host in _YOUTUBE_HOSTS


def extract_video_id(url: str) -> str | None:
    """Extract the single video id from a YouTube URL, or ``None``.

    Handles ``watch?v=<id>``, ``youtu.be/<id>``, and ``shorts/<id>`` forms,
    with or without extra query params (e.g. ``&t=``, ``&list=``, ``?si=``).
    Returns ``None`` for non-YouTube hosts and for pure playlist / channel /
    handle URLs that carry no single video id.

    Args:
        url: The candidate URL.

    Returns:
        The 11-character video id, or ``None`` if the URL is not a resolvable
        single-video YouTube URL.
    """
    try:
        parsed = urlsplit(url)
    except ValueError:
        return None
    host = _normalize_host(parsed.hostname)
    if host not in _YOUTUBE_HOSTS:
        return None

    path = parsed.path or ""

    # youtu.be/<id>
    if host == "youtu.be":
        candidate = path.lstrip("/").split("/", 1)[0]
        return candidate if _VIDEO_ID_RE.match(candidate) else None

    # youtube.com/watch?v=<id>
    if path == "/watch":
        values = parse_qs(parsed.query).get("v", [])
        candidate = values[0] if values else ""
        return candidate if _VIDEO_ID_RE.match(candidate) else None

    # youtube.com/shorts/<id> or /embed/<id> or /v/<id>
    for prefix in ("/shorts/", "/embed/", "/v/"):
        if path.startswith(prefix):
            candidate = path[len(prefix) :].split("/", 1)[0]
            return candidate if _VIDEO_ID_RE.match(candidate) else None

    return None


def _load_youtube_transcript_api() -> Any:
    """Lazily import ``youtube_transcript_api``.

    Returns:
        The imported ``youtube_transcript_api`` module.

    Raises:
        KaguraIngestError: If the optional dependency is not installed, with a
            message pointing at the ``[ingest-youtube]`` extra.
    """
    try:
        import youtube_transcript_api  # type: ignore[import-not-found]
    except ImportError as e:
        raise KaguraIngestError(
            "youtube-transcript-api is not installed. Install with: "
            "pip install 'kagura-memory[ingest-youtube]'"
        ) from e
    return youtube_transcript_api


async def _fetch_oembed(
    url: str,
    *,
    connect_timeout: float = _DEFAULT_CONNECT_TIMEOUT,
    read_timeout: float,
) -> tuple[str | None, str | None]:
    """Fetch ``(title, author_name)`` from YouTube oEmbed; best-effort.

    A failure (network error, non-200, malformed JSON) returns
    ``(None, None)`` rather than raising — oEmbed metadata must never fail the
    whole ingest.

    Args:
        url: The YouTube video URL.
        connect_timeout: Per-request connect timeout in seconds (threaded from
            the caller's ingest connect timeout).
        read_timeout: Per-request read timeout in seconds.

    Returns:
        A ``(title, author_name)`` tuple; either element may be ``None``.
    """
    timeout = httpx.Timeout(connect=connect_timeout, read=read_timeout, write=10.0, pool=10.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(_OEMBED_URL, params={"url": url, "format": "json"})
            response.raise_for_status()
            data = response.json()
    except (httpx.HTTPError, ValueError):
        # ValueError covers JSON decode failures; httpx.HTTPError covers
        # network/status errors. Degrade gracefully.
        return None, None
    title = data.get("title") if isinstance(data, dict) else None
    author = data.get("author_name") if isinstance(data, dict) else None
    return (title or None), (author or None)


def _format_timestamp(seconds: float) -> str:
    """Format ``seconds`` as ``mm:ss`` (or ``h:mm:ss`` past an hour)."""
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _segment_windows(snippets: list[Any]) -> list[tuple[float, str]]:
    """Group transcript snippets into ``(start_sec, text)`` time windows.

    A new window opens once the accumulated duration exceeds
    :data:`_WINDOW_SECONDS` or the accumulated text exceeds
    :data:`_WINDOW_CHARS`, whichever comes first.

    Args:
        snippets: Caption snippets, each with ``.text`` and ``.start``.

    Returns:
        A list of ``(window_start_seconds, window_text)`` tuples.
    """
    windows: list[tuple[float, str]] = []
    window_start: float | None = None
    parts: list[str] = []
    char_count = 0

    def flush() -> None:
        nonlocal parts, char_count, window_start
        if window_start is not None and parts:
            windows.append((window_start, " ".join(parts).strip()))
        parts = []
        char_count = 0
        window_start = None

    for snippet in snippets:
        text = (getattr(snippet, "text", "") or "").strip()
        if not text:
            continue
        start = float(getattr(snippet, "start", 0.0))
        # Close the current window BEFORE adding a snippet that would push it
        # past either window boundary, so the new snippet opens a fresh window
        # anchored at its own start time. The char check looks ahead at the
        # snippet about to be added (char_count + len(text)) rather than the
        # already-accumulated count, so a window never overruns _WINDOW_CHARS by
        # a whole snippet. ``char_count > 0`` guarantees we never flush an empty
        # window, so a single oversized snippet still forms its own window.
        if window_start is not None and (
            start - window_start >= _WINDOW_SECONDS
            or (char_count > 0 and char_count + len(text) > _WINDOW_CHARS)
        ):
            flush()
        if window_start is None:
            window_start = start
        parts.append(text)
        char_count += len(text) + 1
    flush()
    return windows


def _build_markdown(title: str, author: str | None, windows: list[tuple[float, str]]) -> str:
    """Assemble the transcript Markdown document.

    The first line is ``# <title>`` (H1) so ``TextExtractor`` promotes it to
    the overview memory's title. An optional metadata line names the channel.
    Each window renders as a ``## [mm:ss]`` section.

    Args:
        title: The video title (H1).
        author: The channel / author name, or ``None``.
        windows: ``(start_seconds, text)`` time windows from
            :func:`_segment_windows`.

    Returns:
        The UTF-8 Markdown document as a ``str``, capped at
        :data:`_MAX_TOTAL_TEXT_CHARS` UTF-8 encoded bytes (truncated with a
        marker if the transcript would exceed the cap).
    """
    lines: list[str] = [f"# {title}", ""]
    if author:
        lines.append(f"Channel: {author}")
        lines.append("")
    for start, text in windows:
        lines.append(f"## [{_format_timestamp(start)}]")
        lines.append("")
        lines.append(text)
        lines.append("")
    markdown = "\n".join(lines).rstrip() + "\n"

    # Cap on encoded BYTES, not characters: TextExtractor bounds its input at
    # _MAX_TOTAL_TEXT_CHARS *bytes*, so a long multibyte (e.g. Japanese)
    # transcript capped only by character count could still overflow it. Slice
    # the UTF-8 bytes and decode with errors="ignore" to drop a partial
    # trailing multibyte sequence.
    if len(markdown.encode("utf-8")) > _MAX_TOTAL_TEXT_CHARS:
        marker = "\n\n[transcript truncated: exceeded length cap]\n"
        budget = _MAX_TOTAL_TEXT_CHARS - len(marker.encode("utf-8"))
        truncated = markdown.encode("utf-8")[: max(0, budget)].decode("utf-8", errors="ignore")
        markdown = truncated + marker
    return markdown


def _resolve_transcript(transcript_list: Any, languages: tuple[str, ...]) -> Any | None:
    """Pick a transcript: manual (preferred) → generated → first available.

    Args:
        transcript_list: A ``TranscriptList`` from ``YouTubeTranscriptApi.list``.
        languages: Preferred language codes, in descending priority.

    Returns:
        A ``Transcript`` object whose ``.fetch()`` yields snippets, or ``None``
        when the video has no transcript in any language. The caller maps
        ``None`` to a :class:`KaguraFetchError` carrying the original URL.
    """
    youtube_transcript_api = _load_youtube_transcript_api()
    no_transcript = youtube_transcript_api.NoTranscriptFound

    try:
        return transcript_list.find_manually_created_transcript(list(languages))
    except no_transcript:
        pass
    try:
        return transcript_list.find_generated_transcript(list(languages))
    except no_transcript:
        pass
    # Fall back to the first available transcript in any language.
    for transcript in transcript_list:
        return transcript
    # Nothing in any language. Signal "no transcript" to the caller rather than
    # constructing the library's NoTranscriptFound — its __init__ signature
    # varies across versions, so building one via __new__ was fragile.
    return None


def _fetch_transcript_sync(video_id: str, languages: tuple[str, ...], *, url: str) -> list[Any]:
    """Synchronously resolve and fetch a transcript's snippets.

    Runs the blocking ``youtube-transcript-api`` calls; the caller wraps this
    in :func:`asyncio.to_thread`.

    Args:
        video_id: The 11-character YouTube video id.
        languages: Preferred caption languages.
        url: The original YouTube URL. Used as the ``KaguraFetchError.url`` on
            failure so the error (and any derived ``IngestResult.source_uri``)
            surfaces the caller's URL rather than the bare video id.

    Returns:
        The list of caption snippets (each with ``.text`` / ``.start``).

    Raises:
        KaguraFetchError: For caption-disabled / unavailable / restricted
            videos, with an actionable message.
        KaguraIngestError: If the optional dependency is missing.
    """
    youtube_transcript_api = _load_youtube_transcript_api()
    api = youtube_transcript_api.YouTubeTranscriptApi()

    try:
        transcript_list = api.list(video_id)
        transcript = _resolve_transcript(transcript_list, languages)
        if transcript is None:
            raise KaguraFetchError(
                "no transcript/captions found for this video in any language",
                url=url,
            )
        snippets = list(transcript.fetch())
    except youtube_transcript_api.TranscriptsDisabled as e:
        raise KaguraFetchError(
            "captions are disabled for this video; no transcript is available",
            url=url,
        ) from e
    except youtube_transcript_api.AgeRestricted as e:
        raise KaguraFetchError(
            "video is age-restricted; its transcript cannot be retrieved",
            url=url,
        ) from e
    except (
        youtube_transcript_api.VideoUnavailable,
        youtube_transcript_api.VideoUnplayable,
        youtube_transcript_api.InvalidVideoId,
    ) as e:
        raise KaguraFetchError(
            "video is unavailable (private, removed, or region-locked)",
            url=url,
        ) from e
    except youtube_transcript_api.CouldNotRetrieveTranscript as e:
        # Catch-all for the library's other retrieval failures (IP blocked,
        # request blocked, members-only, etc.).
        raise KaguraFetchError(
            f"could not retrieve transcript for this video: {e}",
            url=url,
        ) from e
    return snippets


async def fetch_youtube(
    url: str,
    *,
    max_bytes: int,
    read_timeout: float,
    connect_timeout: float = _DEFAULT_CONNECT_TIMEOUT,
    languages: tuple[str, ...] | None = None,
) -> FetchResult:
    """Resolve a YouTube video URL to a Markdown transcript ``FetchResult``.

    Manual captions are preferred, auto-generated captions are the fallback,
    and the first available transcript in any language is the final fallback.
    The resulting Markdown carries an H1 title (so the overview memory's title
    is populated) and ``## [mm:ss]`` time-window sections.

    Args:
        url: A single-video YouTube URL (``watch?v=``, ``youtu.be/``,
            ``shorts/``). Playlist / channel URLs are rejected.
        max_bytes: Hard upper bound (in bytes) on the assembled transcript
            body, enforced for parity with the byte fetcher — an oversize
            transcript raises :class:`KaguraFetchError` rather than being
            silently truncated. The body is *also* capped at
            :data:`_MAX_TOTAL_TEXT_CHARS` bytes inside :func:`_build_markdown`
            (the ``TextExtractor`` input bound), so the effective limit is the
            smaller of the two.
        read_timeout: Per-request read timeout (seconds). Bounds both the
            blocking transcript fetch (an overall ceiling on the to-thread
            call) and the oEmbed read.
        connect_timeout: Per-request connect timeout (seconds) for the oEmbed
            call, threaded from the caller's ingest connect timeout.
        languages: Preferred caption languages, in descending priority.
            Defaults to ``("en",)``.

    Returns:
        A :class:`FetchResult` with ``content_type="text/markdown"``,
        ``source_type="url"``, and ``source_uri=url``.

    Raises:
        KaguraFetchError: For playlist/channel URLs, for caption-disabled,
            unavailable, age-restricted, or otherwise unretrievable videos, and
            when the blocking transcript request exceeds ``read_timeout``.
        KaguraIngestError: If ``youtube-transcript-api`` is not installed.
    """
    video_id = extract_video_id(url)
    if video_id is None:
        raise KaguraFetchError(
            "playlists/channels are not supported; provide a single video URL",
            url=url,
        )

    langs = languages or _DEFAULT_LANGUAGES

    # Run the transcript fetch and the (best-effort) oEmbed fetch. Thread the
    # original ``url`` through so transcript-failure errors carry it (not the
    # bare video id) as their ``.url``.
    #
    # The transcript fetch is a blocking library call run in a worker thread; if
    # YouTube stalls it can hang indefinitely. Bound it with ``read_timeout`` so
    # the ingest fails fast (the worker thread is left to unwind on its own —
    # ``asyncio.wait_for`` cannot cancel the underlying blocking call).
    try:
        snippets = await asyncio.wait_for(
            asyncio.to_thread(_fetch_transcript_sync, video_id, langs, url=url),
            timeout=read_timeout,
        )
    except TimeoutError as e:
        raise KaguraFetchError(
            f"transcript request timed out after {read_timeout}s",
            url=url,
        ) from e
    title, author = await _fetch_oembed(
        url, connect_timeout=connect_timeout, read_timeout=read_timeout
    )

    windows = _segment_windows(snippets)
    markdown = _build_markdown(title or video_id, author, windows)
    body = markdown.encode("utf-8")

    # Enforce the caller's byte budget, mirroring the byte fetcher's behavior:
    # an oversize transcript is a hard failure, not a silent truncation. The
    # _MAX_TOTAL_TEXT_CHARS cap in _build_markdown is the extractor-input bound;
    # this is the caller's explicit ingest(..., max_bytes=...) bound.
    if len(body) > max_bytes:
        raise KaguraFetchError(
            f"transcript body {len(body)} bytes exceeds max_bytes {max_bytes}",
            url=url,
        )

    return FetchResult(
        body=body,
        content_type="text/markdown",
        source_uri=url,
        source_type="url",
        final_url=url,
        bytes_read=len(body),
    )
