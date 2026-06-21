"""Shared HTTP utilities for Kagura Memory SDK clients."""

from __future__ import annotations

import re
from importlib.metadata import version as _pkg_version
from typing import Any, NoReturn

import httpx

from .exceptions import (
    KaguraAuthError,
    KaguraConnectionError,
    KaguraRateLimitError,
    _exc_message,
)

SDK_VERSION: str = _pkg_version("kagura-memory")
"""Package version string, shared across client modules."""


def base_url_from_mcp(mcp_url: str) -> str:
    """Derive REST API base URL from an MCP URL.

    Strips ``/mcp`` and everything after it (e.g. ``/mcp/w/{workspace}``).

    Args:
        mcp_url: MCP server URL (already stripped of trailing slash).

    Returns:
        Base URL suitable for REST API calls.
    """
    m = re.search(r"/mcp(?=/|$)", mcp_url)
    return mcp_url[: m.start()] if m else mcp_url


def extract_detail(response: httpx.Response) -> str:
    """Return a useful server-supplied error string from an httpx response.

    Handles three response shapes:

    - ``{"detail": "string"}`` — returned as-is (FastAPI HTTPException default).
    - ``{"detail": [{"loc": [...], "msg": "...", ...}, ...]}`` — FastAPI's
      validation-error format (RequestValidationError / Pydantic). Each entry
      is formatted as ``"<loc.path>: <msg>"`` and joined with ``"; "``. Without
      this, a 422 surfaces to the caller as a bare ``"HTTP 422"`` with no hint
      at which field failed validation.
    - Anything else — non-JSON body, non-dict body, missing ``detail``, or a
      ``detail`` of unexpected type — returns an empty string so callers can
      fall back to ``response.text`` or just print the status.
    """
    try:
        body = response.json()
    except (ValueError, UnicodeDecodeError):
        return ""
    if not isinstance(body, dict):
        return ""
    detail = body.get("detail")
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list):
        return _format_validation_errors(detail)
    return ""


def _retry_after_seconds(response: httpx.Response) -> int | None:
    """Parse a numeric ``Retry-After`` header (delta-seconds), else ``None``.

    Only the integer-seconds form is honored; an HTTP-date ``Retry-After`` (rare
    for rate limits) is treated as absent rather than mis-parsed.
    """
    raw = response.headers.get("Retry-After")
    if raw is None:
        return None
    raw = raw.strip()
    return int(raw) if raw.isdigit() else None


def raise_for_kagura_status(e: httpx.HTTPStatusError) -> NoReturn:
    """Translate an httpx ``HTTPStatusError`` into the matching Kagura error.

    Maps ``401`` → :class:`~kagura_memory.exceptions.KaguraAuthError`, ``429`` →
    :class:`~kagura_memory.exceptions.KaguraRateLimitError` (honoring a numeric
    ``Retry-After`` header), and every other status →
    :class:`~kagura_memory.exceptions.KaguraConnectionError`. The server-supplied
    ``detail`` (a FastAPI string or validation-error list) is appended when
    present — surfacing e.g. which field failed a 422 — otherwise the
    exception's own message is used so the status is never left bare. This
    function always raises.
    """
    response = e.response
    status = response.status_code
    if status == 401:
        raise KaguraAuthError("Authentication failed. Check your API key.") from e
    detail = extract_detail(response) or _exc_message(e)
    if status == 429:
        raise KaguraRateLimitError(
            f"Rate limit exceeded (HTTP 429): {detail}",
            retry_after=_retry_after_seconds(response),
        ) from e
    raise KaguraConnectionError(f"HTTP {status}: {detail}") from e


def _format_validation_errors(errors: list[Any]) -> str:
    # Silent-skip malformed entries so a single bad entry doesn't blank the line.
    parts: list[str] = []
    for entry in errors:
        if not isinstance(entry, dict):
            continue
        msg = entry.get("msg")
        if not isinstance(msg, str) or not msg:
            continue
        loc = entry.get("loc")
        if isinstance(loc, list) and loc:
            loc_path = ".".join(str(part) for part in loc)
            parts.append(f"{loc_path}: {msg}")
        else:
            parts.append(msg)
    return "; ".join(parts)


# Plain-HTTP is permitted only for genuine loopback hosts. The host token must be
# followed by a boundary — a port (``:\d+``), a path/query/fragment delimiter, or
# end-of-string — so a prefix-match attack like ``http://localhost.evil.com`` or a
# userinfo trick like ``http://localhost@evil.com`` cannot smuggle an external host
# past the check (#189). Scheme matching stays case-sensitive to preserve prior
# behavior; the trigger below is the lowercase ``http://`` literal.
_LOCALHOST_HTTP_RE = re.compile(r"^http://(?:localhost|127\.0\.0\.1|\[::1\])(?::\d+)?(?:[/?#]|$)")


def validate_https_url(url: str, *, label: str = "URL") -> None:
    """Enforce HTTPS except for localhost development.

    Args:
        url: URL to validate.
        label: Human-readable label for error messages.

    Raises:
        ValueError: If URL uses HTTP and is not a loopback host.
    """
    if url.startswith("http://") and not _LOCALHOST_HTTP_RE.match(url):
        raise ValueError(
            f"{label} must use HTTPS for security (got: {url}). "
            "HTTP is only allowed for localhost development."
        )
