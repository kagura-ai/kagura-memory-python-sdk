"""Shared HTTP utilities for Kagura Memory SDK clients."""

from __future__ import annotations

from importlib.metadata import version as _pkg_version

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
    idx = mcp_url.find("/mcp")
    return mcp_url[:idx] if idx != -1 else mcp_url


def validate_https_url(url: str, *, label: str = "URL") -> None:
    """Enforce HTTPS except for localhost development.

    Args:
        url: URL to validate.
        label: Human-readable label for error messages.

    Raises:
        ValueError: If URL uses HTTP and is not localhost.
    """
    if url.startswith("http://") and not any(
        url.startswith(f"http://{h}") for h in ("localhost", "127.0.0.1", "[::1]")
    ):
        raise ValueError(
            f"{label} must use HTTPS for security (got: {url}). "
            "HTTP is only allowed for localhost development."
        )
