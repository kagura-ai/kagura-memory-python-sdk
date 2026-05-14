"""Credential resolution shared by REST/MCP clients in this SDK.

Centralizes the precedence chain — explicit arg > ``KAGURA_API_KEY`` env >
OAuth profile in ``~/.kagura/credentials.json`` > ``.kagura.json`` config —
in one module so that :class:`KaguraClient` (MCP), :class:`FilesClient`
(REST + R2 PUT), and future REST clients (e.g. ``ResourceClient``) all
resolve credentials identically without import-direction inversion.

Returns one of two result types:

- :class:`_StaticAuth` — long-lived API key, baked into the client's
  ``Authorization`` header at construction.
- :class:`_OAuthAuth` — :class:`auth.credentials.KaguraOAuth` httpx.Auth
  subclass, injects a fresh access_token per request and coordinates
  refresh via a process-wide :class:`asyncio.Lock`.

This module imports only from ``auth.credentials``, ``config``, and
``exceptions`` — never from ``_http``. The reverse direction (``_http``
importing ``_auth``) must also be avoided so the dependency stays
one-way.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .auth.credentials import KaguraOAuth, get_shared_state
from .config import load_config
from .exceptions import KaguraAuthError

_DEFAULT_MCP_URL = "https://memory.kagura-ai.com/mcp"


@dataclass
class _StaticAuth:
    """Long-lived API key resolution result."""

    api_key: str
    mcp_url: str


@dataclass
class _OAuthAuth:
    """OAuth credentials.json resolution result."""

    oauth: KaguraOAuth
    mcp_url: str


def _resolve_auth(
    *,
    api_key: str | None,
    mcp_url: str | None,
    profile: str | None,
) -> _StaticAuth | _OAuthAuth:
    """Pick a credential source per the documented precedence chain.

    See :meth:`KaguraClient.__init__` for the full order. Raises
    :class:`KaguraAuthError` when no source produces credentials.
    """
    # 1. Explicit constructor argument wins absolutely.
    # Empty / whitespace-only api_key is treated the same as None — sending
    # `Authorization: Bearer ` would always 401 and is never what the caller
    # intended; fall through to the env / OAuth / .kagura.json chain instead.
    if api_key is not None and api_key.strip():
        return _StaticAuth(api_key=api_key, mcp_url=mcp_url or _DEFAULT_MCP_URL)

    # 2. KAGURA_API_KEY env var (highest auto-resolution priority).
    # Strip-check mirrors the explicit-arg path above: a whitespace-only
    # env var would otherwise send `Authorization: Bearer ` and 401.
    env_key = os.getenv("KAGURA_API_KEY")
    if env_key and env_key.strip():
        return _StaticAuth(
            api_key=env_key,
            mcp_url=mcp_url or os.getenv("KAGURA_MCP_URL") or _DEFAULT_MCP_URL,
        )

    # 3. OAuth profile from credentials.json: explicit > KAGURA_PROFILE > default.
    target_profile = profile or os.getenv("KAGURA_PROFILE")
    state = get_shared_state(profile=target_profile)
    if state is not None:
        return _OAuthAuth(
            oauth=KaguraOAuth(state),
            mcp_url=mcp_url or state.credentials.mcp_url,
        )
    if target_profile:
        # An explicit profile name (via arg or KAGURA_PROFILE env) was
        # requested but credentials.json has no such profile. Falling
        # through to .kagura.json would silently authenticate with the
        # wrong account, so raise instead.
        source = "profile argument" if profile else "KAGURA_PROFILE env"
        raise KaguraAuthError(
            f"Profile '{target_profile}' (from {source}) not found in credentials.json.\n"
            f"  Run: kagura auth login --profile {target_profile}\n"
            f"  Or inspect ~/.kagura/credentials.json to see which profiles exist."
        )

    # 4. Legacy .kagura.json (which itself env-falls-back internally).
    # Apply the same whitespace-only strip check used for the explicit
    # arg and KAGURA_API_KEY env paths so a stray-whitespace config file
    # doesn't produce a guaranteed-401 `Authorization: Bearer ` header.
    cfg = load_config()
    cfg_key = cfg.get("api_key", "")
    if cfg_key and cfg_key.strip():
        return _StaticAuth(
            api_key=cfg_key,
            mcp_url=mcp_url or cfg.get("mcp_url") or _DEFAULT_MCP_URL,
        )

    raise KaguraAuthError(
        "No credentials found.\n"
        "  Run: kagura auth login\n"
        "  Or set: KAGURA_API_KEY=<your key>\n"
        '  Or create: .kagura.json with {"api_key": "..."}'
    )
