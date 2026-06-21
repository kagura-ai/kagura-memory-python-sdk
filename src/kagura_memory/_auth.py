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

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from .auth.credentials import KaguraOAuth, get_shared_state, load_credentials_file
from .config import load_config
from .exceptions import KaguraAuthError

if TYPE_CHECKING:
    from .auth.credentials import _SharedCredentialsState

_DEFAULT_MCP_URL = "https://memory.kagura-ai.com/mcp"

_logger = logging.getLogger("kagura_memory")

# (credentials-file path, profile name) pairs already warned about this process,
# so the multi-profile ambiguity note (issue #203) fires at most once per active
# profile per file per process rather than on every client construction. Keyed by
# path too so two different credential files that share a default name (e.g.
# "default") don't suppress each other. Reset by :func:`reset_profile_warnings`.
_warned_profiles: set[tuple[str, str]] = set()


def reset_profile_warnings() -> None:
    """Clear the once-per-process ambiguity-warning dedup set (test hook)."""
    _warned_profiles.clear()


def _strict_profile_required() -> bool:
    """True when ``KAGURA_REQUIRE_PROFILE`` opts into strict resolution (#203)."""
    return os.getenv("KAGURA_REQUIRE_PROFILE", "").strip().lower() in {"1", "true", "yes"}


def _check_profile_ambiguity(state: _SharedCredentialsState) -> None:
    """Warn (or, under strict mode, raise) on an implicit multi-profile default.

    Only fires when resolution fell back to ``default_profile`` (no explicit
    ``profile`` arg and no ``KAGURA_PROFILE``) AND more than one profile is
    configured — a single-profile setup is unambiguous and stays silent. The
    note names the active profile + workspace so a misdirected write to the
    wrong account is caught. Honors ``KAGURA_REQUIRE_PROFILE`` (issue #203).
    """
    cf = load_credentials_file()
    if len(cf.profiles) < 2:
        return  # single profile: unambiguous

    name = state.profile_name
    creds = state.credentials
    workspace = creds.workspace_name or creds.workspace_id or "unknown workspace"

    if _strict_profile_required():
        available = ", ".join(sorted(cf.profiles))
        raise KaguraAuthError(
            f"Multiple profiles configured and none selected; refusing to use the "
            f"implicit default '{name}' because KAGURA_REQUIRE_PROFILE is set.\n"
            f"  Select one explicitly: kagura auth use <name>, --profile <name>, "
            f"or KAGURA_PROFILE=<name>\n"
            f"  Available profiles: {available}"
        )

    warn_key = (str(state.path), name)
    if warn_key not in _warned_profiles:
        _warned_profiles.add(warn_key)
        _logger.warning(
            "kagura: using profile '%s' (workspace '%s') — set a default with "
            "'kagura auth use <name>' or pass --profile to silence this notice.",
            name,
            workspace,
        )


# Which precedence branch produced a ``_StaticAuth``. CLI-layer code uses
# this to resolve ``workspace_id`` from the same source as ``api_key`` —
# api_key and workspace_id form an inseparable pair (an api_key is
# provisioned for one specific workspace), so cross-source mixing is the
# class of bug we are preventing (see issue #115).
_StaticSource = Literal["explicit", "env", "config"]

# Superset of :data:`_StaticSource` that also covers the ``_OAuthAuth``
# branch. Used by client code (e.g. :class:`FilesClient`) to remember
# which branch built the client, so a 403 can surface an actionable
# "credential source X expects workspace Y" hint.
_AuthSource = Literal["explicit", "env", "config", "oauth"]

# Human-readable label for each credential source — single source of
# truth shared by the CLI's `_resolve_workspace_from_source` error
# messages and the SDK's 403 hint formatter. Keeping one mapping
# prevents the two surfaces from drifting and surfacing inconsistent
# names for the same source ("KAGURA_API_KEY env" vs "env variable").
_SOURCE_LABEL: dict[_AuthSource, str] = {
    "explicit": "explicit api_key argument",
    "env": "KAGURA_API_KEY env",
    "config": ".kagura.json",
    "oauth": "OAuth profile (~/.kagura/credentials.json)",
}


@dataclass
class _StaticAuth:
    """Long-lived API key resolution result.

    ``source`` records which branch of :func:`_resolve_auth`'s precedence
    chain produced this result. The CLI layer reads it to pick the
    matching ``workspace_id`` source (e.g. ``source="config"`` ⇒
    ``workspace_id`` must come from the same ``.kagura.json``).
    """

    api_key: str
    mcp_url: str
    source: _StaticSource


@dataclass
class _OAuthAuth:
    """OAuth credentials.json resolution result.

    ``workspace_id`` is a snapshot from the OAuth profile at resolution
    time (i.e. what ``kagura auth login`` last stored). It is the
    workspace bound to this api_key/token pair — the CLI layer uses it
    directly so it cannot accidentally mix with an api_key from a
    different source (issue #115).
    """

    oauth: KaguraOAuth
    mcp_url: str
    workspace_id: str | None


def _resolve_auth(
    *,
    api_key: str | None,
    mcp_url: str | None,
    profile: str | None,
    config: dict[str, Any] | None = None,
) -> _StaticAuth | _OAuthAuth:
    """Pick a credential source per the documented precedence chain.

    See :meth:`KaguraClient.__init__` for the full order. Raises
    :class:`KaguraAuthError` when no source produces credentials.

    ``config`` is an optional pre-loaded ``.kagura.json`` dict. When
    provided, the priority-4 fallback uses it directly instead of
    re-invoking :func:`load_config`, so callers that have already
    loaded the config (e.g. CLI commands needing ``context_id`` for
    workspace pairing) avoid a redundant disk read. When ``None``,
    :func:`load_config` is called as before.
    """
    # 1. Explicit constructor argument wins absolutely.
    # Empty / whitespace-only api_key is treated the same as None — sending
    # `Authorization: Bearer ` would always 401 and is never what the caller
    # intended; fall through to the env / OAuth / .kagura.json chain instead.
    if api_key is not None and api_key.strip():
        return _StaticAuth(api_key=api_key, mcp_url=mcp_url or _DEFAULT_MCP_URL, source="explicit")

    # 2. KAGURA_API_KEY env var (highest auto-resolution priority).
    # Strip-check mirrors the explicit-arg path above: a whitespace-only
    # env var would otherwise send `Authorization: Bearer ` and 401.
    env_key = os.getenv("KAGURA_API_KEY")
    if env_key and env_key.strip():
        return _StaticAuth(
            api_key=env_key,
            mcp_url=mcp_url or os.getenv("KAGURA_MCP_URL") or _DEFAULT_MCP_URL,
            source="env",
        )

    # 3. OAuth profile from credentials.json: explicit > KAGURA_PROFILE > default.
    target_profile = profile or os.getenv("KAGURA_PROFILE")
    state = get_shared_state(profile=target_profile)
    if state is not None:
        if target_profile is None:
            # No explicit selection — resolution fell back to default_profile.
            # Warn (or hard-error under strict mode) when the default is
            # ambiguous because multiple profiles are configured (issue #203).
            _check_profile_ambiguity(state)
        return _OAuthAuth(
            oauth=KaguraOAuth(state),
            mcp_url=mcp_url or state.credentials.mcp_url,
            workspace_id=state.credentials.workspace_id,
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
    cfg = config if config is not None else load_config()
    cfg_key = cfg.get("api_key", "")
    if cfg_key and cfg_key.strip():
        return _StaticAuth(
            api_key=cfg_key,
            mcp_url=mcp_url or cfg.get("mcp_url") or _DEFAULT_MCP_URL,
            source="config",
        )

    raise KaguraAuthError(
        "No credentials found.\n"
        "  Run: kagura auth login\n"
        "  Or set: KAGURA_API_KEY=<your key>\n"
        '  Or create: .kagura.json with {"api_key": "..."}'
    )
