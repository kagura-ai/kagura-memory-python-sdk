"""Persistent OAuth credentials for the Kagura SDK.

This module owns three responsibilities:

1. The :class:`OAuthCredentials` / :class:`CredentialsFile` dataclasses
   and their JSON (de)serialization.
2. Atomic, mode ``0600`` read/write of ``~/.kagura/credentials.json``
   with mode ``0700`` parent directory — the first place in this
   SDK that introduces atomic writes and explicit ``chmod`` enforcement.
3. A process-wide cache (:func:`get_shared_state`) so concurrent
   :class:`KaguraClient` instances pointing at the same credentials
   file share a single :class:`asyncio.Lock`, ensuring only one
   ``/oauth2/token`` refresh fires per cycle even when many client
   instances run side by side.

Multi-process coordination (e.g. two ``KaguraClient`` instances in
different worker processes refreshing the same file) is out of scope
for this module — the in-process lock collapses to a no-op across
process boundaries. The follow-up proxy daemon (issue #101) becomes
the single owner of the credentials file in that scenario.
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
import tempfile
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


DEFAULT_CREDENTIALS_PATH: Path = Path.home() / ".kagura" / "credentials.json"
"""Default location of the credentials file. Tests pass an explicit
``path`` argument instead of patching ``Path.home()``."""

REFRESH_SKEW_SEC: int = 300
"""Refresh ``access_token`` when within this many seconds of ``expires_at``.

Default of 5 minutes mirrors common OAuth2 client conventions and gives
enough headroom for a slow refresh round-trip."""

CREDENTIALS_FILE_MODE: int = 0o600
CREDENTIALS_DIR_MODE: int = 0o700


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class OAuthCredentials:
    """A single profile's OAuth2 credentials.

    Attributes are mutable so refresh can rotate ``access_token``,
    ``expires_at``, and ``scope`` in place — callers should mutate
    via :meth:`with_refreshed` to keep the change explicit.
    """

    server: str
    mcp_url: str
    client_id: str
    access_token: str
    refresh_token: str
    token_type: str
    expires_at: datetime
    scope: str
    workspace_id: str
    workspace_name: str
    user_email: str
    issued_at: datetime

    def is_expired(self, skew_seconds: int = 0) -> bool:
        """True when ``access_token`` is past ``expires_at - skew_seconds``."""
        now = datetime.now(UTC)
        threshold = self.expires_at.astimezone(UTC)
        if skew_seconds:
            from datetime import timedelta

            threshold = threshold - timedelta(seconds=skew_seconds)
        return now >= threshold

    def with_refreshed(
        self,
        *,
        access_token: str,
        refresh_token: str | None = None,
        expires_at: datetime,
        scope: str | None = None,
    ) -> OAuthCredentials:
        """Return a copy with the rotated token fields applied.

        ``refresh_token`` is rotated when the server returns a new one
        (RFC 6749 §10.4 recommended). ``scope`` is updated when the
        server narrows or widens the grant (incremental consent).
        """
        return replace(
            self,
            access_token=access_token,
            refresh_token=refresh_token if refresh_token is not None else self.refresh_token,
            expires_at=expires_at,
            scope=scope if scope is not None else self.scope,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "server": self.server,
            "mcp_url": self.mcp_url,
            "client_id": self.client_id,
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "token_type": self.token_type,
            "expires_at": _iso_utc(self.expires_at),
            "scope": self.scope,
            "workspace_id": self.workspace_id,
            "workspace_name": self.workspace_name,
            "user_email": self.user_email,
            "issued_at": _iso_utc(self.issued_at),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> OAuthCredentials:
        return cls(
            server=d["server"],
            mcp_url=d["mcp_url"],
            client_id=d["client_id"],
            access_token=d["access_token"],
            refresh_token=d["refresh_token"],
            token_type=d.get("token_type", "Bearer"),
            expires_at=_parse_iso(d["expires_at"]),
            scope=d.get("scope", ""),
            workspace_id=d.get("workspace_id", ""),
            workspace_name=d.get("workspace_name", ""),
            user_email=d.get("user_email", ""),
            issued_at=_parse_iso(d["issued_at"]) if d.get("issued_at") else datetime.now(UTC),
        )


@dataclass
class CredentialsFile:
    """Top-level wrapper for ``~/.kagura/credentials.json``."""

    version: int = 1
    default_profile: str = "default"
    profiles: dict[str, OAuthCredentials] = field(default_factory=dict)

    def get_profile(self, name: str | None = None) -> OAuthCredentials | None:
        """Return the named profile, or the default, or ``None`` if missing."""
        key = name or self.default_profile
        return self.profiles.get(key)

    def set_profile(self, name: str, creds: OAuthCredentials) -> None:
        """Insert or replace a profile. The first profile becomes default."""
        self.profiles[name] = creds
        if self.default_profile not in self.profiles:
            self.default_profile = name

    def delete_profile(self, name: str) -> None:
        """Remove a profile if present. Does not raise on missing."""
        self.profiles.pop(name, None)
        if self.default_profile == name and self.profiles:
            # Pick any remaining profile as the new default.
            self.default_profile = next(iter(self.profiles))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "default_profile": self.default_profile,
            "profiles": {name: creds.to_dict() for name, creds in self.profiles.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CredentialsFile:
        profiles_raw = d.get("profiles", {})
        profiles = {name: OAuthCredentials.from_dict(p) for name, p in profiles_raw.items()}
        return cls(
            version=d.get("version", 1),
            default_profile=d.get("default_profile", "default"),
            profiles=profiles,
        )


# ---------------------------------------------------------------------------
# datetime helpers (Z-suffixed UTC ISO-8601 for human readability)
# ---------------------------------------------------------------------------


def _iso_utc(dt: datetime) -> str:
    """Serialize a datetime as ``YYYY-MM-DDTHH:MM:SSZ`` (UTC)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(s: str) -> datetime:
    """Parse ISO-8601, accepting both ``...Z`` and ``...+00:00`` suffixes."""
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


# ---------------------------------------------------------------------------
# Filesystem IO
# ---------------------------------------------------------------------------


def _atomic_write_json(path: Path, data: dict[str, Any], mode: int = CREDENTIALS_FILE_MODE) -> None:
    """Write ``data`` to ``path`` atomically.

    Steps: ``mkstemp`` in the same directory → write JSON → ``fsync`` →
    ``chmod`` → ``os.replace``. A kill between any two steps either
    leaves the original file untouched or replaces it cleanly.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    _enforce_dir_perms(path.parent)

    fd, tmp_str = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    tmp = Path(tmp_str)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def _enforce_dir_perms(directory: Path) -> None:
    """Force the directory to mode 0700 even if umask filtered it."""
    try:
        current = stat.S_IMODE(directory.stat().st_mode)
        if current != CREDENTIALS_DIR_MODE:
            os.chmod(directory, CREDENTIALS_DIR_MODE)
    except OSError:
        # Best-effort: a directory we can't chmod is the user's problem,
        # not ours to abort over. The atomic write will fail downstream
        # with a clearer error if perms genuinely block access.
        pass


def _enforce_file_perms(path: Path) -> None:
    """Force the credentials file to mode 0600 (defense in depth)."""
    try:
        current = stat.S_IMODE(path.stat().st_mode)
        if current != CREDENTIALS_FILE_MODE:
            os.chmod(path, CREDENTIALS_FILE_MODE)
    except OSError:
        # Best-effort: a file we cannot chmod (read-only FS, foreign
        # ownership) is the user's problem to fix, not ours to abort
        # over. Downstream reads will fail with a clearer error if
        # the perms genuinely block access.
        pass


def _resolve_path(path: Path | None) -> Path:
    """Return ``path`` or the current value of :data:`DEFAULT_CREDENTIALS_PATH`.

    Functions in this module take ``path: Path | None = None`` rather than
    ``path = DEFAULT_CREDENTIALS_PATH`` because the default value would be
    bound once at function-definition time, making it impossible to
    monkeypatch :data:`DEFAULT_CREDENTIALS_PATH` from tests. Resolving
    inside the function preserves test patchability while keeping the
    one-arg call sites concise.
    """
    return path if path is not None else DEFAULT_CREDENTIALS_PATH


def load_credentials_file(path: Path | None = None) -> CredentialsFile:
    """Read and parse the credentials file.

    Returns an empty :class:`CredentialsFile` when the file is missing,
    unreadable, or malformed — the caller can distinguish "no profile"
    via ``cf.get_profile() is None``. On a present file, parent
    directory and file permissions are coerced to ``0700``/``0600``.
    """
    path = _resolve_path(path)
    if not path.exists():
        return CredentialsFile()

    _enforce_dir_perms(path.parent)
    _enforce_file_perms(path)

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return CredentialsFile()

    if not isinstance(data, dict):
        return CredentialsFile()

    try:
        return CredentialsFile.from_dict(data)
    except (KeyError, ValueError, TypeError):
        # Corrupt or partial file — treat as empty rather than crashing.
        return CredentialsFile()


def save_credentials_file(cf: CredentialsFile, path: Path | None = None) -> None:
    """Atomically persist a :class:`CredentialsFile` to disk."""
    _atomic_write_json(_resolve_path(path), cf.to_dict(), mode=CREDENTIALS_FILE_MODE)


def update_profile(
    profile_name: str,
    creds: OAuthCredentials,
    path: Path | None = None,
) -> None:
    """Load, mutate one profile, and atomically save back.

    The whole-file read-modify-write is intentionally simple — no file
    locking. The in-process :class:`asyncio.Lock` held in
    :class:`_SharedCredentialsState` serializes refreshes within a
    single process; multi-process coordination is out of scope here.
    """
    path = _resolve_path(path)
    cf = load_credentials_file(path)
    cf.set_profile(profile_name, creds)
    save_credentials_file(cf, path)


def delete_profile(
    profile_name: str,
    path: Path | None = None,
) -> None:
    """Remove a profile and save. No-op if the profile is absent."""
    path = _resolve_path(path)
    cf = load_credentials_file(path)
    if profile_name not in cf.profiles:
        return
    cf.delete_profile(profile_name)
    save_credentials_file(cf, path)


def delete_credentials_file(path: Path | None = None) -> None:
    """Remove the entire credentials file (``kagura auth logout --all``).

    Missing file is treated as success — ``logout --all`` is idempotent.
    """
    path = _resolve_path(path)
    try:
        path.unlink()
    except FileNotFoundError:
        # File already absent — logout is idempotent.
        return


# ---------------------------------------------------------------------------
# Shared state cache (in-process single-flight)
# ---------------------------------------------------------------------------


@dataclass
class _SharedCredentialsState:
    """Mutable per-(path, profile) state shared across KaguraClients."""

    credentials: OAuthCredentials
    profile_name: str
    path: Path
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


_state_cache: dict[tuple[Path, str], _SharedCredentialsState] = {}


def get_shared_state(
    path: Path | None = None,
    profile: str | None = None,
) -> _SharedCredentialsState | None:
    """Return the shared state for ``(path, profile)``, loading lazily.

    Returns ``None`` if the credentials file is missing or has no
    matching profile — the caller falls back to other credential
    sources (static API key, ``.kagura.json``, etc.) in that case.

    Concurrent :class:`KaguraClient` instances that resolve to the
    same ``(path, profile)`` pair get the same state object, which
    means they share a single :class:`asyncio.Lock` and any refresh
    fired by one of them benefits all of them.
    """
    path = _resolve_path(path)
    cf = load_credentials_file(path)
    creds = cf.get_profile(profile)
    if creds is None:
        return None

    profile_name = profile or cf.default_profile
    key = (path.resolve(), profile_name)
    cached = _state_cache.get(key)
    if cached is None:
        cached = _SharedCredentialsState(
            credentials=creds,
            profile_name=profile_name,
            path=path,
        )
        _state_cache[key] = cached
    else:
        # Refresh in-memory creds from disk so a recent CLI login is
        # visible to clients already constructed in this process.
        cached.credentials = creds
    return cached


def reset_state_cache() -> None:
    """Clear the module-level cache. Tests call this between cases."""
    _state_cache.clear()


# ---------------------------------------------------------------------------
# httpx.Auth subclass
# ---------------------------------------------------------------------------


class KaguraOAuth(httpx.Auth):
    """``httpx.Auth`` subclass that injects the current access_token.

    Holds a reference to a shared :class:`_SharedCredentialsState`, so
    concurrent ``KaguraClient`` instances pointing at the same
    credentials file all coalesce refresh calls through a single
    :class:`asyncio.Lock`. When the access token is within
    :data:`REFRESH_SKEW_SEC` of expiry the lock is acquired, the
    refresh runs once, and every other in-flight request waits and
    then sees the rotated token.
    """

    requires_request_body = False
    requires_response_body = False

    def __init__(self, state: _SharedCredentialsState):
        self._state = state

    async def async_auth_flow(
        self, request: httpx.Request
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        await self._maybe_refresh()
        request.headers["Authorization"] = f"Bearer {self._state.credentials.access_token}"
        yield request

    async def _maybe_refresh(self) -> None:
        """Refresh the access_token if it's within the skew window.

        The lock is held across the entire read-check-refresh-persist
        sequence so a second caller waiting on the lock will find the
        token already rotated when it acquires.
        """
        async with self._state.lock:
            if not self._state.credentials.is_expired(skew_seconds=REFRESH_SKEW_SEC):
                return

            # Lazy import: ``device_flow`` may depend on this module's
            # public types, so resolve only at refresh time to avoid a
            # cycle at import time.
            from .device_flow import make_oauth_client, refresh_access_token

            async with make_oauth_client() as client:
                token = await refresh_access_token(
                    client,
                    server=self._state.credentials.server,
                    client_id=self._state.credentials.client_id,
                    refresh_token=self._state.credentials.refresh_token,
                )
            # Pass ``None`` when the server omits ``refresh_token`` so the
            # stored token is preserved. ``TokenResponse.refresh_token``
            # defaults to ``""`` when absent, and passing the empty string
            # to ``with_refreshed`` would overwrite a valid stored token
            # with an empty one — breaking every subsequent refresh.
            self._state.credentials = self._state.credentials.with_refreshed(
                access_token=token.access_token,
                refresh_token=token.refresh_token or None,
                expires_at=token.expires_at,
                scope=token.scope,
            )
            update_profile(self._state.profile_name, self._state.credentials, self._state.path)
