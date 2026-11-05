"""Tests for kagura_memory.auth.credentials."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from kagura_memory.auth.credentials import (
    CREDENTIALS_DIR_MODE,
    CREDENTIALS_FILE_MODE,
    REFRESH_SKEW_SEC,
    CredentialsFile,
    KaguraOAuth,
    OAuthCredentials,
    _SharedCredentialsState,
    delete_credentials_file,
    delete_profile,
    get_shared_state,
    load_credentials_file,
    refresh_profile_locked,
    reset_state_cache,
    save_credentials_file,
    update_profile,
)
from kagura_memory.auth.device_flow import TokenResponse
from kagura_memory.exceptions import KaguraAuthError, KaguraAuthExpiredError

# Gate POSIX-specific behaviour to non-Windows (CI runs Linux):
#   - the cross-process refresh-dedup tests rely on fcntl serializing two
#     independent open descriptions of the same lock file (the Windows msvcrt
#     backend is unit-tested in test_filelock.py); and
#   - the 0600/0700 file-mode tests assert POSIX permission bits, which
#     os.chmod cannot set on Windows (NTFS uses ACLs).
_POSIX_ONLY_CREDS = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX-only: fcntl serialization / chmod permission bits"
)


def _sample_creds(
    *,
    access_token: str = "atok-1",
    refresh_token: str = "rtok-1",
    expires_at: datetime | None = None,
    scope: str = "memory:read",
) -> OAuthCredentials:
    return OAuthCredentials(
        server="https://test.example.com",
        mcp_url="https://test.example.com/mcp",
        client_id="kagura-cli",
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="Bearer",
        expires_at=expires_at or (datetime.now(UTC) + timedelta(hours=1)),
        scope=scope,
        workspace_id="ws-1",
        workspace_name="test-ws",
        user_email="test@example.com",
        issued_at=datetime.now(UTC),
    )


# ---------------------------------------------------------------------------
# OAuthCredentials
# ---------------------------------------------------------------------------


def test_oauth_credentials_is_expired_false_when_far_future():
    creds = _sample_creds(expires_at=datetime.now(UTC) + timedelta(hours=1))
    assert creds.is_expired() is False
    assert creds.is_expired(skew_seconds=300) is False


def test_oauth_credentials_is_expired_true_when_past():
    creds = _sample_creds(expires_at=datetime.now(UTC) - timedelta(seconds=10))
    assert creds.is_expired() is True


def test_oauth_credentials_is_expired_with_skew():
    """Token expiring in 2 minutes should be 'expired' with 5-minute skew."""
    creds = _sample_creds(expires_at=datetime.now(UTC) + timedelta(minutes=2))
    assert creds.is_expired() is False
    assert creds.is_expired(skew_seconds=REFRESH_SKEW_SEC) is True


def test_oauth_credentials_with_refreshed_rotates_token_only():
    creds = _sample_creds()
    new_expires = datetime.now(UTC) + timedelta(hours=2)
    rotated = creds.with_refreshed(
        access_token="atok-2",
        expires_at=new_expires,
    )
    assert rotated.access_token == "atok-2"
    assert rotated.refresh_token == creds.refresh_token  # preserved
    assert rotated.expires_at == new_expires
    assert rotated.scope == creds.scope
    # Original is unchanged (replace returns new instance).
    assert creds.access_token == "atok-1"


def test_oauth_credentials_with_refreshed_rotates_refresh_token_and_scope():
    creds = _sample_creds()
    rotated = creds.with_refreshed(
        access_token="atok-2",
        refresh_token="rtok-2",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        scope="memory:read memory:write",
    )
    assert rotated.refresh_token == "rtok-2"
    assert rotated.scope == "memory:read memory:write"


def test_oauth_credentials_round_trip():
    creds = _sample_creds()
    restored = OAuthCredentials.from_dict(creds.to_dict())
    assert restored.access_token == creds.access_token
    assert restored.refresh_token == creds.refresh_token
    assert restored.scope == creds.scope
    # Datetime round-trip through ISO-8601 with Z suffix.
    assert restored.expires_at == creds.expires_at.replace(microsecond=0)


# ---------------------------------------------------------------------------
# CredentialsFile
# ---------------------------------------------------------------------------


def test_credentials_file_get_profile_default():
    cf = CredentialsFile(default_profile="alice", profiles={"alice": _sample_creds()})
    assert cf.get_profile() is not None
    assert cf.get_profile().scope == "memory:read"


def test_credentials_file_get_profile_named():
    creds = _sample_creds()
    cf = CredentialsFile(profiles={"work": creds})
    assert cf.get_profile("work") is creds
    assert cf.get_profile("missing") is None


def test_credentials_file_set_profile_first_becomes_default():
    cf = CredentialsFile()
    cf.set_profile("first", _sample_creds())
    assert cf.default_profile == "first"


def test_credentials_file_delete_profile_updates_default():
    cf = CredentialsFile()
    cf.set_profile("a", _sample_creds())
    cf.set_profile("b", _sample_creds())
    assert cf.default_profile == "a"
    cf.delete_profile("a")
    assert "a" not in cf.profiles
    assert cf.default_profile == "b"


def test_credentials_file_round_trip():
    cf = CredentialsFile()
    cf.set_profile("work", _sample_creds(access_token="atok-work"))
    cf.set_profile("home", _sample_creds(access_token="atok-home"))
    restored = CredentialsFile.from_dict(cf.to_dict())
    assert set(restored.profiles.keys()) == {"work", "home"}
    assert restored.profiles["work"].access_token == "atok-work"


# ---------------------------------------------------------------------------
# Filesystem IO (atomic write + perms)
# ---------------------------------------------------------------------------


def test_save_and_load_round_trip(tmp_path: Path):
    path = tmp_path / ".kagura" / "credentials.json"
    cf = CredentialsFile()
    cf.set_profile("default", _sample_creds())
    save_credentials_file(cf, path)
    restored = load_credentials_file(path)
    assert restored.default_profile == "default"
    assert restored.get_profile().access_token == "atok-1"


@_POSIX_ONLY_CREDS
def test_save_enforces_file_mode_0600(tmp_path: Path):
    path = tmp_path / "creds.json"
    cf = CredentialsFile()
    cf.set_profile("default", _sample_creds())
    save_credentials_file(cf, path)
    assert stat.S_IMODE(path.stat().st_mode) == CREDENTIALS_FILE_MODE


@_POSIX_ONLY_CREDS
def test_save_enforces_dir_mode_0700(tmp_path: Path):
    path = tmp_path / "subdir" / "creds.json"
    cf = CredentialsFile()
    cf.set_profile("default", _sample_creds())
    save_credentials_file(cf, path)
    assert stat.S_IMODE(path.parent.stat().st_mode) == CREDENTIALS_DIR_MODE


def test_load_returns_empty_when_missing(tmp_path: Path):
    path = tmp_path / "nope.json"
    cf = load_credentials_file(path)
    assert cf.profiles == {}
    assert cf.get_profile() is None


def test_load_returns_empty_on_corrupt_file(tmp_path: Path):
    path = tmp_path / "creds.json"
    path.write_text("not json {{{")
    cf = load_credentials_file(path)
    assert cf.profiles == {}


def test_load_returns_empty_on_non_dict_json(tmp_path: Path):
    """Top-level JSON array / scalar must not crash."""
    path = tmp_path / "creds.json"
    path.write_text('["this is", "an array"]')
    cf = load_credentials_file(path)
    assert cf.profiles == {}


def test_load_returns_empty_on_partial_profile(tmp_path: Path):
    """A profile missing required fields falls through to empty CredentialsFile."""
    import json as _json

    path = tmp_path / "creds.json"
    path.write_text(
        _json.dumps(
            {
                "version": 1,
                "default_profile": "broken",
                "profiles": {
                    "broken": {"only_has_one_field": "yes"},  # missing access_token etc.
                },
            }
        )
    )
    cf = load_credentials_file(path)
    assert cf.profiles == {}


def test_delete_profile_no_op_when_missing(tmp_path: Path):
    """Calling delete_profile on a name that isn't in the file is silent."""
    from kagura_memory.auth.credentials import delete_profile

    path = tmp_path / "creds.json"
    cf = CredentialsFile()
    cf.set_profile("alice", _sample_creds())
    save_credentials_file(cf, path)

    delete_profile("does-not-exist", path)
    restored = load_credentials_file(path)
    assert "alice" in restored.profiles  # untouched


@_POSIX_ONLY_CREDS
def test_load_fixes_loose_file_perms(tmp_path: Path):
    """File written 0644 should be coerced back to 0600 on read."""
    path = tmp_path / "creds.json"
    path.write_text(json.dumps({"version": 1, "default_profile": "x", "profiles": {}}))
    os.chmod(path, 0o644)
    load_credentials_file(path)
    assert stat.S_IMODE(path.stat().st_mode) == CREDENTIALS_FILE_MODE


def test_atomic_write_does_not_leave_tmp_on_failure(tmp_path: Path):
    """When os.replace raises, no .tmp files should be left behind."""
    path = tmp_path / "creds.json"
    cf = CredentialsFile()
    cf.set_profile("default", _sample_creds())
    with patch(
        "kagura_memory.auth.credentials.os.replace",
        side_effect=OSError("disk full"),
    ):
        with pytest.raises(OSError):
            save_credentials_file(cf, path)
    leftover = list(tmp_path.glob(".creds.json.*.tmp"))
    assert leftover == [], f"unexpected temp files: {leftover}"


def test_update_profile_inserts_into_existing_file(tmp_path: Path):
    path = tmp_path / "creds.json"
    cf = CredentialsFile()
    cf.set_profile("a", _sample_creds(access_token="atok-a"))
    save_credentials_file(cf, path)

    update_profile("b", _sample_creds(access_token="atok-b"), path)
    restored = load_credentials_file(path)
    assert set(restored.profiles.keys()) == {"a", "b"}


def test_delete_profile_removes_from_file(tmp_path: Path):
    path = tmp_path / "creds.json"
    cf = CredentialsFile()
    cf.set_profile("a", _sample_creds())
    cf.set_profile("b", _sample_creds())
    save_credentials_file(cf, path)

    delete_profile("a", path)
    restored = load_credentials_file(path)
    assert "a" not in restored.profiles
    assert "b" in restored.profiles


def test_delete_credentials_file_removes_file(tmp_path: Path):
    path = tmp_path / "creds.json"
    cf = CredentialsFile()
    cf.set_profile("default", _sample_creds())
    save_credentials_file(cf, path)

    delete_credentials_file(path)
    assert not path.exists()
    # Calling again on missing file is a no-op.
    delete_credentials_file(path)


# ---------------------------------------------------------------------------
# Module-level cache (_SharedCredentialsState)
# ---------------------------------------------------------------------------


def test_get_shared_state_returns_none_when_no_credentials(tmp_path: Path):
    reset_state_cache()
    assert get_shared_state(tmp_path / "creds.json") is None


def test_get_shared_state_returns_same_object_for_same_key(tmp_path: Path):
    """Two calls with the same (path, profile) return the same state object."""
    reset_state_cache()
    path = tmp_path / "creds.json"
    cf = CredentialsFile()
    cf.set_profile("default", _sample_creds())
    save_credentials_file(cf, path)

    a = get_shared_state(path)
    b = get_shared_state(path)
    assert a is b


def test_get_shared_state_profiles_share_one_lock_per_file(tmp_path: Path):
    """Two profiles in the same file must share one asyncio.Lock to prevent
    cross-profile lost-update during concurrent refresh."""
    reset_state_cache()
    path = tmp_path / "creds.json"
    cf = CredentialsFile()
    cf.set_profile("a", _sample_creds(access_token="a"))
    cf.set_profile("b", _sample_creds(access_token="b"))
    save_credentials_file(cf, path)

    state_a = get_shared_state(path, profile="a")
    state_b = get_shared_state(path, profile="b")
    assert state_a is not state_b
    assert state_a.lock is state_b.lock  # shared file-level lock


def test_parse_iso_handles_naive_datetime(tmp_path: Path):
    """Naive timestamps (no tz suffix) must be normalized to UTC."""
    import json as _json

    path = tmp_path / "creds.json"
    creds_dict = _sample_creds().to_dict()
    # Strip the trailing Z so the timestamp is naive.
    creds_dict["expires_at"] = "2027-01-01T00:00:00"
    creds_dict["issued_at"] = "2026-01-01T00:00:00"
    path.write_text(
        _json.dumps(
            {"version": 1, "default_profile": "default", "profiles": {"default": creds_dict}}
        )
    )

    cf = load_credentials_file(path)
    profile = cf.get_profile("default")
    assert profile is not None
    # Must NOT crash on astimezone()
    assert profile.is_expired() is False
    assert profile.expires_at.tzinfo is not None


def test_get_shared_state_separates_different_profiles(tmp_path: Path):
    reset_state_cache()
    path = tmp_path / "creds.json"
    cf = CredentialsFile()
    cf.set_profile("a", _sample_creds(access_token="a"))
    cf.set_profile("b", _sample_creds(access_token="b"))
    save_credentials_file(cf, path)

    sa = get_shared_state(path, profile="a")
    sb = get_shared_state(path, profile="b")
    assert sa is not sb
    assert sa.credentials.access_token == "a"
    assert sb.credentials.access_token == "b"


# ---------------------------------------------------------------------------
# KaguraOAuth — async_auth_flow + single-flight refresh
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_oauth_injects_header_no_refresh_needed(tmp_path: Path):
    reset_state_cache()
    path = tmp_path / "creds.json"
    creds = _sample_creds(access_token="fresh-token")
    cf = CredentialsFile()
    cf.set_profile("default", creds)
    save_credentials_file(cf, path)

    state = get_shared_state(path)
    auth = KaguraOAuth(state)

    request = httpx.Request("GET", "https://test/x")
    flow = auth.async_auth_flow(request)
    out = await flow.__anext__()
    assert out.headers["Authorization"] == "Bearer fresh-token"


@pytest.mark.asyncio
async def test_oauth_refreshes_when_within_skew(tmp_path: Path):
    """When token is within REFRESH_SKEW_SEC of expiry, refresh fires."""
    reset_state_cache()
    path = tmp_path / "creds.json"
    near_expiry = datetime.now(UTC) + timedelta(seconds=60)  # < 300s skew
    creds = _sample_creds(access_token="old-token", expires_at=near_expiry)
    cf = CredentialsFile()
    cf.set_profile("default", creds)
    save_credentials_file(cf, path)

    state = get_shared_state(path)
    auth = KaguraOAuth(state)

    refreshed = TokenResponse(
        access_token="new-token",
        refresh_token="new-rtok",
        token_type="Bearer",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        scope="memory:read",
    )

    with (
        patch(
            "kagura_memory.auth.device_flow.refresh_access_token",
            new=AsyncMock(return_value=refreshed),
        ),
        patch(
            "kagura_memory.auth.device_flow.make_oauth_client",
            return_value=_AsyncContextStub(),
        ),
    ):
        request = httpx.Request("GET", "https://test/x")
        flow = auth.async_auth_flow(request)
        out = await flow.__anext__()

    assert out.headers["Authorization"] == "Bearer new-token"
    assert state.credentials.access_token == "new-token"
    # Persisted to disk too.
    on_disk = load_credentials_file(path).get_profile()
    assert on_disk.access_token == "new-token"


@pytest.mark.asyncio
async def test_oauth_refuses_refresh_without_a_stored_refresh_token(tmp_path: Path):
    """A profile with no refresh token must fail before any network call.

    Posting ``refresh_token=""`` can only ever earn an ``invalid_grant``,
    whose message ("refresh token is no longer valid") describes a token
    that never existed. See typescript-sdk#14.
    """
    reset_state_cache()
    path = tmp_path / "creds.json"
    creds = _sample_creds(
        refresh_token="",
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
    )
    cf = CredentialsFile()
    cf.set_profile("default", creds)
    save_credentials_file(cf, path)

    auth = KaguraOAuth(get_shared_state(path))

    refresh_mock = AsyncMock()
    with (
        patch("kagura_memory.auth.device_flow.refresh_access_token", new=refresh_mock),
        patch(
            "kagura_memory.auth.device_flow.make_oauth_client",
            return_value=_AsyncContextStub(),
        ),
        pytest.raises(KaguraAuthExpiredError, match="without a refresh token"),
    ):
        request = httpx.Request("GET", "https://test/x")
        flow = auth.async_auth_flow(request)
        await flow.__anext__()

    refresh_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_oauth_force_refresh_refuses_without_a_stored_refresh_token(tmp_path: Path):
    reset_state_cache()
    path = tmp_path / "creds.json"
    cf = CredentialsFile()
    cf.set_profile("default", _sample_creds(refresh_token=""))
    save_credentials_file(cf, path)

    auth = KaguraOAuth(get_shared_state(path))

    refresh_mock = AsyncMock()
    with (
        patch("kagura_memory.auth.device_flow.refresh_access_token", new=refresh_mock),
        patch(
            "kagura_memory.auth.device_flow.make_oauth_client",
            return_value=_AsyncContextStub(),
        ),
        pytest.raises(KaguraAuthExpiredError),
    ):
        await auth.force_refresh()

    refresh_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_oauth_refresh_keeps_stored_scope_when_server_omits_it(tmp_path: Path):
    """``TokenResponse.scope`` defaults to "" — that must not blank the grant.

    ``with_refreshed`` falls back only on ``None``, so passing the empty
    string through overwrote a perfectly good stored scope.
    """
    reset_state_cache()
    path = tmp_path / "creds.json"
    near_expiry = datetime.now(UTC) + timedelta(seconds=60)
    cf = CredentialsFile()
    cf.set_profile(
        "default",
        _sample_creds(expires_at=near_expiry, scope="memory:read memory:write"),
    )
    save_credentials_file(cf, path)

    auth = KaguraOAuth(get_shared_state(path))
    refreshed = TokenResponse(
        access_token="new-token",
        refresh_token="new-rtok",
        token_type="Bearer",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        scope="",
    )

    with (
        patch(
            "kagura_memory.auth.device_flow.refresh_access_token",
            new=AsyncMock(return_value=refreshed),
        ),
        patch(
            "kagura_memory.auth.device_flow.make_oauth_client",
            return_value=_AsyncContextStub(),
        ),
    ):
        request = httpx.Request("GET", "https://test/x")
        flow = auth.async_auth_flow(request)
        await flow.__anext__()

    assert load_credentials_file(path).get_profile().scope == "memory:read memory:write"


@pytest.mark.asyncio
async def test_oauth_single_flight_coalesces_concurrent_refresh(tmp_path: Path):
    """5 concurrent requests through KaguraOAuth → 1 refresh call."""
    reset_state_cache()
    path = tmp_path / "creds.json"
    near_expiry = datetime.now(UTC) + timedelta(seconds=10)
    creds = _sample_creds(access_token="old", expires_at=near_expiry)
    cf = CredentialsFile()
    cf.set_profile("default", creds)
    save_credentials_file(cf, path)

    state = get_shared_state(path)
    auth = KaguraOAuth(state)

    refresh_calls = 0

    async def fake_refresh(*args, **kwargs):
        nonlocal refresh_calls
        refresh_calls += 1
        # Simulate non-zero latency so concurrent waiters race for the lock.
        await asyncio.sleep(0.01)
        return TokenResponse(
            access_token="new",
            refresh_token="new-rtok",
            token_type="Bearer",
            expires_at=datetime.now(UTC) + timedelta(hours=1),
            scope="memory:read",
        )

    with (
        patch(
            "kagura_memory.auth.device_flow.refresh_access_token",
            new=fake_refresh,
        ),
        patch(
            "kagura_memory.auth.device_flow.make_oauth_client",
            return_value=_AsyncContextStub(),
        ),
    ):

        async def one_request():
            req = httpx.Request("GET", "https://test/x")
            flow = auth.async_auth_flow(req)
            return await flow.__anext__()

        results = await asyncio.gather(*(one_request() for _ in range(5)))

    assert refresh_calls == 1, f"expected exactly one refresh, got {refresh_calls}"
    assert all(r.headers["Authorization"] == "Bearer new" for r in results)


class _AsyncContextStub:
    """Minimal async-context-manager that returns a MagicMock client."""

    async def __aenter__(self):
        return MagicMock()

    async def __aexit__(self, *_):
        return None


@pytest.mark.asyncio
async def test_force_refresh_refreshes_even_when_not_expired(tmp_path: Path):
    """force_refresh() rotates the token regardless of the skew window (kagura-mcp 401 path)."""
    reset_state_cache()
    path = tmp_path / "creds.json"
    # Token is far from expiry → _maybe_refresh would no-op, force_refresh must not.
    creds = _sample_creds(access_token="old", expires_at=datetime.now(UTC) + timedelta(hours=1))
    cf = CredentialsFile()
    cf.set_profile("default", creds)
    save_credentials_file(cf, path)

    state = get_shared_state(path)
    auth = KaguraOAuth(state)

    refreshed = TokenResponse(
        access_token="forced-new",
        refresh_token="new-rtok",
        token_type="Bearer",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        scope="memory:read",
    )
    with (
        patch(
            "kagura_memory.auth.device_flow.refresh_access_token",
            new=AsyncMock(return_value=refreshed),
        ),
        patch(
            "kagura_memory.auth.device_flow.make_oauth_client",
            return_value=_AsyncContextStub(),
        ),
    ):
        await auth.force_refresh()

    assert state.credentials.access_token == "forced-new"
    assert load_credentials_file(path).get_profile().access_token == "forced-new"


# ---------------------------------------------------------------------------
# Cross-process refresh dedup (#158)
#
# These tests construct two _SharedCredentialsState objects with *separate*
# asyncio.Locks over the SAME credentials file to simulate two processes. The
# cross-process fcntl lock genuinely serializes them — flock treats independent
# open descriptions independently even within one process — so the dedup logic
# is exercised deterministically without spawning subprocesses (the real
# cross-process *lock* itself is covered by the subprocess test in
# tests/test_filelock.py).
# ---------------------------------------------------------------------------


def _independent_state(path: Path, creds: OAuthCredentials, profile: str = "default"):
    """A state with its own asyncio.Lock — a stand-in for a separate process."""
    return _SharedCredentialsState(
        credentials=creds,
        profile_name=profile,
        path=path.resolve(),
        lock=asyncio.Lock(),
    )


def _fresh_token(access_token: str) -> TokenResponse:
    return TokenResponse(
        access_token=access_token,
        refresh_token="new-rtok",
        token_type="Bearer",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        scope="memory:read",
    )


@pytest.mark.asyncio
async def test_refresh_dedup_skew_only_one_network_call(tmp_path: Path):
    """Two 'processes' both within the skew window → exactly one /oauth2/token call.

    The second to acquire the cross-process lock re-reads the freshly-rotated
    token from disk and adopts it instead of hitting the network again.
    """
    reset_state_cache()
    path = tmp_path / "creds.json"
    near = datetime.now(UTC) + timedelta(seconds=10)  # within REFRESH_SKEW_SEC
    cf = CredentialsFile()
    cf.set_profile("default", _sample_creds(access_token="old", expires_at=near))
    save_credentials_file(cf, path)

    auth_a = KaguraOAuth(
        _independent_state(path, _sample_creds(access_token="old", expires_at=near))
    )
    auth_b = KaguraOAuth(
        _independent_state(path, _sample_creds(access_token="old", expires_at=near))
    )

    calls = 0

    async def fake_refresh(*args, **kwargs):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.02)  # hold the lock so the other waits on it
        return _fresh_token("fresh")

    with (
        patch("kagura_memory.auth.device_flow.refresh_access_token", new=fake_refresh),
        patch("kagura_memory.auth.device_flow.make_oauth_client", return_value=_AsyncContextStub()),
    ):
        await asyncio.gather(auth_a._maybe_refresh(), auth_b._maybe_refresh())

    assert calls == 1, f"expected exactly one network refresh across processes, got {calls}"
    assert auth_a._state.credentials.access_token == "fresh"
    # The process that skipped the network still adopts the on-disk token.
    assert auth_b._state.credentials.access_token == "fresh"
    assert load_credentials_file(path).get_profile().access_token == "fresh"


@pytest.mark.asyncio
async def test_refresh_dedup_negative_control_without_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """With the cross-process lock disabled, both processes hit the network.

    This proves the dedup in the positive test is the *lock's* doing, not an
    artifact of the test harness (the recalled TOCTOU 3-layer pattern: a
    positive race test needs a negative control to prove it can fail).
    """
    import contextlib

    import kagura_memory._filelock as fl

    @contextlib.asynccontextmanager
    async def noop_lock(target, *, exclusive=True):
        yield

    monkeypatch.setattr(fl, "async_file_lock", noop_lock)

    reset_state_cache()
    path = tmp_path / "creds.json"
    near = datetime.now(UTC) + timedelta(seconds=10)
    cf = CredentialsFile()
    cf.set_profile("default", _sample_creds(access_token="old", expires_at=near))
    save_credentials_file(cf, path)

    auth_a = KaguraOAuth(
        _independent_state(path, _sample_creds(access_token="old", expires_at=near))
    )
    auth_b = KaguraOAuth(
        _independent_state(path, _sample_creds(access_token="old", expires_at=near))
    )

    calls = 0

    async def fake_refresh(*args, **kwargs):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.02)
        return _fresh_token("fresh")

    with (
        patch("kagura_memory.auth.device_flow.refresh_access_token", new=fake_refresh),
        patch("kagura_memory.auth.device_flow.make_oauth_client", return_value=_AsyncContextStub()),
    ):
        await asyncio.gather(auth_a._maybe_refresh(), auth_b._maybe_refresh())

    assert calls == 2, f"without the lock both processes should refresh, got {calls}"


@pytest.mark.asyncio
async def test_force_refresh_adopts_when_another_process_rotated(tmp_path: Path):
    """401 path: if the on-disk token already differs from the rejected one, adopt + skip.

    A different on-disk token means another proxy already rotated in response
    to the same out-of-band revocation — no second /oauth2/token call needed.
    """
    reset_state_cache()
    path = tmp_path / "creds.json"
    # Disk already holds a rotated token (a peer process refreshed first).
    cf = CredentialsFile()
    cf.set_profile("default", _sample_creds(access_token="peer-rotated"))
    save_credentials_file(cf, path)

    # This process still holds the now-rejected token in memory.
    auth = KaguraOAuth(_independent_state(path, _sample_creds(access_token="rejected-401")))

    calls = 0

    async def fake_refresh(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _fresh_token("should-not-happen")

    with (
        patch("kagura_memory.auth.device_flow.refresh_access_token", new=fake_refresh),
        patch("kagura_memory.auth.device_flow.make_oauth_client", return_value=_AsyncContextStub()),
    ):
        await auth.force_refresh()

    assert calls == 0, "a peer already rotated; this process must not hit the network"
    assert auth._state.credentials.access_token == "peer-rotated"


@pytest.mark.asyncio
async def test_force_refresh_hits_network_when_token_unchanged(tmp_path: Path):
    """401 path: an identical on-disk token means nobody rotated → this process refreshes.

    Guards against the wrong predicate: an expires_at-based skip here would
    wrongly no-op on a 401 while the rejected token is still current, looping.
    """
    reset_state_cache()
    path = tmp_path / "creds.json"
    # On-disk token is the SAME one that just got 401'd, and not near expiry.
    cf = CredentialsFile()
    cf.set_profile("default", _sample_creds(access_token="rejected-401"))
    save_credentials_file(cf, path)

    auth = KaguraOAuth(_independent_state(path, _sample_creds(access_token="rejected-401")))

    calls = 0

    async def fake_refresh(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _fresh_token("forced-new")

    with (
        patch("kagura_memory.auth.device_flow.refresh_access_token", new=fake_refresh),
        patch("kagura_memory.auth.device_flow.make_oauth_client", return_value=_AsyncContextStub()),
    ):
        await auth.force_refresh()

    assert calls == 1, "nobody rotated; the 401 must force a real refresh"
    assert auth._state.credentials.access_token == "forced-new"
    assert load_credentials_file(path).get_profile().access_token == "forced-new"


# ---------------------------------------------------------------------------
# refresh_profile_locked — the one refresh path, shared with `kagura auth` (#313)
# ---------------------------------------------------------------------------


def _single_use_token_endpoint(calls: list[str]):
    """A fake ``refresh_access_token`` that, like memory-cloud v0.79.0+, honours a
    refresh token once: a second use earns ``invalid_grant`` (the #313 failure)."""

    async def fake_refresh(_client, server, *, client_id, refresh_token, scope=None):
        if refresh_token in calls:
            raise KaguraAuthExpiredError("Your login expired (refresh token is no longer valid).")
        calls.append(refresh_token)
        await asyncio.sleep(0.02)  # hold the lock so the other refresher waits on it
        return _fresh_token(f"fresh-from-{refresh_token}")

    return fake_refresh


@pytest.mark.asyncio
@pytest.mark.parametrize("cli_first", [True, False])
async def test_cli_and_proxy_refresh_race_makes_one_token_call(tmp_path: Path, cli_first: bool):
    """`kagura auth refresh` racing a `kagura-mcp` refresh of one profile → one /oauth2/token.

    Both start from the same stale token. Whichever takes the cross-process lock
    second re-reads the file, sees the rotation and adopts it, so the single-use
    refresh token is sent once and neither ends with "login expired" (#313).
    """
    reset_state_cache()
    path = tmp_path / "creds.json"
    stale = _sample_creds(access_token="old", refresh_token="rtok-old")
    cf = CredentialsFile()
    cf.set_profile("default", stale)
    save_credentials_file(cf, path)

    proxy = KaguraOAuth(_independent_state(path, stale))
    calls: list[str] = []

    async def cli_refresh() -> OAuthCredentials:
        # What `kagura auth refresh` does: the forced path, from the token it read.
        result = await refresh_profile_locked(
            "default", expected_stale_token="old", fallback=stale, path=path
        )
        # Second to the lock → it adopts the proxy's rotation; first → it refreshes.
        assert result.refreshed is cli_first
        return result.credentials

    async def proxy_refresh() -> OAuthCredentials:
        await proxy.force_refresh()  # an upstream 401 on the stale token
        return proxy._state.credentials

    with (
        patch(
            "kagura_memory.auth.device_flow.refresh_access_token",
            new=_single_use_token_endpoint(calls),
        ),
        patch("kagura_memory.auth.device_flow.make_oauth_client", return_value=_AsyncContextStub()),
    ):
        first, second = (cli_refresh, proxy_refresh) if cli_first else (proxy_refresh, cli_refresh)
        results = await asyncio.gather(first(), second())

    assert calls == ["rtok-old"], f"expected one token call, got {calls}"
    assert [r.access_token for r in results] == ["fresh-from-rtok-old"] * 2
    on_disk = load_credentials_file(path).get_profile()
    assert on_disk is not None
    assert (on_disk.access_token, on_disk.refresh_token) == ("fresh-from-rtok-old", "new-rtok")


@pytest.mark.asyncio
async def test_cli_and_proxy_refresh_race_without_the_lock_reproduces_313(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Negative control: with the cross-process lock off, the race sends the
    refresh token twice and the loser gets "login expired" — the #313 failure."""
    import contextlib

    import kagura_memory._filelock as fl

    @contextlib.asynccontextmanager
    async def noop_lock(target, *, exclusive=True):
        yield

    monkeypatch.setattr(fl, "async_file_lock", noop_lock)

    reset_state_cache()
    path = tmp_path / "creds.json"
    stale = _sample_creds(access_token="old", refresh_token="rtok-old")
    cf = CredentialsFile()
    cf.set_profile("default", stale)
    save_credentials_file(cf, path)
    proxy = KaguraOAuth(_independent_state(path, stale))

    with (
        patch(
            "kagura_memory.auth.device_flow.refresh_access_token",
            new=_single_use_token_endpoint([]),
        ),
        patch("kagura_memory.auth.device_flow.make_oauth_client", return_value=_AsyncContextStub()),
        pytest.raises(KaguraAuthExpiredError, match="login expired"),
    ):
        await asyncio.gather(
            refresh_profile_locked(
                "default", expected_stale_token="old", fallback=stale, path=path
            ),
            proxy.force_refresh(),
        )


@pytest.mark.asyncio
async def test_refresh_profile_locked_skew_path_adopts_a_fresh_disk_token(tmp_path: Path):
    """Skew path: a token on disk outside the skew window is returned as is."""
    path = tmp_path / "creds.json"
    cf = CredentialsFile()
    cf.set_profile("default", _sample_creds(access_token="peer-fresh"))
    save_credentials_file(cf, path)
    stale = _sample_creds(access_token="old", expires_at=datetime.now(UTC) + timedelta(seconds=10))

    refresh = AsyncMock()
    with patch("kagura_memory.auth.device_flow.refresh_access_token", new=refresh):
        result = await refresh_profile_locked(
            "default", expected_stale_token=None, fallback=stale, path=path
        )

    refresh.assert_not_awaited()
    assert (result.credentials.access_token, result.refreshed) == ("peer-fresh", False)


@pytest.mark.asyncio
async def test_refresh_profile_locked_with_scope_refreshes_from_the_token_on_disk(tmp_path: Path):
    """A requested scope always reaches the server, even after a peer rotated.

    The peer's token carries the grant it had, not the one asked for, so it is
    not adopted; the refresh uses the peer's (current) refresh token, never the
    caller's used one.
    """
    path = tmp_path / "creds.json"
    cf = CredentialsFile()
    cf.set_profile("default", _sample_creds(access_token="peer", refresh_token="rtok-peer"))
    save_credentials_file(cf, path)
    stale = _sample_creds(access_token="old", refresh_token="rtok-old")

    refresh = AsyncMock(return_value=_fresh_token("narrowed"))
    with (
        patch("kagura_memory.auth.device_flow.refresh_access_token", new=refresh),
        patch("kagura_memory.auth.device_flow.make_oauth_client", return_value=_AsyncContextStub()),
    ):
        result = await refresh_profile_locked(
            "default",
            expected_stale_token="old",
            fallback=stale,
            scope="memory:read",
            path=path,
        )

    refresh.assert_awaited_once()
    assert refresh.await_args is not None
    assert refresh.await_args.kwargs["refresh_token"] == "rtok-peer"
    assert refresh.await_args.kwargs["scope"] == "memory:read"
    assert (result.credentials.access_token, result.refreshed) == ("narrowed", True)
    on_disk = load_credentials_file(path).get_profile()
    assert on_disk is not None
    assert on_disk.access_token == "narrowed"


@pytest.mark.asyncio
async def test_refresh_profile_locked_refreshes_the_fallback_when_the_profile_is_gone(
    tmp_path: Path,
):
    """A profile no longer in the file is refreshed from the caller's copy and saved,
    keeping every other profile (the behaviour KaguraOAuth always had)."""
    path = tmp_path / "creds.json"
    cf = CredentialsFile()
    cf.set_profile("other", _sample_creds(access_token="other-token"))
    save_credentials_file(cf, path)

    refresh = AsyncMock(return_value=_fresh_token("fresh"))
    with (
        patch("kagura_memory.auth.device_flow.refresh_access_token", new=refresh),
        patch("kagura_memory.auth.device_flow.make_oauth_client", return_value=_AsyncContextStub()),
    ):
        result = await refresh_profile_locked(
            "default",
            expected_stale_token="old",
            fallback=_sample_creds(access_token="old", refresh_token="rtok-mine"),
            path=path,
        )

    assert refresh.await_args is not None
    assert refresh.await_args.kwargs["refresh_token"] == "rtok-mine"
    assert (result.credentials.access_token, result.refreshed) == ("fresh", True)
    on_disk = load_credentials_file(path)
    mine, other = on_disk.get_profile("default"), on_disk.get_profile("other")
    assert mine is not None and other is not None
    assert (mine.access_token, other.access_token) == ("fresh", "other-token")


@pytest.mark.asyncio
async def test_refresh_profile_locked_saves_before_it_releases_the_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The refreshed profile is on disk when the lock is released.

    The next holder re-reads the file under the lock; a save after the release
    would let a peer in between read the used refresh token and send it again.
    One event loop has no await between the release and the save, so the race
    tests cannot show this — the release itself is observed instead.
    """
    import kagura_memory._filelock as fl

    path = tmp_path / "creds.json"
    stale = _sample_creds(access_token="old", refresh_token="rtok-old")
    cf = CredentialsFile()
    cf.set_profile("default", stale)
    save_credentials_file(cf, path)

    on_disk_at_release: list[str | None] = []
    real_release = fl._release

    def spy_release(target, owner):
        profile = load_credentials_file(path).get_profile()
        on_disk_at_release.append(profile.access_token if profile else None)
        real_release(target, owner)

    monkeypatch.setattr(fl, "_release", spy_release)
    with (
        patch(
            "kagura_memory.auth.device_flow.refresh_access_token",
            new=AsyncMock(return_value=_fresh_token("fresh")),
        ),
        patch("kagura_memory.auth.device_flow.make_oauth_client", return_value=_AsyncContextStub()),
    ):
        await refresh_profile_locked(
            "default", expected_stale_token="old", fallback=stale, path=path
        )

    assert on_disk_at_release == ["fresh"]


@pytest.mark.asyncio
async def test_refresh_not_saved_is_typed_and_keeps_the_new_token_in_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """credentials.json unwritable after the server rotated → KaguraAuthError, OSError chained.

    The rotation revoked the token on file and used up its refresh token, so the
    refreshed profile exists only in memory: KaguraOAuth keeps it (the process
    stays usable until it expires) and the error says to log in again.
    """
    import errno

    import kagura_memory.auth.credentials as creds_mod

    reset_state_cache()
    path = tmp_path / "creds.json"
    stale = _sample_creds(access_token="old", refresh_token="rtok-old")
    cf = CredentialsFile()
    cf.set_profile("default", stale)
    save_credentials_file(cf, path)
    auth = KaguraOAuth(_independent_state(path, stale))

    def disk_full(*_args, **_kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(creds_mod, "save_credentials_file", disk_full)
    with (
        patch(
            "kagura_memory.auth.device_flow.refresh_access_token",
            new=AsyncMock(return_value=_fresh_token("fresh")),
        ),
        patch("kagura_memory.auth.device_flow.make_oauth_client", return_value=_AsyncContextStub()),
        pytest.raises(KaguraAuthError, match="could not save it") as excinfo,
    ):
        await auth.force_refresh()

    assert isinstance(excinfo.value.__cause__, OSError)
    assert "kagura auth login" in str(excinfo.value)
    assert auth._state.credentials.access_token == "fresh"
    on_disk = load_credentials_file(path).get_profile()
    assert on_disk is not None
    assert on_disk.access_token == "old"


def _single_use_endpoint_issuing(sent: list[str], issued: dict[str, tuple[str, str, int]]):
    """A single-use token endpoint: refresh token → (access, refresh, ttl seconds)."""

    async def single_use(_client, server, *, client_id, refresh_token, scope=None):
        if refresh_token in sent:
            raise KaguraAuthExpiredError("Your login expired (refresh token is no longer valid).")
        sent.append(refresh_token)
        access, refresh, ttl = issued[refresh_token]
        return TokenResponse(
            access_token=access,
            refresh_token=refresh,
            token_type="Bearer",
            expires_at=datetime.now(UTC) + timedelta(seconds=ttl),
            scope="memory:read",
        )

    return single_use


def _save_fails_once(monkeypatch: pytest.MonkeyPatch) -> None:
    import kagura_memory.auth.credentials as creds_mod

    real_save = creds_mod.save_credentials_file
    failures = [OSError(28, "No space left on device")]

    def save(cf, path=None):
        if failures:
            raise failures.pop()
        real_save(cf, path)

    monkeypatch.setattr(creds_mod, "save_credentials_file", save)


@pytest.mark.asyncio
@pytest.mark.parametrize("path_kind", ["skew", "401"])
async def test_refresh_after_a_failed_save_sends_the_refresh_token_held_in_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path_kind: str
):
    """After a save failed, the next refresh sends the refresh token held in memory.

    The file still holds the profile the first refresh rotated away: its access
    token is revoked and its refresh token used up. Refreshing from it, or (401
    path) adopting it because it differs from the rejected token, would end in
    "login expired".
    """
    reset_state_cache()
    path = tmp_path / "creds.json"
    soon = datetime.now(UTC) + timedelta(seconds=5)  # within REFRESH_SKEW_SEC
    stale = _sample_creds(access_token="t1", refresh_token="r1", expires_at=soon)
    cf = CredentialsFile()
    cf.set_profile("default", stale)
    save_credentials_file(cf, path)
    auth = KaguraOAuth(_independent_state(path, stale))
    sent: list[str] = []
    # t2 is issued inside the skew window too, so the skew path refreshes again.
    issued = {"r1": ("t2", "r2", 10), "r2": ("t3", "r3", 3600)}

    _save_fails_once(monkeypatch)
    with (
        patch(
            "kagura_memory.auth.device_flow.refresh_access_token",
            new=_single_use_endpoint_issuing(sent, issued),
        ),
        patch("kagura_memory.auth.device_flow.make_oauth_client", return_value=_AsyncContextStub()),
    ):
        with pytest.raises(KaguraAuthError, match="could not save it"):
            await auth._maybe_refresh()
        assert auth._state.credentials.access_token == "t2"
        if path_kind == "skew":
            await auth._maybe_refresh()
        else:
            await auth.force_refresh()  # t2 rejected upstream

    assert sent == ["r1", "r2"]
    assert auth._state.credentials.access_token == "t3"
    assert auth._state.used_refresh_token is None
    on_disk = load_credentials_file(path).get_profile()
    assert on_disk is not None
    assert (on_disk.access_token, on_disk.refresh_token) == ("t3", "r3")


@pytest.mark.asyncio
async def test_after_a_failed_save_a_profile_saved_since_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Once another process saved the profile (here a new login), the file wins
    again: its fresh token is adopted and the in-memory refresh token is not sent."""
    reset_state_cache()
    path = tmp_path / "creds.json"
    soon = datetime.now(UTC) + timedelta(seconds=5)
    stale = _sample_creds(access_token="t1", refresh_token="r1", expires_at=soon)
    cf = CredentialsFile()
    cf.set_profile("default", stale)
    save_credentials_file(cf, path)
    auth = KaguraOAuth(_independent_state(path, stale))
    sent: list[str] = []

    _save_fails_once(monkeypatch)
    with (
        patch(
            "kagura_memory.auth.device_flow.refresh_access_token",
            new=_single_use_endpoint_issuing(sent, {"r1": ("t2", "r2", 10)}),
        ),
        patch("kagura_memory.auth.device_flow.make_oauth_client", return_value=_AsyncContextStub()),
    ):
        with pytest.raises(KaguraAuthError, match="could not save it"):
            await auth._maybe_refresh()
        update_profile(
            "default", _sample_creds(access_token="t-login", refresh_token="r-login"), path
        )
        await auth._maybe_refresh()

    assert sent == ["r1"]
    assert auth._state.credentials.access_token == "t-login"
    assert auth._state.used_refresh_token is None


@pytest.mark.asyncio
async def test_force_refresh_coalesces_with_an_in_flight_skew_refresh(tmp_path: Path):
    """A 401 force_refresh queued behind a skew refresh of one state → one token call.

    force_refresh captures the rejected token before it waits on the in-process
    lock, so it finds the file already rotated and adopts it (kagura-mcp's 401
    path racing a request that entered the skew window).
    """
    reset_state_cache()
    path = tmp_path / "creds.json"
    near = datetime.now(UTC) + timedelta(seconds=10)  # within REFRESH_SKEW_SEC
    cf = CredentialsFile()
    cf.set_profile("default", _sample_creds(access_token="old", expires_at=near))
    save_credentials_file(cf, path)
    auth = KaguraOAuth(_independent_state(path, _sample_creds(access_token="old", expires_at=near)))

    calls = 0

    async def fake_refresh(*args, **kwargs):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.02)  # the 401 arrives while this refresh is in flight
        return _fresh_token(f"fresh-{calls}")

    with (
        patch("kagura_memory.auth.device_flow.refresh_access_token", new=fake_refresh),
        patch("kagura_memory.auth.device_flow.make_oauth_client", return_value=_AsyncContextStub()),
    ):
        await asyncio.gather(auth._maybe_refresh(), auth.force_refresh())

    assert calls == 1, f"expected one token call, got {calls}"
    assert auth._state.credentials.access_token == "fresh-1"


_UNREADABLE_FILES = [
    pytest.param("not json {{{", id="not-json"),
    pytest.param("\udcff", id="not-utf8"),
    pytest.param('["an", "array"]', id="not-an-object"),
    pytest.param('{"profiles": []}', id="profiles-not-a-map"),
    pytest.param('{"profiles": {"default": {"server": "x"}}}', id="profile-missing-fields"),
]


def _write_unreadable(path: Path, text: str) -> bytes:
    raw = text.encode("utf-8", "surrogateescape")
    path.write_bytes(raw)
    return raw


@pytest.mark.parametrize("text", _UNREADABLE_FILES)
def test_load_returns_empty_on_any_unreadable_file(tmp_path: Path, text: str):
    """The lenient reader still treats every unreadable shape as empty."""
    path = tmp_path / "creds.json"
    _write_unreadable(path, text)
    assert load_credentials_file(path).profiles == {}


@pytest.mark.parametrize("text", _UNREADABLE_FILES)
@pytest.mark.parametrize("write", ["update", "set_default", "delete"])
def test_read_modify_write_refuses_a_file_it_cannot_read(tmp_path: Path, text: str, write: str):
    """update_profile / set_default_profile / delete_profile raise, naming the file.

    Loading it as empty and saving would rewrite it with one profile (or none),
    dropping every other profile and the default.
    """
    from kagura_memory.auth.credentials import delete_profile, set_default_profile

    path = tmp_path / "creds.json"
    raw = _write_unreadable(path, text)
    calls = {
        "update": lambda: update_profile("default", _sample_creds(), path),
        "set_default": lambda: set_default_profile("default", path),
        "delete": lambda: delete_profile("default", path),
    }

    with pytest.raises(KaguraAuthError, match="Cannot update") as excinfo:
        calls[write]()

    assert str(path) in str(excinfo.value)
    assert "kagura auth login" in str(excinfo.value)
    assert path.read_bytes() == raw


def test_update_profile_starts_a_missing_file(tmp_path: Path):
    """Only an existing file that cannot be read is refused; a missing one starts empty."""
    path = tmp_path / "sub" / "creds.json"
    update_profile("work", _sample_creds(access_token="first"), path)
    cf = load_credentials_file(path)
    assert cf.default_profile == "work"
    assert list(cf.profiles) == ["work"]


@pytest.mark.asyncio
@pytest.mark.parametrize("text", _UNREADABLE_FILES)
async def test_refresh_profile_locked_refuses_a_file_it_cannot_read(tmp_path: Path, text: str):
    """The refresh fails before the network on a file it cannot read, leaving it as is.

    Refreshing from the caller's copy and saving would consume the refresh token
    and rewrite the file with that one profile.
    """
    path = tmp_path / "creds.json"
    raw = _write_unreadable(path, text)
    refresh = AsyncMock(return_value=_fresh_token("fresh"))

    with (
        patch("kagura_memory.auth.device_flow.refresh_access_token", new=refresh),
        patch("kagura_memory.auth.device_flow.make_oauth_client", return_value=_AsyncContextStub()),
        pytest.raises(KaguraAuthError, match="Cannot update"),
    ):
        await refresh_profile_locked(
            "default", expected_stale_token="old", fallback=_sample_creds(), path=path
        )

    refresh.assert_not_awaited()
    assert path.read_bytes() == raw


@pytest.mark.asyncio
async def test_refresh_profile_locked_keeps_every_other_profile_and_the_default(tmp_path: Path):
    """The refresh saves the file it loaded under the lock: other profiles and the
    default survive (the second load it used to make could come back empty)."""
    path = tmp_path / "creds.json"
    cf = CredentialsFile(default_profile="work")
    cf.set_profile("work", _sample_creds(access_token="work-token"))
    cf.set_profile("default", _sample_creds(access_token="old", refresh_token="rtok-old"))
    save_credentials_file(cf, path)
    loads: list[Path] = []
    import kagura_memory.auth.credentials as creds_mod

    real_read = creds_mod._read_credentials_file

    def counting_read(p: Path) -> CredentialsFile:
        loads.append(p)
        return real_read(p)

    with (
        patch.object(creds_mod, "_read_credentials_file", counting_read),
        patch(
            "kagura_memory.auth.device_flow.refresh_access_token",
            new=AsyncMock(return_value=_fresh_token("fresh")),
        ),
        patch("kagura_memory.auth.device_flow.make_oauth_client", return_value=_AsyncContextStub()),
    ):
        await refresh_profile_locked(
            "default",
            expected_stale_token="old",
            fallback=_sample_creds(access_token="old", refresh_token="rtok-old"),
            path=path,
        )

    assert len(loads) == 1, "the file is read once under the lock"
    after = load_credentials_file(path)
    assert after.default_profile == "work"
    assert {name: p.access_token for name, p in after.profiles.items()} == {
        "work": "work-token",
        "default": "fresh",
    }


def test_delete_credentials_file_is_file_lock_guarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """logout --all unlinks under the lock, so a refresh holding it cannot save
    the profiles back right after."""
    import kagura_memory._filelock as fl

    path = tmp_path / "creds.json"
    cf = CredentialsFile()
    cf.set_profile("default", _sample_creds())
    save_credentials_file(cf, path)
    seen: list[bool] = []
    real_file_lock = fl.file_lock

    def spy_file_lock(target: Path, *, exclusive: bool = True):
        seen.append(path.exists())  # the file is still there when the lock is taken
        return real_file_lock(target, exclusive=exclusive)

    monkeypatch.setattr(fl, "file_lock", spy_file_lock)
    delete_credentials_file(path)

    assert seen == [True]
    assert not path.exists()


def test_update_profile_is_file_lock_guarded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """update_profile wraps its read-modify-write in the cross-process file lock."""
    import kagura_memory._filelock as fl

    calls: list[bool] = []
    real_file_lock = fl.file_lock

    def spy_file_lock(target: Path, *, exclusive: bool = True):
        calls.append(exclusive)
        return real_file_lock(target, exclusive=exclusive)

    monkeypatch.setattr(fl, "file_lock", spy_file_lock)

    path = tmp_path / "creds.json"
    update_profile("default", _sample_creds(access_token="locked-write"), path)

    assert calls == [True]  # acquired exactly once, exclusively
    assert load_credentials_file(path).get_profile().access_token == "locked-write"


def test_delete_profile_is_file_lock_guarded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """delete_profile wraps its read-modify-write in the cross-process file lock.

    Without the lock, a logout (delete) racing a concurrent token refresh (update)
    in a sibling kagura-mcp process can clobber a freshly written token (#190).
    """
    import kagura_memory._filelock as fl
    from kagura_memory.auth.credentials import delete_profile

    path = tmp_path / "creds.json"
    cf = CredentialsFile()
    cf.set_profile("alice", _sample_creds())
    cf.set_profile("bob", _sample_creds())
    save_credentials_file(cf, path)

    calls: list[bool] = []
    real_file_lock = fl.file_lock

    def spy_file_lock(target: Path, *, exclusive: bool = True):
        calls.append(exclusive)
        return real_file_lock(target, exclusive=exclusive)

    monkeypatch.setattr(fl, "file_lock", spy_file_lock)

    delete_profile("alice", path)

    assert calls == [True]  # acquired exactly once, exclusively
    assert "alice" not in load_credentials_file(path).profiles
    assert "bob" in load_credentials_file(path).profiles


def test_delete_profile_no_op_is_lock_guarded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The absent-profile early return still happens under the lock (no TOCTOU)."""
    import kagura_memory._filelock as fl
    from kagura_memory.auth.credentials import delete_profile

    path = tmp_path / "creds.json"
    cf = CredentialsFile()
    cf.set_profile("alice", _sample_creds())
    save_credentials_file(cf, path)

    calls: list[bool] = []
    real_file_lock = fl.file_lock

    def spy_file_lock(target: Path, *, exclusive: bool = True):
        calls.append(exclusive)
        return real_file_lock(target, exclusive=exclusive)

    monkeypatch.setattr(fl, "file_lock", spy_file_lock)

    delete_profile("does-not-exist", path)

    assert calls == [True]  # lock acquired before the membership check
    assert "alice" in load_credentials_file(path).profiles


@pytest.mark.parametrize("full", [True, False])
@pytest.mark.asyncio
async def test_logout_revokes_missing_fallback_even_after_explicit_login(tmp_path, full):
    path = tmp_path / "credentials.json"
    cached = _sample_creds()
    update_profile("mine", cached, path)
    update_profile("other", cached, path)
    if full:
        delete_credentials_file(path)
    else:
        delete_profile("mine", path)
        assert load_credentials_file(path).get_profile("other")
    with patch("kagura_memory.auth.credentials._network_refresh") as request:
        for after_login in (False, True):
            if after_login:
                update_profile("mine", cached, path)
                assert load_credentials_file(path).get_profile("mine")
                path.unlink()
            with pytest.raises(KaguraAuthError, match="logged out"):
                await refresh_profile_locked(
                    "mine", expected_stale_token=None, fallback=cached, path=path
                )
        request.assert_not_called()
    assert not path.exists()


@pytest.mark.asyncio
async def test_corrupt_revocations_refuse_missing_refresh(tmp_path):
    path = tmp_path / "credentials.json"
    (tmp_path / "credentials.json.revocations-v1").write_text('{"version":true}')
    with patch("kagura_memory.auth.credentials._network_refresh") as request:
        with pytest.raises(KaguraAuthError, match="revocations"):
            await refresh_profile_locked(
                "mine", expected_stale_token=None, fallback=_sample_creds(), path=path
            )
        request.assert_not_called()


@pytest.mark.parametrize("first", ["node", "python"])
@pytest.mark.asyncio
async def test_node_sdk_and_python_share_one_refresh(first, tmp_path, monkeypatch):
    """Set KAGURA_NODE_SDK to the installed TypeScript SDK entry point to run."""
    import subprocess
    import time

    import kagura_memory._filelock as locks

    sdk = os.environ.get("KAGURA_NODE_SDK")
    if not sdk:
        pytest.skip("requires the TypeScript SDK interoperability fixture")
    path = tmp_path / "credentials.json"
    cached = _sample_creds(expires_at=datetime.now(UTC) - timedelta(minutes=1))
    update_profile("mine", cached, path)
    update_profile("other", cached, path)

    def launch():
        return subprocess.Popen(
            [
                "node",
                str(Path(__file__).parent / "fixtures/node_credential_refresh.cjs"),
                sdk,
                str(path),
                first,
            ],
            stderr=subprocess.PIPE,
            text=True,
        )

    async def wait_for(suffix):
        deadline = time.monotonic() + 10
        while not Path(str(path) + suffix).exists():
            assert proc is not None and proc.poll() is None, proc.stderr.read() if proc else ""
            assert time.monotonic() < deadline, suffix
            await asyncio.sleep(0.01)

    proc = None

    async def refresh(base, *, scope):
        nonlocal proc
        assert first == "python"
        proc = launch()
        await wait_for(".node-waiting")
        return base.with_refreshed(
            access_token="python", expires_at=datetime.now(UTC) + timedelta(hours=1)
        )

    try:
        if first == "node":
            proc = launch()
            await wait_for(".node-refreshing")
            acquire = locks._try_acquire

            def tried(target, deadline):
                owner = acquire(target, deadline)
                if owner is None:
                    Path(str(path) + ".python-waiting").touch()
                return owner

            monkeypatch.setattr(locks, "_try_acquire", tried)
        with patch("kagura_memory.auth.credentials._network_refresh", refresh):
            result = await refresh_profile_locked(
                "mine", expected_stale_token=None, fallback=cached, path=path
            )
        assert proc is not None
        _, stderr = await asyncio.to_thread(proc.communicate, timeout=15)
        assert proc.returncode == 0, stderr
        assert result.credentials.access_token == first
        assert load_credentials_file(path).profiles["other"].access_token == cached.access_token
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()
