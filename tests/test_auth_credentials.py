"""Tests for kagura_memory.auth.credentials."""

from __future__ import annotations

import asyncio
import json
import os
import stat
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
    delete_credentials_file,
    delete_profile,
    get_shared_state,
    load_credentials_file,
    reset_state_cache,
    save_credentials_file,
    update_profile,
)
from kagura_memory.auth.device_flow import TokenResponse


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


def test_save_enforces_file_mode_0600(tmp_path: Path):
    path = tmp_path / "creds.json"
    cf = CredentialsFile()
    cf.set_profile("default", _sample_creds())
    save_credentials_file(cf, path)
    assert stat.S_IMODE(path.stat().st_mode) == CREDENTIALS_FILE_MODE


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
