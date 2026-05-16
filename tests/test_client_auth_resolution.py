"""Tests for KaguraClient's credential resolution chain."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from kagura_memory import KaguraClient
from kagura_memory._auth import _resolve_auth, _StaticAuth
from kagura_memory.auth.credentials import (
    CredentialsFile,
    KaguraOAuth,
    OAuthCredentials,
    reset_state_cache,
    save_credentials_file,
)
from kagura_memory.exceptions import KaguraAuthError


def _make_creds() -> OAuthCredentials:
    return OAuthCredentials(
        server="https://oauth.example.com",
        mcp_url="https://oauth.example.com/mcp",
        client_id="kagura-cli",
        access_token="atok-from-oauth",
        refresh_token="rtok-from-oauth",
        token_type="Bearer",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        scope="memory:read",
        workspace_id="ws-1",
        workspace_name="oauth-ws",
        user_email="oauth@example.com",
        issued_at=datetime.now(UTC),
    )


@pytest.fixture
def isolated_credentials(tmp_path: Path, monkeypatch):
    """Redirect credentials.json reads/writes to tmp_path; clear all env vars
    that affect resolution so each test starts from a clean slate."""
    fake_path = tmp_path / "credentials.json"
    monkeypatch.setattr("kagura_memory.auth.credentials.DEFAULT_CREDENTIALS_PATH", fake_path)
    monkeypatch.delenv("KAGURA_API_KEY", raising=False)
    monkeypatch.delenv("KAGURA_PROFILE", raising=False)
    monkeypatch.delenv("KAGURA_MCP_URL", raising=False)
    # Also short-circuit load_config so legacy .kagura.json paths don't leak.
    monkeypatch.setattr("kagura_memory._auth.load_config", lambda: {"api_key": ""})
    reset_state_cache()
    yield fake_path
    reset_state_cache()


# ---------------------------------------------------------------------------
# Backwards-compatible explicit path
# ---------------------------------------------------------------------------


def test_explicit_api_key_static_path_unchanged(isolated_credentials):
    """KaguraClient(api_key=..., mcp_url=...) → static bearer header path."""
    client = KaguraClient(api_key="key-1", mcp_url="https://test.com/mcp")
    assert client._client.headers.get("Authorization") == "Bearer key-1"
    # No httpx.Auth was attached for the static path.
    assert client._client.auth is None
    # api_key not stored on instance (per python.md rule).
    assert not hasattr(client, "api_key")


def test_explicit_api_key_overrides_credentials_file(isolated_credentials):
    """Explicit api_key argument wins over a present credentials.json."""
    cf = CredentialsFile()
    cf.set_profile("default", _make_creds())
    save_credentials_file(cf, isolated_credentials)

    client = KaguraClient(api_key="explicit", mcp_url="https://test.com/mcp")
    assert client._client.headers.get("Authorization") == "Bearer explicit"


# ---------------------------------------------------------------------------
# Env var precedence
# ---------------------------------------------------------------------------


def test_kagura_api_key_env_wins(isolated_credentials, monkeypatch):
    """KAGURA_API_KEY env > credentials.json (auto-resolution chain rule 1)."""
    cf = CredentialsFile()
    cf.set_profile("default", _make_creds())
    save_credentials_file(cf, isolated_credentials)

    monkeypatch.setenv("KAGURA_API_KEY", "env-key")
    client = KaguraClient()
    assert client._client.headers.get("Authorization") == "Bearer env-key"


def test_kagura_api_key_env_uses_kagura_mcp_url_env(isolated_credentials, monkeypatch):
    monkeypatch.setenv("KAGURA_API_KEY", "env-key")
    monkeypatch.setenv("KAGURA_MCP_URL", "https://env.example.com/mcp")
    client = KaguraClient()
    assert client.mcp_url == "https://env.example.com/mcp"


def test_whitespace_only_api_key_falls_through(isolated_credentials, monkeypatch):
    """An explicit api_key='   ' must not produce `Authorization: Bearer `."""
    cf = CredentialsFile()
    cf.set_profile("default", _make_creds())
    save_credentials_file(cf, isolated_credentials)

    # Whitespace-only explicit api_key → fall through to OAuth profile.
    client = KaguraClient(api_key="   ")
    assert isinstance(client._client.auth, KaguraOAuth)


def test_whitespace_only_kagura_api_key_env_falls_through(isolated_credentials, monkeypatch):
    """KAGURA_API_KEY='   ' must not produce `Authorization: Bearer `."""
    cf = CredentialsFile()
    cf.set_profile("default", _make_creds())
    save_credentials_file(cf, isolated_credentials)

    monkeypatch.setenv("KAGURA_API_KEY", "   ")
    client = KaguraClient()
    # Falls through to credentials.json OAuth profile.
    assert isinstance(client._client.auth, KaguraOAuth)


# ---------------------------------------------------------------------------
# credentials.json OAuth path
# ---------------------------------------------------------------------------


def test_no_args_picks_up_default_profile(isolated_credentials):
    cf = CredentialsFile()
    cf.set_profile("default", _make_creds())
    save_credentials_file(cf, isolated_credentials)

    client = KaguraClient()
    assert isinstance(client._client.auth, KaguraOAuth)
    # mcp_url derived from the profile.
    assert client.mcp_url == "https://oauth.example.com/mcp"


def test_explicit_profile_arg_overrides_default(isolated_credentials):
    cf = CredentialsFile()
    work_creds = _make_creds()
    work_creds = work_creds.with_refreshed(
        access_token="atok-work",
        expires_at=work_creds.expires_at,
    )
    cf.set_profile("default", _make_creds())
    cf.set_profile("work", work_creds)
    save_credentials_file(cf, isolated_credentials)

    client = KaguraClient(profile="work")
    auth = client._client.auth
    assert isinstance(auth, KaguraOAuth)
    # pyright doesn't narrow client._client.auth through isinstance, so
    # access the private state via the verified KaguraOAuth instance.
    oauth_state = auth._state  # type: ignore[attr-defined]
    assert oauth_state.credentials.access_token == "atok-work"


def test_kagura_profile_env_selects_profile(isolated_credentials, monkeypatch):
    cf = CredentialsFile()
    cf.set_profile("default", _make_creds())
    work_creds = OAuthCredentials(
        server="https://oauth.example.com",
        mcp_url="https://oauth.example.com/mcp",
        client_id="kagura-cli",
        access_token="atok-work-env",
        refresh_token="rtok-work",
        token_type="Bearer",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        scope="memory:read",
        workspace_id="ws-2",
        workspace_name="work-ws",
        user_email="work@example.com",
        issued_at=datetime.now(UTC),
    )
    cf.set_profile("work", work_creds)
    save_credentials_file(cf, isolated_credentials)

    monkeypatch.setenv("KAGURA_PROFILE", "work")
    client = KaguraClient()
    auth = client._client.auth
    assert isinstance(auth, KaguraOAuth)
    # pyright doesn't narrow client._client.auth through isinstance, so
    # access the private state via the verified KaguraOAuth instance.
    oauth_state = auth._state  # type: ignore[attr-defined]
    assert oauth_state.credentials.access_token == "atok-work-env"


def test_explicit_profile_arg_beats_kagura_profile_env(isolated_credentials, monkeypatch):
    cf = CredentialsFile()
    cf.set_profile("default", _make_creds())
    cf.set_profile(
        "alpha",
        OAuthCredentials(
            server="https://oauth.example.com",
            mcp_url="https://oauth.example.com/mcp",
            client_id="kagura-cli",
            access_token="atok-alpha",
            refresh_token="rtok",
            token_type="Bearer",
            expires_at=datetime.now(UTC) + timedelta(hours=1),
            scope="memory:read",
            workspace_id="ws-a",
            workspace_name="alpha-ws",
            user_email="a@example.com",
            issued_at=datetime.now(UTC),
        ),
    )
    cf.set_profile(
        "beta",
        OAuthCredentials(
            server="https://oauth.example.com",
            mcp_url="https://oauth.example.com/mcp",
            client_id="kagura-cli",
            access_token="atok-beta",
            refresh_token="rtok",
            token_type="Bearer",
            expires_at=datetime.now(UTC) + timedelta(hours=1),
            scope="memory:read",
            workspace_id="ws-b",
            workspace_name="beta-ws",
            user_email="b@example.com",
            issued_at=datetime.now(UTC),
        ),
    )
    save_credentials_file(cf, isolated_credentials)

    monkeypatch.setenv("KAGURA_PROFILE", "beta")
    client = KaguraClient(profile="alpha")
    auth = client._client.auth
    assert isinstance(auth, KaguraOAuth)
    # pyright doesn't narrow client._client.auth through isinstance, so
    # access the private state via the verified KaguraOAuth instance.
    oauth_state = auth._state  # type: ignore[attr-defined]
    assert oauth_state.credentials.access_token == "atok-alpha"


# ---------------------------------------------------------------------------
# Failure path
# ---------------------------------------------------------------------------


def test_no_credentials_raises_with_login_hint(isolated_credentials):
    """No api_key + no env + no credentials.json + empty load_config → raise."""
    with pytest.raises(KaguraAuthError, match="kagura auth login"):
        KaguraClient()


def test_explicit_profile_not_found_raises_loud(isolated_credentials):
    """profile= arg pointing at a missing profile must raise, not silently fall through."""
    cf = CredentialsFile()
    cf.set_profile("default", _make_creds())
    save_credentials_file(cf, isolated_credentials)

    with pytest.raises(KaguraAuthError, match="Profile 'missing'"):
        KaguraClient(profile="missing")


def test_kagura_profile_env_not_found_raises_loud(isolated_credentials, monkeypatch):
    """KAGURA_PROFILE env pointing at a missing profile must raise."""
    cf = CredentialsFile()
    cf.set_profile("default", _make_creds())
    save_credentials_file(cf, isolated_credentials)
    monkeypatch.setenv("KAGURA_PROFILE", "ghost")

    with pytest.raises(KaguraAuthError, match="KAGURA_PROFILE"):
        KaguraClient()


# ---------------------------------------------------------------------------
# Explicit mcp_url override
# ---------------------------------------------------------------------------


def test_explicit_mcp_url_overrides_profile_mcp_url(isolated_credentials):
    cf = CredentialsFile()
    cf.set_profile("default", _make_creds())
    save_credentials_file(cf, isolated_credentials)

    client = KaguraClient(mcp_url="https://override.example.com/mcp")
    assert client.mcp_url == "https://override.example.com/mcp"


# ---------------------------------------------------------------------------
# Terminal .kagura.json fallback (static path of last resort)
# ---------------------------------------------------------------------------


def test_kagura_json_fallback_static_path(tmp_path: Path, monkeypatch):
    """Last branch of the resolver: api_key=None, env clear, no profile → .kagura.json wins.

    The credentials precedence ends at ``.kagura.json``; when every
    earlier source (explicit arg, ``KAGURA_API_KEY``, OAuth profile) is
    absent and the legacy config has an ``api_key``, the resolver must
    return a ``_StaticAuth`` carrying that key. ``isolated_credentials``
    blanks ``load_config`` for tests that exercise OAuth paths, so this
    branch needs a dedicated fixture that lets ``load_config`` produce
    a real ``api_key``.
    """
    monkeypatch.setattr(
        "kagura_memory.auth.credentials.DEFAULT_CREDENTIALS_PATH",
        tmp_path / "missing-credentials.json",
    )
    monkeypatch.delenv("KAGURA_API_KEY", raising=False)
    monkeypatch.delenv("KAGURA_PROFILE", raising=False)
    monkeypatch.delenv("KAGURA_MCP_URL", raising=False)
    monkeypatch.setattr(
        "kagura_memory._auth.load_config",
        lambda: {"api_key": "key-from-dot-kagura", "mcp_url": "https://legacy.example.com/mcp"},
    )
    reset_state_cache()

    client = KaguraClient()
    assert client._client.headers.get("Authorization") == "Bearer key-from-dot-kagura"
    assert client._client.auth is None  # static path, no httpx.Auth attached
    assert client.mcp_url == "https://legacy.example.com/mcp"
    reset_state_cache()


def test_resolve_auth_uses_passed_config_without_disk_read(isolated_credentials, monkeypatch):
    """_resolve_auth(config=...) honors a caller-supplied .kagura.json dict.

    Callers that have already loaded the config (the CLI's
    ``_run_files_command`` and ``_get_resource_client``) pass it through
    so the priority-4 fallback never re-reads from disk. Verify the
    contract directly: stub ``load_config`` to fail loudly, then call
    ``_resolve_auth`` with an explicit ``config``. If the kwarg is
    honored, the test passes without the stub firing.
    """
    monkeypatch.setattr(
        "kagura_memory._auth.load_config",
        lambda: pytest.fail("_resolve_auth must not re-read .kagura.json when config= is passed"),
    )

    resolved = _resolve_auth(
        api_key=None,
        mcp_url=None,
        profile=None,
        config={"api_key": "from-passed-config", "mcp_url": "https://passed.example.com/mcp"},
    )
    assert isinstance(resolved, _StaticAuth)
    assert resolved.source == "config"
    assert resolved.api_key == "from-passed-config"
    assert resolved.mcp_url == "https://passed.example.com/mcp"
