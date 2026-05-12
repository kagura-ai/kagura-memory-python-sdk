"""Tests for kagura_memory.auth.cli — the five `kagura auth` sub-commands."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from kagura_memory.auth.credentials import (
    CredentialsFile,
    OAuthCredentials,
    load_credentials_file,
    reset_state_cache,
    save_credentials_file,
)
from kagura_memory.auth.device_flow import (
    DeviceAuthorizationResponse,
    TokenResponse,
)
from kagura_memory.cli import main


def _make_creds(
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
        workspace_id="ws-abcdefgh",
        workspace_name="test-workspace",
        user_email="user@example.com",
        issued_at=datetime.now(UTC),
    )


def _seed_credentials(tmp_path: Path, creds: OAuthCredentials, profile: str = "default") -> Path:
    """Write a credentials file at the standard location relative to tmp_path."""
    path = tmp_path / ".kagura" / "credentials.json"
    cf = CredentialsFile(default_profile=profile, profiles={profile: creds})
    save_credentials_file(cf, path)
    return path


@pytest.fixture
def patched_default_path(tmp_path: Path, monkeypatch):
    """Redirect every credentials.py caller to tmp_path-local credentials.json."""
    fake_path = tmp_path / ".kagura" / "credentials.json"
    monkeypatch.setattr("kagura_memory.auth.credentials.DEFAULT_CREDENTIALS_PATH", fake_path)
    monkeypatch.setattr("kagura_memory.auth.cli.DEFAULT_CREDENTIALS_PATH", fake_path)
    reset_state_cache()
    yield fake_path
    reset_state_cache()


# ---------------------------------------------------------------------------
# login
# ---------------------------------------------------------------------------


def _mock_device_response(**overrides):
    return DeviceAuthorizationResponse(
        device_code="dc-1",
        user_code="ABCD-1234",
        verification_uri="https://test.example.com/device",
        verification_uri_complete="https://test.example.com/device?user_code=ABCD-1234",
        expires_in=600,
        interval=5,
        expires_at=datetime.now(UTC) + timedelta(minutes=10),
        **overrides,
    )


def _mock_token_response(**overrides):
    defaults: dict = {
        "access_token": "atok-fresh",
        "refresh_token": "rtok-fresh",
        "token_type": "Bearer",
        "expires_at": datetime.now(UTC) + timedelta(hours=1),
        "scope": "memory:read",
        "user_email": "user@example.com",
        "workspace_id": "ws-abcdefgh",
        "workspace_name": "test-workspace",
    }
    defaults.update(overrides)
    return TokenResponse(**defaults)


@patch("kagura_memory.auth.cli.webbrowser.open", return_value=True)
@patch("kagura_memory.auth.cli.poll_for_token", new_callable=AsyncMock)
@patch("kagura_memory.auth.cli.authorize_device", new_callable=AsyncMock)
@patch("kagura_memory.auth.cli.make_oauth_client")
def test_login_happy_path(
    mock_client_factory,
    mock_authorize,
    mock_poll,
    mock_browser,
    patched_default_path: Path,
):
    mock_client_factory.return_value = _async_ctx()
    mock_authorize.return_value = _mock_device_response()
    mock_poll.return_value = _mock_token_response()

    result = CliRunner().invoke(main, ["auth", "login"])
    assert result.exit_code == 0, result.output
    # URL+code MUST be printed BEFORE any browser-opening logic.
    output = result.output
    assert "ABCD-1234" in output
    assert "verification_uri_complete" not in output  # we print the value, not the key
    assert "https://test.example.com/device?user_code=ABCD-1234" in output
    assert "user@example.com" in output
    mock_browser.assert_called_once()
    # Credentials persisted.
    cf = load_credentials_file(patched_default_path)
    assert cf.get_profile("default").access_token == "atok-fresh"


@patch("kagura_memory.auth.cli.webbrowser.open")
@patch("kagura_memory.auth.cli.poll_for_token", new_callable=AsyncMock)
@patch("kagura_memory.auth.cli.authorize_device", new_callable=AsyncMock)
@patch("kagura_memory.auth.cli.make_oauth_client")
def test_login_no_browser_does_not_call_webbrowser(
    mock_client_factory,
    mock_authorize,
    mock_poll,
    mock_browser,
    patched_default_path: Path,
):
    mock_client_factory.return_value = _async_ctx()
    mock_authorize.return_value = _mock_device_response()
    mock_poll.return_value = _mock_token_response()

    result = CliRunner().invoke(main, ["auth", "login", "--no-browser"])
    assert result.exit_code == 0
    mock_browser.assert_not_called()
    assert "--no-browser" in result.output
    # URL+code still printed.
    assert "ABCD-1234" in result.output


@patch("kagura_memory.auth.cli.webbrowser.open", return_value=False)
@patch("kagura_memory.auth.cli.poll_for_token", new_callable=AsyncMock)
@patch("kagura_memory.auth.cli.authorize_device", new_callable=AsyncMock)
@patch("kagura_memory.auth.cli.make_oauth_client")
def test_login_falls_back_when_webbrowser_returns_false(
    mock_client_factory,
    mock_authorize,
    mock_poll,
    mock_browser,
    patched_default_path: Path,
):
    mock_client_factory.return_value = _async_ctx()
    mock_authorize.return_value = _mock_device_response()
    mock_poll.return_value = _mock_token_response()

    result = CliRunner().invoke(main, ["auth", "login"])
    assert result.exit_code == 0
    assert "Could not auto-open" in result.output


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def test_status_redacts_access_token_and_hides_refresh_token(patched_default_path: Path):
    creds = _make_creds(access_token="kagura_12345678abcdef", refresh_token="rtok-secret")
    _seed_credentials(patched_default_path.parent.parent, creds)

    result = CliRunner().invoke(main, ["auth", "status"])
    assert result.exit_code == 0
    # Redacted access_token shown as "kagura_1...cdef".
    assert "kagura_1...cdef" in result.output
    # refresh_token must NEVER appear in output, not even redacted.
    assert "rtok-secret" not in result.output
    assert "rtok" not in result.output
    assert "Refresh" not in result.output  # no "Refresh token" line


def test_status_no_credentials_errors_with_login_hint(patched_default_path: Path):
    result = CliRunner().invoke(main, ["auth", "status"])
    assert result.exit_code != 0
    assert "kagura auth login" in result.output


# ---------------------------------------------------------------------------
# logout
# ---------------------------------------------------------------------------


@patch("kagura_memory.auth.cli.revoke_token", new_callable=AsyncMock)
@patch("kagura_memory.auth.cli.make_oauth_client")
def test_logout_revokes_and_deletes_profile(
    mock_client_factory,
    mock_revoke,
    patched_default_path: Path,
):
    mock_client_factory.return_value = _async_ctx()
    mock_revoke.return_value = True
    creds = _make_creds()
    _seed_credentials(patched_default_path.parent.parent, creds)

    result = CliRunner().invoke(main, ["auth", "logout"])
    assert result.exit_code == 0
    assert "removed" in result.output
    mock_revoke.assert_called_once()
    # Profile deleted from file.
    cf = load_credentials_file(patched_default_path)
    assert cf.get_profile("default") is None


def test_logout_all_without_yes_errors(patched_default_path: Path):
    creds = _make_creds()
    _seed_credentials(patched_default_path.parent.parent, creds)

    result = CliRunner().invoke(main, ["auth", "logout", "--all"])
    assert result.exit_code != 0
    assert "--yes" in result.output


@patch("kagura_memory.auth.cli.revoke_token", new_callable=AsyncMock)
@patch("kagura_memory.auth.cli.make_oauth_client")
def test_logout_all_with_yes_deletes_file(
    mock_client_factory,
    mock_revoke,
    patched_default_path: Path,
):
    mock_client_factory.return_value = _async_ctx()
    mock_revoke.return_value = True
    creds = _make_creds()
    _seed_credentials(patched_default_path.parent.parent, creds)
    assert patched_default_path.exists()

    result = CliRunner().invoke(main, ["auth", "logout", "--all", "--yes"])
    assert result.exit_code == 0
    assert not patched_default_path.exists()


def test_logout_no_credentials_errors(patched_default_path: Path):
    result = CliRunner().invoke(main, ["auth", "logout"])
    assert result.exit_code != 0
    assert "No credentials" in result.output


# ---------------------------------------------------------------------------
# refresh
# ---------------------------------------------------------------------------


@patch("kagura_memory.auth.cli.refresh_access_token", new_callable=AsyncMock)
@patch("kagura_memory.auth.cli.make_oauth_client")
def test_refresh_rotates_access_token(
    mock_client_factory,
    mock_refresh,
    patched_default_path: Path,
):
    mock_client_factory.return_value = _async_ctx()
    mock_refresh.return_value = _mock_token_response(access_token="atok-rotated")
    _seed_credentials(patched_default_path.parent.parent, _make_creds())

    result = CliRunner().invoke(main, ["auth", "refresh"])
    assert result.exit_code == 0, result.output
    assert "Refreshed" in result.output
    cf = load_credentials_file(patched_default_path)
    assert cf.get_profile("default").access_token == "atok-rotated"


@patch("kagura_memory.auth.cli.poll_for_token", new_callable=AsyncMock)
@patch("kagura_memory.auth.cli.authorize_device", new_callable=AsyncMock)
@patch("kagura_memory.auth.cli.refresh_access_token", new_callable=AsyncMock)
@patch("kagura_memory.auth.cli.webbrowser.open", return_value=True)
@patch("kagura_memory.auth.cli.make_oauth_client")
def test_refresh_scope_widening_triggers_device_flow(
    mock_client_factory,
    mock_browser,
    mock_refresh,
    mock_authorize,
    mock_poll,
    patched_default_path: Path,
):
    """Wider scope → refresh fails with insufficient_scope → device flow runs."""
    from kagura_memory.exceptions import KaguraAuthError

    mock_client_factory.return_value = _async_ctx()
    # First call (refresh) raises insufficient_scope.
    mock_refresh.side_effect = KaguraAuthError("insufficient_scope")
    mock_authorize.return_value = _mock_device_response()
    mock_poll.return_value = _mock_token_response(
        scope="memory:read memory:write",
        access_token="atok-write",
    )
    _seed_credentials(patched_default_path.parent.parent, _make_creds())

    result = CliRunner().invoke(main, ["auth", "refresh", "--scope", "memory:read memory:write"])
    assert result.exit_code == 0, result.output
    assert "re-running the device flow" in result.output
    mock_authorize.assert_called_once()
    mock_poll.assert_called_once()
    cf = load_credentials_file(patched_default_path)
    assert cf.get_profile("default").access_token == "atok-write"
    assert "memory:write" in cf.get_profile("default").scope


# ---------------------------------------------------------------------------
# token
# ---------------------------------------------------------------------------


def test_token_emits_raw_access_token(patched_default_path: Path):
    creds = _make_creds(access_token="atok-raw")
    _seed_credentials(patched_default_path.parent.parent, creds)

    result = CliRunner().invoke(main, ["auth", "token"])
    assert result.exit_code == 0
    # stdout should contain the raw token (first line).
    assert result.output.splitlines()[0] == "atok-raw"


def test_token_warns_about_expiry_on_stderr(patched_default_path: Path):
    """The expiry warning goes to stderr (not stdout) for clean piping."""
    creds = _make_creds(access_token="atok-raw")
    _seed_credentials(patched_default_path.parent.parent, creds)

    # Click 8.3 splits stdout/stderr automatically — no mix_stderr kwarg.
    result = CliRunner().invoke(main, ["auth", "token"])
    assert result.exit_code == 0
    # stdout: only the raw token on the first line.
    assert result.stdout.strip().splitlines()[0] == "atok-raw"
    # stderr: expiry warning + persist note.
    assert "expires at" in result.stderr.lower()
    assert "Don't persist" in result.stderr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _async_ctx():
    """Build a stub async-context-manager returning a MagicMock client."""

    class _Ctx:
        async def __aenter__(self):
            return MagicMock()

        async def __aexit__(self, *_):
            return None

    return _Ctx()
