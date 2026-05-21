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


@patch("kagura_memory.auth.cli._try_open_browser", return_value=True)
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


@patch("kagura_memory.auth.cli._try_open_browser")
@patch("kagura_memory.auth.cli.poll_for_token", new_callable=AsyncMock)
@patch("kagura_memory.auth.cli.authorize_device", new_callable=AsyncMock)
@patch("kagura_memory.auth.cli.make_oauth_client")
def test_login_no_browser_does_not_call_browser_opener(
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


@patch("kagura_memory.auth.cli._try_open_browser", return_value=False)
@patch("kagura_memory.auth.cli.poll_for_token", new_callable=AsyncMock)
@patch("kagura_memory.auth.cli.authorize_device", new_callable=AsyncMock)
@patch("kagura_memory.auth.cli.make_oauth_client")
def test_login_falls_back_when_browser_open_fails(
    mock_client_factory,
    mock_authorize,
    mock_poll,
    mock_open,
    patched_default_path: Path,
):
    """When every browser-open path fails, print the manual hint."""
    mock_client_factory.return_value = _async_ctx()
    mock_authorize.return_value = _mock_device_response()
    mock_poll.return_value = _mock_token_response()

    result = CliRunner().invoke(main, ["auth", "login"])
    assert result.exit_code == 0
    assert "Could not auto-open" in result.output
    mock_open.assert_called_once()


@patch("kagura_memory.auth.cli.poll_for_token", new_callable=AsyncMock)
@patch("kagura_memory.auth.cli.authorize_device", new_callable=AsyncMock)
@patch("kagura_memory.auth.cli.make_oauth_client")
def test_login_default_scope_is_read_plus_write(
    mock_client_factory,
    mock_authorize,
    mock_poll,
    patched_default_path: Path,
):
    """No flags → request 'memory:read memory:write' (the CLI's main use case)."""
    mock_client_factory.return_value = _async_ctx()
    mock_authorize.return_value = _mock_device_response()
    mock_poll.return_value = _mock_token_response()

    with patch("kagura_memory.auth.cli._try_open_browser", return_value=True):
        result = CliRunner().invoke(main, ["auth", "login"])
    assert result.exit_code == 0, result.output
    # authorize_device was called with the read+write scope.
    _, kwargs = mock_authorize.call_args
    assert kwargs.get("scope") == "memory:read memory:write"


@patch("kagura_memory.auth.cli.poll_for_token", new_callable=AsyncMock)
@patch("kagura_memory.auth.cli.authorize_device", new_callable=AsyncMock)
@patch("kagura_memory.auth.cli.make_oauth_client")
def test_login_read_only_flag_requests_read_scope(
    mock_client_factory,
    mock_authorize,
    mock_poll,
    patched_default_path: Path,
):
    mock_client_factory.return_value = _async_ctx()
    mock_authorize.return_value = _mock_device_response()
    mock_poll.return_value = _mock_token_response()

    with patch("kagura_memory.auth.cli._try_open_browser", return_value=True):
        result = CliRunner().invoke(main, ["auth", "login", "--read-only"])
    assert result.exit_code == 0, result.output
    _, kwargs = mock_authorize.call_args
    assert kwargs.get("scope") == "memory:read"


def test_login_read_only_and_scope_are_mutually_exclusive(patched_default_path: Path):
    result = CliRunner().invoke(main, ["auth", "login", "--read-only", "--scope", "memory:read"])
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


@patch("kagura_memory.auth.cli.poll_for_token", new_callable=AsyncMock)
@patch("kagura_memory.auth.cli.authorize_device", new_callable=AsyncMock)
@patch("kagura_memory.auth.cli.make_oauth_client")
def test_login_explicit_scope_is_honored(
    mock_client_factory,
    mock_authorize,
    mock_poll,
    patched_default_path: Path,
):
    mock_client_factory.return_value = _async_ctx()
    mock_authorize.return_value = _mock_device_response()
    mock_poll.return_value = _mock_token_response()

    with patch("kagura_memory.auth.cli._try_open_browser", return_value=True):
        result = CliRunner().invoke(main, ["auth", "login", "--scope", "memory:read profile:read"])
    assert result.exit_code == 0, result.output
    _, kwargs = mock_authorize.call_args
    assert kwargs.get("scope") == "memory:read profile:read"


@patch("kagura_memory.auth.cli._try_open_browser", return_value=True)
@patch("kagura_memory.auth.cli.poll_for_token", new_callable=AsyncMock)
@patch("kagura_memory.auth.cli.authorize_device", new_callable=AsyncMock)
@patch("kagura_memory.auth.cli.make_oauth_client")
def test_login_does_not_print_pre_login_tip(
    mock_client_factory,
    mock_authorize,
    mock_poll,
    mock_open,
    patched_default_path: Path,
):
    # memory-cloud#772 login-gated /device — the client-side workaround must not reappear.
    mock_client_factory.return_value = _async_ctx()
    mock_authorize.return_value = _mock_device_response()
    mock_poll.return_value = _mock_token_response()

    result = CliRunner().invoke(main, ["auth", "login"])
    assert result.exit_code == 0
    output = result.output
    assert "sign in to the Kagura web UI" not in output
    assert "the consent page assumes" not in output
    assert "https://test.example.com/device?user_code=" in output


# ---------------------------------------------------------------------------
# _try_open_browser + _is_wsl helpers
# ---------------------------------------------------------------------------


def test_try_open_browser_returns_true_when_stdlib_succeeds(monkeypatch):
    from kagura_memory.auth import cli as cli_mod

    monkeypatch.setattr(cli_mod, "_is_wsl", lambda: False)
    with patch.object(cli_mod.webbrowser, "open", return_value=True) as mock_open:
        assert cli_mod._try_open_browser("https://example.com/d?u=ABC") is True
    mock_open.assert_called_once()


def test_try_open_browser_skips_stdlib_on_wsl_even_when_it_would_succeed(monkeypatch):
    """Regression: on WSL, webbrowser.open's True return is not trusted —
    Python sees a registered handler but no Windows browser actually launches.
    The platform fallback must run regardless.
    """
    from kagura_memory.auth import cli as cli_mod

    monkeypatch.setattr(cli_mod, "_is_wsl", lambda: True)
    monkeypatch.setattr(cli_mod.shutil, "which", lambda name: f"/usr/bin/{name}")
    popen_calls: list[list[str]] = []

    def fake_popen(cmd, **kwargs):
        popen_calls.append(cmd)
        return MagicMock()

    with (
        patch.object(cli_mod.webbrowser, "open", return_value=True) as mock_open,
        patch.object(cli_mod.subprocess, "Popen", side_effect=fake_popen),
    ):
        assert cli_mod._try_open_browser("https://example.com/d?u=ABC") is True
    mock_open.assert_not_called()
    assert popen_calls[0][0] == "wslview"


def test_try_open_browser_passes_url_with_shell_metas_safely_via_rundll32(monkeypatch):
    """A URL with shell metacharacters is passed to rundll32 verbatim — argv
    delivery to a non-shell Windows binary keeps the characters literal."""
    from kagura_memory.auth import cli as cli_mod

    monkeypatch.setattr(cli_mod, "_is_wsl", lambda: True)
    monkeypatch.setattr(cli_mod.shutil, "which", lambda name: None)  # no wslview
    popen_calls: list[list[str]] = []

    def fake_popen(cmd, **kwargs):
        popen_calls.append(cmd)
        return MagicMock()

    # URL contains '&' (cmd.exe command chaining) — rundll32 is not a shell so this is safe.
    url_with_amp = "https://example.com/device?user_code=AB&next=/foo"
    with (
        patch.object(cli_mod.webbrowser, "open", return_value=False),
        patch.object(cli_mod.subprocess, "Popen", side_effect=fake_popen),
    ):
        assert cli_mod._try_open_browser(url_with_amp) is True
    assert popen_calls[0] == ["rundll32.exe", "url.dll,FileProtocolHandler", url_with_amp]


def test_try_open_browser_falls_back_to_wsl_opener_when_stdlib_fails(monkeypatch):
    from kagura_memory.auth import cli as cli_mod

    monkeypatch.setattr(cli_mod, "_is_wsl", lambda: True)
    monkeypatch.setattr(cli_mod.shutil, "which", lambda name: f"/usr/bin/{name}")
    popen_calls: list[list[str]] = []

    def fake_popen(cmd, **kwargs):
        popen_calls.append(cmd)
        return MagicMock()

    with (
        patch.object(cli_mod.webbrowser, "open", return_value=False),
        patch.object(cli_mod.subprocess, "Popen", side_effect=fake_popen),
    ):
        assert cli_mod._try_open_browser("https://example.com/d?u=ABC") is True
    # wslview was the first fallback tried.
    assert popen_calls[0][0] == "wslview"
    # URL is passed as a separate argv element — never via shell.
    assert "https://example.com/d?u=ABC" in popen_calls[0]


def test_try_open_browser_falls_back_to_rundll32_when_wslview_missing(monkeypatch):
    from kagura_memory.auth import cli as cli_mod

    monkeypatch.setattr(cli_mod, "_is_wsl", lambda: True)
    monkeypatch.setattr(cli_mod.shutil, "which", lambda name: None)
    popen_calls: list[list[str]] = []

    def fake_popen(cmd, **kwargs):
        popen_calls.append(cmd)
        return MagicMock()

    with (
        patch.object(cli_mod.webbrowser, "open", return_value=False),
        patch.object(cli_mod.subprocess, "Popen", side_effect=fake_popen),
    ):
        assert cli_mod._try_open_browser("https://example.com/d?u=ABC") is True
    assert popen_calls[0] == [
        "rundll32.exe",
        "url.dll,FileProtocolHandler",
        "https://example.com/d?u=ABC",
    ]


def test_try_open_browser_returns_false_when_no_fallback_available(monkeypatch):
    """Linux native without xdg-open: nothing to fall back to."""
    from kagura_memory.auth import cli as cli_mod

    monkeypatch.setattr(cli_mod, "_is_wsl", lambda: False)
    monkeypatch.setattr(cli_mod.sys, "platform", "linux")
    monkeypatch.setattr(cli_mod.shutil, "which", lambda name: None)

    with patch.object(cli_mod.webbrowser, "open", return_value=False):
        assert cli_mod._try_open_browser("https://example.com/d?u=ABC") is False


def test_try_open_browser_macos_uses_open(monkeypatch):
    from kagura_memory.auth import cli as cli_mod

    monkeypatch.setattr(cli_mod, "_is_wsl", lambda: False)
    monkeypatch.setattr(cli_mod.sys, "platform", "darwin")
    popen_calls: list[list[str]] = []

    def fake_popen(cmd, **kwargs):
        popen_calls.append(cmd)
        return MagicMock()

    with (
        patch.object(cli_mod.webbrowser, "open", return_value=False),
        patch.object(cli_mod.subprocess, "Popen", side_effect=fake_popen),
    ):
        assert cli_mod._try_open_browser("https://example.com/d?u=ABC") is True
    assert popen_calls[0][0] == "open"


def test_try_open_browser_linux_uses_xdg_open(monkeypatch):
    from kagura_memory.auth import cli as cli_mod

    monkeypatch.setattr(cli_mod, "_is_wsl", lambda: False)
    monkeypatch.setattr(cli_mod.sys, "platform", "linux")
    monkeypatch.setattr(cli_mod.shutil, "which", lambda name: "/usr/bin/xdg-open")
    popen_calls: list[list[str]] = []

    def fake_popen(cmd, **kwargs):
        popen_calls.append(cmd)
        return MagicMock()

    with (
        patch.object(cli_mod.webbrowser, "open", return_value=False),
        patch.object(cli_mod.subprocess, "Popen", side_effect=fake_popen),
    ):
        assert cli_mod._try_open_browser("https://example.com/d?u=ABC") is True
    assert popen_calls[0][0] == "xdg-open"


def test_try_open_browser_recovers_when_webbrowser_raises(monkeypatch):
    """webbrowser.open raising webbrowser.Error must not crash the device flow."""
    from kagura_memory.auth import cli as cli_mod

    monkeypatch.setattr(cli_mod, "_is_wsl", lambda: False)
    monkeypatch.setattr(cli_mod.sys, "platform", "darwin")
    popen_calls: list[list[str]] = []

    def fake_popen(cmd, **kwargs):
        popen_calls.append(cmd)
        return MagicMock()

    with (
        patch.object(
            cli_mod.webbrowser, "open", side_effect=cli_mod.webbrowser.Error("no browser")
        ),
        patch.object(cli_mod.subprocess, "Popen", side_effect=fake_popen),
    ):
        assert cli_mod._try_open_browser("https://example.com/d?u=ABC") is True
    assert popen_calls[0][0] == "open"


def test_try_open_browser_continues_to_next_fallback_when_popen_fails(monkeypatch):
    """First fallback raising OSError must not stop the loop — try the next one."""
    from kagura_memory.auth import cli as cli_mod

    monkeypatch.setattr(cli_mod, "_is_wsl", lambda: True)
    monkeypatch.setattr(cli_mod.shutil, "which", lambda name: f"/usr/bin/{name}")
    popen_calls: list[list[str]] = []

    def fake_popen(cmd, **kwargs):
        popen_calls.append(cmd)
        if cmd[0] == "wslview":
            raise FileNotFoundError("wslview not actually installed")
        return MagicMock()

    with (
        patch.object(cli_mod.webbrowser, "open", return_value=False),
        patch.object(cli_mod.subprocess, "Popen", side_effect=fake_popen),
    ):
        assert cli_mod._try_open_browser("https://example.com/d?u=ABC") is True
    # First attempt (wslview) raised; second attempt (rundll32.exe) succeeded.
    assert [c[0] for c in popen_calls] == ["wslview", "rundll32.exe"]


def test_is_wsl_detects_via_env(monkeypatch):
    from kagura_memory.auth import cli as cli_mod

    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu-22.04")
    assert cli_mod._is_wsl() is True


def test_is_wsl_detects_via_proc_osrelease(monkeypatch, tmp_path):
    """When WSL_DISTRO_NAME is unset, fall back to reading /proc/sys/kernel/osrelease."""
    from kagura_memory.auth import cli as cli_mod

    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    fake_release = tmp_path / "osrelease"
    fake_release.write_text("5.15.167.4-microsoft-standard-WSL2\n")

    real_open = open

    def fake_open(path, *args, **kwargs):
        if path == "/proc/sys/kernel/osrelease":
            return real_open(fake_release, *args, **kwargs)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)
    assert cli_mod._is_wsl() is True


def test_is_wsl_returns_false_for_native_linux_kernel(monkeypatch, tmp_path):
    """A native Linux kernel string contains neither 'wsl' nor 'microsoft'."""
    from kagura_memory.auth import cli as cli_mod

    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    fake_release = tmp_path / "osrelease"
    fake_release.write_text("6.6.0-generic\n")

    real_open = open

    def fake_open(path, *args, **kwargs):
        if path == "/proc/sys/kernel/osrelease":
            return real_open(fake_release, *args, **kwargs)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)
    assert cli_mod._is_wsl() is False


def test_is_wsl_returns_false_when_no_signal(monkeypatch):
    from kagura_memory.auth import cli as cli_mod

    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    # Force open() to raise so we exit the function via the fallback path.
    fake_open = MagicMock(side_effect=OSError("no /proc/sys/kernel/osrelease"))
    monkeypatch.setattr("builtins.open", fake_open)
    assert cli_mod._is_wsl() is False


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
@patch("kagura_memory.auth.cli._try_open_browser", return_value=True)
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


# ---------------------------------------------------------------------------
# Helpers — _redact_token, _humanize_delta, _resolve_server
# ---------------------------------------------------------------------------


def test_redact_token_branches():
    from kagura_memory.auth.cli import _redact_token

    assert _redact_token("") == "<empty>"
    assert _redact_token("short") == "<redacted>"
    assert _redact_token("kagura_12345678abcdef") == "kagura_1...cdef"


def test_humanize_delta_branches():
    from kagura_memory.auth.cli import _humanize_delta

    # negative clamps to 0
    assert _humanize_delta(-100) == "0m"
    # < 1 hour
    assert _humanize_delta(125) == "2m"
    # hours
    assert _humanize_delta(3600 + 30 * 60) == "1h 30m"
    # days
    assert _humanize_delta(2 * 86400 + 3 * 3600 + 5 * 60) == "2d 3h 5m"


def test_resolve_server_explicit_overrides_env(monkeypatch):
    from kagura_memory.auth.cli import _resolve_server

    monkeypatch.setenv("KAGURA_MCP_URL", "https://env.example.com/mcp")
    assert _resolve_server("https://explicit.example.com") == "https://explicit.example.com"


def test_resolve_server_env_fallback(monkeypatch):
    from kagura_memory.auth.cli import _resolve_server

    monkeypatch.setenv("KAGURA_MCP_URL", "https://env.example.com/mcp")
    # Strips /mcp via base_url_from_mcp
    assert _resolve_server(None) == "https://env.example.com"


def test_resolve_server_default(monkeypatch):
    from kagura_memory.auth.cli import DEFAULT_SERVER, _resolve_server

    monkeypatch.delenv("KAGURA_MCP_URL", raising=False)
    assert _resolve_server(None) == DEFAULT_SERVER


def test_resolve_server_rejects_http():
    from kagura_memory.auth.cli import _resolve_server

    with pytest.raises(ValueError, match="HTTPS"):
        _resolve_server("http://evil.example.com")


# ---------------------------------------------------------------------------
# Error paths in `auth login` / `auth refresh` / `auth token`
# ---------------------------------------------------------------------------


@patch("kagura_memory.auth.cli.poll_for_token", new_callable=AsyncMock)
@patch("kagura_memory.auth.cli.authorize_device", new_callable=AsyncMock)
@patch("kagura_memory.auth.cli.make_oauth_client")
def test_login_denied_error_surfaces_with_login_hint(
    mock_client_factory,
    mock_authorize,
    mock_poll,
    patched_default_path: Path,
):
    from kagura_memory.exceptions import KaguraAuthDeniedError

    mock_client_factory.return_value = _async_ctx()
    mock_authorize.return_value = _mock_device_response()
    mock_poll.side_effect = KaguraAuthDeniedError("Authorization denied at the consent screen.")

    result = CliRunner().invoke(main, ["auth", "login", "--no-browser"])
    assert result.exit_code != 0
    assert "denied" in result.output.lower()
    assert "kagura auth login" in result.output


@patch("kagura_memory.auth.cli.poll_for_token", new_callable=AsyncMock)
@patch("kagura_memory.auth.cli.authorize_device", new_callable=AsyncMock)
@patch("kagura_memory.auth.cli.make_oauth_client")
def test_login_connection_error_surfaces(
    mock_client_factory,
    mock_authorize,
    mock_poll,
    patched_default_path: Path,
):
    from kagura_memory.exceptions import KaguraConnectionError

    mock_client_factory.return_value = _async_ctx()
    mock_authorize.return_value = _mock_device_response()
    mock_poll.side_effect = KaguraConnectionError("Lost connection while waiting for approval.")

    result = CliRunner().invoke(main, ["auth", "login", "--no-browser"])
    assert result.exit_code != 0
    assert "Lost connection" in result.output


def test_refresh_no_profile_errors(patched_default_path: Path):
    result = CliRunner().invoke(main, ["auth", "refresh"])
    assert result.exit_code != 0
    assert "kagura auth login" in result.output


@patch("kagura_memory.auth.cli.refresh_access_token", new_callable=AsyncMock)
@patch("kagura_memory.auth.cli.make_oauth_client")
def test_refresh_expired_error_propagates(
    mock_client_factory,
    mock_refresh,
    patched_default_path: Path,
):
    from kagura_memory.exceptions import KaguraAuthExpiredError

    mock_client_factory.return_value = _async_ctx()
    mock_refresh.side_effect = KaguraAuthExpiredError("Your login expired.")
    _seed_credentials(patched_default_path.parent.parent, _make_creds())

    result = CliRunner().invoke(main, ["auth", "refresh"])
    assert result.exit_code != 0
    assert "expired" in result.output.lower()


def test_token_no_profile_errors(patched_default_path: Path):
    result = CliRunner().invoke(main, ["auth", "token"])
    assert result.exit_code != 0
    assert "kagura auth login" in result.output


@patch("kagura_memory.auth.cli.refresh_access_token", new_callable=AsyncMock)
@patch("kagura_memory.auth.cli.make_oauth_client")
def test_token_refreshes_when_near_expiry(
    mock_client_factory,
    mock_refresh,
    patched_default_path: Path,
):
    """`auth token` should auto-refresh if creds are within REFRESH_SKEW_SEC of expiry."""
    mock_client_factory.return_value = _async_ctx()
    mock_refresh.return_value = _mock_token_response(access_token="atok-refreshed")
    near_expiry = datetime.now(UTC) + timedelta(minutes=2)  # < 5min skew
    _seed_credentials(
        patched_default_path.parent.parent,
        _make_creds(access_token="atok-old", expires_at=near_expiry),
    )

    result = CliRunner().invoke(main, ["auth", "token"])
    assert result.exit_code == 0
    assert result.stdout.strip().splitlines()[0] == "atok-refreshed"
    mock_refresh.assert_awaited()


@patch("kagura_memory.auth.cli.revoke_token", new_callable=AsyncMock)
@patch("kagura_memory.auth.cli.make_oauth_client")
def test_logout_warns_on_revoke_failure(
    mock_client_factory,
    mock_revoke,
    patched_default_path: Path,
):
    """When server-side revoke fails (network 5xx), local logout still succeeds with a warning."""
    mock_client_factory.return_value = _async_ctx()
    mock_revoke.return_value = False  # simulate revoke failure
    _seed_credentials(patched_default_path.parent.parent, _make_creds())

    result = CliRunner().invoke(main, ["auth", "logout"])
    assert result.exit_code == 0
    assert "Warning" in result.output or "revoke failed" in result.output
    # Local profile was still removed.
    cf = load_credentials_file(patched_default_path)
    assert cf.get_profile() is None


def test_logout_warns_about_env_var(monkeypatch, patched_default_path: Path):
    """`logout` should remind the user when KAGURA_API_KEY is set in env."""
    monkeypatch.setenv("KAGURA_API_KEY", "env-key")
    _seed_credentials(patched_default_path.parent.parent, _make_creds())

    with patch("kagura_memory.auth.cli.revoke_token", new_callable=AsyncMock) as mock_revoke:
        mock_revoke.return_value = True
        with patch("kagura_memory.auth.cli.make_oauth_client", return_value=_async_ctx()):
            result = CliRunner().invoke(main, ["auth", "logout"])

    assert result.exit_code == 0
    assert "KAGURA_API_KEY" in result.output
