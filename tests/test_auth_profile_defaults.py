"""Safer multi-profile defaults: `auth use`, ambiguity warning, opt-in strict (#203)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from click.testing import CliRunner

from kagura_memory._auth import _resolve_auth, reset_profile_warnings
from kagura_memory.auth import credentials as creds_mod
from kagura_memory.auth.cli import auth as auth_group
from kagura_memory.auth.credentials import (
    CredentialsFile,
    OAuthCredentials,
    load_credentials_file,
    reset_state_cache,
    save_credentials_file,
    set_default_profile,
)
from kagura_memory.exceptions import KaguraAuthError


def _creds(workspace_name: str = "ws") -> OAuthCredentials:
    return OAuthCredentials(
        server="https://test.example.com",
        mcp_url="https://test.example.com/mcp",
        client_id="kagura-cli",
        access_token="atok",
        refresh_token="rtok",
        token_type="Bearer",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        scope="memory:read",
        workspace_id="ws-id",
        workspace_name=workspace_name,
        user_email="u@example.com",
        issued_at=datetime.now(UTC),
    )


@pytest.fixture
def creds_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    p = tmp_path / "credentials.json"
    monkeypatch.setattr(creds_mod, "DEFAULT_CREDENTIALS_PATH", p)
    monkeypatch.delenv("KAGURA_API_KEY", raising=False)
    monkeypatch.delenv("KAGURA_PROFILE", raising=False)
    monkeypatch.delenv("KAGURA_REQUIRE_PROFILE", raising=False)
    reset_state_cache()
    reset_profile_warnings()
    return p


def _write_profiles(path: Path, names: list[str], default: str | None = None) -> None:
    cf = CredentialsFile()
    for n in names:
        cf.set_profile(n, _creds(workspace_name=f"ws-{n}"))
    if default:
        cf.default_profile = default
    save_credentials_file(cf, path)


# --- set_default_profile -----------------------------------------------------


def test_set_default_profile_sets_existing(creds_path: Path) -> None:
    _write_profiles(creds_path, ["work", "personal"], default="work")
    set_default_profile("personal")
    assert load_credentials_file(creds_path).default_profile == "personal"


def test_set_default_profile_raises_on_missing(creds_path: Path) -> None:
    _write_profiles(creds_path, ["work"])
    with pytest.raises(KeyError):
        set_default_profile("nope")


# --- ambiguity warning (B) ---------------------------------------------------


def test_warns_once_on_multi_profile_implicit_default(creds_path, caplog) -> None:
    _write_profiles(creds_path, ["work", "personal"], default="work")
    with caplog.at_level(logging.WARNING, logger="kagura_memory"):
        _resolve_auth(api_key=None, mcp_url=None, profile=None)
        _resolve_auth(api_key=None, mcp_url=None, profile=None)  # second call: deduped
    msgs = [r.getMessage() for r in caplog.records if "using profile 'work'" in r.getMessage()]
    assert len(msgs) == 1  # once per process, not per client construction
    assert "ws-work" in msgs[0]  # names the workspace


def test_no_warning_single_profile(creds_path, caplog) -> None:
    _write_profiles(creds_path, ["only"])
    with caplog.at_level(logging.WARNING, logger="kagura_memory"):
        _resolve_auth(api_key=None, mcp_url=None, profile=None)
    assert not [r for r in caplog.records if "using profile" in r.getMessage()]


def test_no_warning_when_profile_explicit(creds_path, caplog) -> None:
    _write_profiles(creds_path, ["work", "personal"], default="work")
    with caplog.at_level(logging.WARNING, logger="kagura_memory"):
        _resolve_auth(api_key=None, mcp_url=None, profile="personal")
    assert not [r for r in caplog.records if "using profile" in r.getMessage()]


def test_no_warning_when_env_profile_set(creds_path, caplog, monkeypatch) -> None:
    _write_profiles(creds_path, ["work", "personal"], default="work")
    monkeypatch.setenv("KAGURA_PROFILE", "personal")
    with caplog.at_level(logging.WARNING, logger="kagura_memory"):
        _resolve_auth(api_key=None, mcp_url=None, profile=None)
    assert not [r for r in caplog.records if "using profile" in r.getMessage()]


# --- strict mode (C) ---------------------------------------------------------


def test_strict_mode_raises_on_ambiguous(creds_path, monkeypatch) -> None:
    _write_profiles(creds_path, ["work", "personal"], default="work")
    monkeypatch.setenv("KAGURA_REQUIRE_PROFILE", "1")
    with pytest.raises(KaguraAuthError, match="KAGURA_REQUIRE_PROFILE"):
        _resolve_auth(api_key=None, mcp_url=None, profile=None)


def test_strict_mode_allows_single_profile(creds_path, monkeypatch) -> None:
    _write_profiles(creds_path, ["only"])
    monkeypatch.setenv("KAGURA_REQUIRE_PROFILE", "1")
    result = _resolve_auth(api_key=None, mcp_url=None, profile=None)
    assert result.workspace_id == "ws-id"  # unambiguous → no raise


def test_strict_mode_allows_explicit_profile(creds_path, monkeypatch) -> None:
    _write_profiles(creds_path, ["work", "personal"], default="work")
    monkeypatch.setenv("KAGURA_REQUIRE_PROFILE", "1")
    result = _resolve_auth(api_key=None, mcp_url=None, profile="personal")
    assert result.workspace_id == "ws-id"


def test_strict_raise_not_suppressed_by_prior_warning(creds_path, caplog, monkeypatch) -> None:
    """A prior non-strict warning must not let the dedup set swallow a strict raise."""
    _write_profiles(creds_path, ["work", "personal"], default="work")
    with caplog.at_level(logging.WARNING, logger="kagura_memory"):
        _resolve_auth(api_key=None, mcp_url=None, profile=None)
    assert any("using profile 'work'" in r.getMessage() for r in caplog.records)
    monkeypatch.setenv("KAGURA_REQUIRE_PROFILE", "1")
    with pytest.raises(KaguraAuthError, match="KAGURA_REQUIRE_PROFILE"):
        _resolve_auth(api_key=None, mcp_url=None, profile=None)


def test_stale_default_profile_fails_closed(creds_path, monkeypatch) -> None:
    """default_profile pointing at a missing profile fails closed (characterization).

    With real profiles present but the default name stale, resolution finds no
    OAuth state and falls through; with no api_key/.kagura.json fallback it
    raises rather than silently picking another account.
    """
    _write_profiles(creds_path, ["work", "personal"], default="ghost")
    monkeypatch.setattr("kagura_memory._auth.load_config", lambda: {})
    with pytest.raises(KaguraAuthError, match="No credentials found"):
        _resolve_auth(api_key=None, mcp_url=None, profile=None)


# --- CLI: auth use -----------------------------------------------------------


def test_cli_auth_use_sets_default(creds_path) -> None:
    _write_profiles(creds_path, ["work", "personal"], default="work")
    result = CliRunner().invoke(auth_group, ["use", "personal"])
    assert result.exit_code == 0, result.output
    assert "Default profile set to 'personal'" in result.output
    assert load_credentials_file(creds_path).default_profile == "personal"


def test_cli_auth_use_unknown_profile_errors(creds_path) -> None:
    _write_profiles(creds_path, ["work"])
    result = CliRunner().invoke(auth_group, ["use", "ghost"])
    assert result.exit_code != 0
    assert "not found" in result.output
    assert "work" in result.output  # lists available profiles


def test_cli_auth_use_notes_env_profile_override(creds_path, monkeypatch) -> None:
    """When KAGURA_PROFILE is set, `auth use` warns it overrides the new default."""
    _write_profiles(creds_path, ["work", "personal"], default="work")
    monkeypatch.setenv("KAGURA_PROFILE", "work")
    result = CliRunner().invoke(auth_group, ["use", "personal"])
    assert result.exit_code == 0, result.output
    assert "KAGURA_PROFILE" in result.output
    assert "overrides" in result.output


def test_cli_auth_use_handles_concurrent_delete(creds_path, monkeypatch) -> None:
    """A logout racing the locked write (KeyError) surfaces a clean CLI error."""
    _write_profiles(creds_path, ["work", "personal"], default="work")

    def _raise_keyerror(_name: str) -> None:
        raise KeyError(_name)

    monkeypatch.setattr("kagura_memory.auth.cli.set_default_profile", _raise_keyerror)
    result = CliRunner().invoke(auth_group, ["use", "personal"])
    assert result.exit_code != 0
    assert "no longer exists" in result.output
    assert "Traceback" not in result.output  # clean ClickException, not a crash
