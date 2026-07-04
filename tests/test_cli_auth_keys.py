"""Tests for `kagura auth create-key|list-keys|revoke-key` CLI commands (#201)."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from kagura_memory.cli import main
from kagura_memory.models import MemberAPIKey

WS = "11111111-2222-3333-4444-555555555555"

PLAINTEXT = "kagura_secret_plaintext_value_0123456789"


@pytest.fixture(autouse=True)
def _isolate_oauth_state(isolated_kagura_credentials):
    """Every test here runs against isolated credentials (see conftest)."""


CONFIG = {
    "api_key": "kagura_test",
    "mcp_url": "https://test.com/mcp",
    "context_id": WS,
}


def _key(**overrides) -> MemberAPIKey:
    base = {
        "id": 42,
        "name": "ci-bot",
        "key_prefix": "kagura_abcdef123",
        "plaintext_key": PLAINTEXT,
        "is_visible": False,
        "created_at": datetime(2026, 7, 3, tzinfo=UTC),
        "expires_at": datetime(2026, 10, 1, tzinfo=UTC),
    }
    base.update(overrides)
    return MemberAPIKey(**base)


def _mock_client(mock_cls: MagicMock, **method_returns) -> MagicMock:
    inst = MagicMock()
    inst.__aenter__ = AsyncMock(return_value=inst)
    inst.__aexit__ = AsyncMock(return_value=None)
    for name, value in method_returns.items():
        setattr(inst, name, AsyncMock(return_value=value))
    mock_cls.return_value = inst
    mock_cls._from_resolved_auth.return_value = inst
    mock_cls.from_mcp_url.return_value = inst
    return inst


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.WorkspaceClient")
def test_create_key_prints_plaintext_once_with_warning(mock_cls, mock_config):
    mock_config.return_value = CONFIG
    inst = _mock_client(mock_cls, mint_member_key=_key())
    result = CliRunner().invoke(
        main,
        ["auth", "create-key", "--user", "google_2", "--name", "ci-bot", "--expires-days", "90"],
    )
    assert result.exit_code == 0, result.output
    assert PLAINTEXT in result.output
    assert "cannot be shown again" in result.output  # stderr warning
    assert "2026-10-01" in result.output  # expiry surfaced
    inst.mint_member_key.assert_awaited_once_with(WS, "google_2", "ci-bot", 90)


def test_create_key_requires_expires_days():
    result = CliRunner().invoke(
        main, ["auth", "create-key", "--user", "google_2", "--name", "ci-bot"]
    )
    assert result.exit_code != 0
    assert "--expires-days" in result.output


def test_create_key_rejects_out_of_range_expiry():
    result = CliRunner().invoke(
        main,
        [
            "auth",
            "create-key",
            "--user",
            "google_2",
            "--name",
            "ci-bot",
            "--expires-days",
            "4000",
        ],
    )
    assert result.exit_code != 0
    assert "4000" in result.output  # click.IntRange error


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.WorkspaceClient")
def test_list_keys_table_never_shows_plaintext(mock_cls, mock_config):
    mock_config.return_value = CONFIG
    inst = _mock_client(mock_cls, list_member_keys=[_key(plaintext_key=None)])
    result = CliRunner().invoke(main, ["auth", "list-keys", "--user", "google_2"])
    assert result.exit_code == 0, result.output
    assert "ci-bot" in result.output and "kagura_abcdef123" in result.output
    assert PLAINTEXT not in result.output
    inst.list_member_keys.assert_awaited_once_with(WS, "google_2")


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.WorkspaceClient")
def test_list_keys_json_excludes_plaintext_field(mock_cls, mock_config):
    mock_config.return_value = CONFIG
    # Even if a server ever leaked plaintext on list, the CLI drops the field.
    _mock_client(mock_cls, list_member_keys=[_key()])
    result = CliRunner().invoke(main, ["auth", "list-keys", "--user", "google_2", "--json"])
    assert result.exit_code == 0, result.output
    assert '"plaintext_key"' not in result.output
    assert PLAINTEXT not in result.output
    assert '"key_prefix": "kagura_abcdef123"' in result.output


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.WorkspaceClient")
def test_revoke_key_aborts_without_confirmation(mock_cls, mock_config):
    mock_config.return_value = CONFIG
    inst = _mock_client(mock_cls, revoke_member_key=None)
    result = CliRunner().invoke(
        main, ["auth", "revoke-key", "42", "--user", "google_2"], input="n\n"
    )
    assert result.exit_code != 0
    inst.revoke_member_key.assert_not_called()


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.WorkspaceClient")
def test_revoke_key_yes_skips_prompt(mock_cls, mock_config):
    mock_config.return_value = CONFIG
    inst = _mock_client(mock_cls, revoke_member_key=None)
    result = CliRunner().invoke(main, ["auth", "revoke-key", "42", "--user", "google_2", "--yes"])
    assert result.exit_code == 0, result.output
    assert "Revoked key #42" in result.output
    inst.revoke_member_key.assert_awaited_once_with(WS, "google_2", 42)
