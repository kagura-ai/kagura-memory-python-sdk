"""Tests for `kagura workspace member|invite ...` CLI commands (#225)."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from kagura_memory.auth.credentials import reset_state_cache
from kagura_memory.cli import main
from kagura_memory.models import WorkspaceInvitation, WorkspaceMember

WS = "11111111-2222-3333-4444-555555555555"


@pytest.fixture(autouse=True)
def _isolate_oauth_state(tmp_path, monkeypatch):
    """Isolate from real ``~/.kagura/credentials.json`` and env.

    Same rationale as tests/test_cli_files.py: the workspace CLI walks the
    canonical SDK chain, so a developer's stored OAuth profile would
    pre-empt the config-only fixtures.
    """
    fake_path = tmp_path / "default-credentials.json"
    monkeypatch.setattr("kagura_memory.auth.credentials.DEFAULT_CREDENTIALS_PATH", fake_path)
    monkeypatch.delenv("KAGURA_API_KEY", raising=False)
    monkeypatch.delenv("KAGURA_PROFILE", raising=False)
    monkeypatch.delenv("KAGURA_MCP_URL", raising=False)
    reset_state_cache()
    yield
    reset_state_cache()


CONFIG = {
    "api_key": "kagura_test",
    "mcp_url": "https://test.com/mcp",
    "context_id": WS,
}


def _member(**overrides) -> WorkspaceMember:
    base = {
        "user_id": "google_1",
        "role": "owner",
        "user_email": "o@x.com",
        "user_name": "Owner",
        "joined_at": datetime(2026, 6, 1, tzinfo=UTC),
    }
    base.update(overrides)
    return WorkspaceMember(**base)


def _invitation(**overrides) -> WorkspaceInvitation:
    base = {
        "id": 7,
        "email": "new@x.com",
        "role": "member",
        "token": "tok_0123456789abcdef0123",
        "invitation_url": "https://memory.kagura-ai.com/invite/tok",
        "is_accepted": False,
        "is_expired": False,
        "expires_at": datetime(2026, 7, 10, tzinfo=UTC),
    }
    base.update(overrides)
    return WorkspaceInvitation(**base)


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


# ---------------------------------------------------------------------------
# member
# ---------------------------------------------------------------------------


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.WorkspaceClient")
def test_member_list_renders_table(mock_cls, mock_config):
    mock_config.return_value = CONFIG
    inst = _mock_client(
        mock_cls,
        list_members=[
            _member(),
            _member(user_id="google_2", role="member", user_email=None),
        ],
    )
    result = CliRunner().invoke(main, ["workspace", "member", "list"])
    assert result.exit_code == 0, result.output
    assert "google_1" in result.output and "owner" in result.output
    assert "google_2" in result.output
    inst.list_members.assert_awaited_once_with(WS)


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.WorkspaceClient")
def test_member_list_json(mock_cls, mock_config):
    mock_config.return_value = CONFIG
    _mock_client(mock_cls, list_members=[_member()])
    result = CliRunner().invoke(main, ["workspace", "member", "list", "--json"])
    assert result.exit_code == 0, result.output
    assert '"user_id": "google_1"' in result.output


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.WorkspaceClient")
def test_member_list_workspace_override(mock_cls, mock_config):
    mock_config.return_value = CONFIG
    other_ws = "99999999-8888-7777-6666-555555555555"
    inst = _mock_client(mock_cls, list_members=[])
    result = CliRunner().invoke(main, ["workspace", "member", "list", "--workspace", other_ws])
    assert result.exit_code == 0, result.output
    inst.list_members.assert_awaited_once_with(other_ws)


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.WorkspaceClient")
def test_member_add_forwards_role(mock_cls, mock_config):
    mock_config.return_value = CONFIG
    inst = _mock_client(
        mock_cls, add_member=_member(user_id="google_3", role="viewer", user_email=None)
    )
    result = CliRunner().invoke(
        main, ["workspace", "member", "add", "google_3", "--role", "viewer"]
    )
    assert result.exit_code == 0, result.output
    assert "Added google_3 as viewer" in result.output
    inst.add_member.assert_awaited_once_with(WS, "google_3", role="viewer")


def test_member_add_rejects_owner_role_at_cli():
    result = CliRunner().invoke(main, ["workspace", "member", "add", "google_3", "--role", "owner"])
    assert result.exit_code != 0
    assert "owner" in result.output  # click.Choice error names the invalid value


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.WorkspaceClient")
def test_member_set_role(mock_cls, mock_config):
    mock_config.return_value = CONFIG
    inst = _mock_client(
        mock_cls,
        update_member_role=_member(user_id="google_2", role="admin", user_email=None),
    )
    result = CliRunner().invoke(
        main, ["workspace", "member", "set-role", "google_2", "--role", "admin"]
    )
    assert result.exit_code == 0, result.output
    assert "google_2 is now admin" in result.output
    inst.update_member_role.assert_awaited_once_with(WS, "google_2", role="admin")


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.WorkspaceClient")
def test_member_remove_aborts_without_confirmation(mock_cls, mock_config):
    mock_config.return_value = CONFIG
    inst = _mock_client(mock_cls, remove_member=None)
    result = CliRunner().invoke(main, ["workspace", "member", "remove", "google_2"], input="n\n")
    assert result.exit_code != 0
    inst.remove_member.assert_not_called()


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.WorkspaceClient")
def test_member_remove_yes_skips_prompt(mock_cls, mock_config):
    mock_config.return_value = CONFIG
    inst = _mock_client(mock_cls, remove_member=None)
    result = CliRunner().invoke(main, ["workspace", "member", "remove", "google_2", "--yes"])
    assert result.exit_code == 0, result.output
    assert "Removed google_2" in result.output
    inst.remove_member.assert_awaited_once_with(WS, "google_2")


# ---------------------------------------------------------------------------
# invite
# ---------------------------------------------------------------------------


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.WorkspaceClient")
def test_invite_create_prints_url_once_with_warning(mock_cls, mock_config):
    mock_config.return_value = CONFIG
    inst = _mock_client(mock_cls, create_invitation=_invitation())
    result = CliRunner().invoke(
        main,
        [
            "workspace",
            "invite",
            "create",
            "new@x.com",
            "--role",
            "member",
            "--context",
            "ctx-1",
            "--expires-days",
            "7",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "https://memory.kagura-ai.com/invite/tok" in result.output
    assert "shown once" in result.output  # stderr warning (CliRunner merges streams)
    inst.create_invitation.assert_awaited_once_with(
        WS, "new@x.com", role="member", allowed_context_ids=["ctx-1"], expires_in_days=7
    )


def test_invite_create_rejects_non_preset_expiry_at_cli():
    result = CliRunner().invoke(
        main, ["workspace", "invite", "create", "a@b.com", "--expires-days", "14"]
    )
    assert result.exit_code != 0
    assert "14" in result.output  # click.Choice error


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.WorkspaceClient")
def test_invite_list_hides_token(mock_cls, mock_config):
    mock_config.return_value = CONFIG
    inst = _mock_client(mock_cls, list_invitations=[_invitation()])
    result = CliRunner().invoke(main, ["workspace", "invite", "list"])
    assert result.exit_code == 0, result.output
    assert "new@x.com" in result.output and "pending" in result.output
    assert "tok_0123456789abcdef0123" not in result.output
    assert "invite/tok" not in result.output
    inst.list_invitations.assert_awaited_once_with(WS, include_accepted=False)


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.WorkspaceClient")
def test_invite_list_json_drops_token_fields(mock_cls, mock_config):
    mock_config.return_value = CONFIG
    _mock_client(mock_cls, list_invitations=[_invitation()])
    result = CliRunner().invoke(main, ["workspace", "invite", "list", "--json"])
    assert result.exit_code == 0, result.output
    assert '"token"' not in result.output
    assert '"invitation_url"' not in result.output
    assert '"email": "new@x.com"' in result.output


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.WorkspaceClient")
def test_invite_list_include_accepted_flag(mock_cls, mock_config):
    mock_config.return_value = CONFIG
    inst = _mock_client(mock_cls, list_invitations=[])
    result = CliRunner().invoke(main, ["workspace", "invite", "list", "--include-accepted"])
    assert result.exit_code == 0, result.output
    inst.list_invitations.assert_awaited_once_with(WS, include_accepted=True)


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.WorkspaceClient")
def test_invite_revoke(mock_cls, mock_config):
    mock_config.return_value = CONFIG
    inst = _mock_client(mock_cls, revoke_invitation=None)
    result = CliRunner().invoke(main, ["workspace", "invite", "revoke", "7"])
    assert result.exit_code == 0, result.output
    assert "Revoked invitation #7" in result.output
    inst.revoke_invitation.assert_awaited_once_with(WS, 7)


# ---------------------------------------------------------------------------
# credential-source pairing (#115)
# ---------------------------------------------------------------------------


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.WorkspaceClient")
def test_env_key_without_workspace_names_the_workspace_flag(mock_cls, mock_config, monkeypatch):
    """env api_key has no same-source workspace → actionable error naming --workspace."""
    mock_config.return_value = {}
    monkeypatch.setenv("KAGURA_API_KEY", "kagura_env_key")
    result = CliRunner().invoke(main, ["workspace", "member", "list"])
    assert result.exit_code != 0
    assert "--workspace" in result.output
    assert "--context-id" not in result.output
