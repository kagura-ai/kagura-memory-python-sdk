"""Tests for the OAuth / kagura-mcp path of `kagura setup claude` (#157, PR-A).

Covers the stdio `.mcp.json` writer, the `.mcp.json` mode detector (shared with
`kagura auth status`), the OAuth-specific `.kagura.json` writer, and the
`setup claude --profile` CLI flow.
"""

import json
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from kagura_memory.cli import main
from kagura_memory.setup_claude import (
    MCP_PROXY_COMMAND,
    MCP_SERVER_NAME,
    _create_context,
    _kagura_mcp_on_path,
    _make_client,
    _test_connection,
    _write_kagura_config_oauth,
    _write_mcp_json_stdio,
    detect_mcp_json_mode,
)


@pytest.fixture()
def project_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def _oauth_creds() -> MagicMock:
    creds = MagicMock()
    creds.mcp_url = "https://memory.kagura-ai.com/mcp"
    creds.server = "https://memory.kagura-ai.com"
    creds.user_email = "user@example.com"
    return creds


def _creds_file(creds: MagicMock | None, default_profile: str = "default") -> MagicMock:
    cf = MagicMock()
    cf.get_profile.return_value = creds
    cf.default_profile = default_profile
    return cf


# =============================================================================
# _write_mcp_json_stdio
# =============================================================================


class TestWriteMcpJsonStdio:
    def test_creates_stdio_form(self, project_dir: Path) -> None:
        path = _write_mcp_json_stdio(project_dir, "default")
        server = json.loads(path.read_text())["mcpServers"][MCP_SERVER_NAME]
        assert server["type"] == "stdio"
        assert server["command"] == MCP_PROXY_COMMAND
        assert server["args"] == ["--profile", "default"]
        # No secret is written in the stdio form.
        assert "headers" not in server

    def test_named_profile_in_args(self, project_dir: Path) -> None:
        path = _write_mcp_json_stdio(project_dir, "work")
        server = json.loads(path.read_text())["mcpServers"][MCP_SERVER_NAME]
        assert server["args"] == ["--profile", "work"]

    def test_preserves_other_servers(self, project_dir: Path) -> None:
        existing = {"mcpServers": {"github": {"type": "stdio", "command": "npx github-mcp"}}}
        (project_dir / ".mcp.json").write_text(json.dumps(existing))
        _write_mcp_json_stdio(project_dir, "default")
        servers = json.loads((project_dir / ".mcp.json").read_text())["mcpServers"]
        assert "github" in servers
        assert servers[MCP_SERVER_NAME]["type"] == "stdio"

    def test_overwrites_legacy_url_form(self, project_dir: Path) -> None:
        """Re-running with --profile migrates a legacy static-token entry to stdio."""
        existing = {
            "mcpServers": {
                MCP_SERVER_NAME: {
                    "type": "url",
                    "url": "https://x/mcp",
                    "headers": {"Authorization": "Bearer kagura_old"},
                }
            }
        }
        (project_dir / ".mcp.json").write_text(json.dumps(existing))
        _write_mcp_json_stdio(project_dir, "default")
        server = json.loads((project_dir / ".mcp.json").read_text())["mcpServers"][MCP_SERVER_NAME]
        assert server["type"] == "stdio"
        assert "headers" not in server  # stale token gone


# =============================================================================
# detect_mcp_json_mode
# =============================================================================


class TestDetectMcpJsonMode:
    def test_none_when_no_file(self, project_dir: Path) -> None:
        assert detect_mcp_json_mode(project_dir) == "none"

    def test_stdio(self, project_dir: Path) -> None:
        _write_mcp_json_stdio(project_dir, "default")
        assert detect_mcp_json_mode(project_dir) == "stdio"

    def test_static_token(self, project_dir: Path) -> None:
        existing = {
            "mcpServers": {
                MCP_SERVER_NAME: {
                    "type": "url",
                    "url": "https://x/mcp",
                    "headers": {"Authorization": "Bearer kagura_x"},
                }
            }
        }
        (project_dir / ".mcp.json").write_text(json.dumps(existing))
        assert detect_mcp_json_mode(project_dir) == "static-token"

    def test_static_token_case_insensitive_header(self, project_dir: Path) -> None:
        existing = {
            "mcpServers": {
                MCP_SERVER_NAME: {
                    "type": "url",
                    "url": "https://x/mcp",
                    "headers": {"authorization": "Bearer kagura_x"},
                }
            }
        }
        (project_dir / ".mcp.json").write_text(json.dumps(existing))
        assert detect_mcp_json_mode(project_dir) == "static-token"

    def test_url_without_auth(self, project_dir: Path) -> None:
        existing = {"mcpServers": {MCP_SERVER_NAME: {"type": "url", "url": "https://x/mcp"}}}
        (project_dir / ".mcp.json").write_text(json.dumps(existing))
        assert detect_mcp_json_mode(project_dir) == "url"

    def test_absent_when_no_kagura_entry(self, project_dir: Path) -> None:
        existing = {"mcpServers": {"github": {"type": "stdio", "command": "npx"}}}
        (project_dir / ".mcp.json").write_text(json.dumps(existing))
        assert detect_mcp_json_mode(project_dir) == "absent"

    def test_absent_on_malformed_json(self, project_dir: Path) -> None:
        (project_dir / ".mcp.json").write_text("{not valid json")
        assert detect_mcp_json_mode(project_dir) == "absent"

    def test_url_with_non_dict_headers_not_misclassified(self, project_dir: Path) -> None:
        """A malformed `headers` list must not be read as a static-token header."""
        existing = {
            "mcpServers": {
                MCP_SERVER_NAME: {
                    "type": "url",
                    "url": "https://x/mcp",
                    "headers": ["Authorization"],
                }
            }
        }
        (project_dir / ".mcp.json").write_text(json.dumps(existing))
        assert detect_mcp_json_mode(project_dir) == "url"

    def test_absent_on_unrecognized_type(self, project_dir: Path) -> None:
        """An entry whose `type` is neither stdio nor url is classified absent."""
        existing = {"mcpServers": {MCP_SERVER_NAME: {"type": "sse", "url": "https://x/sse"}}}
        (project_dir / ".mcp.json").write_text(json.dumps(existing))
        assert detect_mcp_json_mode(project_dir) == "absent"


# =============================================================================
# _make_client / _kagura_mcp_on_path (dispatch + PATH probe)
# =============================================================================


class TestMakeClient:
    @patch("kagura_memory.setup_claude.KaguraClient")
    def test_dispatches_to_profile(self, mock_kc: MagicMock) -> None:
        _make_client(None, None, "work")
        mock_kc.assert_called_once_with(profile="work")

    @patch("kagura_memory.setup_claude.KaguraClient")
    def test_dispatches_to_api_key(self, mock_kc: MagicMock) -> None:
        _make_client("kagura_x", "https://x/mcp", None)
        mock_kc.assert_called_once_with(api_key="kagura_x", mcp_url="https://x/mcp")


class TestKaguraMcpOnPath:
    def test_true_when_resolvable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/local/bin/kagura-mcp")
        assert _kagura_mcp_on_path() is True

    def test_false_when_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: None)
        assert _kagura_mcp_on_path() is False


def _async_client_mock(mock_kc: MagicMock, inner: AsyncMock) -> None:
    """Wire a patched KaguraClient so `async with KaguraClient(...) as c` yields `inner`."""
    mock_kc.return_value.__aenter__ = AsyncMock(return_value=inner)
    mock_kc.return_value.__aexit__ = AsyncMock(return_value=False)


class TestConnectionHelpers:
    @pytest.mark.asyncio
    @patch("kagura_memory.setup_claude.KaguraClient")
    async def test_test_connection_lists_contexts(self, mock_kc: MagicMock) -> None:
        inner = AsyncMock()
        inner.list_contexts.return_value = {"count": 2, "contexts": []}
        _async_client_mock(mock_kc, inner)
        out = await _test_connection(profile="work")
        assert out["count"] == 2
        mock_kc.assert_called_once_with(profile="work")

    @pytest.mark.asyncio
    @patch("kagura_memory.setup_claude.KaguraClient")
    async def test_create_context_creates(self, mock_kc: MagicMock) -> None:
        inner = AsyncMock()
        inner.create_context.return_value = {"context_id": "ctx-new"}
        _async_client_mock(mock_kc, inner)
        out = await _create_context(None, None, "proj", None, profile="work")
        assert out["context_id"] == "ctx-new"
        inner.create_context.assert_awaited_once_with(name="proj", summary=None)


# =============================================================================
# _write_kagura_config_oauth
# =============================================================================


class TestWriteKaguraConfigOAuth:
    def test_no_api_key_written(self, project_dir: Path) -> None:
        path = _write_kagura_config_oauth(project_dir, "https://x/mcp", "ctx-1")
        data = json.loads(path.read_text())
        assert "api_key" not in data
        assert data["mcp_url"] == "https://x/mcp"
        assert data["context_id"] == "ctx-1"

    def test_preserves_existing_keys(self, project_dir: Path) -> None:
        existing = {"api_key": "kagura_old", "model": "gpt-5.4-nano"}
        (project_dir / ".kagura.json").write_text(json.dumps(existing))
        _write_kagura_config_oauth(project_dir, "https://x/mcp", "ctx-2")
        data = json.loads((project_dir / ".kagura.json").read_text())
        # OAuth path never rewrites api_key, but also does not delete a pre-existing one
        # (the OAuth profile still wins in the credential-resolution order).
        assert data["api_key"] == "kagura_old"
        assert data["model"] == "gpt-5.4-nano"
        assert data["context_id"] == "ctx-2"


# =============================================================================
# CLI: setup claude --profile
# =============================================================================


def _profile_args(tmp_path: Path, **overrides: str) -> list[str]:
    args = [
        "setup",
        "claude",
        "--profile",
        overrides.get("profile", "default"),
        "--project-dir",
        str(tmp_path),
        "-y",
    ]
    if "context_id" in overrides:
        args.extend(["--context-id", overrides["context_id"]])
    return args


@patch("kagura_memory.setup_claude._kagura_mcp_on_path", return_value=True)
@patch("kagura_memory.setup_claude._test_connection")
@patch("kagura_memory.auth.credentials.load_credentials_file")
def test_setup_claude_profile_writes_stdio_form(
    mock_load: MagicMock,
    mock_conn: AsyncMock,
    mock_on_path: MagicMock,
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """`setup claude --profile` writes the stdio .mcp.json + api-key-less .kagura.json."""
    mock_load.return_value = _creds_file(_oauth_creds())
    mock_conn.return_value = {"count": 1, "contexts": [{"id": "ctx-x", "name": "proj"}]}

    result = runner.invoke(main, _profile_args(tmp_path, context_id="ctx-x"))

    assert result.exit_code == 0, result.output
    assert "kagura-mcp" in result.output

    server = json.loads((tmp_path / ".mcp.json").read_text())["mcpServers"][MCP_SERVER_NAME]
    assert server["type"] == "stdio"
    assert server["command"] == MCP_PROXY_COMMAND
    assert server["args"] == ["--profile", "default"]

    kagura_cfg = json.loads((tmp_path / ".kagura.json").read_text())
    assert "api_key" not in kagura_cfg
    assert kagura_cfg["context_id"] == "ctx-x"
    assert kagura_cfg["mcp_url"] == "https://memory.kagura-ai.com/mcp"

    # Hooks + skills still installed
    assert (tmp_path / ".claude" / "settings.json").exists()
    assert (tmp_path / ".claude" / "commands" / "kagura-recall.md").exists()


@patch("kagura_memory.auth.credentials.load_credentials_file")
def test_setup_claude_profile_missing_profile_errors(
    mock_load: MagicMock,
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """A nonexistent profile points the user at `kagura auth login`."""
    mock_load.return_value = _creds_file(None)

    result = runner.invoke(main, _profile_args(tmp_path, profile="ghost"))

    assert result.exit_code != 0
    assert "No OAuth profile 'ghost'" in result.output
    assert "kagura auth login --profile ghost" in result.output
    assert not (tmp_path / ".mcp.json").exists()


def test_setup_claude_profile_and_api_key_mutually_exclusive(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """--profile and --api-key cannot be combined."""
    result = runner.invoke(
        main,
        [
            "setup",
            "claude",
            "--profile",
            "default",
            "--api-key",
            "kagura_x",
            "--project-dir",
            str(tmp_path),
            "-y",
        ],
    )
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


@patch("kagura_memory.setup_claude._kagura_mcp_on_path", return_value=False)
@patch("kagura_memory.setup_claude._test_connection")
@patch("kagura_memory.auth.credentials.load_credentials_file")
def test_setup_claude_profile_warns_when_not_on_path(
    mock_load: MagicMock,
    mock_conn: AsyncMock,
    mock_on_path: MagicMock,
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """Missing `kagura-mcp` on $PATH warns but does not fail the setup."""
    mock_load.return_value = _creds_file(_oauth_creds())
    mock_conn.return_value = {"count": 1, "contexts": [{"id": "ctx-x", "name": "proj"}]}

    result = runner.invoke(main, _profile_args(tmp_path, context_id="ctx-x"))

    assert result.exit_code == 0, result.output
    assert "not found on $PATH" in result.output
    # Still wrote the config despite the warning
    assert (tmp_path / ".mcp.json").exists()


@patch("kagura_memory.setup_claude._kagura_mcp_on_path", return_value=True)
@patch("kagura_memory.setup_claude._test_connection")
@patch("kagura_memory.auth.credentials.load_credentials_file")
def test_setup_claude_non_default_profile_warns_hooks_use_default(
    mock_load: MagicMock,
    mock_conn: AsyncMock,
    mock_on_path: MagicMock,
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """A non-default profile warns that the CLI hooks still resolve the default profile."""
    mock_load.return_value = _creds_file(_oauth_creds(), default_profile="default")
    mock_conn.return_value = {"count": 1, "contexts": [{"id": "ctx-x", "name": "proj"}]}

    result = runner.invoke(main, _profile_args(tmp_path, profile="work", context_id="ctx-x"))

    assert result.exit_code == 0, result.output
    assert "DEFAULT profile 'default', not 'work'" in result.output
    assert "KAGURA_PROFILE=work" in result.output
    # The proxy entry still binds the named profile despite the hook caveat
    server = json.loads((tmp_path / ".mcp.json").read_text())["mcpServers"][MCP_SERVER_NAME]
    assert server["args"] == ["--profile", "work"]


@patch("kagura_memory.setup_claude._kagura_mcp_on_path", return_value=True)
@patch("kagura_memory.setup_claude._test_connection")
@patch("kagura_memory.auth.credentials.load_credentials_file")
def test_setup_claude_default_profile_no_hook_warning(
    mock_load: MagicMock,
    mock_conn: AsyncMock,
    mock_on_path: MagicMock,
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """The default profile must NOT emit the hook-profile mismatch warning."""
    mock_load.return_value = _creds_file(_oauth_creds(), default_profile="default")
    mock_conn.return_value = {"count": 1, "contexts": [{"id": "ctx-x", "name": "proj"}]}

    result = runner.invoke(main, _profile_args(tmp_path, profile="default", context_id="ctx-x"))

    assert result.exit_code == 0, result.output
    assert "DEFAULT profile" not in result.output


@patch("kagura_memory.setup_claude._kagura_mcp_on_path", return_value=True)
@patch("kagura_memory.setup_claude._test_connection")
@patch("kagura_memory.auth.credentials.load_credentials_file")
def test_setup_claude_profile_auth_failure(
    mock_load: MagicMock,
    mock_conn: AsyncMock,
    mock_on_path: MagicMock,
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """An auth error during the profile connection test surfaces a re-login hint."""
    from kagura_memory.exceptions import KaguraAuthError

    mock_load.return_value = _creds_file(_oauth_creds())
    mock_conn.side_effect = KaguraAuthError("token expired")

    result = runner.invoke(main, _profile_args(tmp_path, context_id="ctx-x"))

    assert result.exit_code != 0
    assert "Authentication failed" in result.output
    assert "kagura auth login --profile default" in result.output
    assert not (tmp_path / ".mcp.json").exists()


@patch("kagura_memory.setup_claude._kagura_mcp_on_path", return_value=True)
@patch("kagura_memory.setup_claude._test_connection")
@patch("kagura_memory.auth.credentials.load_credentials_file")
def test_setup_claude_profile_connection_error(
    mock_load: MagicMock,
    mock_conn: AsyncMock,
    mock_on_path: MagicMock,
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """A connection error names the profile's server and writes nothing."""
    from kagura_memory.exceptions import KaguraConnectionError

    mock_load.return_value = _creds_file(_oauth_creds())
    mock_conn.side_effect = KaguraConnectionError("ECONNREFUSED")

    result = runner.invoke(main, _profile_args(tmp_path, context_id="ctx-x"))

    assert result.exit_code != 0
    assert "Cannot connect to https://memory.kagura-ai.com" in result.output
    assert not (tmp_path / ".mcp.json").exists()


@patch("kagura_memory.setup_claude._kagura_mcp_on_path", return_value=True)
@patch("kagura_memory.setup_claude._test_connection")
@patch("kagura_memory.auth.credentials.load_credentials_file")
def test_setup_claude_profile_generic_error(
    mock_load: MagicMock,
    mock_conn: AsyncMock,
    mock_on_path: MagicMock,
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """An unexpected error surfaces as a generic connection failure."""
    mock_load.return_value = _creds_file(_oauth_creds())
    mock_conn.side_effect = RuntimeError("boom")

    result = runner.invoke(main, _profile_args(tmp_path, context_id="ctx-x"))

    assert result.exit_code != 0
    assert "Connection failed" in result.output
    assert not (tmp_path / ".mcp.json").exists()


@patch("kagura_memory.setup_claude._kagura_mcp_on_path", return_value=True)
@patch("kagura_memory.setup_claude._test_connection")
@patch("kagura_memory.auth.credentials.load_credentials_file")
def test_setup_claude_profile_rejects_malicious_context_id(
    mock_load: MagicMock,
    mock_conn: AsyncMock,
    mock_on_path: MagicMock,
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """The OAuth path rejects a shell-special context_id (parity with the api-key path)."""
    mock_load.return_value = _creds_file(_oauth_creds())
    mock_conn.return_value = {"count": 0, "contexts": []}

    result = runner.invoke(main, _profile_args(tmp_path, context_id="; rm -rf /"))

    assert result.exit_code != 0
    assert "Invalid context_id" in result.output
    assert not (tmp_path / ".mcp.json").exists()
