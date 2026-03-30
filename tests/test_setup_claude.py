"""Tests for kagura setup claude command."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from kagura_memory.cli import main
from kagura_memory.setup_claude import (
    KAGURA_HOOK_MARKER,
    _check_gitignore,
    _install_hooks,
    _install_skills,
    _prompt_api_key,
    _prompt_mcp_url,
    _select_or_create_context,
    _write_kagura_config,
    _write_mcp_json,
)


@pytest.fixture()
def project_dir(tmp_path: Path) -> Path:
    """Create a temporary project directory."""
    return tmp_path


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def _has_kagura_hook(entries: list[dict]) -> bool:
    """Check if any hook entry contains a Kagura-managed hook."""
    return any(
        KAGURA_HOOK_MARKER in h.get("command", "")
        for entry in entries
        for h in entry.get("hooks", [])
    )


# =============================================================================
# Prompt Unit Tests
# =============================================================================


class TestPromptApiKey:
    def test_interactive_with_existing(self) -> None:
        """Interactive mode with existing key should show masked default."""
        with patch("kagura_memory.setup_claude.click") as mock_click:
            mock_click.prompt.return_value = "kagura_new_key"
            mock_click.ClickException = click.ClickException
            result = _prompt_api_key("kagura_12345678abcdef", non_interactive=False)
            assert result == "kagura_new_key"
            mock_click.prompt.assert_called_once()
            call_kwargs = mock_click.prompt.call_args
            assert "kagura_1...cdef" in str(call_kwargs)

    def test_interactive_without_existing(self) -> None:
        """Interactive mode without existing key should prompt with empty default."""
        with patch("kagura_memory.setup_claude.click") as mock_click:
            mock_click.prompt.return_value = "kagura_entered"
            mock_click.ClickException = click.ClickException
            result = _prompt_api_key(None, non_interactive=False)
            assert result == "kagura_entered"


class TestPromptMcpUrl:
    def test_interactive(self) -> None:
        """Interactive mode should prompt with default URL."""
        with patch("kagura_memory.setup_claude.click") as mock_click:
            mock_click.prompt.return_value = "http://custom:8080/mcp"
            result = _prompt_mcp_url(None, non_interactive=False)
            assert result == "http://custom:8080/mcp"


class TestSelectOrCreateContext:
    @patch("kagura_memory.setup_claude._create_context")
    def test_interactive_select_existing(self, mock_create: AsyncMock) -> None:
        """Interactive: user selects an existing context."""
        with patch("kagura_memory.setup_claude.click") as mock_click:
            mock_click.prompt.return_value = 1  # Select first context
            mock_click.echo = MagicMock()
            result = _select_or_create_context(
                {"contexts": [{"id": "ctx-abc", "name": "my-proj"}]},
                "key",
                "url",
                None,
                Path("/tmp/test"),
                non_interactive=False,
            )
            assert result == "ctx-abc"
            mock_create.assert_not_called()

    @patch("kagura_memory.setup_claude.asyncio")
    @patch("kagura_memory.setup_claude._create_context")
    def test_interactive_create_new(self, mock_create: AsyncMock, mock_asyncio: MagicMock) -> None:
        """Interactive: no contexts, user creates a new one."""
        mock_asyncio.run.return_value = {"id": "ctx-new", "name": "test"}
        with patch("kagura_memory.setup_claude.click") as mock_click:
            mock_click.prompt.side_effect = ["my-project", "A summary"]
            mock_click.echo = MagicMock()
            result = _select_or_create_context(
                {"contexts": []},
                "key",
                "url",
                None,
                Path("/tmp/test"),
                non_interactive=False,
            )
            assert result == "ctx-new"

    @patch("kagura_memory.setup_claude.asyncio")
    @patch("kagura_memory.setup_claude._create_context")
    def test_interactive_create_new_from_list(
        self, mock_create: AsyncMock, mock_asyncio: MagicMock
    ) -> None:
        """Interactive: contexts exist, user chooses 'create new'."""
        mock_asyncio.run.return_value = {"id": "ctx-brand-new", "name": "new-proj"}
        with patch("kagura_memory.setup_claude.click") as mock_click:
            # First prompt: select context (choose "create new" = 2)
            # Second prompt: context name
            # Third prompt: summary
            mock_click.prompt.side_effect = [2, "new-proj", ""]
            mock_click.echo = MagicMock()
            mock_click.IntRange = click.IntRange
            result = _select_or_create_context(
                {"contexts": [{"id": "ctx-1", "name": "existing"}]},
                "key",
                "url",
                None,
                Path("/tmp/test"),
                non_interactive=False,
            )
            assert result == "ctx-brand-new"


# =============================================================================
# File Writer Unit Tests
# =============================================================================


class TestWriteKaguraConfig:
    def test_creates_new(self, project_dir: Path) -> None:
        path = _write_kagura_config(project_dir, "kagura_key", "http://localhost:8080/mcp", "ctx-1")
        data = json.loads(path.read_text())
        assert data["api_key"] == "kagura_key"
        assert data["mcp_url"] == "http://localhost:8080/mcp"
        assert data["context_id"] == "ctx-1"

    def test_merges_existing(self, project_dir: Path) -> None:
        existing = {"api_key": "old", "model": "gpt-5.4-nano", "llm_api_key": "sk-xxx"}
        (project_dir / ".kagura.json").write_text(json.dumps(existing))

        _write_kagura_config(project_dir, "new_key", "http://new/mcp", "ctx-2")
        data = json.loads((project_dir / ".kagura.json").read_text())
        assert data["api_key"] == "new_key"
        assert data["model"] == "gpt-5.4-nano"  # preserved
        assert data["llm_api_key"] == "sk-xxx"  # preserved


class TestWriteMcpJson:
    def test_creates_new(self, project_dir: Path) -> None:
        path = _write_mcp_json(project_dir, "kagura_key", "http://localhost:8080/mcp")
        data = json.loads(path.read_text())
        server = data["mcpServers"]["kagura-memory"]
        assert server["type"] == "url"
        assert server["url"] == "http://localhost:8080/mcp"
        assert server["headers"]["Authorization"] == "Bearer kagura_key"

    def test_preserves_other_servers(self, project_dir: Path) -> None:
        existing = {"mcpServers": {"other-server": {"type": "stdio", "command": "npx other"}}}
        (project_dir / ".mcp.json").write_text(json.dumps(existing))

        _write_mcp_json(project_dir, "key", "http://mcp")
        data = json.loads((project_dir / ".mcp.json").read_text())
        assert "other-server" in data["mcpServers"]
        assert "kagura-memory" in data["mcpServers"]


class TestInstallHooks:
    def test_creates_new_settings(self, project_dir: Path) -> None:
        path = _install_hooks(project_dir, "ctx-1")
        data = json.loads(path.read_text())
        assert "SessionStart" in data["hooks"]
        assert "PostToolUse" in data["hooks"]
        session_cmd = data["hooks"]["SessionStart"][0]["hooks"][0]["command"]
        assert "ctx-1" in session_cmd

    def test_preserves_existing_hooks(self, project_dir: Path) -> None:
        claude_dir = project_dir / ".claude"
        claude_dir.mkdir()
        existing = {
            "permissions": {"allow": ["Bash(git *)"]},
            "hooks": {
                "PostToolUse": [
                    {
                        "matcher": "Write|Edit",
                        "hooks": [
                            {
                                "type": "command",
                                "command": 'uv run ruff format --quiet "$file_path"',
                            }
                        ],
                    }
                ]
            },
        }
        (claude_dir / "settings.json").write_text(json.dumps(existing))

        _install_hooks(project_dir, "ctx-1")
        data = json.loads((claude_dir / "settings.json").read_text())

        # Permissions preserved
        assert data["permissions"]["allow"] == ["Bash(git *)"]

        # Existing ruff hook preserved
        post_hooks = data["hooks"]["PostToolUse"]
        assert any(
            "ruff" in h.get("command", "") for entry in post_hooks for h in entry.get("hooks", [])
        )

        # Kagura hook added
        assert _has_kagura_hook(post_hooks)

    def test_idempotent(self, project_dir: Path) -> None:
        _install_hooks(project_dir, "ctx-1")
        _install_hooks(project_dir, "ctx-2")  # Run again with different context

        data = json.loads((project_dir / ".claude" / "settings.json").read_text())

        # Should have exactly 1 kagura hook per event, not duplicates
        for event in ["SessionStart", "PostToolUse"]:
            kagura_count = sum(
                1
                for entry in data["hooks"][event]
                for h in entry.get("hooks", [])
                if KAGURA_HOOK_MARKER in h.get("command", "")
            )
            assert kagura_count == 1, f"Expected 1 kagura hook for {event}, got {kagura_count}"

        # Both hooks should have updated to ctx-2
        session_cmd = data["hooks"]["SessionStart"][0]["hooks"][0]["command"]
        assert "ctx-2" in session_cmd

        post_cmds = [
            h.get("command", "")
            for entry in data["hooks"]["PostToolUse"]
            for h in entry.get("hooks", [])
            if KAGURA_HOOK_MARKER in h.get("command", "")
        ]
        assert post_cmds and "ctx-2" in post_cmds[0]


class TestInstallSkills:
    def test_installs_skills(self, project_dir: Path) -> None:
        paths = _install_skills(project_dir, "ctx-1")
        assert len(paths) == 2
        for p in paths:
            assert p.exists()
            content = p.read_text()
            assert "ctx-1" in content

    def test_skill_filenames(self, project_dir: Path) -> None:
        _install_skills(project_dir, "ctx-1")
        assert (project_dir / ".claude" / "commands" / "kagura-recall.md").exists()
        assert (project_dir / ".claude" / "commands" / "kagura-remember.md").exists()


class TestCheckGitignore:
    def test_not_in_gitignore(self, project_dir: Path) -> None:
        (project_dir / ".gitignore").write_text("node_modules/\n")
        assert _check_gitignore(project_dir) is False

    def test_in_gitignore(self, project_dir: Path) -> None:
        (project_dir / ".gitignore").write_text("node_modules/\n.kagura.json\n")
        assert _check_gitignore(project_dir) is True

    def test_no_gitignore(self, project_dir: Path) -> None:
        assert _check_gitignore(project_dir) is False


# =============================================================================
# CLI Integration Tests
# =============================================================================


def _setup_args(tmp_path: Path, **overrides: str) -> list[str]:
    """Build common setup claude CLI args."""
    args = [
        "setup",
        "claude",
        "--api-key",
        overrides.get("api_key", "kagura_test123"),
        "--mcp-url",
        overrides.get("mcp_url", "http://localhost:8080/mcp"),
        "--project-dir",
        str(tmp_path),
        "-y",
    ]
    if "context_id" in overrides:
        args.extend(["--context-id", overrides["context_id"]])
    return args


@patch("kagura_memory.setup_claude._create_context")
@patch("kagura_memory.setup_claude._test_connection")
@patch("kagura_memory.setup_claude.load_config")
def test_setup_claude_non_interactive(
    mock_config: MagicMock,
    mock_conn: AsyncMock,
    mock_create_ctx: AsyncMock,
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """Non-interactive mode with all flags should succeed without prompts."""
    mock_config.return_value = {}
    mock_conn.return_value = {"count": 0, "contexts": []}
    mock_create_ctx.return_value = {"id": "ctx-new-uuid", "name": "test-project"}

    result = runner.invoke(main, _setup_args(tmp_path))

    assert result.exit_code == 0, result.output
    assert "Setup complete!" in result.output

    # Verify files created
    assert (tmp_path / ".kagura.json").exists()
    assert (tmp_path / ".mcp.json").exists()
    assert (tmp_path / ".claude" / "settings.json").exists()
    assert (tmp_path / ".claude" / "commands" / "kagura-recall.md").exists()


@patch("kagura_memory.setup_claude._test_connection")
@patch("kagura_memory.setup_claude.load_config")
def test_setup_claude_with_existing_context(
    mock_config: MagicMock,
    mock_conn: AsyncMock,
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """With --context-id, should use existing context without prompts."""
    mock_config.return_value = {}
    mock_conn.return_value = {
        "count": 1,
        "contexts": [{"id": "ctx-existing", "name": "my-project"}],
    }

    result = runner.invoke(main, _setup_args(tmp_path, context_id="ctx-existing"))

    assert result.exit_code == 0, result.output
    data = json.loads((tmp_path / ".kagura.json").read_text())
    assert data["context_id"] == "ctx-existing"


@patch("kagura_memory.setup_claude._test_connection")
@patch("kagura_memory.setup_claude.load_config")
def test_setup_claude_connection_failure(
    mock_config: MagicMock,
    mock_conn: AsyncMock,
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """Connection failure should show helpful error and not write files."""
    from kagura_memory.exceptions import KaguraAuthError

    mock_config.return_value = {}
    mock_conn.side_effect = KaguraAuthError("Invalid API key")

    result = runner.invoke(main, _setup_args(tmp_path, api_key="kagura_bad"))

    assert result.exit_code != 0
    assert "Authentication failed" in result.output
    assert not (tmp_path / ".kagura.json").exists()


@patch("kagura_memory.setup_claude.load_config")
def test_setup_claude_non_interactive_missing_key(
    mock_config: MagicMock,
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """Non-interactive without API key should fail."""
    mock_config.return_value = {}

    result = runner.invoke(
        main,
        ["setup", "claude", "--project-dir", str(tmp_path), "-y"],
    )

    assert result.exit_code != 0
    assert "API key required" in result.output


@patch("kagura_memory.setup_claude._create_context")
@patch("kagura_memory.setup_claude._test_connection")
@patch("kagura_memory.setup_claude.load_config")
def test_setup_claude_preserves_existing_mcp_servers(
    mock_config: MagicMock,
    mock_conn: AsyncMock,
    mock_create_ctx: AsyncMock,
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """Should preserve existing MCP servers in .mcp.json."""
    mock_config.return_value = {}
    mock_conn.return_value = {"count": 0, "contexts": []}
    mock_create_ctx.return_value = {"id": "ctx-uuid", "name": "test"}

    existing_mcp = {"mcpServers": {"github": {"type": "stdio", "command": "npx github-mcp"}}}
    (tmp_path / ".mcp.json").write_text(json.dumps(existing_mcp))

    result = runner.invoke(main, _setup_args(tmp_path))

    assert result.exit_code == 0, result.output
    data = json.loads((tmp_path / ".mcp.json").read_text())
    assert "github" in data["mcpServers"]
    assert "kagura-memory" in data["mcpServers"]


@patch("kagura_memory.setup_claude._create_context")
@patch("kagura_memory.setup_claude._test_connection")
@patch("kagura_memory.setup_claude.load_config")
def test_setup_claude_gitignore_warning(
    mock_config: MagicMock,
    mock_conn: AsyncMock,
    mock_create_ctx: AsyncMock,
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """Should warn about .gitignore when .kagura.json not listed."""
    mock_config.return_value = {}
    mock_conn.return_value = {"count": 0, "contexts": []}
    mock_create_ctx.return_value = {"id": "ctx-uuid", "name": "test"}

    result = runner.invoke(main, _setup_args(tmp_path))

    assert result.exit_code == 0
    assert ".gitignore" in result.output


@patch("kagura_memory.setup_claude._test_connection")
@patch("kagura_memory.setup_claude.load_config")
def test_setup_claude_connection_error(
    mock_config: MagicMock,
    mock_conn: AsyncMock,
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """KaguraConnectionError should show server-not-running hint."""
    from kagura_memory.exceptions import KaguraConnectionError

    mock_config.return_value = {}
    mock_conn.side_effect = KaguraConnectionError("ECONNREFUSED")

    result = runner.invoke(main, _setup_args(tmp_path))

    assert result.exit_code != 0
    assert "Cannot connect" in result.output
    assert "docker compose up" in result.output


@patch("kagura_memory.setup_claude._test_connection")
@patch("kagura_memory.setup_claude.load_config")
def test_setup_claude_generic_connection_error(
    mock_config: MagicMock,
    mock_conn: AsyncMock,
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """Generic exception during connection should show error."""
    mock_config.return_value = {}
    mock_conn.side_effect = RuntimeError("unexpected")

    result = runner.invoke(main, _setup_args(tmp_path))

    assert result.exit_code != 0
    assert "Connection failed" in result.output


@patch("kagura_memory.setup_claude._create_context")
@patch("kagura_memory.setup_claude._test_connection")
@patch("kagura_memory.setup_claude.load_config")
def test_setup_claude_config_load_error(
    mock_config: MagicMock,
    mock_conn: AsyncMock,
    mock_create_ctx: AsyncMock,
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """Should gracefully handle broken .kagura.json."""
    mock_config.side_effect = ValueError("Invalid JSON in .kagura.json")
    mock_conn.return_value = {"count": 0, "contexts": []}
    mock_create_ctx.return_value = {"id": "ctx-uuid", "name": "test"}

    result = runner.invoke(main, _setup_args(tmp_path))

    assert result.exit_code == 0, result.output
    assert "Setup complete!" in result.output


@patch("kagura_memory.setup_claude._test_connection")
@patch("kagura_memory.setup_claude.load_config")
def test_setup_claude_rejects_malicious_context_id(
    mock_config: MagicMock,
    mock_conn: AsyncMock,
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """Should reject context_id with shell-special characters."""
    mock_config.return_value = {}
    mock_conn.return_value = {
        "count": 0,
        "contexts": [],
    }

    result = runner.invoke(
        main,
        _setup_args(tmp_path, context_id="; rm -rf /"),
    )

    assert result.exit_code != 0
    assert "Invalid context_id" in result.output
    assert not (tmp_path / ".kagura.json").exists()
