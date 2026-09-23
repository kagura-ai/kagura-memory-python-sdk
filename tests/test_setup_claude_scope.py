"""`kagura setup claude` scope, shadowing, URL query, plugin overlap and opt-outs (#258).

Claude Code's config is never the real one: the autouse ``_isolate_claude_code``
fixture (conftest.py) points ``CLAUDE_CONFIG_DIR`` — where the SDK reads
``~/.claude.json`` — at an empty temp directory and hides the ``claude`` CLI.
Tests that need ``claude`` install :class:`FakeClaude`, which fakes
``subprocess.run``.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from click.testing import CliRunner, Result

from kagura_memory import claude_code
from kagura_memory.claude_code import MCP_SERVER_NAME, claude_json_path
from kagura_memory.cli import main
from kagura_memory.setup_claude import (
    POSTTOOLUSE_HOOK_COMMAND,
    SESSIONSTART_HOOK_COMMAND,
    _install_hooks,
    _plugin_server_url,
)

CTX = "11111111-2222-3333-4444-555555555555"
MCP_URL = "https://memory.example.com/mcp"
API_KEY = "kagura_secret_key_1234"
PLUGIN_ID = "kagura-memory@kagura-memory-cloud"
STDIO_DEFAULT = {"type": "stdio", "command": "kagura-mcp", "args": ["--profile", "default"]}


class FakeClaude:
    """Stands in for the ``claude`` CLI by faking ``subprocess.run`` in claude_code.

    Records every ``claude`` argv (without the executable) and answers
    ``plugin list --json`` with ``plugins``; any other command exits
    ``returncode``.
    """

    def __init__(self, plugins: list[dict[str, Any]] | None = None, returncode: int = 0) -> None:
        self.plugins = plugins or []
        self.returncode = returncode
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert argv[0] == "/usr/bin/claude"
        args = argv[1:]
        self.calls.append(args)
        if args[:2] == ["plugin", "list"]:
            return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(self.plugins), stderr="")
        stderr = "claude failed" if self.returncode else ""
        return subprocess.CompletedProcess(argv, self.returncode, stdout="", stderr=stderr)

    def mcp_calls(self) -> list[list[str]]:
        return [c for c in self.calls if c[0] == "mcp"]


@pytest.fixture()
def fake_claude(monkeypatch: pytest.MonkeyPatch) -> FakeClaude:
    fake = FakeClaude()
    monkeypatch.setattr(claude_code, "claude_executable", lambda: "/usr/bin/claude")
    monkeypatch.setattr(claude_code.subprocess, "run", fake)
    return fake


@pytest.fixture()
def with_plugin(fake_claude: FakeClaude) -> FakeClaude:
    fake_claude.plugins = [{"id": PLUGIN_ID, "enabled": True, "scope": "user"}]
    return fake_claude


@pytest.fixture()
def connection(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Run both auth paths offline: a working OAuth profile and a server with one context."""
    creds = MagicMock()
    creds.mcp_url = MCP_URL
    creds.server = "https://memory.example.com"
    creds.user_email = "user@example.com"
    cf = MagicMock()
    cf.get_profile.return_value = creds
    cf.default_profile = "default"
    monkeypatch.setattr("kagura_memory.auth.credentials.load_credentials_file", lambda: cf)
    monkeypatch.setattr("kagura_memory.setup_claude._kagura_mcp_on_path", lambda: True)
    conn = AsyncMock(return_value={"count": 1, "contexts": [{"id": CTX, "name": "proj"}]})
    monkeypatch.setattr("kagura_memory.setup_claude._test_connection", conn)
    return conn


def run_setup(
    project: Path, *flags: str, oauth: bool = True, yes: bool = True, input: str | None = None
) -> Result:
    args = ["setup", "claude", "--project-dir", str(project), "--context-id", CTX]
    args += ["--profile", "default"] if oauth else ["--api-key", API_KEY, "--mcp-url", MCP_URL]
    if yes:
        args.append("-y")
    return CliRunner().invoke(main, [*args, *flags], input=input)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mcp_entry(project: Path) -> dict[str, Any]:
    return read_json(project / ".mcp.json")["mcpServers"][MCP_SERVER_NAME]


def write_claude_json(data: dict[str, Any]) -> Path:
    path = claude_json_path()
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def write_project_entry(project: Path, entry: dict[str, Any]) -> Path:
    path = project / ".mcp.json"
    path.write_text(json.dumps({"mcpServers": {MCP_SERVER_NAME: entry}}), encoding="utf-8")
    return path


def hook_commands(project: Path, event: str) -> list[str]:
    settings = read_json(project / ".claude" / "settings.json")
    return [h["command"] for entry in settings["hooks"].get(event, []) for h in entry["hooks"]]


# =============================================================================
# E. type "http"; A. --scope project stays the default
# =============================================================================


def test_default_scope_writes_http_entry_to_project(connection, tmp_path: Path) -> None:
    result = run_setup(tmp_path, oauth=False)

    assert result.exit_code == 0, result.output
    entry = mcp_entry(tmp_path)
    assert entry == {
        "type": "http",
        "url": MCP_URL,
        "headers": {"Authorization": f"Bearer {API_KEY}"},
    }
    assert "project scope" in result.output
    assert not claude_json_path().exists()


# =============================================================================
# A. --scope user — through `claude mcp add-json`, never by editing ~/.claude.json
# =============================================================================


def test_scope_user_calls_claude_mcp_add_json(connection, fake_claude, tmp_path: Path) -> None:
    result = run_setup(tmp_path, "--scope", "user", "--guardrails", "off")

    assert result.exit_code == 0, result.output
    [add] = fake_claude.mcp_calls()
    assert add[:5] == ["mcp", "add-json", "--scope", "user", MCP_SERVER_NAME]
    assert json.loads(add[5]) == {
        "type": "stdio",
        "command": "kagura-mcp",
        "args": ["--profile", "default", "--guardrails", "off"],
    }
    # The project keeps its context binding and hooks, but gets no .mcp.json.
    assert not (tmp_path / ".mcp.json").exists()
    assert read_json(tmp_path / ".kagura.json")["context_id"] == CTX
    assert (tmp_path / ".claude" / "settings.json").exists()
    assert not claude_json_path().exists()  # written by claude, never by the SDK
    assert "user scope" in result.output


def test_scope_user_api_key_entry_and_gitignore(connection, fake_claude, tmp_path: Path) -> None:
    result = run_setup(tmp_path, "--scope", "user", oauth=False)

    assert result.exit_code == 0, result.output
    [add] = fake_claude.mcp_calls()
    assert json.loads(add[5]) == {
        "type": "http",
        "url": MCP_URL,
        "headers": {"Authorization": f"Bearer {API_KEY}"},
    }
    assert "Add to .gitignore: .kagura.json" in result.output
    assert "Add to .gitignore: .mcp.json" not in result.output  # no .mcp.json written


@pytest.mark.parametrize("oauth", [True, False], ids=["oauth", "api-key"])
def test_scope_user_without_claude_prints_command_and_writes_nothing(
    connection, tmp_path: Path, oauth: bool
) -> None:
    """The autouse fixture hides ``claude``: print the command, stop, touch nothing."""
    result = run_setup(tmp_path, "--scope", "user", oauth=oauth)

    assert result.exit_code == 1
    assert "claude mcp add-json --scope user kagura-memory" in result.output
    assert "not found on PATH" in result.output
    assert API_KEY not in result.output  # the printed JSON carries a placeholder
    if not oauth:
        assert "<your-api-key>" in result.output
    assert not claude_json_path().exists()
    assert not (tmp_path / ".kagura.json").exists()
    assert not (tmp_path / ".mcp.json").exists()
    connection.assert_not_called()  # stopped before the server was contacted


def test_scope_user_identical_entry_needs_no_claude(connection, tmp_path: Path) -> None:
    path = write_claude_json({"mcpServers": {MCP_SERVER_NAME: STDIO_DEFAULT}})
    before = path.read_bytes()

    result = run_setup(tmp_path, "--scope", "user")

    assert result.exit_code == 0, result.output
    assert "already up to date" in result.output
    assert path.read_bytes() == before


def test_scope_user_replaces_a_different_user_entry(
    connection, fake_claude, tmp_path: Path
) -> None:
    """``claude mcp add-json`` refuses an existing name, so the old entry is removed first."""
    write_claude_json({"mcpServers": {MCP_SERVER_NAME: {**STDIO_DEFAULT, "args": ["x"]}}})

    result = run_setup(tmp_path, "--scope", "user")

    assert result.exit_code == 0, result.output
    assert [c[:4] for c in fake_claude.mcp_calls()] == [
        ["mcp", "remove", "--scope", "user"],
        ["mcp", "add-json", "--scope", "user"],
    ]


def test_scope_user_replacement_declined_interactively(
    connection, fake_claude, tmp_path: Path
) -> None:
    write_claude_json({"mcpServers": {MCP_SERVER_NAME: {**STDIO_DEFAULT, "args": ["x"]}}})

    result = run_setup(tmp_path, "--scope", "user", yes=False, input="n\n")

    assert result.exit_code == 1
    assert "Setup cancelled; nothing was written." in result.output
    assert fake_claude.mcp_calls() == []
    assert not (tmp_path / ".kagura.json").exists()


def test_scope_user_add_json_failure_is_reported_without_the_key(
    connection, fake_claude, tmp_path: Path
) -> None:
    fake_claude.returncode = 1

    result = run_setup(tmp_path, "--scope", "user", oauth=False)

    assert result.exit_code == 1
    assert "`claude mcp add-json` failed: claude failed" in result.output
    assert API_KEY not in result.output


# =============================================================================
# A. Shadow check — both directions, ~/.claude.json only read
# =============================================================================


def test_project_write_under_a_local_entry_is_refused_with_y(connection, tmp_path: Path) -> None:
    path = write_claude_json(
        {"projects": {str(tmp_path.resolve()): {"mcpServers": {MCP_SERVER_NAME: STDIO_DEFAULT}}}}
    )
    before = path.read_bytes()

    result = run_setup(tmp_path)

    assert result.exit_code == 1
    assert "local scope (~/.claude.json)" in result.output
    assert "claude mcp remove --scope local kagura-memory" in result.output
    assert not (tmp_path / ".mcp.json").exists()
    assert not (tmp_path / ".kagura.json").exists()
    assert path.read_bytes() == before
    connection.assert_not_called()


def test_user_write_under_a_project_entry_is_refused_with_y(
    connection, fake_claude, tmp_path: Path
) -> None:
    mcp_json = write_project_entry(tmp_path, STDIO_DEFAULT)
    before = mcp_json.read_bytes()

    result = run_setup(tmp_path, "--scope", "user")

    assert result.exit_code == 1
    assert "project scope (.mcp.json)" in result.output
    assert "claude mcp remove --scope project kagura-memory" in result.output
    assert fake_claude.mcp_calls() == []
    assert mcp_json.read_bytes() == before


def test_user_write_under_a_local_entry_is_refused_with_y(
    connection, fake_claude, tmp_path: Path
) -> None:
    write_claude_json(
        {"projects": {str(tmp_path.resolve()): {"mcpServers": {MCP_SERVER_NAME: STDIO_DEFAULT}}}}
    )

    result = run_setup(tmp_path, "--scope", "user")

    assert result.exit_code == 1
    assert "local scope" in result.output
    assert fake_claude.mcp_calls() == []


@pytest.mark.parametrize(("answer", "code"), [("\n", 1), ("y\n", 0)], ids=["default-no", "yes"])
def test_shadowed_write_asks_interactively(
    connection, fake_claude, tmp_path: Path, answer: str, code: int
) -> None:
    write_project_entry(tmp_path, STDIO_DEFAULT)

    result = run_setup(tmp_path, "--scope", "user", yes=False, input=answer)

    assert result.exit_code == code, result.output
    assert "Write the user-scope entry anyway? [y/N]" in result.output
    assert bool(fake_claude.mcp_calls()) == (code == 0)


def test_project_write_over_a_user_entry_notes_the_hidden_entry(connection, tmp_path: Path) -> None:
    path = write_claude_json({"mcpServers": {MCP_SERVER_NAME: STDIO_DEFAULT}})
    before = path.read_bytes()

    result = run_setup(tmp_path)

    assert result.exit_code == 0, result.output
    assert "hides the kagura-memory entry in user scope (~/.claude.json)" in result.output
    assert mcp_entry(tmp_path) == STDIO_DEFAULT
    assert path.read_bytes() == before


# =============================================================================
# B. --guardrails / --tool-profile
# =============================================================================


def test_stdio_entry_carries_query_flags_and_kagura_json_stays_plain(
    connection, tmp_path: Path
) -> None:
    result = run_setup(tmp_path, "--guardrails", "off", "--tool-profile", "core")

    assert result.exit_code == 0, result.output
    assert mcp_entry(tmp_path)["args"] == [
        "--profile",
        "default",
        "--guardrails",
        "off",
        "--tool-profile",
        "core",
    ]
    assert read_json(tmp_path / ".kagura.json")["mcp_url"] == MCP_URL


def test_static_token_url_carries_query_and_kagura_json_stays_plain(
    connection, tmp_path: Path
) -> None:
    result = run_setup(tmp_path, "--guardrails", CTX.upper(), "--tool-profile", "core", oauth=False)

    assert result.exit_code == 0, result.output
    assert mcp_entry(tmp_path)["url"] == f"{MCP_URL}?guardrails={CTX}&profile=core"
    assert read_json(tmp_path / ".kagura.json")["mcp_url"] == MCP_URL


@pytest.mark.parametrize("bad", ["on", "my-context"])
def test_guardrails_rejects_values_the_server_would_ignore(tmp_path: Path, bad: str) -> None:
    result = run_setup(tmp_path, "--guardrails", bad)

    assert result.exit_code == 2
    assert "'off' or a context UUID" in result.output
    assert not (tmp_path / ".mcp.json").exists()


def test_empty_tool_profile_is_rejected(tmp_path: Path) -> None:
    result = run_setup(tmp_path, "--tool-profile", " ")
    assert result.exit_code == 2


def test_help_documents_the_new_options() -> None:
    result = CliRunner().invoke(main, ["setup", "claude", "--help"])
    text = " ".join(result.output.split())
    for option in (
        "--scope [project|user]",
        "--guardrails",
        "--tool-profile",
        "--session-hook / --no-session-hook",
        "--sync-hook / --no-sync-hook",
        "--commands / --no-commands",
    ):
        assert option in text
    assert "get_context_info" in text


# =============================================================================
# D. Trusted-only SessionStart recall
# =============================================================================


def test_session_hook_is_trusted_only(tmp_path: Path) -> None:
    _install_hooks(tmp_path, CTX)
    [command] = hook_commands(tmp_path, "SessionStart")
    assert f"-c {CTX}" in command
    assert "--trusted-only" in command


def test_rerun_upgrades_the_pre_258_session_hook_in_place(tmp_path: Path) -> None:
    old = (
        'kagura recall "project context and recent decisions" '
        f"-c {CTX} -k 5 2>/dev/null | head -c 2000 || true"
    )
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / "settings.json").write_text(
        json.dumps({"hooks": {"SessionStart": [{"hooks": [{"type": "command", "command": old}]}]}})
    )

    _install_hooks(tmp_path, CTX)

    assert hook_commands(tmp_path, "SessionStart") == [
        SESSIONSTART_HOOK_COMMAND.format(context_id=CTX)
    ]


# =============================================================================
# C. Opt-outs
# =============================================================================


def test_no_session_hook_skips_only_the_session_hook(connection, tmp_path: Path) -> None:
    result = run_setup(tmp_path, "--no-session-hook")

    assert result.exit_code == 0, result.output
    settings = read_json(tmp_path / ".claude" / "settings.json")
    assert "SessionStart" not in settings["hooks"]
    assert hook_commands(tmp_path, "PostToolUse")
    assert (tmp_path / ".claude" / "commands" / "kagura-recall.md").exists()
    assert "at session start" not in result.output


def test_no_sync_hook_skips_only_the_sync_hook(connection, tmp_path: Path) -> None:
    result = run_setup(tmp_path, "--no-sync-hook")

    assert result.exit_code == 0, result.output
    settings = read_json(tmp_path / ".claude" / "settings.json")
    assert "PostToolUse" not in settings["hooks"]
    assert hook_commands(tmp_path, "SessionStart")
    assert "Sync .claude/memory/" not in result.output


def test_all_opt_outs_install_no_hooks_or_commands(connection, tmp_path: Path) -> None:
    result = run_setup(tmp_path, "--no-session-hook", "--no-sync-hook", "--no-commands")

    assert result.exit_code == 0, result.output
    assert not (tmp_path / ".claude" / "settings.json").exists()
    assert not (tmp_path / ".claude" / "commands").exists()
    assert mcp_entry(tmp_path) == STDIO_DEFAULT
    summary = result.output.split("Setup complete!")[1]
    assert "MCP server" in summary
    for item in ("session start", "Sync .claude/memory/", "/kagura-recall"):
        assert item not in summary


def _add_user_items(project: Path) -> None:
    """A user's own hooks and commands next to the SDK's, which must survive an opt-out."""
    path = project / ".claude" / "settings.json"
    settings = read_json(path)
    settings["hooks"]["SessionStart"].append(
        {"hooks": [{"type": "command", "command": 'kagura recall "my notes" -k 3'}]}
    )
    settings["hooks"]["PostToolUse"].append(
        {"matcher": "Write", "hooks": [{"type": "command", "command": "ruff format"}]}
    )
    path.write_text(json.dumps(settings))
    commands = project / ".claude" / "commands"
    (commands / "kagura-remember.md").write_text("my own /kagura-remember\n")
    (commands / "other.md").write_text("other\n")


def test_rerun_with_opt_outs_removes_only_sdk_entries(connection, tmp_path: Path) -> None:
    assert run_setup(tmp_path).exit_code == 0
    _add_user_items(tmp_path)

    result = run_setup(tmp_path, "--no-session-hook", "--no-sync-hook", "--no-commands")

    assert result.exit_code == 0, result.output
    assert hook_commands(tmp_path, "SessionStart") == ['kagura recall "my notes" -k 3']
    assert hook_commands(tmp_path, "PostToolUse") == ["ruff format"]
    commands = tmp_path / ".claude" / "commands"
    assert not (commands / "kagura-recall.md").exists()
    assert (commands / "kagura-remember.md").read_text() == "my own /kagura-remember\n"
    assert (commands / "other.md").exists()
    assert "Removed the SessionStart recall hook and .claude/memory sync hook" in result.output


def test_opt_out_drops_the_emptied_event(connection, tmp_path: Path) -> None:
    assert run_setup(tmp_path).exit_code == 0

    assert run_setup(tmp_path, "--no-sync-hook").exit_code == 0

    settings = read_json(tmp_path / ".claude" / "settings.json")
    assert "PostToolUse" not in settings["hooks"]
    prefix = POSTTOOLUSE_HOOK_COMMAND.split("{context_id}")[0]
    assert prefix not in json.dumps(settings)


@pytest.mark.parametrize(("answer", "kept"), [("n\n", True), ("\n", False)], ids=["no", "yes"])
def test_opt_out_removal_asks_first_interactively(
    connection, tmp_path: Path, answer: str, kept: bool
) -> None:
    assert run_setup(tmp_path).exit_code == 0

    result = run_setup(tmp_path, "--no-session-hook", yes=False, input=answer)

    assert result.exit_code == 0, result.output
    assert "Remove the SDK's SessionStart recall hook from .claude/settings.json?" in result.output
    assert bool(hook_commands(tmp_path, "SessionStart")) == kept


# =============================================================================
# C. kagura-memory plugin detection
# =============================================================================


@pytest.mark.parametrize(
    "plugins",
    [
        None,  # claude missing (autouse fixture)
        [{"id": "other@x", "enabled": True}],  # absent
        [{"id": PLUGIN_ID, "enabled": False}],  # disabled
    ],
    ids=["claude-missing", "absent", "disabled"],
)
def test_no_plugin_means_no_prompt_and_no_change(
    connection, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, plugins
) -> None:
    if plugins is not None:
        fake = FakeClaude(plugins)
        monkeypatch.setattr(claude_code, "claude_executable", lambda: "/usr/bin/claude")
        monkeypatch.setattr(claude_code.subprocess, "run", fake)

    # Interactive with no input: any prompt would abort the run.
    result = run_setup(tmp_path, yes=False, input="")

    assert result.exit_code == 0, result.output
    assert hook_commands(tmp_path, "SessionStart")
    assert hook_commands(tmp_path, "PostToolUse")
    assert (tmp_path / ".claude" / "commands" / "kagura-recall.md").exists()
    assert "server_url" not in result.output
    assert "plugin" not in result.output


def test_plugin_detected_interactive_default_skips_the_duplicates(
    connection, with_plugin, tmp_path: Path
) -> None:
    result = run_setup(tmp_path, yes=False, input="\n")

    assert result.exit_code == 0, result.output
    assert (
        "Skip the SDK's SessionStart recall hook and /kagura-recall, /kagura-remember? "
        "The plugin provides /kagura-memory:session-start, :recall and :remember. [Y/n]"
    ) in " ".join(result.output.split())
    settings = read_json(tmp_path / ".claude" / "settings.json")
    assert "SessionStart" not in settings["hooks"]
    assert hook_commands(tmp_path, "PostToolUse")  # the sync hook has no plugin counterpart
    assert not (tmp_path / ".claude" / "commands" / "kagura-recall.md").exists()
    # Never sets guardrails=off on the user's behalf; recommends it and prints the settings.
    assert mcp_entry(tmp_path) == STDIO_DEFAULT
    assert "--guardrails off" in result.output
    assert f"server_url  {MCP_URL}\n" in result.output
    assert f"context_id  {CTX}\n" in result.output


def test_plugin_detected_interactive_no_installs_as_before(
    connection, with_plugin, tmp_path: Path
) -> None:
    result = run_setup(tmp_path, yes=False, input="n\n")

    assert result.exit_code == 0, result.output
    assert hook_commands(tmp_path, "SessionStart")
    assert (tmp_path / ".claude" / "commands" / "kagura-recall.md").exists()


def test_plugin_detected_with_y_installs_as_before_and_names_the_flags(
    connection, with_plugin, tmp_path: Path
) -> None:
    result = run_setup(tmp_path)

    assert result.exit_code == 0, result.output
    assert "--no-session-hook --no-commands" in result.output
    assert hook_commands(tmp_path, "SessionStart")
    assert (tmp_path / ".claude" / "commands" / "kagura-recall.md").exists()
    assert mcp_entry(tmp_path) == STDIO_DEFAULT


def test_plugin_detected_explicit_flags_skip_the_prompt(
    connection, with_plugin, tmp_path: Path
) -> None:
    result = run_setup(tmp_path, "--session-hook", "--commands", yes=False, input="")

    assert result.exit_code == 0, result.output
    assert "Skip the SDK's" not in result.output
    assert hook_commands(tmp_path, "SessionStart")


def test_plugin_notes_with_guardrails_off_and_an_api_key(
    connection, with_plugin, tmp_path: Path
) -> None:
    result = run_setup(tmp_path, "--guardrails", "off", "--tool-profile", "core", oauth=False)

    assert result.exit_code == 0, result.output
    assert f"server_url  {MCP_URL}?guardrails=off\n" in result.output
    assert "re-run this setup with --guardrails off" not in result.output
    assert API_KEY not in result.output


@pytest.mark.parametrize(
    ("upstream", "expected"),
    [
        (f"{MCP_URL}/w/ws?profile=core&guardrails=off", f"{MCP_URL}/w/ws?guardrails=off"),
        (f"{MCP_URL}?guardrails={CTX}&profile=core", MCP_URL),
        (f"{MCP_URL}?guardrails=OFF", f"{MCP_URL}?guardrails=off"),
        (MCP_URL, MCP_URL),
    ],
)
def test_plugin_server_url_keeps_only_guardrails_off(upstream: str, expected: str) -> None:
    assert _plugin_server_url(upstream) == expected
