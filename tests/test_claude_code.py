"""Tests for kagura_memory.claude_code — the read-only view of Claude Code's MCP config (#258).

``~/.claude.json`` is never the real one: the autouse ``_isolate_claude_code``
fixture (conftest.py) points ``CLAUDE_CONFIG_DIR`` at an empty temp directory
and hides the ``claude`` CLI.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from kagura_memory import claude_code
from kagura_memory.claude_code import (
    MCP_SERVER_NAME,
    McpEntry,
    classify_mcp_entry,
    claude_json_label,
    claude_json_path,
    detect_kagura_plugin,
    detect_mcp_json_mode,
    find_kagura_mcp_entries,
    mcp_add_json_args,
    same_mcp_entry,
)

# Bound at import, before the autouse fixture swaps it for a stub.
from kagura_memory.claude_code import claude_executable as real_claude_executable

_STDIO = {"type": "stdio", "command": "kagura-mcp", "args": ["--profile", "default"]}
_BEARER = {"type": "http", "url": "https://h/mcp", "headers": {"Authorization": "Bearer k"}}


def write_claude_json(data: dict[str, Any]) -> Path:
    """Write the isolated ``~/.claude.json`` the SDK reads (see conftest)."""
    path = claude_json_path()
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def write_mcp_json(project: Path, entry: dict[str, Any]) -> None:
    (project / ".mcp.json").write_text(json.dumps({"mcpServers": {MCP_SERVER_NAME: entry}}))


# =============================================================================
# claude_json_path
# =============================================================================


def test_claude_json_path_follows_claude_config_dir(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    assert claude_json_path() == tmp_path / ".claude.json"


def test_claude_json_path_defaults_to_home(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert claude_json_path() == tmp_path / ".claude.json"


# =============================================================================
# classify_mcp_entry
# =============================================================================


@pytest.mark.parametrize(
    ("entry", "mode"),
    [
        (_STDIO, "stdio"),
        ({"command": "kagura-mcp", "args": []}, "stdio"),  # Claude Code's default type
        ({**_STDIO, "command": "/home/u/.venv/bin/kagura-mcp"}, "stdio"),  # absolute path
        ({**_STDIO, "command": "C:\\venv\\Scripts\\kagura-mcp.exe"}, "stdio"),
        ({"command": "uvx", "args": ["--from", "kagura-memory", "kagura-mcp"]}, "stdio"),
        ({"type": "stdio", "command": "kagura-mcp-other"}, "absent"),
        ({"type": "stdio", "command": "uvx", "args": "kagura-mcp"}, "absent"),  # args not a list
        (_BEARER, "static-token"),
        ({**_BEARER, "type": "url"}, "static-token"),  # legacy SDK form
        ({**_BEARER, "type": "streamable-http"}, "static-token"),
        ({"type": "http", "url": "https://h/mcp"}, "url"),
        ({"type": "url", "url": "https://h/mcp"}, "url"),
        ({"type": "http", "url": "https://h/mcp", "headers": ["Authorization"]}, "url"),
        ({"type": "sse", "url": "https://h/sse"}, "absent"),
        ({"type": "stdio", "command": "other"}, "absent"),
        ("not-a-dict", "absent"),
    ],
)
def test_classify_mcp_entry(entry: object, mode: str) -> None:
    assert classify_mcp_entry(entry) == mode


def test_same_mcp_entry_ignores_empty_values() -> None:
    """``claude mcp add`` stores ``"env": {}``; that must not read as a different entry."""
    assert same_mcp_entry({**_STDIO, "env": {}}, _STDIO)
    assert not same_mcp_entry({**_STDIO, "env": {"A": "1"}}, _STDIO)
    assert not same_mcp_entry({**_STDIO, "args": ["--profile", "work"]}, _STDIO)


def test_legacy_type_flags_only_url() -> None:
    assert McpEntry("project", ".mcp.json", {**_BEARER, "type": "url"}).legacy_type
    assert not McpEntry("project", ".mcp.json", _BEARER).legacy_type


# =============================================================================
# find_kagura_mcp_entries / detect_mcp_json_mode
# =============================================================================


def test_finds_every_scope_strongest_first(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    write_mcp_json(project, _BEARER)
    write_claude_json(
        {
            "mcpServers": {MCP_SERVER_NAME: _STDIO},
            "projects": {str(project.resolve()): {"mcpServers": {MCP_SERVER_NAME: _STDIO}}},
        }
    )

    entries = find_kagura_mcp_entries(project)

    assert [(e.scope, e.source) for e in entries] == [
        ("local", claude_json_label()),
        ("project", ".mcp.json"),
        ("user", claude_json_label()),
    ]
    assert detect_mcp_json_mode(project) == "stdio"  # the local entry wins
    assert detect_mcp_json_mode(project, entries[1:]) == "static-token"  # entries reused


def _local_block(entry: dict[str, Any]) -> dict[str, Any]:
    return {"mcpServers": {MCP_SERVER_NAME: entry}}


def test_local_scope_is_keyed_by_the_git_root(tmp_path: Path) -> None:
    """Claude Code keys local scope by the repository root, not the subdirectory it runs in."""
    repo = tmp_path / "repo"
    sub = repo / "pkg" / "sub"
    sub.mkdir(parents=True)
    (repo / ".git").mkdir()
    stale = {**_STDIO, "args": ["--profile", "stale"]}
    write_claude_json(
        {
            "projects": {
                str(repo.resolve()): _local_block(_STDIO),
                str(sub.resolve()): _local_block(stale),
            }
        }
    )

    [entry] = find_kagura_mcp_entries(sub)

    assert entry.scope == "local"
    assert entry.config == _STDIO  # the root's block, not the ignored subdirectory one


def test_local_scope_of_a_linked_worktree_is_keyed_by_the_main_worktree(tmp_path: Path) -> None:
    main_tree = tmp_path / "main"
    git_dir = main_tree / ".git" / "worktrees" / "wt"
    git_dir.mkdir(parents=True)
    (git_dir / "commondir").write_text("../..\n")
    worktree = tmp_path / "wt"
    (worktree / "deep").mkdir(parents=True)
    (worktree / ".git").write_text(f"gitdir: {git_dir}\n")
    write_claude_json({"projects": {str(main_tree.resolve()): _local_block(_STDIO)}})

    assert [e.scope for e in find_kagura_mcp_entries(worktree / "deep")] == ["local"]


def test_local_scope_of_a_submodule_is_keyed_by_the_submodule(tmp_path: Path) -> None:
    """A ``.git`` file without ``commondir`` (a submodule) keys by its own directory."""
    module = tmp_path / "super" / "mod"
    module.mkdir(parents=True)
    (module / ".git").write_text("gitdir: ../.git/modules/mod\n")
    write_claude_json({"projects": {str(module.resolve()): _local_block(_STDIO)}})

    assert [e.scope for e in find_kagura_mcp_entries(module)] == ["local"]


def test_claude_json_label(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    assert claude_json_label() == "~/.claude.json"

    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "cfg"))
    assert claude_json_label() == "~/cfg/.claude.json"

    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/etc/claude")
    assert claude_json_label() == str(Path("/etc/claude") / ".claude.json")


def test_local_scope_of_another_project_is_ignored(tmp_path: Path) -> None:
    write_claude_json({"projects": {"/somewhere/else": {"mcpServers": {MCP_SERVER_NAME: _STDIO}}}})
    assert find_kagura_mcp_entries(tmp_path) == []
    assert detect_mcp_json_mode(tmp_path) == "none"


def test_user_scope_entry_is_detected_without_a_project_file(tmp_path: Path) -> None:
    write_claude_json({"mcpServers": {MCP_SERVER_NAME: _STDIO}})
    assert [e.scope for e in find_kagura_mcp_entries(tmp_path)] == ["user"]
    assert detect_mcp_json_mode(tmp_path) == "stdio"


def test_project_file_without_entry_is_absent_unless_another_scope_has_one(
    tmp_path: Path,
) -> None:
    (tmp_path / ".mcp.json").write_text(json.dumps({"mcpServers": {"github": {}}}))
    assert detect_mcp_json_mode(tmp_path) == "absent"

    write_claude_json({"mcpServers": {MCP_SERVER_NAME: _BEARER}})
    assert detect_mcp_json_mode(tmp_path) == "static-token"


@pytest.mark.parametrize("content", ["{not json", "[1, 2]", '{"mcpServers": []}'])
def test_malformed_claude_json_is_ignored(tmp_path: Path, content: str) -> None:
    claude_json_path().write_text(content, encoding="utf-8")
    assert find_kagura_mcp_entries(tmp_path) == []


def test_non_dict_entry_is_skipped(tmp_path: Path) -> None:
    write_claude_json({"mcpServers": {MCP_SERVER_NAME: "oops"}})
    assert find_kagura_mcp_entries(tmp_path) == []


def test_detection_never_writes_claude_json(tmp_path: Path) -> None:
    path = write_claude_json({"mcpServers": {MCP_SERVER_NAME: _STDIO}})
    before = (path.read_bytes(), os.stat(path).st_mtime_ns)
    find_kagura_mcp_entries(tmp_path)
    detect_mcp_json_mode(tmp_path)
    assert (path.read_bytes(), os.stat(path).st_mtime_ns) == before


# =============================================================================
# claude CLI helpers
# =============================================================================


def test_mcp_add_json_args() -> None:
    args = mcp_add_json_args("user", _STDIO)
    assert args[:5] == ["mcp", "add-json", "--scope", "user", MCP_SERVER_NAME]
    assert json.loads(args[5]) == _STDIO


def test_claude_executable_looks_up_claude_on_path(monkeypatch) -> None:
    monkeypatch.setattr(claude_code.shutil, "which", lambda name: f"/opt/bin/{name}")
    assert real_claude_executable() == "/opt/bin/claude"


def test_run_claude_without_claude_raises_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        claude_code.run_claude(["plugin", "list"])


def _plugin_list(monkeypatch, *, stdout: str = "", returncode: int = 0, exc=None) -> MagicMock:
    """Make ``claude plugin list --json`` answer ``stdout`` (or raise ``exc``)."""
    monkeypatch.setattr(claude_code, "claude_executable", lambda: "/usr/bin/claude")
    run = MagicMock(
        side_effect=exc,
        return_value=subprocess.CompletedProcess([], returncode, stdout=stdout, stderr=""),
    )
    monkeypatch.setattr(claude_code.subprocess, "run", run)
    return run


def _plugins(*rows: dict[str, Any]) -> str:
    return json.dumps(list(rows))


def test_plugin_present(monkeypatch, tmp_path: Path) -> None:
    run = _plugin_list(
        monkeypatch,
        stdout=_plugins(
            {"id": "other@x", "enabled": True},
            {"id": "kagura-memory@kagura-memory-cloud", "enabled": True, "scope": "user"},
        ),
    )
    assert detect_kagura_plugin(tmp_path) == "kagura-memory@kagura-memory-cloud"
    assert run.call_args.args[0] == ["/usr/bin/claude", "plugin", "list", "--json"]
    assert run.call_args.kwargs["cwd"] == tmp_path


@pytest.mark.parametrize(
    "stdout",
    [
        _plugins({"id": "other@x", "enabled": True}),  # absent
        _plugins({"id": "kagura-memory@kagura-memory-cloud", "enabled": False}),  # disabled
        _plugins({"id": "kagura-memory-extra@x", "enabled": True}),  # a different name
        _plugins({"id": "kagura-memory@x"}),  # no enabled field
        "not json",
        '{"plugins": []}',
        "[1, 2]",
    ],
    ids=["absent", "disabled", "other-name", "no-enabled", "garbage", "object", "non-dicts"],
)
def test_plugin_not_detected(monkeypatch, tmp_path: Path, stdout: str) -> None:
    _plugin_list(monkeypatch, stdout=stdout)
    assert detect_kagura_plugin(tmp_path) is None


def test_plugin_not_detected_when_claude_fails(monkeypatch, tmp_path: Path) -> None:
    _plugin_list(
        monkeypatch, stdout=_plugins({"id": "kagura-memory@x", "enabled": True}), returncode=1
    )
    assert detect_kagura_plugin(tmp_path) is None


@pytest.mark.parametrize(
    "exc", [subprocess.TimeoutExpired(["claude"], 30), PermissionError("denied")]
)
def test_plugin_not_detected_when_claude_errors(monkeypatch, tmp_path: Path, exc) -> None:
    _plugin_list(monkeypatch, exc=exc)
    assert detect_kagura_plugin(tmp_path) is None


def test_plugin_not_detected_without_claude(tmp_path: Path) -> None:
    """The autouse fixture hides ``claude``; detection must quietly say "no"."""
    assert detect_kagura_plugin(tmp_path) is None
