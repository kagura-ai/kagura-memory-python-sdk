"""Read-only view of Claude Code's MCP configuration and plugins (issue #258).

Claude Code resolves an MCP server name by scope, strongest first, and uses
the whole entry from the winning scope with no merging:

* ``local``   — ``~/.claude.json`` → ``projects["<project key>"].mcpServers``, keyed
  by the git repository root (a linked worktree's main working tree) or, outside
  a repository, by the directory itself
* ``project`` — ``<project>/.mcp.json`` → ``mcpServers``
* ``user``    — ``~/.claude.json`` → ``mcpServers``

(then plugin-provided servers). An entry in a stronger scope therefore
silently shadows one of the same name in a weaker scope. ``kagura setup
claude``, ``kagura doctor`` and ``kagura auth status`` read all three here.

The SDK only ever **reads** ``~/.claude.json``: Claude Code owns that file
and rewrites it while running, so user-scope writes go through the ``claude``
CLI (``claude mcp add-json``) instead.
"""

from __future__ import annotations

import json
import ntpath
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

# Single source of truth for the Kagura MCP server entry. The writers in
# ``setup_claude`` and the classifier below key off these so the form that is
# written and the form that is detected can never drift apart.
MCP_SERVER_NAME = "kagura-memory"
MCP_PROXY_COMMAND = "kagura-mcp"

# memory-cloud's Claude Code plugin (``.claude-plugin/plugin.json``), listed
# by ``claude plugin list`` as ``kagura-memory@<marketplace>``.
KAGURA_PLUGIN_NAME = "kagura-memory"

McpScope = Literal["local", "project", "user"]

#: How a ``kagura-memory`` entry authenticates, see :func:`classify_mcp_entry`.
McpMode = Literal["stdio", "static-token", "url", "absent"]

#: Claude Code's precedence for servers of the same name, strongest first.
SCOPE_PRECEDENCE: tuple[McpScope, ...] = ("local", "project", "user")

# Remote types Claude Code accepts (``streamable-http`` is an alias of
# ``http``), plus ``url``, which SDKs before #258 wrote and Claude Code does
# not accept.
_HTTP_TYPES = frozenset({"http", "streamable-http", "url"})

# A stdio entry runs the proxy when one of these is the basename of its command
# or of an argument (an absolute path, or a launcher such as ``uvx``).
_PROXY_NAMES = frozenset({MCP_PROXY_COMMAND, f"{MCP_PROXY_COMMAND}.exe"})

_CLAUDE_TIMEOUT_SEC = 30


def _read_json_safe(path: Path) -> dict[str, Any]:
    """Read a UTF-8 JSON object, returning {} on missing/unreadable/parse error.

    Reads are pinned to UTF-8 so config is decoded identically on every
    locale (issue #197: the OS default codec is cp932 on Japanese Windows,
    which raised UnicodeDecodeError on UTF-8 content). A foreign-encoding,
    corrupt or non-object file falls back to an empty dict rather than
    crashing.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def claude_json_path() -> Path:
    """Claude Code's global config: ``$CLAUDE_CONFIG_DIR`` or ``~``, then ``.claude.json``.

    Returns:
        The path of the file Claude Code keeps user and local scope in.
    """
    return Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home()) / ".claude.json"


def claude_json_label() -> str:
    """:func:`claude_json_path` for messages, with the home directory written ``~``.

    Returns:
        ``~/.claude.json`` by default; the full path when ``$CLAUDE_CONFIG_DIR``
        points outside the home directory.
    """
    path = claude_json_path()
    try:
        return f"~/{path.relative_to(Path.home()).as_posix()}"
    except (ValueError, RuntimeError):
        return str(path)


def _runs_proxy(entry: dict[str, Any]) -> bool:
    """True when the command, or an argument of a launcher, is ``kagura-mcp``.

    Matches ``kagura-mcp`` by name or by path (``/venv/bin/kagura-mcp``, as a
    ``claude mcp add`` entry often has) and ``uvx … kagura-mcp …``, the way
    memory-cloud's ``/kagura-memory:setup`` recognises the proxy.
    """
    args = entry.get("args")
    argv = [entry.get("command"), *(args if isinstance(args, list) else [])]
    # ntpath splits on both "/" and "\", so Windows paths match too.
    return any(isinstance(a, str) and ntpath.basename(a) in _PROXY_NAMES for a in argv)


def classify_mcp_entry(entry: object) -> McpMode:
    """Classify a ``kagura-memory`` server entry.

    ``type`` ``http`` / ``streamable-http`` and the legacy ``url`` are treated
    alike, so entries written before #258 keep being recognised.

    Args:
        entry: The server's value under ``mcpServers``.

    Returns:
        * ``"stdio"``        — refresh-aware ``kagura-mcp`` proxy (OAuth).
        * ``"static-token"`` — http form with a baked ``Authorization`` header.
        * ``"url"``          — http form **without** an ``Authorization`` header
          (e.g. Claude Code's own OAuth).
        * ``"absent"``       — anything else: not a form the SDK recognises.
    """
    if not isinstance(entry, dict):
        return "absent"
    kind = entry.get("type")
    # Claude Code reads an entry without a type as stdio.
    if kind in (None, "stdio") and _runs_proxy(entry):
        return "stdio"
    if kind in _HTTP_TYPES:
        headers = entry.get("headers")
        if isinstance(headers, dict) and any(k.lower() == "authorization" for k in headers):
            return "static-token"
        return "url"
    return "absent"


@dataclass(frozen=True)
class McpEntry:
    """One ``kagura-memory`` definition Claude Code sees for a project."""

    scope: McpScope
    #: Where it lives, for messages: ``.mcp.json`` or :func:`claude_json_label`.
    source: str
    config: dict[str, Any]

    @property
    def mode(self) -> McpMode:
        """See :func:`classify_mcp_entry`."""
        return classify_mcp_entry(self.config)

    @property
    def legacy_type(self) -> bool:
        """True for ``type: "url"``, which Claude Code does not accept."""
        return self.config.get("type") == "url"


def _main_worktree(directory: Path, dot_git: Path) -> Path:
    """The main working tree of the linked worktree at ``directory``, else ``directory``.

    A linked worktree's ``.git`` file names its git dir, whose ``commondir``
    leads to the main repository's ``.git``. A submodule has no ``commondir``.
    """
    try:
        gitdir = dot_git.read_text(encoding="utf-8").strip().removeprefix("gitdir:").strip()
        git_path = (directory / gitdir).resolve()
        common = (git_path / (git_path / "commondir").read_text(encoding="utf-8").strip()).resolve()
    except (OSError, ValueError):
        return directory
    return common.parent if common.name == ".git" else directory


def _local_scope_key(project: Path) -> Path:
    """The path Claude Code keys ``project``'s local-scope block by.

    Inside a git repository that is the repository root (for a linked
    worktree, the main working tree), whichever subdirectory Claude Code runs
    in; elsewhere it is ``project`` itself. Checked against
    ``claude mcp add-json --scope local`` (Claude Code 2.1).
    """
    for directory in (project, *project.parents):
        dot_git = directory / ".git"
        if dot_git.is_dir():
            return directory
        if dot_git.is_file():
            return _main_worktree(directory, dot_git)
    return project


def _local_servers(claude_json: dict[str, Any], project: Path) -> object:
    """The ``mcpServers`` of ``project``'s block in ``~/.claude.json``, if any."""
    projects = claude_json.get("projects")
    if not isinstance(projects, dict):
        return None
    root = _local_scope_key(project)
    # On Windows the key may use forward slashes.
    for key in (str(root), root.as_posix()):
        block = projects.get(key)
        if isinstance(block, dict):
            return block.get("mcpServers")
    return None


def same_mcp_entry(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """True when two entries configure the same server.

    Keys with an empty value (``"env": {}`` as ``claude mcp add`` stores it)
    count as absent.

    Args:
        a: A server entry.
        b: Another server entry.

    Returns:
        Whether they differ only in empty values.
    """

    def significant(entry: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in entry.items() if v not in ({}, [], None)}

    return significant(a) == significant(b)


def find_kagura_mcp_entries(project_dir: Path) -> list[McpEntry]:
    """Every scope that defines ``kagura-memory`` for ``project_dir``, strongest first.

    Reads ``~/.claude.json`` and ``<project>/.mcp.json`` only.

    Args:
        project_dir: The project Claude Code runs in.

    Returns:
        The entries found. The first is the one Claude Code uses and the rest
        are shadowed by it; empty when no scope defines one.
    """
    project = project_dir.resolve()
    claude_json = _read_json_safe(claude_json_path())
    label = claude_json_label()
    candidates: list[tuple[McpScope, str, object]] = [
        ("local", label, _local_servers(claude_json, project)),
        ("project", ".mcp.json", _read_json_safe(project / ".mcp.json").get("mcpServers")),
        ("user", label, claude_json.get("mcpServers")),
    ]
    entries: list[McpEntry] = []
    for scope, source, servers in candidates:
        entry = servers.get(MCP_SERVER_NAME) if isinstance(servers, dict) else None
        if isinstance(entry, dict):
            entries.append(McpEntry(scope, source, entry))
    return entries


def detect_mcp_json_mode(
    project_dir: Path, entries: list[McpEntry] | None = None
) -> McpMode | Literal["none"]:
    """Classify the ``kagura-memory`` entry Claude Code uses in ``project_dir``.

    Used by ``kagura doctor``.

    Args:
        project_dir: The project Claude Code runs in.
        entries: :func:`find_kagura_mcp_entries` for ``project_dir`` when the
            caller already has them; read here otherwise.

    Returns:
        The :func:`classify_mcp_entry` mode of the **effective** entry (the
        strongest scope defining one), or, when no scope defines one:

        * ``"absent"`` — the project has a ``.mcp.json`` (or an unreadable
          one) without a usable ``kagura-memory`` entry.
        * ``"none"``   — no ``.mcp.json`` and no entry in ``~/.claude.json``.
    """
    if entries is None:
        entries = find_kagura_mcp_entries(project_dir)
    if entries:
        return entries[0].mode
    return "absent" if (project_dir / ".mcp.json").exists() else "none"


def claude_executable() -> str | None:
    """Look up the Claude Code CLI.

    Returns:
        The path of ``claude`` on ``$PATH``, or None.
    """
    return shutil.which("claude")


def run_claude(args: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run ``claude <args>`` and capture its output.

    Args:
        args: The arguments after ``claude``.
        cwd: The directory to run in; ``claude`` resolves local and project
            scope from it.

    Returns:
        The finished process, whatever its exit code.

    Raises:
        FileNotFoundError: ``claude`` is not on ``$PATH``.
        subprocess.TimeoutExpired: It did not finish in time.
    """
    exe = claude_executable()
    if exe is None:
        raise FileNotFoundError("claude")
    return subprocess.run(
        [exe, *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=_CLAUDE_TIMEOUT_SEC,
        check=False,
    )


def mcp_add_json_args(scope: McpScope, entry: dict[str, Any]) -> list[str]:
    """Arguments of ``claude mcp add-json --scope <scope> kagura-memory '<json>'``.

    Args:
        scope: The scope to add the entry to.
        entry: The server entry, serialised as the last argument.

    Returns:
        The arguments after ``claude``.
    """
    return ["mcp", "add-json", "--scope", scope, MCP_SERVER_NAME, json.dumps(entry)]


def mcp_remove_args(scope: McpScope) -> list[str]:
    """Arguments of ``claude mcp remove --scope <scope> kagura-memory``.

    Args:
        scope: The scope to remove the entry from.

    Returns:
        The arguments after ``claude``.
    """
    return ["mcp", "remove", "--scope", scope, MCP_SERVER_NAME]


def detect_kagura_plugin(project_dir: Path) -> str | None:
    """Return the id of an installed, enabled ``kagura-memory`` plugin, or None.

    Runs ``claude plugin list --json`` in ``project_dir`` (project-scoped
    installs depend on it) and matches ``kagura-memory@<any marketplace>``.
    Never raises: a missing ``claude``, a failure or unparseable output all
    mean "not detected", so detection can never fail setup.

    Args:
        project_dir: The project to list plugins for.

    Returns:
        The plugin id, such as ``kagura-memory@kagura-memory-cloud``, or None.
    """
    try:
        proc = run_claude(["plugin", "list", "--json"], cwd=project_dir)
        plugins = json.loads(proc.stdout) if proc.returncode == 0 else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    if not isinstance(plugins, list):
        return None
    for plugin in plugins:
        if not isinstance(plugin, dict) or plugin.get("enabled") is not True:
            continue
        plugin_id = plugin.get("id")
        if isinstance(plugin_id, str) and plugin_id.partition("@")[0] == KAGURA_PLUGIN_NAME:
            return plugin_id
    return None
