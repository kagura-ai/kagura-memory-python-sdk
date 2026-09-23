"""Read-only view of Claude Code's MCP configuration and plugins (issue #258).

Claude Code resolves an MCP server name by scope, strongest first, and uses
the whole entry from the winning scope with no merging:

* ``local``   — ``~/.claude.json`` → ``projects["<absolute project path>"].mcpServers``
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

#: Claude Code's precedence for servers of the same name, strongest first.
SCOPE_PRECEDENCE: tuple[McpScope, ...] = ("local", "project", "user")

# Remote types Claude Code accepts (``streamable-http`` is an alias of
# ``http``), plus ``url``, which SDKs before #258 wrote and Claude Code does
# not accept.
_HTTP_TYPES = frozenset({"http", "streamable-http", "url"})

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
    """Claude Code's global config: ``$CLAUDE_CONFIG_DIR`` or ``~``, then ``.claude.json``."""
    return Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home()) / ".claude.json"


def classify_mcp_entry(entry: object) -> str:
    """Classify a ``kagura-memory`` server entry.

    Returns one of:

    * ``"stdio"``        — refresh-aware ``kagura-mcp`` proxy (OAuth).
    * ``"static-token"`` — http form with a baked ``Authorization`` header.
    * ``"url"``          — http form **without** an ``Authorization`` header
      (e.g. Claude Code's own OAuth).
    * ``"absent"``       — anything else: not a form the SDK recognises.

    ``type`` ``http`` / ``streamable-http`` and the legacy ``url`` are treated
    alike, so entries written before #258 keep being recognised.
    """
    if not isinstance(entry, dict):
        return "absent"
    kind = entry.get("type")
    if kind == "stdio" and entry.get("command") == MCP_PROXY_COMMAND:
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
    #: Where it lives, for messages: ``.mcp.json`` or ``~/.claude.json``.
    source: str
    config: dict[str, Any]

    @property
    def mode(self) -> str:
        """See :func:`classify_mcp_entry`."""
        return classify_mcp_entry(self.config)

    @property
    def legacy_type(self) -> bool:
        """True for ``type: "url"``, which Claude Code does not accept."""
        return self.config.get("type") == "url"


def _local_servers(claude_json: dict[str, Any], project: Path) -> object:
    """The ``mcpServers`` of ``project``'s block in ``~/.claude.json``, if any."""
    projects = claude_json.get("projects")
    if not isinstance(projects, dict):
        return None
    # Claude Code keys the block by the project path; on Windows it may use
    # forward slashes.
    for key in (str(project), project.as_posix()):
        block = projects.get(key)
        if isinstance(block, dict):
            return block.get("mcpServers")
    return None


def find_kagura_mcp_entries(project_dir: Path) -> list[McpEntry]:
    """Every scope that defines ``kagura-memory`` for ``project_dir``, strongest first.

    The first element is the entry Claude Code uses; the rest are shadowed by
    it. Reads ``~/.claude.json`` and ``<project>/.mcp.json`` only.
    """
    project = project_dir.resolve()
    claude_json = _read_json_safe(claude_json_path())
    candidates: list[tuple[McpScope, str, object]] = [
        ("local", "~/.claude.json", _local_servers(claude_json, project)),
        ("project", ".mcp.json", _read_json_safe(project / ".mcp.json").get("mcpServers")),
        ("user", "~/.claude.json", claude_json.get("mcpServers")),
    ]
    entries: list[McpEntry] = []
    for scope, source, servers in candidates:
        entry = servers.get(MCP_SERVER_NAME) if isinstance(servers, dict) else None
        if isinstance(entry, dict):
            entries.append(McpEntry(scope, source, entry))
    return entries


def detect_mcp_json_mode(project_dir: Path) -> str:
    """Classify the ``kagura-memory`` entry Claude Code uses in ``project_dir``.

    Returns the :func:`classify_mcp_entry` mode of the **effective** entry
    (the strongest scope defining one, see :func:`find_kagura_mcp_entries`),
    or, when no scope defines one:

    * ``"absent"`` — the project has a ``.mcp.json`` (or an unreadable one)
      without a usable ``kagura-memory`` entry.
    * ``"none"``   — no ``.mcp.json`` and no entry in ``~/.claude.json``.

    Used by ``kagura auth status`` and ``kagura doctor``.
    """
    entries = find_kagura_mcp_entries(project_dir)
    if entries:
        return entries[0].mode
    return "absent" if (project_dir / ".mcp.json").exists() else "none"


def claude_executable() -> str | None:
    """Path of the Claude Code CLI (``claude``) on ``$PATH``, or None."""
    return shutil.which("claude")


def run_claude(args: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run ``claude <args>`` and capture its output.

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
    """Arguments of ``claude mcp add-json --scope <scope> kagura-memory '<json>'``."""
    return ["mcp", "add-json", "--scope", scope, MCP_SERVER_NAME, json.dumps(entry)]


def mcp_remove_args(scope: McpScope) -> list[str]:
    """Arguments of ``claude mcp remove --scope <scope> kagura-memory``."""
    return ["mcp", "remove", "--scope", scope, MCP_SERVER_NAME]


def detect_kagura_plugin(project_dir: Path) -> str | None:
    """Return the id of an installed, enabled ``kagura-memory`` plugin, or None.

    Runs ``claude plugin list --json`` in ``project_dir`` (project-scoped
    installs depend on it) and matches ``kagura-memory@<any marketplace>``.
    Never raises: a missing ``claude``, a failure or unparseable output all
    mean "not detected", so detection can never fail setup.
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
