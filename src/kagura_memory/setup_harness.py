"""``kagura setup codex|hermes|openclaw`` — MCP entries for other harnesses (#260).

memory-cloud's dynamic client registration accepts a loopback client only
when its name carries a known keyword, and OpenAI Codex, Hermes Agent and
OpenClaw register under names it does not know (memory-cloud#1657), so none
of them can sign in on its own. Each can spawn a stdio server, though, and
``kagura-mcp`` already forwards every JSON-RPC message with a fresh bearer
from the ``kagura auth login`` profile. The default entry therefore runs the
proxy; ``--url-form`` writes a URL entry that reads a long-lived API key
from an environment variable instead.

Rules the three commands share:

* **The harness owns its config file.** An entry is written only through
  the harness command documented for it (``codex mcp add``, ``hermes mcp
  add``, ``openclaw mcp add|set``). Without that CLI on ``PATH``, or when
  the command is interactive and there is no terminal (or ``-y``), setup
  prints the block and the file and edits nothing: the SDK has no TOML,
  YAML or JSON5 writer, and text appended to such a file can duplicate a
  key.
* **Filtered environments.** Each harness passes a filtered environment to
  the servers it spawns, so the stdio entry names ``kagura-mcp`` by
  absolute path and always passes ``--profile``; it never depends on
  ``PATH``, ``KAGURA_PROFILE`` or the credentials file's
  ``default_profile``. ``HOME`` needs nothing: ``Path.home()`` falls back
  to the password database when it is unset.
* **No secret is written, printed or passed on a command line.** The stdio
  entry holds none, and the URL form names the variable the key is read
  from. An existing entry is read only to say what kind it is; nothing read
  from it is echoed.
* **Prompts need a terminal.** Setup asks only with a terminal on stdin and
  without ``-y``; otherwise (an agent's shell, CI, ``</dev/null``) it runs
  as ``-y`` does and never stops at a prompt.
* **The entry's credential, on the entry's server.** The profile check, the
  context lookup and the export use the named profile alone, never
  ``KAGURA_API_KEY``, since the stdio entry's ``kagura-mcp`` uses nothing
  else. A URL form entry's context and export must come from a credential
  on its own server: a context id belongs to one deployment.
* **Guardrails.** Codex reads the MCP ``instructions``, so it gets the
  ``--guardrails`` lane. Hermes and OpenClaw do not: they rely on the
  ``guardrails`` block of ``get_context_info``, which ``--guardrails off``
  would remove, and on the opt-in ``AGENTS.md`` export.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tomllib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal
from urllib.parse import quote_plus, urlsplit, urlunsplit

import click

from ._auth import _SOURCE_LABEL, _OAuthAuth, _resolve_auth, _resolve_profile_auth, _StaticAuth
from ._guardrail_export import has_guardrail_block, write_guardrail_block
from ._http import (
    base_url_from_mcp,
    mcp_url_guardrails_off,
    mcp_url_with_query,
    normalize_uuid,
    validate_https_url,
)
from .auth.credentials import CredentialsFile
from .claude_code import MCP_PROXY_COMMAND, _path_label, _runs_proxy, holds_credential
from .exceptions import KaguraAuthError, KaguraNotFoundError, _exc_message
from .memory_client import MemoryClient
from .models import GuardrailDigest
from .setup_claude import _load_profile, _stdio_entry, _verify_profile

HarnessName = Literal["codex", "hermes", "openclaw"]

#: The variable the Codex and OpenClaw URL forms read the API key from, as in
#: memory-cloud's own setup docs. Hermes names its variable itself.
DEFAULT_KEY_ENV = "KAGURA_API_KEY"

# Codex, Hermes and OpenClaw all accept these; a dot would nest a TOML table.
_SERVER_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
# OpenClaw substitutes only upper-case ${VAR} names.
_ENV_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# `openclaw mcp add` starts the server to probe it before it saves.
_HARNESS_TIMEOUT_SEC = 120


def _harness_executable(cli: str) -> str | None:
    """The path of a harness CLI on ``$PATH``, or None."""
    return shutil.which(cli)


def _stdin_is_tty() -> bool:
    """True when setup can hand the terminal to an interactive harness command."""
    return sys.stdin.isatty()


def _proxy_path() -> str:
    """The absolute path of ``kagura-mcp``, for an entry run under a filtered ``PATH``.

    ``$PATH`` first, then the directory of the Python running this setup,
    where the ``kagura-memory`` package installs both console scripts.

    Raises:
        click.ClickException: Neither has ``kagura-mcp``.
    """
    found = shutil.which(MCP_PROXY_COMMAND) or shutil.which(
        MCP_PROXY_COMMAND, path=str(Path(sys.executable).parent)
    )
    if found is None:
        raise click.ClickException(
            f"'{MCP_PROXY_COMMAND}' was not found on $PATH or next to {sys.executable}. "
            "Install kagura-memory where the harness runs (pip install kagura-memory), "
            "or use --url-form with an API key."
        )
    return os.path.abspath(found)


# =============================================================================
# The entry, and what a harness already has
# =============================================================================


@dataclass(frozen=True)
class _Entry:
    """The server entry to write, before a harness formats it."""

    #: stdio: the absolute ``kagura-mcp`` path and its arguments.
    command: str | None = None
    args: tuple[str, ...] = ()
    #: URL form: the endpoint, and the variable holding the API key.
    url: str | None = None
    key_env: str | None = None

    def auth_header(self) -> str:
        """The URL form's ``Authorization`` value: a reference, never a key."""
        return f"Bearer ${{{self.key_env}}}"


@dataclass(frozen=True)
class _Existing:
    """What an entry of the same name already is; never its values."""

    kind: str
    #: A URL entry carrying a bearer, which the Codex plugin's guardrail hooks
    #: can read (a stdio entry turns them into no-ops).
    url_credential: bool = False


_ENV_BEARER = "URL with a bearer token from an environment variable"


def _classify(entry: dict[str, Any]) -> _Existing:
    """Say what kind of entry ``entry`` is, from its keys only.

    Codex keeps the bearer in ``bearer_token_env_var`` / ``env_http_headers``
    (from the environment) or ``http_headers`` (literal); OpenClaw in
    ``headers``, where ``Bearer ${VAR}`` is a reference.
    """

    def authorization(key: str) -> bool:
        headers = entry.get(key)
        return isinstance(headers, dict) and any(
            isinstance(k, str) and k.lower() == "authorization" for k in headers
        )

    if "command" in entry:
        return _Existing("stdio (kagura-mcp)" if _runs_proxy(entry) else "stdio (another command)")
    if "url" not in entry:
        return _Existing("an entry setup does not recognise")
    if entry.get("auth") == "oauth":
        return _Existing("URL with OAuth")
    if entry.get("bearer_token_env_var") or authorization("env_http_headers"):
        return _Existing(_ENV_BEARER, url_credential=True)
    if authorization("headers"):
        static = holds_credential(entry)
        kind = "URL with a static Authorization header" if static else _ENV_BEARER
        return _Existing(kind, url_credential=True)
    if authorization("http_headers"):
        return _Existing("URL with a static Authorization header", url_credential=True)
    return _Existing("URL with OAuth or no credential")


def _capture(exe: str, args: list[str]) -> subprocess.CompletedProcess[str] | None:
    """Run a read-only harness command; None when it could not run."""
    try:
        return subprocess.run(
            [exe, *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_HARNESS_TIMEOUT_SEC,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None


# =============================================================================
# Harnesses
# =============================================================================


class _Harness(ABC):
    """One harness: where its config lives, and how to detect, write and print an entry."""

    key: ClassVar[HarnessName]
    title: ClassVar[str]
    cli: ClassVar[str]
    #: Its add command prompts, so it runs attached to a terminal or not at all.
    interactive_add: ClassVar[bool] = False
    #: Reads the MCP ``instructions``, so a ``--guardrails`` lane reaches it.
    reads_instructions: ClassVar[bool] = False
    #: How much of a context file the harness loads, and in which unit.
    agents_md_cap: ClassVar[tuple[int, str] | None] = None
    #: :meth:`detect` reads the config file, so it works without the CLI.
    detects_from_config: ClassVar[bool] = False
    #: The harness names the URL form's key variable itself (no ``--api-key-env``).
    names_key_env: ClassVar[bool] = False

    @abstractmethod
    def config_path(self) -> Path:
        """The file the harness keeps its MCP servers in."""

    @abstractmethod
    def detect(self, name: str, exe: str | None) -> _Existing | None:
        """The entry ``name`` has now, from its config or a read-only command."""

    @abstractmethod
    def add_args(self, name: str, entry: _Entry) -> list[str]:
        """The harness command (after its name) that adds ``entry`` as ``name``."""

    def replace_args(self, name: str, entry: _Entry) -> list[list[str]]:
        """The commands that replace an existing ``name`` (``--force``)."""
        return [self.add_args(name, entry)]

    @abstractmethod
    def block(self, name: str, entry: _Entry) -> str:
        """``entry`` in the config file's own syntax, to print."""

    def key_env(self, name: str, requested: str | None) -> str:
        """The variable the URL form reads the API key from."""
        return requested or DEFAULT_KEY_ENV

    @abstractmethod
    def key_note(self, entry: _Entry, *, ran: bool) -> str:
        """Where the URL form's key goes; never asks for it."""

    @abstractmethod
    def verify_args(self, name: str) -> list[str]:
        """The harness command that checks the entry."""

    @abstractmethod
    def agents_md_path(self) -> Path:
        """The always-loaded file the export goes into by default."""

    def notes(self) -> list[str]:
        """Harness-specific lines for the end of the run."""
        return []

    def export_notes(self, path: Path) -> list[str]:
        """Harness-specific lines after the AGENTS.md export wrote ``path``."""
        return []

    def plugin_hooks_on(self, name: str) -> bool:
        """True when a plugin's hooks read their credential from the ``name`` entry."""
        return False

    def warn_stdio_entry(
        self, name: str, existing: _Existing | None, hooks_on: bool, *, ask: bool
    ) -> None:
        """Say what a stdio entry costs here; ``ask`` lets the user stop (ClickException)."""

    def saved(self, name: str, exe: str, entry: _Entry) -> bool:
        """After the add command exited 0: whether ``entry`` is now saved as ``name``."""
        return True


def codex_home() -> Path:
    """``$CODEX_HOME``, else ``~/.codex``."""
    return Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")


#: The ``[mcp_servers.<name>]`` table the Codex plugin's hooks read when their
#: config.json names none (memory-cloud ``_codex_adapter.DEFAULT_MCP_SERVER``).
CODEX_HOOKS_DEFAULT_SERVER = "kagura-memory"
# The hooks treat a larger config.json as unreadable.
_CODEX_HOOKS_CONFIG_CAP = 64 * 1024


def codex_hooks_enabled(name: str) -> bool:
    """True when the memory-cloud ``kagura-memory`` Codex plugin's hooks read the ``name`` entry.

    Turning them on writes ``config.json`` into the plugin's data directory
    (memory-cloud ``docs/getting-started.md``); its ``mcp_server`` names the
    ``[mcp_servers.<name>]`` table they take their credential from
    (:data:`CODEX_HOOKS_DEFAULT_SERVER` when absent). A file the hooks cannot
    use (unreadable, over 64 KiB, not a JSON object) leaves them idle, as it
    does in the hooks themselves. Nothing read from it is echoed.

    Args:
        name: The MCP server name setup writes.

    Returns:
        Whether some config.json turns the hooks on for ``name``.
    """
    for path in (codex_home() / "plugins" / "data").glob("kagura-memory-*/config.json"):
        try:
            with path.open("rb") as f:
                raw = f.read(_CODEX_HOOKS_CONFIG_CAP + 1)
            settings = json.loads(raw) if len(raw) <= _CODEX_HOOKS_CONFIG_CAP else None
        except (OSError, ValueError):
            continue
        if not isinstance(settings, dict):
            continue
        server = settings.get("mcp_server")
        if (CODEX_HOOKS_DEFAULT_SERVER if server is None else server) == name:
            return True
    return False


class _Codex(_Harness):
    key = "codex"
    title = "Codex"
    cli = "codex"
    reads_instructions = True
    agents_md_cap = (32 * 1024, "bytes")
    detects_from_config = True

    def config_path(self) -> Path:
        return codex_home() / "config.toml"

    def detect(self, name: str, exe: str | None) -> _Existing | None:
        path = self.config_path()
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as e:
            raise click.ClickException(
                f"Cannot read {_path_label(path)} ({_exc_message(e)}); fix it and re-run."
            ) from None
        servers = data.get("mcp_servers")
        entry = servers.get(name) if isinstance(servers, dict) else None
        return _classify(entry) if isinstance(entry, dict) else None

    def add_args(self, name: str, entry: _Entry) -> list[str]:
        if entry.command is not None:
            return ["mcp", "add", name, "--", entry.command, *entry.args]
        assert entry.url is not None and entry.key_env is not None
        return ["mcp", "add", name, "--url", entry.url, "--bearer-token-env-var", entry.key_env]

    def replace_args(self, name: str, entry: _Entry) -> list[list[str]]:
        return [["mcp", "remove", name], self.add_args(name, entry)]

    def block(self, name: str, entry: _Entry) -> str:
        q = json.dumps  # a JSON string is a TOML basic string
        lines = [f"[mcp_servers.{name}]"]
        if entry.command is not None:
            args = ", ".join(q(a) for a in entry.args)
            lines += [f"command = {q(entry.command)}", f"args = [{args}]"]
        else:
            lines += [f"url = {q(entry.url)}", f"bearer_token_env_var = {q(entry.key_env)}"]
        return "\n".join(lines)

    def key_note(self, entry: _Entry, *, ran: bool) -> str:
        var = entry.key_env
        return (
            f"Codex reads the API key from ${var} when it connects: set it in the environment\n"
            f"  that starts Codex, e.g. `export {var}=<your-api-key>` in your shell profile.\n"
            "  (Codex refuses an inline bearer_token on a URL entry.)"
        )

    def verify_args(self, name: str) -> list[str]:
        return ["mcp", "get", name]

    def agents_md_path(self) -> Path:
        # Codex reads the global AGENTS.override.md in place of AGENTS.md.
        override = codex_home() / "AGENTS.override.md"
        return override if override.is_file() else codex_home() / "AGENTS.md"

    def notes(self) -> list[str]:
        return ["Restart Codex (or start a new session) to load the entry."]

    def plugin_hooks_on(self, name: str) -> bool:
        return codex_hooks_enabled(name)

    def warn_stdio_entry(
        self, name: str, existing: _Existing | None, hooks_on: bool, *, ask: bool
    ) -> None:
        _codex_hook_warning(name, existing, hooks_on, ask=ask)


def hermes_home() -> Path:
    """Where Hermes keeps ``config.yaml`` and ``.env`` for the active Hermes profile.

    ``$HERMES_HOME``, else the sticky profile in ``~/.hermes/active_profile``
    (``~/.hermes/profiles/<name>``), else ``~/.hermes``.
    """
    env = os.environ.get("HERMES_HOME")
    if env:
        return Path(env)
    root = Path.home() / ".hermes"
    try:
        active = (root / "active_profile").read_text(encoding="utf-8").strip().casefold()
    except (OSError, ValueError):
        active = ""
    if active != "default" and re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", active):
        return root / "profiles" / active
    return root


#: The project context files Hermes looks for, in order; it loads only the first.
HERMES_CONTEXT_FILES = (".hermes.md", "HERMES.md", "AGENTS.override.md", "AGENTS.md", "CLAUDE.md")


def hermes_context_file(directory: Path) -> Path:
    """The context file Hermes loads in ``directory``.

    Args:
        directory: The directory Hermes runs in.

    Returns:
        The first of :data:`HERMES_CONTEXT_FILES` that exists, else ``AGENTS.md``.
    """
    for name in HERMES_CONTEXT_FILES:
        if (directory / name).is_file():
            return directory / name
    return directory / "AGENTS.md"


def hermes_key_env(name: str) -> str:
    """The variable Hermes keeps a server's header API key in (its ``_env_key_for_server``)."""
    suffix = re.sub(r"[^A-Za-z0-9_]", "_", name.upper()).strip("_")
    return f"MCP_{suffix}_API_KEY"


class _Hermes(_Harness):
    key = "hermes"
    title = "Hermes Agent"
    cli = "hermes"
    interactive_add = True
    names_key_env = True

    def config_path(self) -> Path:
        return hermes_home() / "config.yaml"

    def detect(self, name: str, exe: str | None) -> _Existing | None:
        # `hermes mcp list` has no --json: match the Name column, and tell
        # stdio from URL by the Transport column (the URL, or the command).
        proc = _capture(exe, ["mcp", "list"]) if exe else None
        if proc is None or proc.returncode != 0:
            return None
        for line in _ANSI_RE.sub("", proc.stdout).splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] == name:
                is_url = parts[1].startswith(("http://", "https://"))
                return _Existing("URL" if is_url else "stdio")
        return None

    def saved(self, name: str, exe: str, entry: _Entry) -> bool:
        # `hermes mcp add` exits 0 when the user cancels an overwrite, declines
        # to save after a failed probe, or hits a validation error; only the
        # list tells. A same-form entry that was kept looks the same, though.
        found = self.detect(name, exe)
        return found is not None and found.kind == ("stdio" if entry.command else "URL")

    def add_args(self, name: str, entry: _Entry) -> list[str]:
        if entry.command is not None:
            # --args takes the rest of the command line, so it goes last.
            return ["mcp", "add", name, "--command", entry.command, "--args", *entry.args]
        assert entry.url is not None
        return ["mcp", "add", name, "--url", entry.url, "--auth", "header"]

    def block(self, name: str, entry: _Entry) -> str:
        q = json.dumps  # a JSON string is a YAML double-quoted scalar
        lines = ["mcp_servers:", f"  {name}:"]
        if entry.command is not None:
            args = ", ".join(q(a) for a in entry.args)
            lines += [f"    command: {q(entry.command)}", f"    args: [{args}]"]
        else:
            lines += [
                f"    url: {q(entry.url)}",
                "    headers:",
                f"      Authorization: {q(entry.auth_header())}",
            ]
        return "\n".join(lines)

    def key_env(self, name: str, requested: str | None) -> str:
        return hermes_key_env(name)

    def key_note(self, entry: _Entry, *, ran: bool) -> str:
        env_file = _path_label(hermes_home() / ".env")
        if ran:
            return (
                f"Hermes asked for the key itself and keeps it in {env_file} as\n"
                f"  {entry.key_env}; setup never saw it."
            )
        return (
            f"Add `{entry.key_env}=<your-api-key>` to {env_file} with an editor:\n"
            "  the entry reads it from there, and setup never sees the key."
        )

    def verify_args(self, name: str) -> list[str]:
        return ["mcp", "test", name]

    def agents_md_path(self) -> Path:
        return hermes_context_file(Path.cwd())

    def export_notes(self, path: Path) -> list[str]:
        return [
            "Hermes scans context files for prompt injection and skips a file it flags;\n"
            f"  if it reports {path.name} as blocked, delete the block between the\n"
            "  kagura-memory:guardrails markers."
        ]


def openclaw_config_path() -> Path:
    """``$OPENCLAW_CONFIG_PATH``, else ``~/.openclaw/openclaw.json``."""
    env = os.environ.get("OPENCLAW_CONFIG_PATH")
    return Path(env) if env else Path.home() / ".openclaw" / "openclaw.json"


class _OpenClaw(_Harness):
    key = "openclaw"
    title = "OpenClaw"
    cli = "openclaw"
    agents_md_cap = (20_000, "characters")

    def config_path(self) -> Path:
        return openclaw_config_path()

    def detect(self, name: str, exe: str | None) -> _Existing | None:
        # `mcp show` fails for a name OpenClaw does not have.
        proc = _capture(exe, ["mcp", "show", name, "--json"]) if exe else None
        if proc is None or proc.returncode != 0 or "{" not in proc.stdout:
            return None
        try:
            entry, _ = json.JSONDecoder().raw_decode(proc.stdout[proc.stdout.index("{") :])
        except ValueError:
            return None
        return _classify(entry) if isinstance(entry, dict) and entry else None

    def server(self, entry: _Entry) -> dict[str, Any]:
        """The ``mcp.servers.<name>`` value."""
        if entry.command is not None:
            return {"command": entry.command, "args": list(entry.args)}
        # OpenClaw defaults a URL entry to SSE; memory-cloud serves Streamable HTTP.
        return {
            "url": entry.url,
            "transport": "streamable-http",
            "headers": {"Authorization": entry.auth_header()},
        }

    def add_args(self, name: str, entry: _Entry) -> list[str]:
        if entry.command is not None:
            args = [a for arg in entry.args for a in ("--arg", arg)]
            return ["mcp", "add", name, "--command", entry.command, *args]
        assert entry.url is not None
        # --no-probe: the key is not in ~/.openclaw/.env yet.
        return [
            *("mcp", "add", name, "--url", entry.url, "--transport", "streamable-http"),
            *("--header", f"Authorization={entry.auth_header()}", "--no-probe"),
        ]

    def replace_args(self, name: str, entry: _Entry) -> list[list[str]]:
        # `mcp add` refuses an existing name; `mcp set` replaces the entry.
        return [["mcp", "set", name, json.dumps(self.server(entry))]]

    def block(self, name: str, entry: _Entry) -> str:
        return json.dumps({"mcp": {"servers": {name: self.server(entry)}}}, indent=2)

    def key_note(self, entry: _Entry, *, ran: bool) -> str:
        return (
            f"Add `{entry.key_env}=<your-api-key>` to ~/.openclaw/.env with an editor: the\n"
            f"  entry sends ${{{entry.key_env}}} (mcp.servers headers take no SecretRef),\n"
            "  and setup never sees the key."
        )

    def verify_args(self, name: str) -> list[str]:
        return ["mcp", "doctor", name, "--probe"]

    def agents_md_path(self) -> Path:
        # agents.defaults.workspace, which OpenClaw loads every session.
        return Path.home() / ".openclaw" / "workspace" / "AGENTS.md"

    def notes(self) -> list[str]:
        return [
            "The Gateway hot-reloads the file. MCP tools appear in OpenClaw's coding and\n"
            "  messaging tool profiles, not in minimal."
        ]


HARNESSES: dict[HarnessName, type[_Harness]] = {
    "codex": _Codex,
    "hermes": _Hermes,
    "openclaw": _OpenClaw,
}


# =============================================================================
# The AGENTS.md export
# =============================================================================


def _export_action(path: Path) -> str:
    """What writing the block would do to ``path``, for ``--dry-run``."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return "create"
    except (OSError, ValueError):
        return "update"
    return "replace the block in" if has_guardrail_block(text) else "append the block to"


def _deployment(mcp_url: str) -> str:
    """The server an MCP URL belongs to: its REST base URL, scheme and host lower-cased."""
    parts = urlsplit(base_url_from_mcp(mcp_url))
    return urlunsplit(parts._replace(scheme=parts.scheme.lower(), netloc=parts.netloc.lower()))


def _export_auth(profile: str | None, entry_url: str | None) -> _StaticAuth | _OAuthAuth:
    """The credential the AGENTS.md export fetches with; reads the credentials only.

    With ``--profile``, that profile alone, never ``KAGURA_API_KEY``: the
    stdio entry's ``kagura-mcp`` uses nothing else. Without it (``--url-form``
    only), the CLI's usual chain, as ``kagura guardrails digest`` uses. A URL
    form entry's server must be the credential's own: a context id belongs to
    one deployment, and setup never sends a credential to another server.

    Raises:
        click.ClickException: There is no credential, or it is for another server.
    """
    try:
        if profile is not None:
            auth: _StaticAuth | _OAuthAuth = _resolve_profile_auth(profile)
        else:
            auth = _resolve_auth(api_key=None, mcp_url=None, profile=None)
    except KaguraAuthError as e:
        raise click.ClickException(
            f"The AGENTS.md export has no credential: {_exc_message(e)}"
        ) from e
    if entry_url is not None and _deployment(auth.mcp_url) != _deployment(entry_url):
        source = _SOURCE_LABEL["oauth" if isinstance(auth, _OAuthAuth) else auth.source]
        raise click.ClickException(
            f"Nothing was written: the AGENTS.md export would use the {source} credential,\n"
            f"  which is for {_deployment(auth.mcp_url)}, but --mcp-url is on "
            f"{_deployment(entry_url)}.\n"
            "  Pass --profile with a login on that server, or set KAGURA_MCP_URL to it for\n"
            "  KAGURA_API_KEY."
        )
    return auth


async def _fetch_digest(auth: _StaticAuth | _OAuthAuth, context_id: str) -> GuardrailDigest:
    async with MemoryClient._from_resolved_auth(auth) as client:
        return await client.get_guardrail_digest(context_id)


def _kagura_command(args: list[str], profile: str | None, cf: CredentialsFile | None) -> str:
    """``kagura <args>``, to re-run later on the credential setup used.

    Without a profile that is the CLI's usual chain. With one, it names the
    profile when it is not the default, and unsets ``KAGURA_API_KEY`` when it
    is set here: the CLI ranks that key above every profile.
    """
    command = shlex.join(["kagura", *args])
    if profile is None or cf is None:
        return command
    if os.environ.get("KAGURA_API_KEY", "").strip():
        return f"env -u KAGURA_API_KEY KAGURA_PROFILE={shlex.quote(profile)} {command}"
    if profile != cf.default_profile:
        return f"KAGURA_PROFILE={shlex.quote(profile)} {command}"
    return command


def _write_export(
    h: _Harness, path: Path, context_id: str, auth: _StaticAuth | _OAuthAuth, refresh: str
) -> None:
    """Fetch the export block and splice it into ``path``, replacing only the marked block.

    Raises:
        click.ClickException: The fetch or the write failed (the MCP entry is
            already set up by then).
    """
    label = _path_label(path)
    failed = "The MCP entry is set up, but the AGENTS.md export failed"
    try:
        digest = asyncio.run(_fetch_digest(auth, context_id))
    except KaguraNotFoundError as e:
        raise click.ClickException(
            f"{failed}: context {context_id} is not visible to this credential on\n"
            f"  {_deployment(auth.mcp_url)} (404), or the server is older than v0.74.0.\n"
            f"  Nothing was written to {label}."
        ) from e
    except Exception as e:
        raise click.ClickException(f"{failed}: {_exc_message(e)}") from e
    if not digest.text.strip():
        click.echo(
            f"\n  Context {context_id} has no tool guardrails this credential can see (none\n"
            f"  marked, or the context is not trusted-tier): nothing was written to {label}."
        )
        try:
            stale = has_guardrail_block(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            stale = False
        if stale:
            click.echo(f"  It still has an earlier block; remove it with:\n    {refresh}")
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        status = write_guardrail_block(path, digest.text)
        text = path.read_text(encoding="utf-8")
    except (OSError, ValueError) as e:
        raise click.ClickException(f"{failed}: {label}: {_exc_message(e)}; left unchanged") from e
    done = "Already up to date:" if status == "unchanged" else "Wrote"
    click.echo(f"\n  {done} the guardrail block for context {context_id} in {label}")
    if h.agents_md_cap is not None:
        cap, unit = h.agents_md_cap
        size = len(text.encode("utf-8")) if unit == "bytes" else len(text)
        if size > cap:
            click.echo(
                f"  Warning: {label} is {size} {unit}; {h.title} reads only the first {cap}."
            )
    click.echo(f"  The block is a snapshot; refresh it with:\n    {refresh}")
    for note in h.export_notes(path):
        click.echo(f"  {note}")


# =============================================================================
# The flow
# =============================================================================


def _check_flags(
    h: _Harness,
    *,
    profile: str | None,
    name: str,
    context_id: str | None,
    guardrails: str | None,
    agents_md: str | None,
    url_form: bool,
    mcp_url: str | None,
    api_key_env: str | None,
    interactive: bool,
) -> None:
    """Reject flag combinations that cannot work before anything is read (exit 2)."""
    if not _SERVER_NAME_RE.fullmatch(name):
        raise click.BadParameter("use 1-64 letters, digits, '-' or '_'", param_hint="'--name'")
    if not url_form:
        if mcp_url is not None or api_key_env is not None:
            raise click.UsageError("--mcp-url and --api-key-env go with --url-form.")
        if profile is None:
            raise click.UsageError(
                "--profile is required: the entry runs `kagura-mcp --profile NAME` "
                "(see `kagura auth list`). Or use --url-form with an API key."
            )
    elif not mcp_url:
        raise click.UsageError(
            "--url-form needs --mcp-url: the MCP URL your API key works with, e.g. "
            "https://memory.kagura-ai.com/mcp/w/<workspace-id>."
        )
    else:
        try:
            validate_https_url(mcp_url, label="MCP URL")
        except ValueError as e:
            raise click.BadParameter(str(e), param_hint="'--mcp-url'") from None
    if api_key_env is not None:
        if h.names_key_env:
            raise click.UsageError(
                f"{h.title} names the variable itself ({h.key_env(name, None)}); "
                "drop --api-key-env."
            )
        if not _ENV_NAME_RE.fullmatch(api_key_env):
            raise click.BadParameter(
                "use an upper-case variable name, e.g. KAGURA_API_KEY",
                param_hint="'--api-key-env'",
            )
    if guardrails == "off" and not h.reads_instructions:
        raise click.UsageError(
            f"{h.title} does not read MCP instructions: its guardrails come only from the "
            "guardrails block of get_context_info, which --guardrails off removes."
        )
    has_context = context_id is not None or guardrails not in (None, "off")
    if agents_md is not None and not interactive and not has_context:
        raise click.UsageError(
            "--agents-md needs --context-id with -y or without a terminal: setup cannot "
            "ask which context."
        )
    if url_form and profile is None and context_id is not None:
        try:
            normalize_uuid(context_id, label="--context-id")
        except ValueError:
            raise click.UsageError(
                "Without --profile, --context-id must be a context UUID: setup cannot "
                "list contexts."
            ) from None


def _check_same_server(profile: str, profile_url: str, mcp_url: str) -> None:
    """With ``--url-form``, the profile only lists contexts and fetches the export,
    so it must be on the entry's server (exit 2 otherwise)."""
    ours, theirs = _deployment(profile_url), _deployment(mcp_url)
    if ours != theirs:
        raise click.UsageError(
            f"--profile {profile} is for {ours}, but --mcp-url is on {theirs}. With "
            "--url-form the profile only lists contexts and fetches the AGENTS.md export, "
            "so it must be on the same server: drop --profile, or log in to that server."
        )


def _codex_hook_warning(
    name: str, existing: _Existing | None, hooks_on: bool, *, ask: bool
) -> None:
    """Say that a stdio entry leaves the Codex plugin's guardrail hooks without a credential.

    Raises:
        click.ClickException: The hooks are on and the user chose not to go on.
    """
    if not hooks_on and not (existing is not None and existing.url_credential):
        return
    click.echo(
        "\n  Warning: the kagura-memory Codex plugin's guardrail hooks read their credential\n"
        "  only from a URL entry (bearer_token_env_var, env_http_headers or http_headers),\n"
        "  so with the stdio entry they do nothing."
    )
    if not hooks_on:
        return
    data = _path_label(codex_home() / "plugins" / "data")
    click.echo(
        f"  They are turned on here for the {name} entry (a config.json under\n"
        f"  {data}/kagura-memory-*/): to keep them, re-run with --url-form."
    )
    if ask and not click.confirm("Write the stdio entry anyway?", default=False):
        raise click.ClickException("Setup cancelled; nothing was written.")


def _warn_guardrails_off_url(h: _Harness, url: str) -> None:
    """Warn when a Hermes/OpenClaw URL already carries ``?guardrails=off``."""
    if mcp_url_guardrails_off(url):
        click.echo(
            f"\n  Warning: --mcp-url has ?guardrails=off, which removes the guardrails block\n"
            f"  from get_context_info: {h.title} then gets no guardrails from Kagura."
        )


def _resolve_context(
    chosen: str | None, contexts: dict[str, Any] | None, *, interactive: bool
) -> str:
    """The context UUID for a guardrails lane or the export.

    ``chosen`` (an id or a name) is looked up in ``contexts``, the profile's
    list; without one, an interactive run picks from that list. It never
    creates a context: a new one has no guardrails to deliver.

    Raises:
        click.UsageError: No context was given, and setup cannot list or ask.
        click.ClickException: The context is not a UUID (an unknown name), or
            the profile can see none to pick.
    """
    listed = [
        c for c in (contexts or {}).get("contexts", []) if isinstance(c, dict) and c.get("id")
    ]
    if chosen is None:
        if contexts is None or not interactive:
            raise click.UsageError("The AGENTS.md export needs --context-id.")
        if not listed:
            raise click.ClickException("The profile can see no context to export from.")
        click.echo("\nWhich context's tool guardrails go into the file?")
        for i, ctx in enumerate(listed, 1):
            click.echo(f"  {i}. {ctx.get('name', '?')} ({str(ctx['id'])[:8]}...)")
        choice = click.prompt("Context", type=click.IntRange(1, len(listed)))
        chosen = str(listed[choice - 1]["id"])
    else:
        match = next((c for c in listed if chosen in (c["id"], c.get("name"))), None)
        if match is not None:
            chosen = str(match["id"])
            click.echo(f"  Using context: {match.get('name', '?')} ({chosen[:8]}...)")
    try:
        return normalize_uuid(chosen, label="context_id")
    except ValueError:
        raise click.ClickException(f"No context {chosen!r} (by id or name).") from None


def _dry_run_context(chosen: str | None) -> str | None:
    """``chosen`` as ``--dry-run`` shows it: a UUID, or a placeholder for a context
    name, which only the real run looks up (a dry run makes no network call)."""
    if chosen is None:
        return None
    try:
        return normalize_uuid(chosen, label="context_id")
    except ValueError:
        return f"<UUID of context {chosen}>"


def _print_reason(
    h: _Harness, exe: str | None, non_interactive: bool, interactive: bool
) -> str | None:
    """Why setup prints the block instead of running the harness command, or None."""
    if exe is None:
        return f"`{h.cli}` is not on PATH"
    if h.interactive_add and not interactive:
        why = "-y was given" if non_interactive else "stdin is not a terminal"
        return f"`{h.cli} mcp add` is interactive and {why}"
    return None


def _run_or_fail(h: _Harness, exe: str, args: list[str], *, attached: bool) -> None:
    """Run ``<harness> <args>`` (attached to the terminal when it prompts).

    Raises:
        click.ClickException: It failed; the message names the subcommand.
    """
    name = f"{h.cli} {' '.join(args[:2])}"
    try:
        if attached:
            proc = subprocess.run([exe, *args], check=False)
        else:
            proc = subprocess.run(
                [exe, *args],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=_HARNESS_TIMEOUT_SEC,
                check=False,
            )
    except subprocess.TimeoutExpired as e:
        raise click.ClickException(f"`{name}` failed: timed out after {e.timeout:g}s") from None
    except (OSError, subprocess.SubprocessError) as e:
        raise click.ClickException(f"`{name}` failed: {_exc_message(e)}") from None
    if proc.returncode != 0:
        detail = "" if attached else (proc.stderr or proc.stdout or "").strip()
        raise click.ClickException(f"`{name}` failed: {detail or f'exit code {proc.returncode}'}")


def _echo_block(block: str) -> None:
    for line in block.splitlines():
        click.echo(f"    {line}")


def _preview_command(
    context_id: str, entry: _Entry, profile: str | None, cf: CredentialsFile | None
) -> str:
    """``kagura guardrails digest <ctx> --target instructions`` on the entry's own credential."""
    args = ["guardrails", "digest", context_id, "--target", "instructions"]
    if entry.url is None:
        return _kagura_command(args, profile, cf)
    # The URL form: the key in the entry's variable, on the entry's server.
    env = [] if entry.key_env == DEFAULT_KEY_ENV else [f'KAGURA_API_KEY="${{{entry.key_env}}}"']
    env.append(f"KAGURA_MCP_URL={shlex.quote(entry.url)}")
    return " ".join([*env, shlex.join(["kagura", *args])])


def run_setup_harness(
    harness: HarnessName,
    *,
    profile: str | None,
    name: str,
    context_id: str | None,
    guardrails: str | None,
    agents_md: str | None,
    url_form: bool,
    mcp_url: str | None,
    api_key_env: str | None,
    force: bool,
    non_interactive: bool,
    dry_run: bool,
) -> None:
    """Set up the ``name`` MCP entry of ``harness``, and the AGENTS.md export if asked.

    ``guardrails`` must already be normalized (``off`` or a canonical UUID).
    ``agents_md`` is None without ``--agents-md`` and ``""`` for the
    harness's default file. Setup prompts only with a terminal on stdin and
    without ``-y``; otherwise it behaves as ``-y`` does. See the module
    docstring for the rules.

    Raises:
        click.UsageError: A flag combination that cannot work (exit 2).
        click.ClickException: A missing profile or ``kagura-mcp``, an existing
            entry without ``force``, or a failed check or harness command.
    """
    h = HARNESSES[harness]()
    interactive = not non_interactive and _stdin_is_tty()
    _check_flags(
        h,
        profile=profile,
        name=name,
        context_id=context_id,
        guardrails=guardrails,
        agents_md=agents_md,
        url_form=url_form,
        mcp_url=mcp_url,
        api_key_env=api_key_env,
        interactive=interactive,
    )
    cf, creds = _load_profile(profile) if profile is not None else (None, None)
    if url_form and creds is not None:
        assert profile is not None and mcp_url is not None
        _check_same_server(profile, creds.mcp_url, mcp_url)
    proxy = None if url_form else _proxy_path()
    exe = _harness_executable(h.cli)
    where = _path_label(h.config_path())
    ask = interactive and not dry_run
    if dry_run:
        click.echo("Dry run: nothing is written, run or fetched.")
    click.echo(f"\nSetting up Kagura Memory for {h.title} ({where})")

    # 1. What the harness has now (read-only)
    existing = h.detect(name, exe)
    if existing is not None:
        click.echo(f"  Existing {name} entry: {existing.kind}")
    elif exe is None and not h.detects_from_config:
        click.echo(f"  `{h.cli}` is not on PATH, so setup cannot look for a {name} entry.")
    else:
        click.echo(f"  No {name} entry yet.")
    hooks_on = h.plugin_hooks_on(name)
    if not url_form:
        # No question when the existing-entry stop below ends the run anyway.
        h.warn_stdio_entry(name, existing, hooks_on, ask=ask and (existing is None or force))
    if existing is not None and not force:
        stop = f"a {name} entry already exists ({existing.kind}); re-run with --force to replace it"
        if not dry_run:
            raise click.ClickException(f"Nothing was written: {stop}.")
        click.echo(f"  Setup would stop here: {stop}.")

    # 2. Guardrails. Only Codex reads MCP instructions; for Hermes and OpenClaw
    # a context UUID only picks the export's context.
    export_context = context_id
    if guardrails not in (None, "off"):
        export_context = export_context or guardrails
        if not h.reads_instructions:
            click.echo(
                f"\n  Warning: {h.title} does not read MCP instructions, so --guardrails has no\n"
                f"  effect there and is not written. Guardrails reach {h.title} through\n"
                "  get_context_info (on by default) and the AGENTS.md export (--agents-md)."
            )
            guardrails = None
    lane_from_context = False
    if h.reads_instructions and guardrails is None:
        if url_form and hooks_on:
            guardrails = "off"
            click.echo(
                "  The plugin's hooks deliver guardrails, so the URL gets ?guardrails=off\n"
                "  (the hooks' own setup asks for it; --guardrails overrides)."
            )
        elif context_id is not None:
            lane_from_context = True

    # 3. The profile check, and whether to export
    contexts = None
    if creds is not None and not dry_run:
        assert profile is not None
        contexts = _verify_profile(profile, creds)
    # Without a profile, setup can only use a context it was given.
    can_pick = contexts is not None or export_context is not None
    export_path = None
    offered = False
    if agents_md is not None:
        export_path = Path(agents_md).expanduser() if agents_md else h.agents_md_path()
    elif not h.reads_instructions and ask and can_pick:
        offered = True
        path = h.agents_md_path()
        click.echo(
            f"\n  {h.title} does not read MCP instructions. A snapshot of a context's tool\n"
            f"  guardrails can go into {_path_label(path)}, which it loads every session."
        )
        if click.confirm("Write the guardrail export block there?", default=False):
            export_path = path

    # 4. The context, when a guardrails lane or the export needs one, and the
    # export's credential: both settled before anything is written.
    if lane_from_context or export_path is not None:
        if dry_run:
            export_context = _dry_run_context(export_context)
        else:
            export_context = _resolve_context(export_context, contexts, interactive=interactive)
    if lane_from_context:
        guardrails = export_context
    export_auth = None
    if export_path is not None:
        export_auth = _export_auth(profile, mcp_url if url_form else None)

    # 5. The entry
    if proxy is not None:
        assert profile is not None
        entry = _Entry(
            command=proxy, args=tuple(_stdio_entry(profile, guardrails=guardrails)["args"])
        )
    else:
        assert mcp_url is not None
        url = mcp_url_with_query(mcp_url, guardrails=guardrails)
        if guardrails is not None and guardrails.startswith("<"):
            # A dry run's placeholder for a context name (a UUID or "off" never
            # starts with "<"), shown as it is rather than URL-encoded.
            url = url.replace(f"guardrails={quote_plus(guardrails)}", f"guardrails={guardrails}")
        entry = _Entry(url=url, key_env=h.key_env(name, api_key_env))
        if not h.reads_instructions:
            _warn_guardrails_off_url(h, url)
    commands = h.replace_args(name, entry) if existing is not None else [h.add_args(name, entry)]
    reason = _print_reason(h, exe, non_interactive, interactive)

    click.echo("")
    if reason is None:
        verb = "Would run" if dry_run else "Running"
        if dry_run and existing is not None and not force:
            verb = "With --force, would run"
        for args in commands:
            click.echo(f"  {verb}: {shlex.join([h.cli, *args])}")
    else:
        replace = " in place of the existing one" if existing is not None else ""
        click.echo(
            f"  Setup does not edit {where} itself ({reason}).\n"
            f"  Add this {name} entry to it{replace}:"
        )
    if dry_run or reason is not None:
        click.echo("")
        _echo_block(h.block(name, entry))
    if dry_run:
        _echo_dry_run_export(h, export_path, export_context, non_interactive, interactive)
        return

    # 6. Write through the harness
    if reason is None:
        assert exe is not None
        for i, args in enumerate(commands):
            try:
                _run_or_fail(h, exe, args, attached=h.interactive_add)
            except click.ClickException as e:
                if i:
                    e.message += (
                        f"\nThe previous {name} entry was removed; re-run setup to add one."
                    )
                raise
        if not h.saved(name, exe, entry):
            skipped = " and skipped the AGENTS.md export" if export_path is not None else ""
            raise click.ClickException(
                f"`{h.cli} mcp list` shows no new {name} entry: `{h.cli} mcp add` was\n"
                f"  cancelled or failed there, so nothing was saved{skipped}."
            )
        click.echo(f"  Done: {h.cli} wrote {name} to {where}.")

    # 7. What the user does next
    click.echo("")
    if entry.url is not None:
        click.echo(f"  {h.key_note(entry, ran=reason is None)}")
    if h.reads_instructions and guardrails not in (None, "off"):
        assert guardrails is not None
        click.echo(
            "  Codex should get the tool guardrail digest of context\n"
            f"  {guardrails} in the MCP instructions when it connects.\n"
            "  The server sends only its base text instead when the entry's credential\n"
            "  cannot read that context, the context has no guardrails, or the deployment\n"
            "  turns the digest off. Preview what it sends:\n"
            f"    {_preview_command(guardrails, entry, profile, cf)}\n"
            "  Use a context whose editor list you control: every editor's guardrail\n"
            "  summaries reach the model."
        )
    for note in h.notes():
        click.echo(f"  {note}")
    click.echo(f"  Check it with: {shlex.join([h.cli, *h.verify_args(name)])}")
    if export_path is None and not h.reads_instructions and not offered:
        click.echo(
            "  Re-run with --agents-md --context-id <id> to put a snapshot of a context's\n"
            f"  tool guardrails into {_path_label(h.agents_md_path())},\n"
            f"  which {h.title} loads every session."
        )

    if export_path is not None:
        assert export_context is not None and export_auth is not None
        refresh = _kagura_command(
            ["guardrails", "digest", export_context, "--out", str(export_path)], profile, cf
        )
        _write_export(h, export_path, export_context, export_auth, refresh)


def _echo_dry_run_export(
    h: _Harness,
    path: Path | None,
    context_id: str | None,
    non_interactive: bool,
    interactive: bool,
) -> None:
    """The AGENTS.md line of ``--dry-run``: the path and what would happen to it."""
    if path is not None:
        context = context_id or "<chosen at the context prompt>"
        click.echo(
            f"\n  AGENTS.md: would {_export_action(path)} {_path_label(path)} "
            f"(the guardrail block for context {context})"
        )
    elif not h.reads_instructions:
        if interactive:
            when = "offered when setup runs"
        else:
            when = "not offered with -y" if non_interactive else "not offered without a terminal"
        click.echo(
            f"\n  AGENTS.md: {_path_label(h.agents_md_path())} ({when}; --agents-md writes it)"
        )
