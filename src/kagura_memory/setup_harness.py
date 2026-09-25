"""``kagura setup codex|hermes|openclaw`` — MCP entries for other harnesses (#260).

OpenAI Codex, Hermes Agent and OpenClaw can each spawn a stdio server, and
``kagura-mcp`` already forwards every JSON-RPC message with a fresh bearer
from the ``kagura auth login`` profile. The default entry therefore runs the
proxy: one device-flow login serves every harness, on any supported server.
``--url-form`` writes a URL entry that reads a long-lived API key from an
environment variable instead.

memory-cloud's dynamic client registration accepts a loopback client only
when its name carries a known keyword. From memory-cloud 0.77.0 the keywords
include Codex, Hermes Agent and OpenClaw (memory-cloud#1657), so their own
OAuth client registration is accepted there; before 0.77.0 it is rejected.
The stdio entry stays the default all the same: that needs no per-harness
browser sign-in and works on older servers. The opt-in ``--url-form --oauth`` writes a
URL entry with no key, which the harness then signs in to itself (from memory-cloud
0.78.0, memory-cloud#1671, Hermes can also do so with its device flow, which needs
no loopback callback); setup first
checks that the entry's server is memory-cloud 0.77.0+. Each harness has
signed in end to end on a ``/mcp/w/<workspace-id>`` URL against memory-cloud
0.78.0 (#284).

Rules the three commands share:

* **The harness owns its config file.** An entry is written only through
  the harness command documented for it (``codex mcp add``, ``hermes mcp
  add``, ``openclaw mcp add|set``). Without that CLI on ``PATH``, or when
  the command is interactive and there is no terminal (or ``-y``), setup
  prints the block and the file and edits nothing: the SDK has no TOML,
  YAML or JSON5 writer, and text appended to such a file can duplicate a
  key. ``codex mcp add`` of an ``--oauth`` entry starts Codex's browser
  sign-in, so it counts as interactive.
* **Filtered environments.** Each harness passes a filtered environment to
  the servers it spawns, so the stdio entry names ``kagura-mcp`` by
  absolute path and always passes ``--profile``; it never depends on
  ``PATH``, ``KAGURA_PROFILE`` or the credentials file's
  ``default_profile``. ``HOME`` needs nothing: ``Path.home()`` falls back
  to the password database when it is unset.
* **No secret is written, printed or passed on a command line.** The stdio
  entry holds none, the URL form names the variable the key is read from,
  and an ``--oauth`` entry has neither a header nor a key variable: the
  harness keeps its own token, which setup never sees. An existing entry is
  read only to say what kind it is; nothing read from it is echoed.
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
import reprlib
import shlex
import shutil
import subprocess
import sys
import textwrap
import tomllib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
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
    mcp_url_without_query_param,
    normalize_url,
    normalize_uuid,
    validate_https_url,
)
from ._version import meets_minimum
from .auth.credentials import CredentialsFile, load_credentials_file
from .auth.device_flow import fetch_system_info, make_oauth_client
from .claude_code import MCP_PROXY_COMMAND, _path_label, _runs_proxy, holds_credential
from .config import load_config
from .exceptions import KaguraAuthError, KaguraNotFoundError, _exc_message
from .memory_client import MemoryClient
from .models import GuardrailDigest
from .setup_claude import _load_profile, _stdio_entry, _verify_profile

HarnessName = Literal["codex", "hermes", "openclaw"]

#: The variable the Codex and OpenClaw URL forms read the API key from, as in
#: memory-cloud's own setup docs. Hermes names its variable itself.
DEFAULT_KEY_ENV = "KAGURA_API_KEY"

HARNESS_OAUTH_MIN_SERVER_VERSION: tuple[int, int, int] = (0, 77, 0)
"""First memory-cloud release whose DCR accepts the harnesses' own OAuth clients.

memory-cloud v0.77.0 (memory-cloud#1657) adds ``codex``, ``hermes`` and
``openclaw`` to the loopback keywords of its dynamic client registration.
Before it, their registration gets 400 ``invalid_client_metadata``, so an
``--oauth`` entry could never sign in; setup checks the version first.
"""

# Codex, Hermes and OpenClaw all accept these; a dot would nest a TOML table.
# The name is a positional in every harness argv, so it starts with a letter or
# digit: `codex mcp add --help …` prints the help and exits 0 with nothing saved.
_SERVER_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
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


def _wrap(text: str) -> str:
    """``text`` wrapped to follow a two-space indent, never breaking a `command`."""
    # textwrap does not split at NUL, so a command's spaces hold while it wraps.
    held = re.sub(r"`[^`]*`", lambda m: m.group(0).replace(" ", "\0"), text)
    lines = textwrap.wrap(held, 78, break_on_hyphens=False, break_long_words=False)
    return "\n  ".join(lines).replace("\0", " ")


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
    #: URL form with ``--oauth``: no key, the harness signs in itself.
    oauth: bool = False

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


def _has_authorization(headers: object) -> bool:
    """True when ``headers`` is a mapping with an ``Authorization`` key, in any case."""
    return isinstance(headers, dict) and any(
        isinstance(k, str) and k.lower() == "authorization" for k in headers
    )


def _classify(entry: dict[str, Any]) -> _Existing:
    """Say what kind of entry ``entry`` is, from its keys only.

    Codex keeps the bearer in ``bearer_token_env_var`` / ``env_http_headers``
    (from the environment) or ``http_headers`` (literal); OpenClaw in
    ``headers``, where ``Bearer ${VAR}`` is a reference.
    """

    def authorization(key: str) -> bool:
        return _has_authorization(entry.get(key))

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

    def attached_add(self, entry: _Entry) -> bool:
        """Adding ``entry`` prompts or signs in, so it runs attached to a terminal or not at all."""
        return self.interactive_add

    @abstractmethod
    def add_args(self, name: str, entry: _Entry) -> list[str]:
        """The harness command (after its name) that adds ``entry`` as ``name``."""

    def replace_args(self, name: str, entry: _Entry) -> list[str]:
        """The harness command that replaces an existing ``name`` (``--force``).

        One command, never a remove and then an add: a failed add would then
        leave no entry at all.
        """
        return self.add_args(name, entry)

    @abstractmethod
    def block(self, name: str, entry: _Entry) -> str:
        """``entry`` in the config file's own syntax, to print."""

    def block_target(self) -> str:
        """Where in the config file the printed block goes."""
        return "it"

    def block_notes(self, name: str) -> list[str]:
        """Lines to print with the block, on how it fits the file as it is."""
        return []

    def key_env(self, name: str, requested: str | None) -> str:
        """The variable the URL form reads the API key from."""
        return requested or DEFAULT_KEY_ENV

    @abstractmethod
    def key_note(self, entry: _Entry, *, ran: bool) -> str:
        """Where the URL form's key goes; never asks for it."""

    @abstractmethod
    def sign_in_note(self, name: str, *, ran: bool) -> str:
        """How the user signs the ``--oauth`` entry in; setup never runs the harness login.

        It names the harness's way round a browser that cannot reach its
        loopback callback (a remote host). ``ran``: setup ran the add command,
        rather than printing the block.
        """

    @abstractmethod
    def token_store(self, name: str) -> str:
        """Where the harness keeps the ``--oauth`` entry's token."""

    def login_note(self, name: str, *, ran: bool) -> str:
        """What replaces :meth:`key_note` for an ``--oauth`` entry: who signs in, and where."""
        return _wrap(
            f"{self.sign_in_note(name, ran=ran)} memory-cloud's consent screen shows the "
            f"client name {self.title} sends, which nothing verifies: approve only a sign-in "
            f"you started. {self.title} keeps the token in {self.token_store(name)}; setup "
            "never sees it."
        )

    def add_failure_note(self, name: str, entry: _Entry) -> str:
        """What to do when the add command failed, for the error message; may be empty."""
        return ""

    @abstractmethod
    def verify_args(self, name: str) -> list[str]:
        """The harness command that checks the entry."""

    @abstractmethod
    def agents_md_path(self) -> Path | None:
        """The always-loaded file the export goes into by default.

        None when every such file would change what the harness loads
        (:meth:`no_agents_md_reason` says why): there is then no offer, and
        ``--agents-md`` needs a PATH.
        """

    def no_agents_md_reason(self) -> str:
        """Why :meth:`agents_md_path` is None, naming the file it would displace."""
        return ""

    def notes(self) -> list[str]:
        """Harness-specific lines for the end of the run."""
        return []

    def export_notes(self, path: Path) -> list[str]:
        """Harness-specific lines after the AGENTS.md export wrote ``path``."""
        return []

    def plugin_hooks_on(self, name: str) -> bool:
        """True when a plugin's hooks read their credential from the ``name`` entry."""
        return False

    def warn_keyless_entry(
        self, name: str, existing: _Existing | None, hooks_on: bool, *, oauth: bool, ask: bool
    ) -> None:
        """Say what an entry without a key (stdio, or ``--oauth``) costs here.

        ``ask`` lets the user stop (ClickException).
        """

    def not_saved(self, name: str, exe: str, entry: _Entry, *, replaced: bool) -> str | None:
        """After the add command exited 0: why ``entry`` is not saved as ``name``, or None.

        ``replaced``: a ``name`` entry existed before the add (``--force``).
        """
        return None


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

    def attached_add(self, entry: _Entry) -> bool:
        # With no bearer_token_env_var, `mcp add --url` saves the entry and then
        # starts Codex's browser sign-in once discovery finds OAuth.
        return entry.oauth

    def add_args(self, name: str, entry: _Entry) -> list[str]:
        # `mcp add` also replaces an entry of the same name, whatever its form.
        if entry.command is not None:
            return ["mcp", "add", name, "--", entry.command, *entry.args]
        assert entry.url is not None
        if entry.oauth:
            return ["mcp", "add", name, "--url", entry.url]
        assert entry.key_env is not None
        return ["mcp", "add", name, "--url", entry.url, "--bearer-token-env-var", entry.key_env]

    def block(self, name: str, entry: _Entry) -> str:
        q = json.dumps  # a JSON string is a TOML basic string
        lines = [f"[mcp_servers.{name}]"]
        if entry.command is not None:
            args = ", ".join(q(a) for a in entry.args)
            lines += [f"command = {q(entry.command)}", f"args = [{args}]"]
        else:
            # An entry without a bearer is an OAuth one: `auth` defaults to "oauth".
            lines.append(f"url = {q(entry.url)}")
            if not entry.oauth:
                lines.append(f"bearer_token_env_var = {q(entry.key_env)}")
        return "\n".join(lines)

    def key_note(self, entry: _Entry, *, ran: bool) -> str:
        var = entry.key_env
        return (
            f"Codex reads the API key from ${var} when it connects: set it in the environment\n"
            f"  that starts Codex, e.g. `export {var}=<your-api-key>` in your shell profile.\n"
            "  (Codex refuses an inline bearer_token on a URL entry.)"
        )

    def sign_in_note(self, name: str, *, ran: bool) -> str:
        login = f"codex mcp login {name}"
        if ran:
            first = (
                "Codex signs in itself: `codex mcp add` above started its sign-in if it found "
                f"OAuth on the server. If it did not log in, run `{login}`."
            )
        else:
            first = f"Once the table is in config.toml, sign in with `{login}`."
        return (
            f"{first} The sign-in redirects the browser to Codex's loopback callback on this "
            "host; when the browser cannot reach it (no browser here, or a remote host), add "
            "--no-browser: Codex then prints the URL and takes the callback URL pasted back. "
            "Codex keys the token on the entry's URL, so changing its ?guardrails= later "
            "(another --guardrails or --context-id) means signing in again."
        )

    def token_store(self, name: str) -> str:
        home = _path_label(codex_home())
        fallback = _path_label(codex_home() / ".credentials.json")
        # On Windows Codex turns its secret_auth_storage feature on by default,
        # which keeps MCP OAuth tokens in an encrypted local store under CODEX_HOME.
        return (
            f'the OS keyring ("Codex MCP Credentials"; on Windows, its encrypted secrets '
            f"store in {home}), else in {fallback}"
        )

    def add_failure_note(self, name: str, entry: _Entry) -> str:
        if not entry.oauth:
            return ""
        return (
            "\n  Codex saves the entry before it signs in, so it may be saved already: check with\n"
            f"  `codex mcp get {name}`, then sign in with `codex mcp login {name}`."
        )

    def verify_args(self, name: str) -> list[str]:
        return ["mcp", "get", name]

    def agents_md_path(self) -> Path | None:
        # Codex reads the global AGENTS.override.md in place of AGENTS.md.
        override = codex_home() / "AGENTS.override.md"
        return override if override.is_file() else codex_home() / "AGENTS.md"

    def notes(self) -> list[str]:
        return ["Restart Codex (or start a new session) to load the entry."]

    def plugin_hooks_on(self, name: str) -> bool:
        return codex_hooks_enabled(name)

    def warn_keyless_entry(
        self, name: str, existing: _Existing | None, hooks_on: bool, *, oauth: bool, ask: bool
    ) -> None:
        _codex_hook_warning(name, existing, hooks_on, oauth=oauth, ask=ask)


# The top-level `mcp_servers:` key of a config.yaml, and what follows its colon.
# A BOM before it and a quoted key are valid YAML too: missing either would print
# a second top-level key.
_YAML_SERVERS_KEY_RE = re.compile(r"""^\ufeff?(["']?)mcp_servers\1[ \t]*:(.*)$""")
# Blank and comment lines neither open nor close a block.
_YAML_SKIP_RE = re.compile(r"^\s*(?:#|$)")
_YAML_INDENT_RE = re.compile(r"^([ \t]+)\S")


@dataclass(frozen=True)
class _YamlServers:
    """The top-level ``mcp_servers:`` key a Hermes ``config.yaml`` already has."""

    #: The indent of the entries under it (two spaces when it has none yet).
    indent: str
    #: Its value is written inline (flow style, or a scalar such as ``null``).
    inline: bool


def _yaml_servers(text: str) -> _YamlServers | None:
    """Find a top-level ``mcp_servers:`` key in ``config.yaml`` text, without parsing YAML.

    YAML keeps the last of two equal keys, so a second top-level
    ``mcp_servers:`` pasted in would drop every server under the first,
    without an error.

    Args:
        text: The file's text.

    Returns:
        The key's entry indent and form, or None when the file has no such key.
    """
    lines = re.split(r"\r?\n", text)
    for i, line in enumerate(lines):
        key = _YAML_SERVERS_KEY_RE.match(line)
        if key is None:
            continue
        inline = not _YAML_SKIP_RE.match(key.group(2))
        after = next((x for x in lines[i + 1 :] if not _YAML_SKIP_RE.match(x)), "")
        indent = _YAML_INDENT_RE.match(after)
        return _YamlServers(indent.group(1) if indent else "  ", inline)
    return None


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


#: The names of each project context file type Hermes looks for, in its order.
_HERMES_OWN_FILES = (".hermes.md", "HERMES.md")
_HERMES_AGENTS_FILES = ("AGENTS.override.md", "AGENTS.md", "agents.md")
_HERMES_CLAUDE_FILES = ("CLAUDE.md", "claude.md")


def _has_text(path: Path) -> bool:
    """True when ``path`` is a file with something left after ``strip()``, as Hermes reads it."""
    try:
        return path.is_file() and bool(path.read_text(encoding="utf-8", errors="replace").strip())
    except OSError:
        return False


def _first_with_text(directory: Path, names: tuple[str, ...]) -> Path | None:
    return next((directory / n for n in names if _has_text(directory / n)), None)


def hermes_cursor_rules(directory: Path) -> Path | None:
    """The Cursor rules file Hermes loads in ``directory``: ``.cursorrules``, else a rule file."""
    rules = sorted((directory / ".cursor" / "rules").glob("*.mdc"))
    return next((f for f in (directory / ".cursorrules", *rules) if _has_text(f)), None)


def hermes_context_file(directory: Path) -> Path | None:
    """The file the export goes into so that Hermes, run in ``directory``, loads it (#278).

    Hermes loads only the first of these types that has a file with text in
    it (``hermes-agent`` ``agent/prompt_builder.py``):

    1. ``.hermes.md`` / ``HERMES.md``: the nearest that exists, from
       ``directory`` up to the git root (``directory`` alone outside a
       repository). The walk stops there even when that file is empty.
    2. ``AGENTS.override.md`` / ``AGENTS.md`` / ``agents.md``: every
       directory from the git root down to ``directory``, the first with text
       in each.
    3. ``CLAUDE.md`` / ``claude.md`` in ``directory``.
    4. ``.cursorrules`` / ``.cursor/rules/*.mdc`` in ``directory``.

    The export goes into the loaded ``.hermes.md`` / ``HERMES.md`` (even one
    in a parent directory), into ``directory``'s AGENTS file that loads (a
    new ``AGENTS.md`` there when the chain loads only from parents), or into
    the loaded ``CLAUDE.md``. With nothing loaded, a new ``AGENTS.md``.
    Writing any other file would stop Hermes loading the user's own.

    Args:
        directory: The directory Hermes runs in.

    Returns:
        The file; None when only Cursor rules load, which a new ``AGENTS.md``
        would stop loading (:func:`hermes_cursor_rules` names them).
    """
    # ``directory`` up to the git root, the nearest ancestor holding .git.
    walk = (directory, *directory.parents)
    root = next((i for i, d in enumerate(walk) if (d / ".git").exists()), None)
    chain = walk[: root + 1] if root is not None else walk[:1]
    for d in chain:
        own = next((d / n for n in _HERMES_OWN_FILES if (d / n).is_file()), None)
        if own is not None:
            if _has_text(own):
                return own
            break  # an empty one ends the lookup: Hermes moves on to the next type
    local = _first_with_text(directory, _HERMES_AGENTS_FILES)
    if local is not None:
        return local
    if any(_first_with_text(d, _HERMES_AGENTS_FILES) for d in chain[1:]):
        return directory / "AGENTS.md"
    claude = _first_with_text(directory, _HERMES_CLAUDE_FILES)
    if claude is not None:
        return claude
    if hermes_cursor_rules(directory) is not None:
        return None
    return directory / "AGENTS.md"


def hermes_key_env(name: str) -> str:
    """The variable Hermes keeps a server's header API key in (its ``_env_key_for_server``)."""
    suffix = re.sub(r"[^A-Za-z0-9_]", "_", name.upper()).strip("_")
    return f"MCP_{suffix}_API_KEY"


#: The ``--connect-timeout`` of an ``--oauth`` ``hermes mcp add``, whose probe
#: runs the browser sign-in: Hermes's ``login_connect_timeout``, the
#: ``oauth.timeout`` callback window (300 s) plus 15 s for the token exchange.
#: Hermes keeps it as the entry's ``connect_timeout`` (default 60 s).
HERMES_OAUTH_CONNECT_TIMEOUT_SEC = 315


@dataclass(frozen=True)
class _HermesEntry:
    """What setup keeps of a Hermes ``mcp_servers.<name>`` entry (#278).

    Only these: never a header value, the ``env`` or any other key, which can
    hold a secret (``hermes config get`` masks credential-shaped keys, but by
    name only). Nothing here is echoed; :attr:`kind` describes it.
    """

    command: str | None
    args: tuple[str, ...]
    url: str | None
    oauth: bool
    #: ``headers`` has an ``Authorization`` key (any case).
    authorization: bool
    enabled: bool

    @classmethod
    def read(cls, value: object) -> _HermesEntry:
        entry = value if isinstance(value, dict) else {}
        args = entry.get("args")
        # As `hermes mcp list` reads it: a string counts only as true/1/yes.
        enabled = entry.get("enabled", True)
        if isinstance(enabled, str):
            enabled = enabled.lower() in {"true", "1", "yes"}
        return cls(
            command=entry["command"] if isinstance(entry.get("command"), str) else None,
            args=tuple(str(a) for a in args) if isinstance(args, list) else (),
            url=entry["url"] if isinstance(entry.get("url"), str) else None,
            oauth=entry.get("auth") == "oauth",
            authorization=_has_authorization(entry.get("headers")),
            enabled=bool(enabled),
        )

    @property
    def kind(self) -> str:
        # A url wins over a command, as in `hermes mcp list`.
        if self.url is not None:
            if self.oauth:
                return "URL with OAuth"
            if self.authorization:
                return "URL with an Authorization header"
            return "URL with no credential"
        if self.command is not None:
            proxy = _runs_proxy({"command": self.command, "args": list(self.args)})
            return "stdio (kagura-mcp)" if proxy else "stdio (another command)"
        return "an entry setup does not recognise"

    def is_(self, entry: _Entry) -> bool:
        """True when this is ``entry``: its command and args, or its url."""
        if entry.command is not None:
            return self.url is None and (self.command, self.args) == (entry.command, entry.args)
        return self.url == entry.url


class _Hermes(_Harness):
    key = "hermes"
    title = "Hermes Agent"
    cli = "hermes"
    interactive_add = True
    names_key_env = True

    def __init__(self) -> None:
        #: :meth:`not_saved` read back a URL entry with no Authorization header.
        self._no_header = False

    def config_path(self) -> Path:
        return hermes_home() / "config.yaml"

    def _read_entry(self, name: str, exe: str) -> tuple[bool, _HermesEntry | None]:
        """``(read, entry)`` from ``hermes config get mcp_servers.<name> --json``.

        ``entry`` is None when Hermes has no such entry; ``read`` is False
        when the command failed otherwise (an older Hermes), and the caller
        falls back to ``hermes mcp list``.
        """
        read, value = self._config_get(name, exe)
        return read, (_HermesEntry.read(value) if read and value is not None else None)

    def detect(self, name: str, exe: str | None) -> _Existing | None:
        if exe is None:
            return None
        read, found = self._read_entry(name, exe)
        if read:
            return None if found is None else _Existing(found.kind)
        return self._list_detect(name, exe)

    def _list_detect(self, name: str, exe: str) -> _Existing | None:
        """The entry from ``hermes mcp list``, for a Hermes whose ``config get`` failed."""
        # `hermes mcp list` has no --json: match the Name column, and tell
        # stdio from URL by the Transport column (the URL, or the command).
        proc = _capture(exe, ["mcp", "list"])
        if proc is None or proc.returncode != 0:
            return None
        for line in _ANSI_RE.sub("", proc.stdout).splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] == name:
                is_url = parts[1].startswith(("http://", "https://"))
                return _Existing("URL" if is_url else "stdio")
        return None

    def not_saved(self, name: str, exe: str, entry: _Entry, *, replaced: bool) -> str | None:
        # `hermes mcp add` exits 0 when the user cancels an overwrite, declines
        # to save after a failed probe, or hits a validation error; only the
        # entry read back tells, compared with what setup asked for.
        read, found = self._read_entry(name, exe)
        if not read:
            return self._list_not_saved(name, exe, entry)
        if found is None:
            return (
                f"Hermes has no {name} entry: `hermes mcp add` was cancelled or failed\n"
                "  there, so nothing was saved"
            )
        if not found.is_(entry):
            if not replaced:
                return _wrap(
                    f"Hermes's {name} entry ({found.kind}) is not the one setup asked for, "
                    "so nothing was saved"
                )
            return _wrap(
                f"Hermes's {name} entry is still the existing one ({found.kind}): Hermes "
                "keeps the existing entry when its overwrite prompt is declined, or when the "
                "add stops before saving, so nothing was saved. Re-run with --force and accept "
                "Hermes's overwrite prompt"
            )
        if entry.oauth and not found.oauth:
            return self._no_oauth(name, replaced=replaced)
        if not found.enabled:
            return self._disabled(name, oauth=entry.oauth)
        self._no_header = found.url is not None and not found.oauth and not found.authorization
        return None

    def _list_not_saved(self, name: str, exe: str, entry: _Entry) -> str | None:
        """:meth:`not_saved` from ``hermes mcp list``, for a Hermes whose ``config get`` failed.

        The list tells the form only: an ``--oauth`` entry cannot be checked.
        """
        found = self._list_detect(name, exe)
        if found is None or found.kind != ("stdio" if entry.command else "URL"):
            return (
                f"`hermes mcp list` shows no new {name} entry: `hermes mcp add` was\n"
                "  cancelled or failed there, so nothing was saved"
            )
        if entry.oauth:
            return _wrap(
                f"Setup could not read back Hermes's {name} entry (`hermes config get "
                f"mcp_servers.{name}` failed), so it cannot tell whether Hermes saved an OAuth "
                "entry it can sign in with: check it with that command or `hermes mcp list`"
            )
        return None

    def _no_oauth(self, name: str, *, replaced: bool) -> str:
        # Hermes writes `auth` only as `oauth` (a header entry is `headers`
        # alone). When it cannot set up OAuth it asks "Continue without
        # authentication?" (default yes) and saves the entry with no `auth`;
        # when its overwrite prompt is declined it keeps the existing entry.
        no_oauth = f"Hermes's {name} entry has no auth: oauth, so it cannot sign in to Kagura: "
        if replaced:
            return _wrap(
                f"{no_oauth}Hermes keeps the existing entry when its overwrite prompt is "
                "declined, and continues without authentication when it cannot set up "
                "OAuth. Re-run with --force and accept Hermes's overwrite prompt"
            )
        return _wrap(
            f"{no_oauth}Hermes continues without authentication when it cannot set up "
            "OAuth. Re-run with --force to replace it"
        )

    def _disabled(self, name: str, *, oauth: bool) -> str:
        # After a failed probe, "Save config anyway?" saves the entry with
        # enabled: false, which Hermes never connects to (for --oauth, the
        # sign-in did not finish; `hermes mcp login` does not turn it back on).
        turn_on = f"`hermes config set mcp_servers.{name}.enabled true`"
        if oauth:
            return _wrap(
                f"Hermes saved {name} disabled, since its sign-in or connection check did not "
                "finish, and it never connects to a disabled entry. Sign in with `hermes mcp "
                f"login {name}` (add --flow device on memory-cloud 0.78.0+ when the browser "
                f"cannot reach this host), then turn the entry on with {turn_on}"
            )
        return _wrap(
            f"Hermes saved {name} disabled, since its connection check did not pass, and it "
            f"never connects to a disabled entry. Check it with `hermes mcp test {name}`, then "
            f"turn it on with {turn_on}"
        )

    def _config_get(self, name: str, exe: str) -> tuple[bool, object]:
        """``mcp_servers.<name>`` from ``hermes config get``: ``(read, value)``.

        ``value`` is None when Hermes has no such entry (it exits 1 with
        "Config key not set"); ``read`` is False when the command failed
        otherwise. Hermes prints the value as one JSON line, credential-shaped
        keys masked; it goes straight to :class:`_HermesEntry`, which keeps no
        header value, env or other key.
        """
        proc = _capture(exe, ["config", "get", f"mcp_servers.{name}", "--json"])
        if proc is not None and proc.returncode == 1 and "Config key not set" in proc.stderr:
            return True, None
        if proc is None or proc.returncode != 0 or not proc.stdout.strip():
            return False, None
        try:
            return True, json.loads(proc.stdout.strip().splitlines()[-1])
        except ValueError:
            return False, None

    def add_args(self, name: str, entry: _Entry) -> list[str]:
        if entry.command is not None:
            # --args takes the rest of the command line, so it goes last.
            return ["mcp", "add", name, "--command", entry.command, "--args", *entry.args]
        assert entry.url is not None
        if entry.oauth:
            # The add's probe runs the browser sign-in, bounded by connect_timeout
            # (30 s unless set); give it the bound `hermes mcp login` uses.
            return [
                *("mcp", "add", name, "--url", entry.url, "--auth", "oauth"),
                *("--connect-timeout", str(HERMES_OAUTH_CONNECT_TIMEOUT_SEC)),
            ]
        return ["mcp", "add", name, "--url", entry.url, "--auth", "header"]

    @cached_property
    def _servers_key(self) -> tuple[_YamlServers | None, str | None]:
        """config.yaml's top-level ``mcp_servers:`` key, and why the file could not be read."""
        try:
            text = self.config_path().read_text(encoding="utf-8")
        except FileNotFoundError:
            return None, None
        except (OSError, ValueError) as e:
            return None, _exc_message(e)
        return _yaml_servers(text), None

    def block(self, name: str, entry: _Entry) -> str:
        q = json.dumps  # a JSON string is a YAML double-quoted scalar
        lines = [f"{name}:"]
        if entry.command is not None:
            args = ", ".join(q(a) for a in entry.args)
            lines += [f"  command: {q(entry.command)}", f"  args: [{args}]"]
        elif entry.oauth:
            lines += [f"  url: {q(entry.url)}", "  auth: oauth"]
        else:
            lines += [
                f"  url: {q(entry.url)}",
                "  headers:",
                f"    Authorization: {q(entry.auth_header())}",
            ]
        servers, _ = self._servers_key
        if servers is None:
            return "\n".join(["mcp_servers:", *(f"  {line}" for line in lines)])
        # The entry alone, to go under the key the file has.
        return "\n".join(f"{servers.indent}{line}" for line in lines)

    def block_target(self) -> str:
        servers, _ = self._servers_key
        return "it" if servers is None else "its mcp_servers: mapping"

    def block_notes(self, name: str) -> list[str]:
        servers, unread = self._servers_key
        where = _path_label(self.config_path())
        if unread is not None:
            return [
                f"Setup could not read {where} ({unread}): if it already has a\n"
                f"  top-level mcp_servers: key, put only the {name} entry under it."
            ]
        if servers is None:
            return []
        notes = [
            f"{where} already has a top-level mcp_servers: key, so only the\n"
            "  entry is printed: a second one would replace the first and every server under it."
        ]
        if servers.inline:
            notes.append(
                "Its mcp_servers value is written inline (flow style or null): rewrite it\n"
                f"  as a block mapping, one server per indented key, before adding {name}."
            )
        return notes

    def key_env(self, name: str, requested: str | None) -> str:
        return hermes_key_env(name)

    def key_note(self, entry: _Entry, *, ran: bool) -> str:
        env_file = _path_label(hermes_home() / ".env")
        if ran and self._no_header:
            # Hermes saves no headers when its "requires authentication?" is
            # answered no or the key is left empty.
            return _wrap(
                "Warning: Hermes saved the entry with no Authorization header (its key prompt "
                "was declined or left empty), so it connects without a key and Kagura refuses "
                "it. Re-run with --force and give Hermes the key when it asks"
            )
        if ran:
            return (
                f"Hermes asked for the key itself and keeps it in {env_file} as\n"
                f"  {entry.key_env}; setup never saw it."
            )
        return (
            f"Add `{entry.key_env}=<your-api-key>` to {env_file} with an editor:\n"
            "  the entry reads it from there, and setup never sees the key."
        )

    def sign_in_note(self, name: str, *, ran: bool) -> str:
        login = f"hermes mcp login {name}"
        if ran:
            first = (
                "Hermes signs in itself: `hermes mcp add` above started its sign-in when it "
                f"probed the server, with --connect-timeout {HERMES_OAUTH_CONNECT_TIMEOUT_SEC} "
                "(the bound `hermes mcp login` uses), which Hermes keeps as the entry's "
                f"connect_timeout. If it did not log in, run `{login}`"
            )
        else:
            first = f"Once the entry is in config.yaml, sign in with `{login}`"
        return (
            f"{first} (the browser flow). The sign-in redirects the browser to Hermes's "
            "loopback callback on this host; when the browser cannot reach it (a remote host), "
            "paste the redirect URL at Hermes's prompt, or (memory-cloud 0.78.0+) run "
            f"`{login} --flow device`, which signs in with a code at the server's /device page."
        )

    def token_store(self, name: str) -> str:
        return _path_label(hermes_home() / "mcp-tokens" / f"{name}.json")

    def verify_args(self, name: str) -> list[str]:
        return ["mcp", "test", name]

    def agents_md_path(self) -> Path | None:
        # Discovery starts from the current directory, as the Hermes CLI's does;
        # its gateway and cron start from terminal.cwd, so they pass a PATH.
        return hermes_context_file(Path.cwd())

    def no_agents_md_reason(self) -> str:
        rules = hermes_cursor_rules(Path.cwd())
        label = _path_label(rules) if rules is not None else "Cursor rules"
        return _wrap(
            f"Hermes loads only {label} here, and it loads the first context file type "
            "it finds: a new AGENTS.md would stop it loading that file. Name the file "
            "with --agents-md PATH"
        )

    def export_notes(self, path: Path) -> list[str]:
        return [
            "Hermes scans context files for prompt injection and skips a file it flags;\n"
            f"  if it reports {path.name} as blocked, delete the block between the\n"
            "  kagura-memory:guardrails markers."
        ]


def _expand_home(value: str) -> Path:
    """``value`` as a path, with a leading ``~`` or ``~/`` read as the home directory.

    ``~user/…`` is kept as written: ``Path.expanduser`` raises for a user
    that does not exist, which failed every ``setup openclaw`` with "Could
    not determine home directory" when such a value was in an OpenClaw path
    variable (#285).
    """
    separators = ("/", "\\") if os.name == "nt" else ("/",)
    if value == "~" or (value[:1] == "~" and value[1:2] in separators):
        return Path.home() / value[2:]
    return Path(value)


def _openclaw_env_path(var: str) -> Path | None:
    """An OpenClaw path variable as OpenClaw reads it: trimmed, a leading ``~`` expanded."""
    value = os.environ.get(var, "").strip()
    return _expand_home(value) if value else None


def openclaw_state_dir() -> Path:
    """``$OPENCLAW_STATE_DIR``, else ``~/.openclaw``: OpenClaw's config and ``.env``."""
    return _openclaw_env_path("OPENCLAW_STATE_DIR") or Path.home() / ".openclaw"


def openclaw_config_path() -> Path:
    """``$OPENCLAW_CONFIG_PATH``, else ``openclaw.json`` in :func:`openclaw_state_dir`."""
    return _openclaw_env_path("OPENCLAW_CONFIG_PATH") or openclaw_state_dir() / "openclaw.json"


def openclaw_workspace_dir() -> Path:
    """OpenClaw's default agent workspace, as its ``resolveDefaultAgentWorkspaceDir`` finds it.

    ``$OPENCLAW_WORKSPACE_DIR``, else ``workspace`` in :func:`openclaw_state_dir`.
    An ``agents.defaults.workspace`` in openclaw.json overrides both there; setup
    has no JSON5 reader, so ``--agents-md PATH`` names such a workspace.
    """
    return _openclaw_env_path("OPENCLAW_WORKSPACE_DIR") or openclaw_state_dir() / "workspace"


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
        # With auth "oauth" it ignores static headers, so the entry has none.
        server: dict[str, Any] = {"url": entry.url, "transport": "streamable-http"}
        if entry.oauth:
            server["auth"] = "oauth"
        else:
            server["headers"] = {"Authorization": entry.auth_header()}
        return server

    def add_args(self, name: str, entry: _Entry) -> list[str]:
        if entry.command is not None:
            args = [a for arg in entry.args for a in ("--arg", arg)]
            return ["mcp", "add", name, "--command", entry.command, *args]
        assert entry.url is not None
        if entry.oauth:
            # `mcp add` never probes an OAuth entry: it saves it for `mcp login`.
            return [
                *("mcp", "add", name, "--url", entry.url, "--transport", "streamable-http"),
                *("--auth", "oauth"),
            ]
        # --no-probe: the key is not in OpenClaw's .env yet.
        return [
            *("mcp", "add", name, "--url", entry.url, "--transport", "streamable-http"),
            *("--header", f"Authorization={entry.auth_header()}", "--no-probe"),
        ]

    def replace_args(self, name: str, entry: _Entry) -> list[str]:
        # `mcp add` refuses an existing name; `mcp set` replaces the entry.
        return ["mcp", "set", name, json.dumps(self.server(entry))]

    def block(self, name: str, entry: _Entry) -> str:
        return json.dumps({"mcp": {"servers": {name: self.server(entry)}}}, indent=2)

    def key_note(self, entry: _Entry, *, ran: bool) -> str:
        env_file = _path_label(openclaw_state_dir() / ".env")
        return (
            f"Add `{entry.key_env}=<your-api-key>` to {env_file} with an editor:\n"
            f"  the entry sends ${{{entry.key_env}}} (mcp.servers headers take no SecretRef),\n"
            "  and setup never sees the key."
        )

    def sign_in_note(self, name: str, *, ran: bool) -> str:
        # `mcp add` and `mcp set` save an OAuth entry without signing in.
        login = f"openclaw mcp login {name}"
        first = "Sign in" if ran else "Once the entry is in openclaw.json, sign in"
        return (
            f"{first} with `{login}`, then check it with the command below. The sign-in "
            "redirects the browser to OpenClaw's loopback callback on this host; when the "
            f"browser cannot reach it (a remote host), `{login} --code <code>` takes the code "
            "from the redirect."
        )

    def token_store(self, name: str) -> str:
        database = _path_label(openclaw_state_dir() / "state" / "openclaw.sqlite")
        return f"its state database ({database})"

    def verify_args(self, name: str) -> list[str]:
        return ["mcp", "doctor", name, "--probe"]

    def agents_md_path(self) -> Path | None:
        # The default agent workspace, which OpenClaw loads every session.
        return openclaw_workspace_dir() / "AGENTS.md"

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
    if profile == cf.default_profile and not os.environ.get("KAGURA_API_KEY", "").strip():
        return command
    return _on_profile(profile, command)


def _on_profile(profile: str, command: str) -> str:
    """``command`` run on ``profile``, even with ``KAGURA_API_KEY`` set here."""
    if os.environ.get("KAGURA_API_KEY", "").strip():
        return f"env -u KAGURA_API_KEY KAGURA_PROFILE={shlex.quote(profile)} {command}"
    return f"KAGURA_PROFILE={shlex.quote(profile)} {command}"


def _on_server(mcp_url: str, server: str) -> bool:
    """True when the stored ``mcp_url`` is on ``server``; False when it is not a URL at all."""
    try:
        return _deployment(mcp_url) == server
    except Exception:  # noqa: BLE001 - a hand-edited file can hold any JSON value
        return False


def _cli_chain_on(mcp_url: str) -> bool:
    """True when the kagura CLI's usual chain has a credential on ``mcp_url``'s server.

    Reads the credentials only, as :func:`_export_auth` does. It runs after
    the entry is written, so a chain it cannot resolve for any reason (no
    credential, a ``.kagura.json`` it cannot read or parse, or one that is
    not a config object) counts as no credential rather than failing setup.
    """
    try:
        auth = _resolve_auth(api_key=None, mcp_url=None, profile=None)
    except Exception:  # noqa: BLE001 - after the write, any failure means no credential
        return False
    return _on_server(auth.mcp_url, _deployment(mcp_url))


def _profiles_on(mcp_url: str) -> list[str]:
    """The stored OAuth profiles on ``mcp_url``'s server, by name; reads the file only.

    Like :func:`_cli_chain_on`, it runs after the write: a profile whose URL
    it cannot read is left out, and a file it cannot read holds none.
    """
    server = _deployment(mcp_url)
    try:
        profiles = load_credentials_file().profiles
    except Exception:  # noqa: BLE001 - after the write, an unusable file has no profile
        return []
    return sorted(name for name, creds in profiles.items() if _on_server(creds.mcp_url, server))


def _broken_config() -> str | None:
    """The ``.kagura.json`` every kagura command loads first, when it cannot be read or parsed.

    Returns:
        Its path and why, never its contents; None when it loads or there is none.
    """
    try:
        load_config()
        return None
    except OSError as e:
        why = e.strerror or type(e).__name__
    except ValueError:
        why = "not UTF-8 JSON"
    except Exception as e:  # noqa: BLE001 - runs after the write: named, never raised
        why = type(e).__name__
    local = Path(".kagura.json")
    return f"{_path_label(local.absolute()) if local.exists() else '~/.kagura.json'} ({why})"


def _write_export(
    h: _Harness,
    path: Path,
    context_id: str,
    auth: _StaticAuth | _OAuthAuth,
    refresh: str,
    *,
    applied: bool,
) -> None:
    """Fetch the export block and splice it into ``path``, replacing only the marked block.

    ``applied``: the harness command wrote the entry, rather than setup
    printing it for the user to add.

    Raises:
        click.ClickException: The fetch or the write failed (the MCP entry is
            already set up, or printed, by then).
    """
    label = _path_label(path)
    failed = (
        "The MCP entry is set up, but the AGENTS.md export failed"
        if applied
        else "The MCP entry is printed for you to add, but the AGENTS.md export failed"
    )
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
        # As `guardrails digest --out` does: an earlier block goes, so the
        # harness stops loading guardrails the server no longer serves. The
        # server sends an empty export only for an empty trusted set, never
        # on a failure (a denied context is a 404, handled above). Without a
        # block, neither the file nor its directory is created (#278).
        try:
            status = write_guardrail_block(path, "")
        except (OSError, ValueError) as e:
            raise click.ClickException(
                f"{failed}: {label}: {_exc_message(e)}; left unchanged"
            ) from e
        none = (
            f"\n  Context {context_id} has no tool guardrails this credential can see (none\n"
            "  marked, or the context is not trusted-tier)"
        )
        if status == "removed":
            click.echo(f"{none}: removed the earlier guardrail block from {label}.")
        else:
            click.echo(f"{none}: nothing was written to {label}.")
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        status = write_guardrail_block(path, digest.text)
    except (OSError, ValueError) as e:
        raise click.ClickException(f"{failed}: {label}: {_exc_message(e)}; left unchanged") from e
    done = "Already up to date:" if status == "unchanged" else "Wrote"
    click.echo(f"\n  {done} the guardrail block for context {context_id} in {label}")
    if h.agents_md_cap is not None:
        cap, unit = h.agents_md_cap
        # As written: read_text would fold CRLF, and undercount (#285).
        raw = path.read_bytes()
        size = len(raw) if unit == "bytes" else len(raw.decode("utf-8"))
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
    oauth: bool,
    interactive: bool,
) -> None:
    """Reject flag combinations that cannot work before anything is read (exit 2)."""
    if not _SERVER_NAME_RE.fullmatch(name):
        raise click.BadParameter(
            "use 1-64 letters, digits, '-' or '_', starting with a letter or digit",
            param_hint="'--name'",
        )
    if oauth and (not url_form or not mcp_url):
        raise click.UsageError(
            "--oauth needs --url-form and --mcp-url: the URL the harness signs in to, e.g. "
            "--url-form --oauth --mcp-url https://memory.kagura-ai.com/mcp/w/<workspace-id>."
        )
    if oauth and api_key_env is not None:
        raise click.UsageError(
            "--oauth and --api-key-env exclude each other: an --oauth entry has no key "
            "variable, since the harness signs in itself."
        )
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
        # It follows --url on the harness argv, where "--help" would read as an option.
        try:
            parts = urlsplit(mcp_url)
        except ValueError:  # e.g. an unclosed IPv6 bracket
            parts = None
        if parts is None or parts.scheme.lower() not in ("http", "https") or not parts.netloc:
            raise click.BadParameter(
                "use an https:// URL, e.g. https://memory.kagura-ai.com/mcp/w/<workspace-id>",
                param_hint="'--mcp-url'",
            )
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


async def _fetch_system_info(base_url: str) -> dict[str, Any] | None:
    async with make_oauth_client() as client:
        return await fetch_system_info(client, base_url)


def _shown_version(version: object) -> str:
    """A version a server reported, safe to print: short and printable, else its repr."""
    if isinstance(version, str) and version.isprintable() and len(version) <= 64:
        return version
    return reprlib.repr(version)


def _check_oauth_server(h: _Harness, mcp_url: str) -> None:
    """Stop unless the ``--oauth`` entry's server is memory-cloud 0.77.0+ (exit 1).

    One unauthenticated ``GET /api/v1/system/info`` on the entry's server,
    as ``kagura auth login --invite`` sends. Before 0.77.0 the harness's
    client registration is rejected, so the entry could never sign in. A
    version setup cannot read (no answer, not a 200, unparseable) stops it
    too: nothing confirms the registration would be accepted.

    Raises:
        click.ClickException: The server is older, or its version is unconfirmed.
    """
    deployment = _deployment(mcp_url)
    info = asyncio.run(_fetch_system_info(base_url_from_mcp(mcp_url)))
    version = None if info is None else info.get("version")
    meets = meets_minimum(version, HARNESS_OAUTH_MIN_SERVER_VERSION)
    if meets is True:
        click.echo(
            f"  {deployment} runs memory-cloud {_shown_version(version)}, which accepts\n"
            f"  {h.title}'s own client registration (0.77.0+)."
        )
        return
    if meets is False:
        why = f"{deployment} runs memory-cloud {_shown_version(version)}"
    elif info is None:
        why = (
            f"setup could not confirm the version of {deployment} (GET /api/v1/system/info "
            "did not answer 200 with a JSON object)"
        )
    else:
        reported = "no version" if version is None else _shown_version(version)
        why = (
            f"setup could not confirm the version of {deployment} (/api/v1/system/info "
            f"reports {reported})"
        )
    raise click.ClickException(
        _wrap(
            f"Nothing was written: --oauth needs memory-cloud 0.77.0+, and {why}. Before "
            f"0.77.0, dynamic client registration rejects {h.title}'s own client "
            "(memory-cloud#1657). Use the default stdio entry (--profile NAME, from "
            "`kagura auth login`) or --url-form with an API key instead."
        )
    )


def _codex_hook_warning(
    name: str, existing: _Existing | None, hooks_on: bool, *, oauth: bool, ask: bool
) -> None:
    """Say that an entry without a key leaves the Codex plugin's guardrail hooks without one.

    The stdio entry holds no bearer, and an ``--oauth`` entry's token stays in
    Codex's own store: the hooks report an OAuth entry as unsupported.

    Raises:
        click.ClickException: The hooks are on and the user chose not to go on.
    """
    if not hooks_on and not (existing is not None and existing.url_credential):
        return
    form = "an --oauth entry" if oauth else "the stdio entry"
    click.echo(
        "\n  Warning: the kagura-memory Codex plugin's guardrail hooks read their credential\n"
        "  only from a URL entry with a bearer (bearer_token_env_var, env_http_headers or\n"
        f"  http_headers), so with {form} they do nothing."
    )
    if not hooks_on:
        return
    data = _path_label(codex_home() / "plugins" / "data")
    click.echo(
        f"  They are turned on here for the {name} entry (a config.json under\n"
        f"  {data}/kagura-memory-*/): to keep them, re-run with --url-form and an API key"
        f"{' (no --oauth)' if oauth else ''}."
    )
    kind = "--oauth" if oauth else "stdio"
    if ask and not click.confirm(f"Write the {kind} entry anyway?", default=False):
        raise click.ClickException("Setup cancelled; nothing was written.")


def _drop_guardrails_context(h: _Harness, *, flag: bool, mcp_url: str | None) -> str | None:
    """``mcp_url`` for a Hermes/OpenClaw entry, without a ``?guardrails=`` context.

    Neither reads MCP instructions, so a context there changes nothing, and a
    context id is never written into their entry: neither the ``--guardrails``
    context (``flag``) nor any ``guardrails`` value in ``mcp_url``. One
    warning names what was dropped. ``off``, when it comes first in
    ``mcp_url`` (the value the server reads), is kept as asked, alone and with
    a warning of its own: it removes the ``get_context_info`` block, the lane
    they have.

    Returns:
        ``mcp_url`` without a guardrails context; None without ``--url-form``.
    """
    dropped = ["--guardrails"] if flag else []
    if mcp_url is not None and mcp_url_guardrails_off(mcp_url):
        click.echo(
            f"\n  Warning: --mcp-url has ?guardrails=off, which removes the guardrails block\n"
            f"  from get_context_info: {h.title} then gets no guardrails from Kagura."
        )
        mcp_url = mcp_url_with_query(mcp_url, guardrails="off")
    elif mcp_url is not None:
        kept = mcp_url_without_query_param(mcp_url, "guardrails")
        if kept != mcp_url:
            dropped.append("the ?guardrails= value in --mcp-url")
            mcp_url = kept
    if dropped:
        has, is_ = ("have", "are") if len(dropped) > 1 else ("has", "is")
        warning = (
            f"Warning: {h.title} does not read MCP instructions, so {' and '.join(dropped)} "
            f"{has} no effect there and {is_} not written. Guardrails reach {h.title} through "
            "get_context_info (on by default) and the AGENTS.md export (--agents-md)."
        )
        # One message for one or both sources, so it is wrapped rather than laid out.
        lines = textwrap.wrap(warning, 80, break_on_hyphens=False)
        click.echo("\n" + "\n".join(f"  {line}" for line in lines))
    return mcp_url


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
    h: _Harness, exe: str | None, entry: _Entry, non_interactive: bool, interactive: bool
) -> str | None:
    """Why setup prints the block instead of running the harness command, or None."""
    if exe is None:
        return f"`{h.cli}` is not on PATH"
    if h.attached_add(entry) and not interactive:
        why = "-y was given" if non_interactive else "stdin is not a terminal"
        what = "is interactive" if h.interactive_add else "starts the sign-in"
        return f"`{h.cli} mcp add` {what} and {why}"
    return None


def _run_or_fail(h: _Harness, exe: str, args: list[str], *, attached: bool, note: str = "") -> None:
    """Run ``<harness> <args>`` (attached to the terminal when it prompts).

    Raises:
        click.ClickException: It failed; the message names the subcommand,
            followed by ``note``.
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
        raise click.ClickException(
            f"`{name}` failed: timed out after {e.timeout:g}s{note}"
        ) from None
    except (OSError, subprocess.SubprocessError) as e:
        raise click.ClickException(f"`{name}` failed: {_exc_message(e)}{note}") from None
    if proc.returncode != 0:
        detail = "" if attached else (proc.stderr or proc.stdout or "").strip()
        raise click.ClickException(
            f"`{name}` failed: {detail or f'exit code {proc.returncode}'}{note}"
        )


def _echo_block(block: str) -> None:
    for line in block.splitlines():
        click.echo(f"    {line}")


def _preview_command(
    context_id: str, entry: _Entry, profile: str | None, cf: CredentialsFile | None
) -> str | None:
    """``kagura guardrails digest <ctx> --target instructions`` on the entry's own credential.

    An ``--oauth`` entry's token stays with the harness, so its preview runs
    on the profile (on the entry's server, as setup checked) or on the CLI's
    usual chain when that is on the entry's server too.

    Returns:
        The command; None for an ``--oauth`` entry without ``--profile`` when
        the chain's credential is not on the entry's server. Pinning the
        server with ``KAGURA_MCP_URL`` would send ``KAGURA_API_KEY``, a key
        for another server, to it.
    """
    args = ["guardrails", "digest", context_id, "--target", "instructions"]
    if entry.url is None:
        return _kagura_command(args, profile, cf)
    if entry.oauth:
        if profile is None and not _cli_chain_on(entry.url):
            return None
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
    oauth: bool,
    force: bool,
    non_interactive: bool,
    dry_run: bool,
) -> None:
    """Set up the ``name`` MCP entry of ``harness``, and the AGENTS.md export if asked.

    ``guardrails`` must already be normalized (``off`` or a canonical UUID).
    ``agents_md`` is None without ``--agents-md`` and ``""`` for the
    harness's default file. Setup prompts only with a terminal on stdin and
    without ``-y``; otherwise it behaves as ``-y`` does. With ``oauth`` (and
    ``url_form``), the entry has no key and the harness signs in itself; the
    server must be memory-cloud 0.77.0+, which a real run checks before it
    detects, runs or writes anything. See the module docstring for the rules.

    Raises:
        click.UsageError: A flag combination that cannot work (exit 2).
        click.ClickException: A missing profile or ``kagura-mcp``, a server
            ``oauth`` cannot use, an existing entry without ``force``, or a
            failed check or harness command.
    """
    h = HARNESSES[harness]()
    interactive = not non_interactive and _stdin_is_tty()
    # The entry gets the URL the HTTPS check passed, not one with padding or
    # control characters a harness might keep.
    mcp_url = normalize_url(mcp_url) if mcp_url is not None else None
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
        oauth=oauth,
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
    if oauth:
        assert mcp_url is not None
        if dry_run:
            click.echo(
                f"  The real run first checks that {_deployment(mcp_url)} runs memory-cloud\n"
                "  0.77.0+ (GET /api/v1/system/info); this dry run sends no request."
            )
        else:
            _check_oauth_server(h, mcp_url)

    # 1. What the harness has now (read-only)
    existing = h.detect(name, exe)
    if existing is not None:
        click.echo(f"  Existing {name} entry: {existing.kind}")
    elif exe is None and not h.detects_from_config:
        click.echo(f"  `{h.cli}` is not on PATH, so setup cannot look for a {name} entry.")
    else:
        click.echo(f"  No {name} entry yet.")
    hooks_on = h.plugin_hooks_on(name)
    if not url_form or oauth:
        # No question when the existing-entry stop below ends the run anyway.
        h.warn_keyless_entry(
            name, existing, hooks_on, oauth=oauth, ask=ask and (existing is None or force)
        )
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
        # _check_flags refused "off" here, so a --guardrails value is a context.
        mcp_url = _drop_guardrails_context(h, flag=guardrails is not None, mcp_url=mcp_url)
        guardrails = None
    lane_from_context = False
    if h.reads_instructions and guardrails is None:
        # The hooks cannot read an --oauth entry, so "off" would leave only the
        # get_context_info block and nothing to deliver the rest.
        if url_form and hooks_on and not oauth:
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
    # Only where it is used: for Hermes it reads the context files.
    wants_default = agents_md == "" or (agents_md is None and not h.reads_instructions)
    default_path = h.agents_md_path() if wants_default else None
    if agents_md is not None:
        export_path = _expand_home(agents_md) if agents_md else default_path
        if export_path is None:
            raise click.ClickException(f"Nothing was written: {h.no_agents_md_reason()}.")
    elif not h.reads_instructions and ask and can_pick and default_path is not None:
        offered = True
        path = default_path
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
        if oauth:
            entry = _Entry(url=url, oauth=True)
        else:
            entry = _Entry(url=url, key_env=h.key_env(name, api_key_env))
    command = h.replace_args(name, entry) if existing is not None else h.add_args(name, entry)
    reason = _print_reason(h, exe, entry, non_interactive, interactive)

    click.echo("")
    show_block = dry_run or reason is not None
    if show_block:
        for note in h.block_notes(name):
            click.echo(f"  {note}")
    if reason is None:
        verb = "Would run" if dry_run else "Running"
        if dry_run and existing is not None and not force:
            verb = "With --force, would run"
        click.echo(f"  {verb}: {shlex.join([h.cli, *command])}")
    else:
        replace = " in place of the existing one" if existing is not None else ""
        click.echo(
            f"  Setup does not edit {where} itself ({reason}).\n"
            f"  Add this {name} entry to {h.block_target()}{replace}:"
        )
    if show_block:
        click.echo("")
        _echo_block(h.block(name, entry))
    if dry_run:
        _echo_dry_run_export(
            h, export_path, default_path, export_context, non_interactive, interactive
        )
        return

    # 6. Write through the harness
    if reason is None:
        assert exe is not None
        _run_or_fail(
            h,
            exe,
            command,
            attached=h.attached_add(entry),
            note=h.add_failure_note(name, entry),
        )
        problem = h.not_saved(name, exe, entry, replaced=existing is not None)
        if problem is not None:
            skipped = "; setup skipped the AGENTS.md export" if export_path is not None else ""
            raise click.ClickException(f"{problem}{skipped}.")
        click.echo(f"  Done: {h.cli} wrote {name} to {where}.")

    # 7. What the user does next
    click.echo("")
    if entry.oauth:
        click.echo(f"  {h.login_note(name, ran=reason is None)}")
    elif entry.url is not None:
        click.echo(f"  {h.key_note(entry, ran=reason is None)}")
    if h.reads_instructions and guardrails not in (None, "off"):
        assert guardrails is not None
        command = _preview_command(guardrails, entry, profile, cf)
        if command is None:
            # An --oauth entry without --profile, and the CLI's usual chain is
            # not on its server: a stored profile there, else a login there.
            assert entry.url is not None
            deployment = _deployment(entry.url)
            on_server = _profiles_on(entry.url)
            if on_server:
                preview = (
                    "The kagura CLI's usual credential is not on\n"
                    f"  {deployment}; preview it on a profile there ({', '.join(on_server)})\n"
                    "  (Codex gets what the account it signed in with can read):"
                )
            else:
                preview = (
                    "The kagura CLI's usual credential is not on\n"
                    f"  {deployment}, and no profile is: log in there with\n"
                    f"  `kagura auth login --server {deployment} --profile NAME`,\n"
                    "  then preview it (Codex gets what the account it signed in with can read):"
                )
            digest = ["guardrails", "digest", guardrails, "--target", "instructions"]
            profile_name = on_server[0] if on_server else "NAME"
            command = _on_profile(profile_name, shlex.join(["kagura", *digest]))
        elif entry.oauth:
            preview = (
                "Preview it on the kagura CLI's credential\n"
                "  (Codex gets what the account it signed in with can read):"
            )
        else:
            preview = "Preview what it sends:"
        fails = ""
        broken = _broken_config()
        if broken is not None:
            fails = "\n  " + _wrap(
                f"The preview fails until {broken} is fixed or removed: every kagura command "
                "reads it first."
            )
        click.echo(
            "  Codex should get the tool guardrail digest of context\n"
            f"  {guardrails} in the MCP instructions when it connects.\n"
            "  The server sends only its base text instead when the entry's credential\n"
            "  cannot read that context, the context has no guardrails, or the deployment\n"
            f"  turns the digest off. {preview}\n"
            f"    {command}{fails}\n"
            "  Use a context whose editor list you control: every editor's guardrail\n"
            "  summaries reach the model."
        )
    for note in h.notes():
        click.echo(f"  {note}")
    click.echo(f"  Check it with: {shlex.join([h.cli, *h.verify_args(name)])}")
    # No hint when no default file fits: a new one would displace the user's.
    if (
        export_path is None
        and not h.reads_instructions
        and not offered
        and default_path is not None
    ):
        click.echo(
            "  Re-run with --agents-md --context-id <id> to put a snapshot of a context's\n"
            f"  tool guardrails into {_path_label(default_path)},\n"
            f"  which {h.title} loads every session."
        )

    if export_path is not None:
        assert export_context is not None and export_auth is not None
        refresh = _kagura_command(
            ["guardrails", "digest", export_context, "--out", str(export_path)], profile, cf
        )
        _write_export(h, export_path, export_context, export_auth, refresh, applied=reason is None)


def _echo_dry_run_export(
    h: _Harness,
    path: Path | None,
    default_path: Path | None,
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
        if default_path is None:
            click.echo(f"\n  AGENTS.md: not offered. {h.no_agents_md_reason()}.")
            return
        if interactive:
            when = "offered when setup runs"
        else:
            when = "not offered with -y" if non_interactive else "not offered without a terminal"
        click.echo(f"\n  AGENTS.md: {_path_label(default_path)} ({when}; --agents-md writes it)")
