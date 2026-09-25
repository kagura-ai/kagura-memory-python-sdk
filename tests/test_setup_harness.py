"""``kagura setup codex|hermes|openclaw`` (#260).

Every test runs against a temporary home: ``HOME`` points into ``tmp_path``
and ``CODEX_HOME``, ``HERMES_HOME``, ``OPENCLAW_STATE_DIR``,
``OPENCLAW_CONFIG_PATH`` and ``OPENCLAW_WORKSPACE_DIR`` are unset unless a
test sets them, the credentials file is a temporary one, no harness CLI is on
the (patched) ``PATH`` unless a test puts it there, ``subprocess.run`` is
a recorder and ``GET /api/v1/system/info`` goes to an in-process mock —
nothing reads or changes the developer's real harness configuration, and
nothing reaches a server.
"""

from __future__ import annotations

import itertools
import json
import os
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from click.testing import CliRunner

from kagura_memory import setup_harness
from kagura_memory._auth import _OAuthAuth, _resolve_profile_auth
from kagura_memory.auth import credentials
from kagura_memory.auth.credentials import CredentialsFile, save_credentials_file
from kagura_memory.cli import main
from kagura_memory.exceptions import KaguraNotFoundError
from kagura_memory.models import GuardrailDigest
from kagura_memory.setup_harness import hermes_context_file, hermes_home, hermes_key_env

from .conftest import make_oauth_creds

# Captured before the autouse fixtures replace them.
REAL_PROXY_PATH = setup_harness._proxy_path
REAL_HARNESS_EXECUTABLE = setup_harness._harness_executable
REAL_STDIN_IS_TTY = setup_harness._stdin_is_tty
REAL_FETCH_DIGEST = setup_harness._fetch_digest
REAL_RUN = subprocess.run

PROXY = "/opt/kagura/bin/kagura-mcp"
CTX = "11111111-2222-3333-4444-555555555555"
OTHER_CTX = "99999999-8888-7777-6666-555555555555"
ACCESS_TOKEN = "atok-SECRET-access-token-value"
API_KEY = "kagura_SECRET_api_key_value"
MCP_URL = "https://memory.kagura-ai.com/mcp/w/ws-1"
CONTEXTS = {"count": 2, "contexts": [{"id": CTX, "name": "proj"}, {"id": OTHER_CTX, "name": "ops"}]}
EXPORT_BLOCK = (
    f"<!-- kagura-memory:guardrails begin context={CTX} tool_triggered_version=abc123 -->\n"
    "- (aaaaaaaa) Squash-merge only after the head SHA matches\n"
    "<!-- kagura-memory:guardrails end -->\n"
)


class Recorder:
    """Stands in for ``subprocess.run``: records argv, answers read-only detection."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        #: stdout of `<cli> mcp list` / `mcp show <name> --json`, per CLI.
        self.detect_out: dict[str, str] = {}
        #: Return code per argv[1:3] joined, e.g. {"mcp add": 1}.
        self.returncodes: dict[str, int] = {}
        #: False: `hermes mcp add` exits 0 without saving (a cancelled overwrite).
        self.hermes_saves = True
        #: False: Hermes cannot set up OAuth and saves the entry with no `auth`.
        self.hermes_oauth_ok = True
        #: False: the header key prompt is declined, and the entry has no `headers`.
        self.hermes_header_ok = True
        #: False: the add's probe (the sign-in) fails, and "Save config anyway"
        #: saves the entry with `enabled: false`.
        self.hermes_probe_ok = True
        #: Hermes's `mcp_servers` entries, as `hermes config get` reads them back.
        self.hermes_entries: dict[str, dict[str, Any]] = {}
        #: The keys whose `hermes config get` fails, e.g. {"url"}; "" is the
        #: whole entry (an older Hermes, which setup reads with `mcp list`).
        self.hermes_unreadable: set[str] = set()

    def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append((list(argv), kwargs))
        cli = Path(argv[0]).name
        sub = " ".join(argv[1:3])
        if sub == "mcp list" or (sub == "mcp show" and "--json" in argv):
            out = self.detect_out.get(cli)
            return subprocess.CompletedProcess(argv, 0 if out else 1, out or "", "")
        if sub == "config get":
            _, server, *rest = argv[3].split(".", 2)
            key = rest[0] if rest else ""
            code = self.returncodes.get(sub) or (2 if key in self.hermes_unreadable else 0)
            if code:
                return subprocess.CompletedProcess(argv, code, "", "boom")
            entry = self.hermes_entries.get(server)
            value = entry if not key else (entry or {}).get(key)
            if value is None:
                return subprocess.CompletedProcess(argv, 1, "", f"Config key not set: {argv[3]}")
            return subprocess.CompletedProcess(argv, 0, json.dumps(value) + "\n", "")
        code = self.returncodes.get(sub, 0)
        if cli == "hermes" and sub == "mcp add" and code == 0 and self.hermes_saves:
            self._hermes_add(argv)
        return subprocess.CompletedProcess(argv, code, "", "boom" if code else "")

    def _hermes_add(self, argv: list[str]) -> None:
        """Save the entry as `hermes mcp add` writes it to config.yaml."""
        name = argv[3]
        auth = argv[argv.index("--auth") + 1] if "--auth" in argv else None
        if "--url" in argv:
            transport = argv[argv.index("--url") + 1]
            entry: dict[str, Any] = {"url": transport}
            if auth == "oauth" and self.hermes_oauth_ok:
                entry["auth"] = "oauth"
            if auth == "header" and self.hermes_header_ok:
                entry["headers"] = {"Authorization": f"Bearer ${{{hermes_key_env(name)}}}"}
        else:
            transport = argv[argv.index("--command") + 1]
            entry = {"command": transport, "args": argv[argv.index("--args") + 1 :]}
        if not self.hermes_probe_ok:
            entry["enabled"] = False
        self.hermes_entries[name] = entry
        self.detect_out["hermes"] = f"  {name}    {transport}   all\n"

    def argvs(self) -> list[list[str]]:
        return [argv for argv, _ in self.calls]

    def mutating(self) -> list[list[str]]:
        """Every call that is not read-only detection."""
        return [
            a
            for a in self.argvs()
            if " ".join(a[1:3]) not in ("mcp list", "config get")
            and not (a[1:3] == ["mcp", "show"] and "--json" in a)
        ]


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch, isolated_kagura_credentials):
    """Temporary home and harness homes; real credentials file in tmp."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    for var in (
        "CODEX_HOME",
        "HERMES_HOME",
        "OPENCLAW_STATE_DIR",
        "OPENCLAW_CONFIG_PATH",
        "OPENCLAW_WORKSPACE_DIR",
    ):
        monkeypatch.delenv(var, raising=False)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    cf = CredentialsFile()
    cf.set_profile("default", make_oauth_creds(access_token=ACCESS_TOKEN))
    cf.set_profile("work", make_oauth_creds(access_token=ACCESS_TOKEN))
    save_credentials_file(cf)
    return home


@pytest.fixture(autouse=True)
def recorder(monkeypatch) -> Recorder:
    rec = Recorder()
    monkeypatch.setattr(setup_harness.subprocess, "run", rec)
    return rec


@pytest.fixture(autouse=True)
def proxy(monkeypatch):
    monkeypatch.setattr(setup_harness, "_proxy_path", lambda: PROXY)


class SystemInfo:
    """``GET /api/v1/system/info`` for ``--oauth``: tests set ``version``, ``status`` or ``exc``."""

    def __init__(self) -> None:
        self.version: str | None = "0.77.0"
        self.status = 200
        self.exc: type[Exception] | None = None
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.exc is not None:
            raise self.exc("connection refused", request=request)
        if request.url.path != "/api/v1/system/info" or self.status != 200:
            return httpx.Response(self.status if self.status != 200 else 404, json={})
        body: dict[str, Any] = {"name": "Kagura Memory Cloud", "features": {}}
        if self.version is not None:
            body["version"] = self.version
        return httpx.Response(200, json=body)


@pytest.fixture(autouse=True)
def system_info(monkeypatch) -> SystemInfo:
    server = SystemInfo()
    monkeypatch.setattr(
        setup_harness,
        "make_oauth_client",
        lambda *a, **k: httpx.AsyncClient(transport=httpx.MockTransport(server)),
    )
    return server


@pytest.fixture
def on_path(monkeypatch):
    """Put harness CLIs on the (patched) PATH: ``on_path("codex")``."""
    present: set[str] = set()
    monkeypatch.setattr(
        setup_harness,
        "_harness_executable",
        lambda cli: f"/usr/bin/{cli}" if cli in present else None,
    )
    return present.add


@pytest.fixture(autouse=True)
def no_harness_on_path(monkeypatch):
    monkeypatch.setattr(setup_harness, "_harness_executable", lambda cli: None)


@pytest.fixture(autouse=True)
def connection():
    with patch("kagura_memory.setup_claude._test_connection") as conn:
        conn.return_value = CONTEXTS
        yield conn


@pytest.fixture(autouse=True)
def digest(monkeypatch):
    """The export block the server would render; tests reassign ``.text`` / ``.error``."""

    class Fake:
        text = EXPORT_BLOCK
        error: Exception | None = None
        #: (profile name, or the static key's source; context id)
        calls: list[tuple[str | None, str]] = []

    async def fetch(auth, context_id):
        who = auth.oauth._state.profile_name if isinstance(auth, _OAuthAuth) else auth.source
        Fake.calls.append((who, context_id))
        if Fake.error is not None:
            raise Fake.error
        return GuardrailDigest(
            context_id=context_id, target="export", text=Fake.text, tool_triggered_version="abc"
        )

    Fake.calls = []
    monkeypatch.setattr(setup_harness, "_fetch_digest", fetch)
    return Fake


@pytest.fixture
def tty(monkeypatch):
    monkeypatch.setattr(setup_harness, "_stdin_is_tty", lambda: True)


def run(*args: str, input: str | None = None):
    return CliRunner().invoke(main, ["setup", *args], input=input)


def flat(output: str) -> str:
    """``output`` on one line, for a message wrapped to fit the terminal."""
    return " ".join(output.split())


def printed_json(output: str) -> Any:
    """The JSON block setup printed (the first ``{`` onwards)."""
    return json.JSONDecoder().raw_decode(output[output.index("{") :])[0]


def codex_config(home: Path) -> Path:
    path = home / ".codex" / "config.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def turn_on_codex_hooks(home: Path, **settings: Any) -> None:
    data = home / ".codex" / "plugins" / "data" / "kagura-memory-kagura-memory-cloud"
    data.mkdir(parents=True)
    body = json.dumps({"context_id": CTX, **settings})
    (data / "config.json").write_text(body, encoding="utf-8")


# =============================================================================
# Codex
# =============================================================================


class TestCodex:
    def test_stdio_add_argv(self, on_path, recorder):
        on_path("codex")
        result = run("codex", "--profile", "default", "-y")
        assert result.exit_code == 0, result.output
        assert recorder.mutating() == [
            ["/usr/bin/codex", "mcp", "add", "kagura-memory", "--", PROXY, "--profile", "default"]
        ]
        assert "codex mcp get kagura-memory" in result.output

    def test_context_id_becomes_the_guardrails_lane(self, on_path, recorder):
        on_path("codex")
        result = run("codex", "--profile", "work", "--context-id", "proj", "-y")
        assert result.exit_code == 0, result.output
        [argv] = recorder.mutating()
        assert argv[argv.index("--") + 1 :] == [PROXY, "--profile", "work", "--guardrails", CTX]
        assert "editor list you control" in result.output
        assert f"guardrail digest of context\n  {CTX} in the MCP" in result.output
        assert (
            f"KAGURA_PROFILE=work kagura guardrails digest {CTX} --target instructions"
            in result.output
        )

    def test_the_preview_names_a_kagura_json_it_cannot_parse(self, env, on_path, recorder):
        # Setup runs on the profile alone, but the preview's kagura command
        # loads ~/.kagura.json (no ./.kagura.json here) before anything else.
        (env / ".kagura.json").write_bytes(b"\xff{")
        on_path("codex")
        result = run("codex", "--profile", "work", "--context-id", CTX, "-y")
        assert result.exit_code == 0, result.output
        assert (
            "The preview fails until ~/.kagura.json (not UTF-8 JSON) is fixed or removed: "
            "every kagura command reads it first."
        ) in flat(result.output)

    def test_url_form_preview_runs_on_the_entry_key(self, on_path, recorder):
        on_path("codex")
        result = run(
            "codex",
            *("--url-form", "--mcp-url", MCP_URL, "--api-key-env", "KAGURA_CODEX_KEY"),
            *("--context-id", CTX, "-y"),
        )
        assert result.exit_code == 0, result.output
        assert (
            'KAGURA_API_KEY="${KAGURA_CODEX_KEY}" '
            f"KAGURA_MCP_URL='{MCP_URL}?guardrails={CTX}' "
            f"kagura guardrails digest {CTX} --target instructions"
        ) in result.output

    def test_explicit_guardrails_wins_over_context_id(self, on_path, recorder):
        on_path("codex")
        result = run(
            "codex", "--profile", "default", "--guardrails", "off", "--context-id", CTX, "-y"
        )
        assert result.exit_code == 0, result.output
        [argv] = recorder.mutating()
        assert argv[-2:] == ["--guardrails", "off"]

    def test_invalid_guardrails_is_a_usage_error(self):
        result = run("codex", "--profile", "default", "--guardrails", "nope", "-y")
        assert result.exit_code == 2

    def test_existing_entry_blocks_without_force(self, env, on_path, recorder):
        on_path("codex")
        codex_config(env).write_text(
            '[mcp_servers.kagura-memory]\ncommand = "kagura-mcp"\nargs = ["--profile", "x"]\n',
            encoding="utf-8",
        )
        result = run("codex", "--profile", "default", "-y")
        assert result.exit_code == 1
        assert "stdio (kagura-mcp)" in result.output
        assert "--force" in result.output
        assert recorder.mutating() == []

    def test_force_replacing_url_bearer_entry_warns_about_plugin_hooks(
        self, env, on_path, recorder
    ):
        on_path("codex")
        codex_config(env).write_text(
            "[mcp_servers.kagura-memory]\n"
            'url = "https://memory.kagura-ai.com/mcp/w/ws"\n'
            'bearer_token_env_var = "KAGURA_API_KEY"\n',
            encoding="utf-8",
        )
        result = run("codex", "--profile", "default", "--force", "-y")
        assert result.exit_code == 0, result.output
        assert "bearer token from an environment variable" in result.output
        assert "guardrail hooks read their credential" in result.output
        # `codex mcp add` overwrites an entry of the same name: no remove first (#274).
        assert recorder.mutating() == [
            ["/usr/bin/codex", "mcp", "add", "kagura-memory", "--", PROXY, "--profile", "default"],
        ]

    def test_failed_force_add_leaves_the_previous_entry(self, env, on_path, recorder):
        on_path("codex")
        codex_config(env).write_text(
            '[mcp_servers.kagura-memory]\ncommand = "kagura-mcp"\n', encoding="utf-8"
        )
        recorder.returncodes["mcp add"] = 1
        result = run("codex", "--profile", "default", "--force", "-y")
        assert result.exit_code == 1
        assert "codex mcp add` failed: boom" in result.output
        assert "removed" not in result.output
        assert [argv[1:3] for argv in recorder.mutating()] == [["mcp", "add"]]

    def test_dry_run_with_an_existing_entry_shows_a_single_add(self, env, on_path, recorder):
        on_path("codex")
        codex_config(env).write_text(
            '[mcp_servers.kagura-memory]\ncommand = "kagura-mcp"\n', encoding="utf-8"
        )
        result = run("codex", "--profile", "default", "--dry-run")
        assert result.exit_code == 0, result.output
        assert (
            f"With --force, would run: codex mcp add kagura-memory -- {PROXY} --profile default"
            in result.output
        )
        assert "mcp remove" not in result.output
        assert recorder.mutating() == []

    def test_name_may_start_with_a_digit(self, on_path, recorder):
        on_path("codex")
        result = run("codex", "--profile", "default", "--name", "9lives", "-y")
        assert result.exit_code == 0, result.output
        assert recorder.mutating()[0][1:4] == ["mcp", "add", "9lives"]

    @pytest.mark.parametrize(
        "given",
        [
            f"  {MCP_URL}\n",
            # What the HTTPS check drops, the entry does not keep either.
            f"\x01{MCP_URL}\x1f",
            MCP_URL.replace("https", "ht\ttps").replace("/mcp", "/m\ncp"),
        ],
        ids=["whitespace", "c0-controls", "tab-and-newline"],
    )
    def test_url_form_mcp_url_is_the_url_the_https_check_read(self, on_path, recorder, given):
        on_path("codex")
        result = run("codex", "--url-form", "--mcp-url", given, "-y")
        assert result.exit_code == 0, result.output
        [argv] = recorder.mutating()
        assert argv[argv.index("--url") + 1] == MCP_URL

    def test_url_form_keeps_a_guardrails_context_already_in_the_url(self, on_path, recorder):
        # Codex reads MCP instructions: the context the URL names still does something.
        on_path("codex")
        url = f"{MCP_URL}?guardrails={CTX}"
        result = run("codex", "--url-form", "--mcp-url", url, "-y")
        assert result.exit_code == 0, result.output
        [argv] = recorder.mutating()
        assert argv[argv.index("--url") + 1] == url
        assert "not written" not in result.output

    def test_static_header_entry_is_described_not_echoed(self, env, on_path):
        on_path("codex")
        codex_config(env).write_text(
            "[mcp_servers.kagura-memory]\n"
            'url = "https://x/mcp"\n'
            f'http_headers = {{ Authorization = "Bearer {API_KEY}" }}\n',
            encoding="utf-8",
        )
        result = run("codex", "--profile", "default", "-y")
        assert result.exit_code == 1
        assert "static Authorization header" in result.output
        assert API_KEY not in result.output

    def test_url_form_argv(self, on_path, recorder):
        on_path("codex")
        result = run("codex", "--url-form", "--mcp-url", MCP_URL, "-y")
        assert result.exit_code == 0, result.output
        assert recorder.mutating() == [
            [
                "/usr/bin/codex",
                *("mcp", "add", "kagura-memory", "--url", MCP_URL),
                *("--bearer-token-env-var", "KAGURA_API_KEY"),
            ]
        ]
        assert "export KAGURA_API_KEY=<your-api-key>" in result.output

    def test_url_form_context_goes_in_the_query(self, on_path, recorder):
        on_path("codex")
        result = run(
            "codex",
            *("--url-form", "--mcp-url", MCP_URL, "--api-key-env", "KAGURA_CODEX_KEY"),
            *("--context-id", CTX, "-y"),
        )
        assert result.exit_code == 0, result.output
        [argv] = recorder.mutating()
        assert argv[argv.index("--url") + 1] == f"{MCP_URL}?guardrails={CTX}"
        assert argv[-1] == "KAGURA_CODEX_KEY"

    def test_url_form_with_hooks_on_defaults_guardrails_off(self, env, on_path, recorder):
        on_path("codex")
        turn_on_codex_hooks(env)
        result = run("codex", "--url-form", "--mcp-url", MCP_URL, "--context-id", CTX, "-y")
        assert result.exit_code == 0, result.output
        [argv] = recorder.mutating()
        assert argv[argv.index("--url") + 1] == f"{MCP_URL}?guardrails=off"

    def test_stdio_with_hooks_on_keeps_stdio_under_y_with_a_note(self, env, on_path, recorder):
        on_path("codex")
        turn_on_codex_hooks(env)
        result = run("codex", "--profile", "default", "-y")
        assert result.exit_code == 0, result.output
        assert "re-run with --url-form" in result.output
        assert recorder.mutating()[0][-3:] == [PROXY, "--profile", "default"]

    def test_stdio_with_hooks_on_asks_and_can_be_declined(self, env, on_path, recorder, tty):
        on_path("codex")
        turn_on_codex_hooks(env)
        result = run("codex", "--profile", "default", input="n\n")
        assert result.exit_code == 1
        assert "Write the stdio entry anyway?" in result.output
        assert recorder.mutating() == []

    def test_stdio_with_hooks_on_without_a_terminal_keeps_stdio(self, env, on_path, recorder):
        on_path("codex")
        turn_on_codex_hooks(env)
        result = run("codex", "--profile", "default")  # no -y, stdin at EOF
        assert result.exit_code == 0, result.output
        assert "re-run with --url-form" in result.output
        assert "Write the stdio entry anyway?" not in result.output
        assert recorder.mutating()[0][-3:] == [PROXY, "--profile", "default"]

    def test_hooks_question_is_skipped_when_the_existing_entry_stops_setup(
        self, env, on_path, recorder, tty
    ):
        on_path("codex")
        turn_on_codex_hooks(env)
        codex_config(env).write_text(
            '[mcp_servers.kagura-memory]\nurl = "https://x/mcp"\nbearer_token_env_var = "K"\n',
            encoding="utf-8",
        )
        result = run("codex", "--profile", "default")
        assert result.exit_code == 1
        assert "guardrail hooks read their credential" in result.output
        assert "Write the stdio entry anyway?" not in result.output
        assert "re-run with --force" in result.output

    def test_hooks_reading_another_table_leave_this_entry_alone(self, env, on_path, recorder):
        on_path("codex")
        turn_on_codex_hooks(env, mcp_server="kagura-work")
        result = run("codex", "--url-form", "--mcp-url", MCP_URL, "--context-id", CTX, "-y")
        assert result.exit_code == 0, result.output
        [argv] = recorder.mutating()
        assert argv[argv.index("--url") + 1] == f"{MCP_URL}?guardrails={CTX}"
        stdio = run("codex", "--profile", "default", "--force", "-y")
        assert "turned on here" not in stdio.output

    def test_hooks_named_table_is_the_one_warned_about(self, env, on_path, recorder):
        on_path("codex")
        turn_on_codex_hooks(env, mcp_server="kagura-work")
        result = run("codex", "--profile", "default", "--name", "kagura-work", "-y")
        assert result.exit_code == 0, result.output
        assert "turned on here for the kagura-work entry" in result.output

    def test_without_codex_prints_the_table_and_edits_nothing(self, env, recorder):
        result = run("codex", "--profile", "default", "--context-id", CTX, "-y")
        assert result.exit_code == 0, result.output
        assert "`codex` is not on PATH" in result.output
        assert "~/.codex/config.toml" in result.output
        assert "[mcp_servers.kagura-memory]" in result.output
        assert f'command = "{PROXY}"' in result.output
        assert f'args = ["--profile", "default", "--guardrails", "{CTX}"]' in result.output
        assert recorder.calls == []
        assert not (env / ".codex").exists()

    def test_codex_home_is_honoured(self, tmp_path, monkeypatch, recorder):
        codex_home = tmp_path / "codex-home"
        monkeypatch.setenv("CODEX_HOME", str(codex_home))
        result = run("codex", "--profile", "default", "-y")
        assert result.exit_code == 0, result.output
        assert str(codex_home / "config.toml") in result.output

    def test_agents_md_defaults_to_the_codex_home_file(self, env, on_path, digest):
        on_path("codex")
        result = run("codex", "--profile", "default", "--context-id", CTX, "--agents-md", "-y")
        assert result.exit_code == 0, result.output
        assert (env / ".codex" / "AGENTS.md").read_text(encoding="utf-8") == EXPORT_BLOCK
        assert digest.calls == [("default", CTX)]
        assert f"kagura guardrails digest {CTX} --out" in result.output

    def test_agents_md_prefers_an_existing_override(self, env, on_path):
        on_path("codex")
        override = env / ".codex" / "AGENTS.override.md"
        override.parent.mkdir(parents=True)
        override.write_text("# Mine\n", encoding="utf-8")
        result = run("codex", "--profile", "default", "--context-id", CTX, "--agents-md", "-y")
        assert result.exit_code == 0, result.output
        assert override.read_text(encoding="utf-8") == "# Mine\n\n" + EXPORT_BLOCK
        assert not (env / ".codex" / "AGENTS.md").exists()


# =============================================================================
# Hermes
# =============================================================================


class TestHermes:
    def test_tty_runs_stdio_add_attached(self, on_path, recorder, tty):
        on_path("hermes")
        result = run("hermes", "--profile", "default", input="n\n")
        assert result.exit_code == 0, result.output
        [(argv, kwargs)] = [c for c in recorder.calls if c[0] in recorder.mutating()]
        assert argv == [
            "/usr/bin/hermes",
            *("mcp", "add", "kagura-memory", "--command", PROXY),
            *("--args", "--profile", "default"),
        ]
        # Attached to the terminal: no captured output, no timeout.
        assert "capture_output" not in kwargs and "timeout" not in kwargs
        assert "hermes mcp test kagura-memory" in result.output

    def test_tty_runs_url_form_add_with_header_auth(self, on_path, recorder, tty):
        on_path("hermes")
        result = run("hermes", "--url-form", "--mcp-url", MCP_URL, input="n\n")
        assert result.exit_code == 0, result.output
        assert recorder.mutating() == [
            ["/usr/bin/hermes", "mcp", "add", "kagura-memory", "--url", MCP_URL, "--auth", "header"]
        ]
        assert "MCP_KAGURA_MEMORY_API_KEY" in result.output

    @pytest.mark.parametrize("flags", [["-y"], []], ids=["-y", "no-tty"])
    def test_without_tty_or_with_y_prints_yaml_and_runs_nothing(
        self, on_path, recorder, env, flags, monkeypatch
    ):
        on_path("hermes")
        # -y alone suffices even with a terminal; without -y, no terminal suffices.
        monkeypatch.setattr(setup_harness, "_stdin_is_tty", lambda: bool(flags))
        # No input: a prompt would hit EOF and abort (an agent's shell, CI, </dev/null).
        result = run("hermes", "--profile", "default", "--context-id", CTX, *flags)
        assert result.exit_code == 0, result.output
        assert "Aborted" not in result.output
        assert "Write the guardrail export block there?" not in result.output
        assert recorder.mutating() == []
        assert "~/.hermes/config.yaml" in result.output
        assert "mcp_servers:\n      kagura-memory:" in result.output
        assert f'command: "{PROXY}"' in result.output
        assert 'args: ["--profile", "default"]' in result.output
        assert "Re-run with --agents-md --context-id <id>" in result.output

    def test_cancelled_add_saves_nothing_and_skips_the_export(self, on_path, recorder, tty):
        on_path("hermes")
        recorder.hermes_saves = False
        result = run("hermes", "--profile", "default", "--context-id", CTX, "--agents-md")
        assert result.exit_code == 1
        assert "cancelled or failed there, so nothing was saved" in flat(result.output)
        assert "skipped the AGENTS.md export" in result.output
        assert not Path("AGENTS.md").exists()

    def test_kept_url_entry_is_not_reported_as_the_new_one(self, on_path, recorder, tty):
        on_path("hermes")
        recorder.hermes_saves = False
        recorder.hermes_entries["kagura-memory"] = {"url": "https://x/mcp"}
        result = run("hermes", "--profile", "default", "--force", input="n\n")
        assert result.exit_code == 1
        out = flat(result.output)
        assert "entry is still the existing one (URL with no credential)" in out
        assert "so nothing was saved" in out
        assert "https://x/mcp" not in out

    def test_kept_same_form_entry_is_not_reported_as_the_new_one(self, on_path, recorder, tty):
        """``mcp list`` showed a kept stdio entry as the new stdio one (#278)."""
        on_path("hermes")
        recorder.hermes_saves = False
        recorder.hermes_entries["kagura-memory"] = {
            "command": "/old/bin/kagura-mcp",
            "args": ["--profile", "old"],
        }
        result = run("hermes", "--profile", "default", "--context-id", CTX, "--agents-md",
                     "--force", input="n\n")  # fmt: skip
        assert result.exit_code == 1
        out = flat(result.output)
        assert "entry is still the existing one (stdio (kagura-mcp))" in out
        assert "setup skipped the AGENTS.md export" in out
        assert "/old/bin" not in out and "Done:" not in out
        assert not Path("AGENTS.md").exists()

    def test_entry_saved_disabled_points_to_mcp_test(self, on_path, recorder, tty, digest):
        on_path("hermes")
        recorder.hermes_probe_ok = False
        result = run("hermes", "--profile", "default", "--context-id", CTX, "--agents-md",
                     input="n\n")  # fmt: skip
        assert result.exit_code == 1
        out = flat(result.output)
        assert "Hermes saved kagura-memory disabled" in out
        assert "Check it with `hermes mcp test kagura-memory`" in out
        assert "`hermes config set mcp_servers.kagura-memory.enabled true`" in out
        assert digest.calls == []

    @pytest.mark.parametrize("value", ["false", "No", "0", "off"])
    def test_a_string_enabled_is_read_as_mcp_list_reads_it(self, on_path, recorder, tty, value):
        on_path("hermes")
        result_entry = {"command": PROXY, "args": ["--profile", "default"], "enabled": value}
        recorder.hermes_saves = False
        recorder.hermes_entries["kagura-memory"] = result_entry
        result = run("hermes", "--profile", "default", "--force", input="n\n")
        assert result.exit_code == 1
        assert "saved kagura-memory disabled" in flat(result.output)

    def test_url_entry_without_a_header_gets_a_warning(self, on_path, recorder, tty):
        """Hermes saves no headers when its key prompt is declined or left empty."""
        on_path("hermes")
        recorder.hermes_header_ok = False
        result = run("hermes", "--url-form", "--mcp-url", MCP_URL, input="n\n")
        assert result.exit_code == 0, result.output
        out = flat(result.output)
        assert "Warning: Hermes saved the entry with no Authorization header" in out
        assert "keeps it in" not in out

    def test_url_entry_with_a_header_gets_the_env_note(self, on_path, recorder, tty):
        on_path("hermes")
        result = run("hermes", "--url-form", "--mcp-url", MCP_URL, input="n\n")
        assert result.exit_code == 0, result.output
        assert "Hermes asked for the key itself and keeps it in" in result.output
        assert "Warning" not in result.output

    @pytest.mark.parametrize(
        ("entry", "kind"),
        [
            ({"command": "/v/bin/kagura-mcp", "args": ["--profile", "p"]}, "stdio (kagura-mcp)"),
            ({"command": "npx", "args": ["-y", "other"]}, "stdio (another command)"),
            ({"url": MCP_URL, "auth": "oauth"}, "URL with OAuth"),
            (
                {"url": MCP_URL, "headers": {"authorization": "x"}},
                "URL with an Authorization header",
            ),
            ({"url": MCP_URL}, "URL with no credential"),
            ({"timeout": 5}, "an entry setup does not recognise"),
        ],
    )
    def test_existing_entry_kind_from_config_get(self, on_path, recorder, entry, kind):
        on_path("hermes")
        recorder.hermes_entries["kagura-memory"] = entry
        result = run("hermes", "--profile", "default", "-y")
        assert result.exit_code == 1
        assert f"Existing kagura-memory entry: {kind}" in result.output
        assert [
            "/usr/bin/hermes", "config", "get", "mcp_servers.kagura-memory", "--json"
        ] in recorder.argvs()  # fmt: skip

    def test_existing_entry_values_are_never_echoed(self, on_path, recorder):
        on_path("hermes")
        recorder.hermes_entries["kagura-memory"] = {
            "url": "https://SECRET-URL.example/mcp",
            "headers": {"Authorization": "Bearer SECRET-HEADER"},
            "env": {"KEY": "SECRET-ENV"},
            "args": ["--token", "SECRET-ARG"],
        }
        result = run("hermes", "--profile", "default", "-y")
        assert result.exit_code == 1
        assert "URL with an Authorization header" in result.output
        assert "SECRET" not in result.output

    def test_printed_url_block_references_the_hermes_variable(self, recorder):
        result = run("hermes", "--url-form", "--mcp-url", MCP_URL, "-y")
        assert result.exit_code == 0, result.output
        assert 'Authorization: "Bearer ${MCP_KAGURA_MEMORY_API_KEY}"' in result.output
        assert "MCP_KAGURA_MEMORY_API_KEY=<your-api-key>` to ~/.hermes/.env" in result.output
        assert recorder.calls == []

    def test_guardrails_off_is_a_usage_error(self):
        result = run("hermes", "--profile", "default", "--guardrails", "off", "-y")
        assert result.exit_code == 2
        assert "get_context_info" in result.output

    def test_guardrails_context_is_not_written_and_warns(self, recorder):
        result = run("hermes", "--profile", "default", "--guardrails", CTX, "-y")
        assert result.exit_code == 0, result.output
        assert "does not read MCP instructions" in result.output
        assert "--guardrails" not in result.output.split("Add this")[1]

    def test_api_key_env_is_refused(self):
        result = run("hermes", "--url-form", "--mcp-url", MCP_URL, "--api-key-env", "MY_KEY", "-y")
        assert result.exit_code == 2
        assert "MCP_KAGURA_MEMORY_API_KEY" in result.output

    def test_existing_entry_from_mcp_list_blocks_without_force(self, on_path, recorder, tty):
        """An older Hermes, whose `config get` fails: its `mcp list` tells."""
        on_path("hermes")
        recorder.hermes_unreadable.add("")
        recorder.detect_out["hermes"] = (
            "\n  MCP Servers:\n\n  Name  Transport  Tools  Status\n"
            "  kagura-memory    https://memory.kagura-ai.com/mcp   all   \x1b[32m✓ enabled\x1b[0m\n"
        )
        result = run("hermes", "--profile", "default")
        assert result.exit_code == 1
        assert "(URL)" in result.output
        assert recorder.mutating() == []

    def test_force_lets_hermes_ask_before_overwriting(self, on_path, recorder, tty):
        on_path("hermes")
        recorder.hermes_entries["kagura-memory"] = {"command": "/x/kagura-mcp", "args": []}
        result = run("hermes", "--profile", "default", "--force", input="n\n")
        assert result.exit_code == 0, result.output
        assert recorder.mutating()[0][1:4] == ["mcp", "add", "kagura-memory"]

    def test_offer_writes_the_file_hermes_loads(self, on_path, recorder, tty):
        on_path("hermes")
        Path("CLAUDE.md").write_text("# Claude\n", encoding="utf-8")
        result = run("hermes", "--profile", "default", "--context-id", CTX, input="y\n")
        assert result.exit_code == 0, result.output
        assert "Write the guardrail export block there?" in result.output
        assert Path("CLAUDE.md").read_text(encoding="utf-8") == "# Claude\n\n" + EXPORT_BLOCK
        assert "prompt injection" in result.output

    def test_offer_defaults_to_no(self, on_path, tty, digest):
        on_path("hermes")
        result = run("hermes", "--profile", "default", "--context-id", CTX, input="\n")
        assert result.exit_code == 0, result.output
        assert digest.calls == []
        assert not Path("AGENTS.md").exists()

    def test_offer_uses_the_context_prompt(self, on_path, tty, digest):
        on_path("hermes")
        # Accept the export; Enter alone picks nothing; then context 2 ("ops").
        result = run("hermes", "--profile", "default", input="y\n\n2\n")
        assert result.exit_code == 0, result.output
        assert digest.calls == [("default", OTHER_CTX)]
        assert "Create new context" not in result.output

    def test_only_cursorrules_agents_md_without_path_stops_before_the_add(
        self, on_path, recorder, tty, digest
    ):
        on_path("hermes")
        Path(".cursorrules").write_text("rules", encoding="utf-8")
        result = run("hermes", "--profile", "default", "--context-id", CTX, "--agents-md")
        assert result.exit_code == 1
        out = flat(result.output)
        assert "Nothing was written: Hermes loads only" in out
        assert ".cursorrules here" in out and "--agents-md PATH" in out
        assert recorder.mutating() == [] and digest.calls == []
        assert not Path("AGENTS.md").exists()

    def test_only_cursorrules_agents_md_with_path_writes_there(self, on_path, recorder, tty):
        on_path("hermes")
        Path(".cursorrules").write_text("rules", encoding="utf-8")
        result = run(
            "hermes", "--profile", "default", "--context-id", CTX, "--agents-md", "RULES.md",
            input="n\n",
        )  # fmt: skip
        assert result.exit_code == 0, result.output
        assert Path("RULES.md").read_text(encoding="utf-8") == EXPORT_BLOCK

    def test_only_cursorrules_no_offer_and_no_hint(self, on_path, recorder, tty, digest):
        on_path("hermes")
        Path(".cursorrules").write_text("rules", encoding="utf-8")
        result = run("hermes", "--profile", "default", "--context-id", CTX, input="n\n")
        assert result.exit_code == 0, result.output
        assert "Write the guardrail export block there?" not in result.output
        assert "Re-run with --agents-md" not in result.output
        assert digest.calls == []

    def test_older_hermes_without_config_get_is_checked_with_mcp_list(self, on_path, recorder, tty):
        on_path("hermes")
        recorder.hermes_unreadable.add("")
        result = run("hermes", "--profile", "default", input="n\n")
        assert result.exit_code == 0, result.output
        assert "Done: hermes wrote kagura-memory" in result.output
        assert ["/usr/bin/hermes", "mcp", "list"] in recorder.argvs()

    def test_older_hermes_cancelled_add_shows_no_new_entry(self, on_path, recorder, tty):
        on_path("hermes")
        recorder.hermes_unreadable.add("")
        recorder.hermes_saves = False
        result = run("hermes", "--profile", "default", input="n\n")
        assert result.exit_code == 1
        assert "`hermes mcp list` shows no new kagura-memory entry" in result.output

    def test_an_entry_other_than_the_new_one_is_not_saved(self, recorder):
        """No entry before the add, and not the one setup asked for after it."""
        recorder.hermes_entries["kagura-memory"] = {"url": "https://x/mcp"}
        entry = setup_harness._Entry(command=PROXY, args=("--profile", "default"))
        problem = setup_harness._Hermes().not_saved(
            "kagura-memory", "/usr/bin/hermes", entry, replaced=False
        )
        assert problem is not None
        assert "(URL with no credential) is not the one setup asked for" in flat(problem)
        assert "https://x/mcp" not in problem

    def test_dry_run_with_only_cursorrules_names_the_file(self, recorder, tty):
        Path(".cursorrules").write_text("rules", encoding="utf-8")
        result = run("hermes", "--profile", "default", "--dry-run")
        assert result.exit_code == 0, result.output
        out = flat(result.output)
        assert "AGENTS.md: not offered. Hermes loads only" in out
        assert ".cursorrules here" in out

    def test_no_offer_with_y(self, digest):
        result = run("hermes", "--profile", "default", "--context-id", CTX, "-y")
        assert result.exit_code == 0, result.output
        assert "Write the guardrail export block there?" not in result.output
        assert digest.calls == []


class TestHermesPaths:
    @pytest.mark.parametrize(
        ("present", "expected"),
        [
            ([], "AGENTS.md"),
            (["CLAUDE.md"], "CLAUDE.md"),
            (["CLAUDE.md", "AGENTS.md"], "AGENTS.md"),
            (["AGENTS.md", "AGENTS.override.md"], "AGENTS.override.md"),
            (["AGENTS.md", "HERMES.md"], "HERMES.md"),
            (["HERMES.md", ".hermes.md", "CLAUDE.md"], ".hermes.md"),
        ],
    )
    def test_context_file_precedence(self, tmp_path, present, expected):
        for name in present:
            (tmp_path / name).write_text("x", encoding="utf-8")
        assert hermes_context_file(tmp_path) == tmp_path / expected

    @staticmethod
    def layout(root: Path, files: dict[str, str]) -> None:
        for name, text in files.items():
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_text(text, encoding="utf-8")

    @staticmethod
    def hermes_loads(cwd: Path) -> list[Path]:
        """What Hermes loads in ``cwd``, stated from its source (#278), apart from the code."""

        def text(f: Path) -> bool:
            return f.is_file() and bool(f.read_text(encoding="utf-8").strip())

        root = next((d for d in (cwd, *cwd.parents) if (d / ".git").exists()), None)
        if root is None:
            dirs = [cwd]
        else:  # the git root down to cwd
            parts = cwd.relative_to(root).parts
            dirs = [root.joinpath(*parts[:i]) for i in range(len(parts) + 1)]
        for d in reversed(dirs):
            found = [d / n for n in (".hermes.md", "HERMES.md") if (d / n).is_file()]
            if found:
                if text(found[0]):
                    return [found[0]]
                break
        agents = []
        for d in dirs:
            first = next(
                (d / n for n in ("AGENTS.override.md", "AGENTS.md", "agents.md") if text(d / n)),
                None,
            )
            if first is not None:
                agents.append(first)
        if agents:
            return agents
        for names in (("CLAUDE.md", "claude.md"), (".cursorrules",)):
            first = next((cwd / n for n in names if text(cwd / n)), None)
            if first is not None:
                return [first]
        return []

    # The layouts in #278's table, and four more from its acceptance list.
    @pytest.mark.parametrize(
        ("files", "cwd", "expected"),
        [
            ({".git/x": "", ".hermes.md": "own", "sub/x": ""}, "sub", ".hermes.md"),
            (
                {".git/x": "", ".hermes.md": "own", "sub/AGENTS.md": "agents"},
                "sub",
                ".hermes.md",
            ),
            (
                {".git/x": "", "AGENTS.md": "root", "sub/CLAUDE.md": "claude"},
                "sub",
                "sub/AGENTS.md",
            ),
            ({".hermes.md": "", "AGENTS.md": "agents"}, ".", "AGENTS.md"),
            ({"AGENTS.override.md": " \n", "AGENTS.md": "agents"}, ".", "AGENTS.md"),
            ({"agents.md": "agents"}, ".", "agents.md"),
            ({"claude.md": "claude"}, ".", "claude.md"),
            # A .hermes.md above the git root is not read.
            ({".hermes.md": "own", "repo/.git/x": ""}, "repo", "repo/AGENTS.md"),
            # Without .git, only the current directory is.
            ({".hermes.md": "own", "sub/x": ""}, "sub", "sub/AGENTS.md"),
            # An empty nearer .hermes.md ends the lookup.
            ({".git/x": "", ".hermes.md": "own", "sub/.hermes.md": ""}, "sub", "sub/AGENTS.md"),
            ({".hermes.md": "", "HERMES.md": "own"}, ".", "AGENTS.md"),
        ],
        ids=[
            "root-hermes-md",
            "root-hermes-md-over-sub-agents",
            "agents-chain-from-root",
            "empty-hermes-md",
            "empty-override",
            "lower-agents-md",
            "lower-claude-md",
            "hermes-md-above-git-root",
            "no-git-cwd-only",
            "empty-sub-hermes-md",
            "empty-hermes-md-hides-HERMES-md",
        ],
    )
    def test_the_export_goes_where_hermes_loads_it(self, tmp_path, files, cwd, expected):
        self.layout(tmp_path, files)
        directory = tmp_path / cwd
        before = self.hermes_loads(directory)
        target = hermes_context_file(directory)
        assert target == tmp_path / expected
        target.write_text(
            (target.read_text(encoding="utf-8") if target.exists() else "") + EXPORT_BLOCK,
            encoding="utf-8",
        )
        after = self.hermes_loads(directory)
        assert target in after
        assert set(before) <= set(after)  # nothing the user had stops loading

    @pytest.mark.skipif(
        not hasattr(os, "geteuid") or os.geteuid() == 0, reason="needs an unreadable file"
    )
    def test_an_unreadable_file_counts_as_empty(self, tmp_path):
        self.layout(tmp_path, {"CLAUDE.md": "claude", "AGENTS.md": "agents"})
        (tmp_path / "AGENTS.md").chmod(0)
        try:
            assert hermes_context_file(tmp_path) == tmp_path / "CLAUDE.md"
        finally:
            (tmp_path / "AGENTS.md").chmod(0o644)

    def test_only_hermes_can_have_no_default_file(self):
        assert setup_harness._Codex().no_agents_md_reason() == ""

    @pytest.mark.parametrize("rules", [".cursorrules", ".cursor/rules/a.mdc"])
    def test_only_cursor_rules_leave_no_default_file(self, tmp_path, rules):
        self.layout(tmp_path, {rules: "rules"})
        assert hermes_context_file(tmp_path) is None
        assert setup_harness.hermes_cursor_rules(tmp_path) == tmp_path / rules

    def test_home_env_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "h"))
        assert hermes_home() == tmp_path / "h"

    def test_sticky_active_profile(self, env):
        (env / ".hermes").mkdir()
        (env / ".hermes" / "active_profile").write_text("Work\n", encoding="utf-8")
        assert hermes_home() == env / ".hermes" / "profiles" / "work"

    @pytest.mark.parametrize("content", [None, "default\n", "../escape\n"])
    def test_default_home(self, env, content):
        (env / ".hermes").mkdir()
        if content is not None:
            (env / ".hermes" / "active_profile").write_text(content, encoding="utf-8")
        assert hermes_home() == env / ".hermes"

    def test_key_env_name(self):
        assert hermes_key_env("kagura-memory") == "MCP_KAGURA_MEMORY_API_KEY"
        assert hermes_key_env("my_srv") == "MCP_MY_SRV_API_KEY"


class TestHermesBlock:
    """The printed block when config.yaml already has a top-level ``mcp_servers:`` (#274).

    A second top-level key would replace the first (YAML keeps the last), and
    every server under it with it, so only the entry is printed.
    """

    STDIO_ENTRY = [
        "kagura-memory:",
        f'  command: "{PROXY}"',
        '  args: ["--profile", "default"]',
    ]

    @staticmethod
    def config(home: Path) -> Path:
        path = home / ".hermes" / "config.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def printed(lines: list[str], indent: str) -> str:
        """``lines`` as setup prints them: its 4-space margin, then ``indent``."""
        return "".join(f"\n    {indent}{line}" for line in lines) + "\n"

    @pytest.mark.parametrize(
        ("text", "indent"),
        [
            ("model: gpt\nmcp_servers:\n  other:\n    command: foo\n", "  "),
            ("mcp_servers:\n    other:\n        url: https://x\n", "    "),
            ("\ufeffmcp_servers:\n  other:\n    command: foo\n", "  "),
            ('"mcp_servers":\n  other:\n    command: foo\n', "  "),
            ("'mcp_servers' :\n  other:\n    command: foo\n", "  "),
            ("mcp_servers:  # mine\n# note\n\n   other:\n     command: foo\n", "   "),
            ("mcp_servers:\r\n  other:\r\n    command: foo\r\n", "  "),
            ("mcp_servers:\nmodel: gpt\n", "  "),  # a key with no entries yet
            ("model: gpt\nmcp_servers:", "  "),  # the last line
        ],
        ids=[
            "after-other-keys",
            "4-space",
            "bom",
            "double-quoted",
            "single-quoted",
            "comments",
            "crlf",
            "empty",
            "last-line",
        ],
    )
    def test_existing_key_gets_the_entry_alone(self, env, recorder, text, indent):
        self.config(env).write_text(text, encoding="utf-8")
        result = run("hermes", "--profile", "default", "-y")
        assert result.exit_code == 0, result.output
        assert self.printed(self.STDIO_ENTRY, indent) in result.output
        assert "\n    mcp_servers:" not in result.output
        assert "Add this kagura-memory entry to its mcp_servers: mapping:" in result.output
        assert "already has a top-level mcp_servers: key" in result.output
        assert "inline" not in result.output
        assert recorder.calls == []

    def test_url_form_entry_alone(self, env, recorder):
        self.config(env).write_text("mcp_servers:\n  other:\n    command: foo\n", encoding="utf-8")
        result = run("hermes", "--url-form", "--mcp-url", MCP_URL, "-y")
        assert result.exit_code == 0, result.output
        entry = [
            "kagura-memory:",
            f'  url: "{MCP_URL}"',
            "  headers:",
            '    Authorization: "Bearer ${MCP_KAGURA_MEMORY_API_KEY}"',
        ]
        assert self.printed(entry, "  ") in result.output

    @pytest.mark.parametrize("value", ["{}", "null", "{other: {command: foo}}", "~"])
    def test_inline_value_must_become_a_block_first(self, env, recorder, value):
        self.config(env).write_text(f"mcp_servers: {value}\nmodel: gpt\n", encoding="utf-8")
        result = run("hermes", "--profile", "default", "-y")
        assert result.exit_code == 0, result.output
        assert self.printed(self.STDIO_ENTRY, "  ") in result.output
        assert "written inline" in result.output
        assert "rewrite it" in result.output

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "model: gpt\n",
            "agents:\n  mcp_servers:\n    x: {}\n",
            "# mcp_servers:\n",
            "mcp_servers_old:\n  x: 1\n",
        ],
        ids=["empty", "other-keys", "nested", "comment", "longer-name"],
    )
    def test_without_a_top_level_key_prints_the_whole_block(self, env, recorder, text):
        self.config(env).write_text(text, encoding="utf-8")
        result = run("hermes", "--profile", "default", "-y")
        assert result.exit_code == 0, result.output
        assert self.printed(["mcp_servers:", *(f"  {x}" for x in self.STDIO_ENTRY)], "") in (
            result.output
        )
        assert "Add this kagura-memory entry to it:" in result.output
        assert "already has" not in result.output

    def test_unreadable_config_prints_the_whole_block_with_a_note(self, env, recorder):
        self.config(env).mkdir()  # a directory where the file should be
        result = run("hermes", "--profile", "default", "-y")
        assert result.exit_code == 0, result.output
        assert "\n    mcp_servers:\n      kagura-memory:\n" in result.output
        assert "could not read ~/.hermes/config.yaml" in result.output
        assert "put only the kagura-memory entry under it" in result.output


# =============================================================================
# OpenClaw
# =============================================================================


class TestOpenClaw:
    def test_stdio_add_argv(self, on_path, recorder):
        on_path("openclaw")
        result = run("openclaw", "--profile", "default", "-y")
        assert result.exit_code == 0, result.output
        assert recorder.mutating() == [
            [
                "/usr/bin/openclaw",
                *("mcp", "add", "kagura-memory", "--command", PROXY),
                *("--arg", "--profile", "--arg", "default"),
            ]
        ]
        assert "openclaw mcp doctor kagura-memory --probe" in result.output
        assert "coding and" in result.output and "not in minimal" in result.output

    def test_url_form_add_argv(self, on_path, recorder):
        on_path("openclaw")
        result = run("openclaw", "--url-form", "--mcp-url", MCP_URL, "-y")
        assert result.exit_code == 0, result.output
        assert recorder.mutating() == [
            [
                "/usr/bin/openclaw",
                *("mcp", "add", "kagura-memory", "--url", MCP_URL),
                *("--transport", "streamable-http"),
                *("--header", "Authorization=Bearer ${KAGURA_API_KEY}", "--no-probe"),
            ]
        ]
        assert "KAGURA_API_KEY=<your-api-key>` to ~/.openclaw/.env" in result.output

    def test_force_uses_mcp_set(self, on_path, recorder):
        on_path("openclaw")
        recorder.detect_out["openclaw"] = json.dumps(
            {"url": "https://x/mcp", "headers": {"Authorization": f"Bearer {API_KEY}"}}
        )
        result = run("openclaw", "--profile", "default", "--force", "-y")
        assert result.exit_code == 0, result.output
        assert "static Authorization header" in result.output
        assert API_KEY not in result.output
        [argv] = recorder.mutating()
        assert argv[:4] == ["/usr/bin/openclaw", "mcp", "set", "kagura-memory"]
        assert json.loads(argv[4]) == {"command": PROXY, "args": ["--profile", "default"]}

    def test_existing_entry_blocks_without_force(self, on_path, recorder):
        on_path("openclaw")
        recorder.detect_out["openclaw"] = json.dumps(
            {"url": "https://x/mcp", "transport": "streamable-http", "auth": "oauth"}
        )
        result = run("openclaw", "--profile", "default", "-y")
        assert result.exit_code == 1
        assert "URL with OAuth" in result.output
        assert recorder.mutating() == []

    @pytest.mark.parametrize("mode", ["add", "set", "block"])
    def test_every_url_form_entry_is_streamable_http(self, on_path, recorder, mode):
        if mode != "block":
            on_path("openclaw")
        if mode == "set":
            recorder.detect_out["openclaw"] = json.dumps({"command": "x"})
        result = run("openclaw", "--url-form", "--mcp-url", MCP_URL, "--force", "-y")
        assert result.exit_code == 0, result.output
        if mode == "add":
            [argv] = recorder.mutating()
            assert argv[argv.index("--transport") + 1] == "streamable-http"
            return
        if mode == "set":
            server = json.loads(recorder.mutating()[0][4])
        else:
            server = printed_json(result.output)["mcp"]["servers"]["kagura-memory"]
        assert server["transport"] == "streamable-http"
        assert server["headers"] == {"Authorization": "Bearer ${KAGURA_API_KEY}"}
        assert "auth" not in server

    def test_guardrails_off_is_a_usage_error(self):
        assert run("openclaw", "--profile", "default", "--guardrails", "off", "-y").exit_code == 2

    def test_without_a_terminal_adds_and_skips_the_offer(self, on_path, recorder):
        on_path("openclaw")
        result = run("openclaw", "--profile", "default", "--context-id", CTX)  # stdin at EOF
        assert result.exit_code == 0, result.output
        assert "Aborted" not in result.output
        assert recorder.mutating()[0][1:3] == ["mcp", "add"]
        assert "~/.openclaw/workspace/AGENTS.md,\n  which OpenClaw loads" in result.output

    def test_guardrails_context_warns_and_picks_the_export_context(
        self, on_path, recorder, env, digest
    ):
        on_path("openclaw")
        result = run("openclaw", "--profile", "default", "--guardrails", CTX, "--agents-md", "-y")
        assert result.exit_code == 0, result.output
        assert "does not read MCP instructions" in result.output
        assert "--guardrails" not in recorder.mutating()[0]
        assert digest.calls == [("default", CTX)]
        agents = env / ".openclaw" / "workspace" / "AGENTS.md"
        assert agents.read_text(encoding="utf-8") == EXPORT_BLOCK

    def test_without_openclaw_prints_the_json_block(self, recorder, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENCLAW_CONFIG_PATH", str(tmp_path / "oc.json"))
        result = run("openclaw", "--profile", "default", "-y")
        assert result.exit_code == 0, result.output
        assert str(tmp_path / "oc.json") in result.output
        assert printed_json(result.output) == {
            "mcp": {
                "servers": {"kagura-memory": {"command": PROXY, "args": ["--profile", "default"]}}
            }
        }
        assert recorder.calls == []

    def test_state_dir_holds_the_config_the_env_file_and_the_workspace(
        self, tmp_path, monkeypatch, recorder, digest
    ):
        # OpenClaw keeps all three under $OPENCLAW_STATE_DIR when it is set (#274).
        state = tmp_path / "oc-state"
        monkeypatch.setenv("OPENCLAW_STATE_DIR", f" {state} ")
        result = run(
            "openclaw",
            *("--profile", "default", "--url-form", "--mcp-url", MCP_URL),
            *("--context-id", CTX, "--agents-md", "-y"),
        )
        assert result.exit_code == 0, result.output
        assert f"OpenClaw ({state / 'openclaw.json'})" in result.output
        assert f"KAGURA_API_KEY=<your-api-key>` to {state / '.env'}" in result.output
        assert "~/.openclaw" not in result.output
        assert (state / "workspace" / "AGENTS.md").read_text(encoding="utf-8") == EXPORT_BLOCK

    def test_state_dir_expands_a_leading_tilde(self, monkeypatch, recorder):
        monkeypatch.setenv("OPENCLAW_STATE_DIR", "~/oc-state")
        result = run("openclaw", "--profile", "default", "-y")
        assert result.exit_code == 0, result.output
        assert "OpenClaw (~/oc-state/openclaw.json)" in result.output
        assert "~/oc-state/workspace/AGENTS.md" in result.output

    def test_workspace_dir_wins_over_state_dir_for_the_export(
        self, tmp_path, monkeypatch, recorder, digest
    ):
        # OpenClaw's resolveDefaultAgentWorkspaceDir reads $OPENCLAW_WORKSPACE_DIR first.
        state, workspace = tmp_path / "oc-state", tmp_path / "ws"
        monkeypatch.setenv("OPENCLAW_STATE_DIR", str(state))
        monkeypatch.setenv("OPENCLAW_WORKSPACE_DIR", f" {workspace} ")
        result = run("openclaw", "--profile", "default", "--context-id", CTX, "--agents-md", "-y")
        assert result.exit_code == 0, result.output
        assert (workspace / "AGENTS.md").read_text(encoding="utf-8") == EXPORT_BLOCK
        assert not (state / "workspace").exists()
        # The config and the key's .env stay in the state directory.
        assert f"OpenClaw ({state / 'openclaw.json'})" in result.output

    @pytest.mark.parametrize(
        "var", ["OPENCLAW_STATE_DIR", "OPENCLAW_CONFIG_PATH", "OPENCLAW_WORKSPACE_DIR"]
    )
    def test_a_tilde_user_value_is_kept_as_written(self, monkeypatch, recorder, var):
        """``Path.expanduser`` raised, failing every run: "Could not determine home" (#285)."""
        monkeypatch.setenv(var, "~no-such-user-285/oc")
        result = run("openclaw", "--profile", "default", "-y")
        assert result.exit_code == 0, result.output
        assert "~no-such-user-285/oc" in result.output

    def test_workspace_dir_expands_a_leading_tilde(self, monkeypatch, recorder):
        monkeypatch.setenv("OPENCLAW_WORKSPACE_DIR", "~/oc-ws")
        result = run("openclaw", "--profile", "default", "-y")
        assert result.exit_code == 0, result.output
        assert "~/oc-ws/AGENTS.md,\n  which OpenClaw loads" in result.output
        assert "OpenClaw (~/.openclaw/openclaw.json)" in result.output

    def test_config_path_wins_over_state_dir(self, tmp_path, monkeypatch, recorder):
        monkeypatch.setenv("OPENCLAW_STATE_DIR", str(tmp_path / "oc-state"))
        monkeypatch.setenv("OPENCLAW_CONFIG_PATH", str(tmp_path / "oc.json"))
        result = run("openclaw", "--url-form", "--mcp-url", MCP_URL, "-y")
        assert result.exit_code == 0, result.output
        assert f"OpenClaw ({tmp_path / 'oc.json'})" in result.output
        # The key still goes in the state directory's .env.
        assert f"to {tmp_path / 'oc-state' / '.env'}" in result.output


def printed_url(harness: str, output: str) -> str:
    """The URL in the Hermes or OpenClaw block setup printed."""
    if harness == "openclaw":
        return printed_json(output)["mcp"]["servers"]["kagura-memory"]["url"]
    [line] = [x for x in output.splitlines() if x.strip().startswith("url: ")]
    return json.loads(line.split("url: ", 1)[1])


@pytest.mark.parametrize("harness", ["hermes", "openclaw"])
class TestNoInstructionsUrl:
    """A ``?guardrails=`` context in ``--mcp-url`` never reaches their entry (#274)."""

    @pytest.mark.parametrize(
        "query",
        [
            f"guardrails={CTX}&profile=core",
            f"profile=core&guardrails={CTX}&guardrails=off",  # the server reads the first
            f"guard%72ails={CTX}&profile=core",  # the server decodes the name
            "guardrails=typo&profile=core",
        ],
        ids=["context", "context-first", "encoded-name", "not-a-context"],
    )
    def test_guardrails_value_is_dropped_with_a_warning(self, harness, recorder, query):
        result = run(harness, "--url-form", "--mcp-url", f"{MCP_URL}?{query}", "-y")
        assert result.exit_code == 0, result.output
        assert printed_url(harness, result.output) == f"{MCP_URL}?profile=core"
        assert "so the ?guardrails= value in --mcp-url has no effect there and is not written" in (
            flat(result.output)
        )
        assert CTX not in result.output

    def test_one_warning_when_guardrails_is_given_both_ways(self, harness, recorder):
        url = f"{MCP_URL}?guardrails={CTX}"
        result = run(harness, "--url-form", "--mcp-url", url, "--guardrails", CTX, "-y")
        assert result.exit_code == 0, result.output
        assert printed_url(harness, result.output) == MCP_URL
        assert result.output.count("does not read MCP instructions") == 1
        assert (
            "so --guardrails and the ?guardrails= value in --mcp-url have no effect there "
            "and are not written"
        ) in flat(result.output)

    def test_url_without_other_parameters_loses_its_query(self, harness, recorder):
        result = run(harness, "--url-form", "--mcp-url", f"{MCP_URL}?guardrails={CTX}", "-y")
        assert result.exit_code == 0, result.output
        assert printed_url(harness, result.output) == MCP_URL

    @pytest.mark.parametrize(
        ("query", "written"),
        [
            ("profile=core&guardrails=off", "profile=core&guardrails=off"),
            # The server reads the first value: only that off is kept, never the context.
            (f"guardrails=OFF&guardrails={CTX}", "guardrails=off"),
        ],
        ids=["off", "off-first"],
    )
    def test_guardrails_off_is_kept_with_a_warning(self, harness, recorder, query, written):
        result = run(harness, "--url-form", "--mcp-url", f"{MCP_URL}?{query}", "-y")
        assert result.exit_code == 0, result.output
        assert printed_url(harness, result.output) == f"{MCP_URL}?{written}"
        assert "--mcp-url has ?guardrails=off" in result.output
        assert "not written" not in result.output
        assert CTX not in result.output

    def test_url_without_guardrails_is_left_alone(self, harness, recorder):
        url = f"{MCP_URL}?profile=core&tools=a,b"
        result = run(harness, "--url-form", "--mcp-url", url, "-y")
        assert result.exit_code == 0, result.output
        assert printed_url(harness, result.output) == url
        assert "Warning" not in result.output


# =============================================================================
# The OAuth URL form (--url-form --oauth, #282)
# =============================================================================

OAUTH = ("--url-form", "--oauth", "--mcp-url", MCP_URL)
#: What an --oauth entry must never carry: a header, a key variable or a reference.
KEY_MARKERS = ("Authorization", "bearer_token_env_var", "bearer-token-env-var", "${", "headers")


def files_under(root: Path) -> dict[Path, bytes]:
    return {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}


def printed_block(output: str) -> list[str]:
    """The block setup printed: the 4-space-indented lines after "Add this …:" and a blank."""
    after = output.split("Add this", 1)[1].splitlines()[2:]
    return list(itertools.takewhile(lambda line: line.startswith("    "), after))


@pytest.mark.parametrize("harness", ["codex", "hermes", "openclaw"])
class TestOAuthFlags:
    @pytest.mark.parametrize(
        "flags",
        [
            ["--oauth", "--profile", "default"],
            ["--oauth", "--mcp-url", MCP_URL],
            ["--url-form", "--oauth"],
        ],
        ids=["no-url-form", "mcp-url-only", "no-mcp-url"],
    )
    def test_oauth_needs_url_form_and_mcp_url(self, harness, on_path, recorder, system_info, flags):
        on_path(harness)
        result = run(harness, *flags, "-y")
        assert result.exit_code == 2
        assert "--oauth needs --url-form and --mcp-url" in flat(result.output)
        assert recorder.calls == [] and system_info.requests == []

    def test_oauth_refuses_api_key_env(self, harness, on_path, recorder, system_info):
        on_path(harness)
        result = run(harness, *OAUTH, "--api-key-env", "KAGURA_API_KEY", "-y")
        assert result.exit_code == 2
        assert "--oauth and --api-key-env exclude each other" in flat(result.output)
        assert recorder.calls == [] and system_info.requests == []

    def test_a_bad_mcp_url_stops_before_the_server_check(self, harness, recorder, system_info):
        result = run(harness, "--url-form", "--oauth", "--mcp-url", "http://example.com/mcp", "-y")
        assert result.exit_code == 2
        assert system_info.requests == []

    def test_profile_on_another_server_stops_before_the_server_check(
        self, harness, recorder, system_info
    ):
        other = "https://kagura.example.com/mcp/w/ws-1"
        result = run(harness, "--profile", "default", *OAUTH[:3], other, "-y")
        assert result.exit_code == 2
        assert "must be on the same server" in flat(result.output)
        assert system_info.requests == []


@pytest.mark.parametrize("harness", ["codex", "hermes", "openclaw"])
class TestOAuthServerCheck:
    def test_0_77_0_proceeds_with_one_unauthenticated_request(
        self, harness, on_path, recorder, system_info, monkeypatch
    ):
        monkeypatch.setenv("KAGURA_API_KEY", API_KEY)
        on_path(harness)
        result = run(harness, "--url-form", "--oauth", "--mcp-url", f"{MCP_URL}?profile=core", "-y")
        assert result.exit_code == 0, result.output
        [request] = system_info.requests
        assert str(request.url) == "https://memory.kagura-ai.com/api/v1/system/info"
        assert "authorization" not in {k.lower() for k in request.headers}
        assert "runs memory-cloud 0.77.0" in result.output

    @pytest.mark.parametrize(
        ("version", "status", "exc", "said"),
        [
            ("0.76.0", 200, None, "runs memory-cloud 0.76.0"),
            ("0.77.0-rc1", 200, None, "runs memory-cloud 0.77.0-rc1"),
            ("main-abc123", 200, None, "reports main-abc123"),
            (None, 200, None, "reports no version"),
            ("0.77.0", 500, None, "could not confirm"),
            ("0.77.0", 200, httpx.ConnectError, "could not confirm"),
        ],
        ids=["0.76.0", "rc", "unparseable", "no-version", "500", "connection-error"],
    )
    def test_anything_else_stops_before_detection_and_writes_nothing(
        self, harness, env, on_path, recorder, system_info, digest, version, status, exc, said
    ):
        system_info.version, system_info.status, system_info.exc = version, status, exc
        on_path(harness)
        # An entry that detection would report, and a file the export would change.
        codex_config(env).write_text(
            '[mcp_servers.kagura-memory]\nurl = "https://x/mcp"\n', encoding="utf-8"
        )
        recorder.hermes_entries["kagura-memory"] = {"url": "https://x/mcp"}
        recorder.detect_out["openclaw"] = json.dumps({"url": "https://x/mcp"})
        before = files_under(env.parent)
        result = run(harness, *OAUTH, "--context-id", CTX, "--agents-md", "--force", "-y")
        assert result.exit_code == 1
        message = flat(result.output)
        assert said in message
        assert "Nothing was written" in message
        assert "--oauth needs memory-cloud 0.77.0+" in message
        assert "memory-cloud#1657" in message
        assert "the default stdio entry (--profile NAME" in message
        assert "--url-form with an API key" in message
        assert "Existing kagura-memory entry" not in result.output
        assert recorder.calls == []
        assert digest.calls == []
        assert files_under(env.parent) == before
        assert len(system_info.requests) == 1

    def test_dry_run_sends_no_request(self, harness, on_path, recorder, system_info, tty):
        on_path(harness)
        result = run(harness, *OAUTH, "--dry-run")
        assert result.exit_code == 0, result.output
        assert system_info.requests == []
        assert "The real run first checks that https://memory.kagura-ai.com runs memory-cloud" in (
            result.output
        )
        assert recorder.mutating() == []

    @pytest.mark.parametrize(
        "form", [["--profile", "default"], ["--url-form", "--mcp-url", MCP_URL]]
    )
    def test_the_other_forms_send_no_request(self, harness, on_path, system_info, form):
        on_path(harness)
        assert run(harness, *form, "-y").exit_code == 0
        assert system_info.requests == []


class TestOAuthCodex:
    def test_tty_runs_a_bare_url_add_attached(self, on_path, recorder, tty):
        on_path("codex")
        result = run("codex", *OAUTH)
        assert result.exit_code == 0, result.output
        [(argv, kwargs)] = [c for c in recorder.calls if c[0] in recorder.mutating()]
        assert argv == ["/usr/bin/codex", "mcp", "add", "kagura-memory", "--url", MCP_URL]
        # It starts Codex's browser sign-in: no captured output, no timeout.
        assert "capture_output" not in kwargs and "timeout" not in kwargs
        out = flat(result.output)
        assert "Codex signs in itself" in out
        assert "If it did not log in, run `codex mcp login kagura-memory`" in out
        assert "when the browser cannot reach it" in out and "add --no-browser" in out
        assert "needs a browser" not in out
        assert "setup never sees it" in out
        assert (
            'the OS keyring ("Codex MCP Credentials"; on Windows, its encrypted secrets store '
            "in ~/.codex), else in ~/.codex/.credentials.json"
        ) in out
        assert "export KAGURA_API_KEY" not in out

    @pytest.mark.parametrize("flags", [["-y"], []], ids=["-y", "no-tty"])
    def test_without_tty_or_with_y_prints_the_table_and_the_login(
        self, on_path, recorder, env, flags, monkeypatch
    ):
        on_path("codex")
        monkeypatch.setattr(setup_harness, "_stdin_is_tty", lambda: bool(flags))
        result = run("codex", *OAUTH, *flags)
        assert result.exit_code == 0, result.output
        assert recorder.mutating() == []
        why = "-y was given" if flags else "stdin is not a terminal"
        assert f"(`codex mcp add` starts the sign-in and {why})" in result.output
        assert printed_block(result.output) == [
            "    [mcp_servers.kagura-memory]",
            f'    url = "{MCP_URL}"',
        ]
        assert "bearer_token_env_var" not in result.output
        assert "sign in with `codex mcp login kagura-memory`" in flat(result.output)
        assert not (env / ".codex").exists()

    def test_without_codex_prints_the_table(self, recorder):
        result = run("codex", *OAUTH, "-y")
        assert result.exit_code == 0, result.output
        assert "`codex` is not on PATH" in result.output
        assert f'url = "{MCP_URL}"' in result.output
        assert "bearer_token_env_var" not in result.output
        assert recorder.calls == []

    def test_force_runs_the_same_add(self, env, on_path, recorder, tty):
        on_path("codex")
        codex_config(env).write_text(
            '[mcp_servers.kagura-memory]\nurl = "https://x/mcp"\nbearer_token_env_var = "K"\n',
            encoding="utf-8",
        )
        result = run("codex", *OAUTH, "--force")
        assert result.exit_code == 0, result.output
        assert recorder.mutating() == [
            ["/usr/bin/codex", "mcp", "add", "kagura-memory", "--url", MCP_URL]
        ]

    def test_failed_add_says_the_entry_may_be_saved(self, on_path, recorder, tty):
        on_path("codex")
        recorder.returncodes["mcp add"] = 1
        result = run("codex", *OAUTH)
        assert result.exit_code == 1
        out = flat(result.output)
        assert "`codex mcp add` failed: exit code 1" in out
        assert "may be saved already" in out
        assert "`codex mcp login kagura-memory`" in out

    def test_context_goes_in_the_url_and_the_preview_uses_the_profile(
        self, on_path, recorder, tty, monkeypatch
    ):
        monkeypatch.setenv("KAGURA_API_KEY", API_KEY)
        on_path("codex")
        result = run("codex", "--profile", "work", *OAUTH, "--context-id", CTX)
        assert result.exit_code == 0, result.output
        [argv] = recorder.mutating()
        assert argv[-2:] == ["--url", f"{MCP_URL}?guardrails={CTX}"]
        out = flat(result.output)
        assert (
            f"env -u KAGURA_API_KEY KAGURA_PROFILE=work kagura guardrails digest {CTX} "
            "--target instructions"
        ) in out
        assert "Codex gets what the account it signed in with can read" in out
        assert "changing its ?guardrails= later" in out
        assert "KAGURA_MCP_URL" not in out

    def test_without_profile_the_preview_uses_the_chain_on_the_entry_server(
        self, on_path, recorder, tty
    ):
        on_path("codex")
        result = run("codex", *OAUTH, "--context-id", CTX)
        assert result.exit_code == 0, result.output
        out = flat(result.output)
        assert f"can read): kagura guardrails digest {CTX} --target instructions" in out
        assert "usual credential is not on" not in out and "KAGURA_MCP_URL" not in out

    @pytest.mark.parametrize("chain", ["api-key", "profile", "none"])
    def test_without_profile_no_preview_runs_on_another_server(
        self, on_path, recorder, tty, monkeypatch, chain
    ):
        # KAGURA_API_KEY (on the default server, no KAGURA_MCP_URL), the default
        # profile (on memory.kagura-ai.com), or no credential at all.
        if chain == "api-key":
            monkeypatch.setenv("KAGURA_API_KEY", API_KEY)
        elif chain == "none":
            save_credentials_file(CredentialsFile())
        other = "https://kagura.example.com/mcp/w/ws-1"
        on_path("codex")
        result = run("codex", "--url-form", "--oauth", "--mcp-url", other, "--context-id", CTX)
        assert result.exit_code == 0, result.output
        out = flat(result.output)
        assert (
            "The kagura CLI's usual credential is not on https://kagura.example.com, and no "
            "profile is: log in there with "
            "`kagura auth login --server https://kagura.example.com --profile NAME`"
        ) in out
        preview = f"KAGURA_PROFILE=NAME kagura guardrails digest {CTX} --target instructions"
        assert (f"env -u KAGURA_API_KEY {preview}" if chain == "api-key" else preview) in out
        # Pinning the server would send the key for another server to this one.
        assert "KAGURA_MCP_URL" not in out
        assert API_KEY not in out

    @pytest.mark.parametrize("chain", ["profile", "api-key", "require-profile"])
    def test_without_profile_the_preview_names_a_profile_on_the_entry_server(
        self, on_path, recorder, tty, monkeypatch, chain
    ):
        # The usual chain is on another server (the default profile, or
        # KAGURA_API_KEY on the default server) or refuses the implicit default
        # (KAGURA_REQUIRE_PROFILE), but the stored profile "work" is on this one.
        other = "https://kagura.example.com"
        cf = CredentialsFile()
        cf.set_profile("default", make_oauth_creds())
        cf.set_profile("work", make_oauth_creds(server=other))
        save_credentials_file(cf)
        if chain == "api-key":
            monkeypatch.setenv("KAGURA_API_KEY", API_KEY)
        elif chain == "require-profile":
            monkeypatch.setenv("KAGURA_REQUIRE_PROFILE", "1")
        on_path("codex")
        mcp_url = f"{other}/mcp/w/ws-1"
        result = run("codex", "--url-form", "--oauth", "--mcp-url", mcp_url, "--context-id", CTX)
        assert result.exit_code == 0, result.output
        out = flat(result.output)
        assert (
            f"The kagura CLI's usual credential is not on {other}; preview it on a profile "
            "there (work)"
        ) in out
        preview = f"KAGURA_PROFILE=work kagura guardrails digest {CTX} --target instructions"
        assert (f"env -u KAGURA_API_KEY {preview}" if chain == "api-key" else preview) in out
        assert "kagura auth login" not in out and "NAME" not in out
        assert "KAGURA_MCP_URL" not in out and API_KEY not in out

    def test_require_profile_names_every_profile_on_the_entry_server(
        self, on_path, recorder, tty, monkeypatch
    ):
        # Both stored profiles are on the entry's server, but the usual chain
        # refuses the implicit default.
        monkeypatch.setenv("KAGURA_REQUIRE_PROFILE", "1")
        on_path("codex")
        result = run("codex", *OAUTH, "--context-id", CTX)
        assert result.exit_code == 0, result.output
        out = flat(result.output)
        assert (
            "The kagura CLI's usual credential is not on https://memory.kagura-ai.com; "
            "preview it on a profile there (default, work)"
        ) in out
        assert f"KAGURA_PROFILE=default kagura guardrails digest {CTX} --target instructions" in out
        assert "kagura auth login" not in out

    @pytest.mark.parametrize(
        "config",
        [
            "{not json",
            None,
            "[]",
            "null",
            '"x"',
            "1",
            '{"api_key": 123}',
            '{"api_key": "k", "mcp_url": 5}',
        ],
        ids=[
            "malformed",
            "unreadable",
            "list",
            "null",
            "string",
            "number",
            "api-key-not-string",
            "mcp-url-not-string",
        ],
    )
    def test_an_unusable_kagura_json_leaves_the_written_entry_done(
        self, on_path, recorder, tty, config
    ):
        # No profile and no key: the usual chain reaches ./.kagura.json, and
        # only after the add has written the entry.
        save_credentials_file(CredentialsFile())
        if config is None:
            Path(".kagura.json").mkdir()
        else:
            Path(".kagura.json").write_text(config, encoding="utf-8")
        on_path("codex")
        result = run("codex", *OAUTH, "--context-id", CTX)
        assert result.exit_code == 0, result.output
        out = flat(result.output)
        assert "Done: codex wrote kagura-memory" in out
        assert "Setup failed" not in out
        assert "usual credential is not on https://memory.kagura-ai.com, and no profile is" in out
        assert f"KAGURA_PROFILE=NAME kagura guardrails digest {CTX} --target instructions" in out
        assert "Check it with: codex mcp get kagura-memory" in out
        # Every kagura command reads .kagura.json first: a file it cannot read
        # or parse fails the preview too. One that parses does not, on a profile.
        broken = f"The preview fails until {Path('.kagura.json').absolute()}"
        if config in ("{not json", None):
            why = "not UTF-8 JSON" if config else "Is a directory"
            assert f"{broken} ({why}) is fixed or removed:" in out
            assert "{not json" not in out
        else:
            assert "The preview fails until" not in out

    def test_a_profile_with_an_unusable_url_is_left_out(self, on_path, recorder, tty):
        # The default profile (the usual chain) has an mcp_url that is not a
        # string; "work" is on the entry's server.
        path = credentials.DEFAULT_CREDENTIALS_PATH
        data = json.loads(path.read_text(encoding="utf-8"))
        data["profiles"]["default"]["mcp_url"] = 5
        path.write_text(json.dumps(data), encoding="utf-8")
        on_path("codex")
        result = run("codex", *OAUTH, "--context-id", CTX)
        assert result.exit_code == 0, result.output
        out = flat(result.output)
        assert "Done: codex wrote kagura-memory" in out
        assert (
            "The kagura CLI's usual credential is not on https://memory.kagura-ai.com; "
            "preview it on a profile there (work)"
        ) in out
        assert f"KAGURA_PROFILE=work kagura guardrails digest {CTX} --target instructions" in out

    @pytest.mark.parametrize("context", [[], ["--context-id", CTX]], ids=["no-context", "context"])
    def test_hooks_on_neither_turn_guardrails_off_nor_stay_silent(
        self, env, on_path, recorder, context
    ):
        on_path("codex")
        turn_on_codex_hooks(env)
        result = run("codex", *OAUTH, *context, "-y")
        assert result.exit_code == 0, result.output
        [line] = [x for x in result.output.splitlines() if x.strip().startswith("url = ")]
        url = json.loads(line.split("url = ", 1)[1])
        assert url == (f"{MCP_URL}?guardrails={CTX}" if context else MCP_URL)
        assert "guardrails=off" not in result.output
        out = flat(result.output)
        assert "guardrail hooks read their credential only from a URL entry with a bearer" in out
        assert "so with an --oauth entry they do nothing" in out
        assert "re-run with --url-form and an API key (no --oauth)" in out

    def test_hooks_on_ask_before_the_oauth_entry(self, env, on_path, recorder, tty):
        on_path("codex")
        turn_on_codex_hooks(env)
        result = run("codex", *OAUTH, input="n\n")
        assert result.exit_code == 1
        assert "Write the --oauth entry anyway?" in result.output
        assert recorder.mutating() == []

    def test_explicit_guardrails_off_is_kept(self, on_path, recorder, tty):
        on_path("codex")
        result = run("codex", *OAUTH, "--guardrails", "off")
        assert result.exit_code == 0, result.output
        [argv] = recorder.mutating()
        assert argv[-1] == f"{MCP_URL}?guardrails=off"


class TestOAuthHermes:
    def test_tty_runs_the_oauth_add_attached_and_reads_auth_back(self, on_path, recorder, tty):
        on_path("hermes")
        result = run("hermes", *OAUTH, input="n\n")
        assert result.exit_code == 0, result.output
        [(argv, kwargs)] = [c for c in recorder.calls if c[0] in recorder.mutating()]
        # The add's probe runs the sign-in: `hermes mcp login`'s bound, not 30 s.
        assert argv == [
            "/usr/bin/hermes",
            *("mcp", "add", "kagura-memory", "--url", MCP_URL, "--auth", "oauth"),
            *("--connect-timeout", "315"),
        ]
        assert "capture_output" not in kwargs and "timeout" not in kwargs
        assert [
            "/usr/bin/hermes",
            *("config", "get", "mcp_servers.kagura-memory", "--json"),
        ] in recorder.argvs()
        out = flat(result.output)
        assert "Done: hermes wrote kagura-memory" in out
        assert "with --connect-timeout 315" in out
        assert "keeps as the entry's connect_timeout" in out
        assert "`hermes mcp login kagura-memory`" in out
        assert "`hermes mcp login kagura-memory --flow device`" in out
        assert "paste the redirect URL at Hermes's prompt" in out
        assert "~/.hermes/mcp-tokens/kagura-memory.json" in out
        assert "MCP_KAGURA_MEMORY_API_KEY" not in out

    def test_an_entry_without_auth_oauth_is_not_saved(self, on_path, recorder, tty, digest):
        """Hermes could not set up OAuth, and continued without authentication."""
        on_path("hermes")
        recorder.hermes_oauth_ok = False
        result = run("hermes", *OAUTH, "--context-id", CTX, "--agents-md", input="n\n")
        assert result.exit_code == 1
        out = flat(result.output)
        assert "Hermes's kagura-memory entry has no auth: oauth" in out
        assert "Hermes continues without authentication when it cannot set up OAuth" in out
        assert "Re-run with --force to replace it" in out
        assert "overwrite prompt" not in out
        assert "setup skipped the AGENTS.md export" in out
        assert "Done:" not in out
        assert digest.calls == []

    def test_a_kept_entry_is_not_called_saved(self, on_path, recorder, tty):
        """--force, and the user declines Hermes's overwrite of a header entry.

        Hermes writes a header entry as ``headers`` with no ``auth`` key.
        """
        on_path("hermes")
        recorder.hermes_entries["kagura-memory"] = {
            "url": MCP_URL,
            "headers": {"Authorization": "Bearer ${MCP_KAGURA_MEMORY_API_KEY}"},
        }
        recorder.hermes_saves = False
        result = run("hermes", *OAUTH, "--force", input="n\n")
        assert result.exit_code == 1
        out = flat(result.output)
        assert "Hermes's kagura-memory entry has no auth: oauth" in out
        assert "Hermes keeps the existing entry when its overwrite prompt is declined" in out
        assert "Re-run with --force and accept Hermes's overwrite prompt" in out
        assert "Re-run with --force to replace it" not in out
        assert "Hermes saved" not in out and "Done:" not in out

    @pytest.fixture
    def kept_oauth_entry(self, recorder) -> str:
        """An OAuth entry of the same name, for another workspace's URL."""
        old = "https://memory.kagura-ai.com/mcp/w/ws-OLD"
        recorder.detect_out["hermes"] = f"  kagura-memory    {old}   all\n"
        recorder.hermes_entries["kagura-memory"] = {"url": old, "auth": "oauth"}
        return old

    def test_a_kept_oauth_entry_is_not_called_saved(self, on_path, recorder, tty, kept_oauth_entry):
        """--force, and the user declines Hermes's overwrite of an OAuth entry.

        It already has ``auth: oauth`` and is enabled: only its url tells.
        """
        on_path("hermes")
        recorder.hermes_saves = False
        result = run("hermes", *OAUTH, "--force", input="n\n")
        assert result.exit_code == 1
        assert [
            "/usr/bin/hermes",
            *("config", "get", "mcp_servers.kagura-memory", "--json"),
        ] in recorder.argvs()
        out = flat(result.output)
        assert "Hermes's kagura-memory entry is still the existing one (URL with OAuth)" in out
        assert "Hermes keeps the existing entry when its overwrite prompt is declined" in out
        assert "Re-run with --force and accept Hermes's overwrite prompt" in out
        assert "no auth: oauth" not in out and "Done:" not in out
        # Described, never echoed.
        assert kept_oauth_entry not in out

    def test_a_replaced_oauth_entry_is_done(self, on_path, recorder, tty, kept_oauth_entry):
        on_path("hermes")
        result = run("hermes", *OAUTH, "--force", input="n\n")
        assert result.exit_code == 0, result.output
        assert recorder.hermes_entries["kagura-memory"]["url"] == MCP_URL
        assert "Done: hermes wrote kagura-memory" in flat(result.output)

    def test_an_unread_entry_is_not_called_replaced(self, on_path, recorder, tty, kept_oauth_entry):
        """An older Hermes whose `config get` fails: `mcp list` shows only the form."""
        on_path("hermes")
        recorder.hermes_unreadable.add("")
        result = run("hermes", *OAUTH, "--force", input="n\n")
        assert result.exit_code == 1
        out = flat(result.output)
        assert "Setup could not read back Hermes's kagura-memory entry" in out
        assert "whether Hermes saved an OAuth entry it can sign in with" in out
        assert "Done:" not in out

    def test_an_unread_auth_is_not_called_missing(self, on_path, recorder, tty):
        on_path("hermes")
        recorder.returncodes["config get"] = 2
        result = run("hermes", *OAUTH, input="n\n")
        assert result.exit_code == 1
        out = flat(result.output)
        assert "Setup could not read back Hermes's kagura-memory entry" in out
        assert "`hermes config get mcp_servers.kagura-memory` failed" in out
        assert "`hermes mcp list`" in out
        assert "no auth: oauth" not in out and "Done:" not in out

    def test_an_entry_saved_disabled_is_not_done(self, on_path, recorder, tty, digest):
        """A sign-in that did not finish, then "Save config anyway?": enabled false."""
        on_path("hermes")
        recorder.hermes_probe_ok = False
        result = run("hermes", *OAUTH, "--context-id", CTX, "--agents-md", input="n\n")
        assert result.exit_code == 1
        out = flat(result.output)
        assert "Hermes saved kagura-memory disabled" in out
        assert "Sign in with `hermes mcp login kagura-memory`" in out
        assert "add --flow device on memory-cloud 0.78.0+" in out
        assert "`hermes config set mcp_servers.kagura-memory.enabled true`" in out
        assert "setup skipped the AGENTS.md export" in out
        assert "Done:" not in out
        assert digest.calls == []

    @pytest.mark.parametrize("flags", [["-y"], []], ids=["-y", "no-tty"])
    def test_printed_block_has_auth_oauth_and_no_headers(
        self, on_path, recorder, flags, monkeypatch
    ):
        on_path("hermes")
        monkeypatch.setattr(setup_harness, "_stdin_is_tty", lambda: bool(flags))
        result = run("hermes", *OAUTH, *flags)
        assert result.exit_code == 0, result.output
        assert recorder.mutating() == []
        assert printed_block(result.output) == [
            "    mcp_servers:",
            "      kagura-memory:",
            f'        url: "{MCP_URL}"',
            "        auth: oauth",
        ]
        assert "headers" not in result.output
        assert "sign in with `hermes mcp login kagura-memory`" in flat(result.output)

    def test_block_under_an_existing_key(self, env, recorder):
        config = env / ".hermes" / "config.yaml"
        config.parent.mkdir(parents=True)
        config.write_text("mcp_servers:\n  other:\n    command: foo\n", encoding="utf-8")
        result = run("hermes", *OAUTH, "-y")
        assert result.exit_code == 0, result.output
        assert f'\n      kagura-memory:\n        url: "{MCP_URL}"\n        auth: oauth\n' in (
            result.output
        )


class TestOAuthOpenClaw:
    def test_add_argv(self, on_path, recorder):
        on_path("openclaw")
        result = run("openclaw", *OAUTH, "-y")
        assert result.exit_code == 0, result.output
        assert recorder.mutating() == [
            [
                "/usr/bin/openclaw",
                *("mcp", "add", "kagura-memory", "--url", MCP_URL),
                *("--transport", "streamable-http", "--auth", "oauth"),
            ]
        ]
        out = flat(result.output)
        assert "Sign in with `openclaw mcp login kagura-memory`" in out
        assert "Check it with: openclaw mcp doctor kagura-memory --probe" in out
        assert "~/.openclaw/state/openclaw.sqlite" in out
        assert "setup never sees it" in out
        assert ".env" not in out

    def test_force_uses_mcp_set(self, on_path, recorder):
        on_path("openclaw")
        recorder.detect_out["openclaw"] = json.dumps({"command": "kagura-mcp"})
        result = run("openclaw", *OAUTH, "--force", "-y")
        assert result.exit_code == 0, result.output
        [argv] = recorder.mutating()
        assert argv[:4] == ["/usr/bin/openclaw", "mcp", "set", "kagura-memory"]
        assert json.loads(argv[4]) == {
            "url": MCP_URL,
            "transport": "streamable-http",
            "auth": "oauth",
        }

    def test_printed_block(self, recorder):
        result = run("openclaw", *OAUTH, "-y")
        assert result.exit_code == 0, result.output
        assert printed_json(result.output) == {
            "mcp": {
                "servers": {
                    "kagura-memory": {
                        "url": MCP_URL,
                        "transport": "streamable-http",
                        "auth": "oauth",
                    }
                }
            }
        }
        assert recorder.calls == []
        out = flat(result.output)
        assert (
            "Once the entry is in openclaw.json, sign in with `openclaw mcp login kagura-memory`"
        ) in out
        assert "`openclaw mcp login kagura-memory --code <code>`" in out


@pytest.mark.parametrize(
    ("harness", "login"),
    [
        ("codex", "codex mcp login NAME"),
        ("hermes", "hermes mcp login NAME"),
        ("openclaw", "openclaw mcp login NAME"),
    ],
)
def test_help_documents_the_oauth_form_and_its_login(harness, login):
    result = run(harness, "--help")
    assert result.exit_code == 0, result.output
    out = flat(result.output)
    assert "--url-form --oauth --mcp-url https://memory.kagura-ai.com/mcp/w/WS_ID" in out
    assert "memory-cloud 0.77.0+" in out
    assert "client registration" in out
    assert login in out
    # Each harness has signed in end to end on a /mcp/w/<workspace-id> URL (#284).
    assert "not verified" not in out


@pytest.mark.parametrize(
    ("version", "shown"),
    [
        ("0.76.0", "0.76.0"),
        ("0.76.0\x1b[2J", "'0.76.0\\x1b[2J'"),
        (76, "76"),
        ("9" * 100, "'" + "9" * 12 + "..."),
    ],
    ids=["plain", "control-characters", "not-a-string", "long"],
)
def test_a_reported_version_is_printed_safely(version, shown):
    assert setup_harness._shown_version(version).startswith(shown)
    assert len(setup_harness._shown_version(version)) <= 64


# =============================================================================
# Shared rules
# =============================================================================


@pytest.mark.parametrize("harness", ["codex", "hermes", "openclaw"])
class TestSharedFlags:
    def test_profile_is_required_without_url_form(self, harness):
        result = run(harness, "-y")
        assert result.exit_code == 2
        assert "--profile is required" in result.output

    def test_url_form_needs_mcp_url(self, harness):
        assert run(harness, "--url-form", "-y").exit_code == 2

    def test_mcp_url_without_url_form(self, harness):
        assert run(harness, "--profile", "default", "--mcp-url", MCP_URL, "-y").exit_code == 2

    def test_plain_http_mcp_url_is_refused(self, harness):
        result = run(harness, "--url-form", "--mcp-url", "http://example.com/mcp", "-y")
        assert result.exit_code == 2

    @pytest.mark.parametrize(
        "url", ["HTTP://example.com/mcp", " http://example.com/mcp", "hTtP://example.com/mcp"]
    )
    def test_plain_http_mcp_url_in_any_spelling_is_refused(self, harness, on_path, recorder, url):
        on_path(harness)
        result = run(harness, "--url-form", "--mcp-url", url, "-y")
        assert result.exit_code == 2
        assert "must use HTTPS" in result.output
        assert recorder.calls == []

    @pytest.mark.parametrize(
        "url",
        [
            "--help",
            "memory.kagura-ai.com/mcp",
            "ftp://memory.kagura-ai.com/mcp",
            "https://",
            "https://[::1/mcp",
        ],
    )
    def test_mcp_url_that_is_no_https_url_is_refused(self, harness, on_path, recorder, url):
        # It goes on the harness argv after --url, where "--help" would read as an option.
        on_path(harness)
        result = run(harness, "--url-form", f"--mcp-url={url}", "-y")
        assert result.exit_code == 2
        assert "use an https:// URL" in result.output
        assert recorder.calls == []

    def test_bad_name(self, harness):
        assert run(harness, "--profile", "default", "--name", "a.b", "-y").exit_code == 2

    @pytest.mark.parametrize("name", ["--help", "-h", "-", "_kagura", "-kagura"])
    def test_name_must_start_with_a_letter_or_digit(self, harness, on_path, recorder, name):
        # The name is a positional in every harness argv: `codex mcp add --help …`
        # prints the help, exits 0 and configures nothing (#274).
        on_path(harness)
        result = run(harness, "--profile", "default", f"--name={name}", "-y")
        assert result.exit_code == 2
        assert "starting with a letter or digit" in result.output
        assert recorder.calls == []

    @pytest.mark.parametrize("flags", [["-y"], []], ids=["-y", "no-tty"])
    def test_agents_md_without_prompts_needs_a_context(self, harness, flags):
        result = run(harness, "--profile", "default", "--agents-md", *flags)
        assert result.exit_code == 2
        assert "--context-id" in result.output

    def test_missing_profile(self, harness):
        result = run(harness, "--profile", "ghost", "-y")
        assert result.exit_code == 1
        assert "kagura auth login --profile ghost" in result.output

    def test_failed_profile_check_writes_nothing(self, harness, on_path, recorder, connection):
        from kagura_memory.exceptions import KaguraAuthError

        on_path(harness)
        connection.side_effect = KaguraAuthError("expired")
        result = run(harness, "--profile", "default", "-y")
        assert result.exit_code == 1
        assert "Authentication failed" in result.output
        assert recorder.mutating() == []


class TestProxyPath:
    def test_absolute_path_from_path(self, monkeypatch):
        monkeypatch.setattr(setup_harness.shutil, "which", lambda cmd, path=None: "bin/kagura-mcp")
        assert REAL_PROXY_PATH() == str(Path("bin/kagura-mcp").absolute())

    def test_falls_back_to_the_python_directory(self, monkeypatch):
        seen = []

        def which(cmd, path=None):
            seen.append(path)
            return None if path is None else f"{path}/kagura-mcp"

        monkeypatch.setattr(setup_harness.shutil, "which", which)
        assert REAL_PROXY_PATH() == f"{Path(sys.executable).parent}/kagura-mcp"
        assert seen == [None, str(Path(sys.executable).parent)]

    def test_missing_kagura_mcp_fails_the_stdio_form(self, monkeypatch):
        monkeypatch.setattr(setup_harness, "_proxy_path", REAL_PROXY_PATH)
        monkeypatch.setattr(setup_harness.shutil, "which", lambda cmd, path=None: None)
        result = run("codex", "--profile", "default", "-y")
        assert result.exit_code == 1
        assert "not found on $PATH" in result.output
        assert "--url-form" in result.output


@pytest.mark.parametrize("harness", ["codex", "hermes", "openclaw"])
def test_dry_run_changes_nothing(harness, env, on_path, recorder, connection, digest, tty):
    on_path(harness)
    Path("AGENTS.md").write_text("# Mine\n", encoding="utf-8")
    before = sorted(p for p in env.parent.rglob("*"))
    result = run(harness, "--profile", "default", "--context-id", CTX, "--agents-md", "--dry-run")
    assert result.exit_code == 0, result.output
    assert "Dry run" in result.output
    assert "Would run:" in result.output
    assert "AGENTS.md: would" in result.output
    assert recorder.mutating() == []
    connection.assert_not_called()
    assert digest.calls == []
    assert sorted(p for p in env.parent.rglob("*")) == before
    assert Path("AGENTS.md").read_text(encoding="utf-8") == "# Mine\n"


def test_dry_run_reports_an_existing_entry_and_prints_the_block(env, recorder):
    codex_config(env).write_text(
        '[mcp_servers.kagura-memory]\nurl = "https://x/mcp"\n', encoding="utf-8"
    )
    result = run("codex", "--profile", "default", "--dry-run")
    assert result.exit_code == 0, result.output
    assert "URL with OAuth or no credential" in result.output
    assert "Setup would stop here" in result.output
    assert "[mcp_servers.kagura-memory]" in result.output


@pytest.mark.parametrize("harness", ["codex", "hermes", "openclaw"])
@pytest.mark.parametrize(
    "form",
    [[], ["--url-form", "--mcp-url", MCP_URL], list(OAUTH)],
    ids=["stdio", "url-form", "oauth"],
)
@pytest.mark.parametrize("flags", [["-y"], []], ids=["-y", "tty"])
def test_no_secret_in_output_argv_or_files(
    harness, form, flags, env, on_path, recorder, monkeypatch, tty
):
    """The access token and an exported API key never reach output, argv or a file.

    An ``--oauth`` entry also carries no header, key variable or reference.
    """
    monkeypatch.setenv("KAGURA_API_KEY", API_KEY)
    on_path(harness)
    result = run(
        harness,
        "--profile",
        "default",
        *form,
        *("--context-id", CTX, "--agents-md", *flags),
        input="n\n",
    )
    assert result.exit_code == 0, result.output
    for secret in (ACCESS_TOKEN, API_KEY):
        assert secret not in result.output
        assert all(secret not in " ".join(argv) for argv in recorder.argvs())
        for path in env.parent.rglob("*"):
            if path.is_file() and "credentials" not in path.name:
                assert secret not in path.read_text(encoding="utf-8", errors="replace")
    if "--oauth" in form:
        for marker in KEY_MARKERS:
            assert marker not in result.output
            assert all(marker not in " ".join(argv) for argv in recorder.argvs())


# =============================================================================
# The AGENTS.md export
# =============================================================================


class TestExport:
    def args(self, path: Path) -> list[str]:
        return [
            "codex",
            "--profile",
            "default",
            "--context-id",
            CTX,
            "--agents-md",
            str(path),
            "-y",
        ]

    def test_replaces_only_the_marked_block_and_is_idempotent(self, digest, tmp_path):
        path = tmp_path / "AGENTS.md"
        old = EXPORT_BLOCK.replace("abc123", "old").replace("head SHA", "old SHA")
        path.write_text(f"# Top\n\n{old}\n## Bottom\n", encoding="utf-8")
        assert run(*self.args(path)).exit_code == 0
        assert path.read_text(encoding="utf-8") == f"# Top\n\n{EXPORT_BLOCK}\n## Bottom\n"
        again = run(*self.args(path))
        assert again.exit_code == 0
        assert "Already up to date" in again.output
        assert path.read_text(encoding="utf-8") == f"# Top\n\n{EXPORT_BLOCK}\n## Bottom\n"

    def test_empty_body_writes_nothing_and_says_so(self, digest, tmp_path):
        digest.text = ""
        path = tmp_path / "AGENTS.md"
        result = run(*self.args(path))
        assert result.exit_code == 0, result.output
        assert "no tool guardrails" in result.output
        assert not path.exists()

    @pytest.mark.parametrize("newline", [b"\n", b"\r\n"], ids=["lf", "crlf"])
    def test_empty_body_removes_an_earlier_block(self, digest, tmp_path, newline):
        """As `guardrails digest --out` does: the harness stops loading a stale block (#278)."""
        digest.text = ""
        path = tmp_path / "AGENTS.md"
        body = EXPORT_BLOCK.encode().replace(b"\n", newline)
        top, bottom = b"# Top" + newline, b"## Bottom" + newline + b"text" + newline
        path.write_bytes(top + newline + body + bottom)
        result = run(*self.args(path))
        assert result.exit_code == 0, result.output
        assert path.read_bytes() == top + bottom
        assert "removed the earlier guardrail block from" in flat(result.output)

    def test_empty_body_without_a_block_creates_no_file_or_directory(self, digest, tmp_path):
        digest.text = ""
        path = tmp_path / "new-dir" / "AGENTS.md"
        result = run(*self.args(path))
        assert result.exit_code == 0, result.output
        assert "nothing was written to" in flat(result.output)
        assert not path.parent.exists()

    @pytest.mark.parametrize(
        "content",
        [
            ("# P\n\n" + EXPORT_BLOCK + "\n" + EXPORT_BLOCK).encode(),
            b"\xff\xfe not utf-8",
        ],
        ids=["two-blocks", "not-utf-8"],
    )
    def test_empty_body_leaves_a_file_it_cannot_splice_unchanged(self, digest, tmp_path, content):
        digest.text = ""
        path = tmp_path / "AGENTS.md"
        path.write_bytes(content)
        result = run(*self.args(path))
        assert result.exit_code == 1
        assert "left unchanged" in result.output
        assert path.read_bytes() == content

    @pytest.mark.parametrize(
        ("codex_on_path", "said"),
        [
            (True, "The MCP entry is set up, but"),
            # Without codex the entry is only printed: it is not set up (#285).
            (False, "The MCP entry is printed for you to add, but"),
        ],
    )
    def test_404_is_a_visibility_error_after_the_entry(
        self, digest, tmp_path, on_path, codex_on_path, said
    ):
        if codex_on_path:
            on_path("codex")
        digest.error = KaguraNotFoundError("Context not found")
        path = tmp_path / "AGENTS.md"
        result = run(*self.args(path))
        assert result.exit_code == 1
        assert "not visible to this credential on\n  https://memory.kagura-ai.com" in (
            result.output
        )
        assert said in result.output
        assert "set up" not in result.output.replace(said, "")
        assert not path.exists()

    @pytest.mark.parametrize("value", ["-y", "--dry-run", "-"])
    def test_an_equals_value_starting_with_a_dash_is_the_path(self, digest, value):
        """click parsed ``--agents-md=-y`` as the default file plus ``-y`` (#285)."""
        result = run("codex", "--profile", "default", "--context-id", CTX, f"--agents-md={value}")
        assert result.exit_code == 0, result.output
        assert Path(value).read_text(encoding="utf-8") == EXPORT_BLOCK
        assert not (Path.home() / ".codex" / "AGENTS.md").exists()
        assert "Dry run" not in result.output

    def test_an_empty_equals_value_is_the_default_file(self, env, digest):
        result = run("codex", "--profile", "default", "--context-id", CTX, "--agents-md=", "-y")
        assert result.exit_code == 0, result.output
        assert (env / ".codex" / "AGENTS.md").read_text(encoding="utf-8") == EXPORT_BLOCK

    def test_a_dash_after_a_space_is_still_the_next_option(self, env, digest):
        result = run("codex", "--profile", "default", "--context-id", CTX, "--agents-md", "-y")
        assert result.exit_code == 0, result.output
        assert (env / ".codex" / "AGENTS.md").read_text(encoding="utf-8") == EXPORT_BLOCK

    @pytest.mark.parametrize("argv", [["--agents-md= "], ["--agents-md", "  "]])
    def test_a_blank_path_is_a_usage_error(self, recorder, digest, argv):
        """It used to create a file named ``' '`` (#285)."""
        result = run("codex", "--profile", "default", "--context-id", CTX, *argv, "-y")
        assert result.exit_code == 2, result.output
        assert "the path is blank" in result.output
        assert recorder.mutating() == []
        assert list(Path.cwd().iterdir()) == []

    def test_a_tilde_user_path_is_kept_as_written(self, digest):
        """``Path.expanduser`` raised for a user that does not exist (#285)."""
        result = run(
            "codex", "--profile", "default", "--context-id", CTX,
            "--agents-md", "~no-such-user-285/AGENTS.md", "-y",
        )  # fmt: skip
        assert result.exit_code == 0, result.output
        written = Path("~no-such-user-285") / "AGENTS.md"
        assert written.read_text(encoding="utf-8") == EXPORT_BLOCK

    def test_non_default_profile_names_it_in_the_refresh_command(self, tmp_path):
        path = tmp_path / "AGENTS.md"
        args = self.args(path)
        args[args.index("default")] = "work"
        result = run(*args)
        assert result.exit_code == 0, result.output
        assert f"KAGURA_PROFILE=work kagura guardrails digest {CTX} --out" in result.output

    def test_openclaw_size_warning(self, env, on_path, digest):
        on_path("openclaw")
        path = env / ".openclaw" / "workspace" / "AGENTS.md"
        path.parent.mkdir(parents=True)
        path.write_text("x" * 20_001 + "\n", encoding="utf-8")
        result = run("openclaw", "--profile", "default", "--context-id", CTX, "--agents-md", "-y")
        assert result.exit_code == 0, result.output
        assert "reads only the first 20000" in result.output


# =============================================================================
# The written command runs under an empty environment
# =============================================================================


class _Upstream(BaseHTTPRequestHandler):
    """A memory-cloud stand-in: answers every JSON-RPC request, records the bearer."""

    seen: list[tuple[str, str | None]] = []

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.seen.append((body.get("method"), self.headers.get("Authorization")))
        if "id" not in body:
            self.send_response(202)
            self.end_headers()
            return
        result = {"tools": [{"name": "recall"}]} if body["method"] == "tools/list" else {}
        payload = json.dumps({"jsonrpc": "2.0", "id": body["id"], "result": result}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("mcp-session-id", "sess-1")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args: Any) -> None:
        pass


def test_written_stdio_command_answers_tools_list_under_an_empty_environment(
    tmp_path, on_path, recorder, monkeypatch
):
    """Run the argv setup writes with no PATH, no KAGURA_* — only a temporary HOME.

    ``env -i`` would also drop ``HOME``, and ``Path.home()`` would then read
    the real home directory from the password database; the test keeps a
    temporary one so it never touches the developer's credentials.
    """
    real = shutil.which("kagura-mcp", path=str(Path(sys.executable).parent))
    if real is None:
        pytest.skip("kagura-mcp console script is not installed in this environment")
    monkeypatch.setattr(setup_harness, "_proxy_path", REAL_PROXY_PATH)
    on_path("codex")

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    home = tmp_path / "isolated-home"
    try:
        port = server.server_address[1]
        creds = make_oauth_creds(server=f"http://127.0.0.1:{port}", access_token=ACCESS_TOKEN)
        cf = CredentialsFile()
        cf.set_profile("ci", creds)
        save_credentials_file(cf, home / ".kagura" / "credentials.json")
        monkeypatch.setattr(
            "kagura_memory.auth.credentials.DEFAULT_CREDENTIALS_PATH",
            home / ".kagura" / "credentials.json",
        )
        monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
        with patch("kagura_memory.setup_claude._test_connection", return_value=CONTEXTS):
            result = run("codex", "--profile", "ci", "-y")
        assert result.exit_code == 0, result.output
        [argv] = recorder.mutating()
        command = argv[argv.index("--") + 1 :]
        assert Path(command[0]).is_absolute()
        assert command[1:] == ["--profile", "ci"]

        requests = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        ]
        proc = REAL_RUN(
            command,
            input="".join(json.dumps(r) + "\n" for r in requests),
            capture_output=True,
            text=True,
            env={"HOME": str(home)},
            timeout=60,
            check=False,
        )
    finally:
        server.shutdown()
        server.server_close()
    assert proc.returncode == 0, proc.stderr
    responses = [json.loads(line) for line in proc.stdout.splitlines()]
    assert responses[-1] == {"jsonrpc": "2.0", "id": 2, "result": {"tools": [{"name": "recall"}]}}
    assert ("tools/list", f"Bearer {ACCESS_TOKEN}") in _Upstream.seen


# =============================================================================
# Failure paths and classification
# =============================================================================


@pytest.mark.parametrize(
    ("entry", "kind", "url_credential"),
    [
        (
            {"command": "/venv/bin/kagura-mcp", "args": ["--profile", "x"]},
            "stdio (kagura-mcp)",
            False,
        ),
        ({"command": "npx", "args": ["other-mcp"]}, "stdio (another command)", False),
        ({"url": "u", "bearer_token_env_var": "K"}, "from an environment variable", True),
        ({"url": "u", "env_http_headers": {"Authorization": "K"}}, "from an environment", True),
        ({"url": "u", "http_headers": {"authorization": f"Bearer {API_KEY}"}}, "static", True),
        (
            {"url": "u", "headers": {"Authorization": "Bearer ${KAGURA_API_KEY}"}},
            "environment",
            True,
        ),
        ({"url": "u", "headers": {"Authorization": f"Bearer {API_KEY}"}}, "static", True),
        ({"url": "u", "auth": "oauth", "headers": {"Authorization": "x"}}, "URL with OAuth", False),
        ({"url": "u"}, "OAuth or no credential", False),
        ({"transport": "sse"}, "does not recognise", False),
    ],
)
def test_classify_names_the_kind_only(entry, kind, url_credential):
    found = setup_harness._classify(entry)
    assert kind in found.kind
    assert found.url_credential is url_credential
    assert API_KEY not in found.kind


def test_unreadable_codex_config_stops_without_echoing_it(env):
    codex_config(env).write_text(f'[mcp_servers.kagura-memory\nkey = "{API_KEY}"\n', "utf-8")
    result = run("codex", "--profile", "default", "-y")
    assert result.exit_code == 1
    assert "Cannot read ~/.codex/config.toml" in result.output
    assert API_KEY not in result.output


def test_harness_command_failure_is_reported(on_path, recorder):
    on_path("openclaw")
    recorder.returncodes["mcp add"] = 1
    result = run("openclaw", "--profile", "default", "-y")
    assert result.exit_code == 1
    assert "`openclaw mcp add` failed: boom" in result.output


def test_harness_command_timeout_is_reported(on_path, monkeypatch):
    on_path("codex")

    def slow(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs.get("timeout", 0))

    monkeypatch.setattr(setup_harness.subprocess, "run", slow)
    result = run("codex", "--profile", "default", "-y")
    assert result.exit_code == 1
    assert "`codex mcp add` failed: timed out" in result.output


def test_url_form_without_profile_needs_a_context_uuid():
    result = run("codex", "--url-form", "--mcp-url", MCP_URL, "--context-id", "proj", "-y")
    assert result.exit_code == 2
    assert "must be a context UUID" in result.output


def test_url_form_without_profile_checks_nothing(on_path, recorder, connection):
    on_path("codex")
    result = run("codex", "--url-form", "--mcp-url", MCP_URL, "--context-id", CTX, "-y")
    assert result.exit_code == 0, result.output
    connection.assert_not_called()
    [argv] = recorder.mutating()
    assert argv[argv.index("--url") + 1] == f"{MCP_URL}?guardrails={CTX}"


def test_unknown_context_name_fails_before_writing(on_path, recorder):
    on_path("codex")
    result = run("codex", "--profile", "default", "--context-id", "nope", "-y")
    assert result.exit_code == 1
    assert "No context 'nope'" in result.output
    assert "Using context: nope" not in result.output
    assert recorder.mutating() == []


def test_context_prompt_with_no_visible_context(on_path, connection, tty):
    on_path("openclaw")
    connection.return_value = {"count": 0, "contexts": []}
    result = run("openclaw", "--profile", "default", "--agents-md")
    assert result.exit_code == 1
    assert "can see no context" in result.output


@pytest.mark.parametrize("harness", ["hermes", "openclaw"])
def test_guardrails_off_already_in_the_url_is_flagged(harness):
    result = run(harness, "--url-form", "--mcp-url", f"{MCP_URL}?guardrails=off", "-y")
    assert result.exit_code == 0, result.output
    assert "gets no guardrails from Kagura" in result.output


def test_api_key_env_must_be_an_upper_case_name():
    result = run("openclaw", "--url-form", "--mcp-url", MCP_URL, "--api-key-env", "my-key", "-y")
    assert result.exit_code == 2


def test_custom_name_flows_into_every_form(on_path, recorder):
    on_path("hermes")
    result = run("hermes", "--url-form", "--mcp-url", MCP_URL, "--name", "kagura_work", "-y")
    assert result.exit_code == 0, result.output
    assert "  kagura_work:" in result.output
    assert "${MCP_KAGURA_WORK_API_KEY}" in result.output
    assert "hermes mcp test kagura_work" in result.output


def test_dry_run_with_an_existing_entry_labels_the_force_commands(on_path, recorder):
    on_path("openclaw")
    recorder.detect_out["openclaw"] = json.dumps({"command": "kagura-mcp"})
    result = run("openclaw", "--profile", "default", "--dry-run")
    assert result.exit_code == 0, result.output
    assert "With --force, would run: openclaw mcp set kagura-memory" in result.output
    assert recorder.mutating() == []


@pytest.mark.parametrize(
    ("content", "action"),
    [(None, "create"), ("# Mine\n", "append the block to"), (EXPORT_BLOCK, "replace the block in")],
)
def test_dry_run_names_the_export_action(tmp_path, content, action):
    path = tmp_path / "AGENTS.md"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    args = ("--context-id", CTX, "--agents-md", str(path), "--dry-run")
    result = run("codex", "--profile", "default", *args)
    assert result.exit_code == 0, result.output
    assert f"AGENTS.md: would {action} {path}" in result.output
    assert (path.read_text(encoding="utf-8") if content is not None else None) == content


def test_export_action_for_each_file_state(tmp_path):
    path = tmp_path / "AGENTS.md"
    assert setup_harness._export_action(path) == "create"
    path.write_text("# Mine\n", encoding="utf-8")
    assert setup_harness._export_action(path) == "append the block to"
    path.write_text(EXPORT_BLOCK, encoding="utf-8")
    assert setup_harness._export_action(path) == "replace the block in"
    path.write_bytes(b"\xff\xfe not utf-8")
    assert setup_harness._export_action(path) == "update"


@pytest.mark.parametrize(
    ("flags", "terminal", "expected"),
    [
        ([], True, "offered when setup runs"),
        (["-y"], True, "not offered with -y"),
        ([], False, "not offered without a terminal"),
    ],
)
def test_dry_run_mentions_the_export_offer(flags, terminal, expected, monkeypatch):
    monkeypatch.setattr(setup_harness, "_stdin_is_tty", lambda: terminal)
    result = run("openclaw", "--profile", "default", "--dry-run", *flags)
    assert result.exit_code == 0, result.output
    assert "~/.openclaw/workspace/AGENTS.md" in result.output
    assert expected in result.output


def test_digest_failure_other_than_404_is_reported_after_the_entry(digest, tmp_path):
    from kagura_memory.exceptions import KaguraConnectionError

    digest.error = KaguraConnectionError("HTTP 503: down")
    path = tmp_path / "AGENTS.md"
    result = run("codex", "--profile", "default", "--context-id", CTX, "--agents-md", str(path))
    assert result.exit_code == 1
    assert (
        "The MCP entry is printed for you to add, but the AGENTS.md export failed: HTTP 503"
    ) in result.output
    assert not path.exists()


def test_broken_target_file_is_left_unchanged(tmp_path):
    path = tmp_path / "AGENTS.md"
    broken = "# P\n\n" + EXPORT_BLOCK + "\n" + EXPORT_BLOCK
    path.write_text(broken, encoding="utf-8")
    result = run("codex", "--profile", "default", "--context-id", CTX, "--agents-md", str(path))
    assert result.exit_code == 1
    assert "fix it by hand; left unchanged" in result.output
    assert path.read_text(encoding="utf-8") == broken


def test_codex_agents_md_size_warning_counts_bytes(env, digest):
    path = env / ".codex" / "AGENTS.md"
    path.parent.mkdir(parents=True)
    path.write_text("あ" * 11_000 + "\n", encoding="utf-8")  # 33,000 bytes, 11,001 chars
    result = run("codex", "--profile", "default", "--context-id", CTX, "--agents-md", "-y")
    assert result.exit_code == 0, result.output
    assert "bytes; Codex reads only the first 32768" in result.output


def test_codex_agents_md_size_warning_counts_crlf_as_two_bytes(env, digest):
    """read_text folded CRLF, so a CRLF file just over the cap got no warning (#285)."""
    path = env / ".codex" / "AGENTS.md"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"x\r\n" * 12_000)  # 36,000 bytes; 24,000 once CRLF is folded
    result = run("codex", "--profile", "default", "--context-id", CTX, "--agents-md", "-y")
    assert result.exit_code == 0, result.output
    size = len(path.read_bytes())
    assert size > 32_768 > len(path.read_text(encoding="utf-8"))
    assert f"is {size} bytes; Codex reads only the first 32768" in result.output


def test_detection_command_that_cannot_run_counts_as_no_entry(on_path, monkeypatch):
    on_path("hermes")
    calls = []

    def broken(argv, **kwargs):
        calls.append(argv)
        raise OSError("exec format error")

    monkeypatch.setattr(setup_harness.subprocess, "run", broken)
    result = run("hermes", "--profile", "default", "-y")
    assert result.exit_code == 0, result.output
    assert "No kagura-memory entry yet." in result.output
    assert calls == [
        ["/usr/bin/hermes", "config", "get", "mcp_servers.kagura-memory", "--json"],
        ["/usr/bin/hermes", "mcp", "list"],
    ]


@pytest.mark.parametrize("stdout", ["not json", "warning: x\n{}", "[1, 2]"])
def test_openclaw_show_output_without_an_entry(on_path, recorder, stdout):
    on_path("openclaw")
    recorder.detect_out["openclaw"] = stdout
    result = run("openclaw", "--profile", "default", "-y")
    assert result.exit_code == 0, result.output
    assert "No kagura-memory entry yet." in result.output


def test_harness_command_that_cannot_start_is_reported(on_path, monkeypatch):
    on_path("openclaw")

    def run_(argv, **kwargs):
        if argv[1:3] == ["mcp", "show"]:
            return subprocess.CompletedProcess(argv, 1, "", "")
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(setup_harness.subprocess, "run", run_)
    result = run("openclaw", "--profile", "default", "-y")
    assert result.exit_code == 1
    assert "`openclaw mcp add` failed: " in result.output
    assert "Permission denied" in result.output


def test_interactive_export_without_profile_or_context_is_a_usage_error(digest, tty):
    result = run("openclaw", "--url-form", "--mcp-url", MCP_URL, "--agents-md", input="\n")
    assert result.exit_code == 2
    assert "The AGENTS.md export needs --context-id" in result.output
    assert digest.calls == []


def test_url_form_without_profile_exports_with_the_default_credential(on_path, digest, tmp_path):
    on_path("codex")
    path = tmp_path / "AGENTS.md"
    args = ("--url-form", "--mcp-url", MCP_URL, "--context-id", CTX, "--agents-md", str(path))
    result = run("codex", *args, "-y")
    assert result.exit_code == 0, result.output
    # The CLI chain: here, the credentials file's default profile.
    assert digest.calls == [("default", CTX)]
    assert f"\n    kagura guardrails digest {CTX} --out" in result.output


@pytest.mark.parametrize("key", [True, False], ids=["KAGURA_API_KEY", "oauth-default"])
def test_url_form_export_credential_for_another_server_writes_nothing(
    on_path, recorder, digest, tmp_path, monkeypatch, key
):
    if key:
        monkeypatch.setenv("KAGURA_API_KEY", API_KEY)  # KAGURA_MCP_URL unset: the cloud host
    on_path("openclaw")
    path = tmp_path / "AGENTS.md"
    self_hosted = "https://kagura.example.com/mcp/w/ws-1"
    args = ("--url-form", "--mcp-url", self_hosted, "--context-id", CTX, "--agents-md", str(path))
    result = run("openclaw", *args, "-y")
    assert result.exit_code == 1
    assert "Nothing was written" in result.output
    assert "https://memory.kagura-ai.com" in result.output
    assert "https://kagura.example.com" in result.output
    assert API_KEY not in result.output
    assert recorder.mutating() == []
    assert digest.calls == []


def test_url_form_profile_on_another_server_is_a_usage_error(recorder):
    self_hosted = "https://kagura.example.com/mcp/w/ws-1"
    result = run("codex", "--profile", "default", "--url-form", "--mcp-url", self_hosted, "-y")
    assert result.exit_code == 2
    assert "must be on the same server" in result.output
    assert recorder.calls == []


def test_url_form_profile_on_the_same_server_lists_its_contexts(on_path, recorder, connection):
    on_path("codex")
    result = run("codex", "--profile", "default", "--url-form", "--mcp-url", MCP_URL, "-y")
    assert result.exit_code == 0, result.output
    connection.assert_called_once()


def test_profile_wins_over_kagura_api_key(on_path, digest, tmp_path, monkeypatch):
    """The entry's kagura-mcp uses the profile alone, so the export does too."""
    monkeypatch.setenv("KAGURA_API_KEY", API_KEY)
    on_path("openclaw")
    path = tmp_path / "AGENTS.md"
    args = ("--profile", "work", "--context-id", CTX, "--agents-md", str(path), "-y")
    result = run("openclaw", *args)
    assert result.exit_code == 0, result.output
    assert digest.calls == [("work", CTX)]
    assert "KAGURA_API_KEY is set" not in result.output
    assert (
        f"env -u KAGURA_API_KEY KAGURA_PROFILE=work kagura guardrails digest {CTX} --out"
        in result.output
    )


@pytest.mark.asyncio
async def test_profile_check_client_ignores_kagura_api_key(monkeypatch):
    from kagura_memory.auth.credentials import KaguraOAuth
    from kagura_memory.setup_claude import _make_client

    monkeypatch.setenv("KAGURA_API_KEY", API_KEY)
    async with _make_client(None, None, "work") as client:
        assert isinstance(client._client.auth, KaguraOAuth)
        assert "Authorization" not in client._client.headers
        assert client.mcp_url == "https://memory.kagura-ai.com/mcp"


def test_profile_auth_for_a_missing_profile():
    from kagura_memory.exceptions import KaguraAuthError

    with pytest.raises(KaguraAuthError, match="kagura auth login --profile ghost"):
        _resolve_profile_auth("ghost")


@pytest.mark.parametrize("url_form", [False, True], ids=["stdio", "url-form"])
def test_dry_run_with_a_context_name_shows_a_placeholder(on_path, recorder, connection, url_form):
    on_path("codex")
    form = ["--url-form", "--mcp-url", MCP_URL] if url_form else []
    result = run("codex", "--profile", "default", *form, "--context-id", "proj", "--dry-run")
    assert result.exit_code == 0, result.output
    connection.assert_not_called()
    placeholder = "<UUID of context proj>"
    assert "--guardrails proj" not in result.output
    assert "guardrails=proj" not in result.output
    if url_form:
        assert f'url = "{MCP_URL}?guardrails={placeholder}"' in result.output
    else:
        assert f"--guardrails '{placeholder}'" in result.output
        assert f'args = ["--profile", "default", "--guardrails", "{placeholder}"]' in result.output


def test_dry_run_with_a_context_uuid_shows_it(on_path):
    on_path("codex")
    result = run("codex", "--profile", "default", "--context-id", CTX.upper(), "--dry-run")
    assert result.exit_code == 0, result.output
    assert f"--guardrails {CTX}" in result.output


def test_unexpected_error_becomes_setup_failed(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr("kagura_memory.cli.run_setup_harness", boom)
    result = run("codex", "--profile", "default", "-y")
    assert result.exit_code == 1
    assert "Setup failed: kaboom" in result.output


def test_codex_hooks_message_names_codex_home(tmp_path, monkeypatch, on_path):
    codex_home = tmp_path / "ch"
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    data = codex_home / "plugins" / "data" / "kagura-memory-x"
    data.mkdir(parents=True)
    (data / "config.json").write_text("{}", encoding="utf-8")
    result = run("codex", "--profile", "default", "-y")
    assert result.exit_code == 0, result.output
    assert f"{codex_home / 'plugins' / 'data'}/kagura-memory-*/" in result.output


@pytest.mark.parametrize(
    ("body", "on"),
    [
        (b"{}", True),
        (b'{"mcp_server": null}', True),
        (b'{"mcp_server": "kagura-memory"}', True),
        (b'{"mcp_server": "other"}', False),
        (b'{"mcp_server": ""}', False),
        (b"[]", False),
        (b"not json", False),
        (b"\xff\xfe", False),
        (b'{"x": "' + b"a" * (64 * 1024) + b'"}', False),
    ],
)
def test_codex_hooks_follow_the_mcp_server_setting(env, body, on):
    data = env / ".codex" / "plugins" / "data" / "kagura-memory-x"
    data.mkdir(parents=True)
    (data / "config.json").write_bytes(body)
    assert setup_harness.codex_hooks_enabled("kagura-memory") is on


def test_codex_hooks_config_that_is_a_directory_is_ignored(env):
    (env / ".codex" / "plugins" / "data" / "kagura-memory-x" / "config.json").mkdir(parents=True)
    assert setup_harness.codex_hooks_enabled("kagura-memory") is False


def test_codex_url_form_block_without_codex():
    result = run("codex", "--url-form", "--mcp-url", MCP_URL, "-y")
    assert result.exit_code == 0, result.output
    assert f'url = "{MCP_URL}"' in result.output
    assert 'bearer_token_env_var = "KAGURA_API_KEY"' in result.output


def test_hermes_list_without_our_name_is_no_entry(on_path, recorder):
    on_path("hermes")
    recorder.hermes_unreadable.add("")
    recorder.detect_out["hermes"] = "  Name  Transport\n  github   npx @mcp/github   all\n"
    result = run("hermes", "--profile", "default", "-y")
    assert result.exit_code == 0, result.output
    assert "No kagura-memory entry yet." in result.output


def test_openclaw_show_with_broken_json_is_no_entry(on_path, recorder):
    on_path("openclaw")
    recorder.detect_out["openclaw"] = '{"command": '
    result = run("openclaw", "--profile", "default", "-y")
    assert result.exit_code == 0, result.output
    assert "No kagura-memory entry yet." in result.output


def test_lookups_use_path_and_stdin(monkeypatch):
    monkeypatch.setattr(setup_harness.shutil, "which", lambda cmd: f"/bin/{cmd}")
    assert REAL_HARNESS_EXECUTABLE("hermes") == "/bin/hermes"
    monkeypatch.setattr(setup_harness.sys.stdin, "isatty", lambda: False)
    assert REAL_STDIN_IS_TTY() is False


@pytest.mark.asyncio
async def test_fetch_digest_uses_the_given_credential(monkeypatch):
    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get_guardrail_digest(self, context_id):
            return GuardrailDigest(context_id=context_id, target="export", text=EXPORT_BLOCK)

    seen = []

    def from_resolved_auth(auth):
        seen.append(auth)
        return Client()

    monkeypatch.setattr(setup_harness.MemoryClient, "_from_resolved_auth", from_resolved_auth)
    auth = _resolve_profile_auth("work")
    digest = await REAL_FETCH_DIGEST(auth, CTX)
    assert digest.text == EXPORT_BLOCK
    assert seen == [auth]


def test_url_form_export_without_any_credential_writes_nothing(
    on_path, recorder, digest, tmp_path, monkeypatch
):
    from kagura_memory.exceptions import KaguraAuthError

    def no_credential(**kwargs):
        raise KaguraAuthError("No credentials found.")

    monkeypatch.setattr(setup_harness, "_resolve_auth", no_credential)
    on_path("codex")
    path = tmp_path / "AGENTS.md"
    args = ("--url-form", "--mcp-url", MCP_URL, "--context-id", CTX, "--agents-md", str(path))
    result = run("codex", *args, "-y")
    assert result.exit_code == 1
    assert "The AGENTS.md export has no credential: No credentials found." in result.output
    assert recorder.mutating() == []
    assert digest.calls == []


def test_dry_run_export_without_a_context_names_the_prompt(tty):
    result = run("openclaw", "--profile", "default", "--agents-md", "--dry-run")
    assert result.exit_code == 0, result.output
    assert "for context <chosen at the context prompt>" in result.output


def test_hermes_config_value_that_is_not_json_is_unread(monkeypatch):
    # An older Hermes prints the value as plain text, not one JSON line.
    monkeypatch.setattr(
        setup_harness,
        "_capture",
        lambda exe, args: subprocess.CompletedProcess([exe, *args], 0, "oauth\n", ""),
    )
    assert setup_harness._Hermes()._config_get("kagura-memory", "hermes") == (False, None)


def test_profiles_on_an_unusable_credentials_file_is_empty(monkeypatch):
    def unusable():
        raise ValueError("credentials.json is not valid JSON")

    monkeypatch.setattr(setup_harness, "load_credentials_file", unusable)
    assert setup_harness._profiles_on(MCP_URL) == []


def test_broken_config_names_any_other_load_failure(monkeypatch):
    def too_deep():
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(setup_harness, "load_config", too_deep)
    assert setup_harness._broken_config() == "~/.kagura.json (RecursionError)"
