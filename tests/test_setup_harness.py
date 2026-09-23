"""``kagura setup codex|hermes|openclaw`` (#260).

Every test runs against a temporary home: ``HOME``, ``CODEX_HOME``,
``HERMES_HOME`` and ``OPENCLAW_CONFIG_PATH`` point into ``tmp_path``, the
credentials file is a temporary one, no harness CLI is on the (patched)
``PATH`` unless a test puts it there, and ``subprocess.run`` is a recorder —
nothing reads or changes the developer's real harness configuration.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from kagura_memory import setup_harness
from kagura_memory.auth.credentials import CredentialsFile, save_credentials_file
from kagura_memory.cli import main
from kagura_memory.exceptions import KaguraNotFoundError
from kagura_memory.models import GuardrailDigest
from kagura_memory.setup_harness import hermes_context_file, hermes_home, hermes_key_env

from .conftest import make_oauth_creds

# Captured before the autouse fixtures replace them.
REAL_PROXY_PATH = setup_harness._proxy_path
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

    def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append((list(argv), kwargs))
        cli = Path(argv[0]).name
        sub = " ".join(argv[1:3])
        if sub == "mcp list" or (sub == "mcp show" and "--json" in argv):
            out = self.detect_out.get(cli)
            return subprocess.CompletedProcess(argv, 0 if out else 1, out or "", "")
        code = self.returncodes.get(sub, 0)
        return subprocess.CompletedProcess(argv, code, "", "boom" if code else "")

    def argvs(self) -> list[list[str]]:
        return [argv for argv, _ in self.calls]

    def mutating(self) -> list[list[str]]:
        """Every call that is not read-only detection."""
        return [
            a
            for a in self.argvs()
            if " ".join(a[1:3]) != "mcp list" and not (a[1:3] == ["mcp", "show"] and "--json" in a)
        ]


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch, isolated_kagura_credentials):
    """Temporary home and harness homes; real credentials file in tmp."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    for var in ("CODEX_HOME", "HERMES_HOME", "OPENCLAW_CONFIG_PATH"):
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
        calls: list[tuple[str | None, str]] = []

    async def fetch(profile, context_id):
        Fake.calls.append((profile, context_id))
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


def printed_json(output: str) -> Any:
    """The JSON block setup printed (the first ``{`` onwards)."""
    return json.JSONDecoder().raw_decode(output[output.index("{") :])[0]


def codex_config(home: Path) -> Path:
    path = home / ".codex" / "config.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def turn_on_codex_hooks(home: Path) -> None:
    data = home / ".codex" / "plugins" / "data" / "kagura-memory-kagura-memory-cloud"
    data.mkdir(parents=True)
    (data / "config.json").write_text(json.dumps({"context_id": CTX}), encoding="utf-8")


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
        assert recorder.mutating() == [
            ["/usr/bin/codex", "mcp", "remove", "kagura-memory"],
            ["/usr/bin/codex", "mcp", "add", "kagura-memory", "--", PROXY, "--profile", "default"],
        ]

    def test_failed_add_after_remove_says_the_entry_is_gone(self, env, on_path, recorder):
        on_path("codex")
        codex_config(env).write_text(
            '[mcp_servers.kagura-memory]\ncommand = "kagura-mcp"\n', encoding="utf-8"
        )
        recorder.returncodes["mcp add"] = 1
        result = run("codex", "--profile", "default", "--force", "-y")
        assert result.exit_code == 1
        assert "codex mcp add` failed: boom" in result.output
        assert "previous kagura-memory entry was removed" in result.output

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

    def test_stdio_with_hooks_on_asks_and_can_be_declined(self, env, on_path, recorder):
        on_path("codex")
        turn_on_codex_hooks(env)
        result = run("codex", "--profile", "default", input="n\n")
        assert result.exit_code == 1
        assert "Write the stdio entry anyway?" in result.output
        assert recorder.mutating() == []

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
        result = run("hermes", "--profile", "default", *flags, input="n\n")
        assert result.exit_code == 0, result.output
        assert recorder.mutating() == []
        assert "~/.hermes/config.yaml" in result.output
        assert "mcp_servers:\n      kagura-memory:" in result.output
        assert f'command: "{PROXY}"' in result.output
        assert 'args: ["--profile", "default"]' in result.output

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
        on_path("hermes")
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
        recorder.detect_out["hermes"] = "  kagura-memory    /x/kagura-mcp --profile   all\n"
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
        # Accept the export, then pick context 2 ("ops") from the list.
        result = run("hermes", "--profile", "default", input="y\n2\n")
        assert result.exit_code == 0, result.output
        assert digest.calls == [("default", OTHER_CTX)]

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

    def test_bad_name(self, harness):
        assert run(harness, "--profile", "default", "--name", "a.b", "-y").exit_code == 2

    def test_agents_md_with_y_needs_a_context(self, harness):
        result = run(harness, "--profile", "default", "--agents-md", "-y")
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
@pytest.mark.parametrize("url_form", [False, True], ids=["stdio", "url-form"])
def test_no_secret_in_output_argv_or_files(
    harness, url_form, env, on_path, recorder, monkeypatch, tty
):
    """The access token and an exported API key never reach output, argv or a file."""
    monkeypatch.setenv("KAGURA_API_KEY", API_KEY)
    on_path(harness)
    form = ["--url-form", "--mcp-url", MCP_URL] if url_form else []
    result = run(
        harness,
        "--profile",
        "default",
        *form,
        *("--context-id", CTX, "--agents-md", "-y"),
    )
    assert result.exit_code == 0, result.output
    for secret in (ACCESS_TOKEN, API_KEY):
        assert secret not in result.output
        assert all(secret not in " ".join(argv) for argv in recorder.argvs())
        for path in env.parent.rglob("*"):
            if path.is_file() and "credentials" not in path.name:
                assert secret not in path.read_text(encoding="utf-8", errors="replace")


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

    def test_empty_body_keeps_an_earlier_block_and_names_the_command(self, digest, tmp_path):
        digest.text = ""
        path = tmp_path / "AGENTS.md"
        path.write_text(EXPORT_BLOCK, encoding="utf-8")
        result = run(*self.args(path))
        assert result.exit_code == 0, result.output
        assert path.read_text(encoding="utf-8") == EXPORT_BLOCK
        assert "earlier block" in result.output

    def test_404_is_a_visibility_error_after_the_entry(self, digest, tmp_path):
        digest.error = KaguraNotFoundError("Context not found")
        path = tmp_path / "AGENTS.md"
        result = run(*self.args(path))
        assert result.exit_code == 1
        assert "not visible to this credential" in result.output
        assert "The MCP entry is set up" in result.output
        assert not path.exists()

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
    assert recorder.mutating() == []


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
