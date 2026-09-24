"""Tests for `kagura doctor`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from kagura_memory._auth import _OAuthAuth, _StaticAuth
from kagura_memory.auth.credentials import CredentialsFile, reset_state_cache, save_credentials_file
from kagura_memory.claude_code import McpEntry, claude_json_label
from kagura_memory.cli import main
from kagura_memory.doctor import DoctorCheck, DoctorReport
from kagura_memory.exceptions import KaguraAuthError, KaguraConnectionError
from kagura_memory.models import ServerInfo
from tests.conftest import make_oauth_creds


@pytest.fixture(autouse=True)
def _isolate_doctor_env(monkeypatch, tmp_path):
    reset_state_cache()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(
        "kagura_memory.auth.credentials.DEFAULT_CREDENTIALS_PATH",
        tmp_path / "credentials.json",
    )
    monkeypatch.delenv("KAGURA_API_KEY", raising=False)
    monkeypatch.delenv("KAGURA_PROFILE", raising=False)
    monkeypatch.delenv("KAGURA_MCP_URL", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    yield
    reset_state_cache()


_STDIO_ENTRY = {"type": "stdio", "command": "kagura-mcp", "args": ["--profile", "default"]}
_BEARER_ENTRY = {"type": "http", "url": "https://h/mcp", "headers": {"Authorization": "Bearer k"}}


def _patch_mcp_entry(monkeypatch, entry: dict) -> None:
    """Make ``entry`` the project-scope kagura-memory entry doctor finds."""
    monkeypatch.setattr(
        "kagura_memory.doctor.find_kagura_mcp_entries",
        lambda project: [McpEntry("project", ".mcp.json", entry, project.resolve() / ".mcp.json")],
    )


def _patch_common_doctor_surface(monkeypatch, *, resolved, creds_file):
    monkeypatch.setattr(
        "kagura_memory.doctor.load_config",
        lambda: {"api_key": "", "mcp_url": resolved.mcp_url},
    )
    monkeypatch.setattr("kagura_memory.doctor.load_credentials_file", lambda path=None: creds_file)
    monkeypatch.setattr("kagura_memory.doctor._resolve_auth", lambda **_: resolved)
    _patch_mcp_entry(monkeypatch, _STDIO_ENTRY)
    monkeypatch.setattr("kagura_memory.doctor._kagura_mcp_on_path", lambda: True)
    monkeypatch.setattr("kagura_memory.doctor.importlib_metadata.version", lambda _: "1.82.6")
    monkeypatch.setattr("kagura_memory.doctor.find_spec", lambda name: object())


def _patch_server(monkeypatch, checks: list[DoctorCheck] | None = None):
    async def _fake_server(_resolved, *, profile=None):
        return checks or [
            DoctorCheck(section="server", status="pass", message="Server reachable"),
            DoctorCheck(section="server", status="pass", message="Version: 0.25.0"),
        ]

    monkeypatch.setattr("kagura_memory.doctor._check_server", _fake_server)


def test_doctor_happy_path(monkeypatch):
    creds_file = CredentialsFile()
    resolved = _StaticAuth(
        api_key="kagura_12345678abcdef",
        mcp_url="https://example.com/mcp",
        source="env",
    )
    _patch_common_doctor_surface(monkeypatch, resolved=resolved, creds_file=creds_file)
    monkeypatch.setenv("KAGURA_API_KEY", "kagura_12345678abcdef")
    _patch_server(monkeypatch)

    from kagura_memory.doctor import run_doctor

    report = run_doctor()

    assert report.exit_code == 0
    assert any(check.message == "Effective Auth: KAGURA_API_KEY env" for check in report.checks)
    assert any(
        check.message == "MCP Mode: stdio (project scope, .mcp.json)" for check in report.checks
    )
    assert any(check.message == "kagura-mcp found on PATH" for check in report.checks)


def test_doctor_warns_on_shadowed_oauth_and_legacy_mcp(monkeypatch):
    creds_file = CredentialsFile()
    creds_file.set_profile("default", make_oauth_creds())
    resolved = _StaticAuth(
        api_key="kagura_12345678abcdef",
        mcp_url="https://example.com/mcp",
        source="env",
    )
    _patch_common_doctor_surface(monkeypatch, resolved=resolved, creds_file=creds_file)
    monkeypatch.setenv("KAGURA_API_KEY", "kagura_12345678abcdef")
    _patch_mcp_entry(monkeypatch, _BEARER_ENTRY)
    _patch_server(monkeypatch)

    from kagura_memory.doctor import run_doctor

    report = run_doctor()

    assert report.exit_code == 0
    assert any("shadowed by KAGURA_API_KEY" in check.message for check in report.checks)
    assert any(
        "Legacy static-token configuration detected" in check.message for check in report.checks
    )


def test_doctor_warns_on_near_expiry_oauth(monkeypatch):
    creds_file = CredentialsFile()
    creds_file.set_profile("default", make_oauth_creds(expires_in_seconds=120))
    resolved = _OAuthAuth(oauth=object(), mcp_url="https://example.com/mcp", workspace_id="ws-1")
    _patch_common_doctor_surface(monkeypatch, resolved=resolved, creds_file=creds_file)
    _patch_server(monkeypatch, checks=[])

    from kagura_memory.doctor import run_doctor

    report = run_doctor()

    assert report.exit_code == 0
    assert any("OAuth token nearing expiration" in check.message for check in report.checks)


def test_doctor_fails_when_kagura_mcp_missing(monkeypatch):
    creds_file = CredentialsFile()
    resolved = _StaticAuth(
        api_key="kagura_12345678abcdef",
        mcp_url="https://example.com/mcp",
        source="env",
    )
    _patch_common_doctor_surface(monkeypatch, resolved=resolved, creds_file=creds_file)
    monkeypatch.setattr("kagura_memory.doctor._kagura_mcp_on_path", lambda: False)
    _patch_server(monkeypatch)

    from kagura_memory.doctor import run_doctor

    report = run_doctor()

    assert report.exit_code == 1
    assert any(
        check.status == "fail" and "kagura-mcp not found" in check.message
        for check in report.checks
    )


def test_doctor_fails_on_old_server_version(monkeypatch):
    creds_file = CredentialsFile()
    resolved = _StaticAuth(
        api_key="kagura_12345678abcdef",
        mcp_url="https://example.com/mcp",
        source="env",
    )
    _patch_common_doctor_surface(monkeypatch, resolved=resolved, creds_file=creds_file)
    _patch_server(
        monkeypatch,
        checks=[
            DoctorCheck(section="server", status="pass", message="Server reachable"),
            DoctorCheck(
                section="server",
                status="fail",
                message="Version: 0.16.0 is below minimum 0.17.1",
            ),
        ],
    )

    from kagura_memory.doctor import run_doctor

    report = run_doctor()

    assert report.exit_code == 1
    assert any(
        check.status == "fail" and "below minimum" in check.message for check in report.checks
    )


def test_doctor_resolves_env_before_config(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".kagura.json").write_text(
        json.dumps(
            {
                "api_key": "kagura_config_should_not_win",
                "mcp_url": "https://config.example.com/mcp",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("KAGURA_API_KEY", "kagura_env_should_win")
    monkeypatch.setattr("kagura_memory.doctor.find_kagura_mcp_entries", lambda _: [])
    monkeypatch.setattr("kagura_memory.doctor.find_spec", lambda name: object())
    monkeypatch.setattr("kagura_memory.doctor.importlib_metadata.version", lambda _: "1.82.6")
    _patch_server(monkeypatch)

    from kagura_memory.doctor import run_doctor

    report = run_doctor(project_dir=tmp_path)

    assert any(check.message == "Effective Auth: KAGURA_API_KEY env" for check in report.checks)
    assert any(".kagura.json api_key is shadowed" in check.message for check in report.checks)
    assert not any("kagura_env_should_win" in check.message for check in report.checks)


def test_doctor_uses_profile_mcp_url_not_config_url(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".kagura.json").write_text(
        json.dumps({"mcp_url": "https://wrong.example.com/mcp"}),
        encoding="utf-8",
    )
    save_credentials_file(
        CredentialsFile(profiles={"dev": make_oauth_creds(server="https://profile.example.com")})
    )
    monkeypatch.setattr("kagura_memory.doctor.find_kagura_mcp_entries", lambda _: [])
    monkeypatch.setattr("kagura_memory.doctor.find_spec", lambda name: object())
    monkeypatch.setattr("kagura_memory.doctor.importlib_metadata.version", lambda _: "1.82.6")
    _patch_server(monkeypatch)

    from kagura_memory.doctor import run_doctor

    report = run_doctor(project_dir=tmp_path, profile="dev")

    assert any(
        check.details.get("mcp_url") == "https://profile.example.com/mcp" for check in report.checks
    )
    assert not any(
        check.details.get("mcp_url") == "https://wrong.example.com/mcp" for check in report.checks
    )


def test_doctor_uses_project_dir_for_config_key_shadow_warning(monkeypatch, tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / ".kagura.json").write_text(
        json.dumps({"api_key": "kagura_config", "mcp_url": "https://config.example.com/mcp"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "kagura_memory.doctor.load_config",
        lambda: {"api_key": "kagura_config", "mcp_url": "https://config.example.com/mcp"},
    )
    monkeypatch.setenv("KAGURA_API_KEY", "kagura_env")
    monkeypatch.setattr("kagura_memory.doctor.find_kagura_mcp_entries", lambda _: [])
    monkeypatch.setattr("kagura_memory.doctor.find_spec", lambda name: object())
    monkeypatch.setattr("kagura_memory.doctor.importlib_metadata.version", lambda _: "1.82.6")
    _patch_server(monkeypatch)

    from kagura_memory.doctor import run_doctor

    report = run_doctor(project_dir=project_dir)

    assert any(".kagura.json api_key is shadowed" in check.message for check in report.checks)


def test_doctor_auth_failure_still_reports_offline_checks(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("kagura_memory.doctor.find_kagura_mcp_entries", lambda _: [])
    monkeypatch.setattr("kagura_memory.doctor.find_spec", lambda name: None)
    monkeypatch.setattr("kagura_memory.doctor.importlib_metadata.version", lambda _: "1.82.6")

    from kagura_memory.doctor import run_doctor

    report = run_doctor(project_dir=tmp_path)

    assert report.exit_code == 1
    assert any("Authentication could not be resolved" in check.message for check in report.checks)
    assert any(check.section == "extras" for check in report.checks)
    assert any("Server connectivity check skipped" in check.message for check in report.checks)


def test_doctor_warns_when_config_key_shadows_oauth(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".kagura.json").write_text(
        json.dumps({"api_key": "kagura_config", "mcp_url": "https://config.example.com/mcp"}),
        encoding="utf-8",
    )
    save_credentials_file(CredentialsFile(profiles={"default": make_oauth_creds()}))
    monkeypatch.setattr("kagura_memory.doctor.find_kagura_mcp_entries", lambda _: [])
    monkeypatch.setattr("kagura_memory.doctor.find_spec", lambda name: object())
    monkeypatch.setattr("kagura_memory.doctor.importlib_metadata.version", lambda _: "1.82.6")
    _patch_server(monkeypatch)

    from kagura_memory.doctor import run_doctor

    report = run_doctor(project_dir=tmp_path)

    assert any(
        ".kagura.json api_key is shadowed by the OAuth profile" in c.message for c in report.checks
    )


def test_doctor_reports_expired_oauth_and_missing_refresh(monkeypatch):
    creds_file = CredentialsFile()
    expired = make_oauth_creds(expires_in_seconds=-10)
    expired.refresh_token = ""
    creds_file.set_profile("default", expired)
    resolved = _OAuthAuth(oauth=object(), mcp_url="https://example.com/mcp", workspace_id="ws-1")
    _patch_common_doctor_surface(monkeypatch, resolved=resolved, creds_file=creds_file)
    _patch_server(monkeypatch, checks=[])

    from kagura_memory.doctor import run_doctor

    report = run_doctor()

    assert report.exit_code == 1
    assert any("missing a refresh token" in check.message for check in report.checks)
    assert any("access token expired" in check.message for check in report.checks)


def test_doctor_warns_on_unusual_api_key_shape(monkeypatch):
    creds_file = CredentialsFile()
    resolved = _StaticAuth(
        api_key="not-a-kagura-key",
        mcp_url="https://example.com/mcp",
        source="env",
    )
    _patch_common_doctor_surface(monkeypatch, resolved=resolved, creds_file=creds_file)
    monkeypatch.setenv("KAGURA_API_KEY", "not-a-kagura-key")
    _patch_server(monkeypatch, checks=[])

    from kagura_memory.doctor import run_doctor

    report = run_doctor()

    assert report.exit_code == 0
    assert any("Unusual API key shape" in check.message for check in report.checks)
    assert not any("not-a-kagura-key" in check.message for check in report.checks)


def test_doctor_insecure_mcp_url_skips_server(monkeypatch):
    creds_file = CredentialsFile()
    resolved = _StaticAuth(
        api_key="kagura_12345678abcdef",
        mcp_url="http://example.com/mcp",
        source="env",
    )
    _patch_common_doctor_surface(monkeypatch, resolved=resolved, creds_file=creds_file)

    from kagura_memory.doctor import run_doctor

    report = run_doctor()

    assert report.exit_code == 0
    assert any(check.status == "warn" and "MCP URL" in check.message for check in report.checks)
    assert any("connectivity check skipped" in check.message for check in report.checks)


def test_doctor_litellm_missing(monkeypatch):
    from importlib.metadata import PackageNotFoundError

    from kagura_memory.doctor import _check_litellm

    monkeypatch.setattr(
        "kagura_memory.doctor.importlib_metadata.version",
        lambda _: (_ for _ in ()).throw(PackageNotFoundError),
    )
    assert _check_litellm().status == "info"


# Every PEP 440 spelling whose release starts 1.82.7 or 1.82.8 is blocked:
# pre-, post-, dev- and local versions too. The ``ingest`` extra's
# ``litellm>=1.50,<1.82.7`` pin excludes all of them as well.
_LITELLM_BLOCKED = [
    "1.82.7",
    "1.82.8",
    "1.82.7rc1",
    "1.82.7.post1",
    "1.82.7-1",  # PEP 440 implicit post-release: 1.82.7.post1
    "1.82.7.dev0",
    "1.82.7+local",
    "1.82.8a1",
    "1.82.8.0",
    "1.82.07",
    "1.82." + "0" * 40 + "7",  # leading zeros past the 32-digit cap
    "0" * 40 + "1.82.7",
    "v1.82.7",
    "V1.82.8",
    "0!1.82.7",  # the default epoch: the same release as 1.82.7
    "1!1.82.8",
    "v0!1.82.7",  # PEP 440 puts the "v" before the epoch
    "V1!1.82.8",
    "v0!1.82.7rc1",
    " 1.82.7 ",
]
_LITELLM_ALLOWED = [
    "1.82.0rc7",
    "1.82.6",
    "1.82.9",
    "1.50.0",
    "1.83.7",
    "1.8.27",
    "1.82",
    "v0!1.82.6",
    "1.82." + "0" * 40,
]


@pytest.mark.parametrize("version", _LITELLM_BLOCKED)
def test_doctor_litellm_blocks_compromised_releases(monkeypatch, version):
    from kagura_memory.doctor import _check_litellm

    monkeypatch.setattr("kagura_memory.doctor.importlib_metadata.version", lambda _: version)
    blocked = _check_litellm()
    assert blocked.status == "fail"
    assert blocked.message == f"LiteLLM {version} is blocked by this SDK"
    assert blocked.details == {"version": version}


@pytest.mark.parametrize("version", _LITELLM_ALLOWED)
def test_doctor_litellm_passes_other_releases(monkeypatch, version):
    from kagura_memory.doctor import _check_litellm

    monkeypatch.setattr("kagura_memory.doctor.importlib_metadata.version", lambda _: version)
    assert _check_litellm().status == "pass"


@pytest.mark.parametrize("version", _LITELLM_BLOCKED + _LITELLM_ALLOWED)
def test_doctor_litellm_blocklist_matches_pep_440_release(monkeypatch, version):
    # PEP 440 is the oracle: blocked exactly when the release segment starts
    # (1, 82, 7) or (1, 82, 8), however the version is spelled.
    pep440 = pytest.importorskip("packaging.version")
    from kagura_memory.doctor import _check_litellm

    release = (pep440.Version(version).release + (0, 0))[:3]
    monkeypatch.setattr("kagura_memory.doctor.importlib_metadata.version", lambda _: version)
    assert (_check_litellm().status == "fail") is (release in {(1, 82, 7), (1, 82, 8)})


def test_doctor_reports_provider_key_presence_with_redaction(monkeypatch):
    from kagura_memory.doctor import _check_provider_keys

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-1234567890")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")

    checks = _check_provider_keys()

    openai = next(check for check in checks if check.details["env"] == "OPENAI_API_KEY")
    anthropic = next(check for check in checks if check.details["env"] == "ANTHROPIC_API_KEY")

    assert openai.status == "info"
    assert openai.details["set"] is True
    assert openai.details["preview"] == "sk-test-...7890"
    assert "sk-test-1234567890" not in openai.message
    assert anthropic.details["set"] is False
    assert anthropic.details["preview"] is None


def test_doctor_warns_when_ingest_text_key_missing(monkeypatch):
    from kagura_memory.doctor import _check_model_key_alignment

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    checks = _check_model_key_alignment()

    assert any(
        check.status == "warn"
        and check.details["feature"] == "ingest-text"
        and check.details["env"] == "ANTHROPIC_API_KEY"
        for check in checks
    )


def test_doctor_env_key_passes_for_ingest_text_model(monkeypatch):
    from kagura_memory.doctor import _check_model_key_alignment

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    checks = _check_model_key_alignment()

    assert any(
        check.status == "pass"
        and check.details["feature"] == "ingest-text"
        and check.details["credential_source"] == "ANTHROPIC_API_KEY"
        for check in checks
    )


def test_doctor_warns_for_audio_gemini_key_missing(monkeypatch):
    from kagura_memory.doctor import _check_model_key_alignment

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    checks = _check_model_key_alignment()

    assert any(
        check.status == "warn"
        and check.details["feature"] == "ingest-audio"
        and check.details["env"] == "GEMINI_API_KEY"
        for check in checks
    )


def test_doctor_ollama_model_does_not_require_key(monkeypatch):
    from kagura_memory.doctor import _check_model_key_alignment

    monkeypatch.setattr("kagura_memory.doctor._DEFAULT_INGEST_TEXT_MODEL", "ollama/qwen3:30b")

    checks = _check_model_key_alignment()

    assert any(
        check.status == "info"
        and check.details["feature"] == "ingest-text"
        and "no API key is required" in check.message
        for check in checks
    )


def test_doctor_unknown_model_provider_is_informational(monkeypatch):
    from kagura_memory.doctor import _check_model_key_alignment

    monkeypatch.setattr("kagura_memory.doctor._DEFAULT_INGEST_TEXT_MODEL", "custom/model")

    checks = _check_model_key_alignment()

    assert any(
        check.status == "info"
        and check.details["feature"] == "ingest-text"
        and check.details["provider"] is None
        for check in checks
    )


def test_doctor_model_provider_handles_empty_model():
    from kagura_memory.doctor import _provider_for_model

    assert _provider_for_model(" ") is None


def test_doctor_model_alignment_skips_empty_model(monkeypatch):
    from kagura_memory.doctor import _check_model_key_alignment

    monkeypatch.setattr("kagura_memory.doctor._DEFAULT_INGEST_VISION_MODEL", None)

    checks = _check_model_key_alignment()

    assert not any(check.details["feature"] == "ingest-vision" for check in checks)


def test_doctor_llm_warnings_do_not_fail_exit(monkeypatch):
    from kagura_memory.doctor import _check_llm_providers

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    checks = _check_llm_providers()
    report = DoctorReport(checks=checks)

    assert any(check.status == "warn" for check in checks)
    assert report.exit_code == 0


def test_run_doctor_includes_llm_section(monkeypatch):
    creds_file = CredentialsFile()
    resolved = _StaticAuth(
        api_key="kagura_12345678abcdef",
        mcp_url="https://example.com/mcp",
        source="env",
    )
    _patch_common_doctor_surface(monkeypatch, resolved=resolved, creds_file=creds_file)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-1234567890")
    _patch_server(monkeypatch, checks=[])

    from kagura_memory.doctor import run_doctor

    report = run_doctor()

    assert report.section_statuses["llm"] == "warn"
    assert any(
        check.section == "llm" and check.details.get("env") == "OPENAI_API_KEY"
        for check in report.checks
    )


def test_doctor_reports_missing_oauth_profile():
    from kagura_memory.doctor import _check_oauth_profile

    checks = _check_oauth_profile(CredentialsFile(), "missing")

    assert checks[0].status == "fail"
    assert "OAuth profile not found" in checks[0].message


def test_doctor_optional_dependencies_reports_missing(monkeypatch):
    monkeypatch.setattr("kagura_memory.doctor.find_spec", lambda name: None)

    from kagura_memory.doctor import _check_optional_dependencies

    check = _check_optional_dependencies()[0]

    assert check.status == "info"
    assert check.details["available"] == []
    assert "ingest-pdf" in check.details["missing"]
    assert "ingest-audio" not in check.details["missing"]


def test_doctor_mcp_modes(tmp_path):
    from kagura_memory.doctor import _check_mcp

    assert any(
        check.message == f"No kagura-memory MCP entry found (.mcp.json, {claude_json_label()})"
        for check in _check_mcp(tmp_path)
    )

    (tmp_path / ".mcp.json").write_text(json.dumps({"mcpServers": {"github": {}}}))
    assert any(
        check.message == "No usable kagura-memory entry found in .mcp.json"
        for check in _check_mcp(tmp_path)
    )

    _write_project_entry(tmp_path, {"type": "http", "url": "https://h/mcp"})
    assert any(
        check.message == "MCP Mode: url (project scope, .mcp.json)"
        for check in _check_mcp(tmp_path)
    )


# ---------------------------------------------------------------------------
# MCP entry scope (#258): the entry Claude Code uses, from every scope
# ---------------------------------------------------------------------------


def _write_claude_json(data: dict) -> None:
    """Write the isolated ~/.claude.json (conftest points CLAUDE_CONFIG_DIR at a temp dir)."""
    from kagura_memory.claude_code import claude_json_path

    claude_json_path().write_text(json.dumps(data), encoding="utf-8")


def _write_project_entry(project: Path, entry: dict) -> None:
    (project / ".mcp.json").write_text(json.dumps({"mcpServers": {"kagura-memory": entry}}))


def test_doctor_reports_user_scope_stdio_entry(monkeypatch, tmp_path):
    """A user-scope entry is the effective one, not "No .mcp.json found"."""
    from kagura_memory.doctor import _check_mcp

    monkeypatch.setattr("kagura_memory.doctor._kagura_mcp_on_path", lambda: True)
    _write_claude_json({"mcpServers": {"kagura-memory": _STDIO_ENTRY}})

    checks = _check_mcp(tmp_path)
    messages = [c.message for c in checks]

    label = claude_json_label()
    assert f"MCP Mode: stdio (user scope, {label})" in messages
    assert "kagura-mcp found on PATH" in messages
    assert not any("No kagura-memory MCP entry" in m or "No .mcp.json" in m for m in messages)
    assert checks[0].details == {"scope": "user", "source": label}


def test_doctor_warns_about_a_shadowed_entry(tmp_path):
    from kagura_memory.doctor import _check_mcp

    _write_project_entry(tmp_path, _STDIO_ENTRY)
    _write_claude_json({"mcpServers": {"kagura-memory": _STDIO_ENTRY}})

    checks = _check_mcp(tmp_path)

    assert checks[0].message == "MCP Mode: stdio (project scope, .mcp.json)"
    shadow = [c for c in checks if "also defined in user scope" in c.message]
    assert len(shadow) == 1
    assert shadow[0].status == "warn"
    assert "project-scope entry" in shadow[0].message


def test_doctor_reports_a_parent_mcp_json_entry_from_a_subdirectory(monkeypatch, tmp_path):
    """Claude Code takes the closest ``.mcp.json`` up the tree; it hides the user entry."""
    from kagura_memory.doctor import _check_mcp

    monkeypatch.setattr("kagura_memory.doctor._kagura_mcp_on_path", lambda: True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path.resolve())
    (tmp_path / "repo" / ".git").mkdir(parents=True)
    sub = tmp_path / "repo" / "sub"
    sub.mkdir()
    _write_project_entry(tmp_path / "repo", _STDIO_ENTRY)
    _write_claude_json({"mcpServers": {"kagura-memory": _BEARER_ENTRY}})

    checks = _check_mcp(sub)

    assert checks[0].status == "pass"
    assert checks[0].message == "MCP Mode: stdio (project scope, ~/repo/.mcp.json)"
    assert any("also defined in user scope" in c.message for c in checks)


def test_doctor_legacy_hint_in_a_parent_mcp_json_names_its_directory(tmp_path):
    """A plain re-run in the subdirectory would write a closer file, not fix this one."""
    from kagura_memory.doctor import _check_mcp

    repo = tmp_path / "my repo"
    (repo / "sub").mkdir(parents=True)
    _write_project_entry(repo, {**_BEARER_ENTRY, "type": "url"})

    [hint] = [c.message for c in _check_mcp(repo / "sub") if 'type "url"' in c.message]

    assert f"re-run `kagura setup claude --project-dir '{repo.resolve()}'`" in hint


@pytest.mark.parametrize("env_set", [False, True], ids=["unset", "set"])
def test_doctor_warns_when_the_entrys_key_variable_is_unset(monkeypatch, tmp_path, env_set):
    """Setup's user-scope API-key entry sends ${KAGURA_MCP_API_KEY}; unset, it sends nothing."""
    from kagura_memory.doctor import _check_mcp

    if env_set:
        monkeypatch.setenv("KAGURA_MCP_API_KEY", "kagura_secret_value")
    else:
        monkeypatch.delenv("KAGURA_MCP_API_KEY", raising=False)
    entry = {**_BEARER_ENTRY, "headers": {"Authorization": "Bearer ${KAGURA_MCP_API_KEY}"}}
    _write_claude_json({"mcpServers": {"kagura-memory": entry}})

    checks = _check_mcp(tmp_path)

    unset = [c for c in checks if "KAGURA_MCP_API_KEY is not set here" in c.message]
    assert len(unset) == (0 if env_set else 1)
    if unset:
        assert unset[0].status == "warn"
        assert unset[0].details == {
            "scope": "user",
            "source": claude_json_label(),
            "env": ("KAGURA_MCP_API_KEY"),
        }
    assert not any("kagura_secret_value" in c.message for c in checks)


def test_doctor_names_the_scope_of_an_unusable_entry(tmp_path):
    from kagura_memory.doctor import _check_mcp

    _write_claude_json({"mcpServers": {"kagura-memory": {"type": "sse", "url": "https://h/sse"}}})

    checks = _check_mcp(tmp_path)

    assert checks[0].status == "warn"
    assert checks[0].message == (
        f"No usable kagura-memory entry found in {claude_json_label()} (user scope)"
    )


def test_doctor_labels_claude_json_where_claude_config_dir_points(monkeypatch, tmp_path):
    """The message names the file really read, not a fixed ``~/.claude.json``."""
    from kagura_memory.doctor import _check_mcp

    config_dir = tmp_path / "elsewhere"
    config_dir.mkdir()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    _write_claude_json({"mcpServers": {"kagura-memory": _STDIO_ENTRY}})

    [first, *_] = _check_mcp(tmp_path)

    assert first.details == {"scope": "user", "source": str(config_dir / ".claude.json")}


@pytest.mark.parametrize("entry_type", ["http", "url"])
def test_doctor_static_token_legacy_type_suggests_rerunning_setup(tmp_path, entry_type):
    """Both types read as static-token; only the legacy "url" one gets the re-run hint."""
    from kagura_memory.doctor import _check_mcp

    _write_project_entry(tmp_path, {**_BEARER_ENTRY, "type": entry_type})

    messages = [c.message for c in _check_mcp(tmp_path)]

    assert any("Legacy static-token configuration detected" in m for m in messages)
    legacy = [m for m in messages if 'type "url"' in m]
    assert bool(legacy) == (entry_type == "url")
    if legacy:
        assert "re-run `kagura setup claude` to write it" in legacy[0]


@pytest.mark.parametrize(
    ("scope", "fix"),
    [
        ("user", "re-run `kagura setup claude --scope user`"),
        ("local", "remove it (`claude mcp remove --scope local kagura-memory`), then re-run"),
    ],
)
def test_doctor_legacy_type_hint_names_the_scope(tmp_path, scope, fix):
    """A plain re-run writes project scope: it would not fix a user or local entry."""
    from kagura_memory.doctor import _check_mcp

    legacy = {**_BEARER_ENTRY, "type": "url"}
    if scope == "user":
        _write_claude_json({"mcpServers": {"kagura-memory": legacy}})
    else:
        key = str(tmp_path.resolve())
        _write_claude_json({"projects": {key: {"mcpServers": {"kagura-memory": legacy}}}})

    [hint] = [c.message for c in _check_mcp(tmp_path) if 'type "url"' in c.message]

    assert fix in hint


# (server version, verdict of the doctor's version check) with
# MIN_SERVER_VERSION = 0.17.1. The agreement test below reuses it.
_SERVER_VERSION_VERDICTS = [
    ("unknown", "info"),
    ("main-abc123", "info"),
    ("0.17", "info"),
    ("0.16.0", "fail"),
    ("v0.16.0", "fail"),
    ("0.16.9-beta", "fail"),
    ("0.17.0-rc1", "fail"),
    ("0.17.1-rc1", "fail"),
    ("0.17.1", "pass"),
    ("0.17.2-rc1", "pass"),
    ("0.25.0", "pass"),
]


@pytest.mark.parametrize(("version", "expected_status"), _SERVER_VERSION_VERDICTS)
def test_check_server_version_branches(monkeypatch, version, expected_status):
    from kagura_memory.doctor import _check_server

    seen = {}

    class FakeClient:
        def __init__(self, api_key=None, mcp_url=None, profile=None):
            seen.update({"api_key": api_key, "mcp_url": mcp_url, "profile": profile})

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def check_server_version(self):
            return ServerInfo(name="memory-cloud", version=version)

    monkeypatch.setattr("kagura_memory.doctor.KaguraClient", FakeClient)
    resolved = _OAuthAuth(
        oauth=object(),
        mcp_url="https://profile.example.com/mcp",
        workspace_id="ws-1",
    )

    import asyncio

    checks = asyncio.run(_check_server(resolved, profile="dev"))

    assert seen["profile"] == "dev"
    assert seen["mcp_url"] == "https://profile.example.com/mcp"
    assert [check.status for check in checks] == ["pass", expected_status]
    assert checks[-1].details["version"] == version
    assert ("minimum" in checks[-1].details) is (expected_status == "fail")


@pytest.mark.parametrize(("version", "expected_status"), _SERVER_VERSION_VERDICTS)
def test_check_server_agrees_with_check_server_version(
    monkeypatch, caplog, version, expected_status
):
    """One string, one verdict: the doctor fails exactly when the client warns."""
    import asyncio
    import logging

    from kagura_memory.client import KaguraClient
    from kagura_memory.doctor import _check_server

    async def fake_server_info(self):
        return ServerInfo(name="memory-cloud", version=version)

    monkeypatch.setattr(KaguraClient, "get_server_info", fake_server_info)
    resolved = _StaticAuth(
        api_key="kagura_test_key", mcp_url="https://api.example.com/mcp", source="env"
    )

    with caplog.at_level(logging.WARNING, logger="kagura_memory"):
        checks = asyncio.run(_check_server(resolved))

    warned = any("tested minimum" in record.getMessage() for record in caplog.records)
    assert checks[-1].status == expected_status
    assert warned is (expected_status == "fail")


@pytest.mark.parametrize(
    ("exc", "message"),
    [
        (KaguraAuthError("bad token"), "bad token"),
        (KaguraConnectionError("offline"), "Server unreachable"),
    ],
)
def test_check_server_failure_branches(monkeypatch, exc, message):
    from kagura_memory.doctor import _check_server

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def check_server_version(self):
            raise exc

    monkeypatch.setattr("kagura_memory.doctor.KaguraClient", FakeClient)
    resolved = _StaticAuth(
        api_key="kagura_12345678abcdef",
        mcp_url="https://example.com/mcp",
        source="env",
    )

    import asyncio

    checks = asyncio.run(_check_server(resolved))

    assert checks[0].status == "fail"
    assert message in checks[0].message


def test_check_server_oauth_auth_error_is_informational(monkeypatch):
    from kagura_memory.doctor import _check_server

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def check_server_version(self):
            raise KaguraAuthError("REST rejected OAuth bearer")

    monkeypatch.setattr("kagura_memory.doctor.KaguraClient", FakeClient)
    resolved = _OAuthAuth(
        oauth=object(),
        mcp_url="https://profile.example.com/mcp",
        workspace_id="ws-1",
    )

    import asyncio

    checks = asyncio.run(_check_server(resolved, profile="dev"))

    assert checks[0].status == "info"
    assert "REST validates API keys" in checks[0].message


def test_check_server_constructor_failure(monkeypatch):
    from kagura_memory.doctor import _check_server

    def bad_client(**kwargs):
        raise ValueError("bad url")

    monkeypatch.setattr("kagura_memory.doctor.KaguraClient", bad_client)
    resolved = _StaticAuth(
        api_key="kagura_12345678abcdef",
        mcp_url="https://example.com/mcp",
        source="env",
    )

    import asyncio

    checks = asyncio.run(_check_server(resolved))

    assert checks[0].status == "fail"
    assert "bad url" in checks[0].message


def test_doctor_cli_json_output(monkeypatch):
    report = DoctorReport(
        checks=[
            DoctorCheck(section="auth", status="pass", message="Effective Auth: OAuth profile"),
            DoctorCheck(section="server", status="pass", message="Server reachable"),
        ]
    )
    monkeypatch.setattr("kagura_memory.cli.run_doctor", lambda profile=None: report)

    runner = CliRunner()
    result = runner.invoke(main, ["doctor", "--json"])

    assert result.exit_code == 0
    assert '"auth": "pass"' in result.output
    assert '"server": "pass"' in result.output
    assert '"sections": {' in result.output


def test_doctor_cli_passes_profile(monkeypatch):
    seen = {}
    report = DoctorReport(
        checks=[DoctorCheck(section="auth", status="pass", message="Effective Auth: OAuth profile")]
    )

    def fake_run_doctor(*, profile=None):
        seen["profile"] = profile
        return report

    monkeypatch.setattr("kagura_memory.cli.run_doctor", fake_run_doctor)

    runner = CliRunner()
    result = runner.invoke(main, ["doctor", "--profile", "dev"])

    assert result.exit_code == 0
    assert seen["profile"] == "dev"
