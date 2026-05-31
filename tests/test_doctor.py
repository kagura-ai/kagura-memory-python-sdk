"""Tests for `kagura doctor`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from kagura_memory._auth import _OAuthAuth, _StaticAuth
from kagura_memory.auth.credentials import CredentialsFile, reset_state_cache, save_credentials_file
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
    yield
    reset_state_cache()


def _patch_common_doctor_surface(monkeypatch, *, resolved, creds_file):
    monkeypatch.setattr(
        "kagura_memory.doctor.load_config",
        lambda: {"api_key": "", "mcp_url": resolved.mcp_url},
    )
    monkeypatch.setattr("kagura_memory.doctor.load_credentials_file", lambda path=None: creds_file)
    monkeypatch.setattr("kagura_memory.doctor._resolve_auth", lambda **_: resolved)
    monkeypatch.setattr("kagura_memory.doctor.detect_mcp_json_mode", lambda _: "stdio")
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
    assert any(check.message == "MCP Mode: stdio" for check in report.checks)
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
    monkeypatch.setattr("kagura_memory.doctor.detect_mcp_json_mode", lambda _: "static-token")
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
    monkeypatch.setattr("kagura_memory.doctor.detect_mcp_json_mode", lambda _: "none")
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
    monkeypatch.setattr("kagura_memory.doctor.detect_mcp_json_mode", lambda _: "none")
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
    monkeypatch.setattr("kagura_memory.doctor.detect_mcp_json_mode", lambda _: "none")
    monkeypatch.setattr("kagura_memory.doctor.find_spec", lambda name: object())
    monkeypatch.setattr("kagura_memory.doctor.importlib_metadata.version", lambda _: "1.82.6")
    _patch_server(monkeypatch)

    from kagura_memory.doctor import run_doctor

    report = run_doctor(project_dir=project_dir)

    assert any(".kagura.json api_key is shadowed" in check.message for check in report.checks)


def test_doctor_auth_failure_still_reports_offline_checks(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("kagura_memory.doctor.detect_mcp_json_mode", lambda _: "none")
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
    monkeypatch.setattr("kagura_memory.doctor.detect_mcp_json_mode", lambda _: "none")
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


def test_doctor_litellm_missing_and_blocked(monkeypatch):
    from importlib.metadata import PackageNotFoundError

    from kagura_memory.doctor import _check_litellm

    monkeypatch.setattr(
        "kagura_memory.doctor.importlib_metadata.version",
        lambda _: (_ for _ in ()).throw(PackageNotFoundError),
    )
    assert _check_litellm().status == "info"

    monkeypatch.setattr("kagura_memory.doctor.importlib_metadata.version", lambda _: "1.82.7")
    blocked = _check_litellm()
    assert blocked.status == "fail"
    assert "blocked" in blocked.message

    monkeypatch.setattr("kagura_memory.doctor.importlib_metadata.version", lambda _: "1.82.9")
    assert _check_litellm().status == "pass"


def test_doctor_version_parser_ignores_extra_components():
    from kagura_memory.doctor import _parse_version_prefix

    assert _parse_version_prefix("1.2.3.4") == (1, 2, 3)


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


def test_doctor_mcp_modes(monkeypatch, tmp_path):
    from kagura_memory.doctor import _check_mcp

    monkeypatch.setattr("kagura_memory.doctor.detect_mcp_json_mode", lambda _: "url")
    assert any(check.message == "MCP Mode: url" for check in _check_mcp(tmp_path))

    monkeypatch.setattr("kagura_memory.doctor.detect_mcp_json_mode", lambda _: "absent")
    assert any("No usable" in check.message for check in _check_mcp(tmp_path))

    monkeypatch.setattr("kagura_memory.doctor.detect_mcp_json_mode", lambda _: "none")
    assert any(check.message == "No .mcp.json found" for check in _check_mcp(tmp_path))


@pytest.mark.parametrize(
    ("version", "expected_status"),
    [
        ("unknown", "info"),
        ("0.16.0", "fail"),
        ("0.25.0", "pass"),
    ],
)
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
    assert any(check.status == expected_status for check in checks)


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
