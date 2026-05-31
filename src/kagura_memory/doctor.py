"""Diagnostic checks for `kagura doctor`."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from datetime import UTC
from importlib import metadata as importlib_metadata
from importlib.metadata import PackageNotFoundError
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Literal

from ._auth import _SOURCE_LABEL, _OAuthAuth, _resolve_auth, _StaticAuth
from ._http import validate_https_url
from .auth.cli import _redact_token
from .auth.credentials import REFRESH_SKEW_SEC, load_credentials_file
from .client import MIN_SERVER_VERSION, KaguraClient
from .config import load_config
from .exceptions import KaguraAuthError, KaguraConnectionError, _exc_message
from .setup_claude import _kagura_mcp_on_path, detect_mcp_json_mode

DoctorStatus = Literal["pass", "warn", "fail", "info"]

_STATUS_ORDER: dict[DoctorStatus, int] = {"fail": 3, "warn": 2, "pass": 1, "info": 0}
_OPTIONAL_INGESTION_DEPENDENCIES: dict[str, str] = {
    "ingest": "pillow",
    "ingest-pdf": "pymupdf",
    "ingest-epub": "fitz",
    "ingest-html": "bs4",
    "ingest-docx": "docx",
    "ingest-xlsx": "openpyxl",
    "ingest-pptx": "pptx",
    "ingest-youtube": "youtube_transcript_api",
    "ingest-browser": "playwright",
}

# Environment variable that litellm reads for each LLM provider the SDK can
# route to. Used only by the optional ingest / agent features; a memory- or
# recall-only user needs none of these.
_PROVIDER_ENV_KEYS: dict[str, str] = {
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "ollama": "OLLAMA_API_KEY",
}

# Default models for the optional LLM-backed features, mirrored from their
# sources of truth: agent.py / config.py (`gpt-5.4-nano`) and
# ingest/_audio.py (`gemini/gemini-2.5-flash`). doctor stays import-light and
# deliberately does NOT import the ingest package (which pulls optional
# extras), so these are mirrored here rather than imported.
_DEFAULT_AGENT_MODEL = "gpt-5.4-nano"
_DEFAULT_AUDIO_MODEL = "gemini/gemini-2.5-flash"


@dataclass
class DoctorCheck:
    section: str
    status: DoctorStatus
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DoctorReport:
    checks: list[DoctorCheck]

    @property
    def section_statuses(self) -> dict[str, DoctorStatus]:
        statuses: dict[str, DoctorStatus] = {}
        for check in self.checks:
            current = statuses.get(check.section)
            if current is None or _STATUS_ORDER[check.status] > _STATUS_ORDER[current]:
                statuses[check.section] = check.status
        return statuses

    @property
    def exit_code(self) -> int:
        return 1 if any(check.status == "fail" for check in self.checks) else 0

    def to_dict(self) -> dict[str, Any]:
        output = {
            "sections": self.section_statuses,
            "checks": [
                {
                    "section": check.section,
                    "status": check.status,
                    "message": check.message,
                    "details": check.details,
                }
                for check in self.checks
            ],
            "exit_code": self.exit_code,
        }
        output.update(self.section_statuses)
        return output


def _looks_like_kagura_key(value: str) -> bool:
    return value.startswith("kagura_") and len(value) >= 10


def _parse_version_prefix(version: str) -> tuple[int, int, int] | None:
    parts: list[int] = []
    for piece in version.split("."):
        if len(parts) == 3:
            break
        digits = "".join(ch for ch in piece if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    if len(parts) != 3:
        return None
    return (parts[0], parts[1], parts[2])


def _check_optional_dependencies() -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    available: list[str] = []
    missing: list[str] = []
    for extra, module_name in _OPTIONAL_INGESTION_DEPENDENCIES.items():
        if find_spec(module_name) is not None:
            available.append(extra)
        else:
            missing.append(extra)
    checks.append(
        DoctorCheck(
            section="extras",
            status="info",
            message=(
                "Optional ingestion dependencies available: "
                + (", ".join(available) if available else "none")
            ),
            details={"available": available, "missing": missing},
        )
    )
    return checks


def _check_litellm() -> DoctorCheck:
    try:
        version = importlib_metadata.version("litellm")
    except PackageNotFoundError:
        return DoctorCheck(
            section="security",
            status="info",
            message="LiteLLM not installed",
        )

    parsed = _parse_version_prefix(version)
    if parsed in {(1, 82, 7), (1, 82, 8)}:
        return DoctorCheck(
            section="security",
            status="fail",
            message=f"LiteLLM {version} is blocked by this SDK",
            details={"version": version},
        )

    return DoctorCheck(
        section="security",
        status="pass",
        message=f"LiteLLM version: {version}",
        details={"version": version},
    )


def _provider_for_model(model: str) -> str | None:
    """Map a litellm model string to a known provider name.

    Returns one of the keys in :data:`_PROVIDER_ENV_KEYS`, or ``None`` when the
    provider cannot be determined from the model string (key check is skipped).
    """

    name = model.strip().lower()
    if name.startswith(("ollama/", "ollama_chat/")):
        return "ollama"
    if name.startswith("gemini/"):
        return "gemini"
    if name.startswith(("anthropic/", "claude")):
        return "anthropic"
    if name.startswith(("openai/", "gpt", "o1", "o3", "o4")):
        return "openai"
    return None


def _check_provider_keys() -> DoctorCheck:
    """Report which optional LLM-provider keys are present (informational).

    Only the env-var names are reported — values are never read into the
    message, so there is nothing to leak. Absence is never a failure: these
    keys are opt-in per feature.
    """

    present = [env for env in _PROVIDER_ENV_KEYS.values() if os.getenv(env)]
    absent = [env for env in _PROVIDER_ENV_KEYS.values() if env not in present]
    return DoctorCheck(
        section="providers",
        status="info",
        message="LLM provider keys set: " + (", ".join(present) if present else "none"),
        details={"present": present, "absent": absent},
    )


def _check_model_provider_match(config: dict[str, Any]) -> list[DoctorCheck]:
    """Warn when a configured default model targets a provider with no key.

    Informational only — no network calls are made to validate a key. Ollama
    needs no key, so it never warns.
    """

    configured: list[tuple[str, str]] = [
        ("agent", config.get("model") or _DEFAULT_AGENT_MODEL),
        ("audio ingest", _DEFAULT_AUDIO_MODEL),
    ]

    checks: list[DoctorCheck] = []
    for feature, model in configured:
        details: dict[str, Any] = {"feature": feature, "model": model}
        provider = _provider_for_model(model)
        if provider is None:
            status: DoctorStatus = "info"
            message = f"{feature} model {model}: provider not recognized; key check skipped"
        else:
            env_var = _PROVIDER_ENV_KEYS[provider]
            details["provider"] = provider
            if provider == "ollama":
                status = "info"
                message = f"{feature} model {model} targets Ollama (no API key required)"
            else:
                details["env"] = env_var
                if os.getenv(env_var):
                    status = "pass"
                    message = f"{feature} model {model}: {env_var} is set"
                else:
                    status = "warn"
                    message = f"{feature} model {model} targets {provider} but {env_var} is not set"
        checks.append(
            DoctorCheck(section="providers", status=status, message=message, details=details)
        )

    return checks


def _check_auth(
    config: dict[str, Any],
    *,
    profile: str | None = None,
    project_dir: Path | None = None,
    creds_file_path: Path | None = None,
) -> tuple[list[DoctorCheck], _StaticAuth | _OAuthAuth | None]:
    checks: list[DoctorCheck] = []
    creds_file = load_credentials_file(creds_file_path)
    env_key = os.getenv("KAGURA_API_KEY") or ""
    target_profile = profile or os.getenv("KAGURA_PROFILE") or None
    oauth_creds = creds_file.get_profile(target_profile)
    config_key = _configured_api_key(config, project_dir=project_dir)

    try:
        resolved = _resolve_auth(
            api_key=None,
            mcp_url=None,
            profile=profile,
            config=config,
        )
    except KaguraAuthError as exc:
        checks.append(
            DoctorCheck(
                section="auth",
                status="fail",
                message=f"Authentication could not be resolved: {exc}",
            )
        )
        checks.extend(_check_api_key_presence(config_key=config_key, env_key=env_key))
        return checks, None

    if isinstance(resolved, _OAuthAuth):
        effective_source = "OAuth profile"
        checks.append(
            DoctorCheck(
                section="auth",
                status="pass",
                message=f"Effective Auth: {effective_source}",
                details={"source": "oauth"},
            )
        )
        checks.extend(_check_oauth_profile(creds_file, target_profile))
    else:
        source_label = _SOURCE_LABEL[resolved.source]
        checks.append(
            DoctorCheck(
                section="auth",
                status="pass",
                message=f"Effective Auth: {source_label}",
                details={"source": resolved.source},
            )
        )
        checks.append(_check_api_key_shape(resolved.api_key, resolved.source))

    checks.extend(_check_api_key_presence(config_key=config_key, env_key=env_key))

    if env_key and oauth_creds is not None and resolved.__class__ is _StaticAuth:
        checks.append(
            DoctorCheck(
                section="auth",
                status="warn",
                message=(
                    "OAuth profile is shadowed by KAGURA_API_KEY; auto-refresh will not be used"
                ),
            )
        )

    if (
        env_key
        and config_key
        and resolved.__class__ is _StaticAuth
        and getattr(resolved, "source") == "env"
    ):
        checks.append(
            DoctorCheck(
                section="auth",
                status="warn",
                message=".kagura.json api_key is shadowed by KAGURA_API_KEY",
            )
        )
    elif oauth_creds is not None and config_key and isinstance(resolved, _OAuthAuth):
        checks.append(
            DoctorCheck(
                section="auth",
                status="warn",
                message=".kagura.json api_key is shadowed by the OAuth profile",
            )
        )

    return checks, resolved


def _configured_api_key(config: dict[str, Any], *, project_dir: Path | None = None) -> str:
    """Return a file-backed api_key candidate, excluding env fallback config."""

    local_config = (project_dir or Path.cwd()) / ".kagura.json"
    if local_config.exists() or (Path.home() / ".kagura.json").exists():
        return config.get("api_key") or ""
    return ""


def _check_api_key_presence(*, config_key: str, env_key: str) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    checks.append(
        DoctorCheck(
            section="auth",
            status="pass" if env_key else "info",
            message="KAGURA_API_KEY env is set" if env_key else "KAGURA_API_KEY env is not set",
            details={"source": "env", "set": bool(env_key)},
        )
    )
    checks.append(
        DoctorCheck(
            section="auth",
            status="pass" if config_key else "info",
            message=".kagura.json api_key is set"
            if config_key
            else ".kagura.json api_key is not set",
            details={"source": "config", "set": bool(config_key)},
        )
    )
    return checks


def _check_api_key_shape(api_key: str, source: str) -> DoctorCheck:
    preview = _redact_token(api_key)
    if _looks_like_kagura_key(api_key):
        return DoctorCheck(
            section="auth",
            status="pass",
            message=f"API key looks valid: {preview}",
            details={"source": source, "preview": preview},
        )
    return DoctorCheck(
        section="auth",
        status="warn",
        message=f"Unusual API key shape: {preview}",
        details={"source": source, "preview": preview},
    )


def _check_oauth_profile(creds_file, env_profile: str | None) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    profile_name = env_profile or creds_file.default_profile
    creds = creds_file.get_profile(env_profile)
    if creds is None:
        checks.append(
            DoctorCheck(
                section="auth",
                status="fail",
                message="OAuth profile not found in ~/.kagura/credentials.json",
                details={"profile": profile_name},
            )
        )
        return checks

    checks.append(
        DoctorCheck(
            section="auth",
            status="pass",
            message=f"OAuth profile valid: {profile_name}",
            details={"profile": profile_name},
        )
    )

    if not creds.refresh_token:
        checks.append(
            DoctorCheck(
                section="auth",
                status="fail",
                message=f"OAuth profile {profile_name} is missing a refresh token",
                details={"profile": profile_name},
            )
        )
    else:
        checks.append(
            DoctorCheck(
                section="auth",
                status="pass",
                message=f"OAuth refresh token present for {profile_name}",
                details={"profile": profile_name},
            )
        )

    expires_at = creds.expires_at.astimezone(UTC)
    if creds.is_expired():
        checks.append(
            DoctorCheck(
                section="auth",
                status="fail",
                message=f"OAuth access token expired for {profile_name}",
                details={"profile": profile_name, "expires_at": expires_at.isoformat()},
            )
        )
    elif creds.is_expired(skew_seconds=REFRESH_SKEW_SEC):
        checks.append(
            DoctorCheck(
                section="auth",
                status="warn",
                message=f"OAuth token nearing expiration for {profile_name}",
                details={"profile": profile_name, "expires_at": expires_at.isoformat()},
            )
        )

    return checks


def _check_https(mcp_url: str) -> DoctorCheck:
    try:
        validate_https_url(mcp_url, label="MCP URL")
    except ValueError as exc:
        return DoctorCheck(
            section="mcp",
            status="warn",
            message=str(exc),
            details={"mcp_url": mcp_url},
        )
    return DoctorCheck(
        section="mcp",
        status="pass",
        message=f"MCP URL is secure: {mcp_url}",
        details={"mcp_url": mcp_url},
    )


def _check_mcp(project_dir: Path) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    mode = detect_mcp_json_mode(project_dir)
    if mode == "stdio":
        checks.append(DoctorCheck(section="mcp", status="pass", message="MCP Mode: stdio"))
    elif mode == "static-token":
        checks.append(
            DoctorCheck(
                section="mcp",
                status="warn",
                message=(
                    "Legacy static-token configuration detected; run "
                    "`kagura setup claude --profile NAME` to migrate"
                ),
            )
        )
    elif mode == "url":
        checks.append(DoctorCheck(section="mcp", status="pass", message="MCP Mode: url"))
    elif mode == "absent":
        checks.append(
            DoctorCheck(
                section="mcp",
                status="warn",
                message="No usable kagura-memory entry found in .mcp.json",
            )
        )
    else:
        checks.append(DoctorCheck(section="mcp", status="info", message="No .mcp.json found"))

    if mode == "stdio":
        if _kagura_mcp_on_path():
            checks.append(
                DoctorCheck(section="mcp", status="pass", message="kagura-mcp found on PATH")
            )
        else:
            checks.append(
                DoctorCheck(section="mcp", status="fail", message="kagura-mcp not found on PATH")
            )
    else:
        checks.append(
            DoctorCheck(
                section="mcp",
                status="info",
                message="kagura-mcp PATH check skipped because .mcp.json is not stdio mode",
            )
        )

    return checks


async def _check_server(
    resolved: _StaticAuth | _OAuthAuth, *, profile: str | None = None
) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    try:
        if isinstance(resolved, _StaticAuth):
            client = KaguraClient(api_key=resolved.api_key, mcp_url=resolved.mcp_url)
        else:
            client = KaguraClient(mcp_url=resolved.mcp_url, profile=profile)
    except Exception as exc:
        return [DoctorCheck(section="server", status="fail", message=_exc_message(exc))]

    async with client:
        try:
            info = await client.check_server_version()
        except KaguraAuthError as exc:
            if isinstance(resolved, _OAuthAuth):
                return [
                    DoctorCheck(
                        section="server",
                        status="info",
                        message=(
                            "Could not verify server version over REST with an OAuth profile "
                            "(expected: REST validates API keys, not OAuth bearers; the MCP "
                            "connection is unaffected)."
                        ),
                    )
                ]
            return [DoctorCheck(section="server", status="fail", message=str(exc))]
        except KaguraConnectionError as exc:
            return [
                DoctorCheck(
                    section="server",
                    status="fail",
                    message=f"Server unreachable: {exc}",
                )
            ]

    checks.append(DoctorCheck(section="server", status="pass", message="Server reachable"))

    server_version = _parse_version_prefix(info.version)
    minimum_version = _parse_version_prefix(MIN_SERVER_VERSION)
    if server_version is None or minimum_version is None:
        checks.append(
            DoctorCheck(
                section="server",
                status="info",
                message=f"Version: {info.version}",
                details={"version": info.version},
            )
        )
        return checks

    if server_version < minimum_version:
        checks.append(
            DoctorCheck(
                section="server",
                status="fail",
                message=(f"Version: {info.version} is below minimum {MIN_SERVER_VERSION}"),
                details={"version": info.version, "minimum": MIN_SERVER_VERSION},
            )
        )
    else:
        checks.append(
            DoctorCheck(
                section="server",
                status="pass",
                message=f"Version: {info.version}",
                details={"version": info.version},
            )
        )

    return checks


def run_doctor(*, project_dir: Path | None = None, profile: str | None = None) -> DoctorReport:
    """Run all `kagura doctor` checks and return a structured report."""

    cwd = project_dir or Path.cwd()
    config = load_config()

    checks: list[DoctorCheck] = []
    auth_checks, resolved = _check_auth(config, profile=profile, project_dir=cwd)
    checks.extend(auth_checks)
    checks.extend(_check_mcp(cwd))
    checks.extend(_check_optional_dependencies())
    checks.append(_check_litellm())
    checks.append(_check_provider_keys())
    checks.extend(_check_model_provider_match(config))

    if resolved is not None:
        https_check = _check_https(resolved.mcp_url)
        checks.append(https_check)
        if https_check.status == "pass":
            checks.extend(asyncio.run(_check_server(resolved, profile=profile)))
        else:
            checks.append(
                DoctorCheck(
                    section="server",
                    status="info",
                    message="Server connectivity check skipped because the MCP URL is insecure",
                )
            )
    else:
        checks.append(
            DoctorCheck(
                section="server",
                status="info",
                message="Server connectivity check skipped because auth resolution failed",
            )
        )

    return DoctorReport(checks=checks)
