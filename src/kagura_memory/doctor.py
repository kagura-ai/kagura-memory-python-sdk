"""Diagnostic checks for `kagura doctor`."""

from __future__ import annotations

import asyncio
import os
import re
import shlex
from dataclasses import dataclass, field
from datetime import UTC
from importlib import metadata as importlib_metadata
from importlib.metadata import PackageNotFoundError
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Literal

from ._auth import _SOURCE_LABEL, _OAuthAuth, _resolve_auth, _StaticAuth
from ._http import validate_https_url
from ._version import meets_minimum, parse_version
from .auth.cli import _redact_token
from .auth.credentials import REFRESH_SKEW_SEC, load_credentials_file
from .claude_code import (
    McpScope,
    claude_json_label,
    detect_mcp_json_mode,
    find_kagura_mcp_entries,
    unset_header_vars,
)
from .client import _MIN_SERVER_VERSION_TUPLE, MIN_SERVER_VERSION, KaguraClient
from .config import load_config
from .exceptions import KaguraAuthError, KaguraConnectionError, _exc_message
from .setup_claude import _kagura_mcp_on_path

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
# The LiteLLM releases compromised in the March 2026 supply-chain attack.
_LITELLM_BLOCKED_RELEASES = frozenset({(1, 82, 7), (1, 82, 8)})
# A PEP 440 epoch: "0!1.82.7" is the release 1.82.7.
_PEP440_EPOCH_RE = re.compile(r"\A\d+!", re.ASCII)
_PROVIDER_ENV_KEYS: dict[str, str] = {
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "ollama": "OLLAMA_API_KEY",
}
_KEYLESS_PROVIDERS = {"ollama"}
# Mirrored to keep doctor import-light. Importing the ingest package runs
# ingest/__init__.py, pulling fetcher/youtube/chunker/files_client onto every
# `kagura` invocation, including memory-only commands.
_DEFAULT_INGEST_TEXT_MODEL = "claude-sonnet-4-6"  # ingest/providers/claude.py
_DEFAULT_INGEST_VISION_MODEL = "gemini/gemini-2.5-flash"  # ingest/providers/gemini.py
_DEFAULT_AUDIO_MODEL = "gemini/gemini-2.5-flash"  # ingest/_audio.py


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

    # Pre-, post-, dev- and local versions share the triple, so they are
    # blocked too, as the ``ingest`` extra's ``<1.82.7`` pin excludes them.
    parsed = parse_version(_PEP440_EPOCH_RE.sub("", version.strip()))
    if parsed in _LITELLM_BLOCKED_RELEASES:
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


def _check_provider_keys() -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    for provider, env_name in _PROVIDER_ENV_KEYS.items():
        value = os.getenv(env_name) or ""
        checks.append(
            DoctorCheck(
                section="llm",
                status="info",
                message=f"{env_name} is set" if value else f"{env_name} is not set",
                details={
                    "provider": provider,
                    "env": env_name,
                    "set": bool(value),
                    "preview": _redact_token(value) if value else None,
                },
            )
        )
    return checks


def _provider_for_model(model: str) -> str | None:
    normalized = model.strip().lower()
    if not normalized:
        return None
    if normalized.startswith(("ollama/", "ollama_chat/")):
        return "ollama"
    if normalized.startswith(("gemini/", "gemini-")):
        return "gemini"
    if normalized.startswith(("claude", "anthropic/")):
        return "anthropic"
    if normalized.startswith(("openai/", "gpt-", "o1", "o3", "o4")):
        return "openai"
    return None


def _check_model_key_alignment() -> list[DoctorCheck]:
    model_checks = [
        ("ingest-text", _DEFAULT_INGEST_TEXT_MODEL),
        ("ingest-vision", _DEFAULT_INGEST_VISION_MODEL),
        ("ingest-audio", _DEFAULT_AUDIO_MODEL),
    ]
    checks: list[DoctorCheck] = []
    for feature, model in model_checks:
        if not model:
            continue
        provider = _provider_for_model(model)
        details: dict[str, Any] = {"feature": feature, "model": model, "provider": provider}
        if provider is None:
            status: DoctorStatus = "info"
            message = f"{feature} model provider could not be inferred: {model}"
        elif provider in _KEYLESS_PROVIDERS:
            status = "info"
            message = f"{feature} model {model} uses {provider}; no API key is required"
        else:
            env_name = _PROVIDER_ENV_KEYS[provider]
            details["env"] = env_name
            if os.getenv(env_name):
                details["credential_source"] = env_name
                status = "pass"
                message = f"{feature} model {model} has credentials via {env_name}"
            else:
                status = "warn"
                message = f"{feature} model {model} expects {env_name}, but it is not set"
        checks.append(DoctorCheck(section="llm", status=status, message=message, details=details))
    return checks


def _check_llm_providers() -> list[DoctorCheck]:
    """Inspect local LLM provider env only; never call provider APIs."""

    return [*_check_provider_keys(), *_check_model_key_alignment()]


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


# How to replace a legacy ``type: "url"`` entry, by the scope it is in. Setup
# writes project and user scope; a local one must go first, or the re-run's
# shadow check refuses to write under it.
_LEGACY_TYPE_FIX: dict[McpScope, str] = {
    "project": "re-run `kagura setup claude`",
    "user": "re-run `kagura setup claude --scope user`",
    "local": (
        "remove it (`claude mcp remove --scope local kagura-memory`), then re-run "
        "`kagura setup claude`"
    ),
}


def _check_mcp(project_dir: Path) -> list[DoctorCheck]:
    """Report the kagura-memory entry Claude Code uses here, from every scope (#258)."""
    checks: list[DoctorCheck] = []
    # The entry Claude Code uses (strongest scope) first, then any it shadows.
    entries = find_kagura_mcp_entries(project_dir)
    mode = detect_mcp_json_mode(project_dir, entries)
    where = f" ({entries[0].scope} scope, {entries[0].source})" if entries else ""
    details = {"scope": entries[0].scope, "source": entries[0].source} if entries else {}
    if mode == "stdio":
        checks.append(
            DoctorCheck(
                section="mcp", status="pass", message=f"MCP Mode: stdio{where}", details=details
            )
        )
    elif mode == "static-token":
        checks.append(
            DoctorCheck(
                section="mcp",
                status="warn",
                message=(
                    f"Legacy static-token configuration detected{where}; run "
                    "`kagura setup claude --profile NAME` to migrate"
                ),
                details=details,
            )
        )
    elif mode == "url":
        checks.append(
            DoctorCheck(
                section="mcp", status="pass", message=f"MCP Mode: url{where}", details=details
            )
        )
    elif mode == "absent":
        checks.append(
            DoctorCheck(
                section="mcp",
                status="warn",
                message=(
                    f"No usable kagura-memory entry found in {entries[0].source} "
                    f"({entries[0].scope} scope)"
                    if entries  # else: a .mcp.json without a kagura-memory entry
                    else "No usable kagura-memory entry found in .mcp.json"
                ),
                details=details,
            )
        )
    else:
        checks.append(
            DoctorCheck(
                section="mcp",
                status="info",
                message=f"No kagura-memory MCP entry found (.mcp.json, {claude_json_label()})",
            )
        )

    if entries and entries[0].legacy_type:
        fix = _LEGACY_TYPE_FIX[entries[0].scope]
        mcp_json_dir = entries[0].path.parent
        if entries[0].scope == "project" and mcp_json_dir != project_dir.resolve():
            # A parent directory's .mcp.json: a re-run here would write a closer file.
            fix = f"re-run `kagura setup claude --project-dir {shlex.quote(str(mcp_json_dir))}`"
        checks.append(
            DoctorCheck(
                section="mcp",
                status="warn",
                message=(
                    'The kagura-memory entry has type "url", which Claude Code does not '
                    f'accept; {fix} to write it as "http"'
                ),
                details=details,
            )
        )
    for name in unset_header_vars(entries[0].config) if entries else []:
        # e.g. setup's user-scope API-key entry, which sends ${KAGURA_MCP_API_KEY}.
        checks.append(
            DoctorCheck(
                section="mcp",
                status="warn",
                message=(
                    f"The kagura-memory entry sends ${{{name}}} in a header, but {name} is not "
                    "set here: set it in the environment that starts Claude Code, or the "
                    "server rejects the request"
                ),
                details={**details, "env": name},
            )
        )
    for hidden in entries[1:]:
        checks.append(
            DoctorCheck(
                section="mcp",
                status="warn",
                message=(
                    f"kagura-memory is also defined in {hidden.scope} scope ({hidden.source}), "
                    f"but Claude Code uses the {entries[0].scope}-scope entry here"
                ),
                details={"scope": hidden.scope, "source": hidden.source},
            )
        )

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
                message="kagura-mcp PATH check skipped because the MCP entry is not stdio mode",
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

    # The same comparison check_server_version() just made, so its warning and
    # this verdict cannot disagree.
    meets = meets_minimum(info.version, _MIN_SERVER_VERSION_TUPLE)
    if meets is None:
        checks.append(
            DoctorCheck(
                section="server",
                status="info",
                message=f"Version: {info.version}",
                details={"version": info.version},
            )
        )
        return checks

    if not meets:
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
    checks.extend(_check_llm_providers())

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
