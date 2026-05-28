"""Click sub-commands for ``kagura auth``.

Five commands form the public surface:

* ``login``   — RFC 8628 device flow, persist a profile.
* ``status``  — print the current profile (token redacted).
* ``logout``  — revoke + delete a profile (or the whole file).
* ``refresh`` — rotate ``access_token``; optional scope expansion.
* ``token``   — emit ``access_token`` to stdout for CI/scripts.

The commands share a thin sync→async bridge: each Click handler
builds parameters, calls ``asyncio.run(_helper(...))``, and translates
known :class:`KaguraAuthError` subclasses into ``click.ClickException``
with concrete next-step CLI guidance (the precedent set by
``setup_claude.py:333-347``).
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import webbrowser
from datetime import UTC, datetime
from pathlib import Path

import click

from ..exceptions import (
    KaguraAuthDeniedError,
    KaguraAuthError,
    KaguraAuthExpiredError,
    KaguraConnectionError,
    _exc_message,
)
from .credentials import (
    DEFAULT_CREDENTIALS_PATH,
    REFRESH_SKEW_SEC,
    CredentialsFile,
    OAuthCredentials,
    delete_credentials_file,
    delete_profile,
    load_credentials_file,
    reset_state_cache,
    save_credentials_file,
    update_profile,
)
from .device_flow import (
    DEFAULT_CLIENT_ID,
    DeviceAuthorizationResponse,
    TokenResponse,
    authorize_device,
    make_oauth_client,
    poll_for_token,
    refresh_access_token,
    revoke_token,
)

DEFAULT_SERVER = "https://memory.kagura-ai.com"
_SCOPE_READ = "memory:read"
_SCOPE_WRITE = "memory:write"
DEFAULT_SCOPE = f"{_SCOPE_READ} {_SCOPE_WRITE}"
READ_ONLY_SCOPE = _SCOPE_READ


# ---------------------------------------------------------------------------
# Click group
# ---------------------------------------------------------------------------


@click.group()
def auth():
    """OAuth2 device-flow authentication for Kagura Memory.

    Logs you in to the Kagura Memory Cloud server and stores
    credentials at ~/.kagura/credentials.json. Use these commands when
    running 'kagura recall', 'kagura remember', and other CLI tools
    that talk to the server directly.

    For Claude Code MCP integration, continue to use 'kagura setup
    claude' with a long-lived API key from the web UI — automatic
    refresh inside Claude Code's MCP client is tracked in a follow-up
    (kagura-mcp proxy daemon).
    """


# ---------------------------------------------------------------------------
# login
# ---------------------------------------------------------------------------


@auth.command(name="login")
@click.option(
    "--profile",
    default="default",
    help="Profile name to store credentials under (default: 'default').",
)
@click.option(
    "--server",
    default=None,
    help="Server URL (default: $KAGURA_MCP_URL base, or https://memory.kagura-ai.com).",
)
@click.option(
    "--scope",
    default=None,
    help=f"OAuth scope (default: '{DEFAULT_SCOPE}'). Override only if you need a custom scope set.",
)
@click.option(
    "--read-only",
    "read_only",
    is_flag=True,
    help=f"Request read-only scope ('{READ_ONLY_SCOPE}') instead of the default read+write.",
)
@click.option(
    "--no-browser",
    is_flag=True,
    help="Don't try to open a browser — just print the URL and code.",
)
def auth_login(
    profile: str,
    server: str | None,
    scope: str | None,
    read_only: bool,
    no_browser: bool,
) -> None:
    """Authenticate via OAuth2 device flow.

    \b
    Examples:
      kagura auth login                      # read + write (default)
      kagura auth login --read-only          # read-only
      kagura auth login --scope "memory:read memory:write profile:read"  # custom
      kagura auth login --profile work
      kagura auth login --no-browser         # for SSH / headless

    \b
    Memory scopes:
      memory:read   — read memories, contexts, files
      memory:write  — create/update/delete memories, contexts, files
      (additional scopes like profile:read may be accepted server-side;
      pass them through --scope.)
    """
    if read_only and scope is not None:
        raise click.ClickException("--read-only and --scope are mutually exclusive; pick one.")
    if scope is not None:
        resolved_scope = scope
    elif read_only:
        resolved_scope = READ_ONLY_SCOPE
    else:
        resolved_scope = DEFAULT_SCOPE
    resolved_server = _resolve_server(server)
    try:
        asyncio.run(
            _run_login(
                server=resolved_server,
                profile=profile,
                scope=resolved_scope,
                open_browser=not no_browser,
            )
        )
    except click.ClickException:
        raise
    except KaguraAuthDeniedError as e:
        raise click.ClickException(
            f"{_exc_message(e)}\n  Re-run: kagura auth login --profile {profile}"
        ) from e
    except KaguraAuthExpiredError as e:
        raise click.ClickException(_exc_message(e)) from e
    except KaguraConnectionError as e:
        raise click.ClickException(_exc_message(e)) from e
    except Exception as e:
        raise click.ClickException(_exc_message(e)) from e


async def _run_login(
    *,
    server: str,
    profile: str,
    scope: str,
    open_browser: bool,
) -> None:
    async with make_oauth_client() as client:
        device = await authorize_device(
            client,
            server,
            client_id=DEFAULT_CLIENT_ID,
            scope=scope,
        )
        _print_device_prompt(device, attempt_browser=open_browser)
        token = await poll_for_token(
            client,
            server,
            client_id=DEFAULT_CLIENT_ID,
            device_code=device.device_code,
            interval=device.interval,
            expires_at=device.expires_at,
        )

    creds = _build_credentials(token, server=server)
    update_profile(profile, creds)
    reset_state_cache()  # invalidate any prior in-process state
    _print_login_success(creds, profile)


def _print_device_prompt(device: DeviceAuthorizationResponse, *, attempt_browser: bool) -> None:
    """Print URL + code FIRST, then optionally try to open a browser.

    The URL+code lines come out unconditionally so that even when the
    browser opens silently, the operator can still copy the code by
    eye if anything else goes wrong.
    """
    click.echo()
    click.echo(f"! First copy your one-time code: {device.user_code}")
    click.echo("  Open this URL in your browser to approve:")
    click.echo(f"    {device.verification_uri_complete}")
    click.echo()

    if not attempt_browser:
        click.echo("  (--no-browser: not opening a browser; polling will continue here.)")
        return

    if not _try_open_browser(device.verification_uri_complete):
        click.echo(
            "  Could not auto-open the browser. Open the URL above manually. "
            "Polling will continue here."
        )


def _try_open_browser(url: str) -> bool:
    """Open ``url`` in the user's default browser.

    On WSL, skip the stdlib :mod:`webbrowser` module entirely and hand
    the URL to a dedicated opener — either ``wslview`` (a Linux-side
    helper from the ``wslu`` package, preferred when available) or
    ``rundll32.exe`` from the Windows side (always available). Python's
    webbrowser can report success on WSL even when no Windows browser
    actually launches, so we don't trust it there. On other platforms,
    try the stdlib first and fall back if it reports failure.

    All fallback openers are non-shell binaries invoked via argv, so a
    URL containing shell metacharacters (``&``, ``|``, ``%``, ``!``,
    etc.) cannot be reinterpreted as command chaining. We intentionally
    do NOT fall back to ``cmd.exe /c start`` — cmd.exe re-parses its
    argv as a shell, and an allowlist tight enough to be safe would
    also reject normal URL characters like ``&``. We also avoid
    ``explorer.exe URL`` because Windows Explorer resolves the argument
    as a path and opens a folder window instead of delegating to the
    URL handler.

    Returns ``True`` if any opener was dispatched successfully.
    """
    on_wsl = _is_wsl()
    if not on_wsl:
        try:
            if webbrowser.open(url):
                return True
        except (webbrowser.Error, OSError):
            # Fall through to the platform-specific openers below.
            pass

    fallback_commands: list[list[str]] = []
    if on_wsl:
        if shutil.which("wslview"):
            fallback_commands.append(["wslview", url])
        # rundll32 calls url.dll's FileProtocolHandler, which hands the
        # URL to ShellExecute — the canonical Windows API for opening
        # the registered protocol handler. rundll32 is not a shell, so
        # shell metacharacters in the URL stay literal, and unlike
        # explorer.exe it never falls back to treating the argument
        # as a filesystem path.
        fallback_commands.append(["rundll32.exe", "url.dll,FileProtocolHandler", url])
    elif sys.platform == "darwin":
        fallback_commands.append(["open", url])
    elif sys.platform.startswith("linux") and shutil.which("xdg-open"):
        fallback_commands.append(["xdg-open", url])

    for cmd in fallback_commands:
        try:
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return True
        except OSError:
            continue

    return False


def _is_wsl() -> bool:
    """Detect Windows Subsystem for Linux (WSL1/WSL2)."""
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        with open("/proc/sys/kernel/osrelease", encoding="ascii", errors="ignore") as f:
            release = f.read().lower()
        return "wsl" in release or "microsoft" in release
    except OSError:
        return False


def _print_login_success(creds: OAuthCredentials, profile: str) -> None:
    expires_utc = creds.expires_at.astimezone(UTC).isoformat()
    click.echo()
    click.echo(f"✓ Logged in as {creds.user_email or '<unknown user>'}")
    if creds.workspace_name:
        click.echo(f"  Workspace: {creds.workspace_name}")
    click.echo(f"  Profile: {profile}")
    click.echo(f"  Server: {creds.server}")
    click.echo(f"  Scope: {creds.scope}")
    click.echo(f"  Expires: {expires_utc} (refreshable)")
    click.echo()
    click.echo(
        "  Note: To use Kagura from Claude Code, continue to use "
        "'kagura setup claude' with an API key from the web UI."
    )


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


@auth.command(name="status")
@click.option("--profile", default=None, help="Profile to inspect (default: default profile).")
def auth_status(profile: str | None) -> None:
    """Show the current profile, server, scope, expiry, and workspace."""
    cf = load_credentials_file()
    creds = cf.get_profile(profile)
    if creds is None:
        target = profile or cf.default_profile or "(none)"
        raise click.ClickException(
            f"No credentials found for profile '{target}'.\n  Run: kagura auth login"
        )

    name = profile or cf.default_profile
    click.echo()
    click.echo(f"Profile: {name}")
    click.echo(f"Server: {creds.server}")
    if creds.workspace_name:
        click.echo(f"Workspace: {creds.workspace_name} ({creds.workspace_id[:8]}...)")
    if creds.user_email:
        click.echo(f"User: {creds.user_email}")
    click.echo(f"Scope: {creds.scope}")
    click.echo(f"Access token: {_redact_token(creds.access_token)}")
    # refresh_token is intentionally NOT shown — even redacted. Its
    # threat model is "can mint new access_tokens until manually revoked",
    # which is strictly higher value than the short-lived access_token.
    expires = creds.expires_at.astimezone(UTC)
    delta = expires - datetime.now(UTC)
    if delta.total_seconds() > 0:
        click.echo(f"Expires: {expires.isoformat()} (in {_humanize_delta(delta.total_seconds())})")
    else:
        click.echo(
            f"Expires: {expires.isoformat()} (EXPIRED — auto-refresh will run on next request)"
        )
    click.echo(f"Issued: {creds.issued_at.astimezone(UTC).isoformat()}")


# ---------------------------------------------------------------------------
# logout
# ---------------------------------------------------------------------------


@auth.command(name="logout")
@click.option("--profile", default=None, help="Profile to log out (default: default profile).")
@click.option(
    "--all",
    "all_profiles",
    is_flag=True,
    help="Delete every profile and remove the credentials file. Requires --yes.",
)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    help="Confirm --all without an interactive prompt.",
)
def auth_logout(profile: str | None, all_profiles: bool, yes: bool) -> None:
    """Revoke server-side and delete the local profile."""
    if all_profiles and not yes:
        raise click.ClickException(
            "Refusing to delete all profiles without explicit confirmation.\n"
            "  Re-run: kagura auth logout --all --yes"
        )

    cf = load_credentials_file()
    if not cf.profiles:
        raise click.ClickException(
            "No credentials to log out of.\n  (~/.kagura/credentials.json is empty or missing.)"
        )

    if all_profiles:
        asyncio.run(_revoke_all_best_effort(cf))
        delete_credentials_file()
        reset_state_cache()
        click.echo("✓ All profiles removed; ~/.kagura/credentials.json deleted.")
        _warn_if_api_key_env()
        return

    target = profile or cf.default_profile
    creds = cf.get_profile(target)
    if creds is None:
        raise click.ClickException(
            f"No profile named '{target}'.\n"
            f"  Inspect ~/.kagura/credentials.json to see which profiles exist."
        )

    asyncio.run(_revoke_one_best_effort(creds))
    delete_profile(target)
    reset_state_cache()
    click.echo(f"✓ Profile '{target}' removed.")
    _warn_if_api_key_env()


def _warn_if_api_key_env() -> None:
    """Remind the user that `KAGURA_API_KEY` env still authenticates after logout."""
    if os.getenv("KAGURA_API_KEY"):
        click.echo(
            "  Note: KAGURA_API_KEY is set in your environment — "
            "the env var will still authenticate kagura commands until you unset it."
        )


async def _revoke_one_best_effort(creds: OAuthCredentials) -> None:
    async with make_oauth_client() as client:
        ok = await revoke_token(
            client,
            creds.server,
            token=creds.access_token,
            client_id=creds.client_id,
        )
    if not ok:
        click.echo(
            "  Warning: server-side revoke failed (network or 5xx). "
            "The local profile was still deleted. "
            "The refresh_token may remain valid until it expires naturally."
        )


async def _revoke_all_best_effort(cf: CredentialsFile) -> None:
    async with make_oauth_client() as client:
        for creds in cf.profiles.values():
            await revoke_token(
                client,
                creds.server,
                token=creds.access_token,
                client_id=creds.client_id,
            )


# ---------------------------------------------------------------------------
# refresh
# ---------------------------------------------------------------------------


@auth.command(name="refresh")
@click.option("--profile", default=None, help="Profile to refresh (default: default profile).")
@click.option(
    "--scope",
    default=None,
    help=(
        "Request this scope (default: keep the current grant unchanged). "
        "Pass space-separated values to ask for multiple "
        "(e.g. 'memory:read memory:write'). "
        "If wider than the current grant, re-runs the device flow."
    ),
)
def auth_refresh(profile: str | None, scope: str | None) -> None:
    """Rotate ``access_token`` (optionally requesting a new scope).

    \b
    Examples:
      kagura auth refresh
      kagura auth refresh --scope "memory:read"                       # narrow to read-only
      kagura auth refresh --scope "memory:read memory:write"          # widen (triggers device flow)
      kagura auth refresh --profile work

    \b
    Memory scopes:
      memory:read   — read memories, contexts, files
      memory:write  — create/update/delete memories, contexts, files
      (additional scopes like profile:read may be accepted server-side;
      pass them through --scope.)
    """
    cf = load_credentials_file()
    target = profile or cf.default_profile
    creds = cf.get_profile(target)
    if creds is None:
        raise click.ClickException(f"No profile named '{target}'.\n  Run: kagura auth login")

    try:
        new_creds = asyncio.run(_run_refresh(creds, scope=scope))
    except click.ClickException:
        raise
    except KaguraAuthExpiredError as e:
        raise click.ClickException(_exc_message(e)) from e
    except KaguraConnectionError as e:
        raise click.ClickException(_exc_message(e)) from e
    except Exception as e:
        raise click.ClickException(_exc_message(e)) from e

    update_profile(target, new_creds)
    reset_state_cache()
    click.echo(f"✓ Refreshed profile '{target}'.")
    click.echo(f"  Scope: {new_creds.scope}")
    click.echo(f"  Expires: {new_creds.expires_at.astimezone(UTC).isoformat()}")


async def _run_refresh(
    creds: OAuthCredentials,
    *,
    scope: str | None,
) -> OAuthCredentials:
    """Refresh; on scope widening, fall back to a full device flow."""
    async with make_oauth_client() as client:
        try:
            token = await refresh_access_token(
                client,
                creds.server,
                client_id=creds.client_id,
                refresh_token=creds.refresh_token,
                scope=scope,
            )
            return _build_credentials(token, server=creds.server, base=creds)
        except KaguraAuthError as e:
            # Heuristic: when the user asked for a wider scope and the
            # server rejected it as insufficient/invalid_scope, fall
            # through to a fresh device flow to collect new consent.
            message = str(e).lower()
            if scope is not None and (
                "insufficient_scope" in message or "invalid_scope" in message
            ):
                click.echo(
                    f"  Requested scope '{scope}' is wider than the current grant — "
                    "re-running the device flow for consent..."
                )
                device = await authorize_device(
                    client,
                    creds.server,
                    client_id=creds.client_id,
                    scope=scope,
                )
                _print_device_prompt(device, attempt_browser=True)
                token = await poll_for_token(
                    client,
                    creds.server,
                    client_id=creds.client_id,
                    device_code=device.device_code,
                    interval=device.interval,
                    expires_at=device.expires_at,
                )
                return _build_credentials(token, server=creds.server)
            raise


# ---------------------------------------------------------------------------
# token
# ---------------------------------------------------------------------------


@auth.command(name="token")
@click.option("--profile", default=None, help="Profile to use (default: default profile).")
def auth_token(profile: str | None) -> None:
    """Emit the raw ``access_token`` to stdout (for CI / scripts).

    \b
    The token will be auto-refreshed if it's within 5 minutes of expiry.
    A warning with the absolute expiry time is written to stderr.

    \b
    Security note: this token is short-lived — do not persist it to
    files or commit it. For long-lived CI, use a Kagura API key
    instead (set KAGURA_API_KEY in your secret store).
    """
    cf = load_credentials_file()
    target = profile or cf.default_profile
    creds = cf.get_profile(target)
    if creds is None:
        raise click.ClickException(f"No profile named '{target}'.\n  Run: kagura auth login")

    if creds.is_expired(skew_seconds=REFRESH_SKEW_SEC):
        try:
            new_creds = asyncio.run(_run_refresh(creds, scope=None))
        except KaguraAuthExpiredError as e:
            raise click.ClickException(_exc_message(e)) from e
        except KaguraConnectionError as e:
            raise click.ClickException(_exc_message(e)) from e
        except Exception as e:
            raise click.ClickException(_exc_message(e)) from e
        update_profile(target, new_creds)
        reset_state_cache()
        creds = new_creds

    click.echo(creds.access_token)
    expires = creds.expires_at.astimezone(UTC)
    delta = expires - datetime.now(UTC)
    click.echo(
        f"⚠ This token expires at {expires.isoformat()} "
        f"(in {_humanize_delta(delta.total_seconds())}).\n"
        f"  Don't persist it to files or shell history. "
        f"For long-lived CI, use a Kagura API key.",
        err=True,
    )


# ---------------------------------------------------------------------------
# Helpers (private)
# ---------------------------------------------------------------------------


def _resolve_server(explicit: str | None) -> str:
    """Pick the OAuth server URL and enforce HTTPS (except localhost).

    Priority: explicit ``--server`` > ``KAGURA_MCP_URL`` (base) > default.
    Validation mirrors :func:`kagura_memory._http.validate_https_url` so a
    user can't accidentally point the device-flow at ``http://evil.com``.
    """
    from .._http import base_url_from_mcp, validate_https_url

    if explicit:
        url = explicit.rstrip("/")
    elif env_mcp := os.getenv("KAGURA_MCP_URL"):
        url = base_url_from_mcp(env_mcp.rstrip("/")) or env_mcp.rstrip("/")
    else:
        url = DEFAULT_SERVER

    validate_https_url(url, label="Server URL")
    return url


def _build_credentials(
    token: TokenResponse,
    *,
    server: str,
    base: OAuthCredentials | None = None,
) -> OAuthCredentials:
    """Turn a successful token response into an :class:`OAuthCredentials`.

    Initial login (``base is None``): construct from scratch, taking
    ``workspace_*`` / ``user_email`` from the token response.

    Refresh (``base`` provided): apply rotated token fields onto the
    existing credentials, preserving ``workspace_*`` / ``user_email``
    and falling back to the prior scope when the server omits it.
    """
    if base is None:
        return OAuthCredentials(
            server=server,
            mcp_url=f"{server.rstrip('/')}/mcp",
            client_id=DEFAULT_CLIENT_ID,
            access_token=token.access_token,
            refresh_token=token.refresh_token,
            token_type=token.token_type,
            expires_at=token.expires_at,
            scope=token.scope,
            workspace_id=token.workspace_id,
            workspace_name=token.workspace_name,
            user_email=token.user_email,
            issued_at=datetime.now(UTC),
        )
    return base.with_refreshed(
        access_token=token.access_token,
        refresh_token=token.refresh_token or None,
        expires_at=token.expires_at,
        scope=token.scope or base.scope,
    )


def _redact_token(token: str) -> str:
    """Show enough prefix/suffix to identify a token without exposing it."""
    if not token:
        return "<empty>"
    if len(token) <= 12:
        return "<redacted>"
    return f"{token[:8]}...{token[-4:]}"


def _humanize_delta(seconds: float) -> str:
    """Render a duration like ``14d 0h 0m`` or ``1h 23m``."""
    seconds = int(max(0, seconds))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


# ---------------------------------------------------------------------------
# These imports keep type-checkers and ``test_cli.py:test_auth_login``-style
# tests happy when patching ``kagura_memory.auth.cli.<symbol>``.
# ---------------------------------------------------------------------------

__all__ = [
    "auth",
    "auth_login",
    "auth_logout",
    "auth_refresh",
    "auth_status",
    "auth_token",
]

# Avoid "imported but unused" warnings for re-exports that exist solely
# so tests can patch them at the cli module level (e.g. via
# ``patch("kagura_memory.auth.cli.<func>", ...)``).
_ = (
    DEFAULT_CREDENTIALS_PATH,
    save_credentials_file,
    Path,
)
