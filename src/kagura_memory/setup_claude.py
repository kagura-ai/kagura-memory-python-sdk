"""Setup Kagura Memory integration for Claude Code."""

import asyncio
import json
import os
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlsplit, urlunsplit

import click

from . import claude_code
from ._http import mcp_url_has_tools_allowlist, mcp_url_with_query
from .claude_code import (
    MCP_API_KEY_ENV,
    MCP_PROXY_COMMAND,
    MCP_SERVER_NAME,
    SCOPE_PRECEDENCE,
    McpEntry,
    McpScope,
    _read_json_safe,
    classify_mcp_entry,
    claude_json_label,
    claude_json_path,
    find_kagura_mcp_entries,
    same_mcp_entry,
)

# Re-exported: the detector lived here before #258, and callers import it from here.
from .claude_code import detect_mcp_json_mode as detect_mcp_json_mode
from .client import KaguraClient
from .exceptions import KaguraAuthError, KaguraConnectionError, _exc_message

DEFAULT_MCP_URL = "http://localhost:8080/mcp"

# Trailing space distinguishes "kagura recall/remember" from "kagura-memory" in MCP config
KAGURA_HOOK_MARKER = "kagura "

_CONTEXT_ID_PATTERN = r"^[a-zA-Z0-9_-]+$"

# --trusted-only (#258): the output is injected into the session unreviewed,
# so connector-ingested memories stay out of it. Removal matches this
# command's fixed prefix (see _sdk_prefix), which the pre-#258 form shares.
SESSIONSTART_HOOK_COMMAND = (
    'kagura recall "project context and recent decisions" '
    "-c {context_id} -k 5 --trusted-only 2>/dev/null | head -c 2000 || true"
)

POSTTOOLUSE_HOOK_COMMAND = (
    "INPUT=$(cat); "
    "FILE=$(echo \"$INPUT\" | jq -r '.tool_input.file_path // empty'); "
    '[ -z "$FILE" ] && exit 0; '
    'case "$FILE" in *.claude/memory/*) ;; *) exit 0;; esac; '
    'BASENAME=$(basename "$FILE" .md); '
    "kagura remember "
    '-s "memory: $BASENAME" '
    '--content "$(cat "$FILE")" '
    "-c {context_id} --type note -i 0.5 2>/dev/null || true; "
    "exit 0"
)

SKILL_RECALL = """\
---
description: Recall Kagura memories relevant to the current task
arguments:
  - name: query
    description: "Search query (optional — defaults to current task context)"
    required: false
---

Search Kagura Memory for context relevant to what we're working on.

```bash
kagura recall "$ARGUMENTS" -c {context_id} -k 10
```

If no arguments provided, recall recent project context:

```bash
kagura recall "project context and recent decisions" -c {context_id} -k 5
```
"""

SKILL_REMEMBER = """\
---
description: Store a memory in Kagura Memory
arguments:
  - name: summary
    description: "What to remember"
    required: true
---

Store a memory about the current work in Kagura Memory.

```bash
kagura remember -s "$ARGUMENTS" \\
  --content "<gather details from context>" -c {context_id}
```

Ask the user what to remember if $ARGUMENTS is empty.
"""


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Write a dict as formatted UTF-8 JSON (locale-independent, issue #197)."""
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _validate_not_empty(value: str, name: str) -> str:
    """Validate that a prompted value is not empty."""
    value = value.strip()
    if not value:
        raise click.UsageError(f"{name} cannot be empty")
    return value


def _prompt_api_key(existing: str | None, non_interactive: bool) -> str:
    """Prompt for API key, or use existing."""
    if non_interactive:
        if not existing:
            raise click.ClickException(
                "API key required in non-interactive mode. Use --api-key or set KAGURA_API_KEY"
            )
        return existing
    if existing:
        masked = f"{existing[:8]}...{existing[-4:]}" if len(existing) > 12 else existing
        click.echo(f"  Existing key: {masked}")
    value = click.prompt("Kagura API Key", default=existing or "")
    return _validate_not_empty(value, "API Key")


def _prompt_mcp_url(existing: str | None, non_interactive: bool) -> str:
    """Prompt for MCP URL, or use existing/default."""
    if non_interactive:
        if not existing:
            raise click.ClickException(
                "MCP URL required in non-interactive mode. Use --mcp-url or set KAGURA_MCP_URL"
            )
        return existing
    value = click.prompt("MCP URL", default=existing or "")
    return _validate_not_empty(value, "MCP URL")


def _make_client(api_key: str | None, mcp_url: str | None, profile: str | None) -> KaguraClient:
    """Build a KaguraClient for either the API-key or OAuth-profile path.

    When ``profile`` is set, authentication and the MCP URL come from the
    OAuth profile in ``~/.kagura/credentials.json``; ``api_key`` / ``mcp_url``
    are ignored. Otherwise the static API-key path is used.
    """
    if profile is not None:
        return KaguraClient(profile=profile)
    return KaguraClient(api_key=api_key, mcp_url=mcp_url)


async def _test_connection(
    api_key: str | None = None,
    mcp_url: str | None = None,
    *,
    profile: str | None = None,
) -> dict[str, Any]:
    """Test connection and return contexts list."""
    async with _make_client(api_key, mcp_url, profile) as client:
        return await client.list_contexts()


async def _create_context(
    api_key: str | None,
    mcp_url: str | None,
    name: str,
    summary: str | None,
    *,
    profile: str | None = None,
) -> dict[str, Any]:
    """Create a new context."""
    async with _make_client(api_key, mcp_url, profile) as client:
        return await client.create_context(name=name, summary=summary)


def _auto_match_context(
    contexts: list[dict[str, Any]], project_dir: Path, threshold: float = 0.65
) -> dict[str, Any] | None:
    """Return the context whose name best matches the project dir name,
    or None if no candidate clears the threshold or there is an ambiguous tie."""
    from difflib import SequenceMatcher

    target = project_dir.name.lower().replace("_", "-")
    if not target:
        # Empty target (e.g., Path("/") or Path(".") with no resolved name) would
        # cause `target in name` to be True for every candidate. Bail early.
        return None
    best: tuple[float, dict[str, Any]] | None = None
    tied = False
    for ctx in contexts:
        name = (ctx.get("name") or "").lower().replace("_", "-")
        if not name:
            continue
        score = SequenceMatcher(None, target, name).ratio()
        # Bonus for substring containment in either direction
        if target in name or name in target:
            score = max(score, 0.85)
        if best is None or score > best[0]:
            best = (score, ctx)
            tied = False
        elif score == best[0]:
            tied = True
    # Reject ambiguous matches (top score ties — fall through to manual prompt)
    if best and best[0] >= threshold and not tied:
        return best[1]
    return None


def _select_or_create_context(
    contexts_response: dict[str, Any],
    api_key: str | None,
    mcp_url: str | None,
    preselected: str | None,
    project_dir: Path,
    non_interactive: bool,
    no_auto_context: bool = False,
    *,
    profile: str | None = None,
) -> str:
    """Select existing context or create a new one. Returns context_id.

    When ``profile`` is set, any context creation authenticates via the OAuth
    profile instead of ``api_key`` / ``mcp_url``.
    """
    contexts = contexts_response.get("contexts", [])

    if preselected:
        for ctx in contexts:
            if ctx.get("id") == preselected or ctx.get("name") == preselected:
                ctx_id = ctx.get("id", preselected)
                click.echo(f"  Using context: {ctx.get('name', '?')} ({ctx_id[:8]}...)")
                return ctx_id
        # Treat as raw UUID if not found in context list
        click.echo(f"  Using context: {preselected}")
        return preselected

    default_name = project_dir.name.lower().replace(" ", "-")

    if non_interactive:
        result = asyncio.run(_create_context(api_key, mcp_url, default_name, None, profile=profile))
        ctx_id = result.get("context_id") or result.get("id", "")
        click.echo(f"  Created context: {default_name} ({ctx_id[:8]}...)")
        return ctx_id

    if contexts and not no_auto_context:
        auto = _auto_match_context(contexts, project_dir)
        if auto:
            auto_id = auto.get("id", "")
            click.echo(f"\nSuggested context: {auto.get('name', '?')} ({auto_id[:8]}...)")
            click.echo("  (use --no-auto-context to disable, or pick a different one below)")
            if click.confirm("Use this suggested context?", default=True):
                return auto_id
        # Fall through to manual selection if auto-match declined

    if contexts:
        click.echo("\nExisting contexts:")
        for i, ctx in enumerate(contexts, 1):
            click.echo(f"  {i}. {ctx.get('name', '?')} ({ctx.get('id', '?')[:8]}...)")
        click.echo(f"  {len(contexts) + 1}. Create new context")

        choice = click.prompt(
            "Select context",
            type=click.IntRange(1, len(contexts) + 1),
            default=len(contexts) + 1,
        )

        if choice <= len(contexts):
            ctx = contexts[choice - 1]
            return ctx.get("id", "")

    name = click.prompt("Context name", default=default_name)
    summary = click.prompt("Context summary (optional)", default="", show_default=False)
    result = asyncio.run(
        _create_context(api_key, mcp_url, name, summary if summary else None, profile=profile)
    )
    ctx_id = result.get("context_id") or result.get("id", "")
    click.echo(f"  Created context: {name} ({ctx_id[:8]}...)")
    return ctx_id


def _write_kagura_config(project_dir: Path, api_key: str, mcp_url: str, context_id: str) -> Path:
    """Write .kagura.json, merging with existing config."""
    path = project_dir / ".kagura.json"
    existing = _read_json_safe(path)
    existing["api_key"] = api_key
    existing["mcp_url"] = mcp_url
    existing["context_id"] = context_id
    _write_json(path, existing)
    return path


def _static_token_entry(api_key: str, url: str) -> dict[str, Any]:
    """The API-key server entry: ``http`` with a baked ``Authorization`` header.

    ``http`` is Claude Code's remote type; SDKs before #258 wrote ``url``,
    which Claude Code does not accept (the detector still recognises it).
    At user scope ``api_key`` is :data:`_API_KEY_REF`, never the key itself.
    """
    return {"type": "http", "url": url, "headers": {"Authorization": f"Bearer {api_key}"}}


#: What a user-scope API-key entry carries in place of the key (see
#: :data:`~kagura_memory.claude_code.MCP_API_KEY_ENV`).
_API_KEY_REF = f"${{{MCP_API_KEY_ENV}}}"


def _stdio_entry(
    profile: str, *, guardrails: str | None = None, tool_profile: str | None = None
) -> dict[str, Any]:
    """The OAuth server entry: Claude Code launches the ``kagura-mcp`` proxy.

    ``--guardrails`` / ``--tool-profile`` make the proxy put ``?guardrails=`` /
    ``?profile=`` on the upstream URL at run time (#258), so the entry never
    copies the profile's URL and changing them never forces a re-login.
    """
    args = ["--profile", profile]
    if guardrails is not None:
        args += ["--guardrails", guardrails]
    if tool_profile is not None:
        args += ["--tool-profile", tool_profile]
    return {"type": "stdio", "command": MCP_PROXY_COMMAND, "args": args}


def _write_project_mcp_entry(project_dir: Path, entry: dict[str, Any]) -> Path:
    """Put ``entry`` into ``<project>/.mcp.json`` as ``kagura-memory``, keeping other servers."""
    path = project_dir / ".mcp.json"
    existing = _read_json_safe(path)
    servers = existing.setdefault("mcpServers", {})
    servers[MCP_SERVER_NAME] = entry
    _write_json(path, existing)
    return path


def _write_mcp_json(project_dir: Path, api_key: str, mcp_url: str) -> Path:
    """Write .mcp.json for Claude Code MCP server config, merging with existing.

    This is the static-token form: the API key is baked into an
    ``Authorization`` header that Claude Code reads once at startup and never
    refreshes. Fine for long-lived API keys (CI / service accounts); for
    short-lived OAuth tokens use :func:`_write_mcp_json_stdio` instead.
    """
    return _write_project_mcp_entry(project_dir, _static_token_entry(api_key, mcp_url))


def _write_mcp_json_stdio(project_dir: Path, profile: str) -> Path:
    """Write .mcp.json pointing Claude Code at the ``kagura-mcp`` stdio proxy.

    The refresh-aware proxy (issue #101) owns ``~/.kagura/credentials.json``
    and injects an always-fresh OAuth bearer per request, so — unlike the
    static-token form written by :func:`_write_mcp_json` — this entry contains
    **no secret** and never goes stale after the access token's ``expires_at``.
    """
    return _write_project_mcp_entry(project_dir, _stdio_entry(profile))


def _kagura_mcp_on_path() -> bool:
    """True when the ``kagura-mcp`` console script is resolvable on ``$PATH``."""
    import shutil

    return shutil.which(MCP_PROXY_COMMAND) is not None


# =============================================================================
# Where the MCP entry goes: --scope and the shadow check (#258)
# =============================================================================


@dataclass
class _McpPlan:
    """The MCP entry to write, checked against every scope before anything is written."""

    scope: McpScope
    entry: dict[str, Any]
    #: Entries the new one hides in this project: weaker scopes' and a parent
    #: directory's ``.mcp.json``.
    hidden: list[McpEntry] = field(default_factory=list)
    #: A different user-scope entry to replace. ``claude mcp add-json``
    #: refuses to overwrite one, so it is removed first (and restored if the
    #: add fails).
    replaces: dict[str, Any] | None = None
    #: The user-scope entry already matches; nothing to write.
    unchanged: bool = False
    #: ``--guardrails`` / ``--tool-profile`` the replaced same-scope entry had
    #: and this run leaves out, e.g. ``["--guardrails off"]``.
    dropped: list[str] = field(default_factory=list)


def _claude_command(args: list[str], *, cwd: Path | None = None) -> str:
    """``claude <args>`` as a POSIX shell line, to print for the user.

    ``claude mcp`` resolves local and project scope from the directory it
    runs in, so ``cwd`` adds a ``cd`` when it is not the current directory.
    """
    command = shlex.join(["claude", *args])
    if cwd is not None and cwd != Path.cwd().resolve():
        command = f"cd {shlex.quote(str(cwd))} && {command}"
    return command


def _redact_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """``entry`` with a baked API key replaced by a placeholder, for printing.

    A ``${VAR}`` reference is kept: it is what the user should run.
    """
    headers = entry.get("headers")
    if not isinstance(headers, dict) or not claude_code.holds_credential(entry):
        return entry
    masked = {
        k: "Bearer <your-api-key>" if k.lower() == "authorization" else v
        for k, v in headers.items()
    }
    return {**entry, "headers": masked}


_QUERY_FLAGS = (("--guardrails", "guardrails"), ("--tool-profile", "profile"))


def _query_flags(entry: dict[str, Any]) -> dict[str, str]:
    """The ``--guardrails`` / ``--tool-profile`` values an entry puts on the upstream URL.

    Read from the ``kagura-mcp`` args of a stdio entry, or from the url's
    query (first value, as the server reads it) of an http one.
    """
    found: dict[str, str] = {}
    if classify_mcp_entry(entry) == "stdio":
        args = entry.get("args")
        argv = [a for a in args if isinstance(a, str)] if isinstance(args, list) else []
        for flag, _ in _QUERY_FLAGS:
            for i, arg in enumerate(argv):
                if arg == flag and i + 1 < len(argv):
                    found[flag] = argv[i + 1]
                elif arg.startswith(f"{flag}="):
                    found[flag] = arg.partition("=")[2]
        return found
    url = entry.get("url")
    params = parse_qsl(urlsplit(url).query) if isinstance(url, str) else []
    for flag, key in _QUERY_FLAGS:
        value = next((v for k, v in params if k == key), None)
        if value is not None:
            found[flag] = value
    return found


def _dropped_query_flags(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """``--guardrails`` / ``--tool-profile`` settings ``old`` has and ``new`` leaves out."""
    kept = _query_flags(new)
    return [f"{flag} {value}" for flag, value in _query_flags(old).items() if flag not in kept]


def _plan_mcp_entry(
    project: Path, scope: McpScope, entry: dict[str, Any], non_interactive: bool
) -> _McpPlan:
    """Check where ``entry`` would land before any file or context is touched.

    Claude Code uses the ``kagura-memory`` entry from the strongest scope
    (local > project > user) and ignores the rest. An entry that would lose is
    refused with ``-y`` (exit 1) and needs an explicit yes otherwise; one that
    hides a weaker entry is written, with a note. ``~/.claude.json`` is only
    read. A user-scope write also needs the ``claude`` CLI, checked here so a
    missing one stops setup before anything is written.

    Raises:
        click.ClickException: The entry would be shadowed and was not
            confirmed, a replacement was declined, or ``claude`` is missing.
    """
    entries = find_kagura_mcp_entries(project)
    rank = SCOPE_PRECEDENCE.index
    stronger = [e for e in entries if rank(e.scope) < rank(scope)]
    if stronger:
        click.echo(
            f"\n  Warning: Claude Code uses the {MCP_SERVER_NAME} entry from the strongest scope,\n"
            f"  so in this project a {scope}-scope entry would be hidden by:"
        )
        for e in stronger:
            # `claude mcp remove` finds a local entry from anywhere in the project,
            # a project one only in the directory whose .mcp.json holds it.
            cwd = {"local": project, "project": e.path.parent, "user": None}[e.scope]
            remove = claude_code.mcp_remove_args(e.scope)
            click.echo(f"    {e.scope} scope ({e.source}) — remove it with:")
            click.echo(f"      {_claude_command(remove, cwd=cwd)}")
        if non_interactive:
            raise click.ClickException(
                f"Nothing was written: the {stronger[0].scope}-scope {MCP_SERVER_NAME} entry "
                f"would hide the {scope}-scope one. Remove it (command above) and re-run."
            )
        if not click.confirm(f"Write the {scope}-scope entry anyway?", default=False):
            raise click.ClickException("Setup cancelled; nothing was written.")

    # The entry this write replaces. A project-scope one in a parent directory's
    # .mcp.json stays where it is: the new, closer file hides it.
    target = project / ".mcp.json" if scope == "project" else claude_json_path()
    current = next((e for e in entries if e.scope == scope and e.path == target), None)
    hidden = [e for e in entries if rank(e.scope) >= rank(scope) and e is not current]
    plan = _McpPlan(scope, entry, hidden=hidden)
    if current is not None:
        plan.dropped = _dropped_query_flags(current.config, entry)
    if scope != "user":
        return plan

    if current is not None and same_mcp_entry(current.config, entry):
        plan.unchanged = True
        return plan
    plan.replaces = current.config if current is not None else None
    if claude_code.claude_executable() is None:
        commands = [claude_code.mcp_add_json_args("user", _redact_entry(entry))]
        if plan.replaces is not None:
            commands.insert(0, claude_code.mcp_remove_args("user"))
        click.echo("\n  Add the user-scope entry yourself, then re-run this setup:")
        for args in commands:
            click.echo(f"    {_claude_command(args)}")
        if entry.get("headers"):
            _echo_api_key_env_note()
        raise click.ClickException(
            "The Claude Code CLI (`claude`) was not found on PATH. A user-scope entry lives "
            f"in {claude_json_label()}, which Claude Code owns, so setup writes it only "
            "through `claude mcp add-json`. Nothing was written."
        )
    if plan.replaces is None:
        return plan
    question = f"Replace the existing user-scope {MCP_SERVER_NAME} entry ({claude_json_label()})?"
    if non_interactive:
        # -y takes the prompt's default; say so, since the old entry is removed.
        click.echo(f"\n  {question} yes (-y)")
    elif not click.confirm(question, default=True):
        raise click.ClickException("Setup cancelled; nothing was written.")
    return plan


def _run_claude_or_fail(args: list[str]) -> None:
    """Run ``claude <args>``; turn any failure into a ClickException.

    The message names only the subcommand: ``args`` can hold an API key, and
    a ``SubprocessError``'s own message quotes the whole command line.
    """
    name = f"claude {' '.join(args[:2])}"
    try:
        proc = claude_code.run_claude(args)
    except subprocess.TimeoutExpired as e:
        raise click.ClickException(f"`{name}` failed: timed out after {e.timeout:g}s") from None
    except subprocess.SubprocessError as e:
        raise click.ClickException(f"`{name}` failed: {type(e).__name__}") from None
    except OSError as e:
        raise click.ClickException(f"`{name}` failed: {_exc_message(e)}") from e
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip() or f"exit code {proc.returncode}"
        raise click.ClickException(f"`{name}` failed: {detail}")


def _add_user_entry(entry: dict[str, Any]) -> None:
    """Run ``claude mcp add-json --scope user`` for ``entry``.

    The entry goes on ``claude``'s command line, which every local user can
    read in the process list while it runs, so one that holds a credential is
    never passed (:func:`~kagura_memory.claude_code.holds_credential`).

    Raises:
        click.ClickException: ``entry`` holds a credential, or ``claude`` failed.
    """
    if claude_code.holds_credential(entry):
        raise click.ClickException(
            "the entry holds an API key, which setup never passes on a command line"
        )
    _run_claude_or_fail(claude_code.mcp_add_json_args("user", entry))


def _write_mcp_entry(project: Path, plan: _McpPlan) -> str:
    """Write the planned entry; return a line saying what was done."""
    if plan.scope == "project":
        path = _write_project_mcp_entry(project, plan.entry)
        return f"Wrote {path.relative_to(project)}"
    if plan.unchanged:
        return f"User-scope {MCP_SERVER_NAME} entry already up to date ({claude_json_label()})"
    if plan.replaces is None:
        _add_user_entry(plan.entry)
    else:
        _run_claude_or_fail(claude_code.mcp_remove_args("user"))
        try:
            _add_user_entry(plan.entry)
        except click.ClickException as failed:
            _restore_user_entry(plan.replaces, failed)
            raise
    return f"Added {MCP_SERVER_NAME} at user scope (claude mcp add-json --scope user)"


def _echo_api_key_env_note(api_key: str | None = None) -> None:
    """Say where a user-scope API-key entry gets the key; never prints it.

    With ``api_key``, also say whether this shell already has it in
    :data:`~kagura_memory.claude_code.MCP_API_KEY_ENV`.
    """
    var = MCP_API_KEY_ENV
    click.echo(
        f"  The entry sends the API key from ${var}, which Claude Code reads when it\n"
        "  connects, so the key is neither in the entry nor on a command line. Set it in\n"
        f"  the environment that starts Claude Code, e.g. `export {var}=<your-api-key>`\n"
        "  in your shell profile. (--profile writes an entry that needs no key at all.)"
    )
    if api_key is None:
        return
    current = os.environ.get(var)
    if not current:
        click.echo(f"  ${var} is not set in this shell.")
    elif current == api_key:
        click.echo(f"  ${var} is already set to this key in this shell.")
    else:
        click.echo(
            f"  Warning: ${var} in this shell holds a different key, which Claude Code\n"
            "  started from here would send."
        )


def _restore_user_entry(old: dict[str, Any], failed: click.ClickException) -> None:
    """Put back the user-scope entry removed before a failed add, rather than leave none.

    An entry that holds an API key is not put back: that would pass the key on
    ``claude``'s command line.

    Raises:
        click.ClickException: The restore failed too, or was not attempted. The
            message keeps the add's error and says the old entry is gone; the
            command to re-add it is printed, with a baked API key replaced by a
            placeholder.
    """
    try:
        _add_user_entry(old)
    except click.ClickException as e:
        click.echo(f"\n  Re-add the previous user-scope {MCP_SERVER_NAME} entry yourself:")
        click.echo(
            f"    {_claude_command(claude_code.mcp_add_json_args('user', _redact_entry(old)))}"
        )
        raise click.ClickException(
            f"{failed.message}\nThe previous user-scope {MCP_SERVER_NAME} entry was removed "
            f"and could not be restored ({e.message}); the command above re-adds it."
        ) from None


def _echo_plan_notes(plan: _McpPlan) -> None:
    """After the write: the weaker entries the new one hides, and the settings it dropped."""
    for e in plan.hidden:
        click.echo(
            f"  Note: in this project it hides the {MCP_SERVER_NAME} entry in {e.scope} "
            f"scope ({e.source}); editing that entry has no effect here."
        )
    if plan.dropped:
        them = "them" if len(plan.dropped) > 1 else "it"
        click.echo(
            f"  Note: the previous {plan.scope}-scope entry also had "
            f"{' and '.join(plan.dropped)}, which this run left out; re-run with {them} "
            f"to keep {them}."
        )


# =============================================================================
# Hooks and commands, and the kagura-memory plugin overlap (#258)
# =============================================================================


@dataclass
class _Extras:
    """Which SDK hooks and commands to install; the rest are removed if the SDK wrote them."""

    session_hook: bool = True
    sync_hook: bool = True
    commands: bool = True
    #: Items turned off whose removal the user declined: still on disk, as they were.
    kept: list[str] = field(default_factory=list)

    def summary(self) -> list[str]:
        lines = []
        if self.session_hook:
            lines.append("  - Recall trusted project memories at session start")
        if self.sync_hook:
            lines.append("  - Sync .claude/memory/ writes to Kagura")
        if self.commands:
            lines.append("  - /kagura-recall and /kagura-remember available as skills")
        lines += [f"  - Kept as it was (removal declined): the SDK's {item}" for item in self.kept]
        return lines


def _resolve_extras(
    project: Path,
    *,
    session_hook: bool | None,
    sync_hook: bool,
    commands: bool | None,
    non_interactive: bool,
) -> tuple[_Extras, str | None]:
    """Decide which hooks and commands to install; return them and the plugin id, if any.

    ``None`` means the flag was not given (on by default). When the
    memory-cloud ``kagura-memory`` plugin is installed, an interactive run asks
    about each of the two that were not given: skip ``/kagura-recall`` and
    ``/kagura-remember``, which duplicate the plugin's ``:recall`` and
    ``:remember`` (default yes), and skip the SessionStart recall hook (default
    no: the plugin recalls nothing automatically, its SessionStart hook only
    announces active guardrails). ``-y`` installs them as before and names the
    flag for the commands. The sync hook has no plugin counterpart and is never
    offered. Detection never fails setup.
    """
    plugin_id = claude_code.detect_kagura_plugin(project)
    if plugin_id is not None and non_interactive:
        if commands is None:
            click.echo(
                f"\n  Detected the {plugin_id} plugin: installing /kagura-recall and "
                "/kagura-remember anyway, which duplicate its :recall and :remember; pass "
                "--no-commands to skip them."
            )
    elif plugin_id is not None and (commands is None or session_hook is None):
        click.echo(f"\n  Detected the {plugin_id} Claude Code plugin.")
        if commands is None:
            click.echo(
                "  /kagura-recall and /kagura-remember duplicate its /kagura-memory:recall and\n"
                "  :remember (resolving the context and profile differently)."
            )
            commands = not click.confirm(
                "Skip the SDK's /kagura-recall and /kagura-remember?", default=True
            )
        if session_hook is None:
            click.echo(
                "  The SDK's SessionStart hook recalls trusted project memories in every\n"
                "  session. The plugin has no automatic recall: its SessionStart hook only\n"
                "  announces active tool guardrails, and /kagura-memory:session-start is a\n"
                "  command you run yourself."
            )
            session_hook = not click.confirm(
                "Skip the SDK's SessionStart recall hook anyway?", default=False
            )
    extras = _Extras(
        session_hook=session_hook is not False, sync_hook=sync_hook, commands=commands is not False
    )
    return extras, plugin_id


def _plugin_server_url(upstream_url: str) -> str:
    """The plugin's ``server_url``: the MCP URL with only ``guardrails=off`` kept in its query."""
    parts = urlsplit(upstream_url)
    values = [v for k, v in parse_qsl(parts.query, keep_blank_values=True) if k == "guardrails"]
    query = "guardrails=off" if values and values[0].strip().lower() == "off" else ""
    return urlunsplit(parts._replace(query=query, fragment=""))


def _echo_plugin_notes(
    plugin_id: str, upstream_url: str, context_id: str, guardrails: str | None
) -> None:
    """Coexistence notes for the plugin: never sets --guardrails, never prints a key."""
    click.echo(
        f"\n  {plugin_id} delivers tool guardrails through its own hooks once you\n"
        "  configure them (/kagura-memory:setup)."
    )
    if guardrails != "off":
        click.echo(
            "  Then re-run this setup with --guardrails off so the server does not also send a\n"
            "  guardrail digest. 'off' also removes the guardrails block from get_context_info,\n"
            "  so set it only once the hooks deliver guardrails."
        )
    click.echo("  Plugin settings (/plugin > kagura-memory > Configure):")
    click.echo(f"    server_url  {_plugin_server_url(upstream_url)}")
    click.echo(
        f"    context_id  {context_id}\n"
        "                The plugin has ONE guardrail context for every project: use this\n"
        "                one only if it holds the guardrails you want everywhere."
    )
    click.echo(
        "    api_key     Enter it yourself: a user API key (kagura_...). The plugin's hooks\n"
        "                authenticate only with one, even when this setup uses --profile."
    )


def _sdk_prefix(template: str) -> str:
    """The fixed start of an SDK-written hook command or command file.

    Everything before the context id; unchanged since the SDK first wrote
    them (#49), so it matches every SDK-written version. Hook install and
    removal, and command-file removal, match on it rather than the loose
    :data:`KAGURA_HOOK_MARKER`, so a user's own ``kagura ...`` hook is never
    replaced or removed, nor a hand-written command file removed.
    """
    return template.split("{context_id}", 1)[0]


def _install_hooks(
    project_dir: Path, context_id: str, *, session: bool = True, sync: bool = True
) -> Path:
    """Install hooks to .claude/settings.json, preserving existing hooks."""
    claude_dir = project_dir / ".claude"
    claude_dir.mkdir(exist_ok=True)
    path = claude_dir / "settings.json"

    existing = _read_json_safe(path)
    hooks = existing.setdefault("hooks", {})

    if session:
        session_start_list = hooks.setdefault("SessionStart", [])
        kagura_session_hook = {
            "type": "command",
            "command": SESSIONSTART_HOOK_COMMAND.format(context_id=context_id),
        }
        _upsert_hook_entry(session_start_list, kagura_session_hook, SESSIONSTART_HOOK_COMMAND)

    if sync:
        post_tool_list = hooks.setdefault("PostToolUse", [])
        kagura_post_hook = {
            "type": "command",
            "command": POSTTOOLUSE_HOOK_COMMAND.format(context_id=context_id),
        }
        _upsert_hook_entry(
            post_tool_list, kagura_post_hook, POSTTOOLUSE_HOOK_COMMAND, matcher="Write|Edit"
        )

    _write_json(path, existing)
    return path


def _upsert_hook_entry(
    hook_list: list[dict[str, Any]],
    new_hook: dict[str, Any],
    template: str,
    matcher: str | None = None,
) -> None:
    """Insert the SDK hook written from ``template``, or update the one already there.

    The existing hook is found by :func:`_sdk_prefix`, the same test removal
    uses, so a user's own ``kagura ...`` hook is left alone.
    """
    prefix = _sdk_prefix(template)
    for entry in hook_list:
        entry_hooks = entry.get("hooks", [])
        for i, h in enumerate(entry_hooks):
            if isinstance(h, dict) and str(h.get("command", "")).startswith(prefix):
                entry_hooks[i] = new_hook
                if matcher:
                    entry["matcher"] = matcher
                return

    entry: dict[str, Any] = {"hooks": [new_hook]}
    if matcher:
        entry["matcher"] = matcher
    hook_list.append(entry)


def _remove_sdk_hook(settings: dict[str, Any], event: str, template: str) -> bool:
    """Drop the SDK hook written from ``template`` under ``event``; True if one was found.

    An entry left without hooks is dropped, and so is an emptied event list.
    """
    hooks = settings.get("hooks")
    if not isinstance(hooks, dict) or not isinstance(hooks.get(event), list):
        return False
    entries: list[Any] = hooks[event]
    prefix = _sdk_prefix(template)
    removed = False
    for entry in list(entries):
        entry_hooks = entry.get("hooks") if isinstance(entry, dict) else None
        if not isinstance(entry_hooks, list):
            continue
        kept = [
            h
            for h in entry_hooks
            if not (isinstance(h, dict) and str(h.get("command", "")).startswith(prefix))
        ]
        if len(kept) == len(entry_hooks):
            continue
        removed = True
        if kept:
            entry["hooks"] = kept
        else:
            entries.remove(entry)
    if removed and not entries:
        del hooks[event]
    return removed


def _remove_disabled_hooks(project: Path, extras: _Extras, non_interactive: bool) -> list[str]:
    """Remove the SDK hooks ``extras`` turns off, asking first unless ``-y``.

    Returns:
        The hooks the user chose to keep (empty unless removal was declined).
    """
    path = project / ".claude" / "settings.json"
    if not path.exists():
        return []
    settings = _read_json_safe(path)
    targets = [
        (
            extras.session_hook,
            "SessionStart recall hook",
            "SessionStart",
            SESSIONSTART_HOOK_COMMAND,
        ),
        (extras.sync_hook, ".claude/memory sync hook", "PostToolUse", POSTTOOLUSE_HOOK_COMMAND),
    ]
    removed = [
        label
        for enabled, label, event, template in targets
        if not enabled and _remove_sdk_hook(settings, event, template)
    ]
    if not removed:
        return []
    what = " and ".join(removed)
    where = path.relative_to(project)
    if non_interactive or click.confirm(f"Remove the SDK's {what} from {where}?", default=True):
        _write_json(path, settings)
        click.echo(f"  Removed the {what} from {where}")
        return []
    return removed


_SKILL_FILES = (("kagura-recall.md", SKILL_RECALL), ("kagura-remember.md", SKILL_REMEMBER))


def _install_skills(project_dir: Path, context_id: str) -> list[Path]:
    """Install kagura skills to .claude/commands/."""
    commands_dir = project_dir / ".claude" / "commands"
    commands_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for filename, template in _SKILL_FILES:
        path = commands_dir / filename
        path.write_text(template.format(context_id=context_id), encoding="utf-8")
        paths.append(path)

    return paths


def _remove_sdk_skills(project: Path, non_interactive: bool) -> list[str]:
    """Remove the SDK-written command files, asking first unless ``-y``.

    Returns:
        The command files the user chose to keep (empty unless removal was
        declined).
    """
    commands_dir = project / ".claude" / "commands"
    paths = []
    for filename, template in _SKILL_FILES:
        path = commands_dir / filename
        try:
            if path.read_text(encoding="utf-8").startswith(_sdk_prefix(template)):
                paths.append(path)
        except (OSError, ValueError):
            continue
    if not paths:
        return []
    names = ", ".join(f"/{p.stem}" for p in paths)
    if non_interactive or click.confirm(
        f"Remove the SDK's {names} command files (.claude/commands/)?", default=True
    ):
        for path in paths:
            path.unlink()
            click.echo(f"  Removed {path.relative_to(project)}")
        return []
    return [f"{names} command files"]


def _apply_extras(project: Path, context_id: str, extras: _Extras, non_interactive: bool) -> None:
    """Install the enabled hooks and commands and remove the disabled SDK-written ones.

    Items whose removal the user declines are recorded in ``extras.kept``.
    """
    if extras.session_hook or extras.sync_hook:
        hooks_path = _install_hooks(
            project, context_id, session=extras.session_hook, sync=extras.sync_hook
        )
        click.echo(f"  Wrote {hooks_path.relative_to(project)}")
    extras.kept += _remove_disabled_hooks(project, extras, non_interactive)

    if extras.commands:
        for p in _install_skills(project, context_id):
            click.echo(f"  Wrote {p.relative_to(project)}")
    else:
        extras.kept += _remove_sdk_skills(project, non_interactive)


def _check_gitignore(
    project_dir: Path, secret_files: tuple[str, ...] = (".kagura.json", ".mcp.json")
) -> list[str]:
    """Check which secret files are missing from .gitignore."""
    try:
        content = (project_dir / ".gitignore").read_text(encoding="utf-8")
    except FileNotFoundError:
        return list(secret_files)
    return [f for f in secret_files if f not in content]


def _warn_tools_allowlist(mcp_url: str, tool_profile: str | None) -> None:
    """Warn that ``--tool-profile`` has no effect on a URL with a ``?tools=`` allowlist."""
    if tool_profile is not None and mcp_url_has_tools_allowlist(mcp_url):
        click.echo(
            "\n  Warning: the MCP URL has a ?tools= allowlist, which the server applies\n"
            f"  instead of --tool-profile {tool_profile}."
        )


def _validate_context_id(context_id: str) -> None:
    """Reject a context_id that is unsafe to interpolate into the hook commands."""
    if not re.match(_CONTEXT_ID_PATTERN, context_id):
        raise click.ClickException(
            f"Invalid context_id: {context_id!r} — "
            "must be alphanumeric, hyphens, or underscores only"
        )


def run_setup_claude(
    api_key: str | None,
    mcp_url: str | None,
    context_id: str | None,
    project_dir: str,
    non_interactive: bool,
    no_auto_context: bool = False,
    profile: str | None = None,
    *,
    scope: McpScope = "project",
    guardrails: str | None = None,
    tool_profile: str | None = None,
    session_hook: bool | None = None,
    sync_hook: bool = True,
    commands: bool | None = None,
) -> None:
    """Run the full setup flow for Claude Code integration.

    Two mutually exclusive auth paths:

    * ``profile`` set → OAuth path: write the refresh-aware ``kagura-mcp``
      stdio entry bound to the named OAuth profile (no API key).
    * ``profile`` unset → API-key path: write the static-token http entry
      (at user scope it names ``$KAGURA_MCP_API_KEY`` instead of the key).

    Both write the entry at ``scope`` (``project``: ``<project>/.mcp.json``;
    ``user``: through ``claude mcp add-json --scope user``), with
    ``guardrails`` / ``tool_profile`` on the upstream URL, then the project's
    ``.kagura.json`` and the hooks and commands that ``session_hook`` /
    ``sync_hook`` / ``commands`` select (``None``: not given, on by default).
    ``guardrails`` must already be normalized (``off`` or a canonical UUID).
    """
    if profile is not None:
        if api_key is not None:
            raise click.UsageError(
                "--profile (OAuth) and --api-key (static token) are mutually exclusive; pick one."
            )
        _run_setup_claude_oauth(
            profile=profile,
            context_id=context_id,
            project_dir=project_dir,
            non_interactive=non_interactive,
            no_auto_context=no_auto_context,
            scope=scope,
            guardrails=guardrails,
            tool_profile=tool_profile,
            session_hook=session_hook,
            sync_hook=sync_hook,
            commands=commands,
        )
        return

    project = Path(project_dir).resolve()

    # Load existing config from project dir (not cwd). _read_json_safe pins
    # UTF-8 and swallows missing/foreign-encoding/corrupt files (issue #197).
    existing_config = _read_json_safe(project / ".kagura.json")

    # 1. API Key
    resolved_api_key = api_key or existing_config.get("api_key")
    resolved_api_key = _prompt_api_key(resolved_api_key, non_interactive)

    # 2. MCP URL
    resolved_mcp_url = mcp_url or existing_config.get("mcp_url")
    resolved_mcp_url = _prompt_mcp_url(resolved_mcp_url, non_interactive)

    # 3. Where the entry goes — settled before any request or write
    _warn_tools_allowlist(resolved_mcp_url, tool_profile)
    upstream_url = mcp_url_with_query(
        resolved_mcp_url, guardrails=guardrails, tool_profile=tool_profile
    )
    # A user-scope entry goes on `claude mcp add-json`'s command line, which other
    # local users can read in the process list, so it names the variable Claude
    # Code reads the key from instead of the key.
    token = resolved_api_key if scope == "project" else _API_KEY_REF
    plan = _plan_mcp_entry(
        project, scope, _static_token_entry(token, upstream_url), non_interactive
    )

    # 4. Test connection
    click.echo("\nVerifying connection...")
    try:
        contexts_response = asyncio.run(_test_connection(resolved_api_key, resolved_mcp_url))
    except KaguraAuthError as e:
        raise click.ClickException(
            f"Authentication failed: {_exc_message(e)}\n"
            "  Check your API key at: Kagura Web UI > Integrations > API Keys"
        ) from e
    except KaguraConnectionError as e:
        raise click.ClickException(
            f"Cannot connect to {resolved_mcp_url}: {_exc_message(e)}\n"
            "  Is the server running? Try: docker compose up -d"
        ) from e
    except Exception as e:
        raise click.ClickException(f"Connection failed: {_exc_message(e)}") from e

    count = contexts_response.get("count", 0)
    click.echo(f"  Connected! ({count} contexts available)")

    # 5. Context selection (reuse existing if not overridden)
    effective_context_id = context_id or existing_config.get("context_id")
    resolved_context_id = _select_or_create_context(
        contexts_response,
        resolved_api_key,
        resolved_mcp_url,
        effective_context_id,
        project,
        non_interactive,
        no_auto_context,
    )

    # 6. Validate context_id before interpolating into shell commands
    _validate_context_id(resolved_context_id)

    # 7. Hooks and commands (may ask about the kagura-memory plugin overlap)
    extras, plugin_id = _resolve_extras(
        project,
        session_hook=session_hook,
        sync_hook=sync_hook,
        commands=commands,
        non_interactive=non_interactive,
    )

    # 8. Write files
    click.echo("\nSetting up Kagura Memory for Claude Code...")

    kagura_path = _write_kagura_config(
        project, resolved_api_key, resolved_mcp_url, resolved_context_id
    )
    click.echo(f"  Wrote {kagura_path.relative_to(project)}")

    click.echo(f"  {_write_mcp_entry(project, plan)}")
    if scope == "user":
        _echo_api_key_env_note(resolved_api_key)
    _echo_plan_notes(plan)
    _apply_extras(project, resolved_context_id, extras, non_interactive)

    if plugin_id is not None:
        _echo_plugin_notes(plugin_id, upstream_url, resolved_context_id, guardrails)

    # 9. Gitignore warning (a user-scope entry keeps the key out of .mcp.json)
    secret_files = (".kagura.json", ".mcp.json") if scope == "project" else (".kagura.json",)
    missing = _check_gitignore(project, secret_files)
    if missing:
        click.echo("\n  Warning: these files contain secrets (API key):")
        for f in missing:
            click.echo(f"  Add to .gitignore: {f}")

    # 10. Summary — only what was installed
    click.echo(
        "\nSetup complete! Claude Code will now:\n"
        f"  - Connect to Kagura Memory ({MCP_SERVER_NAME}, {scope} scope)"
    )
    for line in extras.summary():
        click.echo(line)


def _write_kagura_config_oauth(project_dir: Path, mcp_url: str, context_id: str) -> Path:
    """Write .kagura.json for the OAuth path (no api_key), merging with existing.

    Unlike :func:`_write_kagura_config`, this never writes an ``api_key`` — the
    hooks and skills run ``kagura recall`` / ``kagura remember``, which resolve
    the OAuth profile from ``~/.kagura/credentials.json`` automatically. Any
    pre-existing ``api_key`` in the file is left untouched (the OAuth profile
    still wins in the credential-resolution order).
    """
    path = project_dir / ".kagura.json"
    existing = _read_json_safe(path)
    existing["mcp_url"] = mcp_url
    existing["context_id"] = context_id
    _write_json(path, existing)
    return path


def _run_setup_claude_oauth(
    *,
    profile: str,
    context_id: str | None,
    project_dir: str,
    non_interactive: bool,
    no_auto_context: bool,
    scope: McpScope = "project",
    guardrails: str | None = None,
    tool_profile: str | None = None,
    session_hook: bool | None = None,
    sync_hook: bool = True,
    commands: bool | None = None,
) -> None:
    """Setup flow for the refresh-aware ``kagura-mcp`` (OAuth) integration.

    Resolves auth and the MCP URL from the named OAuth profile in
    ``~/.kagura/credentials.json`` (created by ``kagura auth login``), then
    writes the stdio entry so Claude Code launches ``kagura-mcp`` as the MCP
    server with an always-fresh bearer token.
    """
    from .auth.credentials import load_credentials_file

    project = Path(project_dir).resolve()

    cf = load_credentials_file()
    creds = cf.get_profile(profile)
    if creds is None:
        raise click.ClickException(
            f"No OAuth profile '{profile}' in ~/.kagura/credentials.json.\n"
            f"  Run: kagura auth login --profile {profile}"
        )

    # $PATH check is a warning, never a hard failure: kagura-mcp is a
    # console_script that resolves inside its own venv even when that venv is
    # not on the invoking shell's PATH (a common Claude Code launch setup).
    if not _kagura_mcp_on_path():
        click.echo(
            f"\n  Warning: '{MCP_PROXY_COMMAND}' was not found on $PATH.\n"
            "  Claude Code launches it as the MCP server command — make sure the\n"
            "  environment that starts Claude Code has the kagura-memory package\n"
            "  installed (pip install kagura-memory)."
        )

    _warn_tools_allowlist(creds.mcp_url, tool_profile)
    entry = _stdio_entry(profile, guardrails=guardrails, tool_profile=tool_profile)
    plan = _plan_mcp_entry(project, scope, entry, non_interactive)

    # Verify the profile works and list contexts (auth + URL come from profile)
    click.echo("\nVerifying connection...")
    try:
        contexts_response = asyncio.run(_test_connection(profile=profile))
    except KaguraAuthError as e:
        raise click.ClickException(
            f"Authentication failed: {_exc_message(e)}\n"
            f"  Your token may have expired — re-run: kagura auth login --profile {profile}"
        ) from e
    except KaguraConnectionError as e:
        raise click.ClickException(f"Cannot connect to {creds.server}: {_exc_message(e)}") from e
    except Exception as e:
        raise click.ClickException(f"Connection failed: {_exc_message(e)}") from e

    count = contexts_response.get("count", 0)
    click.echo(f"  Connected as {creds.user_email or '<unknown>'} ({count} contexts available)")

    resolved_context_id = _select_or_create_context(
        contexts_response,
        None,
        creds.mcp_url,
        context_id,
        project,
        non_interactive,
        no_auto_context,
        profile=profile,
    )

    _validate_context_id(resolved_context_id)

    extras, plugin_id = _resolve_extras(
        project,
        session_hook=session_hook,
        sync_hook=sync_hook,
        commands=commands,
        non_interactive=non_interactive,
    )

    click.echo("\nSetting up Kagura Memory for Claude Code (OAuth via kagura-mcp)...")

    kagura_path = _write_kagura_config_oauth(project, creds.mcp_url, resolved_context_id)
    click.echo(f"  Wrote {kagura_path.relative_to(project)}")

    server_command = shlex.join([MCP_PROXY_COMMAND, *entry["args"]])
    click.echo(f"  {_write_mcp_entry(project, plan)} (stdio: {server_command})")
    _echo_plan_notes(plan)
    _apply_extras(project, resolved_context_id, extras, non_interactive)

    # The installed hooks shell out to `kagura recall` / `kagura remember`,
    # which resolve the *default* OAuth profile (they take no --profile flag).
    # When the chosen profile is not the file's default_profile, those hooks
    # would sync under the wrong account — warn rather than silently desync.
    # (The kagura-mcp MCP server itself always uses the named profile.)
    if (extras.session_hook or extras.sync_hook) and profile != cf.default_profile:
        click.echo(
            f"\n  Warning: the session/PostToolUse hooks run `kagura recall` / "
            f"`kagura remember`,\n"
            f"  which use the DEFAULT profile '{cf.default_profile}', not '{profile}'.\n"
            f"  To make them use '{profile}', set KAGURA_PROFILE={profile} in the\n"
            f"  environment that runs Claude Code (the MCP server already uses it)."
        )

    if plugin_id is not None:
        upstream_url = mcp_url_with_query(
            creds.mcp_url, guardrails=guardrails, tool_profile=tool_profile
        )
        _echo_plugin_notes(plugin_id, upstream_url, resolved_context_id, guardrails)

    # Note: the stdio entry carries NO secret (the proxy injects a fresh
    # token per request), so the API-key gitignore warning does not apply here.
    click.echo(
        "\nSetup complete! Claude Code will use the refresh-aware kagura-mcp proxy:\n"
        f"  - MCP server: {server_command} ({scope} scope; "
        "auto-refreshes the OAuth token — no more silent 401s)"
    )
    for line in extras.summary():
        click.echo(line)
