"""Setup Kagura Memory integration for Claude Code."""

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import click

from .client import KaguraClient
from .exceptions import KaguraAuthError, KaguraConnectionError, _exc_message

DEFAULT_MCP_URL = "http://localhost:8080/mcp"

# Trailing space distinguishes "kagura recall/remember" from "kagura-memory" in MCP config
KAGURA_HOOK_MARKER = "kagura "

_CONTEXT_ID_PATTERN = r"^[a-zA-Z0-9_-]+$"

SESSIONSTART_HOOK_COMMAND = (
    'kagura recall "project context and recent decisions" '
    "-c {context_id} -k 5 2>/dev/null | head -c 2000 || true"
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


def _read_json_safe(path: Path) -> dict[str, Any]:
    """Read a JSON file, returning {} on missing file or parse error."""
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Write a dict as formatted JSON."""
    path.write_text(json.dumps(data, indent=2) + "\n")


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


async def _test_connection(api_key: str, mcp_url: str) -> dict[str, Any]:
    """Test connection and return contexts list."""
    client = KaguraClient(api_key=api_key, mcp_url=mcp_url)
    async with client:
        return await client.list_contexts()


async def _create_context(
    api_key: str, mcp_url: str, name: str, summary: str | None
) -> dict[str, Any]:
    """Create a new context."""
    client = KaguraClient(api_key=api_key, mcp_url=mcp_url)
    async with client:
        return await client.create_context(name=name, summary=summary)


def _auto_match_context(
    contexts: list[dict[str, Any]], project_dir: Path, threshold: float = 0.65
) -> dict[str, Any] | None:
    """Return the context whose name best matches the project dir name,
    or None if no candidate clears the threshold."""
    from difflib import SequenceMatcher

    target = project_dir.name.lower().replace("_", "-")
    best: tuple[float, dict[str, Any]] | None = None
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
    if best and best[0] >= threshold:
        return best[1]
    return None


def _select_or_create_context(
    contexts_response: dict[str, Any],
    api_key: str,
    mcp_url: str,
    preselected: str | None,
    project_dir: Path,
    non_interactive: bool,
    no_auto_context: bool = False,
) -> str:
    """Select existing context or create a new one. Returns context_id."""
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
        result = asyncio.run(_create_context(api_key, mcp_url, default_name, None))
        ctx_id = result.get("context_id") or result.get("id", "")
        click.echo(f"  Created context: {default_name} ({ctx_id[:8]}...)")
        return ctx_id

    if contexts and not no_auto_context:
        auto = _auto_match_context(contexts, project_dir)
        if auto:
            auto_id = auto.get("id", "")
            click.echo(f"\nAuto-selected context: {auto.get('name', '?')} ({auto_id[:8]}...)")
            click.echo("  (use --no-auto-context to disable, or pick a different one below)")
            if click.confirm("Use this context?", default=True):
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
    result = asyncio.run(_create_context(api_key, mcp_url, name, summary if summary else None))
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


def _write_mcp_json(project_dir: Path, api_key: str, mcp_url: str) -> Path:
    """Write .mcp.json for Claude Code MCP server config, merging with existing."""
    path = project_dir / ".mcp.json"
    existing = _read_json_safe(path)
    servers = existing.setdefault("mcpServers", {})
    servers["kagura-memory"] = {
        "type": "url",
        "url": mcp_url,
        "headers": {"Authorization": f"Bearer {api_key}"},
    }
    _write_json(path, existing)
    return path


def _is_kagura_hook(hook: dict[str, Any]) -> bool:
    """Check if a hook entry is Kagura-managed."""
    cmd = hook.get("command", "")
    return KAGURA_HOOK_MARKER in cmd


def _install_hooks(project_dir: Path, context_id: str) -> Path:
    """Install hooks to .claude/settings.json, preserving existing hooks."""
    claude_dir = project_dir / ".claude"
    claude_dir.mkdir(exist_ok=True)
    path = claude_dir / "settings.json"

    existing = _read_json_safe(path)
    hooks = existing.setdefault("hooks", {})

    session_start_list = hooks.setdefault("SessionStart", [])
    kagura_session_hook = {
        "type": "command",
        "command": SESSIONSTART_HOOK_COMMAND.format(context_id=context_id),
    }
    _upsert_hook_entry(session_start_list, kagura_session_hook)

    post_tool_list = hooks.setdefault("PostToolUse", [])
    kagura_post_hook = {
        "type": "command",
        "command": POSTTOOLUSE_HOOK_COMMAND.format(context_id=context_id),
    }
    _upsert_hook_entry(post_tool_list, kagura_post_hook, matcher="Write|Edit")

    _write_json(path, existing)
    return path


def _upsert_hook_entry(
    hook_list: list[dict[str, Any]],
    new_hook: dict[str, Any],
    matcher: str | None = None,
) -> None:
    """Insert or update a Kagura hook in a hook event list."""
    for entry in hook_list:
        entry_hooks = entry.get("hooks", [])
        for i, h in enumerate(entry_hooks):
            if _is_kagura_hook(h):
                entry_hooks[i] = new_hook
                if matcher:
                    entry["matcher"] = matcher
                return

    entry: dict[str, Any] = {"hooks": [new_hook]}
    if matcher:
        entry["matcher"] = matcher
    hook_list.append(entry)


def _install_skills(project_dir: Path, context_id: str) -> list[Path]:
    """Install kagura skills to .claude/commands/."""
    commands_dir = project_dir / ".claude" / "commands"
    commands_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for filename, template in [
        ("kagura-recall.md", SKILL_RECALL),
        ("kagura-remember.md", SKILL_REMEMBER),
    ]:
        path = commands_dir / filename
        path.write_text(template.format(context_id=context_id))
        paths.append(path)

    return paths


def _check_gitignore(project_dir: Path) -> list[str]:
    """Check which secret files are missing from .gitignore."""
    secret_files = [".kagura.json", ".mcp.json"]
    try:
        content = (project_dir / ".gitignore").read_text()
    except FileNotFoundError:
        return secret_files
    return [f for f in secret_files if f not in content]


def run_setup_claude(
    api_key: str | None,
    mcp_url: str | None,
    context_id: str | None,
    project_dir: str,
    non_interactive: bool,
    no_auto_context: bool = False,
) -> None:
    """Run the full setup flow for Claude Code integration."""
    project = Path(project_dir).resolve()

    # Load existing config from project dir (not cwd)
    existing_config: dict[str, Any] = {}
    project_config = project / ".kagura.json"
    if project_config.exists():
        try:
            existing_config = json.loads(project_config.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    # 1. API Key
    resolved_api_key = api_key or existing_config.get("api_key")
    resolved_api_key = _prompt_api_key(resolved_api_key, non_interactive)

    # 2. MCP URL
    resolved_mcp_url = mcp_url or existing_config.get("mcp_url")
    resolved_mcp_url = _prompt_mcp_url(resolved_mcp_url, non_interactive)

    # 3. Test connection
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

    # 4. Context selection (reuse existing if not overridden)
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

    # 5. Validate context_id before interpolating into shell commands
    if not re.match(_CONTEXT_ID_PATTERN, resolved_context_id):
        raise click.ClickException(
            f"Invalid context_id: {resolved_context_id!r} — "
            "must be alphanumeric, hyphens, or underscores only"
        )

    # 6. Write files
    click.echo("\nSetting up Kagura Memory for Claude Code...")

    kagura_path = _write_kagura_config(
        project, resolved_api_key, resolved_mcp_url, resolved_context_id
    )
    click.echo(f"  Wrote {kagura_path.relative_to(project)}")

    mcp_path = _write_mcp_json(project, resolved_api_key, resolved_mcp_url)
    click.echo(f"  Wrote {mcp_path.relative_to(project)}")

    hooks_path = _install_hooks(project, resolved_context_id)
    click.echo(f"  Wrote {hooks_path.relative_to(project)}")

    skill_paths = _install_skills(project, resolved_context_id)
    for p in skill_paths:
        click.echo(f"  Wrote {p.relative_to(project)}")

    # 7. Gitignore warning
    missing = _check_gitignore(project)
    if missing:
        click.echo("\n  Warning: these files contain secrets (API key):")
        for f in missing:
            click.echo(f"  Add to .gitignore: {f}")

    # 8. Summary
    click.echo(
        "\nSetup complete! Claude Code will now:\n"
        "  - Recall relevant memories at session start\n"
        "  - Sync .claude/memory/ writes to Kagura\n"
        "  - /kagura-recall and /kagura-remember available as skills"
    )
