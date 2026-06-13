"""CLI for Kagura Memory SDK."""

import asyncio
import json
import sys
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import click

from ._auth import (
    _SOURCE_LABEL,
    _OAuthAuth,
    _resolve_auth,
    _StaticAuth,
)
from .agent import KaguraAgent
from .auth.cli import auth as _auth_group
from .client import KaguraClient
from .config import load_config
from .doctor import run_doctor
from .exceptions import KaguraError, _exc_message
from .files_client import FilesClient
from .logger import VerboseLogger
from .models import FileObject, Message, ProcessResult, ResourceEventRequest, Session
from .resource_client import ResourceClient
from .setup_claude import run_setup_claude

_PROGRESS_CHOICES = ["rich", "json", "none"]


def _resolve_progress_logger(verbose: int, progress: str | None) -> VerboseLogger | None:
    """Resolve ``-v`` / ``--progress`` CLI flags to a ``VerboseLogger`` (or None).

    Issue #108 precedence rule (canonical):

    1. ``--progress`` explicit wins; format is exactly its value.
    2. ``--progress`` omitted → ``rich`` if ``-v`` count ≥ 1, else ``none``.
    3. ``-v`` level controls Rich verbosity (1=actions, 2=details, 3=debug).
       For ``--progress=json``, ``-v`` is ignored — JSON path always emits all
       ``kind`` values (downstream consumers filter).
    4. ``--progress=none`` is always silent regardless of ``-v`` count.

    Returns ``None`` for the silent path so entry points can keep
    ``logger: VerboseLogger | None = None`` and normalize internally.
    """
    if progress is not None:
        progress = progress.lower()
    if progress == "none":
        return None
    if progress == "json":
        # level is irrelevant for JSON; pick the max so explicit -v isn't lost
        # if the caller later switches output_format at the logger level.
        return VerboseLogger(level=max(1, verbose), output_format="json")
    if progress == "rich":
        return VerboseLogger(level=max(1, verbose), output_format="rich")
    # progress was omitted — fall back to -v presence.
    if verbose >= 1:
        return VerboseLogger(level=verbose, output_format="rich")
    return None


# =============================================================================
# Helper Functions
# =============================================================================


def _parse_tags(tags: str | None) -> list[str] | None:
    """Parse comma-separated tags string into a list, or None if empty."""
    if not tags:
        return None
    parsed = [t.strip() for t in tags.split(",") if t.strip()]
    return parsed or None


def _run_client_command(
    operation: Callable[[KaguraClient, str], Awaitable[dict[str, Any]]],
    context_id: str | None,
    *,
    needs_context: bool = True,
) -> None:
    """
    Execute a client operation with standard boilerplate.

    Handles: config loading, context resolution, client lifecycle,
    async execution, JSON output, and error handling. ``KaguraClient``
    itself runs the full credential resolution chain (env → OAuth
    profile → .kagura.json), so commands work seamlessly after a
    ``kagura auth login`` even if .kagura.json is absent.
    """
    try:
        config = load_config()

        ctx_id = ""
        if needs_context:
            ctx_id = context_id or config.get("context_id") or ""
            if not ctx_id:
                raise click.ClickException(
                    "context_id required. Use --context-id or set in .kagura.json"
                )

        client = KaguraClient(
            api_key=config.get("api_key") or None,
            mcp_url=config.get("mcp_url") or None,
        )

        async def _run() -> dict[str, Any]:
            async with client:
                return await operation(client, ctx_id)

        result = asyncio.run(_run())
        click.echo(json.dumps(result, indent=2, ensure_ascii=False))

    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(_exc_message(e)) from e


def _force_utf8_io() -> None:
    """Pin stdout/stderr to UTF-8 so non-ASCII output cannot crash the CLI.

    On Japanese/Chinese Windows the console codec defaults to cp932/cp950/gbk,
    where status glyphs (e.g. the check mark printed by ``kagura auth refresh``)
    raised ``UnicodeEncodeError`` and aborted the command (issue #197). UTF-8
    represents every character; ``errors="replace"`` is a belt-and-braces
    backstop for any stream that still cannot be reconfigured to UTF-8.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, LookupError, OSError):
            # Stream is detached or refuses reconfiguration; leave it as-is.
            pass


@click.group()
@click.version_option(package_name="kagura-memory", prog_name="kagura")
def main():
    """Kagura Memory Cloud CLI - AI-driven memory management."""
    _force_utf8_io()


# Register the `kagura auth` sub-group (OAuth2 device-flow). Defined in
# auth/cli.py so the auth surface stays in its own module; this line is
# the wiring point.
main.add_command(_auth_group, name="auth")


@main.command()
@click.option("--message", "-m", help="Single message to process")
@click.option("--file", "-f", type=click.File("r"), help="Session JSON file")
@click.option("--deep", is_flag=True, help="Enable deep search via memory graph exploration")
@click.option("--verbose", "-v", count=True, help="Increase verbosity (repeatable: -v, -vv, -vvv)")
def process(message, file, deep, verbose):
    """
    Process session with AI analysis.

    Examples:
      kagura process -m "Remember this: FastAPI uses Depends() for DI"
      kagura process -f session.json
      echo '{"messages":[{"role":"user","content":"test"}]}' | kagura process
    """
    try:
        config = load_config()

        # Parse input
        if message:
            session = Session(messages=[Message(role="user", content=message)])
        elif file:
            session = Session(**json.load(file))
        else:
            if sys.stdin.isatty():
                raise click.UsageError(
                    "No input given. Use -m / -f, or pipe a session JSON to stdin. "
                    "Run `kagura process --help` for examples."
                )
            session = Session(**json.load(sys.stdin))

        # KaguraAgent / KaguraClient run the full credential resolution
        # chain when api_key is None — env > OAuth profile > .kagura.json.
        agent_kwargs: dict[str, Any] = {"api_key": config.get("api_key") or None}
        if config.get("mcp_url"):
            agent_kwargs["mcp_url"] = config["mcp_url"]
        if config.get("model"):
            agent_kwargs["model"] = config["model"]
        if config.get("context_id"):
            agent_kwargs["context_id"] = config["context_id"]
        if config.get("llm_api_key"):
            agent_kwargs["llm_api_key"] = config["llm_api_key"]

        async def _run_agent() -> ProcessResult:
            async with KaguraAgent(**agent_kwargs) as agent:
                return await agent.process(session, deep=deep, verbose=verbose)

        result = asyncio.run(_run_agent())
        click.echo(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))

    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(_exc_message(e)) from e


@main.group()
def config():
    """Manage configuration."""
    pass


@config.command(name="show")
def config_show():
    """Show current configuration."""
    try:
        cfg = load_config()
        # Mask API key
        if "api_key" in cfg and cfg["api_key"]:
            cfg["api_key"] = cfg["api_key"][:8] + "..." + cfg["api_key"][-4:]
        click.echo(json.dumps(cfg, indent=2, ensure_ascii=False))
    except Exception as e:
        click.echo(f"Error loading config: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option("--profile", default=None, help="OAuth profile to inspect")
@click.option("--json", "json_output", is_flag=True, help="Emit machine-readable JSON")
def doctor(profile, json_output):
    """Diagnose common setup, auth, MCP, and connectivity issues."""
    report = run_doctor(profile=profile)
    if json_output:
        click.echo(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        for check in report.checks:
            click.echo(f"{check.status.upper()} {check.message}")
    raise SystemExit(report.exit_code)


# =============================================================================
# Direct MCP Tool Commands
# =============================================================================


@main.command()
@click.option("--context-id", "-c", help="Context ID (or set in .kagura.json)")
@click.option("--summary", "-s", required=True, help="Memory summary (for search)")
@click.option("--content", required=True, help="Memory content (full text)")
@click.option("--type", "-t", "memory_type", default="note", help="Memory type")
@click.option("--importance", "-i", type=float, default=0.5, help="Importance 0.0-1.0")
@click.option("--tags", help="Comma-separated tags (e.g., 'python,fastapi')")
@click.option(
    "--source-uri",
    help="Origin URI (e.g., file:///path/to/note.md, vault://my-vault/note)",
)
@click.option(
    "--source-type",
    type=click.Choice(["file", "url", "vault", "api", "manual"], case_sensitive=False),
    default=None,
    help="Origin classification. Opt-in: omitted means no provenance is stamped.",
)
@click.option(
    "--linked-memory-ids",
    help="Comma-separated memory UUIDs to link via declared_link edges",
)
@click.option(
    "--linked-source-uris",
    help="Comma-separated source URIs to resolve to memories and link",
)
def remember(
    context_id,
    summary,
    content,
    memory_type,
    importance,
    tags,
    source_uri,
    source_type,
    linked_memory_ids,
    linked_source_uris,
):
    """
    Store a memory directly (without AI analysis).

    Examples:
      kagura remember -s "FastAPI DI pattern" --content "Use Depends()..."
      kagura remember -c dev -s "OAuth2 setup" --content "..." --tags "auth,oauth"
      kagura remember -s "Spec" --content "$(cat spec.md)" \\
        --source-uri file:///spec.md --source-type file
    """

    async def op(client: KaguraClient, ctx: str) -> dict[str, Any]:
        return await client.remember(
            context_id=ctx,
            summary=summary,
            content=content,
            type=memory_type,
            importance=importance,
            tags=_parse_tags(tags),
            source_uri=source_uri,
            source_type=source_type,
            linked_memory_ids=_parse_tags(linked_memory_ids),
            linked_source_uris=_parse_tags(linked_source_uris),
        )

    _run_client_command(op, context_id)


@main.command()
@click.argument("query")
@click.option("--context-id", "-c", help="Context ID (or set in .kagura.json)")
@click.option("-k", type=int, default=5, help="Number of results (default: 5)")
def recall(query, context_id, k):
    """
    Search memories directly (without AI analysis).

    Examples:
      kagura recall "FastAPI dependency injection"
      kagura recall "OAuth2 implementation" -k 10
      kagura recall -c dev "error handling pattern"
    """
    _run_client_command(
        lambda client, ctx: client.recall(context_id=ctx, query=query, k=k),
        context_id,
    )


@main.command()
@click.option("--context-id", "-c", help="Context ID (or set in .kagura.json)")
@click.option("--memory-id", "-m", required=True, help="Seed memory ID to explore from")
@click.option("--depth", "-d", type=int, default=2, help="Traversal depth 1-5")
@click.option("--min-weight", "-w", type=float, default=0.05, help="Min edge weight")
def explore(context_id, memory_id, depth, min_weight):
    """
    Explore related memories via Neural Memory graph.

    Examples:
      kagura explore -m "abc-123-def"
      kagura explore -c dev -m "abc-123" --depth 3
    """
    _run_client_command(
        lambda client, ctx: client.explore(
            context_id=ctx, memory_id=memory_id, depth=depth, min_weight=min_weight
        ),
        context_id,
    )


@main.command(name="ingest")
@click.argument("source")
@click.option("--context-id", "-c", help="Context ID (or set in .kagura.json)")
@click.option(
    "--text-provider",
    type=click.Choice(["claude", "gemini", "ollama"], case_sensitive=False),
    default="claude",
    show_default=True,
    help="LLM provider for section/overview summarization.",
)
@click.option(
    "--vision-provider",
    type=click.Choice(["claude", "gemini", "ollama"], case_sensitive=False),
    default="gemini",
    show_default=True,
    help=(
        "Vision LLM provider used when image content is extracted "
        "(Phase 2+). Phase 1 extractors do not yet emit images, so this "
        "setting is currently a no-op. Pass --no-vision to skip image "
        "handling configuration entirely."
    ),
)
@click.option(
    "--no-vision",
    is_flag=True,
    default=False,
    help="Skip image content (no image bytes sent to any provider).",
)
@click.option(
    "--no-archive",
    is_flag=True,
    default=False,
    help=(
        "Skip uploading the original source bytes to the workspace's "
        "object store. By default the source is archived to R2 and the "
        "resulting file_id is stamped on the overview memory's details "
        "so callers can resolve memory → original bytes later."
    ),
)
@click.option("--tags", help="Comma-separated tags")
@click.option(
    "--importance",
    "-i",
    type=click.FloatRange(0.0, 1.0),
    default=0.7,
    show_default=True,
    help="Importance 0.0-1.0 for the overview memory; sections inherit lower.",
)
@click.option(
    "--max-bytes",
    type=int,
    default=100 * 1024 * 1024,
    show_default=True,
    help="Body cap (bytes) for URL/file fetch. Default 100 MB.",
)
@click.option(
    "--timeout-connect",
    type=float,
    default=10.0,
    show_default=True,
    help="HTTP connect timeout (seconds).",
)
@click.option(
    "--timeout-read",
    type=float,
    default=60.0,
    show_default=True,
    help="HTTP read timeout (seconds).",
)
@click.option(
    "--allow-http",
    is_flag=True,
    default=False,
    help="Allow http:// URLs (default: HTTPS only).",
)
@click.option(
    "--allow-system-paths",
    is_flag=True,
    default=False,
    help="Allow ingesting paths under /etc, /proc, /root, ~/.ssh, etc.",
)
@click.option(
    "--render/--no-render",
    "render",
    default=False,
    help=(
        "Render the URL in a headless browser before ingesting (for JS-heavy "
        "/ SPA pages). Requires the [ingest-browser] extra plus "
        "`playwright install chromium`. Ignored for local files."
    ),
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Estimate token + page counts without calling any LLM provider.",
)
@click.option(
    "--json",
    "-J",
    "as_json",
    is_flag=True,
    default=False,
    help=(
        "Emit the full IngestResult as JSON (machine-readable). "
        "Default is a human-readable summary."
    ),
)
@click.option(
    "--verbose",
    "-v",
    count=True,
    help="Increase verbosity (repeatable: -v, -vv, -vvv).",
)
@click.option(
    "--progress",
    type=click.Choice(_PROGRESS_CHOICES, case_sensitive=False),
    default=None,
    help=(
        "Progress output format. Default: rich if -v given, none otherwise. "
        "Use json for AI agents / scripts."
    ),
)
def ingest_file(
    source,
    context_id,
    text_provider,
    vision_provider,
    no_vision,
    no_archive,
    tags,
    importance,
    max_bytes,
    timeout_connect,
    timeout_read,
    allow_http,
    allow_system_paths,
    render,
    dry_run,
    as_json,
    verbose,
    progress,
):
    """Ingest a URL or local file into Memory Cloud.

    Extracts structural sections from the source and creates one overview
    memory plus one memory per section, linked via declared_link edges. Heavy
    parsing dependencies are optional — install them with:

      pip install 'kagura-memory[ingest-pdf]'

    Phase 1 ingests text content only. The vision pipeline (Gemini 2.5 Flash
    by default) is configured but the bundled PDF extractor does not yet
    emit page images — image-based OCR memories land in Phase 2. Pass
    --no-vision to skip vision-provider configuration entirely (no
    image bytes will be sent to any provider once Phase 2 lands).

    Examples:
      kagura ingest https://example.com/report.pdf
      kagura ingest ./report.pdf --tags "Q1,report"
      kagura ingest report.pdf --no-vision
      kagura ingest report.pdf --dry-run
    """
    try:
        config = load_config()
        # api_key is only required when we're going to write memories or hit
        # the object store. --dry-run is purely local (fetch + extract +
        # token counter) and must work without authentication.
        if not dry_run and not config.get("api_key"):
            raise click.ClickException(
                "No API key found. Set KAGURA_API_KEY or create .kagura.json"
            )
        ctx_id = context_id or config.get("context_id") or ""
        if not ctx_id and not dry_run:
            raise click.ClickException(
                "context_id required. Use --context-id or set in .kagura.json"
            )
        # Dry-run does not write to Memory Cloud, but still uses the providers'
        # local token counters; allow context_id to be empty in that case.
        ctx_id = ctx_id or "00000000-0000-0000-0000-000000000000"

        api_key = config.get("api_key", "") or "kagura_dry-run-no-auth"
        mcp_url = config.get("mcp_url", "https://memory.kagura-ai.com/mcp")
        # KaguraClient validates the URL is HTTPS (or localhost) — the
        # placeholder api_key above lets it construct cleanly when --dry-run
        # is used without a config file. No network calls will use it.
        client = KaguraClient(api_key=api_key, mcp_url=mcp_url)
        # Archive is disabled in dry-run (no network egress to LLM or R2)
        # and when the caller passes --no-archive.
        archive = not (dry_run or no_archive)
        files_client = (
            FilesClient.from_mcp_url(api_key=api_key, mcp_url=mcp_url) if archive else None
        )

        # Deferred: keeps litellm/pymupdf/pillow off the import path of CLI
        # commands that don't ingest. Hoisting this to the top of the file
        # would re-introduce a startup penalty for every `kagura ...` call.
        from .ingest import FileIngestor

        async def _run() -> Any:
            async with client:
                try:
                    ingestor = FileIngestor(
                        client=client,
                        text_provider_name=text_provider,
                        vision_provider_name=None if no_vision else vision_provider,
                        files_client=files_client,
                    )
                    if dry_run:
                        return await ingestor.estimate_cost(
                            source,
                            max_bytes=max_bytes,
                            connect_timeout=timeout_connect,
                            read_timeout=timeout_read,
                            allow_http=allow_http,
                            allow_system_paths=allow_system_paths,
                            render=render,
                        )
                    return await ingestor.ingest(
                        source,
                        context_id=ctx_id,
                        tags=_parse_tags(tags),
                        importance=importance,
                        max_bytes=max_bytes,
                        connect_timeout=timeout_connect,
                        read_timeout=timeout_read,
                        allow_http=allow_http,
                        allow_system_paths=allow_system_paths,
                        render=render,
                        archive_original=archive,
                        logger=_resolve_progress_logger(verbose, progress),
                    )
                finally:
                    if files_client is not None:
                        await files_client.close()

        result = asyncio.run(_run())
        if as_json:
            click.echo(result.model_dump_json(indent=2))
        else:
            from rich.console import Console

            from .ingest._render import render_dry_run, render_result

            console = Console()
            if result.is_dry_run:
                render_dry_run(result, console)
            else:
                render_result(result, console)

        # Exit code contract:
        #   0 = overview created (partial section errors are still 0,
        #       matching the best-effort design — callers can inspect
        #       result.errors for finer-grained handling)
        #   0 = dry-run completed (no real success/failure)
        #   1 = overview NOT created (fatal)
        if dry_run or result.success:
            return
        sys.exit(1)

    except click.ClickException:
        raise
    except Exception as e:
        # ClickException's __str__ already prefixes "Error: " when printed,
        # so wrapping the message with another "Error: " would render as
        # "Error: Error: <msg>". Pass the raw exception message instead.
        raise click.ClickException(_exc_message(e)) from e


@main.command()
@click.option("--context-id", "-c", help="Context ID (or set in .kagura.json)")
@click.option("--memory-id", "-m", required=True, help="Memory ID to get full details")
def reference(context_id, memory_id):
    """
    Get full details of a specific memory.

    Examples:
      kagura reference -m "abc-123-def"
      kagura reference -c dev -m "abc-123-def"
    """
    _run_client_command(
        lambda client, ctx: client.reference(context_id=ctx, memory_id=memory_id),
        context_id,
    )


@main.command(name="update-memory")
@click.option("--context-id", "-c", help="Context ID (or set in .kagura.json)")
@click.option("--memory-id", "-m", help="Memory UUID to update in-place")
@click.option("--external-id", help="External ID for upsert lookup")
@click.option("--summary", "-s", help="Updated summary")
@click.option("--content", help="Updated content")
@click.option("--type", "-t", "memory_type", help="Updated memory type")
@click.option("--importance", "-i", type=float, help="Updated importance 0.0-1.0")
@click.option("--tags", help="Comma-separated tags")
def update_memory(
    context_id, memory_id, external_id, summary, content, memory_type, importance, tags
):
    """
    Update an existing memory or upsert by external ID.

    Use --memory-id for in-place update, or --external-id for upsert.

    Examples:
      kagura update-memory -m MEM_UUID -s "updated summary"
      kagura update-memory --external-id ext-key -s "summary" --content "..." -t note
    """
    if not memory_id and not external_id:
        raise click.ClickException("Either --memory-id or --external-id is required")
    if memory_id and external_id:
        raise click.ClickException("Provide only one of --memory-id or --external-id")

    tag_list = _parse_tags(tags)

    _run_client_command(
        lambda client, ctx: client.update_memory(
            context_id=ctx,
            memory_id=memory_id,
            external_id=external_id,
            summary=summary,
            content=content,
            type=memory_type,
            importance=importance,
            tags=tag_list,
        ),
        context_id,
    )


@main.command()
@click.option("--context-id", "-c", help="Context ID (or set in .kagura.json)")
@click.option("--memory-id", "-m", help="Memory ID to delete (specific deletion)")
@click.option("--query", "-q", help="Query to find memories to delete (bulk deletion)")
@click.option("-k", type=int, default=10, help="Max memories to delete in query mode")
def forget(context_id, memory_id, query, k):
    """
    Delete memories (soft delete, recoverable for 30 days).

    Use --memory-id for specific deletion or --query for bulk deletion.

    Examples:
      kagura forget -m "abc-123-def"
      kagura forget -q "outdated test data" -k 5
      kagura forget -c dev -m "memory-uuid"
    """
    if not memory_id and not query:
        raise click.ClickException("Either --memory-id or --query is required")

    _run_client_command(
        lambda client, ctx: client.forget(context_id=ctx, memory_id=memory_id, query=query, k=k),
        context_id,
    )


@main.group()
def context():
    """Manage contexts (list, create, update)."""
    pass


@context.command(name="list")
def context_list():
    """
    List available contexts.

    Examples:
      kagura context list
    """
    _run_client_command(
        lambda client, _: client.list_contexts(),
        context_id=None,
        needs_context=False,
    )


@context.command(name="create")
@click.option("--name", "-n", required=True, help="Context name (lowercase, hyphens, underscores)")
@click.option("--display-name", help="Human-readable display name")
@click.option("--description", "-d", help="Context description")
@click.option("--summary", "-s", help="LLM-oriented summary (200-500 chars)")
@click.option("--usage-guide", help="LLM-oriented usage guidelines")
@click.option("--public", is_flag=True, default=False, help="Accessible to workspace members")
def context_create(name, display_name, description, summary, usage_guide, public):
    """
    Create a new context.

    Examples:
      kagura context create -n my-project
      kagura context create -n dev -d "Development notes" -s "Project dev context"
    """
    _run_client_command(
        lambda client, _: client.create_context(
            name=name,
            display_name=display_name,
            description=description,
            summary=summary,
            usage_guide=usage_guide,
            is_private=not public,
        ),
        context_id=None,
        needs_context=False,
    )


@context.command(name="delete")
@click.argument("context_id")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def context_delete(context_id, yes):
    """
    Soft-delete a context and all its memories.

    Examples:
      kagura context delete CTX_UUID
      kagura context delete CTX_UUID -y
    """
    if not yes:
        click.confirm(f"Delete context {context_id}? This is a soft-delete.", abort=True)

    _run_client_command(
        lambda client, _: client.delete_context(context_id=context_id),
        context_id=None,
        needs_context=False,
    )


@context.command(name="update")
@click.argument("context_id")
@click.option("--display-name", help="Updated display name")
@click.option("--description", "-d", help="Updated description")
@click.option("--summary", "-s", help="Updated LLM-oriented summary")
@click.option("--usage-guide", help="Updated LLM-oriented usage guidelines")
@click.option("--lock/--unlock", default=None, help="Lock or unlock the context")
def context_update(context_id, display_name, description, summary, usage_guide, lock):
    """
    Update a context's settings.

    Examples:
      kagura context update CTX_UUID -s "Updated summary"
      kagura context update CTX_UUID --lock
      kagura context update CTX_UUID --unlock
    """
    if all(v is None for v in (display_name, description, summary, usage_guide, lock)):
        raise click.ClickException("At least one update option is required")

    _run_client_command(
        lambda client, _: client.update_context(
            context_id=context_id,
            display_name=display_name,
            description=description,
            summary=summary,
            usage_guide=usage_guide,
            is_locked=lock,
        ),
        context_id=None,
        needs_context=False,
    )


@context.command(name="search-config")
@click.argument("context_id")
@click.option("--semantic", type=click.FloatRange(0.0, 1.0), help="Semantic weight (0.0-1.0)")
@click.option("--bm25", type=click.FloatRange(0.0, 1.0), help="BM25 weight (0.0-1.0)")
@click.option("--fetch-factor", type=click.IntRange(1, 10), help="Fetch multiplier (1-10)")
@click.option("--rerank/--no-rerank", default=None, help="Enable/disable reranking")
@click.option("--reranker", type=click.Choice(["voyage", "cohere"]), help="Reranker provider")
@click.option("--reranker-model", help="Reranker model name")
def context_search_config(
    context_id, semantic, bm25, fetch_factor, rerank, reranker, reranker_model
):
    """
    Update search configuration for a context.

    Weights must sum to 1.0.

    Examples:
      kagura context search-config CTX_UUID --semantic 0.5 --bm25 0.5
      kagura context search-config CTX_UUID --rerank --reranker voyage
    """
    if all(v is None for v in (semantic, bm25, fetch_factor, rerank, reranker, reranker_model)):
        raise click.ClickException("At least one option is required")

    if semantic is not None and bm25 is not None and abs(semantic + bm25 - 1.0) > 0.01:
        raise click.ClickException(
            f"Weights must sum to 1.0 (got {semantic} + {bm25} = {semantic + bm25})"
        )

    _run_client_command(
        lambda client, _: client.update_search_config(
            context_id=context_id,
            semantic_weight=semantic,
            bm25_weight=bm25,
            fetch_factor=fetch_factor,
            use_rerank=rerank,
            reranker_provider=reranker,
            reranker_model=reranker_model,
        ),
        context_id=None,
        needs_context=False,
    )


# Keep backward compat: kagura contexts → kagura context list
@main.command()
def contexts():
    """List available contexts (alias for 'context list')."""
    _run_client_command(
        lambda client, _: client.list_contexts(),
        context_id=None,
        needs_context=False,
    )


# =============================================================================
# Edge Commands
# =============================================================================


@main.group()
def edge():
    """Manage neural memory edges (list, create, update, delete)."""
    pass


@edge.command(name="list")
@click.argument("context_id")
@click.argument("memory_id")
@click.option("--min-weight", type=float, default=0.0, help="Minimum edge weight (0.0-3.0)")
@click.option(
    "--type",
    "edge_types",
    help="Comma-separated edge types to filter (e.g. 'related_to,depends_on')",
)
@click.option(
    "--limit", type=int, help="Max edges per direction (effective max: 2x limit, dedup'd)"
)
def edge_list(context_id, memory_id, min_weight, edge_types, limit):
    """
    List edges connected to a memory.

    Examples:
      kagura edge list CTX_UUID MEM_UUID
      kagura edge list CTX_UUID MEM_UUID --min-weight 0.5 --type related_to
    """

    async def op(client: KaguraClient, _ctx: str) -> dict[str, Any]:
        edges = await client.list_edges(
            context_id=context_id,
            memory_id=memory_id,
            min_weight=min_weight,
            edge_types=_parse_tags(edge_types),
            limit=limit,
        )
        return {"edges": [e.model_dump(mode="json") for e in edges], "count": len(edges)}

    _run_client_command(op, context_id=None, needs_context=False)


@edge.command(name="create")
@click.argument("context_id")
@click.argument("source_id")
@click.argument("target_id")
@click.option("--type", "edge_type", default="related_to", help="Edge type (default: related_to)")
@click.option("--weight", type=float, default=0.5, help="Edge weight 0.0-3.0 (default: 0.5)")
@click.option(
    "--confidence", type=float, default=1.0, help="Edge confidence 0.0-1.0 (default: 1.0)"
)
def edge_create(context_id, source_id, target_id, edge_type, weight, confidence):
    """
    Create or upsert an edge from SOURCE_ID to TARGET_ID.

    If an edge already exists for the same pair, the server applies max-weight
    UPSERT semantics (existing weight is replaced only when the new weight is
    higher). Self-loops are rejected.

    Examples:
      kagura edge create CTX_UUID SRC_UUID TGT_UUID
      kagura edge create CTX_UUID SRC_UUID TGT_UUID --type depends_on --weight 0.8
    """

    async def op(client: KaguraClient, _ctx: str) -> dict[str, Any]:
        result = await client.create_edge(
            context_id=context_id,
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=weight,
            confidence=confidence,
        )
        return result.model_dump(mode="json")

    _run_client_command(op, context_id=None, needs_context=False)


@edge.command(name="update")
@click.argument("context_id")
@click.argument("source_id")
@click.argument("target_id")
@click.option("--weight", type=float, help="New edge weight 0.0-3.0")
@click.option("--type", "edge_type", help="New edge type")
def edge_update(context_id, source_id, target_id, weight, edge_type):
    """
    Update an existing edge's weight and/or type.

    At least one of --weight or --type must be provided.

    Examples:
      kagura edge update CTX_UUID SRC_UUID TGT_UUID --weight 0.9
      kagura edge update CTX_UUID SRC_UUID TGT_UUID --type related_to --weight 0.7
    """
    if weight is None and edge_type is None:
        raise click.ClickException("At least one of --weight or --type must be provided")

    async def op(client: KaguraClient, _ctx: str) -> dict[str, Any]:
        result = await client.update_edge(
            context_id=context_id,
            source_id=source_id,
            target_id=target_id,
            weight=weight,
            edge_type=edge_type,
        )
        return result.model_dump(mode="json")

    _run_client_command(op, context_id=None, needs_context=False)


@edge.command(name="delete")
@click.argument("context_id")
@click.argument("source_id")
@click.argument("target_id")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def edge_delete(context_id, source_id, target_id, yes):
    """
    Delete the edge between SOURCE_ID and TARGET_ID.

    Examples:
      kagura edge delete CTX_UUID SRC_UUID TGT_UUID
      kagura edge delete CTX_UUID SRC_UUID TGT_UUID -y
    """
    if not yes:
        click.confirm(f"Delete edge {source_id} -> {target_id}?", abort=True)

    async def op(client: KaguraClient, _ctx: str) -> dict[str, Any]:
        deleted = await client.delete_edge(
            context_id=context_id,
            source_id=source_id,
            target_id=target_id,
        )
        return {"deleted": deleted}

    _run_client_command(op, context_id=None, needs_context=False)


# =============================================================================
# Sleep Maintenance Commands (issue #85)
# =============================================================================


@main.group()
def sleep():
    """Inspect and roll back Sleep Maintenance runs."""
    pass


@sleep.command(name="history")
@click.argument("context_id")
@click.option(
    "--limit",
    default=10,
    type=click.IntRange(1, 50),
    help="Max runs to return (1-50, default: 10).",
)
def sleep_history(context_id, limit):
    """List recent Sleep Maintenance runs for a context."""

    async def _op(client: KaguraClient, _ctx: str) -> dict[str, Any]:
        reports = await client.get_sleep_history(context_id=context_id, limit=limit)
        return {"reports": [r.model_dump(mode="json") for r in reports]}

    _run_client_command(_op, context_id=None, needs_context=False)


@sleep.command(name="report")
@click.argument("context_id")
@click.argument("report_id")
def sleep_report(context_id, report_id):
    """Get a detailed Sleep Maintenance report including audit log."""

    async def _op(client: KaguraClient, _ctx: str) -> dict[str, Any]:
        detail = await client.get_sleep_report(context_id=context_id, report_id=report_id)
        return detail.model_dump(mode="json")

    _run_client_command(_op, context_id=None, needs_context=False)


@sleep.command(name="rollback")
@click.argument("context_id")
@click.argument("report_id")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt.")
def sleep_rollback(context_id, report_id, yes):
    """Roll back a completed Sleep Maintenance run.

    Reverses edge creation, memory merges, importance updates, scope
    promotions, and archives.
    """
    # Bundled inline (not via ``_run_client_command``) so a "no" answer to
    # ``click.confirm(..., abort=True)`` propagates as ``click.Abort`` and
    # click prints "Aborted!" + exit 1 naturally — the helper's broad
    # ``except Exception`` would otherwise wrap it as "Error: ".
    client = _get_kagura_client()

    async def _run() -> dict[str, Any]:
        async with client:
            # The pre-fetch only feeds the confirm prompt — skip it on --yes.
            if not yes:
                detail = await client.get_sleep_report(context_id=context_id, report_id=report_id)
                click.confirm(
                    f"Roll back sleep run {report_id}? "
                    f"Started {detail.started_at}, {detail.action_count} action(s). "
                    "This reverses edge creation, merges, importance updates, "
                    "promotions, and archives.",
                    abort=True,
                )
            result = await client.rollback_sleep_run(context_id=context_id, report_id=report_id)
            return result.model_dump(mode="json")

    try:
        result = asyncio.run(_run())
        click.echo(json.dumps(result, indent=2, ensure_ascii=False))
    except (click.Abort, click.ClickException):
        raise
    except Exception as e:
        raise click.ClickException(_exc_message(e)) from e


# =============================================================================
# Setup Commands
# =============================================================================


@main.group()
def setup():
    """Set up Kagura integrations for AI coding tools."""
    pass


@setup.command(name="claude")
@click.option("--api-key", help="Kagura API key (skip prompt)")
@click.option("--mcp-url", help="MCP URL (skip prompt)")
@click.option("--context-id", help="Context ID or name (skip prompt)")
@click.option(
    "--profile",
    default=None,
    help=(
        "OAuth profile name (from `kagura auth login`). Writes the refresh-aware "
        "`kagura-mcp` stdio .mcp.json form instead of a static API-key url. "
        "Mutually exclusive with --api-key."
    ),
)
@click.option("--project-dir", default=".", help="Project directory (default: current)")
@click.option("--non-interactive", "-y", is_flag=True, help="No prompts, use defaults/flags")
@click.option("--no-auto-context", is_flag=True, help="Disable auto-select by directory name")
def setup_claude(
    api_key, mcp_url, context_id, profile, project_dir, non_interactive, no_auto_context
):
    """
    Set up Kagura Memory integration for Claude Code.

    Configures .kagura.json, .mcp.json, hooks, and skills in the target project.

    Two auth modes:
      - default (API key): static token baked into .mcp.json (CI / service accounts)
      - --profile NAME (OAuth): refresh-aware `kagura-mcp` stdio proxy, no silent 401s

    Examples:
      kagura setup claude
      kagura setup claude --profile default        # OAuth via kagura-mcp (recommended)
      kagura setup claude --api-key kagura_xxx --mcp-url http://localhost:8080/mcp/w/{workspace_id}
      kagura setup claude -y --api-key kagura_xxx --context-id my-project
      kagura setup claude --no-auto-context  # always show full context list
    """
    try:
        run_setup_claude(
            api_key=api_key,
            mcp_url=mcp_url,
            context_id=context_id,
            project_dir=project_dir,
            non_interactive=non_interactive,
            no_auto_context=no_auto_context,
            profile=profile,
        )
    except (click.Abort, click.ClickException):
        # click.Abort fires on Ctrl+D / explicit abort during prompts — let Click
        # render its default "Aborted!" UX instead of reporting "Setup failed: Abort".
        raise
    except Exception as e:
        raise click.ClickException(f"Setup failed: {_exc_message(e)}") from e


# =============================================================================
# Resource Token Commands
# =============================================================================


def _get_resource_client() -> ResourceClient:
    """Load config and create ResourceClient via the canonical SDK chain.

    Walks :func:`_resolve_auth`'s precedence (``env > OAuth profile >
    .kagura.json``), matching :class:`FilesClient` and :class:`KaguraClient`.
    OAuth-only operators can run all ``kagura resource`` commands except
    ``kagura resource setup``, which currently requires a static api_key
    (see :meth:`ResourceClient.setup_resource`).
    """
    from .exceptions import KaguraAuthError

    config = load_config()
    try:
        # mcp_url=None so each resolver branch pairs its credential with
        # its own URL source (env → KAGURA_MCP_URL, OAuth → profile's
        # stored mcp_url, config → .kagura.json's mcp_url). Forwarding
        # ``config.get("mcp_url")`` here would override every branch
        # with the config / env-default URL — including OAuth profiles
        # bound to a non-default server (see PR #119 review).
        resolved = _resolve_auth(
            api_key=None,
            mcp_url=None,
            profile=None,
            config=config,
        )
    except KaguraAuthError as e:
        raise click.ClickException(_exc_message(e)) from e
    return ResourceClient._from_resolved_auth(resolved)


def _get_kagura_client() -> KaguraClient:
    """Load config and create KaguraClient.

    Unlike :func:`_get_resource_client`, this does NOT pre-check for
    an api_key — ``KaguraClient.__init__`` runs the full resolution
    chain (env > OAuth profile > .kagura.json) so a ``kagura auth
    login``-only setup works without a static api_key. The credential
    error from the chain is translated to a ``ClickException`` so the
    operator sees a clean message instead of a traceback.
    """
    from .exceptions import KaguraAuthError

    config = load_config()
    try:
        return KaguraClient(
            api_key=config.get("api_key") or None,
            mcp_url=config.get("mcp_url") or None,
        )
    except KaguraAuthError as e:
        raise click.ClickException(_exc_message(e)) from e


def _run_resource_command(
    operation: Callable[[ResourceClient], Awaitable[Any]],
) -> None:
    """Execute a ResourceClient operation with standard boilerplate."""
    try:
        client = _get_resource_client()

        async def _run() -> Any:
            async with client:
                return await operation(client)

        result = asyncio.run(_run())
        if result is not None:
            click.echo(result)
    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(_exc_message(e)) from e


@main.group()
def resource():
    """Manage resource tokens and ingest external data."""
    pass


@resource.group(name="tokens")
def resource_tokens():
    """Manage resource tokens (CRUD)."""
    pass


@resource_tokens.command(name="list")
@click.option("--resource-id", "-r", help="Filter by resource ID")
@click.option("--limit", "-l", type=int, default=50, help="Results per page (max 100)")
def tokens_list(resource_id, limit):
    """
    List resource tokens.

    Examples:
      kagura resource tokens list
      kagura resource tokens list --resource-id products
    """

    async def op(client: ResourceClient) -> str:
        result = await client.list_tokens(resource_id=resource_id, limit=limit)
        return result.model_dump_json(indent=2)

    _run_resource_command(op)


@resource_tokens.command(name="create")
@click.option("--resource-id", "-r", required=True, help="Resource ID to scope token to")
@click.option("--description", "-d", help="Token description")
@click.option("--quota", "-q", type=int, default=1000, help="Events per hour (1-10000)")
def tokens_create(resource_id, description, quota):
    """
    Create a new resource token.

    The token is shown ONLY once — save it immediately.

    Examples:
      kagura resource tokens create -r products
      kagura resource tokens create -r slack-messages -d "Slack integration" -q 5000
    """

    async def op(client: ResourceClient) -> str:
        result = await client.create_token(
            resource_id=resource_id,
            description=description,
            quota_events_per_hour=quota,
        )
        return result.model_dump_json(indent=2)

    _run_resource_command(op)


@resource_tokens.command(name="update")
@click.argument("token_id", type=int)
@click.option("--description", "-d", help="Updated description")
@click.option("--quota", "-q", type=int, help="Updated events per hour (1-10000)")
def tokens_update(token_id, description, quota):
    """
    Update a resource token.

    Examples:
      kagura resource tokens update 42 -d "New description"
      kagura resource tokens update 42 -q 2000
    """
    if description is None and quota is None:
        raise click.ClickException("At least --description or --quota is required")

    async def op(client: ResourceClient) -> str:
        result = await client.update_token(
            token_id=token_id,
            description=description,
            quota_events_per_hour=quota,
        )
        return result.model_dump_json(indent=2)

    _run_resource_command(op)


@resource_tokens.command(name="revoke")
@click.argument("token_id", type=int)
def tokens_revoke(token_id):
    """
    Revoke (soft-delete) a resource token.

    Examples:
      kagura resource tokens revoke 42
    """

    async def op(client: ResourceClient) -> str:
        await client.revoke_token(token_id)
        return "Token revoked."

    _run_resource_command(op)


@resource.command(name="ingest")
@click.option("--resource-id", "-r", required=True, help="Resource ID")
@click.option("--api-key", "-k", required=True, help="Resource API key (X-Resource-API-Key)")
@click.option("--doc-id", required=True, help="Document ID")
@click.option("--op", type=click.Choice(["upsert", "delete"]), default="upsert", help="Operation")
@click.option("--version", "-V", type=int, help="Document version (>=1)")
@click.option("--payload", "-p", help="JSON payload string")
@click.option("--importance", "-i", type=float, help="Importance 0.0-1.0")
def ingest(resource_id, api_key, doc_id, op, version, payload, importance):
    """
    Ingest a single resource event.

    Examples:
      kagura resource ingest -r products -k KEY --doc-id SKU-001 -p '{"name":"Widget","price":9.99}'
      kagura resource ingest -r products -k KEY --doc-id SKU-999 --op delete
    """
    try:
        parsed_payload = json.loads(payload) if payload else None
    except json.JSONDecodeError as e:
        raise click.ClickException(f"Invalid JSON payload: {_exc_message(e)}") from e

    event = ResourceEventRequest(
        op=op,
        doc_id=doc_id,
        version=version,
        payload=parsed_payload,
        importance=importance,
    )

    async def op_fn(client: ResourceClient) -> str:
        result = await client.ingest_event(resource_id, api_key, event)
        return result.model_dump_json(indent=2)

    _run_resource_command(op_fn)


@resource.command(name="ingest-batch")
@click.option("--resource-id", "-r", required=True, help="Resource ID")
@click.option("--api-key", "-k", required=True, help="Resource API key (X-Resource-API-Key)")
@click.option("--file", "-f", type=click.File("r"), required=True, help="JSON events array")
def ingest_batch(resource_id, api_key, file):
    """
    Ingest a batch of resource events from a JSON file.

    The file should contain a JSON array of event objects.

    Examples:
      kagura resource ingest-batch -r products -k KEY -f events.json
    """
    try:
        data = json.load(file)
    except json.JSONDecodeError as e:
        raise click.ClickException(f"Invalid JSON file: {_exc_message(e)}") from e

    if not isinstance(data, list):
        raise click.ClickException("JSON file must contain an array of events")

    events = [ResourceEventRequest(**e) for e in data]

    async def op(client: ResourceClient) -> str:
        result = await client.ingest_events(resource_id, api_key, events)
        return result.model_dump_json(indent=2)

    _run_resource_command(op)


@resource.command(name="stats")
@click.option("--resource-id", "-r", required=True, help="Resource ID")
def resource_stats(resource_id):
    """
    Show resource impact statistics.

    Examples:
      kagura resource stats -r products
    """

    async def op(client: ResourceClient) -> str:
        result = await client.get_resource_impact(resource_id)
        return result.model_dump_json(indent=2)

    _run_resource_command(op)


@resource.command(name="list")
def resource_list():
    """
    List all resources in the workspace (owner only).

    Examples:
      kagura resource list
    """

    async def op(client: ResourceClient) -> str:
        result = await client.list_resources()
        return result.model_dump_json(indent=2)

    _run_resource_command(op)


@resource.command(name="indexer-status")
@click.option("--resource-id", "-r", required=True, help="Resource ID")
def resource_indexer_status(resource_id):
    """
    Show indexer state and recent ingest events for a resource.

    The ``state`` field is null when the indexer has never run for this
    resource (this is a normal 200 response, distinct from a 404).

    Examples:
      kagura resource indexer-status -r products
    """

    async def op(client: ResourceClient) -> str:
        result = await client.get_indexer_status(resource_id)
        return result.model_dump_json(indent=2)

    _run_resource_command(op)


@resource.command(name="events")
@click.argument("resource_id")
@click.option(
    "--limit", "-l", type=click.IntRange(1, 100), default=50, help="Events per page (1-100)"
)
@click.option("--cursor", "-c", help="Pagination cursor from a prior next_cursor")
@click.option("--op", type=click.Choice(["upsert", "delete"]), help="Filter by operation")
@click.option("--doc-id", help="Filter by document ID")
@click.option("--version", "-V", type=int, help="Filter by document version")
@click.option("--since", help="Only events at/after this ISO 8601 time (e.g. 2026-06-01T00:00:00Z)")
def resource_events(resource_id, limit, cursor, op, doc_id, version, since):
    """
    List ingested events for a resource (cursor-paginated).

    The response includes a ``next_cursor``; pass it back via --cursor to
    page forward (null on the last page).

    Examples:
      kagura resource events products
      kagura resource events products --op upsert --limit 20
      kagura resource events products --since 2026-06-01T00:00:00Z
      kagura resource events products --cursor "eyJ..."
    """
    parsed_since: datetime | None = None
    if since is not None:
        try:
            parsed_since = datetime.fromisoformat(since)
        except ValueError as e:
            raise click.ClickException(
                f"Invalid --since value (expected ISO 8601, e.g. 2026-06-01T00:00:00Z): "
                f"{_exc_message(e)}"
            ) from e

    async def op_fn(client: ResourceClient) -> str:
        result = await client.list_resource_events(
            resource_id,
            limit=limit,
            cursor=cursor,
            op=op,
            doc_id=doc_id,
            version=version,
            since=parsed_since,
        )
        return result.model_dump_json(indent=2)

    _run_resource_command(op_fn)


@resource.command(name="schema")
@click.option("--resource-id", "-r", required=True, help="Resource ID")
@click.option("--version", "-v", "schema_version", type=int, help="Schema version")
def resource_schema(resource_id, schema_version):
    """
    Show resource field definitions (schema).

    Examples:
      kagura resource schema -r products
      kagura resource schema -r products -v 2
    """

    async def op(client: ResourceClient) -> str:
        result = await client.get_resource_schema(resource_id, schema_version=schema_version)
        if result is None:
            return "No schema registered for this resource."
        return result.model_dump_json(indent=2)

    _run_resource_command(op)


@resource.command(name="setup")
@click.option("--resource-id", "-r", required=True, help="Resource identifier")
@click.option("--summary", "-s", help="Context summary")
@click.option("--description", "-d", help="Token description")
@click.option("--quota", "-q", type=click.IntRange(1, 10000), default=1000, help="Events/hour")
def resource_setup(resource_id, summary, description, quota):
    """
    One-shot resource setup: create context + set resource_id + create token.

    Examples:
      kagura resource setup -r products -s "Product catalog"
      kagura resource setup -r slack-messages -d "Slack sync" -q 5000
    """

    async def op(client: ResourceClient) -> str:
        token = await client.setup_resource(
            resource_id=resource_id,
            summary=summary,
            description=description,
            quota_events_per_hour=quota,
        )
        return token.model_dump_json(indent=2)

    _run_resource_command(op)


@resource.command(name="import")
@click.option("--resource-id", "-r", required=True, help="Resource ID")
@click.option("--api-key", "-k", required=True, help="Resource API key")
@click.option("--file", "-f", "input_file", type=click.File("r"), default="-")
@click.option(
    "--format", "fmt", type=click.Choice(["auto", "csv", "json", "jsonl"]), default="auto"
)
@click.option("--id-column", help="Column name to use as doc_id (default: row number)")
@click.option("--version", "-V", type=click.IntRange(1), default=1, help="Version (>=1)")
@click.option(
    "--verbose",
    "-v",
    count=True,
    help="Increase verbosity (repeatable: -v, -vv, -vvv).",
)
@click.option(
    "--progress",
    type=click.Choice(_PROGRESS_CHOICES, case_sensitive=False),
    default=None,
    help=(
        "Progress output format. Default: rich if -v given, none otherwise. "
        "Use json for AI agents / scripts."
    ),
)
def resource_import(resource_id, api_key, input_file, fmt, id_column, version, verbose, progress):
    """
    Import data from CSV, JSON, or JSONL file.

    Auto-detects format from file extension, or specify --format.
    Each row/object becomes a resource event with op=upsert.

    Examples:
      kagura resource import -r products -k TOKEN -f products.csv
      kagura resource import -r products -k TOKEN -f data.jsonl
      cat items.json | kagura resource import -r products -k TOKEN --format json
    """
    import csv
    from io import StringIO

    # Detect format
    if fmt == "auto":
        name = getattr(input_file, "name", "")
        if name.endswith(".csv"):
            fmt = "csv"
        elif name.endswith(".jsonl"):
            fmt = "jsonl"
        elif name.endswith(".json"):
            fmt = "json"
        else:
            raise click.ClickException("Cannot detect format. Use --format csv|json|jsonl")

    # Parse input
    try:
        raw = input_file.read()
    except Exception as e:
        raise click.ClickException(f"Failed to read input: {_exc_message(e)}") from e

    rows: list[dict] = []
    if fmt == "csv":
        reader = csv.DictReader(StringIO(raw))
        rows = list(reader)
    elif fmt == "json":
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise click.ClickException(f"Invalid JSON: {_exc_message(e)}") from e
        if not isinstance(data, list):
            raise click.ClickException("JSON must be an array of objects")
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise click.ClickException(f"JSON item {i} is not an object: {type(item).__name__}")
        rows = data
    elif fmt == "jsonl":
        for line_num, line in enumerate(raw.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise click.ClickException(
                    f"Invalid JSONL at line {line_num}: {_exc_message(e)}"
                ) from e
            if not isinstance(obj, dict):
                raise click.ClickException(f"JSONL line {line_num} is not an object")
            rows.append(obj)

    if not rows:
        raise click.ClickException("No data found in input")

    # Build events
    events = []
    for i, row in enumerate(rows):
        if id_column:
            if id_column not in row:
                raise click.ClickException(
                    f"Row {i + 1}: column '{id_column}' not found. Keys: {list(row.keys())}"
                )
            doc_id = str(row[id_column])
        else:
            doc_id = str(i + 1)
        events.append(
            ResourceEventRequest(
                op="upsert",
                doc_id=doc_id,
                version=version,
                payload=row,
            )
        )

    logger = _resolve_progress_logger(verbose, progress)
    # Pre-flight announcement: keep stdout strictly machine-readable (the
    # JSON output below) by routing the human-readable count to stderr.
    # Suppress entirely when progress is silent (--progress=none, or no -v
    # and no --progress) so scripts piping stderr to /dev/null see nothing
    # unexpected; otherwise emit it as a structured "start" action through
    # the logger so the rich path stays consistent and json consumers get
    # a parseable event.
    if logger is not None:
        logger.action(
            "Importing events",
            f"{len(events)} event(s)",
            stage="import_start",
        )

    # Batch ingest (100 at a time). The CLI command is ONE user-facing
    # operation, so it must emit exactly one terminal event. ``ingest_events``
    # also has its own per-call terminal contract — calling it N times with
    # the same logger would produce N ``kind=success`` events and break the
    # "wait for the first success/error" reader contract. Resolution: pass
    # ``logger=None`` to the SDK (suppresses its terminal events), emit
    # per-batch progress here at the CLI level, and close with one terminal
    # event at the very end covering the whole import.
    async def op(client: ResourceClient) -> str:
        total_created = 0
        total_failed = 0
        all_errors: list[dict] = []
        batch_count = (len(events) + 99) // 100
        try:
            for i, start in enumerate(range(0, len(events), 100), 1):
                batch = events[start : start + 100]
                if logger is not None:
                    logger.action(
                        "Ingesting batch",
                        f"{i}/{batch_count} ({len(batch)} event(s))",
                        stage="import_batch",
                    )
                result = await client.ingest_events(resource_id, api_key, batch, logger=None)
                total_created += result.created_count
                total_failed += result.failed_count
                all_errors.extend(result.errors[:5])  # Keep first 5 errors per batch
        except BaseException as e:
            if logger is not None:
                logger.error(
                    f"Import failed: {e}",
                    stage="complete",
                    detail={
                        "created_so_far": total_created,
                        "failed_so_far": total_failed,
                        "total_events": len(events),
                    },
                )
            raise
        output: dict = {"created": total_created, "failed": total_failed, "total": len(events)}
        if all_errors:
            output["errors"] = all_errors[:10]  # Show first 10 errors total
        if logger is not None:
            logger.success(
                "Import complete",
                stage="complete",
                detail={
                    "created": total_created,
                    "failed": total_failed,
                    "total": len(events),
                },
            )
        return json.dumps(output, indent=2, ensure_ascii=False)

    _run_resource_command(op)


# =============================================================================
# Files (`kagura files ...`) — server v0.15.1+
# =============================================================================

# Sentinel for `kagura setup claude`-generated `.kagura.json` — means
# "fill in workspace_id from the credential source the resolver picks
# at runtime." Defined as a constant so the CLI, tests, and any future
# config writer share one source of truth and cannot drift.
_CONTEXT_ID_AUTO = "auto"


def _resolve_workspace_from_source(
    auth: _StaticAuth | _OAuthAuth,
    config: dict[str, Any],
    explicit_ctx: str | None,
) -> str:
    """Resolve workspace_id from the SAME source as ``api_key`` (issue #115).

    An api_key is provisioned for one specific workspace; mixing one
    source's api_key with another source's workspace_id is never
    correct (best case: 403; worst case: silently writing to the wrong
    workspace). This function pairs the two so the resolver picks both
    from the same branch:

    1. ``--context-id <uuid>`` always wins — operator override.
    2. ``_OAuthAuth`` → ``workspace_id`` snapshot from the OAuth
       profile (i.e. what ``kagura auth login`` stored).
    3. ``_StaticAuth(source="config")`` → ``.kagura.json``'s
       ``context_id`` (skipping the :data:`_CONTEXT_ID_AUTO` sentinel).
    4. ``_StaticAuth(source="env" | "explicit")`` → error, because env
       / explicit api_keys carry no associated workspace source. The
       operator must pass ``--context-id``.

    Raises ``click.ClickException`` with an actionable message when no
    same-source workspace can be formed.
    """
    # 1. Explicit -c always wins, regardless of source. Normalize via .strip()
    #    so that whitespace-padded inputs (e.g. ``--context-id "  auto  "`` or
    #    a UUID with surrounding spaces) are treated consistently: the
    #    sentinel comparison ignores padding, and the returned value never
    #    carries spaces downstream to FilesClient's UUID validator.
    if explicit_ctx:
        stripped_ctx = explicit_ctx.strip()
        if stripped_ctx and stripped_ctx != _CONTEXT_ID_AUTO:
            return stripped_ctx

    # 2. OAuth path: workspace is a snapshot from the profile.
    if isinstance(auth, _OAuthAuth):
        if auth.workspace_id:
            return auth.workspace_id
        raise click.ClickException(
            "OAuth profile has no workspace bound. Re-run `kagura auth login` "
            "or pass --context-id <uuid>."
        )

    # 3. Static api_key from .kagura.json → workspace must come from the
    #    same .kagura.json (context_id field, not the "auto" sentinel).
    if auth.source == "config":
        cfg_ctx = (config.get("context_id") or "").strip()
        if cfg_ctx and cfg_ctx != _CONTEXT_ID_AUTO:
            return cfg_ctx
        raise click.ClickException(
            '.kagura.json has api_key but context_id is missing or "auto". '
            "Set context_id to the workspace UUID bound to this api_key, "
            "or pass --context-id. (Falling back to the OAuth profile would "
            "mix credential sources — see issue #115.)"
        )

    # 4. env / explicit api_key: no associated workspace source.
    raise click.ClickException(
        f"api_key from {_SOURCE_LABEL[auth.source]} has no associated workspace; "
        "pass --context-id "
        "(mixing api_key and OAuth profile's workspace is not allowed — see issue #115)."
    )


def _bound_workspace_for_hint(auth: _StaticAuth | _OAuthAuth, config: dict[str, Any]) -> str | None:
    """The workspace BOUND to the api_key source — for 403 hint display only.

    Used by :func:`_run_files_command` to thread the right
    "source workspace" prefix into :class:`FilesClient` so a 403 can
    show ``api_key source: .kagura.json (workspace=<source-bound>)``
    next to the request's target workspace. Returns ``None`` when no
    workspace is bound to the source (env / explicit api_key) so the
    formatter renders ``<none>`` instead.

    OAuth callers receive their workspace hint via the resolver
    (``_OAuthAuth.workspace_id``); this helper only fills the
    ``_StaticAuth(source="config")`` case where the bound workspace
    lives in ``.kagura.json``'s ``context_id`` field.
    """
    if isinstance(auth, _StaticAuth) and auth.source == "config":
        cfg_ctx = (config.get("context_id") or "").strip()
        if cfg_ctx and cfg_ctx != _CONTEXT_ID_AUTO:
            return cfg_ctx
    return None


def _run_files_command(
    operation: Callable[[FilesClient, str], Awaitable[Any]],
    context_id: str | None,
    *,
    needs_context: bool = True,
) -> None:
    """Execute a FilesClient operation with standard boilerplate.

    Resolves credentials via :func:`_resolve_auth` (the canonical SDK
    chain ``env > OAuth profile > .kagura.json``). When
    ``needs_context=True`` (the default), workspace_id is resolved
    from the same source via :func:`_resolve_workspace_from_source`.
    When ``needs_context=False`` (file-id-based operations like
    ``download-url`` / ``delete``) only the client is built; ``ctx_id``
    is the empty string and operations must ignore it.
    """
    try:
        config = load_config()
        # mcp_url=None — let the resolver pair each credential source
        # with its matching URL. Forwarding ``config.get("mcp_url")``
        # would override OAuth profile's stored mcp_url with the
        # config / env-default URL, routing OAuth users on a
        # non-default server to the wrong host (see PR #119 review).
        auth = _resolve_auth(
            api_key=None,
            mcp_url=None,
            profile=None,
            config=config,
        )

        ctx_id = ""
        if needs_context:
            ctx_id = _resolve_workspace_from_source(auth, config, context_id)
            client = FilesClient._from_resolved_auth(
                auth, workspace_id_hint=_bound_workspace_for_hint(auth, config)
            )
        else:
            client = FilesClient._from_resolved_auth(auth)

        async def _run() -> Any:
            async with client:
                return await operation(client, ctx_id)

        result = asyncio.run(_run())
        if result is not None:
            click.echo(result)
    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(_exc_message(e)) from e


@main.group()
def files():
    """Upload, list, and manage files in Kagura Memory Cloud."""
    pass


async def _remember_file_object(
    ctx: str,
    path: Path,
    file_obj: FileObject,
    summary: str | None,
    memory_type: str,
    importance: float,
    tags: str | None,
) -> dict[str, Any]:
    """Create a summary memory linked to an uploaded file_object.

    The memory carries ``source_uri``/``source_type`` provenance plus a
    ``details.file_id`` back-reference — the same mechanism ``kagura ingest``
    uses so callers can resolve memory → original bytes. No LLM is invoked;
    the summary defaults to the filename, which keeps this path usable for
    binaries and in environments without LLM credentials.

    ``KaguraClient()`` is constructed with no arguments on purpose: it then
    runs the exact same ``_resolve_auth`` chain (env > OAuth profile >
    .kagura.json) that ``_run_files_command`` used for the upload, so the
    memory write shares the identity that owns the resolved workspace ``ctx``.
    Passing config-derived ``api_key``/``mcp_url`` would force the "explicit"
    branch and could diverge from the upload's source for users with
    credentials in more than one place (cross-workspace 403).
    """
    client = KaguraClient()
    content = (
        f"Uploaded file `{file_obj.filename}` "
        f"({file_obj.size_bytes} bytes, {file_obj.content_type}). "
        f"Stored as file_object {file_obj.id}."
    )
    async with client:
        result = await client.remember(
            context_id=ctx,
            summary=summary or f"File: {file_obj.filename}",
            content=content,
            type=memory_type,
            importance=importance,
            tags=_parse_tags(tags),
            source_uri=path.resolve().as_uri(),
            source_type="file",
            details={
                "file_id": file_obj.id,
                "sha256": file_obj.sha256,
                "size_bytes": file_obj.size_bytes,
                "content_type": file_obj.content_type,
            },
        )
    # remember() now raises KaguraError/KaguraNotFoundError on MCP domain
    # errors (issue #180). A successful response can still in principle lack
    # memory_id; treat that as a failure so the caller's handler can surface
    # the file_id and exit non-zero instead of printing an incomplete payload
    # as success.
    if not isinstance(result, dict) or not result.get("memory_id"):
        raise KaguraError(f"memory write reported an error: {result}")
    return result


@files.command(name="upload")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--context-id", "-c", help="Target context (workspace) UUID")
@click.option("--content-type", "-t", help="MIME type override (default: sniffed)")
@click.option(
    "--remember",
    is_flag=True,
    default=False,
    help="Also create a summary memory linked to the uploaded file_object (no LLM).",
)
@click.option(
    "--summary",
    default=None,
    help="Summary for the --remember memory (default: derived from filename).",
)
@click.option(
    "--type",
    "memory_type",
    default="note",
    show_default=True,
    help="Memory type for the --remember memory.",
)
@click.option(
    "--importance",
    type=click.FloatRange(0.0, 1.0),
    default=0.5,
    show_default=True,
    help="Importance 0.0-1.0 for the --remember memory.",
)
@click.option("--tags", default=None, help="Comma-separated tags for the --remember memory.")
@click.option(
    "--verbose",
    "-v",
    count=True,
    help="Increase verbosity (repeatable: -v, -vv, -vvv).",
)
@click.option(
    "--progress",
    type=click.Choice(_PROGRESS_CHOICES, case_sensitive=False),
    default=None,
    help=(
        "Progress output format. Default: rich if -v given, none otherwise. "
        "Use json for AI agents / scripts."
    ),
)
def files_upload(
    path: Path,
    context_id: str | None,
    content_type: str | None,
    remember: bool,
    summary: str | None,
    memory_type: str,
    importance: float,
    tags: str | None,
    verbose: int,
    progress: str | None,
):
    """
    Upload a file to Kagura Memory Cloud.

    With --remember, also creates one summary memory linked to the
    file_object (provenance + details.file_id back-reference), without
    invoking an LLM — works for binaries and keyless environments. For
    LLM-extracted section memories use `kagura ingest` instead.

    Examples:
      kagura files upload ./report.pdf --context-id ctx-uuid
      kagura files upload ./diagram.png --remember --tags "design,arch"
    """
    # Fail loud rather than silently drop memory-only flags passed without
    # --remember (the None-default flags are the ones that signal clear intent).
    if not remember and (summary is not None or tags is not None):
        raise click.UsageError("--summary and --tags require --remember.")

    logger = _resolve_progress_logger(verbose, progress)

    async def op(client: FilesClient, ctx: str) -> str:
        file_obj = await client.upload(
            context_id=ctx,
            source=path,
            content_type=content_type,
            logger=logger,
        )
        if not remember:
            return file_obj.model_dump_json(indent=2)
        try:
            memory = await _remember_file_object(
                ctx, path, file_obj, summary, memory_type, importance, tags
            )
        except Exception as e:
            # The upload already succeeded — surface the file_id so the user
            # knows the file_object exists and does not re-upload a duplicate.
            raise click.ClickException(
                f"File uploaded (file_id={file_obj.id}), but creating the linked "
                f"memory failed: {_exc_message(e)}. The file_object is stored; "
                f"retry the memory write separately or reference it by file_id."
            ) from e
        return json.dumps(
            {"file": file_obj.model_dump(mode="json"), "memory": memory},
            indent=2,
            ensure_ascii=False,
        )

    _run_files_command(op, context_id)


@files.command(name="download-url")
@click.argument("file_id")
def files_download_url(file_id: str):
    """
    Print a short-lived presigned GET URL for a file.

    Example:
      kagura files download-url <file_id>
    """

    async def op(client: FilesClient, _ctx: str) -> str:
        return await client.download_url(file_id)

    _run_files_command(op, None, needs_context=False)


@files.command(name="delete")
@click.argument("file_id")
def files_delete(file_id: str):
    """
    Soft-delete a file by id.

    Example:
      kagura files delete <file_id>
    """

    async def op(client: FilesClient, _ctx: str) -> str:
        await client.delete(file_id)
        return f"Deleted {file_id}"

    _run_files_command(op, None, needs_context=False)


@files.command(name="list")
@click.option("--context-id", "-c", help="Context (workspace) UUID to list")
@click.option(
    "--limit",
    "-l",
    type=click.IntRange(1, 500),
    default=50,
    help="Max results (1-500)",
)
@click.option("--cursor", help="Forward-compat cursor (server v0.16+)")
def files_list(context_id: str | None, limit: int, cursor: str | None):
    """
    List uploaded files in a context, newest first.

    Example:
      kagura files list --context-id ctx-uuid
    """

    async def op(client: FilesClient, ctx: str) -> str:
        result = await client.list(context_id=ctx, limit=limit, cursor=cursor)
        return result.model_dump_json(indent=2)

    _run_files_command(op, context_id)


if __name__ == "__main__":  # pragma: no cover
    main()
