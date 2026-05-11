"""CLI for Kagura Memory SDK."""

import asyncio
import json
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import click

from .agent import KaguraAgent
from .client import KaguraClient
from .config import load_config
from .files_client import FilesClient
from .models import Message, ProcessResult, ResourceEventRequest, Session
from .resource_client import ResourceClient
from .setup_claude import run_setup_claude

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

    Handles: config loading, API key check, context resolution,
    client lifecycle, async execution, JSON output, and error handling.
    """
    try:
        config = load_config()
        if not config.get("api_key"):
            raise click.ClickException(
                "No API key found. Set KAGURA_API_KEY or create .kagura.json"
            )

        ctx_id = ""
        if needs_context:
            ctx_id = context_id or config.get("context_id") or ""
            if not ctx_id:
                raise click.ClickException(
                    "context_id required. Use --context-id or set in .kagura.json"
                )

        client = KaguraClient(
            api_key=config.get("api_key", ""),
            mcp_url=config.get("mcp_url", "https://memory.kagura-ai.com/mcp"),
        )

        async def _run() -> dict[str, Any]:
            async with client:
                return await operation(client, ctx_id)

        result = asyncio.run(_run())
        click.echo(json.dumps(result, indent=2))

    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(f"Error: {e}") from e


@click.group()
@click.version_option(package_name="kagura-memory", prog_name="kagura")
def main():
    """Kagura Memory Cloud CLI - AI-driven memory management."""
    pass


@main.command()
@click.option("--message", "-m", help="Single message to process")
@click.option("--file", "-f", type=click.File("r"), help="Session JSON file")
@click.option("--deep", is_flag=True, help="Enable deep search (Phase 2)")
@click.option("--verbose", "-v", count=True, help="Increase verbosity (Phase 2)")
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
        if not config.get("api_key"):
            raise click.ClickException(
                "No API key found. Set KAGURA_API_KEY or create .kagura.json"
            )

        # Parse input
        if message:
            session = Session(messages=[Message(role="user", content=message)])
        elif file:
            session = Session(**json.load(file))
        else:
            session = Session(**json.load(sys.stdin))

        # Create agent — extract known keys to avoid TypeError on unknown config keys
        agent_kwargs: dict[str, Any] = {"api_key": config.get("api_key", "")}
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
        click.echo(json.dumps(result.model_dump(), indent=2))

    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(f"Error: {e}") from e


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
        click.echo(json.dumps(cfg, indent=2))
    except Exception as e:
        click.echo(f"Error loading config: {e}", err=True)
        sys.exit(1)


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
def remember(context_id, summary, content, memory_type, importance, tags):
    """
    Store a memory directly (without AI analysis).

    Examples:
      kagura remember -s "FastAPI DI pattern" --content "Use Depends()..."
      kagura remember -c dev -s "OAuth2 setup" --content "..." --tags "auth,oauth"
    """
    tag_list = _parse_tags(tags)

    async def op(client: KaguraClient, ctx: str) -> dict[str, Any]:
        return await client.remember(
            context_id=ctx,
            summary=summary,
            content=content,
            type=memory_type,
            importance=importance,
            tags=tag_list,
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
        click.echo(json.dumps(result, indent=2))
    except (click.Abort, click.ClickException):
        raise
    except Exception as e:
        raise click.ClickException(f"Error: {e}") from e


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
@click.option("--project-dir", default=".", help="Project directory (default: current)")
@click.option("--non-interactive", "-y", is_flag=True, help="No prompts, use defaults/flags")
def setup_claude(api_key, mcp_url, context_id, project_dir, non_interactive):
    """
    Set up Kagura Memory integration for Claude Code.

    Configures .kagura.json, .mcp.json, hooks, and skills in the target project.

    Examples:
      kagura setup claude
      kagura setup claude --api-key kagura_xxx --mcp-url http://localhost:8080/mcp/w/{workspace_id}
      kagura setup claude -y --api-key kagura_xxx --context-id my-project
    """
    try:
        run_setup_claude(
            api_key=api_key,
            mcp_url=mcp_url,
            context_id=context_id,
            project_dir=project_dir,
            non_interactive=non_interactive,
        )
    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(f"Setup failed: {e}") from e


# =============================================================================
# Resource Token Commands
# =============================================================================


def _get_resource_client() -> ResourceClient:
    """Load config and create ResourceClient."""
    config = load_config()
    if not config.get("api_key"):
        raise click.ClickException("No API key found. Set KAGURA_API_KEY or create .kagura.json")
    return ResourceClient.from_mcp_url(
        api_key=config.get("api_key", ""),
        mcp_url=config.get("mcp_url", "https://memory.kagura-ai.com/mcp"),
    )


def _get_kagura_client() -> KaguraClient:
    """Load config and create KaguraClient (mirrors :func:`_get_resource_client`)."""
    config = load_config()
    if not config.get("api_key"):
        raise click.ClickException("No API key found. Set KAGURA_API_KEY or create .kagura.json")
    return KaguraClient(
        api_key=config.get("api_key", ""),
        mcp_url=config.get("mcp_url", "https://memory.kagura-ai.com/mcp"),
    )


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
        raise click.ClickException(f"Error: {e}") from e


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
        raise click.ClickException(f"Invalid JSON payload: {e}") from e

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
        raise click.ClickException(f"Invalid JSON file: {e}") from e

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
def resource_import(resource_id, api_key, input_file, fmt, id_column, version):
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
        raise click.ClickException(f"Failed to read input: {e}") from e

    rows: list[dict] = []
    if fmt == "csv":
        reader = csv.DictReader(StringIO(raw))
        rows = list(reader)
    elif fmt == "json":
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise click.ClickException(f"Invalid JSON: {e}") from e
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
                raise click.ClickException(f"Invalid JSONL at line {line_num}: {e}") from e
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

    click.echo(f"Importing {len(events)} events...")

    # Batch ingest (100 at a time)
    async def op(client: ResourceClient) -> str:
        total_created = 0
        total_failed = 0
        all_errors: list[dict] = []
        for start in range(0, len(events), 100):
            batch = events[start : start + 100]
            result = await client.ingest_events(resource_id, api_key, batch)
            total_created += result.created_count
            total_failed += result.failed_count
            all_errors.extend(result.errors[:5])  # Keep first 5 errors per batch
        output: dict = {"created": total_created, "failed": total_failed, "total": len(events)}
        if all_errors:
            output["errors"] = all_errors[:10]  # Show first 10 errors total
        return json.dumps(output, indent=2)

    _run_resource_command(op)


# =============================================================================
# Files (`kagura files ...`) — server v0.15.1+
# =============================================================================


def _build_files_client(config: dict[str, Any]) -> FilesClient:
    """Build a FilesClient from an already-loaded config dict."""
    if not config.get("api_key"):
        raise click.ClickException("No API key found. Set KAGURA_API_KEY or create .kagura.json")
    return FilesClient.from_mcp_url(
        api_key=config.get("api_key", ""),
        mcp_url=config.get("mcp_url", "https://memory.kagura-ai.com/mcp"),
    )


def _run_files_command(
    operation: Callable[[FilesClient, str], Awaitable[Any]],
    context_id: str | None,
    *,
    needs_context: bool = True,
) -> None:
    """Execute a FilesClient operation with standard boilerplate.

    Validates api_key first (matching ``_run_client_command``'s order)
    so the operator sees a consistent error when both api_key and
    context_id are missing.
    """
    try:
        config = load_config()
        if not config.get("api_key"):
            raise click.ClickException(
                "No API key found. Set KAGURA_API_KEY or create .kagura.json"
            )

        ctx_id = ""
        if needs_context:
            ctx_id = context_id or config.get("context_id") or ""
            if not ctx_id:
                raise click.ClickException(
                    "context_id required. Use --context-id or set in .kagura.json"
                )

        client = _build_files_client(config)

        async def _run() -> Any:
            async with client:
                return await operation(client, ctx_id)

        result = asyncio.run(_run())
        if result is not None:
            click.echo(result)
    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(f"Error: {e}") from e


@main.group()
def files():
    """Upload, list, and manage files in Kagura Memory Cloud."""
    pass


@files.command(name="upload")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--context-id", "-c", help="Target context (workspace) UUID")
@click.option("--content-type", "-t", help="MIME type override (default: sniffed)")
def files_upload(path: Path, context_id: str | None, content_type: str | None):
    """
    Upload a file to Kagura Memory Cloud.

    Example:
      kagura files upload ./report.pdf --context-id ctx-uuid
    """

    async def op(client: FilesClient, ctx: str) -> str:
        result = await client.upload(
            context_id=ctx,
            source=path,
            content_type=content_type,
        )
        return result.model_dump_json(indent=2)

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
