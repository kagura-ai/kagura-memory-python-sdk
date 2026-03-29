"""CLI for Kagura Memory SDK."""

import asyncio
import json
import sys
from collections.abc import Awaitable, Callable
from typing import Any

import click

from .agent import KaguraAgent
from .client import KaguraClient
from .config import load_config
from .models import Message, ProcessResult, ResourceEventRequest, Session
from .resource_client import ResourceClient

# =============================================================================
# Helper Functions
# =============================================================================


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
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
    tag_list = tag_list or None

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


@context.command(name="update")
@click.argument("context_id")
@click.option("--display-name", help="Updated display name")
@click.option("--description", "-d", help="Updated description")
@click.option("--summary", "-s", help="Updated LLM-oriented summary")
@click.option("--usage-guide", help="Updated LLM-oriented usage guidelines")
def context_update(context_id, display_name, description, summary, usage_guide):
    """
    Update a context's settings.

    Examples:
      kagura context update CTX_UUID -s "Updated summary"
      kagura context update CTX_UUID --usage-guide "Store only code snippets"
    """
    if all(v is None for v in (display_name, description, summary, usage_guide)):
        raise click.ClickException("At least one update option is required")

    _run_client_command(
        lambda client, _: client.update_context(
            context_id=context_id,
            display_name=display_name,
            description=description,
            summary=summary,
            usage_guide=usage_guide,
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


if __name__ == "__main__":
    main()
