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
from .models import Message, ResourceEventRequest, Session
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
        agent = KaguraAgent(**agent_kwargs)
        result = asyncio.run(agent.process(session, deep=deep, verbose=verbose))
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
    tag_list = [t.strip() for t in tags.split(",")] if tags else None

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


@main.command()
def contexts():
    """
    List available contexts.

    Examples:
      kagura contexts
    """
    _run_client_command(
        lambda client, _: client.list_contexts(),
        context_id=None,
        needs_context=False,
    )


# =============================================================================
# Resource Token Commands
# =============================================================================


def _get_resource_client() -> tuple[dict[str, Any], ResourceClient]:
    """Load config and create ResourceClient."""
    config = load_config()
    if not config.get("api_key"):
        raise click.ClickException("No API key found. Set KAGURA_API_KEY or create .kagura.json")
    client = ResourceClient.from_mcp_url(
        api_key=config.get("api_key", ""),
        mcp_url=config.get("mcp_url", "https://memory.kagura-ai.com/mcp"),
    )
    return config, client


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
    try:
        _, client = _get_resource_client()

        async def _run() -> str:
            async with client:
                result = await client.list_tokens(resource_id=resource_id, limit=limit)
                return result.model_dump_json(indent=2)

        click.echo(asyncio.run(_run()))
    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(f"Error: {e}") from e


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
    try:
        _, client = _get_resource_client()

        async def _run() -> str:
            async with client:
                result = await client.create_token(
                    resource_id=resource_id,
                    description=description,
                    quota_events_per_hour=quota,
                )
                return result.model_dump_json(indent=2)

        click.echo(asyncio.run(_run()))
    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(f"Error: {e}") from e


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

    try:
        _, client = _get_resource_client()

        async def _run() -> str:
            async with client:
                result = await client.update_token(
                    token_id=token_id,
                    description=description,
                    quota_events_per_hour=quota,
                )
                return result.model_dump_json(indent=2)

        click.echo(asyncio.run(_run()))
    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(f"Error: {e}") from e


@resource_tokens.command(name="revoke")
@click.argument("token_id", type=int)
def tokens_revoke(token_id):
    """
    Revoke (soft-delete) a resource token.

    Examples:
      kagura resource tokens revoke 42
    """
    try:
        _, client = _get_resource_client()

        async def _run() -> None:
            async with client:
                await client.revoke_token(token_id)

        asyncio.run(_run())
        click.echo("Token revoked.")
    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(f"Error: {e}") from e


@resource.command(name="ingest")
@click.option("--resource-id", "-r", required=True, help="Resource ID")
@click.option("--api-key", "-k", required=True, help="Resource API key (X-Resource-API-Key)")
@click.option("--doc-id", required=True, help="Document ID")
@click.option("--op", type=click.Choice(["upsert", "delete"]), default="upsert", help="Operation")
@click.option("--version", "-v", type=int, help="Document version (>=1)")
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
        event = ResourceEventRequest(
            op=op,
            doc_id=doc_id,
            version=version,
            payload=parsed_payload,
            importance=importance,
        )

        _, client = _get_resource_client()

        async def _run() -> str:
            async with client:
                result = await client.ingest_event(resource_id, api_key, event)
                return result.model_dump_json(indent=2)

        click.echo(asyncio.run(_run()))
    except json.JSONDecodeError as e:
        raise click.ClickException(f"Invalid JSON payload: {e}") from e
    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(f"Error: {e}") from e


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
        if not isinstance(data, list):
            raise click.ClickException("JSON file must contain an array of events")

        events = [ResourceEventRequest(**e) for e in data]

        _, client = _get_resource_client()

        async def _run() -> str:
            async with client:
                result = await client.ingest_events(resource_id, api_key, events)
                return result.model_dump_json(indent=2)

        click.echo(asyncio.run(_run()))
    except json.JSONDecodeError as e:
        raise click.ClickException(f"Invalid JSON file: {e}") from e
    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(f"Error: {e}") from e


if __name__ == "__main__":
    main()
