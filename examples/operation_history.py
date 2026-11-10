#!/usr/bin/env python3
"""Workspace operation history — a signed-in web session only (server v0.154.0+).

``WorkspaceClient.list_operation_history`` reads the workspace's operation
history one page at a time. It takes the ``kagura_session`` cookie of a
browser session signed in to the same server: an API key or an OAuth token
is refused before any request. The caller must be an owner or admin of the
workspace, on a plan with the ``operation_history`` feature (Pro and Max by
default).

The script never prints the cookie or the events themselves: only the page
counts and the retention window. History starts when the server began
recording for the workspace (no backfill) and is kept for ``retention_days``.

Usage:
    export KAGURA_SERVER="https://memory.example.com"
    export KAGURA_SESSION_COOKIE="<kagura_session cookie value>"
    export KAGURA_WORKSPACE_ID="<workspace uuid>"
    uv run python examples/operation_history.py
"""

import asyncio
import os
import sys
from datetime import UTC, datetime, timedelta

from kagura_memory import KaguraFeatureNotAvailableError, WorkspaceClient


async def main() -> None:
    server = os.getenv("KAGURA_SERVER")
    cookie = os.getenv("KAGURA_SESSION_COOKIE")
    workspace_id = os.getenv("KAGURA_WORKSPACE_ID")
    if not (server and cookie and workspace_id):
        print("Error: KAGURA_SERVER, KAGURA_SESSION_COOKIE and KAGURA_WORKSPACE_ID required")
        sys.exit(1)

    since = datetime.now(UTC) - timedelta(days=7)  # timezone-aware, as the server requires
    async with WorkspaceClient.from_web_session(base_url=server, session_cookie=cookie) as client:
        events = pages = 0
        cursor: str | None = None
        try:
            while True:
                page = await client.list_operation_history(
                    workspace_id, target_type="memory", since=since, limit=100, cursor=cursor
                )
                pages += 1
                events += len(page.items)
                cursor = page.next_cursor
                if cursor is None:
                    break
        except KaguraFeatureNotAvailableError as e:
            print(f"Operation history is not on this plan (required: {e.required_plan})")
            sys.exit(1)

    total = f"{page.total}+" if page.total_capped else str(page.total)
    print(f"Memory events in the last 7 days: {events} over {pages} page(s), total {total}")
    print(f"Retention: {page.retention_days} days")


if __name__ == "__main__":
    asyncio.run(main())
