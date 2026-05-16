"""Shared pytest fixtures and helpers."""

from datetime import UTC, datetime, timedelta

from kagura_memory.auth.credentials import OAuthCredentials


def make_oauth_creds(
    workspace_id: str = "00000000-0000-0000-0000-0000000000ff",
    expires_in_seconds: int = 3600,
    *,
    server: str = "https://memory.kagura-ai.com",
    access_token: str = "atok-test",
    workspace_name: str = "test-ws",
    user_email: str = "test@example.com",
) -> OAuthCredentials:
    """Build a usable ``OAuthCredentials`` fixture for any client test.

    Defaults match a healthy, non-expired credential bound to a workspace.
    Tests that need a specific scenario (near-expiry, missing workspace,
    etc.) override only the field they care about; everything else has
    sensible neutral defaults shared across client/CLI test suites.
    """
    return OAuthCredentials(
        server=server,
        mcp_url=f"{server.rstrip('/')}/mcp",
        client_id="kagura-cli",
        access_token=access_token,
        refresh_token="rtok-test",
        token_type="Bearer",
        expires_at=datetime.now(UTC) + timedelta(seconds=expires_in_seconds),
        scope="memory:read memory:write",
        workspace_id=workspace_id,
        workspace_name=workspace_name,
        user_email=user_email,
        issued_at=datetime.now(UTC),
    )


def sleep_report_summary_dict(report_id: str = "rid-1") -> dict:
    """Build a server-shaped Sleep Maintenance summary dict for tests.

    Mirrors ``_report_to_summary`` in
    ``memory-cloud/backend/src/mcp_server/tools/sleep.py`` so that both
    SDK-level and CLI-level tests can construct realistic ``SleepReport``
    fixtures without diverging.
    """
    return {
        "report_id": report_id,
        "context_id": "ctx-1",
        "status": "completed",
        "started_at": "2026-04-28T00:00:00",
        "completed_at": "2026-04-28T00:05:00",
        "memories_processed": 10,
        "edges_created": 2,
        "memories_merged": 1,
        "memories_promoted": 0,
        "llm_calls_made": 3,
        "llm_tokens_used": 1234,
    }
