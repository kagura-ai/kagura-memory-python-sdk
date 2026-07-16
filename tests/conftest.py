"""Shared pytest fixtures and helpers."""

from datetime import UTC, datetime, timedelta

import pytest

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


def bootstrap_envelope_dict(
    agent_id: str = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    context_id: str = "11111111-2222-3333-4444-555555555555",
) -> dict:
    """Build a server-shaped ``get_agent_bootstrap`` envelope for tests.

    Mirrors ``build_envelope`` in
    ``memory-cloud/backend/src/services/agent_bootstrap_service.py``
    (v0.49.0, RFC-0002 P0-3) so the MCP-surface tests (test_client.py) and
    the REST-surface tests (test_agents_client.py) consume one wire
    fixture instead of diverging copies. Per-case variants override only
    what they assert, e.g. ``{**bootstrap_envelope_dict(),
    "components": {...}}``.
    """
    return {
        "status": "success",
        "degraded": False,
        "agent": {
            "agent_id": agent_id,
            "name": "ci-agent",
            "binding": {"context_id": context_id, "is_default": True},
        },
        "context": {
            "id": context_id,
            "name": "dev",
            "display_name": "Dev",
            "summary": "Dev knowledge base",
            "usage_guide": "Recall before acting.",
            "is_private": True,
            "is_locked": False,
            "embedding_model": "text-embedding-3-small",
            "embedding_dimensions": 1536,
        },
        "instructions": "Recall before acting.\n\nSTANDARD INSTRUCTIONS",
        "components": {
            "pinned": {
                "status": "ok",
                "memories": [{"memory_id": "m1", "summary": "Guardrail", "type": "note"}],
                "total_available": 1,
                "truncated": False,
                "cap": 100,
            },
            "recall": {"status": "skipped", "reason": "no_query"},
            "state": {"status": "ok", "entries": []},
            "policy": {"status": "skipped", "reason": "no_policy_bundle"},
        },
        "correlation": {
            "agent_id": agent_id,
            "session_id": "run-42",
            "run_id": None,
            "trace_id": None,
            "span_id": None,
        },
        "generated_at": "2026-07-16T00:00:00Z",
    }


@pytest.fixture
def isolated_kagura_credentials(tmp_path, monkeypatch):
    """Isolate a test from real ``~/.kagura/credentials.json`` and env.

    CLI tests walk the canonical SDK chain (``env > OAuth profile >
    .kagura.json``), so a developer's stored OAuth profile or exported
    ``KAGURA_API_KEY`` would pre-empt config-only fixtures and silently
    change behavior. Shared home for the per-file autouse wrappers
    (test_cli_workspace.py / test_cli_auth_keys.py) — one body instead of
    a copy per file.
    """
    from kagura_memory.auth.credentials import reset_state_cache

    fake_path = tmp_path / "default-credentials.json"
    monkeypatch.setattr("kagura_memory.auth.credentials.DEFAULT_CREDENTIALS_PATH", fake_path)
    monkeypatch.delenv("KAGURA_API_KEY", raising=False)
    monkeypatch.delenv("KAGURA_PROFILE", raising=False)
    monkeypatch.delenv("KAGURA_MCP_URL", raising=False)
    reset_state_cache()
    yield
    reset_state_cache()
