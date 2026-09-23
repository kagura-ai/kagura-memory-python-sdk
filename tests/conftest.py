"""Shared pytest fixtures and helpers."""

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
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
        # #1183 (server v0.43.0+): judge-LLM failures behind a degraded run.
        "llm_call_failures": 0,
    }


def sleep_report_detail_dict(report_id: str = "rid-1", **overrides) -> dict:
    """Build a server-shaped Sleep report detail dict for tests.

    Mirrors ``_report_to_detail`` in
    ``memory-cloud/backend/src/mcp_server/tools/sleep.py`` — the object the
    ``get_sleep_report`` tool returns under its ``report`` key. Per-case
    variants pass only the fields they assert, e.g. ``status="degraded"``.
    """
    return {
        **sleep_report_summary_dict(report_id),
        "memories_flagged": 0,
        "embedding_calls_made": 0,
        "error_message": None,
        "edge_discovery_result": None,
        "dedup_result": None,
        "importance_result": None,
        "consolidation_result": None,
        "reindex_result": None,
        "merge_retention_result": None,
        **overrides,
    }


def indexer_status_dict(**metrics) -> dict:
    """Build a server-shaped ``GET /api/v1/resources/{id}/indexer-status`` body.

    Mirrors ``IndexerStatusResponse`` in
    ``memory-cloud/backend/src/api/routes/resource_indexer.py``; ``metrics``
    overrides the per-run counters, e.g. ``skipped_reason=...``.
    """
    return {
        "resource_id": "products",
        "state": {
            "job_status": "idle",
            "last_run_at": "2026-09-20T00:00:00Z",
            "next_run_at": None,
            "active_version": 1,
            "last_offset": 42,
            "lag_seconds": 3.0,
            "metrics": {"applied_upserts": 0, "applied_deletes": 0, "errors": 0, **metrics},
        },
        "recent_events": [],
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


def agent_dict(
    agent_id: str = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    name: str = "ci-agent",
    **overrides,
) -> dict:
    """Build a server-shaped Agent Registry row for tests.

    Mirrors ``_serialize_agent`` in
    ``memory-cloud/backend/src/mcp_server/tools/agent_registry.py`` and
    the REST ``AgentResponse`` (v0.49.0, RFC-0002 P0-1) — MCP and REST
    emit the same field set, so both test suites consume one fixture.
    """
    row = {
        "id": agent_id,
        "workspace_id": "11111111-2222-3333-4444-555555555555",
        "name": name,
        "description": None,
        "owner_user_id": "google_123",
        "framework": "claude-code",
        "environment": "production",
        "version": None,
        "status": "active",
        "enforcement_mode": "enforce",
        "last_seen_at": None,
        "created_at": "2026-07-16T00:00:00Z",
        "updated_at": "2026-07-16T00:00:00Z",
    }
    row.update(overrides)
    return row


def agent_binding_dict(
    binding_id: str = "bbbbbbbb-cccc-dddd-eeee-ffffffffffff",
    agent_id: str = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    context_id: str = "11111111-2222-3333-4444-555555555555",
    **overrides,
) -> dict:
    """Build a server-shaped agent-context binding row for tests.

    Mirrors ``_serialize_binding`` (MCP) and the REST ``BindingResponse``
    (v0.49.0, RFC-0002 P0-2).
    """
    row = {
        "id": binding_id,
        "agent_id": agent_id,
        "context_id": context_id,
        "can_read": True,
        "write_policy": "deny",
        "is_default": False,
        "allowed_memory_types": None,
        "allowed_source_types": None,
        "created_by": "google_123",
        "created_at": "2026-07-16T00:00:00Z",
        "updated_at": "2026-07-16T00:00:00Z",
    }
    row.update(overrides)
    return row


@pytest.fixture(autouse=True)
def _isolate_claude_code(tmp_path_factory, monkeypatch):
    """Keep every test away from the developer's Claude Code setup (#258).

    ``kagura setup claude``, ``kagura doctor`` and ``kagura auth status`` read
    ``~/.claude.json`` and run the ``claude`` CLI (``plugin list``,
    ``mcp add-json``). Point ``CLAUDE_CONFIG_DIR`` (where the SDK, like Claude
    Code, looks for ``.claude.json``) at an empty directory and hide ``claude``
    so no test reads or changes the real configuration. Tests that need either
    write their own ``.claude.json`` there or patch ``claude_executable``.
    """
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path_factory.mktemp("claude-config")))
    monkeypatch.setattr("kagura_memory.claude_code.claude_executable", lambda: None)


def measurement_dict(**overrides) -> dict:
    """Build a server-shaped ``record_measurement`` success envelope for tests.

    Mirrors ``handle_record_measurement`` in
    ``memory-cloud/backend/src/mcp_server/tools/measurement.py`` (v0.54.0,
    #1333) so client and CLI tests share one wire fixture.
    """
    row = {
        "status": "success",
        "measurement_id": "cccccccc-dddd-eeee-ffff-000000000000",
        "metric": "weight_kg",
        "measured_at": "2026-09-01T07:30:00Z",
        "value": 71.5,
        "unit": "kg",
    }
    row.update(overrides)
    return row


def measurement_series_dict(**overrides) -> dict:
    """Build a server-shaped ``recall_series`` success envelope for tests.

    Mirrors ``handle_recall_series`` (v0.54.0, #1333): ``count`` is the
    number of (non-empty) buckets, each bucket's ``count`` the number of
    observations in it.
    """
    row = {
        "status": "success",
        "metric": "weight_kg",
        "period": "week",
        "agg": "avg",
        "series": [
            {"bucket": "2026-08-24T00:00:00Z", "value": 72.0, "count": 3},
            {"bucket": "2026-08-31T00:00:00Z", "value": 71.25, "count": 2},
        ],
        "count": 2,
    }
    row.update(overrides)
    return row


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


# ---------------------------------------------------------------------------
# In-memory MCP endpoint with expiring sessions (#252)
# ---------------------------------------------------------------------------

#: memory-cloud's reply to a request naming a session it no longer holds
#: (``backend/src/mcp_server/transport.py``, the ``POST /mcp`` session check).
SESSION_EXPIRED_BODY: dict[str, Any] = {
    "jsonrpc": "2.0",
    "error": {
        "code": -32603,
        "message": "MCP session not found or expired. Please re-initialize your connection.",
        "data": {"action": "Send a new 'initialize' request without Mcp-Session-Id header"},
    },
    "id": None,
}


class FakeMcpServer:
    """In-memory legacy MCP endpoint whose sessions live until :meth:`restart`.

    Models the MCP Streamable HTTP session contract as memory-cloud's
    ``POST /mcp`` session check implements it: ``initialize``, or any request
    that carries no session id, opens a session and returns its id in
    ``mcp-session-id``; a request naming an unknown session gets the ``404``
    :data:`SESSION_EXPIRED_BODY` before dispatch. (memory-cloud v0.75.0's
    deployed routes skip that check and re-adopt the unknown id instead; the
    SDK's recovery is for a server that enforces it.) Every request is
    recorded as ``(body, session_id)``.
    """

    def __init__(self) -> None:
        self.sessions: set[str] = set()
        self.requests: list[tuple[dict[str, Any], str | None]] = []
        self._opened = 0

    @staticmethod
    def tool_result(session_id: str) -> dict[str, Any]:
        """The ``tools/call`` result served in ``session_id``."""
        text = json.dumps({"status": "success", "session": session_id})
        return {"content": [{"type": "text", "text": text}]}

    def restart(self) -> None:
        """Drop every session, as an idle-hour expiry or a deploy does."""
        self.sessions.clear()

    def record(self, request: httpx.Request) -> dict[str, Any]:
        """Record ``request`` and return its JSON-RPC body."""
        body = json.loads(request.content)
        self.requests.append((body, request.headers.get("mcp-session-id")))
        return body

    def methods(self) -> list[str]:
        return [body["method"] for body, _ in self.requests]

    def calls(self) -> list[tuple[str, str | None]]:
        return [(body["method"], session_id) for body, session_id in self.requests]

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = self.record(request)
        session_id = request.headers.get("mcp-session-id")
        if body["method"] == "initialize" or session_id is None:
            self._opened += 1
            session_id = f"sess-{self._opened}"
            self.sessions.add(session_id)
        elif session_id not in self.sessions:
            return httpx.Response(404, json=SESSION_EXPIRED_BODY)
        if "id" not in body:
            return httpx.Response(202)
        is_initialize = body["method"] == "initialize"
        result = {"serverInfo": {}} if is_initialize else self.tool_result(session_id)
        return httpx.Response(
            200,
            json={"jsonrpc": "2.0", "id": body["id"], "result": result},
            headers={"mcp-session-id": session_id},
        )
