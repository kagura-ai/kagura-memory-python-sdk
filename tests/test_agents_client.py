"""Tests for AgentsClient (#231) — the agent-bootstrap REST companion.

Wire shapes assert the memory-cloud v0.49.0 contract verified against the
server source (RFC-0002 P0-3, memory-cloud #1276): all-optional JSON body,
``agent_id`` in the URL path, the composed envelope with fail-soft
per-component statuses, and the uniform 404 (CWE-639) for agent/context.
"""

import json

import httpx
import pytest

from kagura_memory.agents_client import AgentsClient, _normalize_agent_id
from kagura_memory.exceptions import (
    KaguraAuthError,
    KaguraConnectionError,
    KaguraNotFoundError,
)
from kagura_memory.models import AgentBootstrapResponse

AGENT = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
CTX = "11111111-2222-3333-4444-555555555555"

ENVELOPE = {
    "status": "success",
    "degraded": False,
    "agent": {
        "agent_id": AGENT,
        "name": "ci-agent",
        "binding": {"context_id": CTX, "is_default": True},
    },
    "context": {
        "id": CTX,
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
            "memories": [],
            "total_available": 0,
            "truncated": False,
            "cap": 100,
        },
        "recall": {"status": "skipped", "reason": "no_query"},
        "policy": {"status": "skipped", "reason": "no_policy_bundle"},
    },
    "correlation": {
        "agent_id": AGENT,
        "session_id": "run-42",
        "run_id": None,
        "trace_id": None,
        "span_id": None,
    },
    "generated_at": "2026-07-16T00:00:00Z",
}


def make_client(handler) -> AgentsClient:
    client = AgentsClient(api_key="kagura_test")
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer kagura_test"},
    )
    return client


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def test_envelope_parses_and_ignores_unknown_fields():
    payload = {**ENVELOPE, "some_future_field": "ignored"}
    resp = AgentBootstrapResponse.model_validate(payload)
    assert resp.agent.agent_id == AGENT
    assert resp.agent.binding is not None and resp.agent.binding.context_id == CTX
    assert resp.context is not None and resp.context.embedding_dimensions == 1536
    # Component payloads survive verbatim as dicts.
    assert resp.components["policy"] == {"status": "skipped", "reason": "no_policy_bundle"}
    assert resp.generated_at is not None


def test_envelope_minimal_shape():
    # Only agent is required; a maximally degraded response must still parse.
    resp = AgentBootstrapResponse.model_validate(
        {"status": "success", "degraded": True, "agent": {"agent_id": AGENT, "name": "a"}}
    )
    assert resp.degraded is True
    assert resp.context is None and resp.components == {}


# ---------------------------------------------------------------------------
# agent_id normalization
# ---------------------------------------------------------------------------


def test_normalize_agent_id_canonical():
    assert _normalize_agent_id(AGENT) == AGENT


def test_normalize_agent_id_noncanonical_spellings():
    assert _normalize_agent_id("{" + AGENT + "}") == AGENT
    assert _normalize_agent_id(AGENT.replace("-", "")) == AGENT


def test_normalize_agent_id_rejects_garbage():
    with pytest.raises(ValueError, match="agent_id must be a UUID"):
        _normalize_agent_id("not-a-uuid")


# ---------------------------------------------------------------------------
# bootstrap()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bootstrap_minimal_sends_empty_body():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=ENVELOPE)

    async with make_client(handler) as client:
        result = await client.bootstrap(AGENT)

    assert seen["method"] == "POST"
    assert seen["path"] == f"/api/v1/agents/{AGENT}/bootstrap"
    assert seen["body"] == {}  # all-optional body; server applies defaults
    assert isinstance(result, AgentBootstrapResponse)
    assert result.agent.name == "ci-agent"


@pytest.mark.asyncio
async def test_bootstrap_full_body():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=ENVELOPE)

    async with make_client(handler) as client:
        await client.bootstrap(
            AGENT,
            context_id=CTX,
            session_id="run-42",
            query="current task",
            recall_k=7,
            pinned_cap=50,
            upcoming_until="2026-08-01T00:00:00",
            include=["pinned", "recall"],
        )

    assert seen["body"] == {
        "context_id": CTX,
        "session_id": "run-42",
        "query": "current task",
        "recall_k": 7,
        "pinned_cap": 50,
        "upcoming_until": "2026-08-01T00:00:00",
        "include": ["pinned", "recall"],
    }


@pytest.mark.asyncio
async def test_bootstrap_normalizes_agent_id_in_path():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        return httpx.Response(200, json=ENVELOPE)

    async with make_client(handler) as client:
        await client.bootstrap(AGENT.replace("-", ""))

    assert seen["path"] == f"/api/v1/agents/{AGENT}/bootstrap"


@pytest.mark.asyncio
async def test_bootstrap_rejects_non_uuid_before_request():
    async with make_client(lambda r: httpx.Response(200, json=ENVELOPE)) as client:
        with pytest.raises(ValueError, match="agent_id must be a UUID"):
            await client.bootstrap("../../admin")


@pytest.mark.asyncio
async def test_bootstrap_404_raises_not_found():
    # Uniform 404 — nonexistent and not-yours are indistinguishable (CWE-639).
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "Agent not found"})

    async with make_client(handler) as client:
        with pytest.raises(KaguraNotFoundError, match="Agent not found"):
            await client.bootstrap(AGENT)


@pytest.mark.asyncio
async def test_bootstrap_400_raises_connection_error_with_detail():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"detail": "'session_id' allows only [A-Za-z0-9._-]."})

    async with make_client(handler) as client:
        with pytest.raises(KaguraConnectionError, match="session_id"):
            await client.bootstrap(AGENT, session_id="bad session")


@pytest.mark.asyncio
async def test_bootstrap_401_raises_auth_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "unauthorized"})

    async with make_client(handler) as client:
        with pytest.raises(KaguraAuthError):
            await client.bootstrap(AGENT)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_constructor_requires_credentials():
    with pytest.raises(ValueError, match="from_mcp_url"):
        AgentsClient()


def test_constructor_rejects_plain_http():
    with pytest.raises(Exception, match="HTTPS"):
        AgentsClient(api_key="k", base_url="http://memory.example.com")
