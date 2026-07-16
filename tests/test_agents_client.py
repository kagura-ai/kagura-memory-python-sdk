"""Tests for AgentsClient (#231) — the agent-bootstrap REST companion.

Wire shapes assert the memory-cloud v0.49.0 contract verified against the
server source (RFC-0002 P0-3, memory-cloud #1276): all-optional JSON body,
``agent_id`` in the URL path, the composed envelope with fail-soft
per-component statuses, and the uniform 404 (CWE-639) for agent/context.
"""

import json

import httpx
import pytest

from kagura_memory.agents_client import AgentsClient
from kagura_memory.exceptions import (
    KaguraAuthError,
    KaguraConnectionError,
    KaguraNotFoundError,
)
from kagura_memory.models import Agent, AgentBinding, AgentBootstrapResponse
from tests.conftest import agent_binding_dict, agent_dict, bootstrap_envelope_dict

AGENT = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
CTX = "11111111-2222-3333-4444-555555555555"

ENVELOPE = bootstrap_envelope_dict(agent_id=AGENT, context_id=CTX)


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
# bootstrap()
# ---------------------------------------------------------------------------
# agent_id normalization/rejection is the shared normalize_uuid helper —
# its spelling matrix is unit-tested once in tests/test__http.py; here only
# the client-level behavior (path normalization, pre-request rejection) is
# asserted.


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


# ---------------------------------------------------------------------------
# Registry CRUD (#235, RFC-0002 P0-1 — owner/admin only)
# ---------------------------------------------------------------------------

BINDING = "bbbbbbbb-cccc-dddd-eeee-ffffffffffff"


def test_agent_model_ignores_unknown_fields():
    agent = Agent.model_validate({**agent_dict(), "some_future_field": "ignored"})
    assert agent.id == AGENT
    assert agent.status == "active" and agent.enforcement_mode == "enforce"
    assert agent.last_seen_at is None


def test_agent_binding_model_ignores_unknown_fields():
    binding = AgentBinding.model_validate({**agent_binding_dict(), "future": 1})
    assert binding.write_policy == "deny"
    assert binding.allowed_memory_types is None  # reserved for #1286


@pytest.mark.asyncio
async def test_register_agent_posts_body_and_parses_201():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(201, json=agent_dict())

    async with make_client(handler) as client:
        agent = await client.register_agent("ci-agent", framework="claude-code")

    assert seen["method"] == "POST"
    assert seen["path"] == "/api/v1/agents"
    assert seen["body"] == {"name": "ci-agent", "framework": "claude-code"}
    assert isinstance(agent, Agent) and agent.name == "ci-agent"


@pytest.mark.asyncio
async def test_list_agents_unwraps_envelope():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"agents": [agent_dict()], "count": 1})

    async with make_client(handler) as client:
        agents = await client.list_agents()

    assert len(agents) == 1 and agents[0].id == AGENT


@pytest.mark.asyncio
async def test_list_agents_rejects_malformed_envelope():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"agents": None})

    async with make_client(handler) as client:
        with pytest.raises(KaguraConnectionError, match="agents"):
            await client.list_agents()


@pytest.mark.asyncio
async def test_get_agent_normalizes_id_in_path():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        return httpx.Response(200, json=agent_dict())

    async with make_client(handler) as client:
        await client.get_agent(AGENT.replace("-", ""))

    assert seen["path"] == f"/api/v1/agents/{AGENT}"


@pytest.mark.asyncio
async def test_update_agent_patches_set_fields_only():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=agent_dict(status="suspended"))

    async with make_client(handler) as client:
        agent = await client.update_agent(AGENT, status="suspended")

    assert seen["method"] == "PATCH"
    assert seen["body"] == {"status": "suspended"}
    assert agent.status == "suspended"


@pytest.mark.asyncio
async def test_update_agent_rejects_empty_update():
    """No-op REST update_agent() fails fast — never sends an empty PATCH body."""
    seen = {"called": False}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["called"] = True
        return httpx.Response(200, json=agent_dict())

    async with make_client(handler) as client:
        with pytest.raises(ValueError, match="at least one field"):
            await client.update_agent(AGENT)

    assert seen["called"] is False


@pytest.mark.asyncio
async def test_update_binding_rejects_empty_update():
    """No-op REST update_binding() fails fast — never sends an empty PATCH body."""
    seen = {"called": False}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["called"] = True
        return httpx.Response(200, json=agent_binding_dict())

    async with make_client(handler) as client:
        with pytest.raises(ValueError, match="at least one of"):
            await client.update_binding(AGENT, BINDING)

    assert seen["called"] is False


@pytest.mark.asyncio
async def test_delete_agent_returns_none_on_204():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        return httpx.Response(204)

    async with make_client(handler) as client:
        assert await client.delete_agent(AGENT) is None

    assert seen["method"] == "DELETE"
    assert seen["path"] == f"/api/v1/agents/{AGENT}"


# ---------------------------------------------------------------------------
# Context bindings (#235, RFC-0002 P0-2 — owner/admin only)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bind_context_posts_normalized_body():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(201, json=agent_binding_dict(is_default=True))

    async with make_client(handler) as client:
        binding = await client.bind_context(
            AGENT, CTX.replace("-", ""), write_policy="direct", is_default=True
        )

    assert seen["path"] == f"/api/v1/agents/{AGENT}/bindings"
    # context_id is canonicalized in the BODY too, not just URL paths.
    assert seen["body"] == {
        "context_id": CTX,
        "write_policy": "direct",
        "is_default": True,
    }
    assert isinstance(binding, AgentBinding) and binding.is_default is True


@pytest.mark.asyncio
async def test_list_bindings_unwraps_envelope():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"bindings": [agent_binding_dict()], "count": 1})

    async with make_client(handler) as client:
        bindings = await client.list_bindings(AGENT)

    assert len(bindings) == 1 and bindings[0].agent_id == AGENT


@pytest.mark.asyncio
async def test_update_binding_patches_both_ids_normalized():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=agent_binding_dict(can_read=False))

    async with make_client(handler) as client:
        binding = await client.update_binding(
            AGENT.replace("-", ""), "{" + BINDING + "}", can_read=False
        )

    assert seen["method"] == "PATCH"
    assert seen["path"] == f"/api/v1/agents/{AGENT}/bindings/{BINDING}"
    assert seen["body"] == {"can_read": False}
    assert binding.can_read is False


@pytest.mark.asyncio
async def test_unbind_context_deletes_204():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        return httpx.Response(204)

    async with make_client(handler) as client:
        assert await client.unbind_context(AGENT, BINDING) is None

    assert seen["method"] == "DELETE"
    assert seen["path"] == f"/api/v1/agents/{AGENT}/bindings/{BINDING}"


@pytest.mark.asyncio
async def test_registry_403_maps_to_connection_error():
    """Non-admin key on the owner/admin-gated registry surface → 403 detail passes through."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"detail": "Insufficient permissions"})

    async with make_client(handler) as client:
        with pytest.raises(KaguraConnectionError, match="403"):
            await client.register_agent("ci-agent")
