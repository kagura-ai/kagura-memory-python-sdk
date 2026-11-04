"""Recall shape and response budgets (#298, #295).

memory-cloud changed what ``recall`` returns by default — compact hits from
v0.108.0 (#2002), content on the top results from v0.114.0 (#2124), a
character budget on recall / load_pinned / load_guardrails /
get_agent_bootstrap from v0.82.0 (#1743), ``explore_hints_omitted`` on a URL
without ``explore`` from v0.135.0 (#2470). The replies below are shaped like
memory-cloud v0.141.0's ``utils/recall_projection.py`` and
``mcp_server/tools/memory.py::_recall_envelope``.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from kagura_memory import AgentBootstrapResponse, KaguraClient
from tests.conftest import bootstrap_envelope_dict

CTX = "11111111-2222-3333-4444-555555555555"
MEM_A = "aaaaaaaa-0000-0000-0000-000000000001"
MEM_B = "aaaaaaaa-0000-0000-0000-000000000002"
MEM_C = "aaaaaaaa-0000-0000-0000-000000000003"

_CONFIDENCE = {
    "level": "high",
    "top_score": 0.71,
    "prominence": 0.52,
    "relative_margin": 3.1,
    "result_count": 2,
    "rationale": "Top hit stands well above the candidate-pool background.",
}
_CONTEXT_FIELDS = {
    "context_id": CTX,
    "context_name": "dev",
    "context_display_name": "Dev",
    "context_is_private": False,
    "context_is_locked": False,
}
_EXPLORE_NOT_LISTED = (
    "explore is not listed on this connection: add it to the tools= or add_tools= "
    "list in the MCP URL's query if it has one, otherwise add add_tools=explore or "
    "set profile=full."
)

# verbosity omitted (compact), full omitted (3 on trusted rows), budget forced
# the second hit's content out, and include_explore_hints on a core-profile URL.
COMPACT_REPLY: dict[str, Any] = {
    "confidence": _CONFIDENCE,
    "status": "success",
    "results": [
        {
            "memory_id": MEM_A,
            "summary": "Deploy target moved to ap-northeast-1 after the latency review; the "
            "old us-east-1 stack stays a warm standby until the cut-over…",
            "summary_cut": True,
            "content": "c" * 1500,
            "content_truncated": True,
            "content_next_offset": 1500,
            "type": "decision",
            "importance": 0.8,
            "score": 0.8123,
            "created_at": "2026-10-01T00:00:00Z",
            "updated_at": None,
            "folded": {"count": 1, "memory_ids": [MEM_C]},
        },
        {
            "memory_id": MEM_B,
            "summary": "Runbook: rotate the deploy key",
            "type": "note",
            "importance": 0.5,
            "score": 0.61,
            "created_at": "2026-09-01T00:00:00Z",
            "updated_at": "2026-09-02T00:00:00Z",
            "observed_at": "2026-08-30",
        },
    ],
    "count": 2,
    **_CONTEXT_FIELDS,
    "related_tags": [{"tag": "deploy", "count": 2}],
    "explore_hints_omitted": _EXPLORE_NOT_LISTED,
    "full_applied": 1,
}

# verbosity="full", full=1, the LLM selector on, and a cut budget.
FULL_REPLY: dict[str, Any] = {
    "confidence": {**_CONFIDENCE, "result_count": 1},
    "status": "success",
    "results": [
        {
            "memory_id": MEM_A,
            "summary": "Deploy target moved to ap-northeast-1 after the latency review; the "
            "old us-east-1 stack stays a warm standby until the cut-over in March, when "
            "the DNS weights flip.",
            "content": "The latency review … the DNS weights flip.",
            "spans": [{"start": 0, "end": 18}, {"start": 240, "end": 262}],
            "type": "decision",
            "importance": 0.8,
            "scope": "persistent",
            "score": 0.8123,
            "created_at": "2026-10-01T00:00:00Z",
            "updated_at": None,
            "version": 3,
            "tags": ["deploy", "aws"],
            "superseded_by": MEM_B,
        },
    ],
    "count": 1,
    "selector": {
        "applied": True,
        "model": "judge-small",
        "kept": 1,
        "dropped": 1,
        "dropped_memory_ids": [MEM_C],
    },
    **_CONTEXT_FIELDS,
    "context_summary_omitted": True,
    "truncated": True,
}


def _client() -> KaguraClient:
    client = KaguraClient(api_key="test", mcp_url="https://test.com/mcp")
    client._session_id = "pre-set-session"
    return client


def _wire_client(reply: dict[str, Any], sent: list[dict[str, Any]]) -> KaguraClient:
    """A client whose transport answers every tools/call with ``reply`` as the server sends it."""

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        text = json.dumps(reply, ensure_ascii=False, separators=(",", ":"))
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"content": [{"type": "text", "text": text}]},
            },
        )

    client = _client()
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


# ============================================================================
# recall(verbosity=, full=, max_chars=)
# ============================================================================


@pytest.mark.asyncio
async def test_recall_omits_verbosity_full_and_max_chars_by_default():
    """Unset, the server's defaults apply — and a pre-0.108.0 server never sees them."""
    client = _client()
    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = {"results": []}
            await client.recall(context_id=CTX, query="deploy")
            args = mock.call_args[0][1]
    finally:
        await client.close()
    for key in ("verbosity", "full", "max_chars"):
        assert key not in args


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("verbosity", "full", "max_chars"),
    [("full", 0, 10_000), ("compact", 100, 100_000), ("full", 3, 20_000)],
)
async def test_recall_sends_verbosity_full_and_max_chars_when_set(verbosity, full, max_chars):
    """``full=0`` is a value (summaries only), not "unset", so it goes on the wire."""
    client = _client()
    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = {"results": []}
            await client.recall(
                context_id=CTX, query="deploy", verbosity=verbosity, full=full, max_chars=max_chars
            )
            args = mock.call_args[0][1]
    finally:
        await client.close()
    assert args["verbosity"] == verbosity
    assert args["full"] == full
    assert args["max_chars"] == max_chars


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"verbosity": "verbose"}, "verbosity"),
        ({"verbosity": "FULL"}, "verbosity"),
        ({"full": -1}, "full"),
        ({"full": 101}, "full"),
        ({"full": True}, "full"),  # bool is not an int here, as on the server
        ({"full": 1.0}, "full"),
        ({"full": "3"}, "full"),
        ({"max_chars": 9_999}, "max_chars"),
        ({"max_chars": 100_001}, "max_chars"),
        ({"max_chars": True}, "max_chars"),
        ({"max_chars": "20000"}, "max_chars"),
    ],
)
async def test_recall_rejects_out_of_range_shape_args_before_any_request(kwargs, match):
    client = _client()
    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            with pytest.raises(ValueError, match=match):
                await client.recall(context_id=CTX, query="deploy", **kwargs)
            mock.assert_not_awaited()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_recall_returns_a_compact_v0141_reply_unchanged():
    """A compact hit has no tags / context_summary / scope — nothing in the SDK needs them."""
    sent: list[dict[str, Any]] = []
    client = _wire_client(COMPACT_REPLY, sent)
    try:
        result = await client.recall(context_id=CTX, query="deploy", include_explore_hints=True)
    finally:
        await client.close()

    assert result == COMPACT_REPLY
    assert next(iter(result)) == "confidence"  # the verdict is read before the hits
    arguments = sent[0]["params"]["arguments"]
    assert "verbosity" not in arguments and "full" not in arguments
    first, second = result["results"]
    assert first["summary_cut"] is True and first["content_next_offset"] == 1500
    # Content sits right after the summary it belongs to.
    assert list(first)[:4] == ["memory_id", "summary", "summary_cut", "content"]
    assert first["folded"] == {"count": 1, "memory_ids": [MEM_C]}
    assert "tags" not in first and "context_summary" not in first and "scope" not in first
    assert "content" not in second and second["observed_at"] == "2026-08-30"
    assert result["full_applied"] == 1
    assert "explore_hints" not in result
    assert "add_tools=explore" in result["explore_hints_omitted"]


@pytest.mark.asyncio
async def test_recall_returns_a_full_v0141_reply_unchanged():
    sent: list[dict[str, Any]] = []
    client = _wire_client(FULL_REPLY, sent)
    try:
        result = await client.recall(
            context_id=CTX, query="deploy", verbosity="full", full=1, max_chars=10_000
        )
    finally:
        await client.close()

    assert result == FULL_REPLY
    arguments = sent[0]["params"]["arguments"]
    assert (arguments["verbosity"], arguments["full"], arguments["max_chars"]) == (
        "full",
        1,
        10_000,
    )
    [hit] = result["results"]
    assert hit["tags"] == ["deploy", "aws"] and hit["scope"] == "persistent"
    assert hit["version"] == 3 and "summary_cut" not in hit
    assert hit["spans"][0] == {"start": 0, "end": 18}
    assert result["selector"]["dropped_memory_ids"] == [MEM_C]
    assert result["truncated"] is True and result["context_summary_omitted"] is True
    assert "related_tags" not in result  # omitted when empty since v0.93.0


# ============================================================================
# recall_nearby(include_details=)
# ============================================================================


@pytest.mark.asyncio
async def test_recall_nearby_omits_include_details_by_default():
    client = _client()
    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = {"results": []}
            await client.recall_nearby(context_id=CTX, lat=35.68, lon=139.76)
            await client.recall_nearby(context_id=CTX, lat=35.68, lon=139.76, include_details=False)
            calls = [c.args[1] for c in mock.call_args_list]
    finally:
        await client.close()
    assert all("include_details" not in args for args in calls)


@pytest.mark.asyncio
async def test_recall_nearby_sends_include_details_and_passes_omitted_markers_through():
    """With details, items past the 20k budget carry details_omitted + details_total_chars."""
    reply = {
        "status": "success",
        "results": [
            {
                "memory_id": MEM_A,
                "summary": "Coffee with Sato",
                "type": "note",
                "details": {"location": {"lat": 35.68, "lon": 139.76}},
                "distance_m": 12.5,
            },
            {
                "memory_id": MEM_B,
                "summary": "Site visit",
                "type": "note",
                "distance_m": 80.0,
                "details_omitted": True,
                "details_total_chars": 25_000,
            },
        ],
        **_CONTEXT_FIELDS,
    }
    sent: list[dict[str, Any]] = []
    client = _wire_client(reply, sent)
    try:
        result = await client.recall_nearby(
            context_id=CTX, lat=35.68, lon=139.76, include_details=True
        )
    finally:
        await client.close()
    assert sent[0]["params"]["arguments"]["include_details"] is True
    assert result == reply


@pytest.mark.asyncio
async def test_recall_nearby_default_items_carry_location():
    reply = {
        "status": "success",
        "results": [
            {
                "memory_id": MEM_A,
                "summary": "Coffee with Sato",
                "type": "note",
                "location": {"lat": 35.68, "lon": 139.76, "label": "Tokyo HQ"},
                "distance_m": 12.5,
            }
        ],
        **_CONTEXT_FIELDS,
    }
    client = _wire_client(reply, [])
    try:
        result = await client.recall_nearby(context_id=CTX, lat=35.68, lon=139.76)
    finally:
        await client.close()
    assert result["results"][0]["location"]["label"] == "Tokyo HQ"
    assert "details" not in result["results"][0]


# ============================================================================
# max_chars on load_pinned / get_agent_bootstrap
# ============================================================================


@pytest.mark.asyncio
async def test_load_pinned_sends_max_chars_only_when_set():
    client = _client()
    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = {"status": "success", "memories": [], "truncated": False}
            await client.load_pinned(context_id=CTX)
            assert "max_chars" not in mock.call_args[0][1]
            await client.load_pinned(context_id=CTX, cap=500, max_chars=50_000)
            assert mock.call_args[0][1] == {"context_id": CTX, "cap": 500, "max_chars": 50_000}
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("max_chars", [0, 9_999, 100_001, True, 20_000.0])
async def test_load_pinned_rejects_out_of_range_max_chars(max_chars):
    client = _client()
    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            with pytest.raises(ValueError, match="max_chars"):
                await client.load_pinned(context_id=CTX, max_chars=max_chars)
            mock.assert_not_awaited()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_load_pinned_returns_the_budget_markers():
    reply = {
        "status": "success",
        "memories": [
            {
                "memory_id": MEM_A,
                "summary": "Goal: ship the importer",
                "type": "goal",
                "importance": 0.9,
                "delivery_mode": "always",
            }
        ],
        "total_available": 40,
        "truncated": True,
        "cap": 100,
        **_CONTEXT_FIELDS,
        "context_summary_omitted": True,
    }
    client = _wire_client(reply, [])
    try:
        result = await client.load_pinned(context_id=CTX, max_chars=10_000)
    finally:
        await client.close()
    assert result["truncated"] is True and result["context_summary_omitted"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("max_chars", [9_999, 100_001, False])
async def test_get_agent_bootstrap_rejects_out_of_range_max_chars(max_chars):
    client = _client()
    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            with pytest.raises(ValueError, match="max_chars"):
                await client.get_agent_bootstrap("agent-uuid", max_chars=max_chars)
            mock.assert_not_awaited()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_get_agent_bootstrap_reads_context_summary_omitted():
    envelope = {
        **bootstrap_envelope_dict(),
        "context_summary_omitted": True,
    }
    envelope["components"]["pinned"]["truncated"] = True
    client = _client()
    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = envelope
            result = await client.get_agent_bootstrap("agent-uuid", max_chars=10_000)
            assert mock.call_args[0][1] == {"agent_id": "agent-uuid", "max_chars": 10_000}
    finally:
        await client.close()
    assert isinstance(result, AgentBootstrapResponse)
    assert result.context_summary_omitted is True
    assert result.components["pinned"]["truncated"] is True


def test_bootstrap_envelope_without_the_marker_reads_false():
    assert (
        AgentBootstrapResponse.model_validate(bootstrap_envelope_dict()).context_summary_omitted
        is False
    )
