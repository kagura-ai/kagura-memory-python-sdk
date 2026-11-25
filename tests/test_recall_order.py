"""recall(order=) — date order (memory-cloud v0.182.0, #3004).

``order="date"`` runs the same search and returns the rows oldest first, each
with ``date`` / ``date_source``, plus ``order: "date"`` on the envelope. The
argument is sent only when set, so a default call is the request it was
before, and an older server never sees it.
"""

from __future__ import annotations

import json
from typing import Any, get_args
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from kagura_memory import KaguraClient, RecallDateSource, RecallOrder

CTX = "11111111-2222-3333-4444-555555555555"
MEM_OLD = "aaaaaaaa-0000-0000-0000-000000000001"
MEM_NEW = "aaaaaaaa-0000-0000-0000-000000000002"

# Shaped like memory-cloud v0.182.0's date-ordered recall envelope.
DATE_REPLY: dict[str, Any] = {
    "confidence": {"level": "high", "top_score": 0.81, "result_count": 2},
    "status": "success",
    "results": [
        {
            "memory_id": MEM_OLD,
            "date": "2026-03-02",
            "date_source": "observed",
            "summary": "The team meets on Tuesdays",
            "type": "fact",
            "importance": 0.5,
            "score": 0.62,
            "created_at": "2026-03-05T00:00:00Z",
            "observed_at": "2026-03-02",
            "superseded_by": MEM_NEW,
        },
        {
            "memory_id": MEM_NEW,
            "date": "2026-09-14",
            "date_source": "created",
            "summary": "The team meets on Thursdays from September",
            "type": "fact",
            "importance": 0.6,
            "score": 0.81,
            "created_at": "2026-09-14T08:00:00Z",
        },
    ],
    "count": 2,
    "order": "date",
    "context_id": CTX,
    "context_name": "dev",
}

RELEVANCE_REPLY: dict[str, Any] = {
    "confidence": {"level": "high", "top_score": 0.81, "result_count": 1},
    "status": "success",
    "results": [{"memory_id": MEM_NEW, "summary": "Thursdays", "score": 0.81}],
    "count": 1,
    "context_id": CTX,
}


def _client() -> KaguraClient:
    client = KaguraClient(api_key="test", mcp_url="https://test.com/mcp")
    client._session_id = "pre-set-session"
    return client


def _wire_client(reply: dict[str, Any], sent: list[dict[str, Any]]) -> KaguraClient:
    """A client whose transport answers every tools/call with ``reply``."""

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


async def _sent_args(**kwargs: Any) -> dict[str, Any]:
    client = _client()
    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = {"results": []}
            await client.recall(context_id=CTX, query="team meeting day", **kwargs)
            return mock.call_args[0][1]
    finally:
        await client.close()


def test_order_literals_match_the_server():
    assert get_args(RecallOrder) == ("relevance", "date")
    assert get_args(RecallDateSource) == ("event", "observed", "created")


@pytest.mark.asyncio
async def test_default_recall_request_is_unchanged():
    """No order, no include_superseded: exactly the arguments sent before #3004."""
    assert await _sent_args() == {"query": "team meeting day", "k": 5, "context_id": CTX}


@pytest.mark.asyncio
@pytest.mark.parametrize("order", ["relevance", "date"])
async def test_order_is_sent_when_set(order):
    args = await _sent_args(order=order)
    assert args["order"] == order
    assert "include_superseded" not in args


@pytest.mark.asyncio
async def test_include_superseded_false_is_sent_only_with_order():
    """order="date" includes superseded rows unless told not to; plain recall never needed it."""
    assert "include_superseded" not in await _sent_args(include_superseded=False)
    args = await _sent_args(order="date", include_superseded=False)
    assert args["include_superseded"] is False
    assert args["order"] == "date"
    assert (await _sent_args(order="date", include_superseded=True))["include_superseded"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("order", ["chronological", "DATE", ""])
async def test_unknown_order_is_refused_before_any_request(order):
    client = _client()
    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            with pytest.raises(ValueError, match="order must be"):
                await client.recall(context_id=CTX, query="q", order=order)
            mock.assert_not_called()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_date_order_reply_keys_are_returned():
    sent: list[dict[str, Any]] = []
    client = _wire_client(DATE_REPLY, sent)
    try:
        result = await client.recall(context_id=CTX, query="team meeting day", order="date")
    finally:
        await client.close()
    assert sent[-1]["params"]["arguments"]["order"] == "date"
    assert result["order"] == "date"
    assert [r["date"] for r in result["results"]] == ["2026-03-02", "2026-09-14"]
    assert [r["date_source"] for r in result["results"]] == ["observed", "created"]
    assert result["results"][0]["superseded_by"] == MEM_NEW


@pytest.mark.asyncio
async def test_relevance_reply_has_none_of_the_date_keys():
    """A server that does not send them (older, or relevance order) breaks nothing."""
    sent: list[dict[str, Any]] = []
    client = _wire_client(RELEVANCE_REPLY, sent)
    try:
        result = await client.recall(context_id=CTX, query="team meeting day")
    finally:
        await client.close()
    assert "order" not in sent[-1]["params"]["arguments"]
    assert result.get("order") is None
    assert all("date" not in r and "date_source" not in r for r in result["results"])
