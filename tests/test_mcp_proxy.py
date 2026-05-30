"""Tests for kagura_memory.mcp_proxy (the kagura-mcp stdio MCP proxy)."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from kagura_memory import mcp_proxy
from kagura_memory.mcp_proxy import _error_response, _Upstream, serve


class _FakeOAuth:
    """Stand-in for KaguraOAuth: records force_refresh calls."""

    def __init__(self) -> None:
        self.force_calls = 0

    async def force_refresh(self) -> None:
        self.force_calls += 1


def _upstream(handler: Any, oauth: _FakeOAuth | None = None) -> tuple[_Upstream, _FakeOAuth]:
    oauth = oauth or _FakeOAuth()
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return _Upstream(client, "https://test.example.com/mcp", oauth), oauth  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _Upstream.forward
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_forward_returns_response_and_captures_session_id():
    seen_headers: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.append(dict(request.headers))
        return httpx.Response(
            200,
            json={"jsonrpc": "2.0", "id": 1, "result": {"tools": []}},
            headers={"mcp-session-id": "sess-1"},
        )

    up, _ = _upstream(handler)
    result = await up.forward({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
    assert result == {"jsonrpc": "2.0", "id": 1, "result": {"tools": []}}

    # Session id captured, then replayed on the next request.
    second = await up.forward({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    assert second is not None
    assert "mcp-session-id" not in seen_headers[0]  # first request had none yet
    assert seen_headers[1]["mcp-session-id"] == "sess-1"


@pytest.mark.asyncio
async def test_forward_401_forces_refresh_and_retries_once():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(401, json={"error": "expired"})
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": "ok"})

    up, oauth = _upstream(handler)
    result = await up.forward({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {}})
    assert result == {"jsonrpc": "2.0", "id": 1, "result": "ok"}
    assert oauth.force_calls == 1  # refreshed exactly once
    assert calls["n"] == 2  # one retry


@pytest.mark.asyncio
async def test_forward_persistent_401_raises_after_single_retry():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "invalid_grant"})

    up, oauth = _upstream(handler)
    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await up.forward({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {}})
    assert exc_info.value.response.status_code == 401
    assert oauth.force_calls == 1  # forced refresh once, did not loop


@pytest.mark.asyncio
async def test_forward_notification_returns_none():
    """A 202/empty-body upstream ack (notification) yields no response to write."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(202)

    up, _ = _upstream(handler)
    result = await up.forward({"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert result is None


# ---------------------------------------------------------------------------
# _error_response
# ---------------------------------------------------------------------------


def test_error_response_with_id_builds_jsonrpc_error():
    resp = _error_response({"jsonrpc": "2.0", "id": 7, "method": "x"}, RuntimeError("nope"))
    assert resp is not None
    assert resp["id"] == 7
    assert resp["error"]["code"] == -32000
    assert "nope" in resp["error"]["message"]


def test_error_response_without_id_returns_none():
    """Notifications (no id) get no error reply — replying would break JSON-RPC."""
    assert _error_response({"jsonrpc": "2.0", "method": "notify"}, RuntimeError("x")) is None


def test_error_response_401_is_actionable():
    req = httpx.Request("POST", "https://test/mcp")
    exc = httpx.HTTPStatusError("401", request=req, response=httpx.Response(401, request=req))
    resp = _error_response({"jsonrpc": "2.0", "id": 1}, exc)
    assert resp is not None
    assert "kagura auth login" in resp["error"]["message"]


# ---------------------------------------------------------------------------
# serve loop
# ---------------------------------------------------------------------------


class _FakeUpstream:
    def __init__(self, responder: Any) -> None:
        self._responder = responder

    async def forward(self, message: dict[str, Any]) -> dict[str, Any] | None:
        return self._responder(message)


def _line_reader(lines: list[str]):
    queue = list(lines)

    async def read_line() -> str:
        return queue.pop(0) if queue else ""

    return read_line


@pytest.mark.asyncio
async def test_serve_forwards_and_writes_response():
    written: list[str] = []
    up = _FakeUpstream(lambda m: {"jsonrpc": "2.0", "id": m["id"], "result": "ok"})
    reader = _line_reader(['{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}\n'])
    await serve(up, reader, written.append)  # type: ignore[arg-type]
    assert len(written) == 1
    assert json.loads(written[0]) == {"jsonrpc": "2.0", "id": 1, "result": "ok"}


@pytest.mark.asyncio
async def test_serve_skips_blank_and_unparseable_lines():
    written: list[str] = []
    up = _FakeUpstream(lambda m: {"jsonrpc": "2.0", "id": m["id"], "result": "ok"})
    reader = _line_reader(["\n", "not json\n", '{"jsonrpc":"2.0","id":5,"method":"x"}\n'])
    await serve(up, reader, written.append)  # type: ignore[arg-type]
    assert len(written) == 1
    assert json.loads(written[0])["id"] == 5


@pytest.mark.asyncio
async def test_serve_writes_nothing_for_notification_response():
    written: list[str] = []
    up = _FakeUpstream(lambda m: None)  # forward returns None (notification)
    reader = _line_reader(['{"jsonrpc":"2.0","method":"notifications/initialized"}\n'])
    await serve(up, reader, written.append)  # type: ignore[arg-type]
    assert written == []


@pytest.mark.asyncio
async def test_serve_forward_exception_becomes_error_response():
    written: list[str] = []

    def boom(_message: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("upstream down")

    up = _FakeUpstream(boom)
    reader = _line_reader(['{"jsonrpc":"2.0","id":9,"method":"tools/call"}\n'])
    await serve(up, reader, written.append)  # type: ignore[arg-type]
    assert len(written) == 1
    err = json.loads(written[0])
    assert err["id"] == 9
    assert "upstream down" in err["error"]["message"]


# ---------------------------------------------------------------------------
# _amain — startup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_amain_exits_1_when_no_profile(monkeypatch: pytest.MonkeyPatch, capsys):
    monkeypatch.setattr(mcp_proxy, "get_shared_state", lambda profile=None: None)
    rc = await mcp_proxy._amain(["--profile", "default"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "kagura auth login" in err
    assert "--profile default" in err
