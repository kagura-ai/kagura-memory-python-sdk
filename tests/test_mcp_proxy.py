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
# _Upstream.forward — expired upstream session (#252)
# ---------------------------------------------------------------------------

_SESSION_EXPIRED_BODY: dict[str, Any] = {
    "jsonrpc": "2.0",
    "error": {
        "code": -32603,
        "message": "MCP session not found or expired. Please re-initialize your connection.",
        "data": {"action": "Send a new 'initialize' request without Mcp-Session-Id header"},
    },
    "id": None,
}

_INITIALIZE: dict[str, Any] = {
    "jsonrpc": "2.0",
    "id": 0,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {"roots": {}},
        "clientInfo": {"name": "claude-code", "version": "2.0.0"},
    },
}
_INITIALIZED: dict[str, Any] = {"jsonrpc": "2.0", "method": "notifications/initialized"}


def _tool_call(msg_id: int) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "method": "tools/call",
        "params": {"name": "list_contexts", "arguments": {}},
    }


class _FakeMcpServer:
    """In-memory legacy MCP endpoint whose sessions live until :meth:`restart`.

    Mirrors memory-cloud's transport: ``initialize`` opens a session (returned
    in ``mcp-session-id``); any other request naming an unknown session gets
    the 404 session-expired reply before dispatch. Every request is recorded
    as ``(body, session_id)``.
    """

    def __init__(self) -> None:
        self.sessions: set[str] = set()
        self.requests: list[tuple[dict[str, Any], str | None]] = []
        self._opened = 0

    def restart(self) -> None:
        self.sessions.clear()

    def methods(self) -> list[str]:
        return [body["method"] for body, _ in self.requests]

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        session_id = request.headers.get("mcp-session-id")
        self.requests.append((body, session_id))
        if body["method"] == "initialize":
            self._opened += 1
            new_id = f"sess-{self._opened}"
            self.sessions.add(new_id)
            return httpx.Response(
                200,
                json={"jsonrpc": "2.0", "id": body["id"], "result": {"serverInfo": {}}},
                headers={"mcp-session-id": new_id},
            )
        if session_id not in self.sessions:
            return httpx.Response(404, json=_SESSION_EXPIRED_BODY)
        if "id" not in body:
            return httpx.Response(202)
        return httpx.Response(
            200,
            json={"jsonrpc": "2.0", "id": body["id"], "result": {"session": session_id}},
        )


async def _open_session(up: _Upstream) -> None:
    """Run the downstream handshake Claude Code performs at startup."""
    assert (await up.forward(_INITIALIZE)) is not None
    assert (await up.forward(_INITIALIZED)) is None


@pytest.mark.asyncio
async def test_forward_expired_session_replays_initialize_and_retries_once():
    server = _FakeMcpServer()
    up, _ = _upstream(server.handler)
    await _open_session(up)
    server.restart()  # idle-hour expiry or a deploy drops every session

    result = await up.forward(_tool_call(7))

    assert result == {"jsonrpc": "2.0", "id": 7, "result": {"session": "sess-2"}}
    assert server.methods()[2:] == [
        "tools/call",  # 404: session gone
        "initialize",  # the downstream's own initialize, replayed once
        "notifications/initialized",
        "tools/call",  # retried once, on the new session
    ]
    replayed, replay_session = server.requests[3]
    assert replayed == _INITIALIZE  # byte-for-byte what the downstream sent
    assert replay_session is None  # a new session is opened without the stale id
    assert server.requests[4][1] == "sess-2"
    assert server.requests[5][1] == "sess-2"


@pytest.mark.asyncio
async def test_forward_next_tool_call_after_recovery_needs_no_replay():
    server = _FakeMcpServer()
    up, _ = _upstream(server.handler)
    await _open_session(up)
    server.restart()
    await up.forward(_tool_call(1))

    result = await up.forward(_tool_call(2))

    assert result == {"jsonrpc": "2.0", "id": 2, "result": {"session": "sess-2"}}
    assert server.methods().count("initialize") == 2  # the handshake + one replay


@pytest.mark.asyncio
async def test_forward_persistent_404_forwards_server_error_after_single_replay():
    """A retry that 404s too reaches the downstream as the server's own error, once."""
    server = _FakeMcpServer()

    def handler(request: httpx.Request) -> httpx.Response:
        response = server.handler(request)
        if json.loads(request.content)["method"] == "notifications/initialized":
            server.restart()  # every replayed session dies before the retry lands
        return response

    up, _ = _upstream(handler)
    await _open_session(up)

    result = await up.forward(_tool_call(9))

    assert result == {**_SESSION_EXPIRED_BODY, "id": 9}  # downstream id put back
    assert server.methods().count("initialize") == 2  # replayed once, no loop
    assert server.methods().count("tools/call") == 2


@pytest.mark.asyncio
async def test_forward_replayed_initialize_failure_raises():
    """If the replayed initialize fails, the message is not re-sent and the error surfaces."""
    server = _FakeMcpServer()
    up, _ = _upstream(server.handler)
    await _open_session(up)
    server.restart()

    def down(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content)["method"] == "initialize":
            server.requests.append((json.loads(request.content), None))
            return httpx.Response(503, text="Service Unavailable")
        return server.handler(request)

    up._http = httpx.AsyncClient(transport=httpx.MockTransport(down))
    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await up.forward(_tool_call(3))
    assert exc_info.value.response.status_code == 503
    assert server.methods()[2:] == ["tools/call", "initialize"]


@pytest.mark.asyncio
async def test_forward_404_without_session_is_not_retried():
    """No session was sent, so none can have expired: the 404 is forwarded as-is."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content)["method"])
        return httpx.Response(404, json=_SESSION_EXPIRED_BODY)

    up, _ = _upstream(handler)
    result = await up.forward(_tool_call(4))
    assert result == {**_SESSION_EXPIRED_BODY, "id": 4}
    assert calls == ["tools/call"]


@pytest.mark.asyncio
async def test_forward_modern_method_not_found_404_is_not_a_session_error():
    """memory-cloud's stateless 404 + -32601 (#1544) ignores the session: no replay."""
    server = _FakeMcpServer()
    not_found = {"jsonrpc": "2.0", "id": 5, "error": {"code": -32601, "message": "nope"}}

    def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content)["method"] == "subscriptions/listen":
            server.requests.append((json.loads(request.content), None))
            return httpx.Response(404, json=not_found)
        return server.handler(request)

    up, _ = _upstream(handler)
    await _open_session(up)
    result = await up.forward({"jsonrpc": "2.0", "id": 5, "method": "subscriptions/listen"})
    assert result == not_found
    assert server.methods().count("initialize") == 1


@pytest.mark.asyncio
async def test_forward_downstream_initialize_is_sent_without_stale_session():
    """A downstream re-initialize opens a new session instead of 404ing on the old id."""
    server = _FakeMcpServer()
    up, _ = _upstream(server.handler)
    await _open_session(up)
    server.restart()

    assert (await up.forward({**_INITIALIZE, "id": 10})) is not None
    assert server.requests[-1][1] is None
    assert (await up.forward(_tool_call(11)))["result"] == {"session": "sess-2"}  # type: ignore[index]
    assert server.methods().count("initialize") == 2


@pytest.mark.asyncio
async def test_forward_failed_initialize_is_not_cached_for_replay():
    """Only an initialize the upstream accepted is replayed after an expiry."""
    server = _FakeMcpServer()
    rejected = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"bad": True}}

    def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content).get("params") == {"bad": True}:
            return httpx.Response(
                400, json={"jsonrpc": "2.0", "id": 1, "error": {"code": -32602, "message": "x"}}
            )
        return server.handler(request)

    up, _ = _upstream(handler)
    await _open_session(up)
    assert (await up.forward(rejected))["error"]["code"] == -32602  # type: ignore[index]
    server.restart()

    await up.forward(_tool_call(2))
    initializes = [body for body, _ in server.requests if body["method"] == "initialize"]
    assert initializes == [_INITIALIZE, _INITIALIZE]  # the good one, replayed


# ---------------------------------------------------------------------------
# _Upstream.forward — JSON-RPC error bodies on non-2xx (#252)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "body"),
    [
        pytest.param(
            400,
            {
                "jsonrpc": "2.0",
                "error": {"code": -32600, "message": "Invalid Request: missing method"},
                "id": 8,
            },
            id="invalid-request",
        ),
        pytest.param(
            400,
            {
                "jsonrpc": "2.0",
                "id": 8,
                "error": {
                    "code": -32022,
                    "message": "Unsupported protocol version",
                    "data": {"supported": ["2026-07-28", "2025-06-18"], "requested": "1999"},
                },
            },
            id="unsupported-protocol-version",
        ),
    ],
)
async def test_forward_jsonrpc_error_body_passes_through_unchanged(
    status: int, body: dict[str, Any]
):
    up, _ = _upstream(lambda request: httpx.Response(status, json=body))
    assert await up.forward(_tool_call(8)) == body


@pytest.mark.asyncio
async def test_forward_jsonrpc_error_with_null_id_gets_downstream_id():
    body = {"jsonrpc": "2.0", "error": {"code": -32600, "message": "batch"}, "id": None}
    up, _ = _upstream(lambda request: httpx.Response(400, json=body))
    assert await up.forward(_tool_call(12)) == {**body, "id": 12}


@pytest.mark.asyncio
async def test_forward_jsonrpc_error_for_notification_writes_nothing():
    """A notification never gets a reply, even when the upstream answered it with an error."""
    body = {"jsonrpc": "2.0", "error": {"code": -32600, "message": "x"}, "id": None}
    up, _ = _upstream(lambda request: httpx.Response(400, json=body))
    assert await up.forward({"jsonrpc": "2.0", "method": "notifications/cancelled"}) is None


@pytest.mark.asyncio
async def test_forward_non_jsonrpc_error_body_still_raises():
    up, _ = _upstream(lambda request: httpx.Response(502, text="Bad Gateway"))
    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await up.forward(_tool_call(1))
    assert exc_info.value.response.status_code == 502


@pytest.mark.asyncio
async def test_forward_401_with_jsonrpc_body_keeps_refresh_and_relogin_path():
    """A 401 is never passed through: it refreshes once, then raises for the re-login hint."""
    body = {"jsonrpc": "2.0", "error": {"code": -32001, "message": "unauthorized"}, "id": None}
    up, oauth = _upstream(lambda request: httpx.Response(401, json=body))
    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await up.forward(_tool_call(1))
    assert oauth.force_calls == 1
    error = _error_response(_tool_call(1), exc_info.value)
    assert error is not None
    assert "kagura auth login" in error["error"]["message"]


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
async def test_serve_skips_non_object_messages():
    """A batch array or bare scalar has no single id to answer: drop it, keep serving.

    Before #252 a batch reached ``_error_response``'s ``message.get`` on the
    upstream's 400 and the AttributeError killed the bridge.
    """
    written: list[str] = []
    up = _FakeUpstream(lambda m: {"jsonrpc": "2.0", "id": m["id"], "result": "ok"})
    reader = _line_reader(['[{"jsonrpc":"2.0","id":1,"method":"x"}]\n', "42\n", '{"id":2}\n'])
    await serve(up, reader, written.append)  # type: ignore[arg-type]
    assert [json.loads(line)["id"] for line in written] == [2]


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


def _build_state(tmp_path):
    """A real _SharedCredentialsState backed by a tmp credentials file."""
    from datetime import UTC, datetime, timedelta

    from kagura_memory.auth.credentials import (
        CredentialsFile,
        OAuthCredentials,
        get_shared_state,
        reset_state_cache,
        save_credentials_file,
    )

    reset_state_cache()
    path = tmp_path / "creds.json"
    creds = OAuthCredentials(
        server="https://test.example.com",
        mcp_url="https://test.example.com/mcp",
        client_id="kagura-cli",
        access_token="atok",
        refresh_token="rtok",
        token_type="Bearer",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        scope="memory:read",
        workspace_id="ws-1",
        workspace_name="ws",
        user_email="u@example.com",
        issued_at=datetime.now(UTC),
    )
    cf = CredentialsFile()
    cf.set_profile("default", creds)
    save_credentials_file(cf, path)
    return get_shared_state(path)


@pytest.mark.asyncio
async def test_amain_happy_path_runs_serve_until_eof(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """_amain wires a real httpx client + serve loop; immediate stdin EOF → clean exit 0."""
    import io

    state = _build_state(tmp_path)
    monkeypatch.setattr(mcp_proxy, "get_shared_state", lambda profile=None: state)
    # Empty stdin → readline() returns "" → serve() exits immediately, no upstream call.
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    rc = await mcp_proxy._amain([])
    assert rc == 0


def test_main_returns_amain_exit_code(monkeypatch: pytest.MonkeyPatch):
    """main() drives asyncio.run(_amain(...)) and returns its exit code."""
    monkeypatch.setattr(mcp_proxy, "get_shared_state", lambda profile=None: None)
    assert mcp_proxy.main(["--profile", "x"]) == 1


@pytest.mark.asyncio
async def test_amain_forwards_one_request_and_writes_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    """End-to-end: _amain reads one stdin line, forwards via the real client, writes the reply."""
    import io

    state = _build_state(tmp_path)
    monkeypatch.setattr(mcp_proxy, "get_shared_state", lambda profile=None: state)
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO('{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}\n'),
    )
    out = io.StringIO()
    monkeypatch.setattr("sys.stdout", out)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"jsonrpc": "2.0", "id": 1, "result": {"tools": []}},
            headers={"mcp-session-id": "s1"},
        )

    real_async_client = httpx.AsyncClient

    def fake_async_client(**kwargs: Any) -> httpx.AsyncClient:
        # Inject a MockTransport so no real network call is made; the real
        # KaguraOAuth auth flow still runs (token is fresh, so it no-ops).
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_async_client(**kwargs)

    monkeypatch.setattr(mcp_proxy.httpx, "AsyncClient", fake_async_client)

    rc = await mcp_proxy._amain([])
    assert rc == 0
    assert json.loads(out.getvalue().strip())["result"] == {"tools": []}


@pytest.mark.asyncio
async def test_amain_recovers_from_expired_upstream_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    """End-to-end (#252): the upstream drops the session mid-run and the proxy heals it.

    Claude Code only speaks stdio, so it cannot re-initialize on its own. The
    second ``tools/call`` hits the 404 session-expired reply; the proxy must
    replay the cached ``initialize``, retry the call, and answer it — without a
    restart and without an error on stdout.
    """
    import io

    state = _build_state(tmp_path)
    monkeypatch.setattr(mcp_proxy, "get_shared_state", lambda profile=None: state)
    stdin_lines = [_INITIALIZE, _INITIALIZED, _tool_call(1), _tool_call(2), _tool_call(3)]
    stdin = io.StringIO("".join(json.dumps(m) + "\n" for m in stdin_lines))
    monkeypatch.setattr("sys.stdin", stdin)
    out = io.StringIO()
    monkeypatch.setattr("sys.stdout", out)

    server = _FakeMcpServer()

    def handler(request: httpx.Request) -> httpx.Response:
        response = server.handler(request)
        body = json.loads(request.content)
        if body["method"] == "tools/call" and body.get("id") == 1:
            server.restart()  # the server restarts right after answering call 1
        return response

    real_async_client = httpx.AsyncClient

    def fake_async_client(**kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_async_client(**kwargs)

    monkeypatch.setattr(mcp_proxy.httpx, "AsyncClient", fake_async_client)

    rc = await mcp_proxy._amain([])

    assert rc == 0
    replies = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [r["id"] for r in replies] == [0, 1, 2, 3]
    assert all("error" not in r for r in replies)
    assert [r["result"] for r in replies[1:]] == [
        {"session": "sess-1"},
        {"session": "sess-2"},  # recovered transparently
        {"session": "sess-2"},
    ]
    assert server.methods() == [
        "initialize",
        "notifications/initialized",
        "tools/call",
        "tools/call",  # 404: session expired
        "initialize",  # replayed once
        "notifications/initialized",
        "tools/call",  # retried
        "tools/call",
    ]
    assert server.requests[4][0] == _INITIALIZE
