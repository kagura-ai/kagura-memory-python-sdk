"""Tests for kagura_memory.mcp_proxy (the kagura-mcp stdio MCP proxy)."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from kagura_memory import mcp_proxy
from kagura_memory.mcp_proxy import _error_response, _Upstream, serve
from tests.conftest import SESSION_EXPIRED_BODY, FakeMcpServer


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


def _served_in(session_id: str, msg_id: int) -> dict[str, Any]:
    """The reply the fake server gives a ``tools/call`` answered in ``session_id``."""
    return {"jsonrpc": "2.0", "id": msg_id, "result": FakeMcpServer.tool_result(session_id)}


async def _open_session(up: _Upstream) -> None:
    """Run the downstream handshake Claude Code performs at startup."""
    assert (await up.forward(_INITIALIZE)) is not None
    assert (await up.forward(_INITIALIZED)) is None


@pytest.mark.asyncio
async def test_forward_expired_session_replays_initialize_and_retries_once():
    server = FakeMcpServer()
    up, _ = _upstream(server.handler)
    await _open_session(up)
    server.restart()  # idle-hour expiry or a deploy drops every session

    result = await up.forward(_tool_call(7))

    assert result == _served_in("sess-2", 7)
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
    server = FakeMcpServer()
    up, _ = _upstream(server.handler)
    await _open_session(up)
    server.restart()
    await up.forward(_tool_call(1))

    result = await up.forward(_tool_call(2))

    assert result == _served_in("sess-2", 2)
    assert server.methods().count("initialize") == 2  # the handshake + one replay


@pytest.mark.asyncio
async def test_forward_expired_session_without_cached_initialize_retries_without_session():
    """No ``initialize`` seen yet: forget the stale id and retry once without one.

    The session came from a plain request, so there is nothing to replay; the
    upstream opens a session for the session-less retry and the proxy keeps it.
    """
    server = FakeMcpServer()
    up, _ = _upstream(server.handler)
    await up.forward({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
    server.restart()

    result = await up.forward(_tool_call(2))

    assert result == _served_in("sess-2", 2)
    assert server.calls() == [
        ("tools/list", None),
        ("tools/call", "sess-1"),  # 404: session gone
        ("tools/call", None),  # retried once, without the stale id; no initialize
    ]
    assert up._session_id == "sess-2"  # the session the retry opened is kept


@pytest.mark.asyncio
async def test_forward_persistent_404_forwards_server_error_after_single_replay():
    """A retry that 404s too reaches the downstream as the server's own error, once."""
    server = FakeMcpServer()

    def handler(request: httpx.Request) -> httpx.Response:
        response = server.handler(request)
        if json.loads(request.content)["method"] == "notifications/initialized":
            server.restart()  # every replayed session dies before the retry lands
        return response

    up, _ = _upstream(handler)
    await _open_session(up)

    result = await up.forward(_tool_call(9))

    assert result == {**SESSION_EXPIRED_BODY, "id": 9}  # downstream id put back
    assert server.methods().count("initialize") == 2  # replayed once, no loop
    assert server.methods().count("tools/call") == 2


@pytest.mark.asyncio
async def test_forward_replayed_initialize_failure_raises():
    """If the replayed initialize fails, the message is not re-sent and the error surfaces."""
    server = FakeMcpServer()
    up, _ = _upstream(server.handler)
    await _open_session(up)
    server.restart()

    def down(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content)["method"] == "initialize":
            server.record(request)
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
        return httpx.Response(404, json=SESSION_EXPIRED_BODY)

    up, _ = _upstream(handler)
    result = await up.forward(_tool_call(4))
    assert result == {**SESSION_EXPIRED_BODY, "id": 4}
    assert calls == ["tools/call"]


@pytest.mark.asyncio
async def test_forward_modern_method_not_found_404_is_not_a_session_error():
    """memory-cloud's stateless 404 + -32601 (#1544) ignores the session: no replay."""
    server = FakeMcpServer()
    not_found = {"jsonrpc": "2.0", "id": 5, "error": {"code": -32601, "message": "nope"}}

    def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content)["method"] == "subscriptions/listen":
            server.record(request)
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
    server = FakeMcpServer()
    up, _ = _upstream(server.handler)
    await _open_session(up)
    server.restart()

    assert (await up.forward({**_INITIALIZE, "id": 10})) is not None
    assert server.requests[-1][1] is None
    assert (await up.forward(_tool_call(11))) == _served_in("sess-2", 11)
    assert server.methods().count("initialize") == 2


@pytest.mark.asyncio
async def test_forward_failed_initialize_is_not_cached_for_replay():
    """Only an initialize the upstream accepted is replayed after an expiry."""
    server = FakeMcpServer()
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


def _modern_request(msg_id: int, method: str, version: str) -> dict[str, Any]:
    """A stateless (2026-07-28) request: its protocol version rides in ``params._meta``."""
    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "method": method,
        "params": {"_meta": {"io.modelcontextprotocol/protocolVersion": version}},
    }


# memory-cloud answers -32022 only on the stateless path, i.e. to a request whose
# ``params._meta`` names a version it does not serve there, listing every
# revision it does (``SUPPORTED_PROTOCOL_VERSIONS``, transport_stateless.py).
_UNSUPPORTED_VERSION_REQUEST = _modern_request(8, "tools/list", "2025-06-18")
_UNSUPPORTED_VERSION_BODY: dict[str, Any] = {
    "jsonrpc": "2.0",
    "id": 8,
    "error": {
        "code": -32022,
        "message": "Unsupported protocol version",
        "data": {
            "supported": ["2026-07-28", "2025-03-26", "2024-11-05"],
            "requested": "2025-06-18",
        },
    },
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "status", "body"),
    [
        pytest.param(
            {"jsonrpc": "2.0", "id": 8},
            400,
            {
                "jsonrpc": "2.0",
                "error": {"code": -32600, "message": "Invalid Request: missing method"},
                "id": 8,
            },
            id="invalid-request",
        ),
        pytest.param(
            _UNSUPPORTED_VERSION_REQUEST,
            400,
            _UNSUPPORTED_VERSION_BODY,
            id="unsupported-protocol-version",
        ),
        pytest.param(
            _modern_request(8, "subscriptions/listen", "2026-07-28"),
            404,
            {
                "jsonrpc": "2.0",
                "id": 8,
                "error": {"code": -32601, "message": "Method not found: subscriptions/listen"},
            },
            id="modern-method-not-found-404",
        ),
    ],
)
async def test_forward_jsonrpc_error_body_passes_through_unchanged(
    message: dict[str, Any], status: int, body: dict[str, Any]
):
    up, _ = _upstream(lambda request: httpx.Response(status, json=body))
    assert await up.forward(message) == body


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


def _status_error(response: httpx.Response) -> httpx.HTTPStatusError:
    response.request = httpx.Request("POST", "https://test/mcp")
    return httpx.HTTPStatusError("boom", request=response.request, response=response)


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
    exc = _status_error(httpx.Response(401))
    resp = _error_response({"jsonrpc": "2.0", "id": 1}, exc)
    assert resp is not None
    assert "kagura auth login" in resp["error"]["message"]


def test_error_response_carries_the_servers_explanation():
    """The MCP transport's OAuth-style 403 (workspace URL) keeps its ``error_description``."""
    body = {
        "error": "access_denied",
        "error_description": "You are not a member of this workspace.",
    }
    resp = _error_response(
        {"jsonrpc": "2.0", "id": 1}, _status_error(httpx.Response(403, json=body))
    )
    assert resp is not None
    assert resp["error"] == {
        "code": -32000,
        "message": "kagura-mcp: HTTP 403: You are not a member of this workspace.",
    }


def test_error_response_without_server_detail_keeps_the_httpx_message():
    exc = _status_error(httpx.Response(502, text="Bad Gateway"))
    resp = _error_response({"jsonrpc": "2.0", "id": 1}, exc)
    assert resp is not None
    assert resp["error"]["message"] == "kagura-mcp: boom"


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


async def _run_amain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    stdin: list[dict[str, Any]],
    handler: Any,
    argv: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Run ``_amain`` end to end over ``stdin``; return the replies it wrote to stdout.

    ``handler`` answers the upstream through a MockTransport injected into
    ``_amain``'s real ``httpx.AsyncClient``, so no network call is made while
    the real KaguraOAuth auth flow still runs (the token is fresh, so it no-ops).
    """
    import io

    state = _build_state(tmp_path)
    monkeypatch.setattr(mcp_proxy, "get_shared_state", lambda profile=None: state)
    monkeypatch.setattr("sys.stdin", io.StringIO("".join(json.dumps(m) + "\n" for m in stdin)))
    out = io.StringIO()
    monkeypatch.setattr("sys.stdout", out)
    real_async_client = httpx.AsyncClient

    def fake_async_client(**kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_async_client(**kwargs)

    monkeypatch.setattr(mcp_proxy.httpx, "AsyncClient", fake_async_client)

    assert await mcp_proxy._amain(argv or []) == 0
    return [json.loads(line) for line in out.getvalue().splitlines()]


@pytest.mark.asyncio
async def test_amain_forwards_one_request_and_writes_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    """End-to-end: _amain reads one stdin line, forwards via the real client, writes the reply."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"jsonrpc": "2.0", "id": 1, "result": {"tools": []}},
            headers={"mcp-session-id": "s1"},
        )

    request = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    replies = await _run_amain(monkeypatch, tmp_path, [request], handler)
    assert [reply["result"] for reply in replies] == [{"tools": []}]


@pytest.mark.asyncio
async def test_amain_writes_upstream_jsonrpc_error_to_stdout_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    """End-to-end (#252): a 4xx JSON-RPC error reaches stdout with its code/message/data.

    ``-32022`` carries ``data.supported``, which a modern client needs to pick
    a version to retry with; flattening it into ``-32000`` would lose that.
    """
    replies = await _run_amain(
        monkeypatch,
        tmp_path,
        [_UNSUPPORTED_VERSION_REQUEST],
        lambda request: httpx.Response(400, json=_UNSUPPORTED_VERSION_BODY),
    )
    assert replies == [_UNSUPPORTED_VERSION_BODY]


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
    server = FakeMcpServer()

    def handler(request: httpx.Request) -> httpx.Response:
        response = server.handler(request)
        body = json.loads(request.content)
        if body["method"] == "tools/call" and body.get("id") == 1:
            server.restart()  # the server restarts right after answering call 1
        return response

    stdin = [_INITIALIZE, _INITIALIZED, _tool_call(1), _tool_call(2), _tool_call(3)]
    replies = await _run_amain(monkeypatch, tmp_path, stdin, handler)

    assert [r["id"] for r in replies] == [0, 1, 2, 3]
    assert all("error" not in r for r in replies)
    assert replies[1:] == [
        _served_in("sess-1", 1),
        _served_in("sess-2", 2),  # recovered transparently
        _served_in("sess-2", 3),
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


# ---------------------------------------------------------------------------
# _amain — --guardrails / --tool-profile build the upstream URL (#258)
# ---------------------------------------------------------------------------

_CTX_UUID = "11111111-2222-3333-4444-555555555555"


async def _posted_url(monkeypatch: pytest.MonkeyPatch, tmp_path, argv: list[str]) -> str:
    """Run ``_amain`` with ``argv`` for one ``tools/list``; return the URL it POSTed to."""
    urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {"tools": []}})

    request = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    await _run_amain(monkeypatch, tmp_path, [request], handler, argv)
    assert len(urls) == 1
    return urls[0]


@pytest.mark.asyncio
async def test_amain_without_query_flags_posts_to_the_profile_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    assert await _posted_url(monkeypatch, tmp_path, []) == "https://test.example.com/mcp"


@pytest.mark.asyncio
async def test_amain_guardrails_and_tool_profile_extend_the_profile_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    url = await _posted_url(
        monkeypatch, tmp_path, ["--guardrails", "off", "--tool-profile", "core"]
    )
    assert url == "https://test.example.com/mcp?guardrails=off&profile=core"


@pytest.mark.asyncio
async def test_amain_query_flags_keep_the_server_query_and_replace_guardrails(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    """--server keeps its own query; an earlier ``guardrails`` value is replaced, not
    appended (memory-cloud reads only the first one)."""
    server = f"https://test.example.com/mcp/w/ws-1?guardrails={_CTX_UUID}&tools=recall"
    url = await _posted_url(monkeypatch, tmp_path, ["--server", server, "--guardrails", "off"])
    assert url == "https://test.example.com/mcp/w/ws-1?tools=recall&guardrails=off"


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["?tools=recall", ""], ids=["tools-allowlist", "no-query"])
async def test_amain_warns_when_a_tools_allowlist_overrides_the_tool_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys, query: str
):
    """memory-cloud applies ``tools`` instead of ``profile``: --tool-profile would do nothing."""
    server = f"https://test.example.com/mcp{query}"
    await _posted_url(monkeypatch, tmp_path, ["--server", server, "--tool-profile", "core"])
    warned = "?tools= allowlist" in capsys.readouterr().err
    assert warned == bool(query)


@pytest.mark.asyncio
async def test_amain_guardrails_uuid_is_canonicalized(monkeypatch: pytest.MonkeyPatch, tmp_path):
    url = await _posted_url(monkeypatch, tmp_path, ["--guardrails", _CTX_UUID.upper()])
    assert url == f"https://test.example.com/mcp?guardrails={_CTX_UUID}"


@pytest.mark.parametrize("bad", ["on", "my-context", ""])
def test_parser_rejects_guardrails_that_is_neither_off_nor_uuid(bad: str, capsys):
    """The server silently ignores such a value, so the proxy refuses to start."""
    with pytest.raises(SystemExit) as exc:
        mcp_proxy._build_parser().parse_args(["--guardrails", bad])
    assert exc.value.code == 2
    assert "'off' or a context UUID" in capsys.readouterr().err


def test_parser_rejects_an_empty_tool_profile(capsys):
    with pytest.raises(SystemExit) as exc:
        mcp_proxy._build_parser().parse_args(["--tool-profile", " "])
    assert exc.value.code == 2
    assert "--tool-profile" in capsys.readouterr().err


def test_parser_help_warns_that_off_removes_the_context_info_block(capsys):
    with pytest.raises(SystemExit):
        mcp_proxy._build_parser().parse_args(["--help"])
    assert "get_context_info" in " ".join(capsys.readouterr().out.split())
