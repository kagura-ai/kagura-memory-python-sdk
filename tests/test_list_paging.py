"""Page sizes on ``list_contexts`` / ``get_state`` and ``kagura context list`` paging (#315).

memory-cloud v0.82.0 (#1743) pages both lists and declares ``limit`` on
``list_contexts`` (1-200; 20 with ``include_details`` and no ``name_contains``)
and ``limit`` / ``max_chars`` on ``get_state``. ``list_contexts`` declares no
``max_chars`` up to v0.142.0, so the SDK sends none: from v0.82.0 an undeclared
argument is refused (``invalid_argument``, #1742). The replies are shaped like
v0.142.0's ``mcp_server/tools/context.py::handle_list_contexts`` and
``mcp_server/tools/state.py::handle_get_state``, and every request goes through
the real JSON-RPC transport.
"""

from __future__ import annotations

import inspect
import json
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from click.testing import CliRunner

from kagura_memory import KaguraClient, KaguraError
from kagura_memory.cli import _next_page_footer, main

CTX = "11111111-2222-3333-4444-555555555555"
_CONFIG = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

# A server from 0.82.0 answering an argument its schema does not declare
# (memory-cloud ``find_unknown_arguments``), here a page size.
_REFUSED_LIMIT = {
    "status": "error",
    "error": "invalid_argument",
    "message": (
        "list_contexts does not accept the argument 'limit'. Arguments its inputSchema "
        "declares: cursor, include_details, include_stats, include_summary, name_contains."
    ),
    "unknown_arguments": ["limit"],
    "allowed_arguments": [
        "cursor",
        "include_details",
        "include_stats",
        "include_summary",
        "name_contains",
    ],
    "suggestions": {},
    "help": (
        "Remove the unknown arguments or rename them as suggested, then call the tool "
        "again. tools/list shows each tool's inputSchema."
    ),
}


def _context(n: int) -> dict[str, Any]:
    return {
        "id": f"00000000-0000-0000-0000-{n:012d}",
        "name": f"ctx-{n}",
        "is_private": False,
        "is_locked": False,
        "last_used_at": "2026-11-01T00:00:00Z",
    }


class _Server:
    """A ``/mcp`` endpoint that records each ``tools/call`` and answers via ``reply``."""

    def __init__(self, reply: Any) -> None:
        self.reply = reply
        self.arguments: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        arguments = body["params"]["arguments"]
        self.arguments.append(arguments)
        payload = self.reply(arguments) if callable(self.reply) else self.reply
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": body["id"],
                "result": {"content": [{"type": "text", "text": text}]},
            },
        )

    def client(self, **_: Any) -> KaguraClient:
        client = KaguraClient(**_CONFIG)
        client._session_id = "pre-set-session"
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(self.handler))
        return client


def _directory(contexts: list[dict[str, Any]]):
    """``list_contexts`` paging as v0.142.0 does it: an offset cursor, default 100."""

    def reply(arguments: dict[str, Any]) -> dict[str, Any]:
        offset = int(arguments.get("cursor") or 0)
        limit = arguments.get("limit", 100)
        page = contexts[offset : offset + limit]
        has_more = offset + len(page) < len(contexts)
        return {
            "status": "success",
            "contexts": page,
            "count": len(contexts),
            "total": len(page),
            "has_more": has_more,
            "next_cursor": str(offset + len(page)) if has_more else None,
            "limit": 500,
            "can_create": True,
        }

    return reply


def _keyword_only(method: Any, name: str) -> bool:
    return inspect.signature(method).parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


# ===========================================================================
# KaguraClient.list_contexts(limit=)
# ===========================================================================


@pytest.mark.asyncio
async def test_list_contexts_sends_no_paging_argument_by_default():
    """Unset, the server's page size applies and an older server sees nothing new."""
    server = _Server(_directory([_context(1)]))
    client = server.client()
    try:
        await client.list_contexts()
        await client.list_contexts(include_details=True)
    finally:
        await client.close()
    assert server.arguments == [{}, {"include_details": True}]


@pytest.mark.parametrize("limit", [1, 20, 200])
@pytest.mark.asyncio
async def test_list_contexts_sends_limit_and_cursor_when_set(limit: int):
    server = _Server(_directory([_context(n) for n in range(400)]))
    client = server.client()
    try:
        result = await client.list_contexts(cursor="100", limit=limit)
    finally:
        await client.close()
    assert server.arguments == [{"cursor": "100", "limit": limit}]
    assert result["total"] == limit
    assert result["next_cursor"] == str(100 + limit)
    assert _keyword_only(KaguraClient.list_contexts, "limit")


@pytest.mark.parametrize(
    ("kwargs", "sent"),
    [
        ({"include_details": True, "limit": 20}, {"include_details": True, "limit": 20}),
        # name_contains lifts the 20-row cap on include_details pages.
        (
            {"include_details": True, "name_contains": "auth", "limit": 200},
            {"include_details": True, "name_contains": "auth", "limit": 200},
        ),
        ({"include_summary": True, "limit": 200}, {"include_summary": True, "limit": 200}),
    ],
    ids=["details-20", "details-filtered-200", "summary-200"],
)
@pytest.mark.asyncio
async def test_list_contexts_include_details_pages(kwargs: dict, sent: dict):
    server = _Server(_directory([_context(1)]))
    client = server.client()
    try:
        await client.list_contexts(**kwargs)
    finally:
        await client.close()
    assert server.arguments == [sent]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"limit": 0}, "between 1 and 200"),
        ({"limit": 201}, "between 1 and 200"),
        ({"limit": True}, "between 1 and 200"),  # bool is not an int, as on the server
        ({"limit": 50.0}, "between 1 and 200"),
        ({"limit": "50"}, "between 1 and 200"),
        # The server refuses this page with validation_error.
        ({"include_details": True, "limit": 21}, "at most 20"),
        # A blank name_contains is trimmed to no filter on the server.
        ({"include_details": True, "name_contains": "   ", "limit": 21}, "at most 20"),
        # So is a non-string one (the cap is checked before its type there).
        ({"include_details": True, "name_contains": 123, "limit": 21}, "at most 20"),
    ],
)
@pytest.mark.asyncio
async def test_list_contexts_rejects_a_limit_before_any_request(kwargs: dict, match: str):
    server = _Server(_directory([]))
    client = server.client()
    try:
        with pytest.raises(ValueError, match=match):
            await client.list_contexts(**kwargs)
    finally:
        await client.close()
    assert server.arguments == []


def test_list_contexts_takes_no_max_chars():
    """v0.82.0-v0.142.0 declare no ``max_chars`` on list_contexts, and refuse an undeclared one."""
    assert "max_chars" not in inspect.signature(KaguraClient.list_contexts).parameters


@pytest.mark.asyncio
async def test_a_refused_limit_raises_invalid_argument_without_a_retry():
    server = _Server(_REFUSED_LIMIT)
    client = server.client()
    try:
        with pytest.raises(KaguraError) as excinfo:
            await client.list_contexts(limit=50)
    finally:
        await client.close()
    assert excinfo.value.code == "invalid_argument"
    assert excinfo.value.details["unknown_arguments"] == ["limit"]
    assert server.arguments == [{"limit": 50}]  # one call: the SDK never retries


# ===========================================================================
# KaguraClient.get_state(limit=, max_chars=)
# ===========================================================================

_STATE_PAGE = {
    "status": "success",
    "context_id": CTX,
    "states": {"a": 1, "b": {"step": 2}},
    "count": 2,
    "has_more": True,
    "next_cursor": "b",
}


@pytest.mark.asyncio
async def test_get_state_sends_limit_and_max_chars_only_when_set():
    server = _Server(_STATE_PAGE)
    client = server.client()
    try:
        await client.get_state(CTX)
        result = await client.get_state(CTX, cursor="a", limit=2, max_chars=10_000)
        await client.get_state(CTX, limit=200)
        await client.get_state(CTX, max_chars=100_000)
    finally:
        await client.close()
    assert server.arguments == [
        {"context_id": CTX},
        {"context_id": CTX, "cursor": "a", "limit": 2, "max_chars": 10_000},
        {"context_id": CTX, "limit": 200},
        {"context_id": CTX, "max_chars": 100_000},
    ]
    assert (result["has_more"], result["next_cursor"]) == (True, "b")
    assert _keyword_only(KaguraClient.get_state, "limit")
    assert _keyword_only(KaguraClient.get_state, "max_chars")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"limit": 0}, "limit must be an integer between 1 and 200"),
        ({"limit": 201}, "limit must be an integer between 1 and 200"),
        ({"limit": True}, "limit must be an integer between 1 and 200"),
        ({"max_chars": 9_999}, "max_chars must be an integer between 10000 and 100000"),
        ({"max_chars": 100_001}, "max_chars must be an integer between 10000 and 100000"),
        ({"max_chars": True}, "max_chars must be an integer between 10000 and 100000"),
        ({"max_chars": "20000"}, "max_chars must be an integer between 10000 and 100000"),
    ],
)
@pytest.mark.asyncio
async def test_get_state_rejects_out_of_range_paging_before_any_request(kwargs, match):
    server = _Server(_STATE_PAGE)
    client = server.client()
    try:
        with pytest.raises(ValueError, match=match):
            await client.get_state(CTX, **kwargs)
    finally:
        await client.close()
    assert server.arguments == []


@pytest.mark.parametrize(
    ("kwargs", "named"),
    [
        ({"limit": 10}, "limit: list mode only"),
        ({"max_chars": 20_000}, "max_chars: list mode only"),
        ({"cursor": "a", "limit": 10}, "cursor, limit: list mode only"),
    ],
)
@pytest.mark.asyncio
async def test_get_state_refuses_list_arguments_with_a_key(kwargs: dict, named: str):
    """The server reads them in list mode only; with a key they would be dropped."""
    server = _Server(_STATE_PAGE)
    client = server.client()
    try:
        with pytest.raises(ValueError, match=named):
            await client.get_state(CTX, "step", **kwargs)
    finally:
        await client.close()
    assert server.arguments == []


# ===========================================================================
# kagura context list --limit/--cursor
# ===========================================================================


def _invoke(server: _Server, *args: str):
    with (
        patch("kagura_memory.cli.load_config", return_value=dict(_CONFIG)),
        patch("kagura_memory.cli.KaguraClient", side_effect=server.client),
    ):
        return CliRunner().invoke(main, ["context", "list", *args])


def test_context_list_pages_with_the_footer_cursor():
    """Page 1 names next_cursor on stderr; passing it back returns page 2 and no footer."""
    contexts = [_context(n) for n in range(3)]
    server = _Server(_directory(contexts))

    first = _invoke(server, "--limit", "2")
    assert first.exit_code == 0, first.output
    page1 = json.loads(first.stdout)  # stdout stays the reply's JSON
    assert [c["name"] for c in page1["contexts"]] == ["ctx-0", "ctx-1"]
    assert (page1["has_more"], page1["next_cursor"]) == (True, "2")
    assert first.stderr == (
        'has_more: true; next_cursor: "2". For the next page, run the command again '
        'with the same options and --cursor "2".\n'
    )

    second = _invoke(server, "--limit", "2", "--cursor", page1["next_cursor"])
    assert second.exit_code == 0, second.output
    page2 = json.loads(second.stdout)
    assert [c["name"] for c in page2["contexts"]] == ["ctx-2"]
    assert (page2["has_more"], page2["next_cursor"]) == (False, None)
    assert second.stderr == ""

    assert server.arguments == [{"limit": 2}, {"cursor": "2", "limit": 2}]


def test_context_list_forwards_the_paging_flags_with_the_filters():
    server = _Server(_directory([_context(1)]))
    result = _invoke(
        server, "--name-contains", "ctx", "--details", "--limit", "50", "--cursor", "100"
    )
    assert result.exit_code == 0, result.output
    assert server.arguments == [
        {"name_contains": "ctx", "include_details": True, "cursor": "100", "limit": 50}
    ]


def test_context_list_from_an_older_server_prints_no_footer():
    """Before 0.82.0 the reply has no has_more / next_cursor and nothing is paged."""
    server = _Server({"status": "success", "contexts": [_context(1)], "count": 1})
    result = _invoke(server)
    assert result.exit_code == 0, result.output
    assert server.arguments == [{}]
    assert result.stderr == ""


def test_context_list_refused_limit_is_a_clean_error():
    server = _Server(_REFUSED_LIMIT)
    result = _invoke(server, "--limit", "50")
    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr.startswith(
        "Error: list_contexts failed (invalid_argument): list_contexts does not accept "
        "the argument 'limit'."
    )
    assert "  Help: Remove the unknown arguments" in result.stderr
    assert "Traceback" not in result.output
    assert server.arguments == [{"limit": 50}]


def test_context_list_refuses_a_details_page_over_20_before_any_request():
    server = _Server(_directory([]))
    result = _invoke(server, "--details", "--limit", "21")
    assert result.exit_code == 1
    assert "pass name_contains or a limit of at most 20, got 21" in result.stderr
    assert server.arguments == []


@pytest.mark.parametrize("limit", ["0", "201"])
def test_context_list_limit_is_checked_by_the_option(limit: str):
    server = _Server(_directory([]))
    result = _invoke(server, "--limit", limit)
    assert result.exit_code == 2
    assert "1<=x<=200" in result.stderr
    assert server.arguments == []


@pytest.mark.parametrize(
    "result",
    [
        {"has_more": False, "next_cursor": None},
        {"has_more": True, "next_cursor": None},  # nothing to continue from
        {"has_more": True, "next_cursor": ""},
        {"has_more": "true", "next_cursor": "2"},  # not the JSON boolean
        {"contexts": []},
    ],
)
def test_next_page_footer_needs_has_more_and_a_cursor(result: dict):
    assert _next_page_footer(result) is None


@pytest.mark.parametrize(
    "cursor",
    [
        "2\x1b[31m",  # a terminal escape
        "$(touch pwned)",  # expanded by a shell inside double quotes
        "`id`",
        "2\n3",
        "\u0662",  # an Arabic-Indic digit: isdigit(), but not ASCII
    ],
)
def test_next_page_footer_shows_only_a_cursor_the_server_takes_back(cursor: str):
    """memory-cloud takes back ASCII digits only; anything else stays in the JSON."""
    note = _next_page_footer({"has_more": True, "next_cursor": cursor})
    assert note == (
        "has_more: true. For the next page, run the command again with the same "
        "options and --cursor set to next_cursor from the JSON."
    )
