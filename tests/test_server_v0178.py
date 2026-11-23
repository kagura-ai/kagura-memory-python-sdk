"""memory-cloud v0.177.0 / v0.178.0: fusion, list_workspaces, list_contexts(workspace_id) (#350).

**Fusion** (memory-cloud v0.178.0, #2980). ``update_search_config(fusion=)`` is
keyword-only and sent only when set, as given; the reply's ``config`` and
``get_context_info``'s ``search_config`` carry ``fusion`` (stored) and
``fusion_effective`` (what recall resolves to), which :class:`SearchConfig`
reads as plain ``str`` (``None`` from an older server).

**Workspaces** (memory-cloud v0.177.0, #2972). ``list_workspaces()`` calls the
MCP tool with no arguments and reads ``{status, workspaces, total}`` into
:class:`AccountWorkspaceList`; ``list_contexts(workspace_id=)`` is sent only
when set. The replies below are shaped like v0.178.0's
``mcp_server/tools/workspaces.py``, ``context.py`` and ``search_config.py``.
"""

from __future__ import annotations

import inspect
import json
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from click.testing import CliRunner

import kagura_memory
from kagura_memory import (
    AccountWorkspace,
    AccountWorkspaceList,
    ContextInfo,
    KaguraClient,
    KaguraError,
    KaguraFeatureNotAvailableError,
    SearchConfig,
)
from kagura_memory.cli import main

CTX = "11111111-2222-3333-4444-555555555555"
WS = "aaaaaaaa-0000-0000-0000-000000000001"
WS2 = "aaaaaaaa-0000-0000-0000-000000000002"


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

    client = KaguraClient(api_key="test", mcp_url="https://test.com/mcp")
    client._session_id = "pre-set-session"
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


def _call(sent: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    """The tool name and arguments of the one ``tools/call`` sent."""
    assert len(sent) == 1
    assert sent[0]["method"] == "tools/call"
    return sent[0]["params"]["name"], sent[0]["params"]["arguments"]


def _keyword_only(method: Any, name: str) -> bool:
    return inspect.signature(method).parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


async def _run(reply: dict[str, Any], call: Any) -> tuple[Any, str, dict[str, Any]]:
    sent: list[dict[str, Any]] = []
    client = _wire_client(reply, sent)
    try:
        result = await call(client)
    finally:
        await client.close()
    name, arguments = _call(sent)
    return result, name, arguments


# ===========================================================================
# update_search_config(fusion=) and SearchConfig.fusion / fusion_effective
# ===========================================================================

_SEARCH_CONFIG_REPLY = {
    "status": "success",
    "message": "Search configuration updated.",
    "context_id": CTX,
    "config": {
        "semantic_weight": 0.6,
        "bm25_weight": 0.4,
        "fetch_factor": 3,
        "use_rerank": False,
        "reranker_provider": "voyage",
        "reranker_model": "rerank-2",
        "reinforce_enabled": True,
        "reinforce_max_boost": 0.15,
        "reinforce_require_host_arbitration": False,
        "routing_mode": "off",
        "fusion": "rrf",
        "fusion_effective": "rrf",
        "content_chunks_enabled": False,
        "content_chunks_effective": False,
        "llm_selector_enabled": False,
        "llm_selector_effective": False,
    },
}


def test_update_search_config_fusion_is_keyword_only() -> None:
    assert _keyword_only(KaguraClient.update_search_config, "fusion")
    assert inspect.signature(KaguraClient.update_search_config).parameters["fusion"].default is None


@pytest.mark.asyncio
async def test_update_search_config_omits_fusion_by_default() -> None:
    """The request an older server gets is unchanged when ``fusion`` is not set."""
    _, name, arguments = await _run(
        _SEARCH_CONFIG_REPLY, lambda c: c.update_search_config(CTX, use_rerank=True)
    )
    assert name == "update_search_config"
    assert arguments == {"context_id": CTX, "use_rerank": True}


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["auto", "rrf", "weighted"])
async def test_update_search_config_sends_fusion(value: str) -> None:
    result, _, arguments = await _run(
        _SEARCH_CONFIG_REPLY, lambda c: c.update_search_config(CTX, fusion=value)
    )
    assert arguments == {"context_id": CTX, "fusion": value}
    assert result["config"]["fusion"] == "rrf"


@pytest.mark.asyncio
async def test_update_search_config_sends_an_unknown_fusion_as_given() -> None:
    """No local value check: the server owns the rule and refuses with its own code."""
    refusal = {
        "status": "error",
        "error": "invalid_search_config",
        "message": "fusion: Input should be 'auto', 'rrf' or 'weighted'",
    }
    sent: list[dict[str, Any]] = []
    client = _wire_client(refusal, sent)
    try:
        with pytest.raises(KaguraError) as excinfo:
            await client.update_search_config(CTX, fusion="keyword_floor")
    finally:
        await client.close()
    assert _call(sent)[1] == {"context_id": CTX, "fusion": "keyword_floor"}
    assert excinfo.value.code == "invalid_search_config"


@pytest.mark.asyncio
async def test_update_search_config_fusion_on_an_older_server_is_an_unknown_argument() -> None:
    """Before v0.178.0 the tool's schema has no ``fusion``: ``invalid_argument`` (#1742)."""
    refusal = {
        "status": "error",
        "error": "invalid_argument",
        "message": "Unknown argument for update_search_config: 'fusion'.",
        "unknown_arguments": ["fusion"],
        "accepted_arguments": ["context_id", "semantic_weight", "bm25_weight"],
    }
    sent: list[dict[str, Any]] = []
    client = _wire_client(refusal, sent)
    try:
        with pytest.raises(KaguraError) as excinfo:
            await client.update_search_config(CTX, fusion="rrf")
    finally:
        await client.close()
    assert _call(sent)[1] == {"context_id": CTX, "fusion": "rrf"}
    assert excinfo.value.code == "invalid_argument"
    assert excinfo.value.details["unknown_arguments"] == ["fusion"]


@pytest.mark.parametrize(
    ("sent", "fusion", "effective"),
    [
        ({"fusion": "auto", "fusion_effective": "weighted"}, "auto", "weighted"),
        ({"fusion": "rrf", "fusion_effective": "rrf"}, "rrf", "rrf"),
        # auto resolves to the deployment's HYBRID_FUSION, which has a third value.
        ({"fusion": "auto", "fusion_effective": "keyword_floor"}, "auto", "keyword_floor"),
        ({"fusion": "newer", "fusion_effective": "newer"}, "newer", "newer"),
        ({}, None, None),
    ],
    ids=["auto", "rrf", "deployment-value", "newer-value", "older-server"],
)
def test_search_config_reads_fusion(sent: dict, fusion: str | None, effective: str | None) -> None:
    config = SearchConfig.model_validate({"semantic_weight": 0.6, "bm25_weight": 0.4, **sent})
    assert config.fusion == fusion
    assert config.fusion_effective == effective


@pytest.mark.asyncio
async def test_get_context_info_reads_fusion_from_search_config() -> None:
    reply = {
        "status": "success",
        "context": {
            "id": CTX,
            "name": "dev",
            "search_config": {
                "semantic_weight": 0.8,
                "bm25_weight": 0.2,
                "fetch_factor": 3,
                "use_rerank": False,
                "reranker_provider": "voyage",
                "reranker_model": "rerank-2",
                "fusion": "auto",
                "fusion_effective": "weighted",
                "llm_selector_enabled": False,
                "llm_selector_effective": False,
            },
        },
        "workspace": {"id": WS, "name": "Main"},
        "stats": {"total_memories": 0},
        "instructions": "",
    }
    info, name, _ = await _run(reply, lambda c: c.get_context_info(CTX))
    assert name == "get_context_info"
    assert isinstance(info, ContextInfo)
    assert info.context.search_config.fusion == "auto"
    assert info.context.search_config.fusion_effective == "weighted"


# ===========================================================================
# list_workspaces()
# ===========================================================================

_WORKSPACES_REPLY = {
    "status": "success",
    "workspaces": [
        {"id": WS, "name": "Main", "role": "owner", "current": True, "usable": True},
        {
            "id": WS2,
            "name": "Team",
            "role": "viewer",
            "current": False,
            "usable": False,
            "reason": "workspace_suspended",
            "a_key_from_a_newer_server": 1,
        },
    ],
    "total": 2,
}


def test_account_workspace_models_are_exported() -> None:
    assert "AccountWorkspace" in kagura_memory.__all__
    assert "AccountWorkspaceList" in kagura_memory.__all__


@pytest.mark.asyncio
async def test_list_workspaces_calls_the_tool_with_no_arguments() -> None:
    result, name, arguments = await _run(_WORKSPACES_REPLY, lambda c: c.list_workspaces())
    assert name == "list_workspaces"
    assert arguments == {}
    assert isinstance(result, AccountWorkspaceList)
    assert result.total == 2
    main_ws, team = result.workspaces
    assert isinstance(main_ws, AccountWorkspace)
    assert (main_ws.id, main_ws.name, main_ws.role) == (WS, "Main", "owner")
    assert main_ws.current is True and main_ws.usable is True and main_ws.reason is None
    assert team.current is False and team.usable is False
    assert team.reason == "workspace_suspended"
    assert not hasattr(team, "a_key_from_a_newer_server")


def test_account_workspace_reason_is_a_plain_str() -> None:
    """A reason a newer server adds still parses."""
    row = AccountWorkspace.model_validate(
        {"id": WS, "name": "x", "role": "member", "current": False, "usable": False, "reason": "z"}
    )
    assert row.reason == "z"


@pytest.mark.asyncio
async def test_list_workspaces_on_an_older_server_is_an_unknown_tool() -> None:
    """Before v0.177.0 there is no such tool: ``unknown_tool`` (as v0.176.0 answers)."""
    refusal = {
        "status": "error",
        "error": "unknown_tool",
        "message": "Unknown tool: list_workspaces",
        "help": "Call tools/list to see the tools this URL lists; ...",
    }
    sent: list[dict[str, Any]] = []
    client = _wire_client(refusal, sent)
    try:
        with pytest.raises(KaguraError) as excinfo:
            await client.list_workspaces()
    finally:
        await client.close()
    assert excinfo.value.code == "unknown_tool"


# ===========================================================================
# list_contexts(workspace_id=)
# ===========================================================================

_CONTEXTS_REPLY = {
    "status": "success",
    "contexts": [],
    "count": 0,
    "total": 0,
    "limit": 25,
    "can_create": True,
    "has_more": False,
    "next_cursor": None,
    "other_workspaces": 1,
    "workspaces_hint": "You are a member of 1 other workspace: list_workspaces() lists them ...",
}


def test_list_contexts_workspace_id_is_keyword_only() -> None:
    assert _keyword_only(KaguraClient.list_contexts, "workspace_id")


@pytest.mark.asyncio
async def test_list_contexts_omits_workspace_id_by_default() -> None:
    result, name, arguments = await _run(_CONTEXTS_REPLY, lambda c: c.list_contexts())
    assert name == "list_contexts"
    assert arguments == {}
    assert result["other_workspaces"] == 1
    assert result["workspaces_hint"].startswith("You are a member of 1 other workspace")


@pytest.mark.asyncio
async def test_list_contexts_sends_workspace_id() -> None:
    _, _, arguments = await _run(
        _CONTEXTS_REPLY, lambda c: c.list_contexts(workspace_id=WS2, limit=10)
    )
    assert arguments == {"workspace_id": WS2, "limit": 10}


@pytest.mark.asyncio
async def test_list_contexts_of_another_workspace_passes_through() -> None:
    """Another workspace: no ``limit`` / ``can_create``, its own empty ``hint``."""
    reply = {
        "status": "success",
        "contexts": [],
        "count": 0,
        "total": 0,
        "has_more": False,
        "next_cursor": None,
        "hint": f"No contexts of that workspace are visible to you. ... (/mcp/w/{WS2}) ...",
        "other_workspaces": 1,
    }
    result, _, arguments = await _run(reply, lambda c: c.list_contexts(workspace_id=WS2))
    assert arguments == {"workspace_id": WS2}
    assert "limit" not in result and "can_create" not in result
    assert result["hint"].startswith("No contexts of that workspace")


@pytest.mark.asyncio
async def test_list_contexts_google_view_sends_the_count_only() -> None:
    """``?profile=google`` / ``microsoft``: ``other_workspaces`` without ``workspaces_hint``."""
    reply = {k: v for k, v in _CONTEXTS_REPLY.items() if k != "workspaces_hint"}
    result, _, _ = await _run(reply, lambda c: c.list_contexts())
    assert result["other_workspaces"] == 1
    assert "workspaces_hint" not in result


@pytest.mark.asyncio
async def test_list_contexts_of_a_workspace_not_reachable_is_permission_denied() -> None:
    refusal = {
        "status": "error",
        "error": "permission_denied",
        "message": "Cannot list contexts: workspace not accessible.",
        "your_role": "not_a_member",
        "required_role": "viewer",
    }
    sent: list[dict[str, Any]] = []
    client = _wire_client(refusal, sent)
    try:
        with pytest.raises(KaguraError) as excinfo:
            await client.list_contexts(workspace_id=WS2)
    finally:
        await client.close()
    assert excinfo.value.code == "permission_denied"
    assert excinfo.value.details["your_role"] == "not_a_member"


@pytest.mark.asyncio
async def test_list_contexts_of_a_workspace_without_a_seat_is_a_feature_gate() -> None:
    refusal = {
        "status": "error",
        "error": "feature_not_available",
        "message": "Your seat on this workspace's plan is paused.",
        "feature": "members",
    }
    client = _wire_client(refusal, [])
    try:
        with pytest.raises(KaguraFeatureNotAvailableError):
            await client.list_contexts(workspace_id=WS2)
    finally:
        await client.close()


# ===========================================================================
# CLI
# ===========================================================================


@pytest.fixture
def _cli_client(isolated_kagura_credentials):
    """``KaguraClient`` replaced in the CLI by an async mock, with a stored config."""
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    with (
        patch("kagura_memory.cli.load_config") as load_config,
        patch("kagura_memory.cli.KaguraClient", return_value=mock_client),
    ):
        load_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}
        yield mock_client


@pytest.mark.parametrize("value", ["auto", "rrf", "weighted"])
def test_cli_search_config_fusion(_cli_client: AsyncMock, value: str) -> None:
    """``--fusion`` alone is enough and is passed through."""
    _cli_client.update_search_config.return_value = {"status": "success"}
    result = CliRunner().invoke(main, ["context", "search-config", CTX, "--fusion", value])
    assert result.exit_code == 0, result.output
    assert _cli_client.update_search_config.await_args.kwargs["fusion"] == value


def test_cli_search_config_without_fusion_sends_none(_cli_client: AsyncMock) -> None:
    _cli_client.update_search_config.return_value = {"status": "success"}
    result = CliRunner().invoke(main, ["context", "search-config", CTX, "--rerank"])
    assert result.exit_code == 0, result.output
    assert _cli_client.update_search_config.await_args.kwargs["fusion"] is None


def test_cli_search_config_refuses_another_fusion(_cli_client: AsyncMock) -> None:
    result = CliRunner().invoke(main, ["context", "search-config", CTX, "--fusion", "fast"])
    assert result.exit_code == 2
    _cli_client.update_search_config.assert_not_called()


def test_cli_context_list_workspace_id(_cli_client: AsyncMock) -> None:
    _cli_client.list_contexts.return_value = _CONTEXTS_REPLY
    result = CliRunner().invoke(main, ["context", "list", "--workspace-id", WS2])
    assert result.exit_code == 0, result.output
    assert _cli_client.list_contexts.await_args.kwargs["workspace_id"] == WS2
    assert json.loads(result.output)["other_workspaces"] == 1


def test_cli_context_list_without_workspace_id_sends_none(_cli_client: AsyncMock) -> None:
    _cli_client.list_contexts.return_value = _CONTEXTS_REPLY
    result = CliRunner().invoke(main, ["context", "list"])
    assert result.exit_code == 0, result.output
    assert _cli_client.list_contexts.await_args.kwargs["workspace_id"] is None


def test_cli_workspace_list_table(_cli_client: AsyncMock) -> None:
    _cli_client.list_workspaces.return_value = AccountWorkspaceList.model_validate(
        _WORKSPACES_REPLY
    )
    result = CliRunner().invoke(main, ["workspace", "list"])
    assert result.exit_code == 0, result.output
    _cli_client.list_workspaces.assert_awaited_once_with()
    lines = result.output.splitlines()
    assert lines[0].split() == ["ID", "NAME", "ROLE", "CURRENT", "USABLE"]
    assert lines[1].split() == [WS, "Main", "owner", "yes", "yes"]
    assert lines[2].split() == [WS2, "Team", "viewer", "-", "no", "(workspace_suspended)"]


def test_cli_workspace_list_json(_cli_client: AsyncMock) -> None:
    _cli_client.list_workspaces.return_value = AccountWorkspaceList.model_validate(
        _WORKSPACES_REPLY
    )
    result = CliRunner().invoke(main, ["workspace", "list", "--json"])
    assert result.exit_code == 0, result.output
    body = json.loads(result.output)
    assert body["total"] == 2
    assert body["workspaces"][1]["reason"] == "workspace_suspended"
    assert body["workspaces"][0]["reason"] is None


def test_cli_workspace_list_error_is_a_click_error(_cli_client: AsyncMock) -> None:
    error = KaguraError("list_workspaces failed (unknown_tool): Unknown tool: list_workspaces")
    _cli_client.list_workspaces.side_effect = error
    result = CliRunner().invoke(main, ["workspace", "list"])
    assert result.exit_code == 1
    assert "unknown_tool" in result.output
