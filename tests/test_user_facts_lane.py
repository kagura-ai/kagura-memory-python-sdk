"""memory-cloud v0.184.0: the user facts lane (memory-cloud#3014).

**Recall reply.** With a context's ``user_facts_lane`` on, ``recall`` returns an
optional top-level ``user_facts`` key: rows of the ``results`` shape, never
also in ``results`` and not counted by ``count``. Present (possibly empty)
when the lane ran; absent (never ``null``) when it did not. A row carries
``content`` only when the call's ``full`` is 1 or more. The SDK passes the
key through in the reply dict as sent, kept apart from ``results``.

**Search config.** ``update_search_config(user_facts_lane=)`` is keyword-only
and sent only when not ``None``; ``SearchConfig.user_facts_lane`` reads it
(``None`` from an older server). CLI ``kagura context search-config
--[no-]user-facts-lane``; ``kagura recall`` names the user facts on stderr
after the JSON when the key is present.
"""

from __future__ import annotations

import inspect
import json
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from click.testing import CliRunner

from kagura_memory import ContextInfo, KaguraClient, SearchConfig
from kagura_memory.cli import main

CTX = "11111111-2222-3333-4444-555555555555"
WS = "aaaaaaaa-0000-0000-0000-000000000001"
MEM_HIT = "aaaaaaaa-0000-0000-0000-000000000010"
MEM_FACT = "aaaaaaaa-0000-0000-0000-000000000020"

_HIT = {
    "memory_id": MEM_HIT,
    "summary": "Moving day checklist for the new flat",
    "type": "note",
    "importance": 0.5,
    "score": 0.74,
    "created_at": "2026-11-20T09:00:00Z",
}
_FACT = {
    "memory_id": MEM_FACT,
    "summary": "The user injured a knee and avoids stairs for now",
    "type": "fact",
    "importance": 0.7,
    "score": 0.31,
    "created_at": "2026-11-18T08:00:00Z",
    "observed_at": "2026-11-17",
}
_FACT_WITH_CONTENT = {**_FACT, "content": "Physio said no heavy loads on stairs for six weeks."}


def _reply(**extra: Any) -> dict[str, Any]:
    """Shaped like memory-cloud v0.184.0's recall envelope (user_facts last)."""
    return {
        "confidence": {"level": "high", "top_score": 0.74, "result_count": 1},
        "status": "success",
        "results": [_HIT],
        "count": 1,
        "context_id": CTX,
        "context_name": "dev",
        **extra,
    }


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


async def _run(reply: dict[str, Any], call: Any) -> tuple[Any, str, dict[str, Any]]:
    sent: list[dict[str, Any]] = []
    client = _wire_client(reply, sent)
    try:
        result = await call(client)
    finally:
        await client.close()
    assert len(sent) == 1
    assert sent[0]["method"] == "tools/call"
    return result, sent[0]["params"]["name"], sent[0]["params"]["arguments"]


# ===========================================================================
# recall: the user_facts key
# ===========================================================================


@pytest.mark.asyncio
async def test_recall_returns_user_facts_rows_apart_from_results() -> None:
    result, name, arguments = await _run(
        _reply(user_facts=[_FACT]), lambda c: c.recall(context_id=CTX, query="carry bags upstairs")
    )
    assert name == "recall"
    # Nothing new is sent: the lane is a context setting, not a recall argument.
    assert arguments == {"query": "carry bags upstairs", "k": 5, "context_id": CTX}
    assert result["user_facts"] == [_FACT]
    assert result["results"] == [_HIT]
    assert result["count"] == 1
    fact_ids = {row["memory_id"] for row in result["user_facts"]}
    assert fact_ids.isdisjoint(row["memory_id"] for row in result["results"])


@pytest.mark.asyncio
async def test_recall_user_facts_row_carries_content_with_full() -> None:
    result, _, arguments = await _run(
        _reply(user_facts=[_FACT_WITH_CONTENT]),
        lambda c: c.recall(context_id=CTX, query="carry bags upstairs", full=1),
    )
    assert arguments["full"] == 1
    assert result["user_facts"][0]["content"].startswith("Physio said")


@pytest.mark.asyncio
async def test_recall_user_facts_row_without_content_with_full_zero() -> None:
    result, _, arguments = await _run(
        _reply(user_facts=[_FACT]),
        lambda c: c.recall(context_id=CTX, query="carry bags upstairs", full=0),
    )
    assert arguments["full"] == 0
    assert "content" not in result["user_facts"][0]


@pytest.mark.asyncio
async def test_recall_without_the_key_has_no_user_facts() -> None:
    """The lane did not run (or an older server): the key is absent, not an empty list."""
    result, _, _ = await _run(_reply(), lambda c: c.recall(context_id=CTX, query="q"))
    assert "user_facts" not in result
    assert result.get("user_facts") is None


@pytest.mark.asyncio
async def test_recall_with_an_empty_lane_keeps_the_empty_list() -> None:
    """The lane ran and found nothing: present and empty, unlike a lane that did not run."""
    result, _, _ = await _run(_reply(user_facts=[]), lambda c: c.recall(context_id=CTX, query="q"))
    assert "user_facts" in result
    assert result["user_facts"] == []


def test_recall_docstring_describes_user_facts() -> None:
    doc = inspect.getdoc(KaguraClient.recall) or ""
    for term in ("user_facts", "v0.184.0", "#3014", "user_facts_lane"):
        assert term in doc, f"recall's docstring lacks {term!r}"


# ===========================================================================
# update_search_config(user_facts_lane=) and SearchConfig.user_facts_lane
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
        "fusion": "auto",
        "fusion_effective": "weighted",
        "user_facts_lane": True,
        "content_chunks_enabled": False,
        "content_chunks_effective": False,
        "llm_selector_enabled": False,
        "llm_selector_effective": False,
    },
}


def test_update_search_config_user_facts_lane_is_keyword_only() -> None:
    param = inspect.signature(KaguraClient.update_search_config).parameters["user_facts_lane"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is None


@pytest.mark.asyncio
async def test_update_search_config_omits_user_facts_lane_by_default() -> None:
    """The request an older server gets is unchanged when the lane is not set."""
    _, name, arguments = await _run(
        _SEARCH_CONFIG_REPLY, lambda c: c.update_search_config(CTX, use_rerank=True)
    )
    assert name == "update_search_config"
    assert arguments == {"context_id": CTX, "use_rerank": True}


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [True, False])
async def test_update_search_config_sends_user_facts_lane(value: bool) -> None:
    result, _, arguments = await _run(
        _SEARCH_CONFIG_REPLY, lambda c: c.update_search_config(CTX, user_facts_lane=value)
    )
    assert arguments == {"context_id": CTX, "user_facts_lane": value}
    assert result["config"]["user_facts_lane"] is True
    assert SearchConfig.model_validate(result["config"]).user_facts_lane is True


@pytest.mark.parametrize(
    ("sent", "expected"),
    [({"user_facts_lane": True}, True), ({"user_facts_lane": False}, False), ({}, None)],
    ids=["on", "off", "older-server"],
)
def test_search_config_reads_user_facts_lane(sent: dict, expected: bool | None) -> None:
    config = SearchConfig.model_validate({"semantic_weight": 0.6, "bm25_weight": 0.4, **sent})
    assert config.user_facts_lane is expected


def test_search_config_round_trips_user_facts_lane() -> None:
    config = SearchConfig.model_validate({"user_facts_lane": True})
    dumped = config.model_dump()
    assert dumped["user_facts_lane"] is True
    assert SearchConfig.model_validate(dumped) == config


@pytest.mark.asyncio
async def test_get_context_info_reads_user_facts_lane() -> None:
    reply = {
        "status": "success",
        "context": {
            "id": CTX,
            "name": "dev",
            "search_config": {
                "semantic_weight": 0.6,
                "bm25_weight": 0.4,
                "fetch_factor": 3,
                "use_rerank": False,
                "user_facts_lane": True,
            },
        },
        "workspace": {"id": WS, "name": "Main"},
        "stats": {"total_memories": 0},
        "instructions": "",
    }
    info, name, _ = await _run(reply, lambda c: c.get_context_info(CTX))
    assert name == "get_context_info"
    assert isinstance(info, ContextInfo)
    assert info.context.search_config.user_facts_lane is True


# ===========================================================================
# CLI
# ===========================================================================


def _mock_client(mock_client_cls: Any, mock_config: Any, method: str, reply: Any) -> AsyncMock:
    mock_config.return_value = {
        "api_key": "key",
        "mcp_url": "https://test.com/mcp",
        "context_id": "ctx",
    }
    mock_client = AsyncMock()
    getattr(mock_client, method).return_value = reply
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client
    return mock_client


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_cli_recall_prints_user_facts_after_the_results(mock_client_cls, mock_config):
    _mock_client(mock_client_cls, mock_config, "recall", _reply(user_facts=[_FACT]))
    result = CliRunner().invoke(main, ["recall", "carry bags upstairs"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["user_facts"] == [_FACT]
    assert payload["results"] == [_HIT]
    assert result.stdout.index('"results"') < result.stdout.index('"user_facts"')
    assert "user facts (1, not in results):" in result.stderr
    assert f"- {_FACT['summary']} ({MEM_FACT})" in result.stderr


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_cli_recall_names_an_empty_user_facts_lane(mock_client_cls, mock_config):
    _mock_client(mock_client_cls, mock_config, "recall", _reply(user_facts=[]))
    result = CliRunner().invoke(main, ["recall", "q"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["user_facts"] == []
    assert "user facts: none" in result.stderr


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_cli_recall_without_user_facts_prints_no_section(mock_client_cls, mock_config):
    _mock_client(mock_client_cls, mock_config, "recall", _reply())
    result = CliRunner().invoke(main, ["recall", "q"])
    assert result.exit_code == 0, result.output
    assert "user_facts" not in json.loads(result.stdout)
    assert "user facts" not in result.stderr


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_cli_recall_user_facts_row_without_summary(mock_client_cls, mock_config):
    """A row missing a summary or id still prints (newer or trimmed rows)."""
    _mock_client(mock_client_cls, mock_config, "recall", _reply(user_facts=[{"type": "fact"}]))
    result = CliRunner().invoke(main, ["recall", "q"])
    assert result.exit_code == 0, result.output
    assert "user facts (1, not in results):" in result.stderr
    assert "- (no summary)" in result.stderr


@pytest.mark.parametrize(
    ("flags", "expected"),
    [(("--user-facts-lane",), True), (("--no-user-facts-lane",), False)],
    ids=["on", "off"],
)
@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_cli_search_config_user_facts_lane(mock_client_cls, mock_config, flags, expected):
    mock_client = _mock_client(
        mock_client_cls, mock_config, "update_search_config", _SEARCH_CONFIG_REPLY
    )
    result = CliRunner().invoke(main, ["context", "search-config", CTX, *flags])
    assert result.exit_code == 0, result.output
    kwargs = mock_client.update_search_config.call_args.kwargs
    assert kwargs["user_facts_lane"] is expected
    assert json.loads(result.output)["config"]["user_facts_lane"] is True


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_cli_search_config_without_the_flag_leaves_it_unset(mock_client_cls, mock_config):
    mock_client = _mock_client(
        mock_client_cls, mock_config, "update_search_config", _SEARCH_CONFIG_REPLY
    )
    result = CliRunner().invoke(main, ["context", "search-config", CTX, "--fusion", "rrf"])
    assert result.exit_code == 0, result.output
    assert mock_client.update_search_config.call_args.kwargs["user_facts_lane"] is None


def test_cli_search_config_help_names_the_flag():
    result = CliRunner().invoke(main, ["context", "search-config", "--help"])
    assert result.exit_code == 0, result.output
    assert "--user-facts-lane / --no-user-facts-lane" in result.output
    assert "v0.184.0" in result.output
