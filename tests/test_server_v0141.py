"""memory-cloud v0.141.0 realignment: response models, paging and write caps (#299).

**Models.** A v0.141.0-shaped payload parses into every changed model, and a
field an older server does not send reads as its default.

**Paging and caps** (memory-cloud v0.82.0, #1743). A cursor argument is sent
only when set; ``list_agents`` follows ``next_cursor`` and stops on a repeated
cursor or after 50 pages; ``list_edges`` (no cursor on the server) warns when a
direction was cut; the ``remember`` / ``update_memory`` tag caps and the
``set_state`` value cap are checked before any request, at the server's bound.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, get_args
from unittest.mock import AsyncMock, patch

import pytest

from kagura_memory import (
    Agent,
    AgentStatus,
    ContextInfo,
    EmbeddingStatus,
    IndexerStatusResponse,
    KaguraClient,
    KaguraResponseError,
    MemberAPIKey,
    ResourceEventBatchResponse,
    SleepAction,
    SleepReport,
    SleepReportSource,
    WorkspaceMember,
)
from tests.conftest import (
    agent_dict,
    indexer_status_dict,
    sleep_report_detail_dict,
    sleep_report_summary_dict,
)

CTX = "11111111-2222-3333-4444-555555555555"


def _client() -> KaguraClient:
    """A client whose MCP session is already open (``_call_tool`` is patched)."""
    client = KaguraClient(api_key="test", mcp_url="https://test.com/mcp")
    client._session_id = "pre-set-session"
    return client


def _keyword_only(method: Any, name: str) -> bool:
    return inspect.signature(method).parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


# ===========================================================================
# Models
# ===========================================================================


def test_known_value_aliases_track_the_server():
    # sleep_reports.source CHECK (d05_523) and the #2219 filter; agent statuses
    # from the list_agents tool description (v0.141.0).
    assert set(get_args(SleepReportSource)) == {"sleep", "analysis"}
    assert set(get_args(AgentStatus)) == {"active", "suspended", "retired"}


@pytest.mark.parametrize(
    ("sent", "expected"),
    [({"source": "analysis"}, "analysis"), ({"source": "future"}, "future"), ({}, None)],
    ids=["analysis", "unknown-value", "older-server"],
)
def test_sleep_report_reads_source(sent: dict, expected: str | None):
    """``source`` (v0.138.0+, #2219) is a ``str``; an older server omits it."""
    report = SleepReport.model_validate({**sleep_report_summary_dict(), **sent})
    assert report.source == expected


_ANALYSIS_NOTE = (
    "This is an analysis report (source 'analysis'): it records the LLM spend of a "
    "Memory Analysis run (cluster labeling)."
)


@pytest.mark.asyncio
async def test_get_sleep_report_reads_a_v0141_page():
    """Paging fields, ``note`` and an omitted action's marker reach the model."""
    client = _client()
    omitted = {
        "id": "57",
        "phase": "dedup",
        "action_type": "merge",
        "memory_id": "m-1",
        "target_id": "m-2",
        "created_at": "2026-11-03T00:00:00Z",
        "details_omitted": True,
        "details_total_chars": 31_000,
    }
    payload = {
        "status": "success",
        "report": {**sleep_report_detail_dict("rid-9"), "source": "analysis"},
        "actions": [omitted],
        "action_count": 120,
        "actions_has_more": True,
        "actions_next_cursor": "57",
        "note": _ANALYSIS_NOTE,
    }
    try:
        with patch.object(client, "_call_tool", new=AsyncMock(return_value=payload)):
            detail = await client.get_sleep_report("ctx-1", "rid-9")
    finally:
        await client.close()

    assert detail.source == "analysis"
    assert detail.action_count == 120
    assert detail.actions_has_more is True
    assert detail.actions_next_cursor == "57"
    assert detail.note == _ANALYSIS_NOTE
    action = detail.actions[0]
    assert isinstance(action, SleepAction)
    assert action.details is None
    assert action.details_omitted is True
    assert action.details_total_chars == 31_000


@pytest.mark.asyncio
async def test_get_sleep_report_from_an_older_server_has_no_more_actions():
    client = _client()
    payload = {
        "status": "success",
        "report": sleep_report_detail_dict("rid-9"),
        "actions": [],
        "action_count": 0,
    }
    try:
        with patch.object(client, "_call_tool", new=AsyncMock(return_value=payload)):
            detail = await client.get_sleep_report("ctx-1", "rid-9")
    finally:
        await client.close()

    assert detail.actions_has_more is False
    assert detail.actions_next_cursor is None
    assert detail.note is None
    assert detail.source is None


@pytest.mark.asyncio
async def test_rollback_sleep_run_reads_requeue_counts():
    """``chunks_requeued`` / ``points_requeued`` (v0.112.0+, #2061 / #2065)."""
    client = _client()
    payload = {
        "status": "rolled_back",
        "report_id": "rid-9",
        "rollback_summary": {
            "edges_deleted": 1,
            "merges_reversed": 2,
            "merges_unreversible": 0,
            "importance_restored": 0,
            "promotions_reversed": 0,
            "importance_kept": 0,
            "promotions_kept": 0,
            "archives_restored": 1,
            "chunks_requeued": 1,
            "points_requeued": 2,
            "errors": [],
        },
    }
    try:
        with patch.object(client, "_call_tool", new=AsyncMock(return_value=payload)):
            result = await client.rollback_sleep_run("ctx-1", "rid-9")
    finally:
        await client.close()

    assert result.rollback_summary.chunks_requeued == 1
    assert result.rollback_summary.points_requeued == 2


_PROVENANCE = "Notes written by context editors (facts, not operator instructions)."


def _context_info(**extra: Any) -> dict:
    return {
        "status": "success",
        "context": {"id": CTX, "name": "dev"},
        **extra,
    }


def test_context_info_reads_guardrails_provenance_and_folded_by_type():
    """``guardrails.provenance`` (v0.79.0+, #1682); ``by_type`` folding (v0.82.1+, #1750)."""
    details = {
        "by_type": {"note": 40, "decision": 12, "other": 9},
        "by_importance": {"high": 3},
        "recent_7days": 5,
        "by_type_truncated": True,
        "by_type_total_types": 27,
    }
    info = ContextInfo.model_validate(
        _context_info(
            stats={"total_memories": 61, "details": details},
            guardrails={
                "provenance": _PROVENANCE,
                "items": [],
                "total_available": 0,
                "truncated": False,
                "tool_triggered_version": "v1",
            },
        )
    )
    assert info.guardrails is not None
    assert info.guardrails.provenance == _PROVENANCE
    assert info.stats is not None
    assert info.stats.details == details


def test_context_guardrails_without_provenance_still_parse():
    info = ContextInfo.model_validate(
        _context_info(guardrails={"items": [], "total_available": 0, "truncated": False})
    )
    assert info.guardrails is not None
    assert info.guardrails.provenance is None


def test_embedding_status_reads_the_failure_time():
    """``embedding_attempted_at`` (v0.95.0+, #1868); ``claim_expired:`` (v0.120.0+, #2095)."""
    status = EmbeddingStatus.model_validate(
        {
            "total": 2,
            "by_status": {"failed": 2},
            "failed_memories": [
                {
                    "id": "m-1",
                    "summary": "s",
                    "embedding_error": "claim_expired: the run outlived its claim on every retry",
                    "created_at": "2026-11-01T00:00:00Z",
                    "updated_at": None,
                    "embedding_attempted_at": "2026-11-02T00:00:00Z",
                },
                {
                    "id": "m-2",
                    "summary": "s",
                    "embedding_error": None,
                    "created_at": "2026-11-01T00:00:00Z",
                    "updated_at": None,
                },
            ],
        }
    )
    first, older = status.failed_memories
    assert first.embedding_attempted_at is not None
    assert first.embedding_attempted_at.day == 2
    assert first.embedding_error is not None
    assert first.embedding_error.startswith("claim_expired:")
    assert older.embedding_attempted_at is None


@pytest.mark.parametrize(("metrics", "expected"), [({"failed_events": 3}, 3), ({}, 0)])
def test_indexer_metrics_read_failed_events(metrics: dict, expected: int):
    """``failed_events`` (v0.135.0+, #2254); ``0`` from an older server."""
    status = IndexerStatusResponse.model_validate(indexer_status_dict(**metrics))
    assert status.state is not None
    assert status.state.metrics.failed_events == expected


def test_indexer_metrics_keep_a_plan_suspended_skip():
    status = IndexerStatusResponse.model_validate(
        indexer_status_dict(skipped_reason="plan_suspended")
    )
    assert status.state is not None
    assert status.state.metrics.skipped_reason == "plan_suspended"


@pytest.mark.parametrize(("sent", "expected"), [({"seat_paused": True}, True), ({}, False)])
def test_workspace_member_reads_seat_paused(sent: dict, expected: bool):
    """``seat_paused`` (v0.100.0+, #1943)."""
    member = WorkspaceMember.model_validate(
        {"user_id": "u-1", "role": "member", "joined_at": "2026-11-01T00:00:00Z", **sent}
    )
    assert member.seat_paused is expected


def test_member_api_key_reads_v0141_fields():
    """Default context + agent (v0.118.0+, #2188) and connector (v0.129.0+, #2299)."""
    row = {
        "id": "akey_1",
        "name": "connector:github",
        "key_prefix": "kgr_abc",
        "plaintext_key": None,
        "is_visible": False,
        "visibility_expires_at": None,
        "created_at": "2026-11-01T00:00:00Z",
        "revoked_at": None,
    }
    key = MemberAPIKey.model_validate(
        {
            **row,
            "default_context_id": CTX,
            "default_context_name": "dev",
            "agent_id": None,
            "connector_id": "conn-1",
        }
    )
    assert key.default_context_id == CTX
    assert key.default_context_name == "dev"
    assert key.agent_id is None
    assert key.connector_id == "conn-1"

    older = MemberAPIKey.model_validate(row)
    assert (older.default_context_id, older.default_context_name) == (None, None)
    assert (older.agent_id, older.connector_id) == (None, None)


def test_batch_ingest_error_rows_stay_mappings():
    """The REST batch route's rows are ``{index, doc_id, error}`` (no #1742 cause)."""
    rows = [{"index": 1, "doc_id": "SKU-1", "error": "Duplicate version 3"}]
    batch = ResourceEventBatchResponse.model_validate(
        {
            "status": "success",
            "created_count": 1,
            "failed_count": 1,
            "event_ids": [7],
            "errors": rows,
        }
    )
    assert batch.errors == rows


# ===========================================================================
# Sleep arguments
# ===========================================================================

_GET_SLEEP_REPORT_DECLARED = {"report_id", "actions_cursor", "actions_limit", "max_chars"}


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, {"report_id": "rid-9"}),
        ({"actions_cursor": "57"}, {"report_id": "rid-9", "actions_cursor": "57"}),
        (
            {"actions_cursor": "57", "actions_limit": 200, "max_chars": 50_000},
            {
                "report_id": "rid-9",
                "actions_cursor": "57",
                "actions_limit": 200,
                "max_chars": 50_000,
            },
        ),
    ],
    ids=["none", "cursor", "all"],
)
@pytest.mark.asyncio
async def test_get_sleep_report_sends_paging_arguments_only_when_set(kwargs: dict, expected: dict):
    """Only declared arguments go out (v0.82.0+ refuses others) — never ``context_id`` (#304)."""
    client = _client()
    payload = {
        "status": "success",
        "report": sleep_report_detail_dict("rid-9"),
        "actions": [],
        "action_count": 0,
    }
    try:
        with patch.object(client, "_call_tool", new=AsyncMock(return_value=payload)) as mock:
            await client.get_sleep_report("ctx-1", "rid-9", **kwargs)
    finally:
        await client.close()

    mock.assert_called_once_with("get_sleep_report", expected)
    assert set(expected) <= _GET_SLEEP_REPORT_DECLARED
    for name in ("actions_cursor", "actions_limit", "max_chars"):
        assert _keyword_only(KaguraClient.get_sleep_report, name)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (None, {"context_id": "ctx-1", "limit": 10}),
        ("all", {"context_id": "ctx-1", "limit": 10, "source": "all"}),
    ],
)
@pytest.mark.asyncio
async def test_get_sleep_history_sends_source_only_when_set(source: str | None, expected: dict):
    """``source`` (v0.138.0+, #2219) is omitted when unset, so older servers never see it."""
    client = _client()
    payload = {"status": "success", "reports": [sleep_report_summary_dict()], "count": 1}
    try:
        with patch.object(client, "_call_tool", new=AsyncMock(return_value=payload)) as mock:
            await client.get_sleep_history("ctx-1", source=source)
    finally:
        await client.close()

    mock.assert_called_once_with("get_sleep_history", expected)
    assert _keyword_only(KaguraClient.get_sleep_history, "source")


# ===========================================================================
# Cursor paging
# ===========================================================================


@pytest.mark.parametrize(
    ("cursor", "expected"), [(None, {}), ("100", {"cursor": "100"})], ids=["unset", "set"]
)
@pytest.mark.asyncio
async def test_list_contexts_sends_cursor_only_when_set(cursor: str | None, expected: dict):
    client = _client()
    page = {
        "status": "success",
        "contexts": [{"id": CTX, "name": "dev"}],
        "count": 150,
        "total": 1,
        "has_more": True,
        "next_cursor": "101",
    }
    try:
        with patch.object(client, "_call_tool", new=AsyncMock(return_value=page)) as mock:
            result = await client.list_contexts(cursor=cursor)
    finally:
        await client.close()

    mock.assert_called_once_with("list_contexts", expected)
    assert (result["has_more"], result["next_cursor"]) == (True, "101")
    assert _keyword_only(KaguraClient.list_contexts, "cursor")


@pytest.mark.asyncio
async def test_get_state_sends_cursor_in_list_mode():
    client = _client()
    page = {"status": "success", "states": {"b": 2}, "count": 1, "has_more": False}
    try:
        with patch.object(client, "_call_tool", new=AsyncMock(return_value=page)) as mock:
            result = await client.get_state(CTX, cursor="a")
    finally:
        await client.close()

    mock.assert_called_once_with("get_state", {"context_id": CTX, "cursor": "a"})
    assert result["states"] == {"b": 2}
    assert _keyword_only(KaguraClient.get_state, "cursor")


@pytest.mark.asyncio
async def test_get_state_refuses_a_cursor_with_a_key():
    client = _client()
    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            with pytest.raises(ValueError, match="omit key"):
                await client.get_state(CTX, "step", cursor="a")
    finally:
        await client.close()
    mock.assert_not_called()


def _agents_page(*names: str, next_cursor: str | None = None) -> dict:
    return {
        "status": "success",
        "agents": [agent_dict(name=n) for n in names],
        "count": len(names),
        "has_more": next_cursor is not None,
        "next_cursor": next_cursor,
    }


@pytest.mark.asyncio
async def test_list_agents_follows_next_cursor():
    """Every page is read; the first call sends nothing, then only ``cursor``."""
    client = _client()
    pages = [_agents_page("a", "b", next_cursor="2"), _agents_page("c")]
    try:
        with patch.object(client, "_call_tool", new=AsyncMock(side_effect=pages)) as mock:
            agents = await client.list_agents()
    finally:
        await client.close()

    assert [a.name for a in agents] == ["a", "b", "c"]
    assert all(isinstance(a, Agent) for a in agents)
    assert [c.args for c in mock.call_args_list] == [
        ("list_agents", {}),
        ("list_agents", {"cursor": "2"}),
    ]


@pytest.mark.asyncio
async def test_list_agents_from_an_older_server_is_one_call():
    client = _client()
    page = {"status": "success", "agents": [agent_dict()], "count": 1}
    try:
        with patch.object(client, "_call_tool", new=AsyncMock(return_value=page)) as mock:
            agents = await client.list_agents()
    finally:
        await client.close()

    assert len(agents) == 1
    mock.assert_called_once_with("list_agents", {})


def test_list_agents_row_reads_a_cut_description():
    """``description`` is a 200-character preview in a v0.82.0+ list row."""
    agent = Agent.model_validate(
        agent_dict(description="x" * 200 + "…", description_truncated=True)
    )
    assert agent.description_truncated is True
    assert Agent.model_validate(agent_dict()).description_truncated is False


@pytest.mark.parametrize(
    ("pages", "match"),
    [
        (
            [_agents_page("a", next_cursor="1"), _agents_page("b", next_cursor="1")],
            "repeated an earlier next_cursor",
        ),
        ([{**_agents_page("a"), "has_more": True}], "has_more without a next_cursor"),
    ],
    ids=["repeated-cursor", "missing-cursor"],
)
@pytest.mark.asyncio
async def test_list_agents_stops_on_a_broken_cursor(pages: list[dict], match: str):
    client = _client()
    try:
        with patch.object(client, "_call_tool", new=AsyncMock(side_effect=pages)):
            with pytest.raises(KaguraResponseError, match=match) as exc:
                await client.list_agents()
    finally:
        await client.close()
    assert exc.value.operation == "list_agents"


@pytest.mark.asyncio
async def test_list_agents_gives_up_after_50_pages():
    client = _client()
    pages = [_agents_page(f"a{i}", next_cursor=str(i + 1)) for i in range(51)]
    try:
        with patch.object(client, "_call_tool", new=AsyncMock(side_effect=pages)) as mock:
            with pytest.raises(KaguraResponseError, match="did not end within 50 pages"):
                await client.list_agents()
    finally:
        await client.close()
    assert mock.call_count == 50


@pytest.mark.parametrize(
    ("flags", "warned"),
    [
        ({"outgoing_has_more": True, "incoming_has_more": False}, "outgoing"),
        ({"outgoing_has_more": True, "incoming_has_more": True}, "outgoing and incoming"),
        ({"outgoing_has_more": False, "incoming_has_more": False}, None),
        ({}, None),
    ],
    ids=["outgoing", "both", "complete", "older-server"],
)
@pytest.mark.asyncio
async def test_list_edges_warns_when_a_direction_was_cut(
    flags: dict, warned: str | None, caplog: pytest.LogCaptureFixture
):
    """The tool has no cursor: a cut direction is logged, the list returned as is."""
    client = _client()
    edge = {
        "source_id": "m-1",
        "target_id": "m-2",
        "edge_type": "related_to",
        "weight": 0.5,
        "confidence": 1.0,
    }
    payload = {"status": "success", "memory_id": "m-1", "edges": [edge], "count": 1, **flags}
    try:
        with (
            patch.object(client, "_call_tool", new=AsyncMock(return_value=payload)),
            caplog.at_level(logging.WARNING, logger="kagura_memory"),
        ):
            edges = await client.list_edges(CTX, "m-1")
    finally:
        await client.close()

    assert len(edges) == 1
    if warned is None:
        assert "list_edges" not in caplog.text
    else:
        assert f"more {warned} edges exist" in caplog.text


@pytest.mark.parametrize(("sent", "expected"), [({"has_more": True}, True), ({}, False)])
@pytest.mark.asyncio
async def test_list_tags_reads_has_more(sent: dict, expected: bool):
    """A v0.82.0+ MCP reply holds at most 200 tags and says when more matched."""
    client = _client()
    payload = {
        "status": "success",
        "context_id": CTX,
        "context_name": "dev",
        "tags": [{"tag": "python", "count": 3, "last_used_at": None}],
        "total": 1,
        **sent,
    }
    try:
        with patch.object(client, "_call_tool", new=AsyncMock(return_value=payload)):
            result = await client.list_tags(CTX)
    finally:
        await client.close()
    assert result.has_more is expected


@pytest.mark.asyncio
async def test_list_tags_still_accepts_the_servers_limit_of_500():
    """The server's bound is 1-500 on MCP and REST; only an MCP page is held to 200."""
    client = _client()
    payload = {
        "status": "success",
        "context_id": CTX,
        "context_name": "dev",
        "tags": [],
        "total": 0,
    }
    try:
        with patch.object(client, "_call_tool", new=AsyncMock(return_value=payload)) as mock:
            await client.list_tags(CTX, limit=500)
    finally:
        await client.close()
    assert mock.call_args.args[1]["limit"] == 500


# ===========================================================================
# Write caps
# ===========================================================================


def _tags(count: int, length: int = 5) -> list[str]:
    return [f"{i:0{length}d}"[-length:] for i in range(count)]


@pytest.mark.parametrize(
    ("tags", "match"),
    [
        (_tags(50, 100), None),
        (_tags(51), "tags accepts at most 50 tags, got 51"),
        (["x" * 101], "each tag must be at most 100 characters, got 101"),
    ],
    ids=["at-the-bound", "51-tags", "101-chars"],
)
@pytest.mark.asyncio
async def test_remember_checks_the_tag_caps(tags: list[str], match: str | None):
    client = _client()
    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = {"status": "success", "memory_id": "m-1"}
            if match is None:
                await client.remember(CTX, "summary text", "content", tags=tags)
                assert mock.call_args.args[1]["tags"] == tags
            else:
                with pytest.raises(ValueError, match=match):
                    await client.remember(CTX, "summary text", "content", tags=tags)
                mock.assert_not_called()
    finally:
        await client.close()


@pytest.mark.parametrize(
    ("tags", "match"),
    [
        (_tags(50, 100), None),
        (_tags(51), "tags accepts at most 50 tags, got 51"),
        (["ok", "y" * 101], "each tag must be at most 100 characters, got 101"),
    ],
    ids=["at-the-bound", "51-tags", "101-chars"],
)
@pytest.mark.asyncio
async def test_update_memory_checks_the_tag_caps(tags: list[str], match: str | None):
    client = _client()
    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = {"status": "success", "memory_id": "m-1"}
            if match is None:
                await client.update_memory(CTX, memory_id="m-1", tags=tags)
                assert mock.call_args.args[1]["tags"] == tags
            else:
                with pytest.raises(ValueError, match=match):
                    await client.update_memory(CTX, memory_id="m-1", tags=tags)
                mock.assert_not_called()
    finally:
        await client.close()


# Compact JSON of a string is its characters plus two quotes, so these sit at
# the 16,384-character bound and one past it. A non-ASCII value counts one
# character per code point (``ensure_ascii=False``), as the server counts it.
@pytest.mark.parametrize(
    ("value", "allowed"),
    [
        ("x" * 16_382, True),
        ("x" * 16_383, False),
        ("あ" * 16_382, True),
        ({"k": "x" * 16_376}, True),
        ({"k": "x" * 16_377}, False),
    ],
    ids=["ascii-at-bound", "ascii-over", "non-ascii-at-bound", "object-at-bound", "object-over"],
)
@pytest.mark.asyncio
async def test_set_state_checks_the_value_cap(value: Any, allowed: bool):
    client = _client()
    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = {"status": "success", "key": "k"}
            if allowed:
                await client.set_state(CTX, "k", value)
                assert mock.call_args.args[1]["value"] == value
            else:
                with pytest.raises(ValueError, match="at most 16384 characters as compact JSON"):
                    await client.set_state(CTX, "k", value)
                mock.assert_not_called()
    finally:
        await client.close()
