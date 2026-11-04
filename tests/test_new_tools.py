"""Session tools on ``KaguraClient`` (#301) and the plan headroom notice (#314).

``bootstrap``, ``changes_since``, ``list`` (``list_matching``),
``remember_batch``, ``forget(memory_ids=)``, ``describe_tools``, ``guide``,
``list_revisions`` and ``rollback_session``. Every reply below is shaped like
memory-cloud v0.142.0's handler for that tool (``mcp_server/tools/bootstrap.py``
+ ``services/session_bootstrap_service.py``, ``listing.py``, ``batch.py``,
``memory.py::_handle_forget_many``, ``describe.py``, ``guide.py``,
``revisions.py``, ``session_rollback.py``) and the ``plan_headroom`` entries
like ``services/plan_headroom.py``. Calls go through the real JSON-RPC
transport so the exact arguments on the wire are asserted, an unset argument
being absent.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from kagura_memory import (
    BootstrapResponse,
    ContextGuardrails,
    ForgetBatchResult,
    KaguraClient,
    KaguraError,
    KaguraNotFoundError,
    KaguraPartialRollbackError,
    KaguraPartialSessionRollbackError,
    KaguraQuotaError,
    KaguraResponseError,
    PlanHeadroomNotice,
)

CTX = "11111111-2222-3333-4444-555555555555"
MEM_A = "aaaaaaaa-0000-0000-0000-000000000001"
MEM_B = "aaaaaaaa-0000-0000-0000-000000000002"
MEM_C = "aaaaaaaa-0000-0000-0000-000000000003"
MEM_D = "aaaaaaaa-0000-0000-0000-000000000004"

_CONTEXT_FIELDS: dict[str, Any] = {
    "context_id": CTX,
    "context_name": "dev",
    "context_display_name": "Dev",
    "context_is_private": False,
    "context_is_locked": False,
}
_CONTEXT_BLOCK: dict[str, Any] = {
    "id": CTX,
    "name": "dev",
    "display_name": "Dev",
    "summary": "SDK development notes",
    "usage_guide": "Store decisions and lessons.",
    "is_private": False,
    "is_locked": False,
    "embedding_model": "text-embedding-3-small",
    "embedding_dimensions": 1536,
}
_PROVENANCE = "Notes written by context editors (facts, not operator instructions)."

# plan_headroom as services/plan_headroom.py renders it: the daily cap carries
# resets_at, and a bootstrap whose notice passed a quarter of max_chars keeps
# help on the most severe entry only (fit_headroom).
_HEADROOM_CRITICAL: dict[str, Any] = {
    "dimension": "memories_per_day",
    "level": "critical",
    "used": 97,
    "limit": 100,
    "percentage": 97.0,
    "resets_at": "2026-11-05T00:00:00Z",
    "plan": "free",
    "remedies": ["wait", "upgrade"],
    "help": "Wait for the daily reset at 2026-11-05T00:00:00Z, or upgrade the plan from the "
    "workspace plan page in the web UI (Workspace settings → Plan). At 100 % remember will be "
    "refused until the reset.",
}
_HEADROOM_WARNING_NO_HELP: dict[str, Any] = {
    "dimension": "memories",
    "level": "warning",
    "used": 4100,
    "limit": 5000,
    "percentage": 82.0,
    "plan": "free",
    "remedies": ["cleanup", "upgrade"],
}

# format="index", max_chars=5000, with a cut changes lane (#2126, #2485).
INDEX_REPLY: dict[str, Any] = {
    "status": "success",
    "format": "index",
    "degraded": False,
    "context": _CONTEXT_BLOCK,
    "guardrails": {
        "provenance": _PROVENANCE,
        "items": [
            {
                "memory_id": MEM_D,
                "summary": "Never run `git push --force` on main; open a PR from a branch and a…",
                "summary_cut": True,
                "importance": 0.9,
            }
        ],
        "total_available": 1,
        "truncated": False,
        "tool_triggered_version": "tt-1",
    },
    "components": {
        "pinned": {
            "status": "ok",
            "memories": [
                {
                    "memory_id": MEM_A,
                    "summary": "Goal: ship v0.44.0 with the session tools; every argument is "
                    "sent only when set and every reply is typed.",
                    "type": "goal",
                    "importance": 0.9,
                }
            ],
            "total_available": 1,
            "truncated": False,
            "cap": 20,
        },
        "upcoming": {
            "status": "ok",
            "results": [
                {
                    "memory_id": MEM_B,
                    "summary": "Release review",
                    "from": "2026-11-10T00:00:00",
                    "until": "2026-11-10T23:59:59",
                }
            ],
            "from": "2026-11-04T09:00:00",
        },
        "changes": {
            "status": "ok",
            "changes": [
                {
                    "memory_id": MEM_C,
                    "kind": "created",
                    "at": "2026-11-01T10:00:00Z",
                    "summary": "Decision: bootstrap wraps the bootstrap tool, not "
                    "get_agent_bootstrap, and keeps…",
                    "summary_cut": True,
                },
                {"memory_id": MEM_D, "kind": "forgotten", "at": "2026-11-02T10:00:00Z"},
            ],
            "has_more": True,
            "next_cursor": "MjAyNi0xMS0wMlQxMDowMDowMHxmb3Jnb3R0ZW58YWFh",
            "truncated": True,
        },
    },
    "since": "2026-10-28T09:00:00Z",
    "generated_at": "2026-11-04T09:00:00Z",
    **_CONTEXT_FIELDS,
    "plan_headroom": [_HEADROOM_WARNING_NO_HELP, _HEADROOM_CRITICAL],
}

# Full format (max_chars >= 10,000): instructions, uncut guardrails, and the
# pinned rows' context_summary dropped by the budget.
FULL_REPLY: dict[str, Any] = {
    "status": "success",
    "degraded": False,
    "context": _CONTEXT_BLOCK,
    "instructions": "Kagura Memory quick reference …",
    "guardrails": {
        "provenance": _PROVENANCE,
        "items": [
            {
                "memory_id": MEM_D,
                "summary": "Never run `git push --force` on main; open a PR from a branch and "
                "merge it after review.",
                "importance": 0.9,
                "authored_by_caller": True,
                "source_type": "manual",
            }
        ],
        "total_available": 1,
        "truncated": False,
        "tool_triggered_version": "tt-1",
    },
    "components": {
        "pinned": {
            "status": "ok",
            "memories": [
                {
                    "memory_id": MEM_A,
                    "summary": "Goal: ship v0.44.0",
                    "type": "goal",
                    "importance": 0.9,
                    "delivery_mode": "always",
                }
            ],
            "total_available": 1,
            "truncated": False,
            "cap": 20,
        },
        "upcoming": {"status": "ok", "results": [], "from": "2026-11-04T09:00:00"},
        "changes": {"status": "ok", "changes": [], "has_more": False, "next_cursor": None},
    },
    "since": "2026-10-28T09:00:00Z",
    "generated_at": "2026-11-04T09:00:00Z",
    "context_summary_omitted": True,
    **_CONTEXT_FIELDS,
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
        result: dict[str, Any] = {"content": [{"type": "text", "text": text}]}
        if reply.get("status") == "error":
            result["isError"] = True
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": result})

    client = _client()
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


async def _call(reply: dict[str, Any], method: str, *args: Any, **kwargs: Any) -> tuple[Any, dict]:
    """Run ``client.<method>`` against ``reply``; return its result and the tools/call params."""
    sent: list[dict[str, Any]] = []
    client = _wire_client(reply, sent)
    try:
        result = await getattr(client, method)(*args, **kwargs)
    finally:
        await client.close()
    assert len(sent) == 1
    assert sent[0]["method"] == "tools/call"
    return result, sent[0]["params"]


async def _refused_locally(method: str, *args: Any, error: type[Exception], **kwargs: Any) -> str:
    """``client.<method>`` raises ``error`` before any request; return its message."""
    sent: list[dict[str, Any]] = []
    client = _wire_client({"status": "success"}, sent)
    try:
        with pytest.raises(error) as info:
            await getattr(client, method)(*args, **kwargs)
    finally:
        await client.close()
    assert sent == []
    return str(info.value)


def _error(code: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"status": "error", "error": code, "message": message, **extra}


# ============================================================================
# PlanHeadroomNotice
# ============================================================================


def test_plan_headroom_notice_reads_a_v0142_entry():
    notice = PlanHeadroomNotice.model_validate(_HEADROOM_CRITICAL)
    assert notice.dimension == "memories_per_day" and notice.level == "critical"
    assert (notice.used, notice.limit, notice.percentage) == (97, 100, 97.0)
    assert notice.plan == "free" and notice.remedies == ["wait", "upgrade"]
    assert notice.resets_at == datetime(2026, 11, 5, tzinfo=UTC)
    assert notice.help is not None and "remember will be refused" in notice.help


def test_plan_headroom_notice_tolerates_new_values_and_left_out_fields():
    """``level`` / ``dimension`` are plain str; ``plan`` / ``help`` / ``resets_at`` optional."""
    notice = PlanHeadroomNotice.model_validate(
        {
            "dimension": "storage_bytes",
            "level": "exhausted",
            "used": 9,
            "limit": 10,
            "percentage": 90.0,
            "remedies": ["addon"],
            "future_key": 1,
        }
    )
    assert notice.dimension == "storage_bytes" and notice.level == "exhausted"
    assert notice.plan is None and notice.help is None and notice.resets_at is None


# ============================================================================
# bootstrap
# ============================================================================


@pytest.mark.asyncio
async def test_bootstrap_sends_only_the_context_by_default():
    _, params = await _call(FULL_REPLY, "bootstrap", CTX)
    assert params == {"name": "bootstrap", "arguments": {"context_id": CTX}}


@pytest.mark.asyncio
async def test_bootstrap_sends_every_argument_that_is_set():
    _, params = await _call(
        INDEX_REPLY,
        "bootstrap",
        CTX,
        since="7d",
        include=["pinned", "changes"],
        max_chars=5_000,
        format="index",
    )
    assert params["arguments"] == {
        "context_id": CTX,
        "since": "7d",
        "include": ["pinned", "changes"],
        "max_chars": 5_000,
        "format": "index",
    }


@pytest.mark.asyncio
async def test_bootstrap_sends_a_datetime_since_as_iso():
    _, params = await _call(FULL_REPLY, "bootstrap", CTX, since=datetime(2026, 10, 28, 9, 0))
    assert params["arguments"]["since"] == "2026-10-28T09:00:00"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_chars": 1_999}, "max_chars"),
        ({"max_chars": 100_001}, "max_chars"),
        ({"max_chars": True}, "max_chars"),
        ({"max_chars": "5000"}, "max_chars"),
        ({"format": "compact"}, "format"),
        ({"format": "full", "max_chars": 9_999}, "at least 10000"),
        ({"include": []}, "include"),
        ({"include": ["pinned", "recall"]}, "include"),
    ],
)
async def test_bootstrap_refuses_what_the_server_would_before_any_request(kwargs, match):
    message = await _refused_locally("bootstrap", CTX, error=ValueError, **kwargs)
    assert match in message


@pytest.mark.asyncio
@pytest.mark.parametrize("include", ["pinned", ("pinned",), ["pinned", 1]])
async def test_bootstrap_refuses_an_include_that_is_not_a_list_of_str(include):
    """A bare ``"pinned"`` is not iterated into its characters."""
    message = await _refused_locally("bootstrap", CTX, error=TypeError, include=include)
    assert "include" in message


@pytest.mark.asyncio
async def test_bootstrap_accepts_the_index_floor_and_full_at_ten_thousand():
    _, params = await _call(INDEX_REPLY, "bootstrap", CTX, max_chars=2_000)
    assert params["arguments"]["max_chars"] == 2_000
    _, params = await _call(FULL_REPLY, "bootstrap", CTX, format="full", max_chars=10_000)
    assert params["arguments"]["format"] == "full"


@pytest.mark.asyncio
async def test_bootstrap_reads_an_index_reply():
    result, _ = await _call(INDEX_REPLY, "bootstrap", CTX, format="index", max_chars=5_000)

    assert isinstance(result, BootstrapResponse)
    assert result.format == "index" and result.degraded is False
    assert result.instructions is None
    assert result.context is not None and result.context.usage_guide is not None
    assert result.context_id == CTX and result.context_name == "dev"
    assert result.since == datetime(2026, 10, 28, 9, tzinfo=UTC)

    assert isinstance(result.guardrails, ContextGuardrails)
    (rule,) = result.guardrails.items
    assert rule.summary_cut is True and rule.authored_by_caller is None
    assert result.guardrails.provenance == _PROVENANCE

    pinned = result.components.pinned
    assert pinned is not None and pinned.status == "ok" and pinned.cap == 20
    assert pinned.memories[0]["summary"].endswith("every reply is typed.")  # pinned keeps its text

    upcoming = result.components.upcoming
    assert upcoming is not None and upcoming.from_ == "2026-11-04T09:00:00"
    assert upcoming.results[0]["until"] == "2026-11-10T23:59:59"

    changes = result.components.changes
    assert changes is not None and changes.truncated is True
    assert changes.has_more is True and changes.next_cursor
    created, forgotten = changes.changes
    assert created.summary_cut is True and created.at == datetime(2026, 11, 1, 10, tzinfo=UTC)
    assert forgotten.kind == "forgotten" and forgotten.summary is None  # #1876

    assert result.plan_headroom is not None
    warning, critical = result.plan_headroom
    assert warning.help is None and warning.resets_at is None  # fit_headroom trimmed it
    assert critical.level == "critical" and critical.help


@pytest.mark.asyncio
async def test_bootstrap_reads_a_full_reply_without_plan_headroom():
    result, _ = await _call(FULL_REPLY, "bootstrap", CTX)

    assert result.format is None
    assert result.instructions == "Kagura Memory quick reference …"
    assert result.context_summary_omitted is True
    assert result.guardrails is not None
    assert result.guardrails.items[0].source_type == "manual"
    assert result.components.pinned is not None
    assert result.components.pinned.memories[0]["delivery_mode"] == "always"
    assert result.components.changes is not None
    assert result.components.changes.changes == [] and result.components.changes.has_more is False
    assert result.plan_headroom is None  # below 80 %: the key is absent


@pytest.mark.asyncio
async def test_bootstrap_reports_a_failed_lane_and_leaves_out_unrequested_ones():
    reply = {
        **FULL_REPLY,
        "degraded": True,
        "components": {"upcoming": {"status": "error", "error": "component_failed"}},
    }
    result, _ = await _call(reply, "bootstrap", CTX, include=["upcoming"])

    assert result.degraded is True
    upcoming = result.components.upcoming
    assert upcoming is not None
    assert upcoming.status == "error" and upcoming.error == "component_failed"
    assert upcoming.results == []
    assert result.components.pinned is None and result.components.changes is None


@pytest.mark.asyncio
async def test_bootstrap_guardrails_states():
    """``null`` (read failed) and absent (``?guardrails=off``) both read as None."""
    failed, _ = await _call({**FULL_REPLY, "guardrails": None}, "bootstrap", CTX)
    assert failed.guardrails is None and "guardrails" in failed.model_fields_set

    off_reply = {k: v for k, v in FULL_REPLY.items() if k != "guardrails"}
    off, _ = await _call(off_reply, "bootstrap", CTX)
    assert off.guardrails is None and "guardrails" not in off.model_fields_set

    unreadable, _ = await _call({**FULL_REPLY, "guardrails": {"items": "x"}}, "bootstrap", CTX)
    assert unreadable.guardrails is None


@pytest.mark.asyncio
async def test_bootstrap_plan_headroom_is_advisory():
    """An entry the SDK cannot read is dropped; nothing readable reads as None."""
    broken = {k: v for k, v in _HEADROOM_CRITICAL.items() if k != "used"}
    some, _ = await _call(
        {**FULL_REPLY, "plan_headroom": [broken, _HEADROOM_WARNING_NO_HELP]}, "bootstrap", CTX
    )
    assert some.plan_headroom is not None
    assert [n.dimension for n in some.plan_headroom] == ["memories"]

    none, _ = await _call({**FULL_REPLY, "plan_headroom": [broken]}, "bootstrap", CTX)
    assert none.plan_headroom is None
    odd, _ = await _call({**FULL_REPLY, "plan_headroom": {"x": 1}}, "bootstrap", CTX)
    assert odd.plan_headroom is None


@pytest.mark.asyncio
async def test_bootstrap_keeps_a_lane_a_newer_server_adds():
    reply = {**FULL_REPLY, "components": {**FULL_REPLY["components"], "digest": {"status": "ok"}}}
    result, _ = await _call(reply, "bootstrap", CTX)
    assert result.components.model_extra == {"digest": {"status": "ok"}}


@pytest.mark.asyncio
async def test_bootstrap_drift_raises_response_error():
    reply = {**FULL_REPLY, "components": {"changes": {"status": "ok", "changes": "x"}}}
    sent: list[dict[str, Any]] = []
    client = _wire_client(reply, sent)
    try:
        with pytest.raises(KaguraResponseError) as info:
            await client.bootstrap(CTX)
    finally:
        await client.close()
    assert info.value.operation == "bootstrap"


@pytest.mark.asyncio
async def test_bootstrap_without_components_is_drift():
    reply = {k: v for k, v in FULL_REPLY.items() if k != "components"}
    sent: list[dict[str, Any]] = []
    client = _wire_client(reply, sent)
    try:
        with pytest.raises(KaguraResponseError) as info:
            await client.bootstrap(CTX)
    finally:
        await client.close()
    assert info.value.operation == "bootstrap"


@pytest.mark.asyncio
async def test_bootstrap_context_not_found_raises_not_found():
    reply = _error("context_not_found", "Context not found or you don't have access to it.")
    sent: list[dict[str, Any]] = []
    client = _wire_client(reply, sent)
    try:
        with pytest.raises(KaguraNotFoundError):
            await client.bootstrap(CTX)
    finally:
        await client.close()


# ============================================================================
# changes_since
# ============================================================================

CHANGES_REPLY: dict[str, Any] = {
    "status": "success",
    "changes": [
        {
            "memory_id": MEM_A,
            "kind": "superseded",
            "at": "2026-11-01T10:00:00Z",
            "summary": "Old deploy target",
            "superseded_by": MEM_B,
        },
        {"memory_id": MEM_C, "kind": "forgotten", "at": "2026-11-02T10:00:00Z"},
        {"memory_id": MEM_D, "kind": "restored", "at": "2026-11-03T10:00:00Z", "summary": "Back"},
    ],
    "count": 3,
    "has_more": True,
    "next_cursor": "cursor-3",
    "since": "2026-11-01T00:00:00Z",
    "until": None,
    "session_id": "mcp-0123456789abcdef",
    **_CONTEXT_FIELDS,
}


@pytest.mark.asyncio
async def test_changes_since_sends_only_context_and_since_by_default():
    _, params = await _call(CHANGES_REPLY, "changes_since", CTX, "2026-11-01T00:00:00Z")
    assert params == {
        "name": "changes_since",
        "arguments": {"context_id": CTX, "since": "2026-11-01T00:00:00Z"},
    }


@pytest.mark.asyncio
async def test_changes_since_sends_every_argument_that_is_set():
    _, params = await _call(
        CHANGES_REPLY,
        "changes_since",
        CTX,
        datetime(2026, 11, 1, tzinfo=UTC),
        until=datetime(2026, 11, 4),
        kinds=["created", "restored"],
        cursor="cursor-0",
        limit=10,
        session_id="mcp-0123456789abcdef",
        by_me=True,
        max_chars=10_000,
    )
    assert params["arguments"] == {
        "context_id": CTX,
        "since": "2026-11-01T00:00:00+00:00",
        "until": "2026-11-04T00:00:00",
        "kinds": ["created", "restored"],
        "cursor": "cursor-0",
        "limit": 10,
        "session_id": "mcp-0123456789abcdef",
        "by_me": True,
        "max_chars": 10_000,
    }


@pytest.mark.asyncio
async def test_changes_since_reads_rows_without_a_summary():
    result, _ = await _call(CHANGES_REPLY, "changes_since", CTX, "2026-11-01T00:00:00Z")
    superseded, forgotten, restored = result.changes
    assert superseded.superseded_by == MEM_B and superseded.summary == "Old deploy target"
    assert forgotten.summary is None and forgotten.superseded_by is None
    assert restored.kind == "restored"
    assert result.has_more is True and result.next_cursor == "cursor-3"
    assert result.until is None and result.session_id == "mcp-0123456789abcdef"
    assert result.by_me is False and result.truncated is False
    assert result.context_name == "dev"


@pytest.mark.asyncio
async def test_changes_since_refuses_an_out_of_range_budget_locally():
    await _refused_locally(
        "changes_since", CTX, "2026-11-01T00:00:00Z", error=ValueError, max_chars=9_999
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kinds", ["created", ("created",), ["created", None]])
async def test_changes_since_refuses_kinds_that_are_not_a_list_of_str(kinds):
    """``list("created")`` would send the characters as kinds."""
    message = await _refused_locally(
        "changes_since", CTX, "2026-11-01T00:00:00Z", error=TypeError, kinds=kinds
    )
    assert "kinds" in message


@pytest.mark.asyncio
async def test_changes_since_without_has_more_is_drift():
    reply = {k: v for k, v in CHANGES_REPLY.items() if k != "has_more"}
    sent: list[dict[str, Any]] = []
    client = _wire_client(reply, sent)
    try:
        with pytest.raises(KaguraResponseError) as info:
            await client.changes_since(CTX, "2026-11-01T00:00:00Z")
    finally:
        await client.close()
    assert info.value.operation == "changes_since"


# ============================================================================
# list (list_matching)
# ============================================================================

LIST_REPLY: dict[str, Any] = {
    "status": "success",
    "memories": [
        {
            "memory_id": MEM_A,
            "summary": "Fix flaky upload test",
            "type": "tech-debt",
            "importance": 0.6,
            "scope": "working",
            "tags": ["status:open"],
            "delivery_mode": "on_recall",
            "source_type": None,
            "created_at": "2026-11-01T10:00:00Z",
            "updated_at": None,
            "version": 1,
            "observed_at": "2026-10-31",
        }
    ],
    "count": 1,
    "total": 7,
    "has_more": True,
    "next_cursor": "1",
    "order_by": "observed_at",
    "direction": "asc",
    "truncated": True,
    "details_omitted": True,
    **_CONTEXT_FIELDS,
}


@pytest.mark.asyncio
async def test_list_matching_sends_only_the_context_by_default():
    _, params = await _call(LIST_REPLY, "list_matching", CTX)
    assert params == {"name": "list", "arguments": {"context_id": CTX}}


@pytest.mark.asyncio
async def test_list_matching_sends_every_argument_that_is_set():
    filters = {"type": ["tech-debt"], "tags": ["status:open"], "details.priority": 2}
    _, params = await _call(
        LIST_REPLY,
        "list_matching",
        CTX,
        filters=filters,
        order_by="observed_at",
        direction="asc",
        cursor="0",
        limit=1,
        include_details=True,
        max_chars=10_000,
    )
    assert params["arguments"] == {
        "context_id": CTX,
        "filters": filters,
        "order_by": "observed_at",
        "direction": "asc",
        "cursor": "0",
        "limit": 1,
        "include_details": True,
        "max_chars": 10_000,
    }


@pytest.mark.asyncio
async def test_list_matching_reads_a_page():
    result, _ = await _call(LIST_REPLY, "list_matching", CTX)
    assert result.total == 7 and result.count == 1
    assert result.has_more is True and result.next_cursor == "1"
    assert result.memories[0]["observed_at"] == "2026-10-31"
    assert result.order_by == "observed_at" and result.direction == "asc"
    assert result.truncated is True and result.details_omitted is True


@pytest.mark.asyncio
async def test_list_matching_refuses_an_out_of_range_budget_locally():
    await _refused_locally("list_matching", CTX, error=ValueError, max_chars=100_001)


@pytest.mark.asyncio
async def test_list_matching_without_total_is_drift():
    reply = {k: v for k, v in LIST_REPLY.items() if k != "total"}
    sent: list[dict[str, Any]] = []
    client = _wire_client(reply, sent)
    try:
        with pytest.raises(KaguraResponseError) as info:
            await client.list_matching(CTX)
    finally:
        await client.close()
    assert info.value.operation == "list"


# ============================================================================
# remember_batch
# ============================================================================

_ITEM = {"summary": "Batch item one is a decision", "content": "evidence", "type": "decision"}

BATCH_REPLY: dict[str, Any] = {
    "status": "partial",
    "results": [
        {
            "index": 0,
            "status": "success",
            "memory_id": MEM_A,
            "scope": "working",
            "persistence": {"scope": "working", "committed": True},
            "lint": [{"code": "no_tags", "hint": "Add tags."}],
        },
        {
            "index": 1,
            "status": "error",
            "error": "invalid_argument",
            "message": "item does not accept importnace (did you mean 'importance'?).",
        },
        {
            "index": 2,
            "status": "duplicate_candidate",
            "candidate": {"memory_id": MEM_B, "similarity": 0.97},
        },
        {
            "index": 3,
            "status": "error",
            "error": "payload_too_large",
            "message": "memory exceeds 1 MB",
            "limit": 1048576,
            "actual": 2000000,
        },
    ],
    "count": 4,
    "succeeded": 1,
    "candidates": 1,
    "failed": 2,
    "skipped": 0,
    "session_id": "mcp-0123456789abcdef",
    **_CONTEXT_FIELDS,
    "plan_headroom": [_HEADROOM_CRITICAL],
}


@pytest.mark.asyncio
async def test_remember_batch_sends_the_items_as_given_and_nothing_unset():
    items = [_ITEM, {**_ITEM, "tags": ["a"], "details": {"tool_trigger": {"tool": "Bash"}}}]
    _, params = await _call(BATCH_REPLY, "remember_batch", CTX, items)
    assert params == {"name": "remember_batch", "arguments": {"context_id": CTX, "items": items}}


@pytest.mark.asyncio
async def test_remember_batch_sends_the_batch_level_arguments_that_are_set():
    _, params = await _call(
        BATCH_REPLY,
        "remember_batch",
        CTX,
        [_ITEM],
        atomic=False,
        dedupe="check",
        tags_normalize=True,
        verbose=False,
    )
    assert params["arguments"] == {
        "context_id": CTX,
        "items": [_ITEM],
        "atomic": False,
        "dedupe": "check",
        "tags_normalize": True,
        "verbose": False,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("items", "error", "match"),
    [
        ([], ValueError, "1 to 50"),
        ([_ITEM] * 51, ValueError, "1 to 50"),
        ({"summary": "x"}, TypeError, "list"),
        ([_ITEM, "summary"], TypeError, r"items\[1\]"),
        ([{**_ITEM, "context_id": str(uuid.uuid4())}], ValueError, "another context_id"),
        ([_ITEM, {**_ITEM, "dedupe": "off"}], ValueError, r"items\[1\] carries dedupe"),
        ([{**_ITEM, "tags_normalize": True}], ValueError, "tags_normalize"),
        ([{**_ITEM, "verbose": True}], ValueError, "verbose"),
        ([_ITEM, {**_ITEM, "tags": [f"t{i}" for i in range(51)]}], ValueError, r"items\[1\]"),
        ([{**_ITEM, "tags": ["x" * 101]}], ValueError, "100 characters"),
    ],
)
async def test_remember_batch_refuses_bad_items_before_any_request(items, error, match):
    message = await _refused_locally("remember_batch", CTX, items, error=error)
    assert re.search(match, message)


@pytest.mark.asyncio
async def test_remember_batch_accepts_fifty_items_and_tags_at_the_cap():
    items = [{**_ITEM, "tags": [f"t{i}" for i in range(50)]}] * 50
    _, params = await _call(BATCH_REPLY, "remember_batch", CTX, items)
    assert len(params["arguments"]["items"]) == 50


@pytest.mark.asyncio
async def test_remember_batch_sends_an_item_that_repeats_the_batch_context_id():
    """The server accepts an item ``context_id`` equal to the batch's (``_request_from_item``)."""
    items = [{**_ITEM, "context_id": CTX}]
    _, params = await _call(BATCH_REPLY, "remember_batch", CTX, items)
    assert params["arguments"]["items"] == items


@pytest.mark.asyncio
async def test_remember_batch_leaves_tags_sent_as_a_json_string_to_the_server():
    """The server coerces ``tags`` sent as a JSON string per item; only a list is measured."""
    items = [{**_ITEM, "tags": '["a", "b"]'}]
    _, params = await _call(BATCH_REPLY, "remember_batch", CTX, items)
    assert params["arguments"]["items"] == items


@pytest.mark.asyncio
async def test_remember_batch_reads_a_partial_reply_with_plan_headroom():
    result, _ = await _call(BATCH_REPLY, "remember_batch", CTX, [_ITEM] * 4)

    assert result.status == "partial"
    assert (result.count, result.succeeded, result.candidates, result.failed, result.skipped) == (
        4,
        1,
        1,
        2,
        0,
    )
    ok, invalid, candidate, too_large = result.results
    assert ok.memory_id == MEM_A and ok.persistence == {"scope": "working", "committed": True}
    assert ok.lint == [{"code": "no_tags", "hint": "Add tags."}]
    assert invalid.error == "invalid_argument" and invalid.memory_id is None
    assert candidate.candidate == {"memory_id": MEM_B, "similarity": 0.97}
    assert too_large.model_extra == {"limit": 1048576, "actual": 2000000}
    assert result.session_id == "mcp-0123456789abcdef" and result.context_id == CTX
    assert result.plan_headroom is not None and result.plan_headroom[0].level == "critical"


@pytest.mark.asyncio
async def test_remember_batch_plan_headroom_is_advisory_too():
    """The same fail-open field type as bootstrap's: an unreadable entry is dropped."""
    broken = {k: v for k, v in _HEADROOM_CRITICAL.items() if k != "limit"}
    reply = {**BATCH_REPLY, "plan_headroom": [broken, _HEADROOM_WARNING_NO_HELP]}
    result, _ = await _call(reply, "remember_batch", CTX, [_ITEM] * 4)
    assert result.plan_headroom is not None
    assert [n.dimension for n in result.plan_headroom] == ["memories"]
    nothing, _ = await _call({**BATCH_REPLY, "plan_headroom": "x"}, "remember_batch", CTX, [_ITEM])
    assert nothing.plan_headroom is None


@pytest.mark.asyncio
async def test_remember_batch_reads_an_atomic_reply_committed_after_timeout():
    reply = {
        "status": "success",
        "results": [
            {"index": 0, "status": "success", "memory_id": MEM_A, "scope": "working"},
            {"index": 1, "status": "success", "memory_id": MEM_B, "scope": "working"},
        ],
        "count": 2,
        "succeeded": 2,
        "candidates": 0,
        "failed": 0,
        "skipped": 0,
        "committed_after_timeout": True,
        "message": "The call passed its time limit after the batch was committed: …",
        **_CONTEXT_FIELDS,
    }
    result, params = await _call(reply, "remember_batch", CTX, [_ITEM, _ITEM], atomic=True)
    assert params["arguments"]["atomic"] is True
    assert result.committed_after_timeout is True
    assert [r.memory_id for r in result.results] == [MEM_A, MEM_B]
    assert result.plan_headroom is None


@pytest.mark.asyncio
async def test_remember_batch_reads_a_pre_096_reply_without_skipped():
    """An older server sends no ``skipped`` / ``candidates``: they read as 0."""
    reply = {
        "status": "success",
        "results": [{"index": 0, "status": "success", "memory_id": MEM_A}],
        "count": 1,
        "succeeded": 1,
        "failed": 0,
    }
    result, _ = await _call(reply, "remember_batch", CTX, [_ITEM])
    assert result.skipped == 0 and result.candidates == 0


@pytest.mark.asyncio
async def test_remember_batch_with_nothing_written_raises_with_the_rows():
    reply = _error(
        "batch_failed",
        "No item was written; see results.",
        results=[{"index": 0, "status": "error", "error": "dedupe_unavailable", "message": "x"}],
        count=1,
        succeeded=0,
        candidates=0,
        failed=1,
        skipped=0,
        **_CONTEXT_FIELDS,
    )
    sent: list[dict[str, Any]] = []
    client = _wire_client(reply, sent)
    try:
        with pytest.raises(KaguraError) as info:
            await client.remember_batch(CTX, [_ITEM])
    finally:
        await client.close()
    assert info.value.code == "batch_failed"
    assert info.value.details["results"][0]["error"] == "dedupe_unavailable"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reply", "rows"),
    [
        (
            _error(
                "batch_refused",
                "1 of 2 items are invalid; atomic=true writes nothing.",
                results=[
                    {"index": 0, "status": "error", "error": "missing_fields", "message": "x"},
                    {
                        "index": 1,
                        "status": "skipped",
                        "message": "batch refused: another item is invalid",
                    },
                ],
                count=2,
                succeeded=0,
                candidates=0,
                failed=1,
                skipped=1,
            ),
            True,
        ),
        (
            _error(
                "timeout",
                "The atomic batch did not finish in time and was rolled back; nothing was "
                "written. Send fewer items, or drop atomic.",
                count=2,
                **_CONTEXT_FIELDS,
            ),
            False,
        ),
    ],
    ids=["batch_refused", "timeout"],
)
async def test_remember_batch_atomic_refusals_raise_with_their_code(reply, rows):
    """``batch_refused`` carries the rows; an atomic ``timeout`` (batch.py) carries none."""
    sent: list[dict[str, Any]] = []
    client = _wire_client(reply, sent)
    try:
        with pytest.raises(KaguraError) as info:
            await client.remember_batch(CTX, [_ITEM, _ITEM], atomic=True)
    finally:
        await client.close()
    err = info.value
    assert type(err) is KaguraError and err.code == reply["error"]
    assert ("results" in err.details) is rows
    assert err.details["count"] == 2


@pytest.mark.asyncio
async def test_remember_batch_over_quota_atomic_batch_is_a_quota_error():
    reply = _error(
        "quota_exceeded",
        "Daily memory creation limit reached.",
        gate="quota",
        quota_type="memories_per_day",
        limit=100,
        current=99,
        resets_at="2026-11-05T00:00:00Z",
        count=5,
        **_CONTEXT_FIELDS,
    )
    sent: list[dict[str, Any]] = []
    client = _wire_client(reply, sent)
    try:
        with pytest.raises(KaguraQuotaError) as info:
            await client.remember_batch(CTX, [_ITEM] * 5, atomic=True)
    finally:
        await client.close()
    assert info.value.quota_type == "memories_per_day"


@pytest.mark.asyncio
async def test_remember_batch_drift_raises_response_error():
    reply = {k: v for k, v in BATCH_REPLY.items() if k != "results"}
    sent: list[dict[str, Any]] = []
    client = _wire_client(reply, sent)
    try:
        with pytest.raises(KaguraResponseError) as info:
            await client.remember_batch(CTX, [_ITEM])
    finally:
        await client.close()
    assert info.value.operation == "remember_batch"


# ============================================================================
# forget(memory_ids=)
# ============================================================================

FORGET_BATCH_REPLY: dict[str, Any] = {
    "status": "partial",
    "deleted_count": 1,
    "memory_ids": [MEM_A],
    "results": [
        {"memory_id": MEM_A, "status": "deleted"},
        {"memory_id": MEM_B, "status": "skipped"},
        {
            "memory_id": MEM_C,
            "status": "error",
            "error": "not_attempted",
            "message": "Not attempted: the call's time budget was spent on earlier ids. Retry it "
            "in a new call.",
        },
        {
            "memory_id": MEM_D,
            "status": "error",
            "error": "timeout",
            "message": "forget timed out",
            "correlation_id": "c0ffee",
        },
    ],
    "count": 4,
    "succeeded": 1,
    "skipped": 1,
    "failed": 2,
    "session_id": "mcp-0123456789abcdef",
    "context_id": CTX,
    "context_name": "dev",
}


@pytest.mark.asyncio
async def test_forget_memory_ids_sends_only_the_ids():
    ids = [MEM_A, MEM_B, MEM_C, MEM_D]
    _, params = await _call(FORGET_BATCH_REPLY, "forget", CTX, memory_ids=ids)
    assert params == {"name": "forget", "arguments": {"context_id": CTX, "memory_ids": ids}}


@pytest.mark.asyncio
async def test_forget_memory_ids_reads_per_id_rows():
    result, _ = await _call(FORGET_BATCH_REPLY, "forget", CTX, memory_ids=[MEM_A, MEM_B])

    assert isinstance(result, ForgetBatchResult)
    assert result.status == "partial" and result.memory_ids == [MEM_A]
    assert (result.deleted_count, result.succeeded, result.skipped, result.failed) == (1, 1, 1, 2)
    deleted, skipped, unreached, timed_out = result.results
    assert deleted.status == "deleted" and skipped.status == "skipped"
    assert unreached.status == "error" and unreached.error == "not_attempted"
    assert timed_out.correlation_id == "c0ffee"
    assert result.session_id == "mcp-0123456789abcdef" and result.context_name == "dev"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"memory_ids": [MEM_A], "memory_id": MEM_B}, ValueError, "exactly one"),
        ({"memory_ids": [MEM_A], "query": "old"}, ValueError, "exactly one"),
        ({"memory_ids": [MEM_A], "k": 5}, ValueError, "k applies to query mode only"),
        ({"memory_ids": [MEM_A], "k": 10}, ValueError, "k applies to query mode only"),
        ({"memory_ids": []}, ValueError, "1 to 50"),
        ({"memory_ids": [str(uuid.uuid4()) for _ in range(51)]}, ValueError, "50"),
        ({"memory_ids": MEM_A}, TypeError, "list"),
        ({"memory_ids": [MEM_A, 7]}, TypeError, r"memory_ids\[1\]"),
        ({"memory_ids": [MEM_A, "not-a-uuid"]}, ValueError, r"memory_ids\[1\] is not"),
        ({"memory_ids": [MEM_A, MEM_A.upper()]}, ValueError, "more than once"),
    ],
)
async def test_forget_memory_ids_refuses_bad_ids_before_any_request(kwargs, error, match):
    message = await _refused_locally("forget", CTX, error=error, **kwargs)
    assert re.search(match, message)


@pytest.mark.asyncio
async def test_forget_memory_ids_accepts_fifty_ids_and_other_uuid_spellings():
    """50 is the server's cap (``FORGET_MAX_IDS``); ids go out as written."""
    ids = [str(uuid.uuid4()) for _ in range(49)] + [MEM_A.replace("-", "").upper()]
    _, params = await _call(FORGET_BATCH_REPLY, "forget", CTX, memory_ids=ids)
    assert params["arguments"]["memory_ids"] == ids


@pytest.mark.asyncio
async def test_forget_memory_ids_all_failed_raises_with_the_rows():
    reply = _error(
        "batch_failed",
        "No id was deleted; see results.",
        deleted_count=0,
        memory_ids=[],
        results=[{"memory_id": MEM_A, "status": "error", "error": "permission_denied"}],
        count=1,
        succeeded=0,
        skipped=0,
        failed=1,
        context_id=CTX,
        context_name="dev",
    )
    sent: list[dict[str, Any]] = []
    client = _wire_client(reply, sent)
    try:
        with pytest.raises(KaguraError) as info:
            await client.forget(CTX, memory_ids=[MEM_A])
    finally:
        await client.close()
    assert info.value.code == "batch_failed"
    assert info.value.details["results"][0]["error"] == "permission_denied"


@pytest.mark.asyncio
async def test_forget_memory_ids_drift_raises_response_error():
    reply = {k: v for k, v in FORGET_BATCH_REPLY.items() if k != "results"}
    sent: list[dict[str, Any]] = []
    client = _wire_client(reply, sent)
    try:
        with pytest.raises(KaguraResponseError) as info:
            await client.forget(CTX, memory_ids=[MEM_A])
    finally:
        await client.close()
    assert info.value.operation == "forget"


@pytest.mark.asyncio
async def test_forget_single_mode_still_returns_the_dict():
    reply = {"status": "success", "deleted_count": 1, "memory_ids": [MEM_A], "context_id": CTX}
    result, params = await _call(reply, "forget", CTX, memory_id=MEM_A)
    assert result == reply
    assert params["arguments"] == {"context_id": CTX, "memory_id": MEM_A}


@pytest.mark.asyncio
async def test_forget_query_mode_still_sends_k_ten_by_default():
    reply = {"status": "success", "deleted_count": 0, "memory_ids": []}
    _, params = await _call(reply, "forget", CTX, query="old data")
    assert params["arguments"] == {"context_id": CTX, "query": "old data", "k": 10}
    _, params = await _call(reply, "forget", CTX, None, "old data", 3)
    assert params["arguments"]["k"] == 3


# ============================================================================
# describe_tools / guide
# ============================================================================

_URL_HINT = {
    "list_all": "?profile=full on the MCP URL lists every tool",
    "allowlist": "?tools=<name>,<name> lists exactly those (wins over profile)",
    "add": "?add_tools=<name>,<name> lists the profile plus those (ignored next to ?tools=)",
    "note": "A profile only picks what tools/list shows; …",
}


@pytest.mark.asyncio
async def test_describe_tools_lists_hidden_tools_with_no_arguments():
    reply = {
        "status": "success",
        "tools": [{"name": "get_usage", "title": "Get usage", "summary": "Plan usage."}],
        "count": 1,
        "listed": ["bootstrap", "recall"],
        "hint": 'describe_tools(names=["<tool>"]) returns a tool\'s full schema; …',
        "url": _URL_HINT,
    }
    result, params = await _call(reply, "describe_tools")
    assert params == {"name": "describe_tools", "arguments": {}}
    assert result.tools[0]["name"] == "get_usage" and result.count == 1
    assert result.listed == ["bootstrap", "recall"] and result.url == _URL_HINT
    assert result.definitions == [] and result.url_error is None


@pytest.mark.asyncio
async def test_describe_tools_returns_definitions_for_names():
    reply = {
        "status": "success",
        "definitions": [{"name": "get_usage", "inputSchema": {"type": "object"}}],
        "url": _URL_HINT,
        "url_error": "Unknown tool profile 'typo'.",
        "unknown": ["nope"],
        "hint": "describe_tools() with no arguments lists every tool you can ask for.",
    }
    result, params = await _call(
        reply, "describe_tools", query="usage", names=["get_usage", "nope"]
    )
    assert params["arguments"] == {"query": "usage", "names": ["get_usage", "nope"]}
    assert result.definitions[0]["inputSchema"] == {"type": "object"}
    assert result.unknown == ["nope"] and result.url_error == "Unknown tool profile 'typo'."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "kwargs", "error"),
    [
        ("describe_tools", {"names": "get_usage"}, TypeError),
        ("describe_tools", {"names": ["a", 1]}, TypeError),
        ("describe_tools", {"names": [f"t{i}" for i in range(21)]}, ValueError),
        ("guide", {"topics": "recall"}, TypeError),
        ("guide", {"topics": [None]}, TypeError),
        ("guide", {"topics": [f"t{i}" for i in range(21)]}, ValueError),
    ],
)
async def test_name_lists_are_checked_before_any_request(method, kwargs, error):
    await _refused_locally(method, error=error, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize(("method", "key"), [("describe_tools", "names"), ("guide", "topics")])
async def test_name_lists_take_twenty_names(method, key):
    """20 is the server's cap (``MAX_NAMES`` / ``MAX_TOPICS``)."""
    names = [f"t{i}" for i in range(20)]
    _, params = await _call({"status": "success", "topics": []}, method, **{key: names})
    assert params["arguments"] == {key: names}


@pytest.mark.asyncio
async def test_guide_sends_topics_only_when_set_and_reads_sections():
    reply = {
        "status": "success",
        "topics": [{"topic": "index", "text": "Topics: …"}],
        "unknown": ["recal"],
        "hint": 'guide(["index"]) lists every topic.',
    }
    result, params = await _call(reply, "guide")
    assert params == {"name": "guide", "arguments": {}}
    assert result.topics[0].topic == "index" and result.unknown == ["recal"]

    _, params = await _call(reply, "guide", ["recall", "recal"])
    assert params["arguments"] == {"topics": ["recall", "recal"]}


@pytest.mark.asyncio
async def test_guide_drift_raises_response_error():
    sent: list[dict[str, Any]] = []
    client = _wire_client({"status": "success", "topics": "index"}, sent)
    try:
        with pytest.raises(KaguraResponseError) as info:
            await client.guide()
    finally:
        await client.close()
    assert info.value.operation == "guide"


# ============================================================================
# list_revisions
# ============================================================================

REVISIONS_REPLY: dict[str, Any] = {
    "status": "success",
    "memory_id": MEM_A,
    "current_version": 3,
    "revisions": [
        {
            "version": 3,
            "kind": "updated",
            "at": "2026-11-03T10:00:00Z",
            "changed_fields": ["summary", "tags"],
            "writer": "api",
            "by_you": True,
            "session_id": "mcp-0123456789abcdef",
        },
        {
            "version": 2,
            "kind": "updated",
            "at": "2026-11-02T10:00:00Z",
            "changed_fields": [],
            "writer": "erasure",
            "by_you": False,
        },
    ],
    "count": 2,
    "has_more": True,
    "next_cursor": "41",
    **_CONTEXT_FIELDS,
}


@pytest.mark.asyncio
async def test_list_revisions_sends_set_arguments_and_reads_the_page():
    result, params = await _call(REVISIONS_REPLY, "list_revisions", CTX, MEM_A)
    assert params == {
        "name": "list_revisions",
        "arguments": {"context_id": CTX, "memory_id": MEM_A},
    }
    assert result.current_version == 3 and result.has_more is True
    mine, erased = result.revisions
    assert mine.changed_fields == ["summary", "tags"] and mine.by_you is True
    assert mine.session_id == "mcp-0123456789abcdef"
    assert erased.writer == "erasure" and erased.agent_id is None

    _, params = await _call(
        REVISIONS_REPLY, "list_revisions", CTX, MEM_A, limit=5, cursor="41", max_chars=10_000
    )
    assert params["arguments"] == {
        "context_id": CTX,
        "memory_id": MEM_A,
        "limit": 5,
        "cursor": "41",
        "max_chars": 10_000,
    }


@pytest.mark.asyncio
async def test_list_revisions_of_a_forgotten_memory_is_not_found():
    reply = _error("memory_not_found", f"Memory not found or you don't have access: {MEM_A}")
    sent: list[dict[str, Any]] = []
    client = _wire_client(reply, sent)
    try:
        with pytest.raises(KaguraNotFoundError):
            await client.list_revisions(CTX, MEM_A)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_list_revisions_checks_max_chars_and_drift():
    await _refused_locally("list_revisions", CTX, MEM_A, error=ValueError, max_chars=5)
    reply = {k: v for k, v in REVISIONS_REPLY.items() if k != "current_version"}
    sent: list[dict[str, Any]] = []
    client = _wire_client(reply, sent)
    try:
        with pytest.raises(KaguraResponseError) as info:
            await client.list_revisions(CTX, MEM_A)
    finally:
        await client.close()
    assert info.value.operation == "list_revisions"


# ============================================================================
# rollback_session
# ============================================================================

DRY_RUN_REPLY: dict[str, Any] = {
    "status": "dry_run",
    "dry_run": True,
    "plan": {"forget": 1, "restore": 1, "unforget": 0, "skip": 1},
    "items": [
        {"memory_id": MEM_A, "kind": "created", "action": "forget"},
        {
            "memory_id": MEM_B,
            "kind": "updated",
            "action": "restore",
            "fields": ["summary"],
        },
        {"memory_id": MEM_C, "kind": "updated", "action": "skip", "cause": "not_permitted"},
    ],
    "session_id": "mcp-0123456789abcdef",
    "events": 4,
    "remaining": 0,
    **_CONTEXT_FIELDS,
}

_SUMMARY = {
    "forgotten": 1,
    "restored": 1,
    "updated_restored": 1,
    "skipped": 1,
    "failed": 0,
    "chunks_requeued": 0,
    "points_requeued": 0,
    "skipped_items": [{"memory_id": MEM_C, "kind": "updated", "cause": "not_permitted"}],
    "errors": [],
}


@pytest.mark.asyncio
async def test_rollback_session_always_sends_dry_run():
    result, params = await _call(
        DRY_RUN_REPLY, "rollback_session", CTX, session_id="mcp-0123456789abcdef"
    )
    assert params == {
        "name": "rollback_session",
        "arguments": {"context_id": CTX, "dry_run": True, "session_id": "mcp-0123456789abcdef"},
    }
    assert result.status == "dry_run" and result.dry_run is True
    assert result.plan == {"forget": 1, "restore": 1, "unforget": 0, "skip": 1}
    forget_row, restore_row, skip_row = result.items
    assert forget_row.action == "forget" and restore_row.fields == ["summary"]
    assert skip_row.cause == "not_permitted"
    assert result.summary is None and result.items_truncated is False


@pytest.mark.asyncio
async def test_rollback_session_window_mode_reads_a_run():
    reply = {
        "status": "in_progress",
        "dry_run": False,
        "rollback_status": "partial",
        "summary": _SUMMARY,
        "totals": {**{k: v for k, v in _SUMMARY.items() if isinstance(v, int)}, "calls": 1},
        "since": "2026-11-03T00:00:00Z",
        "until": "2026-11-04T00:00:00Z",
        "until_from": "unfinished_rollback",
        "events": 500,
        "remaining": 12,
        "rollback_id": 9,
        "help": "Call again with the same since and until to continue.",
        **_CONTEXT_FIELDS,
    }
    result, params = await _call(
        reply,
        "rollback_session",
        CTX,
        since=datetime(2026, 11, 3),
        until="2026-11-04T00:00:00Z",
        dry_run=False,
    )
    assert params["arguments"] == {
        "context_id": CTX,
        "dry_run": False,
        "since": "2026-11-03T00:00:00",
        "until": "2026-11-04T00:00:00Z",
    }
    assert result.status == "in_progress" and result.remaining == 12
    assert result.since == datetime(2026, 11, 3, tzinfo=UTC)
    assert result.until_from == "unfinished_rollback" and result.rollback_id == 9
    assert result.summary is not None
    assert result.summary.skipped_items[0]["cause"] == "not_permitted"
    assert result.totals is not None and result.totals["calls"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({}, "exactly one"),
        ({"session_id": "s", "since": "2026-11-03T00:00:00Z"}, "exactly one"),
        ({"session_id": "s", "until": "2026-11-04T00:00:00Z"}, "until needs since"),
    ],
)
async def test_rollback_session_needs_one_mode(kwargs, match):
    message = await _refused_locally("rollback_session", CTX, error=ValueError, **kwargs)
    assert match in message


@pytest.mark.asyncio
async def test_rollback_session_partial_raises_with_the_session_summary():
    reply = _error(
        "partial_rollback",
        "Rollback finished this call with 1 failed item(s); summary shows what was reversed.",
        help="Every item not listed in summary.errors was handled and saved. …",
        retryable=True,
        rollback_status="partial",
        summary={**_SUMMARY, "failed": 1, "errors": [f"Memory {MEM_D} (forget) failed: x"]},
        totals={"forgotten": 1, "failed": 1, "calls": 1},
        correlation_id="c0ffee",
        session_id="mcp-0123456789abcdef",
        events=4,
        remaining=1,
        rollback_id=9,
        **_CONTEXT_FIELDS,
    )
    sent: list[dict[str, Any]] = []
    client = _wire_client(reply, sent)
    try:
        with pytest.raises(KaguraPartialSessionRollbackError) as info:
            await client.rollback_session(CTX, session_id="mcp-0123456789abcdef", dry_run=False)
    finally:
        await client.close()
    err = info.value
    assert not isinstance(err, KaguraPartialRollbackError)  # the Sleep run's sibling
    assert err.code == "partial_rollback" and err.retryable is True
    assert err.session_summary is not None
    assert err.session_summary.failed == 1 and err.session_summary.forgotten == 1
    assert err.session_summary.errors == [f"Memory {MEM_D} (forget) failed: x"]
    assert err.totals == {"forgotten": 1, "failed": 1, "calls": 1}
    assert (err.rollback_id, err.remaining) == (9, 1)
    assert err.correlation_id == "c0ffee" and err.__cause__ is None
    assert err.details["rollback_status"] == "partial"


@pytest.mark.asyncio
async def test_rollback_session_partial_with_unreadable_fields_still_raises_typed():
    """The rollback is committed: a field the SDK cannot read is None, not a second error."""
    reply = _error(
        "partial_rollback",
        "Rollback finished this call with 1 failed item(s); summary shows what was reversed.",
        summary={"failed": "one"},
        totals="x",
        rollback_id=True,
        **_CONTEXT_FIELDS,
    )
    sent: list[dict[str, Any]] = []
    client = _wire_client(reply, sent)
    try:
        with pytest.raises(KaguraPartialSessionRollbackError) as info:
            await client.rollback_session(CTX, session_id="s", dry_run=False)
    finally:
        await client.close()
    err = info.value
    assert err.session_summary is None and err.totals is None
    assert err.rollback_id is None and err.remaining is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("code", "error"),
    [
        ("session_not_found", KaguraNotFoundError),
        ("no_writes_in_window", KaguraError),
        ("already_rolled_back", KaguraError),
    ],
)
async def test_rollback_session_refusals(code, error):
    sent: list[dict[str, Any]] = []
    client = _wire_client(_error(code, "refused", retryable=False), sent)
    try:
        with pytest.raises(error) as info:
            await client.rollback_session(CTX, since="2026-11-03T00:00:00Z")
    finally:
        await client.close()
    assert info.value.code == code


@pytest.mark.asyncio
async def test_rollback_session_drift_raises_response_error():
    reply = {k: v for k, v in DRY_RUN_REPLY.items() if k != "dry_run"}
    sent: list[dict[str, Any]] = []
    client = _wire_client(reply, sent)
    try:
        with pytest.raises(KaguraResponseError) as info:
            await client.rollback_session(CTX, session_id="s")
    finally:
        await client.close()
    assert info.value.operation == "rollback_session"
