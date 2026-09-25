"""Tests for tool guardrails (#253, memory-cloud v0.74.0+ #1619/#1621).

Wire shapes assert the memory-cloud contract verified against the server
source (``mcp_server/tools/memory.py::handle_load_guardrails``,
``api/routes/memory.py`` ``/guardrails`` + ``/guardrails/digest``,
``services/guardrail_digest.py::render_context_info_block`` and
``utils/tool_trigger.py``): the two independently capped lanes of
``load_guardrails``, the ``get_context_info.guardrails`` block, the digest's
``X-Kagura-Guardrails-Tool-Triggered-Version`` header, and the reserved
``details.tool_trigger`` write key.
"""

import json
import os
import stat
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from click.testing import CliRunner

from kagura_memory import (
    GUARDRAIL_FORMAT,
    ContextGuardrails,
    ContextInfo,
    GuardrailDigest,
    GuardrailItem,
    GuardrailSet,
    KaguraClient,
    KaguraError,
    KaguraNotFoundError,
    KaguraResponseError,
    MemoryClient,
    ToolTrigger,
)
from kagura_memory._guardrail_export import splice_guardrail_block, write_guardrail_block
from kagura_memory.auth.credentials import reset_state_cache
from kagura_memory.cli import main

CTX = "11111111-2222-3333-4444-555555555555"
MEM_A = "aaaaaaaa-0000-0000-0000-000000000001"
MEM_B = "bbbbbbbb-0000-0000-0000-000000000002"
VERSION = "3f9c1a7b2d4e6f80"


def _item(memory_id: str, *, tool_trigger: dict | None = None, pinned: bool = False) -> dict:
    """One server-shaped ``GuardrailItem`` (``_guardrail_item_payload``)."""
    return {
        "memory_id": memory_id,
        "summary": "Squash-merge only after the head SHA matches",
        "context_summary": "why this exists" if pinned else None,
        "type": "decision",
        "importance": 0.9,
        "delivery_mode": "always" if pinned else "on_recall",
        "tool_trigger": tool_trigger,
        "source_type": "manual",
        "authored_by_caller": True,
        "created_at": "2026-09-01T00:00:00Z",
        "updated_at": "2026-09-02T00:00:00Z",
    }


TRIGGER = {"tool": "Bash|PowerShell", "on": "pre", "match": "gh pr merge", "action": "inform"}


def guardrail_set_dict(**overrides) -> dict:
    """A server-shaped ``load_guardrails`` envelope with BOTH lanes truncated."""
    payload = {
        "status": "success",
        "format": 1,
        "version": VERSION,
        "pinned": [_item(MEM_A, pinned=True)],
        "tool_triggered": [_item(MEM_B, tool_trigger=TRIGGER)],
        "total_available": 7,
        "truncated": True,
        "cap": 1,
        "pinned_cap": 1,
        "pinned_total_available": 3,
        "pinned_truncated": True,
        "tool_triggered_total_available": 4,
        "tool_triggered_truncated": True,
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def test_guardrail_set_parses_both_lanes_and_truncation_flags():
    result = GuardrailSet.model_validate(guardrail_set_dict())

    assert result.format == 1
    assert result.version == VERSION
    assert [i.memory_id for i in result.pinned] == [MEM_A]
    assert [i.memory_id for i in result.tool_triggered] == [MEM_B]
    # Per-lane flags say which protection is incomplete; the top-level pair is
    # the either-lane summary.
    assert result.truncated is True
    assert result.pinned_truncated is True
    assert result.tool_triggered_truncated is True
    assert result.pinned_total_available == 3
    assert result.tool_triggered_total_available == 4
    assert result.total_available == 7
    assert result.cap == 1
    assert result.pinned_cap == 1


def test_guardrail_set_only_tool_triggered_lane_truncated():
    result = GuardrailSet.model_validate(
        guardrail_set_dict(
            truncated=True,
            pinned_truncated=False,
            tool_triggered_truncated=True,
        )
    )
    assert result.pinned_truncated is False
    assert result.tool_triggered_truncated is True


def test_guardrail_set_mcp_context_fields_and_unknown_keys():
    # The MCP envelope adds the _context_response_fields block; REST omits it.
    payload = guardrail_set_dict(
        context_id=CTX,
        context_name="dev",
        context_is_locked=False,
        some_future_field={"x": 1},
    )
    result = GuardrailSet.model_validate(payload)
    assert result.context_id == CTX
    assert result.context_name == "dev"
    rest = GuardrailSet.model_validate(guardrail_set_dict())
    assert rest.context_id is None


def test_guardrail_set_missing_truncation_flag_is_an_error():
    # A hook must never read a silently-complete set: the flags are required.
    payload = guardrail_set_dict()
    del payload["tool_triggered_truncated"]
    with pytest.raises(ValueError):
        GuardrailSet.model_validate(payload)


def test_guardrail_item_typed_trigger_and_provenance():
    item = GuardrailItem.model_validate(_item(MEM_B, tool_trigger=TRIGGER))
    assert isinstance(item.tool_trigger, ToolTrigger)
    assert item.tool_trigger.tool == "Bash|PowerShell"
    assert item.tool_trigger.match == "gh pr merge"
    assert item.authored_by_caller is True
    assert item.context_summary is None
    assert item.created_at is not None


def test_guardrail_item_keeps_unknown_on_and_action_values():
    # Forward compatibility: a future server may add an ``on`` / ``action``
    # value; the contract says consumers SKIP such items, never fail the read.
    trigger = {"tool": "Bash", "on": "post", "action": "warn", "future_key": 1}
    item = GuardrailItem.model_validate(_item(MEM_B, tool_trigger=trigger))
    assert item.tool_trigger is not None
    assert item.tool_trigger.on == "post"
    assert item.tool_trigger.action == "warn"
    # Unknown keys are kept (and ignored by a matcher), not dropped.
    assert item.tool_trigger.model_extra == {"future_key": 1}


@pytest.mark.parametrize(
    "legacy",
    [{"foo": 1}, {"tool": 5}, "Bash", ["Bash"]],
    ids=["no-tool", "non-string-tool", "string", "list"],
)
def test_guardrail_item_unusable_trigger_is_none_not_a_failed_read(legacy):
    item = GuardrailItem.model_validate(_item(MEM_B, tool_trigger=legacy))
    assert item.tool_trigger is None
    assert item.memory_id == MEM_B


def test_guardrail_item_absent_authored_by_caller_is_unknown():
    payload = _item(MEM_A)
    del payload["authored_by_caller"]
    assert GuardrailItem.model_validate(payload).authored_by_caller is None


def test_tool_trigger_defaults_match_server_normalization():
    trigger = ToolTrigger(tool="Edit|Write")
    assert trigger.on == "pre"
    assert trigger.action == "inform"
    assert trigger.match is None


def test_guardrail_format_mirrors_server():
    # memory-cloud utils/tool_trigger.py GUARDRAIL_FORMAT; a greater format on
    # the wire means "treat the set as absent" (fail-open).
    assert GUARDRAIL_FORMAT == 1
    assert GuardrailSet.model_validate(guardrail_set_dict()).format == GUARDRAIL_FORMAT


# ---------------------------------------------------------------------------
# ContextInfo.guardrails (get_context_info block, #1621)
# ---------------------------------------------------------------------------


def _context_info_dict(**extra) -> dict:
    return {
        "status": "success",
        "context": {"id": CTX, "name": "dev"},
        **extra,
    }


GUARDRAILS_BLOCK = {
    "items": [
        {
            "memory_id": MEM_B,
            "summary": "gh pr merge --delete-branch closes the child PR",
            "importance": 0.8,
            "authored_by_caller": False,
            "source_type": "manual",
        }
    ],
    "total_available": 12,
    "truncated": True,
    "tool_triggered_version": VERSION,
}


def test_context_info_keeps_guardrails_block():
    info = ContextInfo.model_validate(_context_info_dict(guardrails=GUARDRAILS_BLOCK))
    assert isinstance(info.guardrails, ContextGuardrails)
    assert info.guardrails.total_available == 12
    assert info.guardrails.truncated is True
    assert info.guardrails.tool_triggered_version == VERSION
    [entry] = info.guardrails.items
    assert entry.memory_id == MEM_B
    assert entry.importance == 0.8
    assert entry.authored_by_caller is False
    assert entry.source_type == "manual"


def test_context_info_guardrails_absent_vs_null():
    # Absent: the endpoint URL carries ?guardrails=off. Null: the read failed or
    # no context resolved. model_fields_set tells the two apart.
    absent = ContextInfo.model_validate(_context_info_dict())
    null = ContextInfo.model_validate(_context_info_dict(guardrails=None))
    assert absent.guardrails is None and "guardrails" not in absent.model_fields_set
    assert null.guardrails is None and "guardrails" in null.model_fields_set


def test_context_info_guardrails_ignores_unknown_keys():
    block = {**GUARDRAILS_BLOCK, "future": True}
    block["items"] = [{**GUARDRAILS_BLOCK["items"][0], "tags": ["x"]}]
    info = ContextInfo.model_validate(_context_info_dict(guardrails=block))
    assert info.guardrails is not None
    assert info.guardrails.items[0].summary.startswith("gh pr merge")


@pytest.mark.parametrize("missing", ["items", "total_available", "truncated"])
def test_context_info_guardrails_block_without_counts_is_none_not_complete(missing):
    # A block missing its counts must never read as a complete (untruncated) set.
    block = {k: v for k, v in GUARDRAILS_BLOCK.items() if k != missing}
    info = ContextInfo.model_validate(_context_info_dict(guardrails=block))
    assert info.guardrails is None
    assert info.context.id == CTX


def test_context_info_malformed_guardrails_block_degrades_to_none():
    # The block is fail-open on the server; a shape the SDK cannot read must not
    # fail the whole get_context_info call (ingest steering depends on it).
    info = ContextInfo.model_validate(_context_info_dict(guardrails="oops"))
    assert info.guardrails is None
    assert info.context.id == CTX


# ---------------------------------------------------------------------------
# KaguraClient (MCP)
# ---------------------------------------------------------------------------


def _mcp_client() -> KaguraClient:
    client = KaguraClient(api_key="test", mcp_url="https://test.com/mcp")
    client._session_id = "pre-set-session"
    return client


@pytest.mark.asyncio
async def test_load_guardrails_mcp_minimal():
    client = _mcp_client()
    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = guardrail_set_dict(context_id=CTX)
        result = await client.load_guardrails(CTX)
        name, args = mock.call_args[0]
    await client.close()

    assert name == "load_guardrails"
    assert args == {"context_id": CTX}  # cap omitted → server default (50)
    assert isinstance(result, GuardrailSet)
    assert result.pinned_truncated is True
    assert result.tool_triggered_truncated is True
    assert result.tool_triggered[0].tool_trigger is not None


@pytest.mark.asyncio
async def test_load_guardrails_mcp_with_cap():
    client = _mcp_client()
    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = guardrail_set_dict()
        await client.load_guardrails(CTX, cap=10)
        args = mock.call_args[0][1]
    await client.close()
    assert args == {"context_id": CTX, "cap": 10}


@pytest.mark.asyncio
async def test_load_guardrails_mcp_missing_flag_is_a_response_error():
    # Drift never reads as a complete set, and never escapes as a raw
    # pydantic ValidationError (#250).
    payload = guardrail_set_dict()
    del payload["pinned_truncated"]
    client = _mcp_client()
    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = payload
        with pytest.raises(KaguraResponseError) as exc:
            await client.load_guardrails(CTX)
    await client.close()
    assert exc.value.operation == "load_guardrails"


@pytest.mark.asyncio
async def test_load_guardrails_mcp_context_not_found():
    client = _mcp_client()
    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {
            "status": "error",
            "error": "context_not_found",
            "message": "Context not found or you don't have access to it.",
        }
        with pytest.raises(KaguraNotFoundError):
            await client.load_guardrails(CTX)
    await client.close()


@pytest.mark.asyncio
async def test_get_context_info_exposes_guardrails():
    client = _mcp_client()
    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = _context_info_dict(guardrails=GUARDRAILS_BLOCK)
        info = await client.get_context_info(CTX)
    await client.close()
    assert info.guardrails is not None
    assert info.guardrails.tool_triggered_version == VERSION


# ---- remember / update_memory tool_trigger= --------------------------------


@pytest.mark.asyncio
async def test_remember_tool_trigger_model_sends_details_tool_trigger():
    client = _mcp_client()
    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"memory_id": "m1"}
        await client.remember(
            context_id=CTX,
            summary="Never force-push to main",
            content="...",
            tool_trigger=ToolTrigger(tool="Bash", match=r"git\s+push\s+--force", action="block"),
        )
        args = mock.call_args[0][1]
    await client.close()
    assert args["details"] == {
        "tool_trigger": {
            "tool": "Bash",
            "on": "pre",
            "match": r"git\s+push\s+--force",
            "action": "block",
        }
    }


@pytest.mark.asyncio
async def test_remember_tool_trigger_omits_unset_match():
    # The server rejects "match": null (match_not_string) — never send it.
    client = _mcp_client()
    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"memory_id": "m1"}
        await client.remember(
            context_id=CTX, summary="s" * 10, content="c", tool_trigger=ToolTrigger(tool="Edit")
        )
        trigger = mock.call_args[0][1]["details"]["tool_trigger"]
    await client.close()
    assert "match" not in trigger
    assert trigger == {"tool": "Edit", "on": "pre", "action": "inform"}


@pytest.mark.parametrize(
    "trigger",
    [
        ToolTrigger.model_validate({"tool": "Bash", "matches": r"rm\s+-rf", "action": "block"}),
        ToolTrigger(tool="Bash", action="block", matches=r"rm\s+-rf"),  # type: ignore[call-arg]
    ],
    ids=["model_validate", "constructor"],
)
@pytest.mark.asyncio
async def test_remember_tool_trigger_forwards_misspelt_key(trigger):
    # A typo must reach the server (tool_trigger_unknown_key), never be dropped
    # into an unscoped guardrail that fires on every Bash call.
    client = _mcp_client()
    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"memory_id": "m1"}
        await client.remember(context_id=CTX, summary="s" * 10, content="c", tool_trigger=trigger)
        sent = mock.call_args[0][1]["details"]["tool_trigger"]
    await client.close()
    assert sent == {"tool": "Bash", "on": "pre", "action": "block", "matches": r"rm\s+-rf"}


@pytest.mark.asyncio
async def test_remember_tool_trigger_dict_merges_with_other_details():
    client = _mcp_client()
    details = {"location": {"lat": 35.68, "lon": 139.76}}
    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"memory_id": "m1"}
        await client.remember(
            context_id=CTX,
            summary="s" * 10,
            content="c",
            details=details,
            tool_trigger={"tool": "Bash", "match": "rm -rf"},
        )
        args = mock.call_args[0][1]
    await client.close()
    # A dict is forwarded verbatim — the server validates and normalizes it.
    assert args["details"] == {
        "location": {"lat": 35.68, "lon": 139.76},
        "tool_trigger": {"tool": "Bash", "match": "rm -rf"},
    }
    assert details == {"location": {"lat": 35.68, "lon": 139.76}}  # caller's dict untouched


@pytest.mark.asyncio
async def test_remember_tool_trigger_conflicts_with_details_key():
    client = _mcp_client()
    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        with pytest.raises(ValueError, match="tool_trigger"):
            await client.remember(
                context_id=CTX,
                summary="s" * 10,
                content="c",
                details={"tool_trigger": {"tool": "Bash"}},
                tool_trigger=ToolTrigger(tool="Edit"),
            )
        mock.assert_not_called()
    await client.close()


@pytest.mark.asyncio
async def test_remember_details_tool_trigger_alone_passes_through():
    client = _mcp_client()
    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"memory_id": "m1"}
        await client.remember(
            context_id=CTX, summary="s" * 10, content="c", details={"tool_trigger": {"tool": "X"}}
        )
        args = mock.call_args[0][1]
    await client.close()
    assert args["details"] == {"tool_trigger": {"tool": "X"}}


@pytest.mark.asyncio
async def test_remember_guardrail_permission_denied_surfaces_code():
    client = _mcp_client()
    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {
            "status": "error",
            "error": "permission_denied",
            "message": (
                "Cannot mark a memory as a tool guardrail: tool guardrails require "
                "context editor or above."
            ),
            "required_role": "editor",
        }
        with pytest.raises(KaguraError, match="permission_denied"):
            await client.remember(
                context_id=CTX, summary="s" * 10, content="c", tool_trigger={"tool": "Bash"}
            )
    await client.close()


@pytest.mark.asyncio
async def test_remember_invalid_trigger_surfaces_stable_code():
    client = _mcp_client()
    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {
            "status": "error",
            "error": "validation_error",
            "message": "invalid details.tool_trigger: block_requires_match: action='block' ...",
        }
        with pytest.raises(KaguraError, match="block_requires_match"):
            await client.remember(
                context_id=CTX,
                summary="s" * 10,
                content="c",
                tool_trigger=ToolTrigger(tool="Bash", action="block"),
            )
    await client.close()


@pytest.mark.parametrize(
    "id_kwargs",
    [{"memory_id": MEM_A}, {"external_id": "ext-1", "summary": "s" * 10, "content": "c"}],
    ids=["in-place", "upsert"],
)
@pytest.mark.asyncio
async def test_update_memory_tool_trigger_sends_details(id_kwargs):
    client = _mcp_client()
    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"status": "success"}
        await client.update_memory(
            context_id=CTX, tool_trigger=ToolTrigger(tool="Bash", match="gh pr merge"), **id_kwargs
        )
        args = mock.call_args[0][1]
    await client.close()
    assert args["details"] == {
        "tool_trigger": {"tool": "Bash", "on": "pre", "match": "gh pr merge", "action": "inform"}
    }


@pytest.mark.asyncio
async def test_update_memory_tool_trigger_merges_with_details():
    client = _mcp_client()
    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"status": "success"}
        await client.update_memory(
            context_id=CTX,
            memory_id=MEM_A,
            details={"note": "keep"},
            tool_trigger={"tool": "Edit|Write"},
        )
        args = mock.call_args[0][1]
    await client.close()
    assert args["details"] == {"note": "keep", "tool_trigger": {"tool": "Edit|Write"}}


@pytest.mark.asyncio
async def test_update_memory_tool_trigger_conflicts_with_details_key():
    client = _mcp_client()
    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        with pytest.raises(ValueError, match="tool_trigger"):
            await client.update_memory(
                context_id=CTX,
                memory_id=MEM_A,
                details={"tool_trigger": None},
                tool_trigger={"tool": "Edit"},
            )
        mock.assert_not_called()
    await client.close()


@pytest.mark.asyncio
async def test_update_memory_without_tool_trigger_leaves_details_alone():
    client = _mcp_client()
    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
        mock.return_value = {"status": "success"}
        await client.update_memory(context_id=CTX, memory_id=MEM_A, summary="s" * 10)
        args = mock.call_args[0][1]
    await client.close()
    assert "details" not in args


# ---------------------------------------------------------------------------
# MemoryClient (REST twin)
# ---------------------------------------------------------------------------


def make_rest_client(handler) -> MemoryClient:
    client = MemoryClient(api_key="kagura_test")
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer kagura_test"},
    )
    return client


@pytest.mark.asyncio
async def test_rest_load_guardrails_minimal():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=guardrail_set_dict())

    async with make_rest_client(handler) as client:
        result = await client.load_guardrails(CTX)

    assert seen["method"] == "POST"
    assert seen["path"] == "/api/v1/memory/guardrails"
    assert seen["body"] == {"context_id": CTX}
    assert isinstance(result, GuardrailSet)
    assert result.pinned_truncated is True
    assert result.tool_triggered_truncated is True
    assert result.context_id is None  # REST carries no context block


@pytest.mark.asyncio
async def test_rest_load_guardrails_cap_and_uuid_normalization():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=guardrail_set_dict(truncated=False))

    async with make_rest_client(handler) as client:
        await client.load_guardrails(CTX.upper(), cap=25)

    assert seen["body"] == {"context_id": CTX, "cap": 25}


@pytest.mark.asyncio
async def test_rest_load_guardrails_rejects_non_uuid_before_request():
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("no request expected")

    async with make_rest_client(handler) as client:
        with pytest.raises(ValueError, match="context_id"):
            await client.load_guardrails("not-a-uuid")


@pytest.mark.asyncio
async def test_rest_load_guardrails_missing_flag_is_a_response_error():
    payload = guardrail_set_dict()
    del payload["tool_triggered_truncated"]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    async with make_rest_client(handler) as client:
        with pytest.raises(KaguraResponseError) as exc:
            await client.load_guardrails(CTX)
    assert exc.value.operation == "MemoryClient.load_guardrails"


@pytest.mark.asyncio
async def test_rest_load_guardrails_uniform_404():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "Context not found"})

    async with make_rest_client(handler) as client:
        with pytest.raises(KaguraNotFoundError):
            await client.load_guardrails(CTX)


EXPORT_BLOCK = (
    f"<!-- kagura-memory:guardrails begin context={CTX} tool_triggered_version={VERSION} -->\n"
    "- (bbbbbbbb) gh pr merge --delete-branch closes the child PR\n"
    "<!-- kagura-memory:guardrails end -->\n"
)


@pytest.mark.asyncio
async def test_rest_guardrail_digest_returns_text_and_version_header():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["params"] = dict(request.url.params)
        return httpx.Response(
            200,
            text=EXPORT_BLOCK,
            headers={
                "Content-Type": "text/markdown; charset=utf-8",
                "X-Kagura-Guardrails-Tool-Triggered-Version": VERSION,
            },
        )

    async with make_rest_client(handler) as client:
        digest = await client.get_guardrail_digest(CTX)

    assert seen["method"] == "GET"
    assert seen["path"] == "/api/v1/memory/guardrails/digest"
    assert seen["params"] == {"context_id": CTX, "target": "export"}
    assert isinstance(digest, GuardrailDigest)
    assert digest.text == EXPORT_BLOCK
    assert digest.tool_triggered_version == VERSION
    assert digest.context_id == CTX
    assert digest.target == "export"
    assert digest.content_type is not None and digest.content_type.startswith("text/markdown")


@pytest.mark.asyncio
async def test_rest_guardrail_digest_instructions_target_forwards_tool_view():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["params"] = dict(request.url.params)
        return httpx.Response(
            200,
            text="base text",
            headers={
                "Content-Type": "text/plain; charset=utf-8",
                "X-Kagura-Guardrails-Tool-Triggered-Version": VERSION,
            },
        )

    async with make_rest_client(handler) as client:
        digest = await client.get_guardrail_digest(
            CTX, target="instructions", profile="core", tools="remember,recall"
        )

    assert seen["params"] == {
        "context_id": CTX,
        "target": "instructions",
        "profile": "core",
        "tools": "remember,recall",
    }
    assert digest.text == "base text"


@pytest.mark.asyncio
async def test_rest_guardrail_digest_empty_set():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text="",
            headers={"X-Kagura-Guardrails-Tool-Triggered-Version": "4f53cda18c2baa0c"},
        )

    async with make_rest_client(handler) as client:
        digest = await client.get_guardrail_digest(CTX)

    assert digest.text == ""
    assert digest.tool_triggered_version == "4f53cda18c2baa0c"


@pytest.mark.asyncio
async def test_rest_guardrail_digest_missing_header_is_none():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="")

    async with make_rest_client(handler) as client:
        digest = await client.get_guardrail_digest(CTX)

    assert digest.tool_triggered_version is None


@pytest.mark.asyncio
async def test_rest_guardrail_digest_uniform_404():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "Context not found"})

    async with make_rest_client(handler) as client:
        with pytest.raises(KaguraNotFoundError):
            await client.get_guardrail_digest(CTX)


# ---------------------------------------------------------------------------
# AGENTS.md block splice (mirrors the server's Codex cloud recipe)
# ---------------------------------------------------------------------------


NEW_BLOCK = EXPORT_BLOCK.replace(VERSION, "0123456789abcdef").replace(
    "closes the child PR", "closes the stacked child PR"
)


def test_splice_appends_to_file_without_block():
    text = "# Project\n\nRules.\n"
    assert splice_guardrail_block(text, EXPORT_BLOCK) == text + "\n" + EXPORT_BLOCK


def test_splice_adds_missing_trailing_newline_before_block():
    assert splice_guardrail_block("# P", EXPORT_BLOCK) == "# P\n\n" + EXPORT_BLOCK


def test_splice_into_empty_file_writes_only_the_block():
    assert splice_guardrail_block("", EXPORT_BLOCK) == EXPORT_BLOCK


def test_splice_replaces_existing_block_in_place():
    text = "# Project\n\n" + EXPORT_BLOCK + "\n## After\n"
    assert splice_guardrail_block(text, NEW_BLOCK) == "# Project\n\n" + NEW_BLOCK + "\n## After\n"


def test_splice_same_block_is_identity():
    text = "# Project\n\n" + EXPORT_BLOCK
    assert splice_guardrail_block(text, EXPORT_BLOCK) == text


def test_splice_empty_digest_removes_block_and_preceding_newline():
    text = "# Project\n\n" + EXPORT_BLOCK
    assert splice_guardrail_block(text, "") == "# Project\n"


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("## Guardrails\n", "## Next section\nbody\n"),
        ("## Guardrails\n", ""),
        ("", "# Project\n"),
        ("# P\n\n", "\n## After\n"),
    ],
    ids=["under-heading", "under-heading-at-eof", "file-start", "blank-lines-around"],
)
def test_splice_empty_digest_keeps_the_line_above_intact(before, after):
    # Only a blank line separating the block goes with it; the newline ending
    # the line above (e.g. the user's own heading) stays.
    text = before + EXPORT_BLOCK + after
    expected = (before[:-1] if before.endswith("\n\n") else before) + after
    assert splice_guardrail_block(text, "") == expected


def test_splice_empty_digest_without_block_is_identity():
    assert splice_guardrail_block("# Project\n", "") == "# Project\n"


@pytest.mark.parametrize(
    "block",
    [
        "- (x) no markers\n",
        EXPORT_BLOCK + EXPORT_BLOCK,
        EXPORT_BLOCK.replace("<!-- kagura-memory:guardrails end -->\n", ""),
        "<!-- kagura-memory:guardrails end -->\n"
        + EXPORT_BLOCK.replace("<!-- kagura-memory:guardrails end -->\n", ""),
    ],
    ids=["no-markers", "two-blocks", "no-end", "end-first"],
)
def test_splice_rejects_malformed_fetched_block(block):
    # An end-first block used to be written, and the next run then refused
    # the file as broken (#285).
    with pytest.raises(ValueError, match="marker"):
        splice_guardrail_block("# Project\n", block)


@pytest.mark.parametrize(
    "text",
    [
        "# P\n\n" + EXPORT_BLOCK + "\n" + EXPORT_BLOCK,
        "# P\n\n" + EXPORT_BLOCK.replace("<!-- kagura-memory:guardrails end -->\n", ""),
        "<!-- kagura-memory:guardrails end -->\n# P\n<!-- kagura-memory:guardrails begin x -->\n",
    ],
    ids=["two-blocks", "unterminated", "end-before-begin"],
)
def test_splice_refuses_ambiguous_file(text):
    with pytest.raises(ValueError, match="by hand"):
        splice_guardrail_block(text, NEW_BLOCK)


def test_write_block_keeps_crlf_line_endings(tmp_path):
    out = tmp_path / "AGENTS.md"
    out.write_bytes(b"# Title\r\n\r\nLine one\r\nLine two\r\n")

    assert write_guardrail_block(out, EXPORT_BLOCK) == "written"
    assert out.read_bytes() == (
        b"# Title\r\n\r\nLine one\r\nLine two\r\n\r\n" + EXPORT_BLOCK.replace("\n", "\r\n").encode()
    )
    # The CRLF file with this block is recognized as up to date...
    assert write_guardrail_block(out, EXPORT_BLOCK) == "unchanged"
    # ...and a removal keeps CRLF too.
    assert write_guardrail_block(out, "") == "removed"
    assert out.read_bytes() == b"# Title\r\n\r\nLine one\r\nLine two\r\n"


def test_write_block_keeps_lf_line_endings(tmp_path):
    existing = tmp_path / "AGENTS.md"
    existing.write_bytes(b"# Title\n")
    assert write_guardrail_block(existing, EXPORT_BLOCK) == "written"
    assert existing.read_bytes() == b"# Title\n\n" + EXPORT_BLOCK.encode()

    created = tmp_path / "NEW.md"
    assert write_guardrail_block(created, EXPORT_BLOCK) == "written"
    assert created.read_bytes() == EXPORT_BLOCK.encode()  # LF on every platform


# ---------------------------------------------------------------------------
# CLI: kagura guardrails load / digest
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_credential_state(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "kagura_memory.auth.credentials.DEFAULT_CREDENTIALS_PATH",
        tmp_path / "default-credentials.json",
    )
    monkeypatch.delenv("KAGURA_API_KEY", raising=False)
    monkeypatch.delenv("KAGURA_PROFILE", raising=False)
    monkeypatch.delenv("KAGURA_MCP_URL", raising=False)
    reset_state_cache()
    yield
    reset_state_cache()


CONFIG = {"api_key": "key", "mcp_url": "https://test.com/mcp", "context_id": CTX}


def _wire_kagura_client(mock_cls: MagicMock) -> AsyncMock:
    client = AsyncMock()
    client.load_guardrails.return_value = GuardrailSet.model_validate(guardrail_set_dict())
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    mock_cls.return_value = client
    return client


@patch("kagura_memory.cli.load_config", return_value=CONFIG)
@patch("kagura_memory.cli.KaguraClient")
def test_cli_guardrails_load(mock_cls, _config):
    client = _wire_kagura_client(mock_cls)
    result = CliRunner().invoke(main, ["guardrails", "load", CTX, "--cap", "5"])
    assert result.exit_code == 0, result.output
    client.load_guardrails.assert_awaited_once_with(CTX, cap=5)
    out = json.loads(result.output)
    assert out["tool_triggered_truncated"] is True
    assert out["tool_triggered"][0]["tool_trigger"]["tool"] == "Bash|PowerShell"


@patch("kagura_memory.cli.load_config", return_value=CONFIG)
@patch("kagura_memory.cli.KaguraClient")
def test_cli_guardrails_load_falls_back_to_config_context(mock_cls, _config):
    client = _wire_kagura_client(mock_cls)
    result = CliRunner().invoke(main, ["guardrails", "load"])
    assert result.exit_code == 0, result.output
    client.load_guardrails.assert_awaited_once_with(CTX, cap=None)


def _wire_memory_client(mock_cls: MagicMock, text: str, version: str = VERSION) -> AsyncMock:
    client = AsyncMock()
    client.get_guardrail_digest.return_value = GuardrailDigest(
        context_id=CTX, target="export", text=text, tool_triggered_version=version
    )
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    mock_cls._from_resolved_auth.return_value = client
    return client


def _digest(*args: str, text: str = EXPORT_BLOCK, config: dict | None = None):
    with (
        patch("kagura_memory.cli.load_config", return_value=config or CONFIG),
        patch("kagura_memory.cli.MemoryClient") as mock_cls,
    ):
        client = _wire_memory_client(mock_cls, text)
        result = CliRunner().invoke(main, ["guardrails", "digest", *args])
    return result, client


def test_cli_guardrails_digest_prints_block():
    result, client = _digest(CTX)
    assert result.exit_code == 0, result.output
    assert result.output == EXPORT_BLOCK
    client.get_guardrail_digest.assert_awaited_once_with(
        CTX, target="export", profile=None, tools=None
    )


def test_cli_guardrails_digest_instructions_target():
    result, client = _digest(CTX, "--target", "instructions", text="base text")
    assert result.exit_code == 0, result.output
    assert result.output == "base text\n"  # no trailing newline in the body → CLI adds one
    client.get_guardrail_digest.assert_awaited_once_with(
        CTX, target="instructions", profile=None, tools=None
    )


def test_cli_guardrails_digest_instructions_forwards_tool_view():
    result, client = _digest(
        CTX,
        "--target",
        "instructions",
        "--profile",
        "core",
        "--tools",
        "remember,recall",
        text="base text",
    )
    assert result.exit_code == 0, result.output
    client.get_guardrail_digest.assert_awaited_once_with(
        CTX, target="instructions", profile="core", tools="remember,recall"
    )


@pytest.mark.parametrize("flag", ["--profile", "--tools"])
def test_cli_guardrails_digest_tool_view_needs_instructions_target(flag):
    result, client = _digest(CTX, flag, "core")
    assert result.exit_code != 0
    assert "--target instructions" in result.output
    client.get_guardrail_digest.assert_not_called()


@pytest.mark.parametrize("value", ["", "  "])
def test_cli_guardrails_digest_blank_out_is_a_usage_error(value, tmp_path, monkeypatch):
    """``--out ''`` passed click.Path as ``Path('.')`` and failed at the write (#285)."""
    monkeypatch.chdir(tmp_path)
    result, client = _digest(CTX, "--out", value)
    assert result.exit_code == 2, result.output
    assert "the path is blank" in result.output
    client.get_guardrail_digest.assert_not_called()
    assert list(tmp_path.iterdir()) == []


def test_cli_guardrails_digest_server_error_is_a_clean_message():
    with (
        patch("kagura_memory.cli.load_config", return_value=CONFIG),
        patch("kagura_memory.cli.MemoryClient") as mock_cls,
    ):
        client = _wire_memory_client(mock_cls, EXPORT_BLOCK)
        client.get_guardrail_digest.side_effect = KaguraNotFoundError("Context not found")
        result = CliRunner().invoke(main, ["guardrails", "digest", CTX])
    assert result.exit_code == 1
    assert "Context not found" in result.output
    assert "Traceback" not in result.output


def test_cli_guardrails_digest_requires_a_context():
    result, client = _digest(config={"api_key": "key", "mcp_url": "https://test.com/mcp"})
    assert result.exit_code != 0
    assert "context_id required" in result.output
    client.get_guardrail_digest.assert_not_called()


def test_cli_guardrails_digest_out_rejects_instructions_target(tmp_path):
    out = tmp_path / "AGENTS.md"
    result, client = _digest(CTX, "--target", "instructions", "--out", str(out))
    assert result.exit_code != 0
    assert "--out" in result.output
    client.get_guardrail_digest.assert_not_called()
    assert not out.exists()


def test_cli_guardrails_digest_out_writes_then_skips_unchanged(tmp_path):
    out = tmp_path / "AGENTS.md"
    out.write_text("# Project\n", encoding="utf-8")

    first, _ = _digest(CTX, "--out", str(out))
    assert first.exit_code == 0, first.output
    assert json.loads(first.output) == {
        "path": str(out),
        "status": "written",
        "tool_triggered_version": VERSION,
    }
    assert out.read_text(encoding="utf-8") == "# Project\n\n" + EXPORT_BLOCK

    mtime = out.stat().st_mtime_ns
    second, _ = _digest(CTX, "--out", str(out))
    assert second.exit_code == 0, second.output
    assert json.loads(second.output)["status"] == "unchanged"
    assert out.stat().st_mtime_ns == mtime


def test_cli_guardrails_digest_empty_set_says_so_on_stderr():
    result, _ = _digest(CTX, text="")
    assert result.exit_code == 0, result.output
    assert result.stdout == ""
    assert "No tool guardrails" in result.stderr


def test_cli_guardrails_digest_out_creates_missing_file(tmp_path):
    out = tmp_path / "AGENTS.md"
    result, _ = _digest(CTX, "--out", str(out))
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "written"
    assert out.read_text(encoding="utf-8") == EXPORT_BLOCK


def test_cli_guardrails_digest_out_empty_set_removes_block(tmp_path):
    out = tmp_path / "AGENTS.md"
    out.write_text("# Project\n\n" + EXPORT_BLOCK, encoding="utf-8")
    result, _ = _digest(CTX, "--out", str(out), text="")
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "removed"
    assert out.read_text(encoding="utf-8") == "# Project\n"


def test_cli_guardrails_digest_out_empty_set_keeps_heading_above_block(tmp_path):
    out = tmp_path / "AGENTS.md"
    out.write_text("## Guardrails\n" + EXPORT_BLOCK + "## Next section\nbody\n", encoding="utf-8")
    result, _ = _digest(CTX, "--out", str(out), text="")
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "removed"
    assert out.read_text(encoding="utf-8") == "## Guardrails\n## Next section\nbody\n"


def test_cli_guardrails_digest_out_empty_set_never_creates_file(tmp_path):
    out = tmp_path / "AGENTS.md"
    result, _ = _digest(CTX, "--out", str(out), text="")
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "unchanged"
    assert not out.exists()


def test_cli_guardrails_digest_out_refuses_ambiguous_file(tmp_path):
    out = tmp_path / "AGENTS.md"
    original = "# P\n\n" + EXPORT_BLOCK + "\n" + EXPORT_BLOCK
    out.write_text(original, encoding="utf-8")
    result, _ = _digest(CTX, "--out", str(out), text=NEW_BLOCK)
    assert result.exit_code != 0
    assert "by hand" in result.output
    assert out.read_text(encoding="utf-8") == original


def test_cli_guardrails_digest_out_failed_replace_keeps_file_and_cleans_tmp(tmp_path, monkeypatch):
    out = tmp_path / "AGENTS.md"
    out.write_text("# Project\n", encoding="utf-8")

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr("kagura_memory._guardrail_export.os.replace", boom)
    result, _ = _digest(CTX, "--out", str(out))
    assert result.exit_code == 1
    assert "disk full" in result.output
    assert out.read_text(encoding="utf-8") == "# Project\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["AGENTS.md"]


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes and symlinks")
def test_cli_guardrails_digest_out_keeps_symlink_and_mode(tmp_path):
    real = tmp_path / "CLAUDE.md"
    real.write_text("# Project\n", encoding="utf-8")
    real.chmod(0o640)
    link = tmp_path / "AGENTS.md"
    link.symlink_to(real.name)

    result, _ = _digest(CTX, "--out", str(link))
    assert result.exit_code == 0, result.output
    assert link.is_symlink()
    assert real.read_text(encoding="utf-8") == "# Project\n\n" + EXPORT_BLOCK
    assert stat.S_IMODE(real.stat().st_mode) == 0o640
    assert [p.name for p in Path(tmp_path).iterdir() if p.name.startswith(".")] == []
