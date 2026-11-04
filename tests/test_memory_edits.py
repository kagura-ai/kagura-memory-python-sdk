"""New write and read arguments on tools the SDK already wraps (#302).

``remember`` (``observed_at``, ``tags_normalize``, ``dedupe``, ``verbose``),
``update_memory`` (partial edits, ``expected_version``, the supersede accept),
``reference`` (``version``, ``content_offset``), the ``observed_at`` list
column and the content-chunk / LLM-selector search settings. Every argument
is keyword-only and sent only when set; the replies below are shaped like
memory-cloud v0.142.0's ``mcp_server/tools/memory.py`` and
``search_config.py``.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, date, datetime, timedelta, timezone, tzinfo
from typing import Any, get_args

import httpx
import pytest

from kagura_memory import (
    DedupeMode,
    KaguraClient,
    KaguraConnectionError,
    KaguraError,
    KaguraResponseError,
    KaguraVersionConflictError,
    MemoryClient,
    SearchConfig,
)
from kagura_memory.client import _observed_at_arg

CTX = "11111111-2222-3333-4444-555555555555"
MEM = "aaaaaaaa-0000-0000-0000-000000000001"
OLD = "aaaaaaaa-0000-0000-0000-000000000002"

_CONTEXT_FIELDS = {
    "context_id": CTX,
    "context_name": "dev",
    "context_display_name": "Dev",
    "context_is_private": False,
    "context_is_locked": False,
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


def _arguments(sent: list[dict[str, Any]]) -> dict[str, Any]:
    """The ``tools/call`` arguments of the one request sent."""
    assert len(sent) == 1
    assert sent[0]["method"] == "tools/call"
    return sent[0]["params"]["arguments"]


async def _remember(reply: dict[str, Any], **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    sent: list[dict[str, Any]] = []
    client = _wire_client(reply, sent)
    try:
        result = await client.remember(CTX, "Deploy target moved", "after the review", **kwargs)
    finally:
        await client.close()
    return result, _arguments(sent)


_REMEMBERED = {
    "status": "success",
    "memory_id": MEM,
    "scope": "working",
    "version": 1,
    "persistence": {
        "scope": "working",
        "committed": True,
        "promotes_via": "sleep",
        "consolidation_archive_min_age_days": 30,
    },
    **_CONTEXT_FIELDS,
}

# ============================================================================
# observed_at: the argument's shape, checked as the server checks it
# ============================================================================


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2024-03-07", "2024-03-07"),
        ("2024-03-07T10:00:00Z", "2024-03-07T10:00:00Z"),
        ("2024-03-07T10:00:00.123+09:00", "2024-03-07T10:00:00.123+09:00"),
        ("2024-03-07T10:00:00", "2024-03-07T10:00:00"),
        (date(2024, 3, 7), "2024-03-07"),
        (datetime(2024, 3, 7, 10, 0, 0, 500), "2024-03-07T10:00:00Z"),
        (datetime(2024, 3, 7, 19, 0, tzinfo=timezone(timedelta(hours=9))), "2024-03-07T10:00:00Z"),
        (datetime(2024, 3, 7, 10, 0, tzinfo=UTC), "2024-03-07T10:00:00Z"),
        # The server strips the value; the 40-character cap is on it as sent.
        (" 2024-03-07 ", " 2024-03-07 "),
        ("2024-03-07T10:00:00." + "1" * 20, "2024-03-07T10:00:00." + "1" * 20),
    ],
    ids=[
        "date",
        "utc",
        "offset-fraction",
        "naive",
        "date-object",
        "naive-datetime-is-utc",
        "aware-datetime-converted",
        "utc-datetime",
        "padded",
        "40-chars",
    ],
)
def test_observed_at_arg_renders_the_accepted_forms(value: object, expected: str) -> None:
    """A string goes as written; an object becomes the server's stored form."""
    assert _observed_at_arg(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "20240307",
        "2024-03-07 10:00:00",
        "2024-03-07T10:00",
        "2024-W10",
        "2024-02-30",
        "2024-03-07T25:00:00Z",
        "",
        "   ",
        "2024-03-07T10:00:00." + "1" * 30,
        # Stripped it is short, but the request model caps the string as sent.
        " 2024-03-07" + " " * 30,
        # Python 3.14 reads it as the next midnight; the server (3.11) refuses it.
        "2024-03-07T24:00:00Z",
        # In UTC the instant falls before year 1 / after year 9999 (OverflowError there).
        "0001-01-01T00:30:00+01:00",
        "9999-12-31T23:30:00-01:00",
        20240307,
        True,
        None,
    ],
    ids=[
        "basic-form",
        "space-separator",
        "no-seconds",
        "week-date",
        "not-a-day",
        "not-an-hour",
        "empty",
        "blank",
        "over-40-chars",
        "over-40-as-sent",
        "hour-24",
        "before-year-1",
        "after-year-9999",
        "int",
        "bool",
        "none",
    ],
)
def test_observed_at_arg_refuses_what_the_server_refuses(value: object) -> None:
    with pytest.raises(ValueError, match="ISO 8601"):
        _observed_at_arg(value)


@pytest.mark.parametrize(
    "value",
    [
        datetime(1, 1, 1, 0, 30, tzinfo=timezone(timedelta(hours=1))),
        datetime(9999, 12, 31, 23, 30, tzinfo=timezone(timedelta(hours=-1))),
    ],
    ids=["before-year-1", "after-year-9999"],
)
def test_observed_at_arg_out_of_range_datetime_is_a_value_error(value: datetime) -> None:
    """Converting to UTC overflows: the documented ``ValueError``, never ``OverflowError``."""
    with pytest.raises(ValueError, match="ISO 8601"):
        _observed_at_arg(value)


class _NoOffset(tzinfo):
    """A tzinfo that knows no offset: Python counts such a datetime as naive."""

    def utcoffset(self, dt: datetime | None) -> timedelta | None:
        return None

    def dst(self, dt: datetime | None) -> timedelta | None:
        return None


@pytest.mark.skipif(not hasattr(time, "tzset"), reason="time.tzset is POSIX-only")
@pytest.mark.parametrize(
    "value",
    [datetime(2024, 3, 7, 10, 0), datetime(2024, 3, 7, 10, 0, tzinfo=_NoOffset())],
    ids=["naive", "tzinfo-without-offset"],
)
def test_observed_at_arg_naive_is_utc_whatever_the_local_zone(
    monkeypatch: pytest.MonkeyPatch, value: datetime
) -> None:
    """Naive means UTC, as on the server, never the machine's own zone."""
    monkeypatch.setenv("TZ", "Asia/Tokyo")
    time.tzset()
    try:
        assert _observed_at_arg(value) == "2024-03-07T10:00:00Z"
    finally:
        monkeypatch.undo()
        time.tzset()


# ============================================================================
# remember(observed_at=, tags_normalize=, dedupe=, verbose=)
# ============================================================================


@pytest.mark.asyncio
async def test_remember_omits_the_new_arguments_by_default() -> None:
    """Unset, a server before v0.82.0 never sees them and one from v0.82.0 never refuses them."""
    _, arguments = await _remember(_REMEMBERED)
    for key in ("observed_at", "tags_normalize", "dedupe", "verbose"):
        assert key not in arguments


@pytest.mark.asyncio
async def test_remember_sends_the_new_arguments_when_set() -> None:
    reply = {
        **_REMEMBERED,
        "persistence": {**_REMEMBERED["persistence"], "detail": "Working memories ..."},
        "lint": [{"code": "tag_normalized", "subject": "Dev_Env", "replacement": "dev-env"}],
    }
    result, arguments = await _remember(
        reply,
        tags=["Dev_Env"],
        observed_at=date(2024, 3, 7),
        tags_normalize=True,
        dedupe="off",
        verbose=True,
    )
    assert arguments["observed_at"] == "2024-03-07"
    assert arguments["tags_normalize"] is True
    assert arguments["dedupe"] == "off"
    assert arguments["verbose"] is True
    assert result["version"] == 1
    assert result["persistence"]["detail"] == "Working memories ..."
    assert result["lint"][0]["replacement"] == "dev-env"


@pytest.mark.asyncio
async def test_remember_false_flags_stay_off_the_wire() -> None:
    """``False`` is the server default: sending it would only risk an invalid_argument."""
    _, arguments = await _remember(_REMEMBERED, tags_normalize=False, verbose=False)
    assert "tags_normalize" not in arguments
    assert "verbose" not in arguments


@pytest.mark.asyncio
async def test_remember_dedupe_check_returns_the_candidate() -> None:
    """``duplicate_candidate`` is a decision for the caller, not an error: nothing raises."""
    reply = {
        "status": "duplicate_candidate",
        "candidate": {"memory_id": OLD, "summary": "Deploy target moved", "similarity": 0.93},
        **_CONTEXT_FIELDS,
    }
    result, arguments = await _remember(reply, dedupe="check")
    assert arguments["dedupe"] == "check"
    assert result["status"] == "duplicate_candidate"
    assert result["candidate"]["memory_id"] == OLD


@pytest.mark.asyncio
async def test_remember_observed_at_may_repeat_details() -> None:
    """The same value in ``details`` agrees; the server folds one into the other."""
    _, arguments = await _remember(
        _REMEMBERED, details={"observed_at": "2024-03-07", "k": 1}, observed_at="2024-03-07"
    )
    assert arguments["details"] == {"observed_at": "2024-03-07", "k": 1}
    assert arguments["observed_at"] == "2024-03-07"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"details": {"observed_at": "2024-03-08"}, "observed_at": "2024-03-07"},
            "disagree",
        ),
        (
            # The server compares the values as sent, not as instants.
            {
                "details": {"observed_at": "2024-03-07T10:00:00+00:00"},
                "observed_at": "2024-03-07T10:00:00Z",
            },
            "disagree",
        ),
        ({"observed_at": "07/03/2024"}, "ISO 8601"),
        ({"dedupe": "skip"}, "dedupe"),
        ({"dedupe": "CHECK"}, "dedupe"),
    ],
    ids=["details-disagree", "same-instant-other-spelling", "bad-shape", "bad-dedupe", "case"],
)
async def test_remember_refuses_locally(kwargs: dict[str, Any], match: str) -> None:
    sent: list[dict[str, Any]] = []
    client = _wire_client(_REMEMBERED, sent)
    try:
        with pytest.raises(ValueError, match=match):
            await client.remember(CTX, "Deploy target moved", "after the review", **kwargs)
    finally:
        await client.close()
    assert sent == []


def test_dedupe_mode_lists_the_server_values() -> None:
    assert set(get_args(DedupeMode)) == {"suggest", "check", "off"}


# ============================================================================
# update_memory: partial edits, expected_version, the supersede accept
# ============================================================================

_UPDATED = {
    "status": "success",
    "memory_id": MEM,
    "operation": "updated",
    "re_embedded": False,
    "scope": "working",
    "version": 4,
    **_CONTEXT_FIELDS,
}


async def _update(reply: dict[str, Any], **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    sent: list[dict[str, Any]] = []
    client = _wire_client(reply, sent)
    try:
        result = await client.update_memory(CTX, **kwargs)
    finally:
        await client.close()
    return result, _arguments(sent)


_NEW_UPDATE_ARGS = (
    "observed_at",
    "add_tags",
    "remove_tags",
    "details_patch",
    "expected_version",
    "accept_supersede_candidate",
    "verbose",
)


@pytest.mark.asyncio
async def test_update_memory_omits_the_new_arguments_by_default() -> None:
    _, arguments = await _update(_UPDATED, memory_id=MEM, summary="Deploy target moved")
    for key in _NEW_UPDATE_ARGS:
        assert key not in arguments
    assert arguments == {"context_id": CTX, "memory_id": MEM, "summary": "Deploy target moved"}


@pytest.mark.asyncio
async def test_update_memory_sends_partial_edits() -> None:
    """One call flips a status: nothing else of the stored tags or details is resent."""
    result, arguments = await _update(
        _UPDATED,
        memory_id=MEM,
        add_tags=["status:done"],
        remove_tags=["status:open"],
        details_patch={"status": "done", "assignee": None},
        observed_at="2026-11-04",
        expected_version=3,
        verbose=True,
    )
    assert arguments["add_tags"] == ["status:done"]
    assert arguments["remove_tags"] == ["status:open"]
    assert arguments["details_patch"] == {"status": "done", "assignee": None}
    assert arguments["observed_at"] == "2026-11-04"
    assert arguments["expected_version"] == 3
    assert arguments["verbose"] is True
    for key in ("tags", "details", "accept_supersede_candidate"):
        assert key not in arguments
    assert result["version"] == 4


@pytest.mark.asyncio
async def test_update_memory_tool_trigger_goes_into_the_patch() -> None:
    """With ``details_patch`` the guardrail is merged into it, so other stored keys survive."""
    _, arguments = await _update(
        _UPDATED,
        memory_id=MEM,
        details_patch={"status": "done"},
        tool_trigger={"tool": "Bash", "pattern": "rm -rf"},
    )
    assert arguments["details_patch"] == {
        "status": "done",
        "tool_trigger": {"tool": "Bash", "pattern": "rm -rf"},
    }
    assert "details" not in arguments


@pytest.mark.asyncio
async def test_update_memory_observed_at_alone_edits_only_that_key() -> None:
    _, arguments = await _update(
        _UPDATED, memory_id=MEM, observed_at=datetime(2026, 11, 4, 9, 30, tzinfo=UTC)
    )
    assert arguments == {"context_id": CTX, "memory_id": MEM, "observed_at": "2026-11-04T09:30:00Z"}


@pytest.mark.asyncio
async def test_update_memory_accept_supersede_candidate() -> None:
    """A graph action: only the flag and a version guard go out; the reply names the old memory."""
    reply = {**_UPDATED, "version": 3, "superseded_memory_id": OLD}
    result, arguments = await _update(
        reply, memory_id=MEM, accept_supersede_candidate=True, expected_version=3
    )
    assert arguments == {
        "context_id": CTX,
        "memory_id": MEM,
        "accept_supersede_candidate": True,
        "expected_version": 3,
    }
    assert result["superseded_memory_id"] == OLD


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"external_id": "e", "add_tags": ["a"]}, "add_tags require memory_id"),
        ({"external_id": "e", "remove_tags": ["a"]}, "remove_tags require memory_id"),
        ({"external_id": "e", "details_patch": {"a": 1}}, "details_patch require memory_id"),
        ({"external_id": "e", "expected_version": 2}, "expected_version requires memory_id"),
        (
            {"external_id": "e", "accept_supersede_candidate": True},
            "accept_supersede_candidate requires memory_id",
        ),
        (
            {"external_id": "e", "dismiss_supersede_candidate": True},
            "dismiss_supersede_candidate requires memory_id",
        ),
        ({"memory_id": MEM, "tags": ["a"], "add_tags": ["b"]}, "give one or the other"),
        ({"memory_id": MEM, "tags": [], "remove_tags": ["b"]}, "give one or the other"),
        ({"memory_id": MEM, "details": {}, "details_patch": {"a": 1}}, "give one or the other"),
        (
            {
                "memory_id": MEM,
                "accept_supersede_candidate": True,
                "dismiss_supersede_candidate": True,
            },
            "cannot both be true",
        ),
        (
            {"memory_id": MEM, "accept_supersede_candidate": True, "summary": "x" * 20},
            r"field edits \(summary\)",
        ),
        (
            {"memory_id": MEM, "accept_supersede_candidate": True, "add_tags": ["a"]},
            r"field edits \(add_tags\)",
        ),
        (
            {"memory_id": MEM, "accept_supersede_candidate": True, "observed_at": "2024-03-07"},
            r"field edits \(observed_at\)",
        ),
        (
            {
                "memory_id": MEM,
                "accept_supersede_candidate": True,
                "tool_trigger": {"tool": "Bash", "pattern": "x"},
            },
            r"field edits \(details\)",
        ),
        ({"memory_id": MEM, "expected_version": 0}, "expected_version must be an integer"),
        ({"memory_id": MEM, "expected_version": True}, "expected_version must be an integer"),
        ({"memory_id": MEM, "expected_version": "3"}, "expected_version must be an integer"),
        (
            {
                "memory_id": MEM,
                "details_patch": {"observed_at": None},
                "observed_at": "2024-03-07",
            },
            r"details_patch\['observed_at'\] and observed_at disagree",
        ),
        (
            {
                "memory_id": MEM,
                "details": {"observed_at": "2024-03-08"},
                "observed_at": "2024-03-07",
            },
            r"details\['observed_at'\] and observed_at disagree",
        ),
        ({"memory_id": MEM, "observed_at": "March 7"}, "ISO 8601"),
        (
            {"memory_id": MEM, "add_tags": [f"t{i}" for i in range(51)]},
            "add_tags accepts at most 50 tags, got 51",
        ),
        ({"memory_id": MEM, "add_tags": ["x" * 101]}, "at most 100 characters"),
        (
            {
                "memory_id": MEM,
                "details_patch": {"tool_trigger": None},
                "tool_trigger": {"tool": "Bash", "pattern": "x"},
            },
            "not both",
        ),
    ],
    ids=[
        "upsert-add-tags",
        "upsert-remove-tags",
        "upsert-details-patch",
        "upsert-expected-version",
        "upsert-accept",
        "upsert-dismiss",
        "tags-and-add-tags",
        "tags-and-remove-tags",
        "details-and-details-patch",
        "accept-and-dismiss",
        "accept-with-summary",
        "accept-with-add-tags",
        "accept-with-observed-at",
        "accept-with-tool-trigger",
        "version-zero",
        "version-bool",
        "version-str",
        "patch-deletes-observed-at",
        "details-disagree",
        "observed-at-shape",
        "add-tags-count",
        "add-tags-length",
        "tool-trigger-twice-in-patch",
    ],
)
async def test_update_memory_refuses_locally(kwargs: dict[str, Any], match: str) -> None:
    """Each mirrors a server validation_error; nothing goes on the wire."""
    sent: list[dict[str, Any]] = []
    client = _wire_client(_UPDATED, sent)
    try:
        with pytest.raises(ValueError, match=match):
            await client.update_memory(CTX, **kwargs)
    finally:
        await client.close()
    assert sent == []


@pytest.mark.asyncio
async def test_update_memory_remove_tags_is_not_capped() -> None:
    """The server caps the tags a call adds, never the ones it removes (#2585)."""
    long_tag = "x" * 150
    _, arguments = await _update(_UPDATED, memory_id=MEM, remove_tags=[long_tag])
    assert arguments["remove_tags"] == [long_tag]


@pytest.mark.asyncio
async def test_update_memory_accept_conflicting_edge_is_a_kagura_error() -> None:
    """``edge_exists`` keeps the existing edge in ``details``."""
    edge = {"source_id": MEM, "target_id": OLD, "edge_type": "declared_link"}
    sent: list[dict[str, Any]] = []
    client = _wire_client(
        {"status": "error", "error": "edge_exists", "message": "...", "existing_edge": edge}, sent
    )
    try:
        with pytest.raises(KaguraError) as excinfo:
            await client.update_memory(CTX, memory_id=MEM, accept_supersede_candidate=True)
    finally:
        await client.close()
    assert excinfo.value.code == "edge_exists"
    assert excinfo.value.details["existing_edge"] == edge


# ============================================================================
# KaguraVersionConflictError: MCP version_conflict and REST 409 RES-002
# ============================================================================

_MCP_CONFLICT = {
    "status": "error",
    "error": "version_conflict",
    "message": f"Memory {MEM} is at version 5, not the expected version 3. Read it again "
    "and resend with its current version.",
    "memory_id": MEM,
    "expected_version": 3,
    "current_version": 5,
    "retryable": False,
    "help": "reference() the memory and resend with its current version, "
    "or omit expected_version to overwrite.",
}


@pytest.mark.asyncio
async def test_mcp_version_conflict_is_typed() -> None:
    sent: list[dict[str, Any]] = []
    client = _wire_client(_MCP_CONFLICT, sent)
    try:
        with pytest.raises(KaguraVersionConflictError) as excinfo:
            await client.update_memory(CTX, memory_id=MEM, summary="x" * 20, expected_version=3)
    finally:
        await client.close()
    err = excinfo.value
    assert isinstance(err, KaguraError)
    assert err.code == "version_conflict"
    assert err.current_version == 5
    assert err.expected_version == 3
    assert err.memory_id == MEM
    assert err.retryable is False
    assert err.help is not None and "reference()" in err.help
    assert "version 5" in str(err)


def _rest_memory_client(status: int, body: Any):
    client = MemoryClient(api_key="kagura_test", base_url="https://test.com")
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _req: httpx.Response(status, json=body))
    )
    return client


@pytest.mark.asyncio
async def test_rest_409_version_conflict_is_typed() -> None:
    """``PATCH /api/v1/memory/{id}``'s refusal, through the shared REST spine."""
    body = {
        "error": "RES-002",
        "message": f"Memory {MEM} is at version 5, not the expected version 3.",
        "details": {"reason": "version_conflict", "expected_version": 3, "current_version": 5},
    }
    async with _rest_memory_client(409, body) as client:
        with pytest.raises(KaguraVersionConflictError) as excinfo:
            await client._request("PATCH", f"/api/v1/memory/{MEM}", json={"expected_version": 3})
    err = excinfo.value
    assert err.code == "RES-002"
    assert err.details["reason"] == "version_conflict"
    assert err.current_version == 5
    assert err.expected_version == 3
    assert err.memory_id is None
    assert "version 5" in str(err)
    assert isinstance(err.__cause__, httpx.HTTPStatusError)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        {"error": "RES-002", "message": "A file with this sha256 exists.", "details": {}},
        {"error": "RES-002", "message": "edge", "details": {"reason": "edge_exists"}},
        {"detail": "conflict"},
    ],
    ids=["files-dedup", "edge-exists", "fastapi-detail"],
)
async def test_rest_other_409s_are_not_version_conflicts(body: dict[str, Any]) -> None:
    """The files dedup 409 shares ``RES-002``: it must stay what ``FilesClient.upload`` reads."""
    async with _rest_memory_client(409, body) as client:
        with pytest.raises(KaguraConnectionError) as excinfo:
            await client._request("POST", "/api/v1/files/reserve", json={})
    assert not isinstance(excinfo.value, KaguraVersionConflictError)


def test_version_conflict_properties_ignore_malformed_details() -> None:
    err = KaguraVersionConflictError("conflict")
    err.details = {"current_version": True, "expected_version": "3", "memory_id": 7}
    assert err.current_version is None
    assert err.expected_version is None
    assert err.memory_id is None


# ============================================================================
# reference(version=, content_offset=)
# ============================================================================


async def _reference(reply: dict[str, Any], **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    sent: list[dict[str, Any]] = []
    client = _wire_client(reply, sent)
    try:
        result = await client.reference(CTX, MEM, **kwargs)
    finally:
        await client.close()
    return result, _arguments(sent)


@pytest.mark.asyncio
async def test_reference_omits_version_and_offset_by_default() -> None:
    reply = {"status": "success", "memory": {"memory_id": MEM, "version": 5}}
    result, arguments = await _reference(reply)
    assert arguments == {"context_id": CTX, "memory_id": MEM}
    assert result["memory"]["version"] == 5


@pytest.mark.asyncio
async def test_reference_reads_an_earlier_version() -> None:
    reply = {
        "status": "success",
        "memory": {
            "memory_id": MEM,
            "summary": "Deploy target was us-east-1",
            "version": 2,
            "historical": True,
            "current_version": 5,
        },
    }
    result, arguments = await _reference(reply, version=2)
    assert arguments == {"context_id": CTX, "memory_id": MEM, "version": 2}
    assert result["memory"]["historical"] is True
    assert result["memory"]["current_version"] == 5


@pytest.mark.asyncio
async def test_reference_continues_content() -> None:
    """``content_offset=0`` is a value (the start), so it goes on the wire."""
    reply = {
        "status": "success",
        "memory": {
            "memory_id": MEM,
            "content": "c" * 100,
            "content_offset": 1500,
            "content_total_chars": 1600,
            "content_truncated": False,
        },
    }
    _, arguments = await _reference(reply, content_offset=1500)
    assert arguments["content_offset"] == 1500
    _, arguments = await _reference(reply, content_offset=0)
    assert arguments["content_offset"] == 0


@pytest.mark.asyncio
async def test_reference_unknown_version_is_a_validation_error() -> None:
    reply = {
        "status": "error",
        "error": "validation_error",
        "message": "Memory has no version 9.",
        "received": 9,
        "current_version": 5,
    }
    sent: list[dict[str, Any]] = []
    client = _wire_client(reply, sent)
    try:
        with pytest.raises(KaguraError) as excinfo:
            await client.reference(CTX, MEM, version=9)
    finally:
        await client.close()
    assert excinfo.value.code == "validation_error"
    assert excinfo.value.details["current_version"] == 5


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"version": 0}, "version must be an integer of at least 1"),
        ({"version": True}, "version must be an integer"),
        ({"version": 2.0}, "version must be an integer"),
        ({"content_offset": -1}, "content_offset must be an integer of at least 0"),
        ({"content_offset": False}, "content_offset must be an integer"),
    ],
    ids=["version-zero", "version-bool", "version-float", "offset-negative", "offset-bool"],
)
async def test_reference_refuses_locally(kwargs: dict[str, Any], match: str) -> None:
    sent: list[dict[str, Any]] = []
    client = _wire_client({"status": "success", "memory": {}}, sent)
    try:
        with pytest.raises(ValueError, match=match):
            await client.reference(CTX, MEM, **kwargs)
    finally:
        await client.close()
    assert sent == []


# ============================================================================
# list_memories: MemoryListItem.observed_at (the REST route has no filter)
# ============================================================================

_LIST_ROW = {
    "id": MEM,
    "summary": "Deploy target moved",
    "type": "decision",
    "scope": "persistent",
    "importance": 0.8,
    "created_at": "2026-10-01T00:00:00Z",
    "updated_at": "2026-10-02T00:00:00Z",
}


async def _list_memories(rows: list[dict[str, Any]], **kwargs: Any):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"memories": rows, "total": len(rows), "has_more": False})

    client = KaguraClient(api_key="test", mcp_url="https://test.com/mcp")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        page = await client.list_memories(context_id=CTX, **kwargs)
    finally:
        await client.close()
    return page, requests


@pytest.mark.asyncio
async def test_list_memories_reads_observed_at() -> None:
    """A datetime keeps its instant; a date is the start of that UTC day; absent is None."""
    rows = [
        {**_LIST_ROW, "observed_at": "2024-03-07T10:00:00Z"},
        {**_LIST_ROW, "observed_at": "2024-03-07"},
        {**_LIST_ROW, "observed_at": None},
        _LIST_ROW,
    ]
    page, requests = await _list_memories(rows)
    observed = [row.observed_at for row in page.memories]
    assert observed == [
        datetime(2024, 3, 7, 10, tzinfo=UTC),
        datetime(2024, 3, 7, tzinfo=UTC),
        None,
        None,
    ]
    # Every value is aware, so rows sort and compare without a TypeError.
    assert all(value.tzinfo is not None for value in observed if value is not None)
    params = dict(requests[0].url.params)
    assert not any(key.startswith("observed") for key in params)


@pytest.mark.asyncio
async def test_list_memories_ignores_fields_newer_than_the_sdk() -> None:
    rows = [{**_LIST_ROW, "observed_at": "2024-03-07", "version": 4, "tags": ["deploy"]}]
    page, _ = await _list_memories(rows)
    assert page.memories[0].observed_at == datetime(2024, 3, 7, tzinfo=UTC)
    assert not hasattr(page.memories[0], "version")


@pytest.mark.asyncio
async def test_list_memories_legacy_observed_at_reads_as_none() -> None:
    """The server lists any value in its stored shape, a pre-v0.127.0 one unchecked.

    One such row must not fail the page: it reads as None, the others parse.
    """
    rows = [
        {**_LIST_ROW, "observed_at": "2024-02-30"},
        {**_LIST_ROW, "observed_at": "2024-03-07T24:00:00Z"},
        {**_LIST_ROW, "observed_at": "2024-13-01T10:00:00Z"},
        {**_LIST_ROW, "observed_at": "2024-03-07"},
    ]
    page, _ = await _list_memories(rows)
    assert [row.observed_at for row in page.memories] == [
        None,
        None,
        None,
        datetime(2024, 3, 7, tzinfo=UTC),
    ]


@pytest.mark.asyncio
async def test_list_memories_unreadable_observed_at_is_response_drift() -> None:
    with pytest.raises(KaguraResponseError, match="list_memories"):
        await _list_memories([{**_LIST_ROW, "observed_at": "last spring"}])


# ============================================================================
# update_search_config(content_chunks_enabled=, llm_selector_enabled=) + SearchConfig
# ============================================================================

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
        "content_chunks_enabled": True,
        "content_chunks_effective": False,
        "llm_selector_enabled": False,
        "llm_selector_effective": False,
    },
}


async def _search_config(**kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    sent: list[dict[str, Any]] = []
    client = _wire_client(_SEARCH_CONFIG_REPLY, sent)
    try:
        result = await client.update_search_config(CTX, **kwargs)
    finally:
        await client.close()
    return result, _arguments(sent)


@pytest.mark.asyncio
async def test_update_search_config_omits_the_toggles_by_default() -> None:
    _, arguments = await _search_config(use_rerank=True)
    assert arguments == {"context_id": CTX, "use_rerank": True}


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [True, False], ids=["on", "off"])
async def test_update_search_config_sends_the_toggles(value: bool) -> None:
    """``False`` turns a setting off, so it goes on the wire like ``True``."""
    result, arguments = await _search_config(
        content_chunks_enabled=value, llm_selector_enabled=value
    )
    assert arguments == {
        "context_id": CTX,
        "content_chunks_enabled": value,
        "llm_selector_enabled": value,
    }
    assert result["config"]["content_chunks_effective"] is False


@pytest.mark.asyncio
async def test_update_search_config_owner_only_refusal() -> None:
    sent: list[dict[str, Any]] = []
    client = _wire_client(
        {
            "status": "error",
            "error": "permission_denied",
            "message": "Only the context owner or a workspace owner/admin may change it.",
            "required_role": "owner",
            "help": "Ask the context owner or a workspace owner/admin to change ...",
        },
        sent,
    )
    try:
        with pytest.raises(KaguraError) as excinfo:
            await client.update_search_config(CTX, content_chunks_enabled=True)
    finally:
        await client.close()
    assert excinfo.value.code == "permission_denied"
    assert excinfo.value.details["required_role"] == "owner"


def test_search_config_reads_every_v0142_field() -> None:
    """The REST search-config shape: every field reaches the model."""
    config = SearchConfig.model_validate(
        {
            "context_id": CTX,
            "semantic_weight": 0.5,
            "bm25_weight": 0.5,
            "fetch_factor": 3,
            "use_rerank": True,
            "reranker_provider": "cohere",
            "reranker_model": "rerank-v3.5",
            "embedding_model": "text-embedding-3-small",
            "content_chunks_enabled": True,
            "content_chunks_effective": False,
            "content_chunks_limit": 1,
            "content_chunks_in_use": 1,
            "content_chunks_disabled_reason": "cap",
            "content_chunks_can_change": True,
            "content_chunks_supported": True,
            "content_chunks_backfill_pending": 12,
            "embedding_provider": "openai",
            "llm_selector_enabled": True,
            "llm_selector_effective": True,
            "llm_selector_limit": None,
            "llm_selector_in_use": 2,
            "llm_selector_disabled_reason": None,
            "llm_selector_can_change": False,
            "created_at": "2026-10-01T00:00:00Z",
            "a_field_from_a_newer_server": {"x": 1},
        }
    )
    assert config.content_chunks_enabled is True
    assert config.content_chunks_effective is False
    assert config.content_chunks_limit == 1
    assert config.content_chunks_in_use == 1
    assert config.content_chunks_disabled_reason == "cap"
    assert config.content_chunks_can_change is True
    assert config.content_chunks_supported is True
    assert config.content_chunks_backfill_pending == 12
    assert config.embedding_provider == "openai"
    assert config.llm_selector_enabled is True
    assert config.llm_selector_effective is True
    assert config.llm_selector_limit is None
    assert config.llm_selector_in_use == 2
    assert config.llm_selector_can_change is False
    assert not hasattr(config, "a_field_from_a_newer_server")


def test_search_config_unknown_reason_passes_through() -> None:
    config = SearchConfig.model_validate({"content_chunks_disabled_reason": "region"})
    assert config.content_chunks_disabled_reason == "region"


@pytest.mark.asyncio
async def test_get_context_info_reads_the_selector_and_leaves_chunks_unreported() -> None:
    """At v0.142.0 get_context_info sends the selector pair only; the rest read None."""
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
                "reranker_provider": "voyage",
                "reranker_model": "rerank-2",
                "llm_selector_enabled": True,
                "llm_selector_effective": False,
            },
        },
    }
    sent: list[dict[str, Any]] = []
    client = _wire_client(reply, sent)
    try:
        info = await client.get_context_info(CTX)
    finally:
        await client.close()
    config = info.context.search_config
    assert config.llm_selector_enabled is True
    assert config.llm_selector_effective is False
    assert config.content_chunks_enabled is None
    assert config.content_chunks_supported is None
