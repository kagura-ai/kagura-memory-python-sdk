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
from datetime import UTC, date, datetime, timedelta, timezone
from typing import Any

import httpx
import pytest

from kagura_memory import DedupeMode, KaguraClient
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
        "int",
        "bool",
        "none",
    ],
)
def test_observed_at_arg_refuses_what_the_server_refuses(value: object) -> None:
    with pytest.raises(ValueError, match="ISO 8601"):
        _observed_at_arg(value)


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
    from typing import get_args

    assert set(get_args(DedupeMode)) == {"suggest", "check", "off"}
