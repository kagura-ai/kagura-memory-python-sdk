"""Workspace operation history reader (#332, memory-cloud v0.154.0+).

``WorkspaceClient.list_operation_history`` wraps ``GET
/api/v1/workspaces/{id}/operation-history``: a web session only (owner or
admin, a plan with the ``operation_history`` feature). The contract below was
read from the server source at tag v0.155.0; every test runs against an
``httpx.MockTransport``, never a live server.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from typing import Any, get_args

import httpx
import pytest

import kagura_memory
from kagura_memory import (
    KaguraAuthError,
    KaguraConnectionError,
    KaguraFeatureNotAvailableError,
    KaguraNotFoundError,
    KaguraResponseError,
    OperationHistoryItem,
    OperationHistoryOutcome,
    OperationHistoryPage,
    OperationHistoryTargetType,
)
from kagura_memory.workspace_client import WorkspaceClient

WS = "11111111-2222-3333-4444-555555555555"
CTX = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
AGENT = "bbbbbbbb-cccc-dddd-eeee-ffffffffffff"
EVENT_A = "cccccccc-0000-0000-0000-000000000001"
EVENT_B = "cccccccc-0000-0000-0000-000000000002"
COOKIE = "test-session"
PATH = f"/api/v1/workspaces/{WS}/operation-history"
# An opaque server cursor: sent back verbatim, never parsed.
CURSOR = "MjAyNi0xMS0wMVQwMDowMDowMCswMDowMHxjY2NjY2NjYw=="


def _item(**overrides: Any) -> dict[str, Any]:
    """One item as the v0.155.0 route serializes it (every key, nulls included)."""
    item: dict[str, Any] = {
        "id": EVENT_A,
        "occurred_at": "2026-11-01T09:30:00Z",
        "context_id": CTX,
        "actor_user_id": "google_123",
        "actor_name": "Owner",
        "subject_user_id": None,
        "agent_id": None,
        "principal_kind": "api_key",
        "credential_id": "akey_0123",
        "client_id": None,
        "surface": "mcp",
        "operation": "memory.updated",
        "target_type": "memory",
        "target_id": "dddddddd-0000-0000-0000-000000000001",
        "outcome": "success",
        "count": 1,
        "changed_fields": ["summary", "tags"],
    }
    item.update(overrides)
    return item


def _page(**overrides: Any) -> dict[str, Any]:
    page: dict[str, Any] = {
        "items": [_item()],
        "total": 1,
        "total_capped": False,
        "next_cursor": None,
        "retention_days": 90,
    }
    page.update(overrides)
    return page


class _Recorder:
    """A MockTransport handler that records each request and answers in turn."""

    def __init__(self, *responses: httpx.Response) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.responses.pop(0)


def _ok(payload: Any) -> httpx.Response:
    return httpx.Response(200, json=payload, headers={"Cache-Control": "no-store"})


def _session_client(handler: _Recorder, monkeypatch: pytest.MonkeyPatch) -> WorkspaceClient:
    constructor = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: constructor(transport=httpx.MockTransport(handler), **kw),
    )
    return WorkspaceClient.from_web_session(base_url="https://test.example", session_cookie=COOKIE)


async def _list(
    handler: _Recorder, monkeypatch: pytest.MonkeyPatch, **kwargs: Any
) -> OperationHistoryPage:
    async with _session_client(handler, monkeypatch) as client:
        return await client.list_operation_history(WS, **kwargs)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def test_models_and_aliases_are_exported():
    for name in (
        "OperationHistoryItem",
        "OperationHistoryPage",
        "OperationHistoryTargetType",
        "OperationHistoryOutcome",
    ):
        assert name in kagura_memory.__all__


def test_known_value_aliases_document_the_v0155_sets():
    assert get_args(OperationHistoryTargetType) == (
        "memory",
        "context",
        "member",
        "credential",
        "integration",
        "sleep",
    )
    assert get_args(OperationHistoryOutcome) == ("success", "failure", "denied")


def test_item_reads_every_field():
    item = OperationHistoryItem.model_validate(
        _item(
            subject_user_id="google_456",
            agent_id=AGENT,
            client_id="kmc_client_1",
        )
    )
    assert item.id == EVENT_A
    assert item.occurred_at == datetime(2026, 11, 1, 9, 30, tzinfo=UTC)
    assert item.context_id == CTX
    assert item.actor_user_id == "google_123" and item.actor_name == "Owner"
    assert item.subject_user_id == "google_456"
    assert item.agent_id == AGENT
    assert item.principal_kind == "api_key"
    assert item.credential_id == "akey_0123" and item.client_id == "kmc_client_1"
    assert item.surface == "mcp"
    assert item.operation == "memory.updated"
    assert item.target_type == "memory"
    assert item.target_id == "dddddddd-0000-0000-0000-000000000001"
    assert item.outcome == "success"
    assert item.count == 1
    assert item.changed_fields == ["summary", "tags"]


def test_item_nullable_fields_read_none_whether_null_or_absent():
    nullable = (
        "context_id",
        "actor_user_id",
        "actor_name",
        "subject_user_id",
        "agent_id",
        "credential_id",
        "client_id",
        "target_id",
    )
    as_null = OperationHistoryItem.model_validate(_item(**dict.fromkeys(nullable)))
    absent = OperationHistoryItem.model_validate(
        {k: v for k, v in _item().items() if k not in nullable}
    )
    for name in nullable:
        assert getattr(as_null, name) is None
        assert getattr(absent, name) is None


def test_item_keeps_values_a_newer_server_adds():
    """Value sets are plain ``str``; unknown keys do not fail parsing."""
    item = OperationHistoryItem.model_validate(
        _item(
            principal_kind="future_kind",
            surface="future_surface",
            operation="future.op",
            target_type="future_target",
            outcome="future_outcome",
            future_field={"x": 1},
        )
    )
    assert (item.principal_kind, item.surface, item.operation) == (
        "future_kind",
        "future_surface",
        "future.op",
    )
    assert (item.target_type, item.outcome) == ("future_target", "future_outcome")


@pytest.mark.parametrize(
    "missing",
    [
        "id",
        "occurred_at",
        "principal_kind",
        "surface",
        "operation",
        "target_type",
        "outcome",
        "count",
        "changed_fields",
    ],
)
def test_item_requires_the_fields_the_server_always_sends(missing: str):
    with pytest.raises(ValueError):
        OperationHistoryItem.model_validate({k: v for k, v in _item().items() if k != missing})


def test_page_defaults_and_required_fields():
    page = OperationHistoryPage.model_validate(
        {"items": [], "total": 0, "next_cursor": None, "retention_days": 90, "later": 1}
    )
    assert page.total_capped is False
    assert page.next_cursor is None
    for missing in ("items", "total", "retention_days"):
        with pytest.raises(ValueError):
            OperationHistoryPage.model_validate({k: v for k, v in _page().items() if k != missing})


# ---------------------------------------------------------------------------
# Session only
# ---------------------------------------------------------------------------


def _fail(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"no request may be sent, got {request.method} {request.url}")


@pytest.mark.asyncio
async def test_an_api_key_client_is_refused_before_any_request():
    client = WorkspaceClient(api_key="kagura_test")
    await client._client.aclose()
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(_fail))
    async with client:
        with pytest.raises(ValueError, match=r"requires WorkspaceClient\.from_web_session\(\)"):
            # The refusal comes first: even an invalid workspace id is not checked.
            await client.list_operation_history("not-a-uuid", limit=0)


@pytest.mark.asyncio
async def test_an_oauth_client_is_refused_before_any_request():
    def fake_auth(request):  # httpx accepts a callable auth
        return request

    client = WorkspaceClient(_oauth=fake_auth)  # type: ignore[arg-type]
    await client._client.aclose()
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(_fail))
    async with client:
        with pytest.raises(ValueError, match=r"requires WorkspaceClient\.from_web_session\(\)"):
            await client.list_operation_history(WS)


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_bare_call_sends_the_session_cookie_and_no_query(monkeypatch):
    handler = _Recorder(_ok(_page()))
    page = await _list(handler, monkeypatch)
    (request,) = handler.requests
    assert request.method == "GET"
    assert request.url.path == PATH
    assert request.url.query == b""
    assert request.headers["cookie"] == f"kagura_session={COOKIE}"
    assert "authorization" not in request.headers
    assert isinstance(page, OperationHistoryPage)
    assert page.total == 1 and page.retention_days == 90
    assert page.items[0].operation == "memory.updated"


@pytest.mark.asyncio
async def test_the_workspace_id_is_sent_canonical(monkeypatch):
    handler = _Recorder(_ok(_page()))
    async with _session_client(handler, monkeypatch) as client:
        await client.list_operation_history(WS.upper())
    assert handler.requests[0].url.path == PATH


@pytest.mark.asyncio
async def test_every_filter_combined_is_sent(monkeypatch):
    jst = timezone(timedelta(hours=9))
    since = datetime(2026, 11, 1, 9, 0, tzinfo=jst)
    until = datetime(2026, 11, 2, 0, 0, tzinfo=UTC)
    handler = _Recorder(_ok(_page()))
    await _list(
        handler,
        monkeypatch,
        query="  summary  ",
        context_id=CTX.upper(),
        actor="Owner",
        agent_id=AGENT.replace("-", ""),
        operation="memory.updated",
        target_type="memory",
        outcome="success",
        since=since,
        until=until,
        limit=100,
        cursor=CURSOR,
    )
    params = handler.requests[0].url.params
    assert dict(params) == {
        "query": "  summary  ",  # the server trims it
        "context_id": CTX,
        "actor": "Owner",
        "agent_id": AGENT,
        "operation": "memory.updated",
        "target_type": "memory",
        "outcome": "success",
        "since": "2026-11-01T09:00:00+09:00",
        "until": "2026-11-02T00:00:00+00:00",
        "limit": "100",
        "cursor": CURSOR,
    }
    # The offset survives URL encoding: "+" is percent-encoded, not a space.
    assert b"%2B09%3A00" in handler.requests[0].url.query


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["query", "actor", "operation", "cursor"])
@pytest.mark.parametrize("blank", [None, "", "   "])
async def test_unset_and_blank_text_filters_are_omitted(monkeypatch, name, blank):
    handler = _Recorder(_ok(_page()))
    await _list(handler, monkeypatch, **{name: blank})
    assert handler.requests[0].url.query == b""


@pytest.mark.asyncio
async def test_target_type_and_outcome_are_sent_as_given(monkeypatch):
    """Type hints only: a value a newer server adds needs no SDK release."""
    handler = _Recorder(_ok(_page()))
    await _list(handler, monkeypatch, target_type="future_target", outcome="future_outcome")
    params = handler.requests[0].url.params
    assert dict(params) == {"target_type": "future_target", "outcome": "future_outcome"}


@pytest.mark.asyncio
async def test_an_unknown_operation_is_left_to_the_server(monkeypatch):
    handler = _Recorder(
        httpx.Response(
            422,
            json={
                "error": "VAL-001",
                "message": "Invalid operation",
                "details": {"field": "operation"},
            },
        )
    )
    with pytest.raises(KaguraConnectionError, match="HTTP 422") as caught:
        await _list(handler, monkeypatch, operation="memory.future")
    assert handler.requests[0].url.params["operation"] == "memory.future"
    assert caught.value.code == "VAL-001"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "cap"), [("query", 120), ("actor", 255), ("operation", 40), ("cursor", 512)]
)
async def test_text_filters_at_their_cap_are_sent_and_over_it_refused(monkeypatch, name, cap):
    handler = _Recorder(_ok(_page()))
    async with _session_client(handler, monkeypatch) as client:
        await client.list_operation_history(WS, **{name: "x" * cap})
        with pytest.raises(ValueError, match=f"{name} must be at most {cap} characters"):
            await client.list_operation_history(WS, **{name: "x" * (cap + 1)})
    (request,) = handler.requests  # the refused call sent nothing
    assert request.url.params[name] == "x" * cap


@pytest.mark.asyncio
async def test_the_length_is_checked_on_the_value_as_sent(monkeypatch):
    """Surrounding spaces count: the server's ``max_length`` reads the raw value."""
    async with _session_client(_Recorder(), monkeypatch) as client:
        with pytest.raises(ValueError, match="query must be at most 120"):
            await client.list_operation_history(WS, query=" " + "x" * 120)


@pytest.mark.asyncio
async def test_a_refused_cursor_is_not_echoed(monkeypatch):
    secret_cursor = "c" * 513
    async with _session_client(_Recorder(), monkeypatch) as client:
        with pytest.raises(ValueError) as caught:
            await client.list_operation_history(WS, cursor=secret_cursor)
    assert secret_cursor not in str(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["query", "actor", "operation", "cursor"])
async def test_a_text_filter_must_be_a_str(monkeypatch, name):
    async with _session_client(_Recorder(), monkeypatch) as client:
        with pytest.raises(ValueError, match=f"{name} must be a string"):
            await client.list_operation_history(WS, **{name: 7})


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [0, 101, True, 2.0, "5"])
async def test_limit_outside_1_100_is_refused(monkeypatch, limit):
    async with _session_client(_Recorder(), monkeypatch) as client:
        with pytest.raises(ValueError, match="limit must be an integer between 1 and 100"):
            await client.list_operation_history(WS, limit=limit)


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [1, 100])
async def test_limit_bounds_are_sent(monkeypatch, limit):
    handler = _Recorder(_ok(_page()))
    await _list(handler, monkeypatch, limit=limit)
    assert handler.requests[0].url.params["limit"] == str(limit)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [{"context_id": "nope"}, {"agent_id": "nope"}],
    ids=["context_id", "agent_id"],
)
async def test_ids_must_be_uuids(monkeypatch, kwargs):
    (name,) = kwargs
    async with _session_client(_Recorder(), monkeypatch) as client:
        with pytest.raises(ValueError, match=f"{name} must be a UUID"):
            await client.list_operation_history(WS, **kwargs)
        with pytest.raises(ValueError, match="workspace_id must be a UUID"):
            await client.list_operation_history("nope")


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["since", "until"])
async def test_a_naive_datetime_is_refused(monkeypatch, name):
    """The server refuses a naive time; it is not taken as UTC here."""
    async with _session_client(_Recorder(), monkeypatch) as client:
        with pytest.raises(ValueError, match=f"{name} must be timezone-aware"):
            await client.list_operation_history(WS, **{name: datetime(2026, 11, 1)})


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["since", "until"])
async def test_a_time_filter_must_be_a_datetime(monkeypatch, name):
    async with _session_client(_Recorder(), monkeypatch) as client:
        with pytest.raises(ValueError, match=f"{name} must be a datetime"):
            await client.list_operation_history(WS, **{name: "2026-11-01T00:00:00Z"})


@pytest.mark.asyncio
async def test_since_after_until_is_refused(monkeypatch):
    jst = timezone(timedelta(hours=9))
    async with _session_client(_Recorder(), monkeypatch) as client:
        with pytest.raises(ValueError, match="since must not be after until"):
            await client.list_operation_history(
                WS,
                since=datetime(2026, 11, 1, 1, tzinfo=UTC),
                until=datetime(2026, 11, 1, 9, 30, tzinfo=jst),  # 00:30 UTC
            )


@pytest.mark.asyncio
async def test_equal_since_and_until_are_sent(monkeypatch):
    at = datetime(2026, 11, 1, tzinfo=UTC)
    handler = _Recorder(_ok(_page()))
    await _list(handler, monkeypatch, since=at, until=at)
    assert handler.requests[0].url.params["since"] == handler.requests[0].url.params["until"]


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_empty_page(monkeypatch):
    handler = _Recorder(_ok(_page(items=[], total=0)))
    page = await _list(handler, monkeypatch)
    assert page.items == [] and page.total == 0
    assert page.next_cursor is None and page.total_capped is False


@pytest.mark.asyncio
async def test_paging_round_trips_the_opaque_cursor(monkeypatch):
    handler = _Recorder(
        _ok(_page(items=[_item()], total=2, next_cursor=CURSOR)),
        _ok(_page(items=[_item(id=EVENT_B)], total=2, next_cursor=None)),
    )
    seen: list[str] = []
    async with _session_client(handler, monkeypatch) as client:
        cursor: str | None = None
        while True:
            page = await client.list_operation_history(
                WS, operation="memory.updated", cursor=cursor
            )
            seen += [item.id for item in page.items]
            cursor = page.next_cursor
            if cursor is None:
                break
    assert seen == [EVENT_A, EVENT_B]
    first, second = handler.requests
    assert "cursor" not in first.url.params
    assert second.url.params["cursor"] == CURSOR
    # The filters go again with every page: the server rechecks them.
    assert second.url.params["operation"] == "memory.updated"


@pytest.mark.asyncio
async def test_a_capped_total(monkeypatch):
    handler = _Recorder(_ok(_page(total=1000, total_capped=True, next_cursor=CURSOR)))
    page = await _list(handler, monkeypatch)
    assert page.total == 1000 and page.total_capped is True
    assert page.next_cursor == CURSOR


@pytest.mark.asyncio
async def test_retention_days_reports_the_add_on(monkeypatch):
    handler = _Recorder(_ok(_page(retention_days=455)))
    assert (await _list(handler, monkeypatch)).retention_days == 455


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"items": None, "total": 0, "next_cursor": None, "retention_days": 90},
        _page(total="many"),
        _page(items=[{"id": EVENT_A}]),
        _page(items=[_item(occurred_at="yesterday")]),
    ],
    ids=["array", "null-items", "bad-total", "partial-item", "bad-time"],
)
async def test_a_malformed_page_is_a_response_error(monkeypatch, payload):
    handler = _Recorder(_ok(payload))
    with pytest.raises(KaguraResponseError) as caught:
        await _list(handler, monkeypatch)
    assert caught.value.operation == "WorkspaceClient.list_operation_history"


@pytest.mark.asyncio
async def test_a_non_json_body_is_a_connection_error(monkeypatch):
    handler = _Recorder(httpx.Response(200, text="<html>maintenance</html>"))
    with pytest.raises(KaguraConnectionError, match="non-JSON body"):
        await _list(handler, monkeypatch)


# ---------------------------------------------------------------------------
# Refusals keep the server's typed envelope; none becomes an empty page
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_expired_session_is_an_auth_error(monkeypatch):
    handler = _Recorder(
        httpx.Response(401, json={"error": "AUTH-001", "message": "Session expired", "details": {}})
    )
    with pytest.raises(KaguraAuthError, match="Renew the signed-in web session"):
        await _list(handler, monkeypatch)


def _uniform_403() -> httpx.Response:
    """``PermissionService.check_workspace_admin``'s refusal at v0.155.0.

    ``AuthorizationError`` (403 ``AUTH-101``) with the uniform message and its
    ``details`` stripped by the exception handler (CWE-639): the same body for
    a member below admin, a non-member, and an unknown or deleted workspace.
    """
    return httpx.Response(
        403,
        json={"error": "AUTH-101", "message": "Insufficient permissions", "details": {}},
    )


@pytest.mark.asyncio
async def test_a_member_below_admin_is_refused(monkeypatch):
    handler = _Recorder(_uniform_403())
    with pytest.raises(KaguraConnectionError, match="HTTP 403: Insufficient permissions") as caught:
        await _list(handler, monkeypatch)
    assert caught.value.code == "AUTH-101"
    # The session path never adds the owner-API-key hint.
    assert "API key" not in str(caught.value)


@pytest.mark.asyncio
async def test_an_unknown_workspace_gets_the_same_uniform_403(monkeypatch):
    """The route answers no 404: an unknown workspace is the admin check's 403."""
    handler = _Recorder(_uniform_403())
    with pytest.raises(KaguraConnectionError, match="HTTP 403") as caught:
        await _list(handler, monkeypatch)
    assert not isinstance(caught.value, KaguraNotFoundError)
    assert caught.value.code == "AUTH-101"


@pytest.mark.asyncio
async def test_the_plan_gate_is_a_feature_error(monkeypatch):
    handler = _Recorder(
        httpx.Response(
            403,
            json={
                "error": "FEAT-001",
                # FeatureNotAvailableError.for_feature at v0.155.0 (OSS tier labels).
                "message": (
                    "Feature 'operation_history' not available on M plan. "
                    "Upgrade to L plan to access this feature."
                ),
                "details": {
                    "gate": "plan",
                    "feature": "operation_history",
                    "required_plan": "pro",
                    "required_plan_display": "L",
                    "current_plan": "basic",
                },
            },
        )
    )
    with pytest.raises(KaguraFeatureNotAvailableError) as caught:
        await _list(handler, monkeypatch)
    err = caught.value
    assert err.feature == "operation_history"
    assert err.required_plan == "pro" and err.current_plan == "basic"
    assert err.gate == "plan"
    assert err.details["required_plan_display"] == "L"


@pytest.mark.asyncio
async def test_a_refusal_is_sent_once(monkeypatch):
    """No retry and no second request: the server owns the history writes."""
    handler = _Recorder(httpx.Response(500, json={"error": "SYS-001", "message": "boom"}))
    with pytest.raises(KaguraConnectionError, match="HTTP 500"):
        await _list(handler, monkeypatch)
    assert len(handler.requests) == 1
