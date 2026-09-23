"""Typed quota, plan-gate and partial-rollback errors (#256).

The wire shapes below follow memory-cloud's error contract
(``docs/api-surface-1.0/error-responses.md``, "Gate refusals (#1644)"):

- v0.75.0+ stamps every plan/quota refusal with a ``gate`` descriptor —
  REST in ``details``, MCP as top-level envelope fields — and the SDK
  branches on it.
- Older servers (v0.68.0-v0.74.0) send only the legacy fields, so the SDK
  falls back to the error code: ``quota_exceeded`` with a frozen
  ``quota_type``, ``plan_required`` with ``required_plan``, ``FEAT-001``
  with ``feature``.
- A quota refusal with neither a ``gate`` nor a frozen ``quota_type`` (the
  1 MB memory-size guard) is a request limit no tier lifts, never a
  "wait until reset" quota.
"""

from __future__ import annotations

import copy
import pickle
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from kagura_memory import (
    KaguraFeatureNotAvailableError,
    KaguraPartialRollbackError,
    RollbackResult,
    RollbackSummary,
)
from kagura_memory._http import raise_for_kagura_status
from kagura_memory.client import KaguraClient
from kagura_memory.exceptions import (
    KaguraConnectionError,
    KaguraError,
    KaguraNotFoundError,
    KaguraQuotaError,
    KaguraRateLimitError,
    KaguraResponseError,
)
from kagura_memory.files_client import FilesClient
from kagura_memory.models import ResourceEventRequest
from kagura_memory.resource_client import ResourceClient
from kagura_memory.workspace_client import WorkspaceClient

WS = "11111111-2222-3333-4444-555555555555"
CTX = "22222222-3333-4444-5555-666666666666"


def _in_hours(hours: float) -> str:
    return (datetime.now(UTC) + timedelta(hours=hours)).isoformat()


# ---------------------------------------------------------------------------
# Exception classes
# ---------------------------------------------------------------------------


def test_new_errors_are_kagura_errors():
    assert issubclass(KaguraFeatureNotAvailableError, KaguraError)
    assert issubclass(KaguraPartialRollbackError, KaguraError)
    assert issubclass(KaguraQuotaError, KaguraError)


def test_quota_error_keeps_its_positional_signature():
    """The pre-#256 ``KaguraQuotaError(message, retry_after)`` call still works."""
    err = KaguraQuotaError("Quota exceeded.", 60)
    assert str(err) == "Quota exceeded."
    assert err.retry_after == 60
    assert err.quota_type is None
    assert err.limit is None
    assert err.used_today is None
    assert err.resets_at is None
    assert err.gate is None


def test_quota_error_derives_retry_after_from_resets_at():
    resets_at = datetime.now(UTC) + timedelta(hours=2)
    err = KaguraQuotaError("over", resets_at=resets_at)
    assert err.retry_after is not None
    assert 7100 < err.retry_after <= 7200


def test_quota_error_reset_in_the_past_means_retry_now():
    err = KaguraQuotaError("over", resets_at=datetime.now(UTC) - timedelta(minutes=5))
    assert err.retry_after == 0


def test_quota_error_naive_resets_at_is_read_as_utc():
    naive = (datetime.now(UTC) + timedelta(hours=1)).replace(tzinfo=None)
    err = KaguraQuotaError("over", resets_at=naive)
    assert err.resets_at is not None and err.resets_at.tzinfo is not None
    assert err.retry_after is not None and 3500 < err.retry_after <= 3600


def test_quota_error_explicit_retry_after_wins():
    err = KaguraQuotaError("over", 30, resets_at=datetime.now(UTC) + timedelta(hours=2))
    assert err.retry_after == 30


def test_feature_error_attributes_default_to_none():
    err = KaguraFeatureNotAvailableError("nope")
    assert (err.feature, err.required_plan, err.required_plan_display) == (None, None, None)
    assert (err.current_plan, err.gate) == (None, None)
    assert err.details == {}


def _round_trips(err: KaguraError) -> list[KaguraError]:
    # Unpickles only bytes this test just produced from its own object.
    return [pickle.loads(pickle.dumps(err)), copy.copy(err)]


def test_quota_error_pickles_with_its_attributes():
    """Exceptions cross process boundaries (multiprocessing, task queues)."""
    resets_at = datetime(2099, 1, 2, tzinfo=UTC)
    err = KaguraQuotaError(
        "over",
        quota_type="memories_per_day",
        limit=5,
        current=5,
        resets_at=resets_at,
        required_plan="pro",
        details={"requested": 1},
    )
    for clone in _round_trips(err):
        assert isinstance(clone, KaguraQuotaError)
        assert str(clone) == "over"
        assert (clone.quota_type, clone.limit, clone.current) == ("memories_per_day", 5, 5)
        assert clone.resets_at == resets_at
        assert clone.retry_after == err.retry_after
        assert clone.required_plan == "pro"
        assert clone.details == {"requested": 1}


def test_feature_error_pickles_with_its_attributes():
    err = KaguraFeatureNotAvailableError("nope", feature="resources", gate="plan")
    for clone in _round_trips(err):
        assert isinstance(clone, KaguraFeatureNotAvailableError)
        assert (clone.feature, clone.gate) == ("resources", "plan")


def test_partial_rollback_error_pickles_with_its_summary():
    summary = RollbackSummary(edges_deleted=2, errors=["Action 1 (merge): db error"])
    err = KaguraPartialRollbackError("partial", report_id="rid-1", summary=summary)
    for clone in _round_trips(err):
        assert isinstance(clone, KaguraPartialRollbackError)
        assert str(clone) == "partial"
        assert clone.report_id == "rid-1"
        assert clone.summary == summary


# ---------------------------------------------------------------------------
# MCP envelopes → KaguraClient
# ---------------------------------------------------------------------------


def _client() -> KaguraClient:
    client = KaguraClient(api_key="test", mcp_url="https://test.com/mcp")
    client._session_id = "pre-set-session"
    return client


async def _remember_with(result: dict[str, Any]) -> None:
    client = _client()
    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = result
            await client.remember(context_id=CTX, summary="s", content="c")
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_mcp_quota_exceeded_with_gate_descriptor():
    """memory-cloud v0.75.0+: ``gate`` plus the legacy daily-quota fields."""
    resets_at = _in_hours(3)
    with pytest.raises(KaguraQuotaError, match=r"remember failed \(quota_exceeded\)") as exc:
        await _remember_with(
            {
                "status": "error",
                "error": "quota_exceeded",
                "message": f"Daily memory limit reached. Resets at {resets_at}.",
                "gate": "quota",
                "quota_type": "memories_per_day",
                "current": 500,
                "limit": 500,
                "used_today": 500,
                "requested": 1,
                "resets_at": resets_at,
                "required_plan": "pro",
                "required_plan_display": "L",
                "current_plan": "basic",
            }
        )
    err = exc.value
    assert err.gate == "quota"
    assert err.quota_type == "memories_per_day"
    assert err.limit == 500
    assert err.current == 500
    assert err.used_today == 500
    assert err.resets_at == datetime.fromisoformat(resets_at)
    assert err.retry_after is not None and 10700 < err.retry_after <= 10800
    assert err.required_plan == "pro"
    assert err.required_plan_display == "L"
    assert err.current_plan == "basic"
    # Fields without an attribute stay reachable; the envelope framing does not.
    assert err.details["requested"] == 1
    assert err.details["quota_type"] == "memories_per_day"
    assert not {"status", "error", "message"} & err.details.keys()


@pytest.mark.asyncio
async def test_mcp_daily_call_cap_is_a_quota_error():
    """The in-band daily MCP call cap: ``rate_limit_exceeded`` with ``used_today`` /
    ``daily_limit`` and no ``gate``, resetting at midnight UTC."""
    with pytest.raises(KaguraQuotaError, match=r"remember failed \(rate_limit_exceeded\)") as exc:
        await _remember_with(
            {
                "status": "error",
                "error": "rate_limit_exceeded",
                "message": "Daily MCP call limit reached (100/100). Resets at midnight UTC.",
                "used_today": 100,
                "daily_limit": 100,
                "help": "Use get_usage() to check your current quota.",
            }
        )
    err = exc.value
    assert err.quota_type == "api_mcp_daily"
    assert (err.current, err.used_today, err.limit) == (100, 100, 100)
    tomorrow = datetime.now(UTC).date() + timedelta(days=1)
    assert err.resets_at == datetime(tomorrow.year, tomorrow.month, tomorrow.day, tzinfo=UTC)
    assert err.retry_after is not None and 0 < err.retry_after <= 86400
    # ``details`` is what the server sent, not the names the SDK filled in.
    assert err.details == {
        "used_today": 100,
        "daily_limit": 100,
        "help": "Use get_usage() to check your current quota.",
    }


def test_mcp_transport_daily_quota_429_stays_a_rate_limit_error():
    """Deliberately unchanged: the middleware's HTTP 429 on ``/mcp`` keeps raising
    KaguraRateLimitError, so existing ``except KaguraRateLimitError`` handlers hold."""
    request = httpx.Request("POST", "https://test.com/mcp")
    body = {
        "error": "QUOTA-001",
        "message": "Daily MCP quota exceeded: 101/100. Resets at midnight UTC.",
        "details": {"gate": "quota", "quota_type": "api_mcp_daily", "retry_after": 86400},
    }
    response = httpx.Response(429, json=body, headers={"Retry-After": "86400"}, request=request)
    with pytest.raises(KaguraRateLimitError) as exc:
        raise_for_kagura_status(httpx.HTTPStatusError("429", request=request, response=response))
    assert exc.value.retry_after == 86400
    assert not isinstance(exc.value, KaguraQuotaError)


@pytest.mark.asyncio
async def test_mcp_quota_exceeded_legacy_server_without_gate():
    """memory-cloud v0.68.0-v0.74.0: no ``gate``; the frozen ``quota_type`` decides."""
    with pytest.raises(KaguraQuotaError) as exc:
        await _remember_with(
            {
                "status": "error",
                "error": "quota_exceeded",
                "message": "Daily memory limit reached.",
                "quota_type": "memories_per_day",
                "limit": 100,
                "used_today": 100,
                "requested": 1,
                "resets_at": "2099-01-02T00:00:00Z",
            }
        )
    err = exc.value
    assert err.gate is None
    assert (err.quota_type, err.limit, err.used_today) == ("memories_per_day", 100, 100)
    assert err.resets_at == datetime(2099, 1, 2, tzinfo=UTC)
    assert err.retry_after is not None and err.retry_after > 0


@pytest.mark.asyncio
async def test_mcp_untyped_quota_exceeded_is_not_a_tier_quota():
    """The 1 MB memory-size guard: no ``gate``, no ``quota_type`` — waiting cannot help."""
    with pytest.raises(KaguraError, match=r"quota_exceeded.*1MB") as exc:
        await _remember_with(
            {
                "status": "error",
                "error": "quota_exceeded",
                "message": "Memory size 2,000,000 bytes exceeds limit 1,048,576 bytes (1MB).",
            }
        )
    assert not isinstance(exc.value, KaguraQuotaError)


def test_mcp_quota_exceeded_with_a_retry_window_is_a_quota_error():
    """``ingest_events``' events-per-hour ceiling names ``retry_after_seconds``."""
    with pytest.raises(KaguraQuotaError) as exc:
        KaguraClient._raise_for_mcp_error(
            {
                "status": "error",
                "error": "quota_exceeded",
                "message": "Event quota exceeded (1000/hour).",
                "retry_after_seconds": 3600,
            },
            "ingest_events",
        )
    assert exc.value.retry_after == 3600
    assert exc.value.quota_type is None


def test_mcp_count_cap_carries_no_retry_after():
    """The server's context-cap envelope (``create_context`` past the SDK's
    pre-check, ``setup_resource``): a count cap, not a time window."""
    with pytest.raises(KaguraQuotaError) as exc:
        KaguraClient._raise_for_mcp_error(
            {
                "status": "error",
                "error": "quota_exceeded",
                "message": "Context limit reached.",
                "help": "Delete unused contexts or upgrade your plan.",
                "gate": "quota",
                "quota_type": "contexts",
                "current": 1,
                "limit": 1,
                "required_plan": "basic",
                "required_plan_display": "M",
                "current_plan": "free",
            },
            "create_context",
        )
    err = exc.value
    assert (err.quota_type, err.current, err.limit) == ("contexts", 1, 1)
    assert err.retry_after is None
    assert err.resets_at is None


def test_mcp_gate_wins_over_the_error_code():
    """``setup_connector``'s seat cap keeps its ``CONNECTOR-001`` code."""
    with pytest.raises(KaguraQuotaError) as exc:
        KaguraClient._raise_for_mcp_error(
            {
                "status": "error",
                "error": "CONNECTOR-001",
                "message": "Connector limit reached.",
                "gate": "quota",
                "quota_type": "connectors",
                "current": 2,
                "limit": 2,
                "max_connectors": 2,
                "active_connectors": 2,
            },
            "setup_connector",
        )
    assert exc.value.quota_type == "connectors"


def test_mcp_legacy_connector_cap_is_a_quota_error_by_code():
    """Before #1644, ``CONNECTOR-001`` named exactly one cap — the code is enough."""
    with pytest.raises(KaguraQuotaError):
        KaguraClient._raise_for_mcp_error(
            {
                "status": "error",
                "error": "CONNECTOR-001",
                "message": "Connector limit reached.",
                "max_connectors": 2,
                "active_connectors": 2,
            },
            "setup_connector",
        )


@pytest.mark.asyncio
async def test_mcp_plan_required_with_gate_descriptor():
    client = _client()
    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = {
                "status": "error",
                "error": "plan_required",
                "message": "Feature 'resources' not available on L plan.",
                "gate": "plan",
                "feature": "resources",
                "required_plan": "promax",
                "required_plan_display": "XL",
                "current_plan": "pro",
            }
            with pytest.raises(
                KaguraFeatureNotAvailableError, match=r"setup_resource failed \(plan_required\)"
            ) as exc:
                await client.setup_resource(resource_id="products")
    finally:
        await client.close()
    err = exc.value
    assert err.gate == "plan"
    assert err.feature == "resources"
    assert err.required_plan == "promax"
    assert err.required_plan_display == "XL"
    assert err.current_plan == "pro"
    assert err.details["feature"] == "resources"


def _contexts_at(*, count: Any, limit: Any) -> dict[str, Any]:
    return {
        "status": "success",
        "contexts": [],
        "count": count,
        "limit": limit,
        "can_create": False,
    }


@pytest.mark.asyncio
async def test_create_context_precheck_carries_the_quota_fields():
    """The pre-check refuses before the server does, so it fills the fields itself."""
    client = _client()
    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = _contexts_at(count=3, limit=3)
            with pytest.raises(KaguraQuotaError, match=r"Context limit reached \(3/3\)") as exc:
                await client.create_context(name="over-limit")
            mock.assert_awaited_once()
    finally:
        await client.close()
    err = exc.value
    assert (err.quota_type, err.current, err.limit) == ("contexts", 3, 3)
    assert err.retry_after is None


@pytest.mark.asyncio
async def test_create_context_precheck_drops_non_integer_counts():
    client = _client()
    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = _contexts_at(count=True, limit="3")
            with pytest.raises(KaguraQuotaError) as exc:
                await client.create_context(name="over-limit")
    finally:
        await client.close()
    assert exc.value.quota_type == "contexts"
    assert (exc.value.current, exc.value.limit) == (None, None)


@pytest.mark.asyncio
async def test_create_context_precheck_defers_a_failed_quota_lookup_to_the_server():
    """``limit: 0`` + ``can_create: false`` is list_contexts' lookup-failure signal,
    not a cap: the server's own check decides, and its typed refusal comes through."""
    client = _client()
    cap = {
        "status": "error",
        "error": "quota_exceeded",
        "message": "Context limit reached.",
        "gate": "quota",
        "quota_type": "contexts",
        "current": 1,
        "limit": 1,
        "required_plan": "basic",
        "required_plan_display": "M",
    }
    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.side_effect = [_contexts_at(count=1, limit=0), cap]
            with pytest.raises(KaguraQuotaError, match=r"create_context failed") as exc:
                await client.create_context(name="new-ctx")
            assert mock.call_args_list[1].args[0] == "create_context"
    finally:
        await client.close()
    assert exc.value.required_plan_display == "M"


@pytest.mark.asyncio
async def test_create_context_precheck_lookup_failure_lets_the_create_through():
    client = _client()
    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.side_effect = [_contexts_at(count=1, limit=0), {"id": "uuid-1", "name": "c"}]
            result = await client.create_context(name="c")
    finally:
        await client.close()
    assert result["id"] == "uuid-1"


def test_mcp_plan_required_legacy_server():
    """memory-cloud v0.68.0-v0.74.0 sent ``required_plan`` only."""
    with pytest.raises(KaguraFeatureNotAvailableError) as exc:
        KaguraClient._raise_for_mcp_error(
            {
                "status": "error",
                "error": "plan_required",
                "message": "Upgrade to XL.",
                "required_plan": "promax",
            },
            "update_context",
        )
    err = exc.value
    assert err.required_plan == "promax"
    assert err.feature is None
    assert err.gate is None


def test_mcp_plan_required_with_no_upgrade_path():
    """``required_plan`` is ``null`` when no tier carries the feature."""
    with pytest.raises(KaguraFeatureNotAvailableError) as exc:
        KaguraClient._raise_for_mcp_error(
            {
                "status": "error",
                "error": "plan_required",
                "message": "Public contexts are not available.",
                "gate": "plan",
                "feature": "public_contexts",
                "required_plan": None,
                "required_plan_display": None,
                "current_plan": "pro",
            },
            "update_context",
        )
    assert exc.value.required_plan is None
    assert exc.value.feature == "public_contexts"


def test_mcp_feature_not_available_allowlist_gate():
    with pytest.raises(KaguraFeatureNotAvailableError) as exc:
        KaguraClient._raise_for_mcp_error(
            {
                "status": "error",
                "error": "feature_not_available",
                "message": "Memory analysis is not yet enabled for this workspace.",
                "gate": "allowlist",
                "feature": "memory_analysis",
            },
            "analyze_context",
        )
    assert exc.value.gate == "allowlist"
    assert exc.value.required_plan is None


def test_mcp_deployment_gate_on_a_validation_error_code():
    """Branch on ``gate``, not the code: the managed-LLM refusal stays ``validation_error``."""
    with pytest.raises(KaguraFeatureNotAvailableError) as exc:
        KaguraClient._raise_for_mcp_error(
            {
                "status": "error",
                "error": "validation_error",
                "message": "No managed model on this deployment.",
                "gate": "deployment",
                "feature": "managed_llm",
            },
            "analyze_context",
        )
    assert exc.value.gate == "deployment"


def test_mcp_unknown_gate_kind_falls_back_to_the_code():
    with pytest.raises(KaguraError) as exc:
        KaguraClient._raise_for_mcp_error(
            {
                "status": "error",
                "error": "validation_error",
                "message": "bad",
                "gate": "something_new",
            },
            "remember",
        )
    assert type(exc.value) is KaguraError


def test_mcp_malformed_gate_fields_are_dropped_not_raised():
    """Drifted field types never mask the refusal itself."""
    with pytest.raises(KaguraQuotaError) as exc:
        KaguraClient._raise_for_mcp_error(
            {
                "status": "error",
                "error": "quota_exceeded",
                "message": "over",
                "gate": "quota",
                "quota_type": 7,
                "limit": "100",
                "current": True,
                "resets_at": "not-a-date",
            },
            "remember",
        )
    err = exc.value
    assert (err.quota_type, err.limit, err.current, err.resets_at) == (None, None, None, None)
    assert err.retry_after is None


def test_mcp_not_found_mapping_is_unchanged():
    with pytest.raises(KaguraNotFoundError):
        KaguraClient._raise_for_mcp_error(
            {"status": "error", "error": "context_not_found", "message": "gone"}, "recall"
        )


# ---------------------------------------------------------------------------
# rollback_sleep_run
# ---------------------------------------------------------------------------

_UNREVERSIBLE = "shadow merge m-1 → m-2 not reversed — the edge was changed by a later writer"

# A partial rollback: each unreversible merge is also an ``errors`` entry.
_FULL_SUMMARY = {
    "edges_deleted": 3,
    "merges_reversed": 1,
    "merges_unreversible": 1,
    "importance_restored": 4,
    "promotions_reversed": 1,
    "importance_kept": 1,
    "promotions_kept": 5,
    "archives_restored": 2,
    "errors": [_UNREVERSIBLE, "Action 42 (merge): db error"],
}


def test_rollback_summary_round_trips_the_new_counters():
    summary = RollbackSummary.model_validate(_FULL_SUMMARY)
    assert summary.merges_unreversible == 1
    assert summary.importance_kept == 1
    assert summary.promotions_kept == 5
    assert summary.model_dump() == _FULL_SUMMARY


def test_rollback_summary_new_counters_default_to_zero():
    summary = RollbackSummary.model_validate({"edges_deleted": 1})
    assert (summary.merges_unreversible, summary.importance_kept, summary.promotions_kept) == (
        0,
        0,
        0,
    )


@pytest.mark.asyncio
async def test_rollback_success_carries_the_new_counters():
    client = _client()
    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            # A success never carries an unreversible merge (that is an error).
            mock.return_value = {
                "status": "success",
                "report_id": "rid-9",
                "rollback_summary": {**_FULL_SUMMARY, "merges_unreversible": 0, "errors": []},
            }
            result = await client.rollback_sleep_run(context_id=CTX, report_id="rid-9")
    finally:
        await client.close()
    assert isinstance(result, RollbackResult)
    assert result.rollback_summary.merges_unreversible == 0
    assert result.rollback_summary.importance_kept == 1
    assert result.rollback_summary.promotions_kept == 5


@pytest.mark.asyncio
async def test_partial_rollback_raises_with_the_summary():
    client = _client()
    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = {
                "status": "error",
                "error": "partial_rollback",
                "message": "Rollback completed with 1 error(s).",
                "report_id": "rid-9",
                "rollback_summary": _FULL_SUMMARY,
            }
            with pytest.raises(
                KaguraPartialRollbackError, match=r"rollback_sleep_run failed \(partial_rollback\)"
            ) as exc:
                await client.rollback_sleep_run(context_id=CTX, report_id="rid-9")
    finally:
        await client.close()
    err = exc.value
    assert err.report_id == "rid-9"
    assert isinstance(err.summary, RollbackSummary)
    assert err.summary.model_dump() == _FULL_SUMMARY
    assert err.summary.merges_unreversible == 1
    assert err.summary.errors == [_UNREVERSIBLE, "Action 42 (merge): db error"]


@pytest.mark.parametrize(
    "extra",
    [{"rollback_summary": {"edges_deleted": "three"}}, {}],
    ids=["drifted-summary", "missing-summary"],
)
def test_partial_rollback_with_an_unreadable_summary_is_still_a_partial_rollback(extra):
    """The partial reversal is committed either way — drift must not hide it."""
    with pytest.raises(
        KaguraPartialRollbackError,
        match=r"failed \(partial_rollback\): Rollback completed.*could not be read",
    ) as exc:
        KaguraClient._raise_for_mcp_error(
            {
                "status": "error",
                "error": "partial_rollback",
                "message": "Rollback completed with 1 error(s).",
                "report_id": "rid-9",
                **extra,
            },
            "rollback_sleep_run",
        )
    err = exc.value
    assert err.report_id == "rid-9"
    assert err.summary is None
    assert isinstance(err.__cause__, KaguraResponseError)


# ---------------------------------------------------------------------------
# REST envelopes → KaguraRestClient subclasses
# ---------------------------------------------------------------------------


def _rest(cls, handler):
    client = cls(api_key="kagura_test", base_url="https://test.com")
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer kagura_test"},
    )
    return client


def _respond(status: int, code: str, message: str, details: dict[str, Any], **headers: str):
    body = {"error": code, "message": message, "details": details}
    return lambda _req: httpx.Response(status, json=body, headers=headers)


_TOKEN_CAP_DETAILS = {
    "gate": "quota",
    "quota_type": "resource_tokens",
    "current": 3,
    "limit": 3,
    "required_plan": "promax",
    "required_plan_display": "XL",
    "current_plan": "pro",
    "feature": "resources",
    "resets_at": None,
}


@pytest.mark.asyncio
async def test_rest_token_cap_is_a_quota_error_not_a_connection_error():
    """v0.75.0: ``POST /resource-tokens`` at the cap answers 403 ``QUOTA-001``."""
    msg = "Token limit reached. Your L plan allows 3 active tokens."
    async with _rest(ResourceClient, _respond(403, "QUOTA-001", msg, _TOKEN_CAP_DETAILS)) as c:
        with pytest.raises(KaguraQuotaError, match="Token limit reached") as exc:
            await c.create_token("products")
    err = exc.value
    assert (err.gate, err.quota_type, err.current, err.limit) == (
        "quota",
        "resource_tokens",
        3,
        3,
    )
    assert err.feature == "resources"
    assert err.required_plan_display == "XL"
    assert err.retry_after is None
    assert err.details == _TOKEN_CAP_DETAILS


@pytest.mark.asyncio
async def test_rest_embedding_spend_cap_keeps_its_usd_fields_in_details():
    """``QUOTA-002`` has no count pair: its numbers live only in ``details``."""
    details = {
        "gate": "quota",
        "quota_type": "embedding_spend_daily",
        "period": "daily",
        "cap_usd": 1.0,
        "current_usd": 1.2,
    }
    handler = _respond(429, "QUOTA-002", "Daily embedding spend cap reached.", details)
    async with _rest(ResourceClient, handler) as c:
        with pytest.raises(KaguraQuotaError) as exc:
            await c.list_tokens()
    err = exc.value
    assert (err.limit, err.current) == (None, None)
    assert (err.details["period"], err.details["cap_usd"], err.details["current_usd"]) == (
        "daily",
        1.0,
        1.2,
    )


@pytest.mark.asyncio
async def test_rest_feat_001_with_gate_descriptor():
    details = {
        "gate": "plan",
        "feature": "resources",
        "required_plan": "promax",
        "required_plan_display": "XL",
        "current_plan": "basic",
    }
    msg = "Feature 'resources' not available on M plan. Upgrade to XL plan."
    async with _rest(ResourceClient, _respond(403, "FEAT-001", msg, details)) as c:
        with pytest.raises(KaguraFeatureNotAvailableError, match="Upgrade to XL") as exc:
            await c.create_token("products")
    err = exc.value
    assert (err.gate, err.feature, err.required_plan) == ("plan", "resources", "promax")
    assert (err.required_plan_display, err.current_plan) == ("XL", "basic")


@pytest.mark.asyncio
async def test_rest_feat_001_legacy_server_without_gate():
    """memory-cloud v0.68.0-v0.74.0: ``details`` carried ``feature`` only."""
    handler = _respond(403, "FEAT-001", "Upgrade to XL.", {"feature": "resources"})
    async with _rest(ResourceClient, handler) as c:
        with pytest.raises(KaguraFeatureNotAvailableError) as exc:
            await c.create_token("products")
    assert exc.value.feature == "resources"
    assert exc.value.gate is None
    assert exc.value.required_plan is None


@pytest.mark.asyncio
async def test_rest_legacy_placeholder_403_keeps_its_mapping():
    """Pre-v0.75.0 token cap: ``HTTP-403`` with no details — nothing to type it by."""
    handler = _respond(403, "HTTP-403", "Token limit reached.", {})
    async with _rest(ResourceClient, handler) as c:
        with pytest.raises(KaguraConnectionError, match="HTTP 403: Token limit reached"):
            await c.create_token("products")


@pytest.mark.asyncio
async def test_rest_rate_limit_429_keeps_its_mapping():
    """``RATE-001`` carries no gate: still a KaguraQuotaError from the 429 hook."""
    handler = _respond(
        429,
        "RATE-001",
        "Rate limit exceeded: 61/60 requests per minute",
        {"retry_after": 60, "limit": 60, "remaining": 0},
        **{"Retry-After": "60"},
    )
    async with _rest(ResourceClient, handler) as c:
        with pytest.raises(KaguraQuotaError) as exc:
            event = ResourceEventRequest(op="upsert", doc_id="d", version=1, payload={"x": 1})
            await c.ingest_event("res", "key", event)
    assert exc.value.retry_after == 60
    assert exc.value.gate is None


@pytest.mark.asyncio
async def test_rest_daily_api_quota_uses_retry_after_header():
    details = {"gate": "quota", "quota_type": "api_rest_daily", "retry_after": 86400}
    handler = _respond(
        429, "QUOTA-001", "Daily REST quota exceeded.", details, **{"Retry-After": "86400"}
    )
    async with _rest(ResourceClient, handler) as c:
        with pytest.raises(KaguraQuotaError, match="Daily REST quota exceeded") as exc:
            await c.list_tokens()
    assert exc.value.retry_after == 86400
    assert exc.value.quota_type == "api_rest_daily"


@pytest.mark.asyncio
async def test_rest_legacy_daily_quota_without_quota_type_is_still_a_quota_error():
    """Pre-v0.75.0 middleware replaced ``details`` with ``{retry_after}``."""
    handler = _respond(
        429,
        "QUOTA-001",
        "Daily REST quota exceeded.",
        {"retry_after": 86400},
        **{"Retry-After": "86400"},
    )
    async with _rest(ResourceClient, handler) as c:
        with pytest.raises(KaguraQuotaError) as exc:
            await c.list_tokens()
    assert exc.value.retry_after == 86400


@pytest.mark.asyncio
async def test_rest_untyped_quota_001_is_not_a_quota_error():
    """No ``gate``, no frozen ``quota_type``, no retry window: not a tier quota."""
    handler = _respond(429, "QUOTA-001", "Memory size exceeds 1MB.", {"quota_type": None})
    async with _rest(ResourceClient, handler) as c:
        with pytest.raises(KaguraError, match="Memory size exceeds 1MB") as exc:
            await c.list_tokens()
    assert not isinstance(exc.value, KaguraQuotaError)


@pytest.mark.asyncio
async def test_rest_typed_quota_without_gate_is_a_quota_error():
    """A server predating #1644 that already named a frozen ``quota_type``."""
    handler = _respond(429, "QUOTA-001", "Agent limit reached.", {"quota_type": "agents"})
    async with _rest(ResourceClient, handler) as c:
        with pytest.raises(KaguraQuotaError) as exc:
            await c.list_tokens()
    assert exc.value.quota_type == "agents"


@pytest.mark.asyncio
async def test_rest_deployment_gate_on_a_422():
    """The managed-LLM refusal stays ``VAL-001``/422 but carries a gate."""
    handler = _respond(
        422,
        "VAL-001",
        "No managed model on this deployment.",
        {"gate": "deployment", "feature": "managed_llm"},
    )
    async with _rest(ResourceClient, handler) as c:
        with pytest.raises(KaguraFeatureNotAvailableError) as exc:
            await c.list_tokens()
    assert exc.value.gate == "deployment"


@pytest.mark.asyncio
async def test_rest_gate_message_with_credential_markers_is_dropped():
    handler = _respond(403, "FEAT-001", "echo Bearer kagura_secret", {"feature": "resources"})
    async with _rest(ResourceClient, handler) as c:
        with pytest.raises(KaguraFeatureNotAvailableError) as exc:
            await c.create_token("products")
    assert "kagura_secret" not in str(exc.value)
    assert str(exc.value) == "HTTP 403"


@pytest.mark.asyncio
async def test_workspace_invite_without_team_invitations_feature():
    """v0.75.0: ``POST /invitations`` without the feature is ``FEAT-001``, not ``HTTP-403``."""
    details = {
        "gate": "plan",
        "feature": "team_invitations",
        "required_plan": "pro",
        "required_plan_display": "L",
        "current_plan": "basic",
    }
    msg = "Feature 'team_invitations' not available on M plan. Upgrade to L plan."
    async with _rest(WorkspaceClient, _respond(403, "FEAT-001", msg, details)) as c:
        with pytest.raises(KaguraFeatureNotAvailableError, match="Upgrade to L plan") as exc:
            await c.create_invitation(WS, "a@b.com", role="admin")
    assert exc.value.feature == "team_invitations"
    assert exc.value.required_plan_display == "L"


@pytest.mark.asyncio
async def test_workspace_invite_at_the_seat_cap():
    """v0.75.0: the member cap is ``QUOTA-001`` with ``quota_type: "members"``."""
    details = {
        "gate": "quota",
        "quota_type": "members",
        "current": 5,
        "limit": 5,
        "required_plan": "pro",
        "required_plan_display": "L",
        "current_plan": "basic",
    }
    msg = "Member limit reached (5 seats)."
    async with _rest(WorkspaceClient, _respond(429, "QUOTA-001", msg, details)) as c:
        with pytest.raises(KaguraQuotaError, match="Member limit reached") as exc:
            await c.create_invitation(WS, "a@b.com", role="admin")
    err = exc.value
    assert (err.quota_type, err.current, err.limit) == ("members", 5, 5)
    assert err.retry_after is None


@pytest.mark.asyncio
async def test_files_feature_gate_skips_the_workspace_mismatch_hint():
    handler = _respond(
        403, "FEAT-001", "Storage is not on your plan.", {"gate": "plan", "feature": "files"}
    )
    async with _rest(FilesClient, handler) as c:
        with pytest.raises(KaguraFeatureNotAvailableError, match="Storage is not on your plan"):
            await c.list(context_id=CTX)
