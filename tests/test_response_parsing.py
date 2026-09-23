"""Response parsing against a newer server (#250).

Two contracts, pinned on every client surface:

1. **Forward tolerance.** Server-growing value sets — the Sleep run status
   and the indexer job status / skipped reason — are typed ``str`` on the
   response models, so a value added by a newer server parses instead of
   failing the whole call. The ``Literal`` aliases stay as the documented
   known-value sets and must track memory-cloud (v0.75.0 here).
2. **Drift wrapping.** A 2xx payload that still fails validation raises
   :class:`KaguraResponseError` (a ``KaguraError``) naming the operation,
   never a raw ``pydantic.ValidationError``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, get_args
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from pydantic import ValidationError

from kagura_memory import (
    AgentsClient,
    FilesClient,
    IndexerJobStatus,
    IndexerSkippedReason,
    IndexerStatusResponse,
    KaguraClient,
    KaguraError,
    KaguraResponseError,
    ResourceClient,
    RollbackResult,
    SleepReport,
    SleepReportDetail,
    SleepRunStatus,
    WorkspaceClient,
)
from kagura_memory._http import parse_response
from kagura_memory.secrets.client import SecretClient
from tests.conftest import sleep_report_summary_dict

AGENT = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
WS = "11111111-2222-3333-4444-555555555555"


def _indexer_payload(**metrics: Any) -> dict[str, Any]:
    """Server-shaped ``indexer-status`` body (memory-cloud resource_indexer.py)."""
    return {
        "resource_id": "products",
        "state": {
            "job_status": "idle",
            "last_run_at": "2026-09-20T00:00:00Z",
            "next_run_at": None,
            "active_version": 1,
            "last_offset": 42,
            "lag_seconds": 3.0,
            "metrics": {"applied_upserts": 0, "applied_deletes": 0, "errors": 0, **metrics},
        },
        "recent_events": [],
    }


def _detail_payload(**overrides: Any) -> dict[str, Any]:
    """Flattened ``get_sleep_report`` shape (``_report_to_detail`` + actions)."""
    return {
        **sleep_report_summary_dict("rid-d"),
        "memories_flagged": 0,
        "embedding_calls_made": 0,
        "error_message": None,
        "edge_discovery_result": None,
        "dedup_result": None,
        "merge_retention_result": None,
        "importance_result": None,
        "consolidation_result": None,
        "reindex_result": None,
        "actions": [],
        "action_count": 0,
        **overrides,
    }


# ---------------------------------------------------------------------------
# Known-value aliases track the server
# ---------------------------------------------------------------------------


def test_sleep_run_status_alias_tracks_server_check_constraint():
    # memory-cloud backend/src/models/sleep.py ``valid_sleep_report_status``;
    # ``degraded`` since v0.43.0 (#1183).
    assert set(get_args(SleepRunStatus)) == {
        "running",
        "completed",
        "degraded",
        "failed",
        "cancelled",
        "rolled_back",
    }


def test_indexer_skipped_reason_alias_tracks_server_enum():
    # memory-cloud backend/src/api/routes/resource_indexer.py
    # ``IndexerSkippedReason``; ``memories_per_day_exceeded`` since v0.68.0 (#1549).
    assert set(get_args(IndexerSkippedReason)) == {
        "no_pending_events",
        "schema_not_found",
        "context_not_found",
        "empty_valid_points",
        "resource_entity_missing",
        "memories_per_day_exceeded",
    }


def test_indexer_job_status_alias_tracks_server_check_constraint():
    assert set(get_args(IndexerJobStatus)) == {"idle", "queued", "running", "failed"}


# ---------------------------------------------------------------------------
# Sleep models
# ---------------------------------------------------------------------------


def test_sleep_report_parses_degraded_run_with_failure_count():
    report = SleepReport.model_validate(
        {**sleep_report_summary_dict(), "status": "degraded", "llm_call_failures": 2}
    )
    assert report.status == "degraded"
    assert report.llm_call_failures == 2


def test_sleep_report_llm_call_failures_is_none_before_server_v043():
    payload = sleep_report_summary_dict()
    del payload["llm_call_failures"]
    assert SleepReport.model_validate(payload).llm_call_failures is None


def test_sleep_report_detail_parses_degraded_run_with_merge_retention():
    detail = SleepReportDetail.model_validate(
        _detail_payload(
            status="degraded",
            error_message="phase_failure: dedup_merge",
            merge_retention_result={"purged": 3, "window_days": 30},
        )
    )
    assert detail.status == "degraded"
    assert detail.llm_call_failures == 0
    assert detail.error_message == "phase_failure: dedup_merge"
    assert detail.merge_retention_result == {"purged": 3, "window_days": 30}


@pytest.mark.parametrize(
    "model, payload",
    [
        (SleepReport, {**sleep_report_summary_dict(), "status": "paused"}),
        (SleepReportDetail, _detail_payload(status="paused")),
        (
            RollbackResult,
            {"report_id": "rid-1", "status": "paused", "rollback_summary": {}},
        ),
    ],
    ids=["SleepReport", "SleepReportDetail", "RollbackResult"],
)
def test_unseen_sleep_status_does_not_raise(model, payload):
    assert model.model_validate(payload).status == "paused"


# ---------------------------------------------------------------------------
# Indexer models
# ---------------------------------------------------------------------------


def test_indexer_status_parses_memories_per_day_exceeded():
    status = IndexerStatusResponse.model_validate(
        _indexer_payload(skipped_reason="memories_per_day_exceeded")
    )
    assert status.state is not None
    assert status.state.metrics.skipped_reason == "memories_per_day_exceeded"


def test_indexer_status_unseen_values_pass_through():
    # A newer server's reason must not raise — and must not degrade to None
    # either: None means "the last run was NOT skipped" on this wire.
    payload = _indexer_payload(skipped_reason="some_future_reason")
    payload["state"]["job_status"] = "paused"
    status = IndexerStatusResponse.model_validate(payload)
    assert status.state is not None
    assert status.state.job_status == "paused"
    assert status.state.metrics.skipped_reason == "some_future_reason"


# ---------------------------------------------------------------------------
# parse_response
# ---------------------------------------------------------------------------


def test_response_error_is_a_small_kagura_error():
    err = KaguraResponseError("boom")
    assert isinstance(err, KaguraError)
    assert err.operation is None
    assert KaguraResponseError("boom", operation="op").operation == "op"


def test_parse_response_returns_the_model():
    report = parse_response(SleepReport, sleep_report_summary_dict(), operation="op")
    assert isinstance(report, SleepReport)


def test_parse_response_wraps_validation_error():
    payload = {**sleep_report_summary_dict(), "memories_processed": "many"}
    with pytest.raises(KaguraResponseError) as exc_info:
        parse_response(SleepReport, payload, operation="get_sleep_history")

    err = exc_info.value
    assert isinstance(err, KaguraError)
    assert err.operation == "get_sleep_history"
    assert isinstance(err.__cause__, ValidationError)
    message = str(err)
    assert message.startswith("get_sleep_history: ")
    assert "SleepReport" in message
    assert "memories_processed" in message


def test_parse_response_message_omits_payload_values():
    # Payloads can carry sensitive values (secret ciphertext, key plaintext);
    # the message names the field only — the full detail stays on __cause__.
    payload = {**sleep_report_summary_dict(), "memories_processed": "sensitive-value-xyz"}
    with pytest.raises(KaguraResponseError) as exc_info:
        parse_response(SleepReport, payload, operation="op")
    assert "sensitive-value-xyz" not in str(exc_info.value)


def test_parse_response_caps_the_listed_errors():
    # Every required field missing: the message lists a few, then a count.
    with pytest.raises(KaguraResponseError, match=r"\+\d+ more"):
        parse_response(SleepReport, {}, operation="op")


def test_parse_response_non_mapping_payload():
    with pytest.raises(KaguraResponseError, match="op: "):
        parse_response(SleepReport, None, operation="op")


# ---------------------------------------------------------------------------
# MCP surface: every KaguraClient model path wraps drift
# ---------------------------------------------------------------------------


_MCP_DRIFT_CASES: list[tuple[str, Callable[[KaguraClient], Awaitable[Any]], dict[str, Any]]] = [
    (
        "get_sleep_history",
        lambda c: c.get_sleep_history(context_id="ctx-1"),
        {"reports": [sleep_report_summary_dict("ok"), {"report_id": "broken"}]},
    ),
    (
        "get_sleep_report",
        lambda c: c.get_sleep_report(context_id="ctx-1", report_id="rid-1"),
        {"report": {"report_id": "rid-1"}, "actions": [], "action_count": 0},
    ),
    (
        "rollback_sleep_run",
        lambda c: c.rollback_sleep_run(context_id="ctx-1", report_id="rid-1"),
        {"report_id": "rid-1"},
    ),
    ("get_usage", lambda c: c.get_usage(), {"plan": "free"}),
    ("get_context_info", lambda c: c.get_context_info("ctx-1"), {}),
    ("list_tags", lambda c: c.list_tags("ctx-1"), {"tags": []}),
    ("get_agent_bootstrap", lambda c: c.get_agent_bootstrap(AGENT), {}),
    ("register_agent", lambda c: c.register_agent("a"), {"agent": {"id": AGENT}}),
    ("get_agent", lambda c: c.get_agent(AGENT), {"agent": {"id": AGENT}}),
    ("list_agents", lambda c: c.list_agents(), {"agents": [{"id": AGENT}]}),
    (
        "update_agent",
        lambda c: c.update_agent(AGENT, name="b"),
        {"agent": {"id": AGENT}},
    ),
    (
        "bind_agent_context",
        lambda c: c.bind_agent_context(AGENT, WS),
        {"binding": {"id": "b"}},
    ),
    (
        "list_agent_bindings",
        lambda c: c.list_agent_bindings(AGENT),
        {"bindings": [{"id": "b"}]},
    ),
    (
        "update_agent_binding",
        lambda c: c.update_agent_binding(AGENT, "b", can_read=False),
        {"binding": {"id": "b"}},
    ),
    (
        "list_edges",
        lambda c: c.list_edges(context_id="ctx-1", memory_id="m"),
        {"edges": [{"source_id": "a"}]},
    ),
    (
        "create_edge",
        lambda c: c.create_edge(context_id="ctx-1", source_id="a", target_id="b"),
        {"edge": {"source_id": "a"}},
    ),
    (
        "update_edge",
        lambda c: c.update_edge(context_id="ctx-1", source_id="a", target_id="b", weight=1.0),
        {"edge": {"source_id": "a"}},
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool, call, payload", _MCP_DRIFT_CASES, ids=[case[0] for case in _MCP_DRIFT_CASES]
)
async def test_mcp_model_paths_wrap_drift(tool, call, payload):
    client = KaguraClient(api_key="test", mcp_url="https://test.com/mcp")
    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = {"status": "success", **payload}
            with pytest.raises(KaguraResponseError) as exc_info:
                await call(client)
        assert exc_info.value.operation == tool
        assert isinstance(exc_info.value.__cause__, ValidationError)
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# REST surface: every KaguraRestClient subclass wraps drift
# ---------------------------------------------------------------------------


def _rest_client(cls: type, body: Any) -> Any:
    client = cls(api_key="kagura_test", base_url="https://test.com")
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=body)),
        headers={"Authorization": "Bearer kagura_test"},
    )
    return client


_REST_DRIFT_CASES: list[tuple[str, type, Callable[[Any], Awaitable[Any]], Any]] = [
    (
        "ResourceClient.get_indexer_status",
        ResourceClient,
        lambda c: c.get_indexer_status("products"),
        {**_indexer_payload(), "state": {"job_status": "idle"}},
    ),
    ("ResourceClient.list_tokens", ResourceClient, lambda c: c.list_tokens(), {"tokens": []}),
    ("ResourceClient.list_resources", ResourceClient, lambda c: c.list_resources(), {}),
    (
        "ResourceClient.get_resource_impact",
        ResourceClient,
        lambda c: c.get_resource_impact("products"),
        {},
    ),
    (
        "ResourceClient.list_resource_events",
        ResourceClient,
        lambda c: c.list_resource_events("products"),
        {"events": [{"id": "not-an-int"}]},
    ),
    ("FilesClient.list", FilesClient, lambda c: c.list(context_id=WS), [{"id": "f"}]),
    (
        # The forward-compatible ``{files, next_cursor}`` envelope branch.
        "FilesClient.list",
        FilesClient,
        lambda c: c.list(context_id=WS),
        {"files": [{"id": "f"}], "next_cursor": None},
    ),
    (
        "FilesClient.download_url",
        FilesClient,
        lambda c: c.download_url("f", context_id=WS),
        {},
    ),
    (
        "WorkspaceClient.list_members",
        WorkspaceClient,
        lambda c: c.list_members(WS),
        [{"role": "member"}],
    ),
    (
        "WorkspaceClient.list_invitations",
        WorkspaceClient,
        lambda c: c.list_invitations(WS),
        [{"id": "x"}],
    ),
    ("AgentsClient.get_agent", AgentsClient, lambda c: c.get_agent(AGENT), {"id": AGENT}),
    ("AgentsClient.bootstrap", AgentsClient, lambda c: c.bootstrap(AGENT), {}),
    ("SecretClient.list_pubkeys", SecretClient, lambda c: c.list_pubkeys(), [{"id": "p"}]),
    ("SecretClient.verify_audit", SecretClient, lambda c: c.verify_audit(), {}),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation, cls, call, body",
    _REST_DRIFT_CASES,
    ids=[case[0] for case in _REST_DRIFT_CASES],
)
async def test_rest_model_paths_wrap_drift(operation, cls, call, body):
    client = _rest_client(cls, body)
    try:
        with pytest.raises(KaguraResponseError) as exc_info:
            await call(client)
        assert exc_info.value.operation == operation
        assert str(exc_info.value).startswith(f"{operation}: ")
    finally:
        await client.close()
