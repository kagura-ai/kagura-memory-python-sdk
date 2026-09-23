"""Response parsing against a newer server (#250).

**Forward tolerance.** Server-growing value sets — the Sleep run status and
the indexer job status / skipped reason — are typed ``str`` on the response
models, so a value added by a newer server parses instead of failing the
whole call. The ``Literal`` aliases stay as the documented known-value sets
and must track memory-cloud (v0.75.0 here).
"""

from __future__ import annotations

from typing import Any, get_args

import pytest

from kagura_memory import (
    IndexerJobStatus,
    IndexerSkippedReason,
    IndexerStatusResponse,
    RollbackResult,
    SleepReport,
    SleepReportDetail,
    SleepRunStatus,
)
from tests.conftest import sleep_report_summary_dict


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
