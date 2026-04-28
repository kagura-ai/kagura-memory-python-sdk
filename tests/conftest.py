"""Shared pytest fixtures and helpers."""


def sleep_report_summary_dict(report_id: str = "rid-1") -> dict:
    """Build a server-shaped Sleep Maintenance summary dict for tests.

    Mirrors ``_report_to_summary`` in
    ``memory-cloud/backend/src/mcp_server/tools/sleep.py`` so that both
    SDK-level and CLI-level tests can construct realistic ``SleepReport``
    fixtures without diverging.
    """
    return {
        "report_id": report_id,
        "context_id": "ctx-1",
        "status": "completed",
        "started_at": "2026-04-28T00:00:00",
        "completed_at": "2026-04-28T00:05:00",
        "memories_processed": 10,
        "edges_created": 2,
        "memories_merged": 1,
        "memories_promoted": 0,
        "llm_calls_made": 3,
        "llm_tokens_used": 1234,
    }
