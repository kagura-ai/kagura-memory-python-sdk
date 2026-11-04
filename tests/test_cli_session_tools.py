"""CLI for the session tools (#301): ``kagura bootstrap`` / ``changes`` / ``list``
and ``kagura forget --ids``.

The client is mocked: these tests pin what each command passes to
:class:`KaguraClient` and that the typed reply is printed as JSON (the
client side is covered by ``tests/test_new_tools.py``).
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

from click.testing import CliRunner

from kagura_memory import (
    BootstrapResponse,
    ChangesSinceResponse,
    ForgetBatchResult,
    ListMatchingResponse,
)
from kagura_memory.cli import main

CTX = "11111111-2222-3333-4444-555555555555"
MEM_A = "aaaaaaaa-0000-0000-0000-000000000001"
MEM_B = "aaaaaaaa-0000-0000-0000-000000000002"


def _invoke(*args: str, **replies: Any) -> tuple[Any, AsyncMock]:
    """Run ``kagura <args>`` with a mocked client whose methods return ``replies``."""
    client = AsyncMock()
    for name, value in replies.items():
        getattr(client, name).return_value = value
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    config = {"api_key": "key", "mcp_url": "https://test.com/mcp", "context_id": CTX}
    with (
        patch("kagura_memory.cli.load_config", return_value=config),
        patch("kagura_memory.cli.KaguraClient", return_value=client),
    ):
        result = CliRunner().invoke(main, list(args))
    return result, client


# ---------------------------------------------------------------- forget --ids


def test_forget_ids_sends_the_list_and_prints_the_rows():
    reply = ForgetBatchResult.model_validate(
        {
            "status": "partial",
            "deleted_count": 1,
            "memory_ids": [MEM_A],
            "results": [
                {"memory_id": MEM_A, "status": "deleted"},
                {"memory_id": MEM_B, "status": "error", "error": "not_attempted"},
            ],
            "count": 2,
            "succeeded": 1,
            "skipped": 0,
            "failed": 1,
        }
    )
    result, client = _invoke("forget", "--ids", f"{MEM_A}, {MEM_B},", forget=reply)

    assert result.exit_code == 0, result.output
    client.forget.assert_awaited_once_with(context_id=CTX, memory_ids=[MEM_A, MEM_B])
    printed = json.loads(result.output)
    assert printed["results"][1]["error"] == "not_attempted"


def test_forget_ids_refuses_another_target_and_an_empty_list():
    for args in (["--ids", MEM_A, "-m", MEM_B], ["--ids", MEM_A, "-q", "old"]):
        result, client = _invoke("forget", *args)
        assert result.exit_code != 0
        assert "only one of --memory-id, --ids or --query" in result.output
        client.forget.assert_not_called()
    result, client = _invoke("forget", "--ids", " , ")
    assert result.exit_code != 0 and "--ids needs at least one memory ID" in result.output


def test_forget_without_a_target_names_ids_too():
    result, _ = _invoke("forget")
    assert result.exit_code != 0
    assert "Either --memory-id or --query is required (or --ids)" in result.output


def test_forget_memory_id_mode_is_unchanged():
    reply = {"status": "success", "deleted_count": 1, "memory_ids": [MEM_A]}
    result, client = _invoke("forget", "-m", MEM_A, forget=reply)
    assert result.exit_code == 0
    client.forget.assert_awaited_once_with(context_id=CTX, memory_id=MEM_A, query=None, k=10)


# ------------------------------------------------------------------- bootstrap


def test_bootstrap_passes_every_option_and_prints_wire_keys():
    reply = BootstrapResponse.model_validate(
        {
            "status": "success",
            "format": "index",
            "components": {
                "upcoming": {"status": "ok", "results": [], "from": "2026-11-04T09:00:00"}
            },
        }
    )
    result, client = _invoke(
        "bootstrap",
        "--since",
        "7d",
        "--include",
        "upcoming,changes",
        "--format",
        "index",
        "--max-chars",
        "5000",
        bootstrap=reply,
    )
    assert result.exit_code == 0, result.output
    client.bootstrap.assert_awaited_once_with(
        CTX, since="7d", include=["upcoming", "changes"], max_chars=5000, format="index"
    )
    printed = json.loads(result.output)
    assert printed["components"]["upcoming"]["from"] == "2026-11-04T09:00:00"  # by alias


def test_bootstrap_defaults_send_nothing_extra():
    reply = BootstrapResponse.model_validate({"components": {}})
    result, client = _invoke("bootstrap", bootstrap=reply)
    assert result.exit_code == 0, result.output
    client.bootstrap.assert_awaited_once_with(
        CTX, since=None, include=None, max_chars=None, format=None
    )


def test_bootstrap_max_chars_floor_is_two_thousand():
    result, client = _invoke("bootstrap", "--max-chars", "1999")
    assert result.exit_code == 2
    client.bootstrap.assert_not_called()


# --------------------------------------------------------------------- changes


def test_changes_passes_every_option():
    reply = ChangesSinceResponse.model_validate({"changes": [], "has_more": False})
    result, client = _invoke(
        "changes",
        "2026-11-01T00:00:00Z",
        "--until",
        "2026-11-04T00:00:00Z",
        "--kinds",
        "created,restored",
        "--cursor",
        "c1",
        "--limit",
        "20",
        "--session-id",
        "mcp-0123456789abcdef",
        "--by-me",
        "--max-chars",
        "10000",
        changes_since=reply,
    )
    assert result.exit_code == 0, result.output
    client.changes_since.assert_awaited_once_with(
        CTX,
        "2026-11-01T00:00:00Z",
        until="2026-11-04T00:00:00Z",
        kinds=["created", "restored"],
        cursor="c1",
        limit=20,
        session_id="mcp-0123456789abcdef",
        by_me=True,
        max_chars=10_000,
    )
    assert json.loads(result.output)["has_more"] is False


def test_changes_without_by_me_leaves_it_unset():
    reply = ChangesSinceResponse.model_validate({"changes": [], "has_more": False})
    result, client = _invoke("changes", "2026-11-01", changes_since=reply)
    assert result.exit_code == 0, result.output
    assert client.changes_since.await_args.kwargs["by_me"] is None
    assert client.changes_since.await_args.kwargs["kinds"] is None


# ------------------------------------------------------------------------ list


def test_list_passes_filters_and_options():
    reply = ListMatchingResponse.model_validate(
        {"memories": [{"memory_id": MEM_A}], "total": 1, "has_more": False}
    )
    result, client = _invoke(
        "list",
        "--filters",
        '{"type": "task", "details.status": "open"}',
        "--order-by",
        "observed_at",
        "--direction",
        "asc",
        "--cursor",
        "50",
        "--limit",
        "10",
        "--details",
        "--max-chars",
        "20000",
        list_matching=reply,
    )
    assert result.exit_code == 0, result.output
    client.list_matching.assert_awaited_once_with(
        CTX,
        filters={"type": "task", "details.status": "open"},
        order_by="observed_at",
        direction="asc",
        cursor="50",
        limit=10,
        include_details=True,
        max_chars=20_000,
    )
    assert json.loads(result.output)["total"] == 1


def test_list_refuses_filters_that_are_not_a_json_object():
    result, client = _invoke("list", "--filters", '["task"]')
    assert result.exit_code == 2
    assert "--filters must be a JSON object, got list" in result.output
    client.list_matching.assert_not_called()
    result, _ = _invoke("list", "--filters", "{type")
    assert result.exit_code == 2 and "Invalid JSON for --filters" in result.output


def test_list_defaults_send_nothing_extra():
    reply = ListMatchingResponse.model_validate({"memories": [], "total": 0, "has_more": False})
    result, client = _invoke("list", list_matching=reply)
    assert result.exit_code == 0, result.output
    client.list_matching.assert_awaited_once_with(
        CTX,
        filters=None,
        order_by=None,
        direction=None,
        cursor=None,
        limit=None,
        include_details=False,
        max_chars=None,
    )
