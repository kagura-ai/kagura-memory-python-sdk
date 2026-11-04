"""Tests for CLI commands."""

import json
import re
import shlex
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from kagura_memory.auth.credentials import reset_state_cache
from kagura_memory.cli import _cli_error_message, _parse_tags, _unknown_arguments, main
from kagura_memory.exceptions import (
    KaguraConnectionError,
    KaguraError,
    KaguraNotFoundError,
    KaguraVersionConflictError,
)
from kagura_memory.setup_claude import SESSIONSTART_HOOK_COMMAND
from tests.conftest import (
    indexer_status_dict,
    measurement_dict,
    measurement_series_dict,
    sleep_report_detail_dict,
    sleep_report_summary_dict,
)


@pytest.fixture(autouse=True)
def _isolate_credential_state(tmp_path, monkeypatch):
    """Isolate every test from real ``~/.kagura/credentials.json`` and env vars.

    Several CLI helpers (``_get_resource_client``, ``_run_files_command``)
    now walk the canonical SDK chain (``env > OAuth profile > .kagura.json``)
    via ``_resolve_auth``. Without this isolation, a developer's stored
    OAuth profile or a stale ``KAGURA_PROFILE`` could pre-empt the
    config-only fixtures used by the resource / context tests below
    and make them flaky across machines.
    """
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


def test_parse_tags():
    """_parse_tags should handle various inputs."""
    assert _parse_tags(None) is None
    assert _parse_tags("") is None
    assert _parse_tags(",,,") is None
    assert _parse_tags("a, b, c") == ["a", "b", "c"]
    assert _parse_tags("single") == ["single"]
    assert _parse_tags(" spaced , tags ") == ["spaced", "tags"]


@patch("kagura_memory.cli.load_config")
def test_config_show(mock_config):
    """config show should display masked API key."""
    mock_config.return_value = {
        "api_key": "kagura_12345678abcdef",
        "mcp_url": "https://memory.kagura-ai.com/mcp",
        "model": "gpt-5.4-nano",
    }
    runner = CliRunner()
    result = runner.invoke(main, ["config", "show"])
    assert result.exit_code == 0
    assert "kagura_1...cdef" in result.output
    assert "gpt-5.4-nano" in result.output


@patch("kagura_memory.cli.load_config")
def test_config_show_no_api_key(mock_config):
    """config show should work without API key."""
    mock_config.return_value = {"api_key": "", "mcp_url": "https://test.com/mcp"}
    runner = CliRunner()
    result = runner.invoke(main, ["config", "show"])
    assert result.exit_code == 0


@patch("kagura_memory.cli.load_config")
def test_remember_missing_api_key(mock_config, monkeypatch, tmp_path):
    """remember should fail when neither api_key nor OAuth credentials exist."""
    mock_config.return_value = {"api_key": "", "context_id": "ctx-1"}
    # Make sure env vars + credentials.json paths cannot rescue.
    monkeypatch.delenv("KAGURA_API_KEY", raising=False)
    monkeypatch.delenv("KAGURA_PROFILE", raising=False)
    monkeypatch.setattr(
        "kagura_memory.auth.credentials.DEFAULT_CREDENTIALS_PATH",
        tmp_path / "missing-credentials.json",
    )
    # Also short-circuit the .kagura.json fallback inside _resolve_auth.
    monkeypatch.setattr("kagura_memory._auth.load_config", lambda: {"api_key": ""})
    runner = CliRunner()
    result = runner.invoke(main, ["remember", "-s", "test", "--content", "data"])
    assert result.exit_code != 0
    assert "No credentials found" in result.output or "kagura auth login" in result.output


@patch("kagura_memory.cli.load_config")
def test_remember_missing_context(mock_config):
    """remember should fail without context_id."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}
    runner = CliRunner()
    result = runner.invoke(main, ["remember", "-s", "test", "--content", "data"])
    assert result.exit_code != 0
    assert "context_id required" in result.output


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_contexts_command(mock_client_cls, mock_config):
    """contexts command should list available contexts."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    mock_client = AsyncMock()
    mock_client.list_contexts.return_value = {"contexts": [{"id": "c1", "name": "test"}]}
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    runner = CliRunner()
    result = runner.invoke(main, ["contexts"])
    assert result.exit_code == 0
    assert "c1" in result.output


def _recall_cli(mock_client_cls, mock_config, *args):
    """Invoke `kagura recall` with a wired mock client and return (result, client)."""
    mock_config.return_value = {
        "api_key": "key",
        "mcp_url": "https://test.com/mcp",
        "context_id": "ctx",
    }
    mock_client = AsyncMock()
    mock_client.recall.return_value = {"results": [{"summary": "found"}]}
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client
    result = CliRunner().invoke(main, ["recall", "test query", *args])
    return result, mock_client


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_recall_command(mock_client_cls, mock_config):
    """recall command should search and output results."""
    result, _ = _recall_cli(mock_client_cls, mock_config)
    assert result.exit_code == 0
    assert "found" in result.output


@pytest.mark.parametrize(
    ("flags", "expected"),
    [((), None), (("--rerank",), True), (("--no-rerank",), False)],
    ids=["omitted", "rerank", "no-rerank"],
)
@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_recall_rerank_flag(mock_client_cls, mock_config, flags, expected):
    """Issue #251: --rerank/--no-rerank forward True/False; no flag forwards None.

    None leaves ``use_rerank`` off the wire, so the server follows the
    context's search config (memory-cloud v0.69.0+).
    """
    result, mock_client = _recall_cli(mock_client_cls, mock_config, *flags)
    assert result.exit_code == 0, result.output
    assert mock_client.recall.call_args.kwargs.get("use_rerank") is expected


@pytest.mark.parametrize(
    ("flags", "expected"),
    [((), None), (("--trusted-only",), {"trust_tier": "trusted"})],
    ids=["default", "trusted-only"],
)
@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_recall_trusted_only_flag(mock_client_cls, mock_config, flags, expected):
    """Issue #258: --trusted-only excludes connector-ingested memories; no flag sends no filter."""
    result, mock_client = _recall_cli(mock_client_cls, mock_config, *flags)
    assert result.exit_code == 0, result.output
    assert mock_client.recall.call_args.kwargs.get("filters") == expected


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ((), {"verbosity": None, "full": None, "max_chars": None}),
        (
            ("--verbosity", "full", "--full", "0", "--max-chars", "10000"),
            {"verbosity": "full", "full": 0, "max_chars": 10_000},
        ),
        (
            ("--verbosity", "compact", "--full", "100", "--max-chars", "100000"),
            {"verbosity": "compact", "full": 100, "max_chars": 100_000},
        ),
    ],
    ids=["omitted", "full-low", "compact-high"],
)
@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_recall_shape_flags(mock_client_cls, mock_config, flags, expected):
    """Issue #298: each flag reaches recall(); an omitted one is None, so it stays off the wire."""
    result, mock_client = _recall_cli(mock_client_cls, mock_config, *flags)
    assert result.exit_code == 0, result.output
    kwargs = mock_client.recall.call_args.kwargs
    assert {key: kwargs[key] for key in expected} == expected


@pytest.mark.parametrize(
    "flags",
    [
        ("--verbosity", "verbose"),
        ("--full", "-1"),
        ("--full", "101"),
        ("--max-chars", "9999"),
        ("--max-chars", "100001"),
    ],
)
@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_recall_shape_flags_out_of_range_are_usage_errors(mock_client_cls, mock_config, flags):
    result, mock_client = _recall_cli(mock_client_cls, mock_config, *flags)
    assert result.exit_code == 2, result.output
    assert flags[0] in result.output
    mock_client.recall.assert_not_called()


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_recall_prints_content_right_after_its_summary(mock_client_cls, mock_config):
    """Issue #298: the JSON keeps the server's key order, so content follows its summary."""
    hit = {
        "memory_id": "m1",
        "summary": "Deploy target moved",
        "summary_cut": True,
        "content": "The latency review showed…",
        "content_truncated": True,
        "content_next_offset": 1500,
        "type": "decision",
        "score": 0.81,
    }
    mock_config.return_value = {
        "api_key": "key",
        "mcp_url": "https://test.com/mcp",
        "context_id": "ctx",
    }
    mock_client = AsyncMock()
    mock_client.recall.return_value = {"confidence": {"level": "high"}, "results": [hit]}
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    result = CliRunner().invoke(main, ["recall", "deploy", "--full", "1"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["results"] == [hit]
    out = result.output
    assert out.index('"summary"') < out.index('"content"') < out.index('"type"')


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_session_start_hook_command_is_a_valid_recall(mock_client_cls, mock_config):
    """The hook `kagura setup claude` writes must parse with this CLI's flags (#258, #298)."""
    command = SESSIONSTART_HOOK_COMMAND.format(context_id="ctx")
    argv = shlex.split(command.split(" 2>/dev/null", 1)[0])
    assert argv[:3] == ["kagura", "recall", "project context and recent decisions"]

    # _recall_cli supplies its own query; pass the hook's flags after it.
    result, mock_client = _recall_cli(mock_client_cls, mock_config, *argv[3:])

    assert result.exit_code == 0, result.output
    kwargs = mock_client.recall.call_args.kwargs
    assert kwargs["context_id"] == "ctx"
    assert kwargs["k"] == 5
    assert kwargs["filters"] == {"trust_tier": "trusted"}
    assert kwargs["verbosity"] == "full"
    assert kwargs["full"] == 0


def _full_row(i: int) -> dict:
    """A verbosity="full", full=0 recall row: 200-character summary, 140 of context_summary."""
    return {
        "memory_id": f"00000000-0000-0000-0000-00000000000{i}",
        "summary": f"Decision {i}: " + "x" * 188,
        "context_summary": "y" * 140,
        "type": "decision",
        "importance": 0.8,
        "scope": "persistent",
        "score": 0.8123,
        "created_at": "2026-10-01T00:00:00Z",
        "updated_at": "2026-10-02T00:00:00Z",
        "version": 3,
        "tags": ["architecture", "decision", f"row-{i}"],
    }


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_session_start_hook_cap_keeps_five_full_rows_with_tags(mock_client_cls, mock_config):
    """#298: the hook's `head -c` cap holds all five uncut-summary rows and their tags."""
    cap = int(re.search(r"head -c (\d+)", SESSIONSTART_HOOK_COMMAND).group(1))
    mock_config.return_value = {
        "api_key": "key",
        "mcp_url": "https://test.com/mcp",
        "context_id": "ctx",
    }
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client
    mock_client.recall.return_value = {
        "confidence": {
            "level": "high",
            "top_score": 0.71,
            "prominence": 0.52,
            "relative_margin": 3.1,
            "result_count": 5,
            "rationale": "Top hit is well above the background of the candidate pool.",
        },
        "status": "success",
        "results": [_full_row(i) for i in range(5)],
        "count": 5,
        "context_id": "00000000-0000-0000-0000-000000000099",
        "context_name": "dev",
        "context_display_name": "Dev",
        "context_is_private": False,
        "context_is_locked": False,
        "related_tags": [{"tag": "architecture", "count": 3}],
    }

    result = CliRunner().invoke(main, ["recall", "project context"])

    assert result.exit_code == 0, result.output
    head = result.output.encode("utf-8")[:cap].decode("utf-8", errors="ignore")
    for i in range(5):
        assert f'"row-{i}"\n      ]' in head  # row i's tags list, closed


def _server_error(
    code: str | None, message: str = "refused", unknown: list[str] | None = None
) -> KaguraError:
    """A KaguraError as the client raises it for an MCP error envelope with ``code``.

    ``unknown`` is the envelope's ``unknown_arguments`` (memory-cloud
    ``find_unknown_arguments``, top level since 0.82.0), which the client keeps
    in ``KaguraError.details``; ``None`` leaves ``details`` unset.
    """
    err = KaguraError(f"recall failed ({code}): {message}")
    err.code = code  # what the client's MCP error mapping sets (KaguraError.code)
    if unknown is not None:
        err.details = {"unknown_arguments": unknown, "allowed_arguments": ["context_id", "query"]}
    return err


def _recall_cli_raising(mock_client_cls, mock_config, side_effect, *args):
    """`_recall_cli` with ``client.recall`` answering in turn from ``side_effect``."""
    mock_config.return_value = {
        "api_key": "key",
        "mcp_url": "https://test.com/mcp",
        "context_id": "ctx",
    }
    mock_client = AsyncMock()
    mock_client.recall.side_effect = side_effect
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client
    result = CliRunner().invoke(main, ["recall", "test query", *args])
    return result, mock_client


_HOOK_SHAPE_FLAGS = ("--trusted-only", "--verbosity", "full", "--full", "0")


def _shape(call) -> tuple:
    return call.kwargs["verbosity"], call.kwargs["full"], call.kwargs["max_chars"]


@pytest.mark.parametrize(
    ("unknown", "retried", "note"),
    [
        # 0.108.0-0.113.x: verbosity is declared, full is not.
        (
            ["full"],
            ("full", None, None),
            "retried without --full (--full needs memory-cloud 0.114.0+).",
        ),
        # 0.82.0-0.107.x: neither is declared.
        (
            ["full", "verbosity"],
            (None, None, None),
            "retried without --verbosity, --full (--verbosity needs memory-cloud 0.108.0+, "
            "--full needs memory-cloud 0.114.0+).",
        ),
        # No readable list (details absent): every shape flag that was sent goes.
        (
            None,
            (None, None, None),
            "retried without --verbosity, --full (--verbosity needs memory-cloud 0.108.0+, "
            "--full needs memory-cloud 0.114.0+).",
        ),
    ],
    ids=["0.108-0.113-drops-full", "0.82-0.107-drops-both", "no-details-drops-all"],
)
@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_recall_retries_once_without_the_refused_shape_flags(
    mock_client_cls, mock_config, unknown, retried, note
):
    """#298: the SessionStart hook's flags on an older server — one retry, still rows."""
    reply = {"results": [{"memory_id": "m1", "summary": "found"}]}
    error = _server_error("invalid_argument", unknown=unknown)
    result, mock_client = _recall_cli_raising(
        mock_client_cls, mock_config, [error, reply], *_HOOK_SHAPE_FLAGS
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == reply
    first, second = mock_client.recall.await_args_list
    assert _shape(first) == ("full", 0, None)
    assert _shape(second) == retried
    same = ("context_id", "query", "k", "use_rerank", "filters")
    assert {key: second.kwargs[key] for key in same} == {key: first.kwargs[key] for key in same}
    assert second.kwargs["filters"] == {"trust_tier": "trusted"}
    assert result.stderr.count("note:") == 1
    assert note in result.stderr
    assert "--max-chars" not in result.stderr


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_recall_keeps_an_accepted_shape_flag_on_retry(mock_client_cls, mock_config):
    """--max-chars is declared from 0.82.0, so a refusal of --full alone keeps it."""
    reply = {"results": []}
    result, mock_client = _recall_cli_raising(
        mock_client_cls,
        mock_config,
        [_server_error("invalid_argument", unknown=["full"]), reply],
        "--full",
        "1",
        "--max-chars",
        "40000",
    )

    assert result.exit_code == 0, result.output
    assert [_shape(call) for call in mock_client.recall.await_args_list] == [
        (None, 1, 40_000),
        (None, None, 40_000),
    ]


@pytest.mark.parametrize(
    "unknown", [["k"], [], ["verbosity"]], ids=["other-arg", "empty", "unsent"]
)
@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_recall_does_not_retry_a_refusal_of_other_arguments(mock_client_cls, mock_config, unknown):
    """A listed refusal that names none of the shape flags sent is not about them."""
    result, mock_client = _recall_cli_raising(
        mock_client_cls,
        mock_config,
        _server_error("invalid_argument", unknown=unknown),
        "--full",
        "0",
    )

    assert result.exit_code == 1
    assert mock_client.recall.await_count == 1
    assert "note:" not in result.stderr


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_recall_does_not_retry_when_no_shape_flag_was_sent(mock_client_cls, mock_config):
    result, mock_client = _recall_cli_raising(
        mock_client_cls, mock_config, _server_error("invalid_argument", unknown=["verbosity"])
    )

    assert result.exit_code == 1
    assert mock_client.recall.await_count == 1
    assert "invalid_argument" in result.output
    assert "note:" not in result.stderr


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_recall_retry_error_propagates_unchanged(mock_client_cls, mock_config):
    result, mock_client = _recall_cli_raising(
        mock_client_cls,
        mock_config,
        [
            _server_error("invalid_argument", unknown=["full"]),
            _server_error("invalid_argument", "second failure", unknown=["full"]),
        ],
        "--full",
        "0",
    )

    # The second refusal is not retried again, even though it names --full.
    assert result.exit_code == 1
    assert mock_client.recall.await_count == 2
    assert "recall failed (invalid_argument): second failure" in result.output
    assert result.stderr.count("note:") == 1


@pytest.mark.parametrize(
    "error",
    [
        _server_error("validation_error", unknown=["verbosity"]),
        _server_error(None),
        KaguraNotFoundError("recall: Context not found"),
    ],
    ids=["other-code", "no-code", "not-found"],
)
@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_recall_does_not_retry_other_errors(mock_client_cls, mock_config, error):
    result, mock_client = _recall_cli_raising(
        mock_client_cls, mock_config, error, "--verbosity", "full"
    )

    assert result.exit_code == 1
    assert mock_client.recall.await_count == 1
    assert "note:" not in result.stderr


def test_forget_requires_memory_id_or_query():
    """forget should fail without --memory-id or --query."""
    runner = CliRunner()
    result = runner.invoke(main, ["forget", "-c", "ctx"])
    assert result.exit_code != 0
    assert "Either --memory-id or --query" in result.output


def test_forget_help_describes_deployment_retention():
    """Retention is set per deployment (server v0.66.0+), not a fixed 30 days (#257)."""
    result = CliRunner().invoke(main, ["forget", "--help"])
    assert result.exit_code == 0
    text = " ".join(result.output.split())
    assert "recoverable for 30 days" not in text
    assert "CLEANUP_DELETED_MEMORIES_RETENTION_DAYS" in text
    assert "degraded" in text


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_context_list(mock_client_cls, mock_config):
    """context list should work."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    mock_client = AsyncMock()
    mock_client.list_contexts.return_value = {"contexts": [{"id": "c1"}]}
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    runner = CliRunner()
    result = runner.invoke(main, ["context", "list"])
    assert result.exit_code == 0
    assert "c1" in result.output
    # Issue #255: no flags → every option at its default, so the client omits them all.
    mock_client.list_contexts.assert_awaited_once_with(
        name_contains=None,
        include_summary=False,
        include_details=False,
        include_stats=False,
        cursor=None,
        limit=None,
    )
    assert result.stderr == ""  # no has_more, no footer


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_context_list_forwards_filter_and_detail_flags(mock_client_cls, mock_config):
    """Issue #255: context list maps --name-contains/--summary/--details/--stats."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    mock_client = AsyncMock()
    mock_client.list_contexts.return_value = {"contexts": [], "count": 1, "total": 0}
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["context", "list", "--name-contains", "auth", "--summary", "--details", "--stats"],
    )
    assert result.exit_code == 0, result.output
    mock_client.list_contexts.assert_awaited_once_with(
        name_contains="auth",
        include_summary=True,
        include_details=True,
        include_stats=True,
        cursor=None,
        limit=None,
    )


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_context_create(mock_client_cls, mock_config):
    """context create should call create_context."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    mock_client = AsyncMock()
    mock_client.create_context.return_value = {"id": "new-uuid", "name": "dev"}
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    runner = CliRunner()
    result = runner.invoke(main, ["context", "create", "-n", "dev", "-s", "Dev context"])
    assert result.exit_code == 0
    assert "new-uuid" in result.output


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_context_update(mock_client_cls, mock_config):
    """context update should call update_context."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    mock_client = AsyncMock()
    mock_client.update_context.return_value = {"id": "uuid-1", "summary": "updated"}
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    runner = CliRunner()
    result = runner.invoke(main, ["context", "update", "uuid-1", "-s", "updated"])
    assert result.exit_code == 0
    assert "updated" in result.output


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_context_update_lock(mock_client_cls, mock_config):
    """context update --lock should pass is_locked=True."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    mock_client = AsyncMock()
    mock_client.update_context.return_value = {"id": "uuid-1", "is_locked": True}
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    runner = CliRunner()
    result = runner.invoke(main, ["context", "update", "uuid-1", "--lock"])
    assert result.exit_code == 0
    call_kwargs = mock_client.update_context.call_args[1]
    assert call_kwargs["is_locked"] is True


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_context_update_unlock(mock_client_cls, mock_config):
    """context update --unlock should pass is_locked=False."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    mock_client = AsyncMock()
    mock_client.update_context.return_value = {"id": "uuid-1", "is_locked": False}
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    runner = CliRunner()
    result = runner.invoke(main, ["context", "update", "uuid-1", "--unlock"])
    assert result.exit_code == 0
    call_kwargs = mock_client.update_context.call_args[1]
    assert call_kwargs["is_locked"] is False


def test_context_update_requires_option():
    """context update should fail without any update option."""
    runner = CliRunner()
    result = runner.invoke(main, ["context", "update", "uuid-1"])
    assert result.exit_code != 0
    assert "At least one update option" in result.output


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_update_memory_by_id(mock_client_cls, mock_config):
    """update-memory with --memory-id should call update_memory."""
    mock_config.return_value = {
        "api_key": "key",
        "mcp_url": "https://test.com/mcp",
        "context_id": "ctx",
    }

    mock_client = AsyncMock()
    mock_client.update_memory.return_value = {"status": "success"}
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    runner = CliRunner()
    result = runner.invoke(main, ["update-memory", "-m", "mem-1", "-s", "updated summary"])
    assert result.exit_code == 0
    mock_client.update_memory.assert_called_once()


def test_update_memory_requires_id():
    """update-memory without --memory-id or --external-id should fail."""
    runner = CliRunner()
    result = runner.invoke(main, ["update-memory", "-s", "test"])
    assert result.exit_code != 0
    assert "Either --memory-id or --external-id" in result.output


def test_update_memory_rejects_both_ids():
    """update-memory with both --memory-id and --external-id should fail."""
    runner = CliRunner()
    result = runner.invoke(main, ["update-memory", "-m", "mem-1", "--external-id", "ext-1"])
    assert result.exit_code != 0
    assert "only one" in result.output.lower()


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_update_memory_dismiss_supersede_candidate(mock_client_cls, mock_config):
    """Issue #255: --dismiss-supersede-candidate forwards the flag (a dismissal-only call)."""
    mock_config.return_value = {
        "api_key": "key",
        "mcp_url": "https://test.com/mcp",
        "context_id": "ctx",
    }

    mock_client = AsyncMock()
    mock_client.update_memory.return_value = {
        "status": "success",
        "supersede_candidate_dismissed": "mem-old",
    }
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    runner = CliRunner()
    result = runner.invoke(main, ["update-memory", "-m", "mem-1", "--dismiss-supersede-candidate"])
    assert result.exit_code == 0, result.output
    assert "mem-old" in result.output
    kwargs = mock_client.update_memory.await_args.kwargs
    assert kwargs["memory_id"] == "mem-1"
    assert kwargs["dismiss_supersede_candidate"] is True


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_update_memory_dismiss_defaults_off(mock_client_cls, mock_config):
    """Issue #255: without the flag the CLI passes False, which the client omits."""
    mock_config.return_value = {
        "api_key": "key",
        "mcp_url": "https://test.com/mcp",
        "context_id": "ctx",
    }

    mock_client = AsyncMock()
    mock_client.update_memory.return_value = {"status": "success"}
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    runner = CliRunner()
    result = runner.invoke(main, ["update-memory", "-m", "mem-1", "-s", "updated summary"])
    assert result.exit_code == 0, result.output
    assert mock_client.update_memory.await_args.kwargs["dismiss_supersede_candidate"] is False


@patch("kagura_memory.cli.KaguraClient")
def test_update_memory_dismiss_rejects_external_id(mock_client_cls):
    """Issue #255: --dismiss-supersede-candidate needs --memory-id; with --external-id
    the CLI fails before building a client."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "update-memory",
            "--external-id",
            "ext-1",
            "-s",
            "summary text",
            "--dismiss-supersede-candidate",
        ],
    )
    assert result.exit_code != 0
    assert "--dismiss-supersede-candidate requires --memory-id" in result.output
    mock_client_cls.assert_not_called()


# ---------------------------------------------------------------------------
# update-memory --details / --location / --merge-details (Issue #247)
# ---------------------------------------------------------------------------


def _update_memory_cli(
    mock_client_cls,
    mock_config,
    *args,
    reference=None,
    reference_exc=None,
    update_side_effect=None,
):
    """Invoke `kagura update-memory` with a wired mock client and return (result, client).

    ``args`` is the full argument list after ``update-memory`` (the id flag
    included, so a test can choose --memory-id or --external-id). ``reference``
    is the reply the mock's ``reference`` returns for the merge fallback; by
    default a memory whose details are ``null``. ``reference_exc`` makes that
    read raise instead (a failed reference() call). ``update_memory`` answers
    as memory-cloud 0.113.0+ does (with a ``version``) unless
    ``update_side_effect`` says otherwise.
    """
    mock_config.return_value = {
        "api_key": "key",
        "mcp_url": "https://test.com/mcp",
        "context_id": "ctx",
    }
    mock_client = AsyncMock()
    mock_client.update_memory.return_value = {
        "status": "success",
        "memory_id": "mem-1",
        "version": 2,
    }
    if update_side_effect is not None:
        mock_client.update_memory.side_effect = update_side_effect
    mock_client.reference.return_value = (
        {"status": "success", "memory": {"memory_id": "mem-1", "details": None}}
        if reference is None
        else reference
    )
    if reference_exc is not None:
        mock_client.reference.side_effect = reference_exc
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client
    runner = CliRunner()
    result = runner.invoke(main, ["update-memory", *args])
    return result, mock_client


def test_update_memory_help_states_wholesale_replace():
    """The wholesale-replace semantic and the --merge-details caveats are in --help.

    Acceptance criterion (2) of #247: a user reading the help must learn that
    --details replaces details wholesale, that --merge-details keeps the
    other top-level keys, and that the merge is two calls, not one atomic
    update.
    """
    result = CliRunner().invoke(main, ["update-memory", "--help"])
    assert result.exit_code == 0
    text = " ".join(result.output.split())
    assert "REPLACES the memory's details wholesale" in text
    assert "--merge-details" in text
    assert "top-level keys" in text
    assert "two calls, not one atomic update" in text


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_update_memory_parses_details_json(mock_client_cls, mock_config):
    """--details accepts inline JSON and forwards it as a dict under details=."""
    result, mock_client = _update_memory_cli(
        mock_client_cls,
        mock_config,
        "-m",
        "mem-1",
        "--details",
        '{"location": {"lat": 35.68, "lon": 139.76}, "client": "acme"}',
    )
    assert result.exit_code == 0, result.output
    kwargs = mock_client.update_memory.await_args.kwargs
    assert kwargs["memory_id"] == "mem-1"
    assert kwargs["details"] == {"location": {"lat": 35.68, "lon": 139.76}, "client": "acme"}
    mock_client.reference.assert_not_called()


@pytest.mark.parametrize(
    "args",
    [
        (),
        ("--details", ""),
        ("--location", ""),
        ("--details", "   "),
        ("--location", "  "),
    ],
    ids=[
        "neither-flag",
        "empty-details",
        "empty-location",
        "blank-details",
        "blank-location",
    ],
)
@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_update_memory_details_none_when_unset(mock_client_cls, mock_config, args):
    """No flag — or a blank value, as remember treats it — means details=None (unchanged).

    Only ``None`` leaves the memory's details alone; ``{}`` would clear them,
    so an empty shell variable expanding must not become a clear.
    """
    result, mock_client = _update_memory_cli(
        mock_client_cls, mock_config, "-m", "mem-1", "-s", "updated summary", *args
    )
    assert result.exit_code == 0, result.output
    assert mock_client.update_memory.await_args.kwargs["details"] is None


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_update_memory_rejects_invalid_details_json(mock_client_cls, mock_config):
    """Malformed --details is a usage error, not a traceback, and nothing is sent."""
    result, mock_client = _update_memory_cli(
        mock_client_cls, mock_config, "-m", "mem-1", "--details", "{not json"
    )
    assert result.exit_code != 0
    assert "Invalid JSON" in result.output
    mock_client.update_memory.assert_not_called()


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_update_memory_rejects_non_object_details(mock_client_cls, mock_config):
    """--details must be a JSON object — a list or scalar cannot carry named fields."""
    result, mock_client = _update_memory_cli(
        mock_client_cls, mock_config, "-m", "mem-1", "--details", "[1, 2]"
    )
    assert result.exit_code != 0
    assert "JSON object" in result.output
    mock_client.update_memory.assert_not_called()


@pytest.mark.parametrize(
    ("location", "expected"),
    [
        ("35.68,139.76", {"lat": 35.68, "lon": 139.76}),
        ("35.68, 139.76, Tokyo HQ", {"lat": 35.68, "lon": 139.76, "label": "Tokyo HQ"}),
    ],
    ids=["lat-lon", "lat-lon-label"],
)
@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_update_memory_location_shorthand(mock_client_cls, mock_config, location, expected):
    """--location lat,lon[,label] builds details.location with numeric coordinates."""
    result, mock_client = _update_memory_cli(
        mock_client_cls, mock_config, "-m", "mem-1", "--location", location
    )
    assert result.exit_code == 0, result.output
    details = mock_client.update_memory.await_args.kwargs["details"]
    assert details["location"] == expected
    assert isinstance(details["location"]["lat"], float)


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_update_memory_bare_location_replaces_details(mock_client_cls, mock_config):
    """A bare --location without --merge-details sends ONLY the location.

    This is the destructive replace path the --help warns about: the server
    replaces ``details`` wholesale, so every other key the memory carried is
    dropped. The CLI does not read the memory first unless --merge-details is
    given, so ``reference`` is never called here.
    """
    result, mock_client = _update_memory_cli(
        mock_client_cls, mock_config, "-m", "mem-1", "--location", "35.68,139.76"
    )
    assert result.exit_code == 0, result.output
    assert mock_client.update_memory.await_args.kwargs["details"] == {
        "location": {"lat": 35.68, "lon": 139.76}
    }
    mock_client.reference.assert_not_called()


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_update_memory_rejects_conflicting_location(mock_client_cls, mock_config):
    """location in --details plus --location is ambiguous — fail, don't silently drop one."""
    result, mock_client = _update_memory_cli(
        mock_client_cls,
        mock_config,
        "-m",
        "mem-1",
        "--details",
        '{"location": {"lat": 1.0, "lon": 2.0}}',
        "--location",
        "35.68,139.76",
    )
    assert result.exit_code != 0
    assert "--location conflicts" in result.output
    mock_client.update_memory.assert_not_called()


@pytest.mark.parametrize("bad", ["35.68", "abc,139.76", "91.0,139.76"])
@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_update_memory_rejects_malformed_location(mock_client_cls, mock_config, bad):
    """--location rejects wrong arity, non-numeric, and out-of-range coordinates."""
    result, mock_client = _update_memory_cli(
        mock_client_cls, mock_config, "-m", "mem-1", "--location", bad
    )
    assert result.exit_code != 0
    assert "--location" in result.output
    mock_client.update_memory.assert_not_called()


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (("--details", '{"a": 1}'), {"a": 1}),
        (("--location", "35.68,139.76"), {"location": {"lat": 35.68, "lon": 139.76}}),
    ],
    ids=["details", "location"],
)
@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_update_memory_details_with_external_id(mock_client_cls, mock_config, args, expected):
    """--details/--location ride along on the --external-id upsert path.

    Without --merge-details the payload is forwarded next to ``external_id``
    as-is: no read first, so ``reference`` is never called.
    """
    result, mock_client = _update_memory_cli(
        mock_client_cls,
        mock_config,
        "--external-id",
        "ext-1",
        "-s",
        "sum",
        "--content",
        "c",
        "-t",
        "note",
        *args,
    )
    assert result.exit_code == 0, result.output
    kwargs = mock_client.update_memory.await_args.kwargs
    assert kwargs["external_id"] == "ext-1"
    assert kwargs["memory_id"] is None
    assert kwargs["details"] == expected
    mock_client.reference.assert_not_called()


def _reference_reply(details, **extra):
    """A `reference` reply whose memory carries ``details`` plus any bound markers."""
    return {"status": "success", "memory": {"memory_id": "mem-1", "details": details, **extra}}


def _bounded_reference_reply(**markers):
    """A bounded `reference` reply as memory-cloud 0.78.0+ sends it: markers, no ``details`` key.

    The server places ``{"details": <whole object>}`` OR the marker dict —
    ``details_omitted``/``details_total_chars``/``details_next_offset``, or a
    ``details_json`` page with its offsets — never the key next to the markers.
    """
    return {"status": "success", "memory": {"memory_id": "mem-1", **markers}}


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_update_memory_merge_details(mock_client_cls, mock_config):
    """--merge-details sends --details/--location as details_patch in one call (#302).

    No reference() read first: the server merges the top-level keys over the
    current details. The other flags compose with it: --dismiss-supersede-candidate
    still rides along.
    """
    result, mock_client = _update_memory_cli(
        mock_client_cls,
        mock_config,
        "-m",
        "mem-1",
        "--location",
        "35.68,139.76,Tokyo HQ",
        "--details",
        '{"n": 2, "stale": null}',
        "--merge-details",
        "--dismiss-supersede-candidate",
    )
    assert result.exit_code == 0, result.output
    mock_client.reference.assert_not_called()
    mock_client.update_memory.assert_awaited_once()
    kwargs = mock_client.update_memory.await_args.kwargs
    assert kwargs["memory_id"] == "mem-1"
    assert kwargs["details_patch"] == {
        "n": 2,
        "stale": None,
        "location": {"lat": 35.68, "lon": 139.76, "label": "Tokyo HQ"},
    }
    assert "details" not in kwargs
    assert kwargs["dismiss_supersede_candidate"] is True


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_update_memory_details_patch(mock_client_cls, mock_config):
    """--details-patch is forwarded as written, nulls included."""
    result, mock_client = _update_memory_cli(
        mock_client_cls,
        mock_config,
        "-m",
        "mem-1",
        "--details-patch",
        '{"status": "done", "assignee": null}',
        "--expected-version",
        "3",
    )
    assert result.exit_code == 0, result.output
    kwargs = mock_client.update_memory.await_args.kwargs
    assert kwargs["details_patch"] == {"status": "done", "assignee": None}
    assert kwargs["expected_version"] == 3
    mock_client.reference.assert_not_called()


@pytest.mark.parametrize(
    "args",
    [
        ("--details", '{"a": 1}'),
        ("--location", "35.68,139.76"),
        ("--merge-details", "--details", '{"a": 1}'),
    ],
    ids=["details", "location", "merge-details"],
)
@patch("kagura_memory.cli.KaguraClient")
def test_update_memory_details_patch_is_exclusive(mock_client_cls, args):
    result = CliRunner().invoke(
        main, ["update-memory", "-m", "mem-1", "--details-patch", '{"b": 2}', *args]
    )
    assert result.exit_code == 1, result.output
    assert "--details-patch cannot be combined" in result.output
    mock_client_cls.assert_not_called()


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_update_memory_rejects_non_object_details_patch(mock_client_cls, mock_config):
    result, mock_client = _update_memory_cli(
        mock_client_cls, mock_config, "-m", "mem-1", "--details-patch", "[1]"
    )
    assert result.exit_code != 0
    assert "--details-patch must be a JSON object" in result.output
    mock_client.update_memory.assert_not_called()


class _OddDetailsError(Exception):
    """An error whose ``details`` is not an object (a subclass outside the SDK)."""

    details = ["unknown_arguments"]


@pytest.mark.parametrize(
    ("details", "expected"),
    [
        ({"unknown_arguments": ["details_patch", "add_tags"]}, ["details_patch", "add_tags"]),
        ({"unknown_arguments": ["details_patch", 7, None]}, ["details_patch"]),
        ({"unknown_arguments": "details_patch"}, None),
        ({}, None),
    ],
    ids=["listed", "non-str-dropped", "not-a-list", "absent"],
)
def test_unknown_arguments_reads_the_envelope_list(details, expected):
    err = KaguraError("refused")
    err.code, err.details = "invalid_argument", details
    assert _unknown_arguments(err) == expected


def test_unknown_arguments_tolerates_details_that_are_not_an_object():
    assert _unknown_arguments(_OddDetailsError()) is None
    assert _unknown_arguments(ValueError("no details at all")) is None


def _patch_refused(*names: str) -> KaguraError:
    """memory-cloud 0.82.0-0.139.x refusing undeclared arguments (#1742)."""
    err = KaguraError("update_memory failed (invalid_argument): Unknown argument(s)")
    err.code = "invalid_argument"
    err.details = {"unknown_arguments": list(names)}
    return err


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_update_memory_merge_details_falls_back_when_refused(mock_client_cls, mock_config):
    """Before 0.140.0 the refused call wrote nothing: read, merge, write the whole object.

    The merge follows details_patch's rule (a null deletes the key), and the
    other flags — --expected-version included — go with the write.
    """
    current = {"location": {"lat": 1.0, "lon": 2.0, "label": "old"}, "client": "acme", "n": 1}
    result, mock_client = _update_memory_cli(
        mock_client_cls,
        mock_config,
        "-m",
        "mem-1",
        "-s",
        "updated summary",
        "--location",
        "35.68,139.76,Tokyo HQ",
        "--details",
        '{"n": 2, "client": null}',
        "--merge-details",
        "--expected-version",
        "4",
        reference=_reference_reply(current),
        update_side_effect=[_patch_refused("details_patch"), {"status": "success", "version": 5}],
    )
    assert result.exit_code == 0, result.output
    assert "this server refused it, merging via reference() instead" in result.stderr
    mock_client.reference.assert_awaited_once_with(context_id="ctx", memory_id="mem-1")
    first, second = mock_client.update_memory.await_args_list
    assert "details_patch" in first.kwargs
    assert "details_patch" not in second.kwargs
    assert second.kwargs["details"] == {
        "location": {"lat": 35.68, "lon": 139.76, "label": "Tokyo HQ"},
        "n": 2,
    }
    assert second.kwargs["summary"] == "updated summary"
    assert second.kwargs["expected_version"] == 4
    # The read is not mutated in place: the merge builds a new dict.
    assert current["n"] == 1


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_update_memory_merge_details_falls_back_when_ignored(mock_client_cls, mock_config):
    """A server before 0.82.0 drops details_patch and answers without a version.

    Its other edits were applied, so the second call writes the merged details only.
    """
    current = {"location": {"lat": 1.0, "lon": 2.0}, "client": "acme"}
    result, mock_client = _update_memory_cli(
        mock_client_cls,
        mock_config,
        "-m",
        "mem-1",
        "-s",
        "updated summary",
        "--location",
        "35.68,139.76",
        "--merge-details",
        reference=_reference_reply(current),
        update_side_effect=[{"status": "success"}, {"status": "success", "memory_id": "mem-1"}],
    )
    assert result.exit_code == 0, result.output
    assert "this server ignored it (before 0.82.0)" in result.stderr
    first, second = mock_client.update_memory.await_args_list
    assert first.kwargs["summary"] == "updated summary"
    assert second.kwargs == {
        "context_id": "ctx",
        "memory_id": "mem-1",
        "details": {"location": {"lat": 35.68, "lon": 139.76}, "client": "acme"},
    }


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_update_memory_fallback_folds_observed_at_into_the_details(mock_client_cls, mock_config):
    """The server folds observed_at into the patch; the fallback's merge must too.

    Otherwise a stored observed_at would contradict the new one in the write.
    """
    current = {"observed_at": "2024-01-01", "status": "open"}
    result, mock_client = _update_memory_cli(
        mock_client_cls,
        mock_config,
        "-m",
        "mem-1",
        "--details-patch",
        '{"status": "done"}',
        "--observed-at",
        "2026-11-04",
        reference=_reference_reply(current),
        update_side_effect=[_patch_refused("details_patch"), {"status": "success", "version": 3}],
    )
    assert result.exit_code == 0, result.output
    second = mock_client.update_memory.await_args
    assert second.kwargs["details"] == {"observed_at": "2026-11-04", "status": "done"}
    assert second.kwargs["observed_at"] == "2026-11-04"


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_update_memory_details_patch_fallback_names_its_flag(mock_client_cls, mock_config):
    result, mock_client = _update_memory_cli(
        mock_client_cls,
        mock_config,
        "-m",
        "mem-1",
        "--details-patch",
        '{"status": "done"}',
        reference=_bounded_reference_reply(details_omitted=True, details_total_chars=30000),
        update_side_effect=[_patch_refused("details_patch")],
    )
    assert result.exit_code == 1, result.output
    assert "--details-patch: the memory's current details could not be read" in result.output
    assert "without --details-patch" in result.output
    mock_client.update_memory.assert_awaited_once()


_WIRE_CTX = "11111111-2222-3333-4444-555555555555"
_WIRE_MEM = "aaaaaaaa-0000-0000-0000-000000000001"
# memory-cloud 0.82.0-0.139.x: update_memory's closed schema refuses details_patch
# (mcp_server/tools/__init__.py + _arg_coercion.find_unknown_arguments, #1742).
_UNKNOWN_DETAILS_PATCH = {
    "status": "error",
    "error": "invalid_argument",
    "message": "update_memory does not accept the argument 'details_patch' (did you mean "
    "'details'?). Arguments its inputSchema declares: context_id, details, ...",
    "unknown_arguments": ["details_patch"],
    "allowed_arguments": ["context_id", "details", "expected_version", "memory_id"],
    "suggestions": {"details_patch": "details"},
    "help": "Remove the unknown arguments or rename them as suggested, then call the tool again.",
}
_WIRE_UPDATED = {
    "status": "success",
    "memory_id": _WIRE_MEM,
    "operation": "updated",
    "re_embedded": False,
    "scope": "working",
}


@pytest.mark.parametrize(
    ("first_reply", "expected_calls"),
    [
        (
            {**_WIRE_UPDATED, "version": 5},
            [("update_memory", "patch")],
        ),
        (
            _UNKNOWN_DETAILS_PATCH,
            [("update_memory", "patch"), ("reference", None), ("update_memory", "merged+edits")],
        ),
        (
            _WIRE_UPDATED,
            [("update_memory", "patch"), ("reference", None), ("update_memory", "merged")],
        ),
    ],
    ids=["0.140.0+-one-call", "0.82.0-0.139.x-refused", "before-0.82.0-ignored"],
)
def test_update_memory_merge_details_on_the_wire(monkeypatch, first_reply, expected_calls):
    """--merge-details through the real client and transport, per server generation.

    The refusal is read from the MCP envelope as the client keeps it
    (``KaguraError.details["unknown_arguments"]``), and every request carries
    exactly the arguments named here.
    """
    import httpx

    from kagura_memory import KaguraClient

    stored = {"location": {"lat": 1.0, "lon": 2.0}, "client": "acme", "n": 1}
    sent: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        params = json.loads(request.content)["params"]
        sent.append((params["name"], params["arguments"]))
        if params["name"] == "reference":
            reply = {"status": "success", "memory": {"memory_id": _WIRE_MEM, "details": stored}}
        elif len(sent) == 1:
            reply = first_reply
        else:
            reply = {**_WIRE_UPDATED, "version": 5}
        text = json.dumps(reply)
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"content": [{"type": "text", "text": text}]},
            },
        )

    def wired_client(**_kwargs):
        client = KaguraClient(api_key="test", mcp_url="https://test.com/mcp")
        client._session_id = "pre-set-session"
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return client

    monkeypatch.setattr("kagura_memory.cli.KaguraClient", wired_client)
    monkeypatch.setattr(
        "kagura_memory.cli.load_config",
        lambda: {"api_key": "test", "mcp_url": "https://test.com/mcp"},
    )
    result = CliRunner().invoke(
        main,
        [
            "update-memory",
            "-c",
            _WIRE_CTX,
            "-m",
            _WIRE_MEM,
            "--location",
            "35.68,139.76",
            "--details",
            '{"n": 2, "client": null}',
            "--merge-details",
            "--expected-version",
            "4",
        ],
    )
    assert result.exit_code == 0, result.output
    ids = {"context_id": _WIRE_CTX, "memory_id": _WIRE_MEM}
    new_location = {"lat": 35.68, "lon": 139.76}
    bodies = {
        None: ids,
        "patch": {
            **ids,
            "details_patch": {"n": 2, "client": None, "location": new_location},
            "expected_version": 4,
        },
        # The refused call wrote nothing: the merged object goes with every flag.
        "merged+edits": {
            **ids,
            "details": {"location": new_location, "n": 2},
            "expected_version": 4,
        },
        # The ignored call applied the other flags: only the details are left.
        "merged": {**ids, "details": {"location": new_location, "n": 2}},
    }
    assert sent == [(name, bodies[body]) for name, body in expected_calls]


@pytest.mark.parametrize(
    ("code", "details"),
    [
        ("invalid_argument", {"unknown_arguments": ["add_tags"]}),
        ("invalid_argument", {"unknown_arguments": ["details_patch", "add_tags"]}),
        ("invalid_argument", {}),
        ("validation_error", {}),
    ],
    ids=["other-argument", "another-argument-too", "no-list", "other-code"],
)
@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_update_memory_merge_details_no_fallback_on_other_errors(
    mock_client_cls, mock_config, code, details
):
    """Only a refusal that names details_patch retries; nothing is read or rewritten on a guess."""
    err = KaguraError(f"update_memory failed ({code}): refused")
    err.code, err.details = code, details
    result, mock_client = _update_memory_cli(
        mock_client_cls,
        mock_config,
        "-m",
        "mem-1",
        "--location",
        "35.68,139.76",
        "--merge-details",
        update_side_effect=err,
    )
    assert result.exit_code == 1, result.output
    assert f"({code})" in result.output
    mock_client.reference.assert_not_called()
    mock_client.update_memory.assert_awaited_once()


@pytest.mark.parametrize(
    "reference",
    [_reference_reply(None), {"status": "success", "memory": {"memory_id": "mem-1"}}],
    ids=["details-null", "details-key-absent"],
)
@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_update_memory_merge_details_onto_none(mock_client_cls, mock_config, reference):
    """In the fallback, null (or no) details merge onto ``{}``: the patch is sent as-is."""
    result, mock_client = _update_memory_cli(
        mock_client_cls,
        mock_config,
        "-m",
        "mem-1",
        "--location",
        "35.68,139.76",
        "--merge-details",
        reference=reference,
        update_side_effect=[_patch_refused("details_patch"), {"status": "success"}],
    )
    assert result.exit_code == 0, result.output
    assert mock_client.update_memory.await_args.kwargs["details"] == {
        "location": {"lat": 35.68, "lon": 139.76}
    }


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_update_memory_merge_details_empty_object_resends_current(mock_client_cls, mock_config):
    """--details '{}' under --merge-details is an empty patch, never a clear.

    Without --merge-details the same '{}' clears details, as the client documents.
    """
    current = {"location": {"lat": 1.0, "lon": 2.0}, "client": "acme"}
    result, mock_client = _update_memory_cli(
        mock_client_cls, mock_config, "-m", "mem-1", "--details", "{}", "--merge-details"
    )
    assert result.exit_code == 0, result.output
    assert mock_client.update_memory.await_args.kwargs["details_patch"] == {}

    result, mock_client = _update_memory_cli(
        mock_client_cls,
        mock_config,
        "-m",
        "mem-1",
        "--details",
        "{}",
        "--merge-details",
        reference=_reference_reply(current),
        update_side_effect=[_patch_refused("details_patch"), {"status": "success"}],
    )
    assert result.exit_code == 0, result.output
    assert mock_client.update_memory.await_args.kwargs["details"] == current


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        # The two shapes memory-cloud 0.78.0+ (#1685) produces (no ``details`` key at all).
        (
            _bounded_reference_reply(
                details_omitted=True, details_total_chars=24000, details_next_offset=0
            ),
            "24000 characters",
        ),
        (
            _bounded_reference_reply(
                details_json='{"client": "ac',
                details_offset=0,
                details_total_chars=1000,
                details_truncated=True,
                details_next_offset=14,
            ),
            "1000 characters",
        ),
        # Drift no server produces today, still a bounded read: a truthy
        # non-bool flag, and the size marker without the flag.
        (_bounded_reference_reply(details_omitted=1, details_total_chars=100), "100 characters"),
        (_bounded_reference_reply(details_total_chars=100), "100 characters"),
        # The marker wins even when a ``details`` key rides along, and a
        # missing size leaves the count out of the message.
        (_reference_reply(None, details_omitted=True), "could not be read in full;"),
        (
            _reference_reply(None, details_json='{"client": "ac', details_next_offset=14),
            "could not be read in full;",
        ),
    ],
    ids=[
        "omitted",
        "paged-details-json",
        "omitted-truthy-marker",
        "total-chars-only",
        "omitted-next-to-null-details",
        "paged-next-to-null-details",
    ],
)
@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_update_memory_merge_details_refuses_partial_read(
    mock_client_cls, mock_config, reference, expected
):
    """In the fallback, a bounded reference reply (0.78.0+) that left details out is refused.

    Merging onto a partial read would silently drop the keys that were not
    returned, so the CLI stops before writing anything and points at the
    full-object path (--details without --merge-details). The first cases use
    the server's real shape — markers in place of the ``details`` key — so a
    guard that treats an absent key as "no details" before checking the
    markers would merge onto ``{}`` and fail here.
    """
    result, mock_client = _update_memory_cli(
        mock_client_cls,
        mock_config,
        "-m",
        "mem-1",
        "--location",
        "35.68,139.76",
        "--merge-details",
        reference=reference,
        update_side_effect=[_patch_refused("details_patch")],
    )
    assert result.exit_code == 1, result.output
    assert expected in result.output
    assert "without --merge-details" in result.output
    # Only the refused one-call attempt went out; no details were written.
    mock_client.update_memory.assert_awaited_once()
    assert "details" not in mock_client.update_memory.await_args.kwargs


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_update_memory_merge_details_rejects_non_object_current(mock_client_cls, mock_config):
    """Current details that are not a JSON object cannot be merged onto — nothing is written."""
    result, mock_client = _update_memory_cli(
        mock_client_cls,
        mock_config,
        "-m",
        "mem-1",
        "--location",
        "35.68,139.76",
        "--merge-details",
        reference=_reference_reply(["not", "an", "object"]),
        update_side_effect=[_patch_refused("details_patch")],
    )
    assert result.exit_code == 1, result.output
    assert "current details are not a JSON object" in result.output
    mock_client.update_memory.assert_awaited_once()


@pytest.mark.parametrize(
    "reference",
    [{"status": "success"}, {"status": "success", "memory": "mem-1"}, ["not", "a", "dict"]],
    ids=["memory-missing", "memory-not-a-dict", "reply-not-a-dict"],
)
@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_update_memory_merge_details_requires_memory_object(
    mock_client_cls, mock_config, reference
):
    """A reference reply without a memory object is an error, never a merge onto nothing."""
    result, mock_client = _update_memory_cli(
        mock_client_cls,
        mock_config,
        "-m",
        "mem-1",
        "--location",
        "35.68,139.76",
        "--merge-details",
        reference=reference,
        update_side_effect=[_patch_refused("details_patch")],
    )
    assert result.exit_code == 1, result.output
    assert "carried no memory object" in result.output
    mock_client.update_memory.assert_awaited_once()


@pytest.mark.parametrize(
    "exc",
    [
        KaguraNotFoundError("reference: Memory not found or you don't have access: mem-1"),
        KaguraConnectionError("timed out"),
    ],
    ids=["not-found", "transport"],
)
@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_update_memory_merge_details_failed_read_never_writes(mock_client_cls, mock_config, exc):
    """A fallback reference() that RAISES fails the command; no details are written.

    ``_call_tool_checked`` turns a memory_not_found / permission_denied
    envelope into ``KaguraNotFoundError`` / ``KaguraError`` and a transport
    failure is ``KaguraConnectionError``. Either must surface as the
    command's error (exit 1, the message printed), never be swallowed into
    a merge onto ``{}`` — that would send a wholesale replace after a
    transient read failure, the exact data loss --merge-details exists to
    prevent.
    """
    result, mock_client = _update_memory_cli(
        mock_client_cls,
        mock_config,
        "-m",
        "mem-1",
        "--location",
        "35.68,139.76",
        "--merge-details",
        reference_exc=exc,
        update_side_effect=[_patch_refused("details_patch")],
    )
    assert result.exit_code == 1, result.output
    assert str(exc) in result.output
    mock_client.reference.assert_awaited_once_with(context_id="ctx", memory_id="mem-1")
    mock_client.update_memory.assert_awaited_once()
    assert "details" not in mock_client.update_memory.await_args.kwargs


@patch("kagura_memory.cli.KaguraClient")
def test_update_memory_merge_details_rejects_external_id(mock_client_cls):
    """--merge-details needs --memory-id (reference takes a memory id); with --external-id
    the CLI fails before building a client."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "update-memory",
            "--external-id",
            "ext-1",
            "--location",
            "35.68,139.76",
            "--merge-details",
        ],
    )
    assert result.exit_code == 1, result.output
    assert "--merge-details requires --memory-id" in result.output
    mock_client_cls.assert_not_called()


@pytest.mark.parametrize(
    "args",
    [(), ("--details", "  "), ("--location", "")],
    ids=["no-payload", "blank-details", "blank-location"],
)
@patch("kagura_memory.cli.KaguraClient")
def test_update_memory_merge_details_needs_payload(mock_client_cls, args):
    """--merge-details with nothing to merge is an error before any client call.

    A blank --details/--location is unset (details=None), and merging nothing
    would be a pointless reference() round-trip that changes no details.
    """
    runner = CliRunner()
    result = runner.invoke(main, ["update-memory", "-m", "mem-1", "--merge-details", *args])
    assert result.exit_code == 1, result.output
    assert "--merge-details needs --details or --location" in result.output
    mock_client_cls.assert_not_called()


# ---------------------------------------------------------------------------
# update-memory --add-tag / --remove-tag / --expected-version / --observed-at /
# --accept-supersede-candidate, remember --observed-at / --dedupe /
# --tags-normalize, context search-config --content-chunks / --llm-selector (#302)
# ---------------------------------------------------------------------------


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_update_memory_tag_edits_and_guards(mock_client_cls, mock_config):
    """Repeatable --add-tag / --remove-tag (blanks dropped) and the other new flags."""
    result, mock_client = _update_memory_cli(
        mock_client_cls,
        mock_config,
        "-m",
        "mem-1",
        "--add-tag",
        "status:done",
        "--add-tag",
        " ",
        "--add-tag",
        " release ",
        "--remove-tag",
        "status:open",
        "--expected-version",
        "7",
        "--observed-at",
        "2026-11-04",
    )
    assert result.exit_code == 0, result.output
    kwargs = mock_client.update_memory.await_args.kwargs
    assert kwargs["add_tags"] == ["status:done", "release"]
    assert kwargs["remove_tags"] == ["status:open"]
    assert kwargs["expected_version"] == 7
    assert kwargs["observed_at"] == "2026-11-04"
    assert kwargs["tags"] is None
    assert kwargs["accept_supersede_candidate"] is False


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_update_memory_new_flags_default_off(mock_client_cls, mock_config):
    """Unset (or blank), each new flag reaches the client as its unset value."""
    result, mock_client = _update_memory_cli(
        mock_client_cls, mock_config, "-m", "mem-1", "-s", "updated summary", "--observed-at", " "
    )
    assert result.exit_code == 0, result.output
    kwargs = mock_client.update_memory.await_args.kwargs
    assert kwargs["add_tags"] is None
    assert kwargs["remove_tags"] is None
    assert kwargs["expected_version"] is None
    assert kwargs["observed_at"] is None
    assert kwargs["accept_supersede_candidate"] is False
    assert kwargs["details"] is None
    assert "details_patch" not in kwargs


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_update_memory_accept_supersede_candidate(mock_client_cls, mock_config):
    result, mock_client = _update_memory_cli(
        mock_client_cls, mock_config, "-m", "mem-1", "--accept-supersede-candidate"
    )
    assert result.exit_code == 0, result.output
    assert mock_client.update_memory.await_args.kwargs["accept_supersede_candidate"] is True


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("--add-tag", "a"), "--add-tag requires --memory-id"),
        (("--remove-tag", "a"), "--remove-tag requires --memory-id"),
        (("--details-patch", '{"a": 1}'), "--details-patch requires --memory-id"),
        (("--expected-version", "2"), "--expected-version requires --memory-id"),
        (("--accept-supersede-candidate",), "--accept-supersede-candidate requires --memory-id"),
        (
            ("--add-tag", "a", "--expected-version", "2"),
            "--add-tag, --expected-version require --memory-id",
        ),
    ],
    ids=["add-tag", "remove-tag", "details-patch", "expected-version", "accept", "two-flags"],
)
@patch("kagura_memory.cli.KaguraClient")
def test_update_memory_in_place_flags_reject_external_id(mock_client_cls, args, message):
    result = CliRunner().invoke(
        main, ["update-memory", "--external-id", "ext-1", "-s", "summary text", *args]
    )
    assert result.exit_code == 1, result.output
    assert message in result.output
    mock_client_cls.assert_not_called()


@pytest.mark.parametrize("flag", ["--add-tag", "--remove-tag"])
@patch("kagura_memory.cli.KaguraClient")
def test_update_memory_tags_and_tag_edits_are_exclusive(mock_client_cls, flag):
    result = CliRunner().invoke(main, ["update-memory", "-m", "mem-1", "--tags", "a,b", flag, "c"])
    assert result.exit_code == 1, result.output
    assert "--tags replaces the whole list" in result.output
    mock_client_cls.assert_not_called()


@patch("kagura_memory.cli.KaguraClient")
def test_update_memory_expected_version_must_be_positive(mock_client_cls):
    result = CliRunner().invoke(main, ["update-memory", "-m", "mem-1", "--expected-version", "0"])
    assert result.exit_code == 2, result.output
    assert "--expected-version" in result.output
    mock_client_cls.assert_not_called()


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_update_memory_version_conflict_prints_the_current_version(mock_client_cls, mock_config):
    err = KaguraVersionConflictError(
        "update_memory failed (version_conflict): Memory mem-1 is at version 5, not the "
        "expected version 3."
    )
    err.code = "version_conflict"
    err.details = {
        "current_version": 5,
        "expected_version": 3,
        "help": "reference() the memory and resend with its current version.",
    }
    result, _ = _update_memory_cli(
        mock_client_cls,
        mock_config,
        "-m",
        "mem-1",
        "-s",
        "updated summary",
        "--expected-version",
        "3",
        update_side_effect=err,
    )
    assert result.exit_code == 1, result.output
    assert "is at version 5" in result.output
    assert "Current version: 5" in result.output
    assert "Help: reference() the memory" in result.output


def test_version_conflict_without_a_current_version_prints_no_line():
    assert "Current version" not in _cli_error_message(KaguraVersionConflictError("conflict"))


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_remember_new_flags(mock_client_cls, mock_config):
    mock_config.return_value = {
        "api_key": "key",
        "mcp_url": "https://test.com/mcp",
        "context_id": "ctx",
    }
    mock_client = AsyncMock()
    mock_client.remember.return_value = {
        "status": "duplicate_candidate",
        "candidate": {"memory_id": "mem-old", "summary": "Deploy key rotated", "similarity": 0.9},
    }
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    result = CliRunner().invoke(
        main,
        [
            "remember",
            "-s",
            "Deploy key rotated",
            "--content",
            "...",
            "--observed-at",
            "2026-11-01",
            "--dedupe",
            "check",
            "--tags-normalize",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "duplicate_candidate" in result.output
    kwargs = mock_client.remember.await_args.kwargs
    assert kwargs["observed_at"] == "2026-11-01"
    assert kwargs["dedupe"] == "check"
    assert kwargs["tags_normalize"] is True


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_remember_new_flags_default_off(mock_client_cls, mock_config):
    mock_config.return_value = {
        "api_key": "key",
        "mcp_url": "https://test.com/mcp",
        "context_id": "ctx",
    }
    mock_client = AsyncMock()
    mock_client.remember.return_value = {"status": "success"}
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    result = CliRunner().invoke(main, ["remember", "-s", "summary text", "--content", "..."])
    assert result.exit_code == 0, result.output
    kwargs = mock_client.remember.await_args.kwargs
    assert kwargs["observed_at"] is None
    assert kwargs["dedupe"] is None
    assert kwargs["tags_normalize"] is False


def test_remember_rejects_an_unknown_dedupe_mode():
    result = CliRunner().invoke(
        main, ["remember", "-s", "summary text", "--content", "...", "--dedupe", "skip"]
    )
    assert result.exit_code == 2
    assert "--dedupe" in result.output


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["--content-chunks"], {"content_chunks_enabled": True, "llm_selector_enabled": None}),
        (["--no-content-chunks"], {"content_chunks_enabled": False, "llm_selector_enabled": None}),
        (["--llm-selector"], {"content_chunks_enabled": None, "llm_selector_enabled": True}),
        (["--no-llm-selector"], {"content_chunks_enabled": None, "llm_selector_enabled": False}),
        (
            ["--semantic", "0.5", "--bm25", "0.5"],
            {"content_chunks_enabled": None, "llm_selector_enabled": None},
        ),
    ],
    ids=["chunks-on", "chunks-off", "selector-on", "selector-off", "neither"],
)
@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_context_search_config_toggles(mock_client_cls, mock_config, args, expected):
    """A toggle alone is enough; ``--no-...`` sends False, which turns the setting off."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}
    mock_client = AsyncMock()
    mock_client.update_search_config.return_value = {"status": "success"}
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    result = CliRunner().invoke(main, ["context", "search-config", "uuid-1", *args])
    assert result.exit_code == 0, result.output
    kwargs = mock_client.update_search_config.await_args.kwargs
    for key, value in expected.items():
        assert kwargs[key] is value


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_context_delete(mock_client_cls, mock_config):
    """context delete should call delete_context with -y flag."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    mock_client = AsyncMock()
    mock_client.delete_context.return_value = {
        "status": "success",
        "message": "Context 'test' has been soft-deleted.",
        "context_id": "uuid-1",
        "context_name": "test",
    }
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    runner = CliRunner()
    result = runner.invoke(main, ["context", "delete", "uuid-1", "-y"])
    assert result.exit_code == 0
    mock_client.delete_context.assert_called_once_with(context_id="uuid-1")


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_context_delete_prompts_confirmation(mock_client_cls, mock_config):
    """context delete without -y should prompt for confirmation."""
    runner = CliRunner()
    result = runner.invoke(main, ["context", "delete", "uuid-1"], input="n\n")
    assert result.exit_code != 0


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_context_search_config(mock_client_cls, mock_config):
    """context search-config should call update_search_config."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    mock_client = AsyncMock()
    mock_client.update_search_config.return_value = {"status": "success"}
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    runner = CliRunner()
    result = runner.invoke(
        main, ["context", "search-config", "uuid-1", "--semantic", "0.5", "--bm25", "0.5"]
    )
    assert result.exit_code == 0
    assert "success" in result.output


@pytest.mark.parametrize("provider", ["voyage", "cohere", "self_hosted"])
@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_context_search_config_reranker_choices(mock_client_cls, mock_config, provider):
    """--reranker accepts the server's provider enum, incl. keyless self_hosted (v0.42.0+)."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    mock_client = AsyncMock()
    mock_client.update_search_config.return_value = {"status": "success"}
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    result = CliRunner().invoke(
        main, ["context", "search-config", "uuid-1", "--rerank", "--reranker", provider]
    )
    assert result.exit_code == 0, result.output
    kwargs = mock_client.update_search_config.call_args.kwargs
    assert kwargs["reranker_provider"] == provider
    assert kwargs["use_rerank"] is True


def test_context_search_config_rejects_legacy_ollama_reranker():
    """The server renamed ollama → self_hosted (v0.42.0) and rejects the old key."""
    result = CliRunner().invoke(
        main, ["context", "search-config", "uuid-1", "--reranker", "ollama"]
    )
    assert result.exit_code != 0
    assert "Invalid value for '--reranker'" in result.output


def test_context_search_config_requires_option():
    """context search-config should fail without options."""
    runner = CliRunner()
    result = runner.invoke(main, ["context", "search-config", "uuid-1"])
    assert result.exit_code != 0
    assert "At least one option" in result.output


def test_context_search_config_invalid_weight_sum():
    """context search-config should reject weights that don't sum to 1.0."""
    runner = CliRunner()
    result = runner.invoke(
        main, ["context", "search-config", "uuid-1", "--semantic", "0.5", "--bm25", "0.3"]
    )
    assert result.exit_code != 0
    assert "must sum to 1.0" in result.output


def test_context_search_config_invalid_range():
    """context search-config should reject out-of-range values."""
    runner = CliRunner()
    result = runner.invoke(main, ["context", "search-config", "uuid-1", "--semantic", "1.5"])
    assert result.exit_code != 0


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_remember_with_empty_tags(mock_client_cls, mock_config):
    """remember with empty tags should not pass empty strings."""
    mock_config.return_value = {
        "api_key": "key",
        "mcp_url": "https://test.com/mcp",
        "context_id": "ctx",
    }

    mock_client = AsyncMock()
    mock_client.remember.return_value = {"memory_id": "m1"}
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    runner = CliRunner()
    result = runner.invoke(main, ["remember", "-s", "test", "--content", "data", "--tags", ",,,"])
    assert result.exit_code == 0
    # tags should be None (filtered out), not ['', '', '']
    call_kwargs = mock_client.remember.call_args
    assert call_kwargs[1].get("tags") is None or call_kwargs[1].get("tags") == []


def _remember_mock_client(mock_client_cls):
    """Wire a mock KaguraClient whose ``remember`` returns a memory id."""
    mock_client = AsyncMock()
    mock_client.remember.return_value = {"memory_id": "m1"}
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client
    return mock_client


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_remember_passes_source_uri_and_type(mock_client_cls, mock_config):
    """remember should forward --source-uri / --source-type to the client."""
    mock_config.return_value = {
        "api_key": "key",
        "mcp_url": "https://test.com/mcp",
        "context_id": "ctx",
    }
    mock_client = _remember_mock_client(mock_client_cls)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "remember",
            "-s",
            "test",
            "--content",
            "data",
            "--source-uri",
            "file:///foo.md",
            "--source-type",
            "file",
        ],
    )
    assert result.exit_code == 0
    kwargs = mock_client.remember.call_args[1]
    assert kwargs["source_uri"] == "file:///foo.md"
    assert kwargs["source_type"] == "file"


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_remember_parses_linked_memory_ids_with_whitespace(mock_client_cls, mock_config):
    """--linked-memory-ids should split on comma and strip whitespace."""
    mock_config.return_value = {
        "api_key": "key",
        "mcp_url": "https://test.com/mcp",
        "context_id": "ctx",
    }
    mock_client = _remember_mock_client(mock_client_cls)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "remember",
            "-s",
            "test",
            "--content",
            "data",
            "--linked-memory-ids",
            "uuid-1, uuid-2 , uuid-3",
        ],
    )
    assert result.exit_code == 0
    kwargs = mock_client.remember.call_args[1]
    assert kwargs["linked_memory_ids"] == ["uuid-1", "uuid-2", "uuid-3"]


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_remember_parses_linked_source_uris(mock_client_cls, mock_config):
    """--linked-source-uris should split on comma into a list."""
    mock_config.return_value = {
        "api_key": "key",
        "mcp_url": "https://test.com/mcp",
        "context_id": "ctx",
    }
    mock_client = _remember_mock_client(mock_client_cls)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "remember",
            "-s",
            "test",
            "--content",
            "data",
            "--linked-source-uris",
            "file:///a.md,vault://v/b",
        ],
    )
    assert result.exit_code == 0
    kwargs = mock_client.remember.call_args[1]
    assert kwargs["linked_source_uris"] == ["file:///a.md", "vault://v/b"]


# ---------------------------------------------------------------------------
# remember --details / --location (Issue #241)
# ---------------------------------------------------------------------------


def _remember_cli(mock_client_cls, mock_config, *args):
    """Invoke `kagura remember` with a wired mock client and return (result, client)."""
    mock_config.return_value = {
        "api_key": "key",
        "mcp_url": "https://test.com/mcp",
        "context_id": "ctx",
    }
    mock_client = _remember_mock_client(mock_client_cls)
    runner = CliRunner()
    result = runner.invoke(main, ["remember", "-s", "test", "--content", "data", *args])
    return result, mock_client


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_remember_parses_details_json(mock_client_cls, mock_config):
    """--details accepts inline JSON and forwards it as a dict."""
    result, mock_client = _remember_cli(
        mock_client_cls,
        mock_config,
        "--details",
        '{"location": {"lat": 35.68, "lon": 139.76}, "note": "HQ"}',
    )
    assert result.exit_code == 0
    details = mock_client.remember.call_args[1]["details"]
    assert details == {"location": {"lat": 35.68, "lon": 139.76}, "note": "HQ"}


@pytest.mark.parametrize(
    "args",
    [
        (),
        ("--details", ""),
        ("--location", ""),
        ("--details", "   "),
        ("--location", "  "),
    ],
    ids=[
        "neither-flag",
        "empty-details",
        "empty-location",
        "blank-details",
        "blank-location",
    ],
)
@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_remember_details_none_when_unset(mock_client_cls, mock_config, args):
    """No flag — or a blank value, as --tags already treats it — means details=None.

    A whitespace-only value is usually an empty shell variable expanding, not a
    deliberate request; failing it would be an arbitrary split from ``""``.
    """
    result, mock_client = _remember_cli(mock_client_cls, mock_config, *args)
    assert result.exit_code == 0
    assert mock_client.remember.call_args[1]["details"] is None


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_remember_rejects_invalid_details_json(mock_client_cls, mock_config):
    """Malformed --details is a usage error, not a traceback."""
    result, mock_client = _remember_cli(mock_client_cls, mock_config, "--details", "{not json")
    assert result.exit_code != 0
    assert "Invalid JSON" in result.output
    mock_client.remember.assert_not_called()


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_remember_rejects_non_object_details(mock_client_cls, mock_config):
    """--details must be a JSON object — a list or scalar cannot carry named fields."""
    result, mock_client = _remember_cli(mock_client_cls, mock_config, "--details", "[1, 2]")
    assert result.exit_code != 0
    assert "JSON object" in result.output
    mock_client.remember.assert_not_called()


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_remember_location_shorthand(mock_client_cls, mock_config):
    """--location lat,lon builds details.location with numeric coordinates."""
    result, mock_client = _remember_cli(mock_client_cls, mock_config, "--location", "35.68,139.76")
    assert result.exit_code == 0
    details = mock_client.remember.call_args[1]["details"]
    assert details == {"location": {"lat": 35.68, "lon": 139.76}}
    assert isinstance(details["location"]["lat"], float)


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_remember_location_shorthand_with_label(mock_client_cls, mock_config):
    """--location lat,lon,label attaches the optional label."""
    result, mock_client = _remember_cli(
        mock_client_cls, mock_config, "--location", "35.68, 139.76, Tokyo HQ"
    )
    assert result.exit_code == 0
    details = mock_client.remember.call_args[1]["details"]
    assert details["location"] == {"lat": 35.68, "lon": 139.76, "label": "Tokyo HQ"}


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_remember_location_trailing_comma_omits_label(mock_client_cls, mock_config):
    """'lat,lon,' is the 2-field form with a stray comma — no empty label key."""
    result, mock_client = _remember_cli(mock_client_cls, mock_config, "--location", "35.68,139.76,")
    assert result.exit_code == 0
    assert mock_client.remember.call_args[1]["details"]["location"] == {
        "lat": 35.68,
        "lon": 139.76,
    }


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_remember_location_merges_into_details(mock_client_cls, mock_config):
    """--location and --details compose: other details keys are preserved."""
    result, mock_client = _remember_cli(
        mock_client_cls,
        mock_config,
        "--details",
        '{"note": "HQ"}',
        "--location",
        "35.68,139.76",
    )
    assert result.exit_code == 0
    details = mock_client.remember.call_args[1]["details"]
    assert details == {"note": "HQ", "location": {"lat": 35.68, "lon": 139.76}}


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_remember_rejects_conflicting_location(mock_client_cls, mock_config):
    """location in --details plus --location is ambiguous — fail, don't silently drop one."""
    result, mock_client = _remember_cli(
        mock_client_cls,
        mock_config,
        "--details",
        '{"location": {"lat": 1.0, "lon": 2.0}}',
        "--location",
        "35.68,139.76",
    )
    assert result.exit_code != 0
    assert "--location" in result.output
    mock_client.remember.assert_not_called()


@pytest.mark.parametrize(
    "bad",
    ["35.68", "abc,139.76", "35.68,139.76,label,extra", "91.0,139.76", "35.68,181.0"],
)
@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_remember_rejects_malformed_location(mock_client_cls, mock_config, bad):
    """--location rejects wrong arity, non-numeric, and out-of-range coordinates."""
    result, mock_client = _remember_cli(mock_client_cls, mock_config, "--location", bad)
    assert result.exit_code != 0
    mock_client.remember.assert_not_called()


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_remember_omits_provenance_when_not_given(mock_client_cls, mock_config):
    """Without provenance flags, remember must not stamp source_type/uri/links.

    Guards against a silent behavior change: a plain ``kagura remember`` should
    pass None for these so the server stores no misleading provenance.
    """
    mock_config.return_value = {
        "api_key": "key",
        "mcp_url": "https://test.com/mcp",
        "context_id": "ctx",
    }
    mock_client = _remember_mock_client(mock_client_cls)

    runner = CliRunner()
    result = runner.invoke(main, ["remember", "-s", "test", "--content", "data"])
    assert result.exit_code == 0
    kwargs = mock_client.remember.call_args[1]
    assert kwargs["source_uri"] is None
    assert kwargs["source_type"] is None
    assert kwargs["linked_memory_ids"] is None
    assert kwargs["linked_source_uris"] is None


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.ResourceClient")
def test_resource_stats(mock_rc_cls, mock_config):
    """resource stats should call get_resource_impact."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    mock_rc = AsyncMock()
    json_out = '{"token_count": 2, "memory_count": 50}'
    mock_rc.get_resource_impact.return_value = MagicMock(
        model_dump_json=lambda indent=None: json_out
    )
    mock_rc.__aenter__ = AsyncMock(return_value=mock_rc)
    mock_rc.__aexit__ = AsyncMock(return_value=None)
    mock_rc_cls.from_mcp_url.return_value = mock_rc
    mock_rc_cls._from_resolved_auth.return_value = mock_rc

    runner = CliRunner()
    result = runner.invoke(main, ["resource", "stats", "-r", "products"])
    assert result.exit_code == 0
    assert "token_count" in result.output
    mock_rc.get_resource_impact.assert_called_once_with("products")


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.ResourceClient")
def test_resource_schema(mock_rc_cls, mock_config):
    """resource schema should call get_resource_schema."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    mock_rc = AsyncMock()
    json_out = '{"resource_id": "products", "schema_version": 1}'
    mock_rc.get_resource_schema.return_value = MagicMock(
        model_dump_json=lambda indent=None: json_out
    )
    mock_rc.__aenter__ = AsyncMock(return_value=mock_rc)
    mock_rc.__aexit__ = AsyncMock(return_value=None)
    mock_rc_cls.from_mcp_url.return_value = mock_rc
    mock_rc_cls._from_resolved_auth.return_value = mock_rc

    runner = CliRunner()
    result = runner.invoke(main, ["resource", "schema", "-r", "products"])
    assert result.exit_code == 0
    assert "schema_version" in result.output
    mock_rc.get_resource_schema.assert_called_once_with("products", schema_version=None)


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.ResourceClient")
def test_resource_schema_not_registered(mock_rc_cls, mock_config):
    """resource schema should show message when schema is not registered."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    mock_rc = AsyncMock()
    mock_rc.get_resource_schema.return_value = None
    mock_rc.__aenter__ = AsyncMock(return_value=mock_rc)
    mock_rc.__aexit__ = AsyncMock(return_value=None)
    mock_rc_cls.from_mcp_url.return_value = mock_rc
    mock_rc_cls._from_resolved_auth.return_value = mock_rc

    runner = CliRunner()
    result = runner.invoke(main, ["resource", "schema", "-r", "no-schema"])
    assert result.exit_code == 0
    assert "No schema registered" in result.output


def _setup_resource_client(mock_rc_cls) -> AsyncMock:
    """Route ``kagura resource setup`` to a ResourceClient mock that returns a token."""
    mock_rc = AsyncMock()
    mock_rc.setup_resource.return_value = MagicMock(
        model_dump_json=lambda indent=None: '{"token": "kagura_resource_abc", "id": 1}'
    )
    mock_rc.__aenter__ = AsyncMock(return_value=mock_rc)
    mock_rc.__aexit__ = AsyncMock(return_value=None)
    mock_rc_cls.from_mcp_url.return_value = mock_rc
    mock_rc_cls._from_resolved_auth.return_value = mock_rc
    return mock_rc


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.ResourceClient")
def test_resource_setup(mock_rc_cls, mock_config):
    """resource setup names the context after the resource unless --name is given (#273)."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}
    mock_rc = _setup_resource_client(mock_rc_cls)

    runner = CliRunner()
    result = runner.invoke(main, ["resource", "setup", "-r", "products"])
    assert result.exit_code == 0, result.output
    assert "kagura_resource_abc" in result.output
    assert result.stderr == ""
    mock_rc.setup_resource.assert_called_once_with(
        resource_id="products", context_name=None, description=None, quota_events_per_hour=1000
    )


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.ResourceClient")
def test_resource_setup_name(mock_rc_cls, mock_config):
    """--name sets the context name the server requires (#273)."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}
    mock_rc = _setup_resource_client(mock_rc_cls)

    runner = CliRunner()
    result = runner.invoke(
        main, ["resource", "setup", "-r", "products", "--name", "product-catalog", "-q", "50"]
    )
    assert result.exit_code == 0, result.output
    mock_rc.setup_resource.assert_called_once_with(
        resource_id="products",
        context_name="product-catalog",
        description=None,
        quota_events_per_hour=50,
    )


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.ResourceClient")
def test_resource_setup_summary_is_ignored_with_a_note(mock_rc_cls, mock_config):
    """--summary is still accepted, but not sent; stderr names `kagura context update` (#273)."""
    from kagura_memory.cli import _SETUP_SUMMARY_IGNORED_NOTE

    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}
    mock_rc = _setup_resource_client(mock_rc_cls)

    runner = CliRunner()
    result = runner.invoke(main, ["resource", "setup", "-r", "products", "-s", "catalog"])
    assert result.exit_code == 0, result.output
    assert result.stderr.strip() == _SETUP_SUMMARY_IGNORED_NOTE
    assert "kagura context update <context_id> --summary" in _SETUP_SUMMARY_IGNORED_NOTE
    assert "kagura_resource_abc" in result.stdout
    assert "summary" not in mock_rc.setup_resource.call_args.kwargs


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.ResourceClient")
def test_resource_list(mock_rc_cls, mock_config):
    """resource list should call list_resources."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    mock_rc = AsyncMock()
    json_out = '{"resources": [], "total": 0}'
    mock_rc.list_resources.return_value = MagicMock(model_dump_json=lambda indent=None: json_out)
    mock_rc.__aenter__ = AsyncMock(return_value=mock_rc)
    mock_rc.__aexit__ = AsyncMock(return_value=None)
    mock_rc_cls.from_mcp_url.return_value = mock_rc
    mock_rc_cls._from_resolved_auth.return_value = mock_rc

    runner = CliRunner()
    result = runner.invoke(main, ["resource", "list"])
    assert result.exit_code == 0
    assert '"total": 0' in result.output
    mock_rc.list_resources.assert_called_once_with()


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.ResourceClient")
def test_resource_indexer_status(mock_rc_cls, mock_config):
    """resource indexer-status should call get_indexer_status."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    mock_rc = AsyncMock()
    json_out = '{"resource_id": "products", "state": null, "recent_events": []}'
    mock_rc.get_indexer_status.return_value = MagicMock(
        model_dump_json=lambda indent=None: json_out
    )
    mock_rc.__aenter__ = AsyncMock(return_value=mock_rc)
    mock_rc.__aexit__ = AsyncMock(return_value=None)
    mock_rc_cls.from_mcp_url.return_value = mock_rc
    mock_rc_cls._from_resolved_auth.return_value = mock_rc

    runner = CliRunner()
    result = runner.invoke(main, ["resource", "indexer-status", "-r", "products"])
    assert result.exit_code == 0
    assert '"resource_id"' in result.output
    mock_rc.get_indexer_status.assert_called_once_with("products")


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.ResourceClient")
def test_resource_import_csv(mock_rc_cls, mock_config):
    """resource import should parse CSV and batch ingest."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    mock_rc = AsyncMock()
    mock_rc.ingest_events.return_value = MagicMock(created_count=3, failed_count=0)
    mock_rc.__aenter__ = AsyncMock(return_value=mock_rc)
    mock_rc.__aexit__ = AsyncMock(return_value=None)
    mock_rc_cls.from_mcp_url.return_value = mock_rc
    mock_rc_cls._from_resolved_auth.return_value = mock_rc

    csv_content = "name,price\nWidget,9.99\nGadget,19.99\nThing,29.99"
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["resource", "import", "-r", "products", "-k", "TOKEN", "--format", "csv"],
        input=csv_content,
    )
    assert result.exit_code == 0
    assert '"created": 3' in result.output


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.ResourceClient")
def test_resource_import_jsonl(mock_rc_cls, mock_config):
    """resource import should parse JSONL."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    mock_rc = AsyncMock()
    mock_rc.ingest_events.return_value = MagicMock(created_count=2, failed_count=0)
    mock_rc.__aenter__ = AsyncMock(return_value=mock_rc)
    mock_rc.__aexit__ = AsyncMock(return_value=None)
    mock_rc_cls.from_mcp_url.return_value = mock_rc
    mock_rc_cls._from_resolved_auth.return_value = mock_rc

    jsonl = '{"name":"A"}\n{"name":"B"}'
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["resource", "import", "-r", "res", "-k", "KEY", "--format", "jsonl"],
        input=jsonl,
    )
    assert result.exit_code == 0
    assert '"created": 2' in result.output


def test_resource_import_no_format_stdin():
    """resource import from stdin without --format should error."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["resource", "import", "-r", "res", "-k", "KEY"],
        input="some data",
    )
    assert result.exit_code != 0
    assert "Cannot detect format" in result.output


def test_resource_import_bad_id_column():
    """resource import with nonexistent --id-column should error."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["resource", "import", "-r", "res", "-k", "KEY", "--format", "csv", "--id-column", "nope"],
        input="name,price\nWidget,9.99",
    )
    assert result.exit_code != 0
    assert "not found" in result.output


def test_resource_import_json_non_object_items():
    """resource import should reject JSON array with non-object items."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["resource", "import", "-r", "res", "-k", "KEY", "--format", "json"],
        input="[1, 2, 3]",
    )
    assert result.exit_code != 0
    assert "not an object" in result.output


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.ResourceClient")
def test_resource_import_json(mock_rc_cls, mock_config):
    """resource import should parse JSON array."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    mock_rc = AsyncMock()
    mock_rc.ingest_events.return_value = MagicMock(created_count=2, failed_count=0)
    mock_rc.__aenter__ = AsyncMock(return_value=mock_rc)
    mock_rc.__aexit__ = AsyncMock(return_value=None)
    mock_rc_cls.from_mcp_url.return_value = mock_rc
    mock_rc_cls._from_resolved_auth.return_value = mock_rc

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["resource", "import", "-r", "res", "-k", "KEY", "--format", "json"],
        input='[{"name":"A"},{"name":"B"}]',
    )
    assert result.exit_code == 0
    assert '"created": 2' in result.output


def test_resource_ingest_invalid_payload_surfaces_json_error():
    """`resource ingest -p '<not json>'` translates JSONDecodeError via _exc_message (#130)."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "resource",
            "ingest",
            "-r",
            "products",
            "-k",
            "TOKEN",
            "--doc-id",
            "SKU-1",
            "-p",
            "not json",
        ],
    )
    assert result.exit_code != 0
    assert "Invalid JSON payload" in result.output


def test_resource_ingest_batch_invalid_json_file_surfaces_error(tmp_path):
    """`resource ingest-batch -f <bad json>` translates JSONDecodeError via _exc_message (#130)."""
    bad = tmp_path / "events.json"
    bad.write_text("not json")
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["resource", "ingest-batch", "-r", "res", "-k", "TOKEN", "-f", str(bad)],
    )
    assert result.exit_code != 0
    assert "Invalid JSON file" in result.output


def test_resource_import_read_failure_surfaces_class_name(monkeypatch):
    """`resource import` translates a read() OSError into ClickException via _exc_message (#130)."""
    import io

    class _RaisingFile(io.StringIO):
        def read(self, *_args, **_kwargs):
            raise OSError("disk gone")

    runner = CliRunner()
    # Inject a file-like that raises on read by monkeypatching click.File.convert
    # to swap the wrapped file just before resource_import reads it.
    real_convert = __import__("click").File.convert

    def _fake_convert(self, value, param, ctx):
        if isinstance(value, str):
            return _RaisingFile("placeholder")
        return real_convert(self, value, param, ctx)

    monkeypatch.setattr("click.File.convert", _fake_convert)

    result = runner.invoke(
        main,
        ["resource", "import", "-r", "res", "-k", "TOKEN", "--format", "json", "-f", "any.json"],
    )
    assert result.exit_code != 0
    assert "Failed to read input" in result.output
    assert "disk gone" in result.output


def test_resource_import_invalid_json():
    """resource import with invalid JSON should error."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["resource", "import", "-r", "res", "-k", "KEY", "--format", "json"],
        input="not json",
    )
    assert result.exit_code != 0
    assert "Invalid JSON" in result.output


def test_resource_import_json_not_array():
    """resource import with JSON object (not array) should error."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["resource", "import", "-r", "res", "-k", "KEY", "--format", "json"],
        input='{"not": "array"}',
    )
    assert result.exit_code != 0
    assert "must be an array" in result.output


def test_resource_import_invalid_jsonl():
    """resource import with invalid JSONL line should error."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["resource", "import", "-r", "res", "-k", "KEY", "--format", "jsonl"],
        input='{"ok":1}\nnot json\n{"ok":2}',
    )
    assert result.exit_code != 0
    assert "Invalid JSONL at line 2" in result.output


def test_resource_import_empty_data():
    """resource import with empty input should error."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["resource", "import", "-r", "res", "-k", "KEY", "--format", "jsonl"],
        input="",
    )
    assert result.exit_code != 0
    assert "No data found" in result.output


def test_resource_import_auto_detect_csv(tmp_path):
    """resource import should auto-detect CSV from extension."""
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("name\nWidget")

    runner = CliRunner()
    # Will fail at ResourceClient but format detection should pass
    result = runner.invoke(
        main,
        ["resource", "import", "-r", "res", "-k", "KEY", "-f", str(csv_file)],
    )
    # It gets past format detection (would fail at connection)
    assert "Cannot detect format" not in (result.output or "")


# ============================================================================
# Sleep Maintenance CLI (issue #85)
# ============================================================================


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_sleep_history(mock_client_cls, mock_config):
    """`kagura sleep history` echoes a JSON object with a top-level ``reports`` array."""
    from kagura_memory import SleepReport

    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    mock_client = AsyncMock()
    mock_client.get_sleep_history.return_value = [SleepReport(**sleep_report_summary_dict("rid-9"))]
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    runner = CliRunner()
    result = runner.invoke(main, ["sleep", "history", "ctx-1", "--limit", "5"])
    assert result.exit_code == 0, result.output
    assert "rid-9" in result.output
    mock_client.get_sleep_history.assert_called_once_with(context_id="ctx-1", limit=5, source=None)


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_sleep_report(mock_client_cls, mock_config):
    """`kagura sleep report` echoes the flattened detail JSON."""
    from kagura_memory import SleepReportDetail

    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    detail = SleepReportDetail(
        **sleep_report_summary_dict("rid-9"),
        memories_flagged=0,
        embedding_calls_made=2,
        actions=[],
        action_count=0,
    )
    mock_client = AsyncMock()
    mock_client.get_sleep_report.return_value = detail
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    runner = CliRunner()
    result = runner.invoke(main, ["sleep", "report", "ctx-1", "rid-9"])
    assert result.exit_code == 0, result.output
    assert "rid-9" in result.output
    assert "memories_flagged" in result.output
    mock_client.get_sleep_report.assert_called_once_with(
        context_id="ctx-1", report_id="rid-9", actions_cursor=None, actions_limit=None
    )


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_sleep_rollback_with_yes_flag(mock_client_cls, mock_config):
    """`kagura sleep rollback -y` skips both the prompt and the pre-fetch."""
    from kagura_memory import RollbackResult, RollbackSummary

    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    rollback = RollbackResult(
        report_id="rid-9",
        status="rolled_back",
        rollback_summary=RollbackSummary(edges_deleted=2, merges_reversed=1),
    )

    mock_client = AsyncMock()
    mock_client.rollback_sleep_run.return_value = rollback
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    runner = CliRunner()
    result = runner.invoke(main, ["sleep", "rollback", "ctx-1", "rid-9", "-y"])
    assert result.exit_code == 0, result.output
    assert "rolled_back" in result.output
    mock_client.rollback_sleep_run.assert_called_once_with(context_id="ctx-1", report_id="rid-9")
    # --yes must skip the cosmetic pre-fetch — no wasted round trip.
    mock_client.get_sleep_report.assert_not_called()


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_sleep_rollback_aborts_on_no(mock_client_cls, mock_config):
    """Without -y, answering 'n' aborts before calling rollback_sleep_run."""
    from kagura_memory import SleepReportDetail

    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    detail = SleepReportDetail(
        **sleep_report_summary_dict("rid-9"),
        memories_flagged=0,
        embedding_calls_made=2,
        actions=[],
        action_count=3,
    )
    mock_client = AsyncMock()
    mock_client.get_sleep_report.return_value = detail
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    runner = CliRunner()
    result = runner.invoke(main, ["sleep", "rollback", "ctx-1", "rid-9"], input="n\n")
    assert result.exit_code != 0
    mock_client.rollback_sleep_run.assert_not_called()


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_sleep_rollback_wraps_unexpected_exception(mock_client_cls, mock_config):
    """Unexpected exceptions inside _run() surface as click.ClickException."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    mock_client = AsyncMock()
    mock_client.rollback_sleep_run.side_effect = RuntimeError("boom")
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    runner = CliRunner()
    result = runner.invoke(main, ["sleep", "rollback", "ctx-1", "rid-9", "-y"])
    assert result.exit_code != 0
    assert "Error: boom" in result.output


# ----------------------------------------------------------------------------
# Degraded runs end-to-end (#250): the real KaguraClient parses the server
# payload, so these pin the CLI against the wire shape — not a pre-built model.
# ----------------------------------------------------------------------------


def _degraded_tool_result(tool_name: str, _arguments: dict) -> dict:
    """Server-shaped MCP results for a ``degraded`` run (memory-cloud v0.43.0+)."""
    summary = {**sleep_report_summary_dict("rid-d"), "status": "degraded", "llm_call_failures": 2}
    if tool_name == "get_sleep_history":
        return {"status": "success", "reports": [sleep_report_summary_dict("rid-c"), summary]}
    if tool_name == "get_sleep_report":
        report = sleep_report_detail_dict(
            "rid-d", status="degraded", llm_call_failures=2, merge_retention_result={"purged": 1}
        )
        return {"status": "success", "report": report, "actions": [], "action_count": 4}
    if tool_name == "rollback_sleep_run":
        return {
            "status": "rolled_back",
            "report_id": "rid-d",
            "rollback_summary": {"edges_deleted": 4, "errors": []},
        }
    raise AssertionError(f"unexpected tool {tool_name}")


@pytest.fixture
def degraded_mcp(monkeypatch):
    """Real KaguraClient from config, with ``_call_tool`` answering as the server."""
    from kagura_memory.client import KaguraClient

    monkeypatch.setattr(
        "kagura_memory.cli.load_config",
        lambda: {"api_key": "key", "mcp_url": "https://test.com/mcp"},
    )
    mock = AsyncMock(side_effect=_degraded_tool_result)
    monkeypatch.setattr(KaguraClient, "_call_tool", mock)
    return mock


def test_sleep_history_lists_degraded_run(degraded_mcp):
    result = CliRunner().invoke(main, ["sleep", "history", "ctx-1"])
    assert result.exit_code == 0, result.output
    assert '"status": "degraded"' in result.output
    assert '"llm_call_failures": 2' in result.output
    assert "rid-c" in result.output


def test_sleep_report_shows_degraded_run(degraded_mcp):
    result = CliRunner().invoke(main, ["sleep", "report", "ctx-1", "rid-d"])
    assert result.exit_code == 0, result.output
    assert '"status": "degraded"' in result.output
    assert '"merge_retention_result"' in result.output
    # CONTEXT_ID is not sent: memory-cloud v0.82.0+ refuses it (#296).
    degraded_mcp.assert_called_once_with("get_sleep_report", {"report_id": "rid-d"})


def test_sleep_history_source_option_is_sent(degraded_mcp):
    """``--source`` reaches get_sleep_history (memory-cloud v0.138.0+, #2219)."""
    result = CliRunner().invoke(main, ["sleep", "history", "ctx-1", "--source", "all"])
    assert result.exit_code == 0, result.output
    degraded_mcp.assert_called_once_with(
        "get_sleep_history", {"context_id": "ctx-1", "limit": 10, "source": "all"}
    )


def test_sleep_history_rejects_an_unknown_source(degraded_mcp):
    result = CliRunner().invoke(main, ["sleep", "history", "ctx-1", "--source", "everything"])
    assert result.exit_code == 2
    degraded_mcp.assert_not_called()


def test_sleep_report_paging_options_are_sent(degraded_mcp):
    """``--actions-cursor`` / ``--actions-limit`` page the audit log (v0.82.0+, #1743)."""
    result = CliRunner().invoke(
        main,
        ["sleep", "report", "ctx-1", "rid-d", "--actions-cursor", "41", "--actions-limit", "20"],
    )
    assert result.exit_code == 0, result.output
    degraded_mcp.assert_called_once_with(
        "get_sleep_report", {"report_id": "rid-d", "actions_cursor": "41", "actions_limit": 20}
    )


def test_sleep_report_rejects_an_actions_limit_over_200(degraded_mcp):
    result = CliRunner().invoke(
        main, ["sleep", "report", "ctx-1", "rid-d", "--actions-limit", "201"]
    )
    assert result.exit_code == 2
    degraded_mcp.assert_not_called()


def test_sleep_rollback_prefetches_degraded_run_for_confirm(degraded_mcp):
    """Without -y the confirm prompt pre-fetches the report — degraded must parse."""
    result = CliRunner().invoke(main, ["sleep", "rollback", "ctx-1", "rid-d"], input="y\n")
    assert result.exit_code == 0, result.output
    assert "4 action(s)" in result.output
    assert '"status": "rolled_back"' in result.output
    # CONTEXT_ID is not sent to either tool: memory-cloud v0.82.0+ refuses it (#296).
    assert [c.args for c in degraded_mcp.call_args_list] == [
        ("get_sleep_report", {"report_id": "rid-d"}),
        ("rollback_sleep_run", {"report_id": "rid-d"}),
    ]


def test_sleep_history_drift_is_a_clean_error_naming_the_operation(monkeypatch):
    """A payload the SDK cannot parse is an ``Error:`` line, not a pydantic traceback."""
    from kagura_memory.client import KaguraClient

    monkeypatch.setattr(
        "kagura_memory.cli.load_config",
        lambda: {"api_key": "key", "mcp_url": "https://test.com/mcp"},
    )
    monkeypatch.setattr(
        KaguraClient,
        "_call_tool",
        AsyncMock(return_value={"status": "success", "reports": [{"report_id": "x"}]}),
    )
    result = CliRunner().invoke(main, ["sleep", "history", "ctx-1"])
    assert result.exit_code != 0
    assert "Error: get_sleep_history: " in result.output
    assert "Traceback" not in result.output


# ----------------------------------------------------------------------------
# Typed gate / partial-rollback errors (#256)
# ----------------------------------------------------------------------------

_ROLLBACK_SUMMARY = {
    "edges_deleted": 3,
    "merges_reversed": 1,
    "merges_unreversible": 0,
    "importance_restored": 4,
    "promotions_reversed": 1,
    "importance_kept": 1,
    "promotions_kept": 5,
    "archives_restored": 0,
    "errors": [],
}


@pytest.fixture
def rollback_mcp(monkeypatch):
    """Real KaguraClient from config; the test sets ``_call_tool``'s return value."""
    from kagura_memory.client import KaguraClient

    monkeypatch.setattr(
        "kagura_memory.cli.load_config",
        lambda: {"api_key": "key", "mcp_url": "https://test.com/mcp"},
    )
    mock = AsyncMock()
    monkeypatch.setattr(KaguraClient, "_call_tool", mock)
    return mock


def test_sleep_rollback_shows_the_unreversible_and_kept_counters(rollback_mcp):
    rollback_mcp.return_value = {
        "status": "success",
        "report_id": "rid-9",
        "rollback_summary": _ROLLBACK_SUMMARY,
    }
    result = CliRunner().invoke(main, ["sleep", "rollback", "ctx-1", "rid-9", "-y"])
    assert result.exit_code == 0, result.output
    assert '"merges_unreversible": 0' in result.output
    assert '"importance_kept": 1' in result.output
    assert '"promotions_kept": 5' in result.output


def test_sleep_rollback_partial_prints_what_was_reversed(rollback_mcp):
    """A partial rollback exits non-zero but still shows the per-category summary."""
    rollback_mcp.return_value = {
        "status": "error",
        "error": "partial_rollback",
        "message": "Rollback completed with 1 error(s).",
        "report_id": "rid-9",
        "rollback_summary": {
            **_ROLLBACK_SUMMARY,
            "merges_unreversible": 1,
            "errors": ["shadow merge m-1 → m-2 not reversed"],
        },
    }
    result = CliRunner().invoke(main, ["sleep", "rollback", "ctx-1", "rid-9", "-y"])
    assert result.exit_code != 0
    assert '"error": "partial_rollback"' in result.output
    assert '"merges_reversed": 1' in result.output
    assert '"merges_unreversible": 1' in result.output
    assert "shadow merge m-1 → m-2 not reversed" in result.output
    assert "Error: rollback_sleep_run failed (partial_rollback)" in result.output


def test_sleep_rollback_partial_with_an_unreadable_summary(rollback_mcp):
    """Summary drift still reports the partial rollback, with a null summary."""
    rollback_mcp.return_value = {
        "status": "error",
        "error": "partial_rollback",
        "message": "Rollback completed with 1 error(s).",
        "report_id": "rid-9",
        "rollback_summary": {"edges_deleted": "three"},
    }
    result = CliRunner().invoke(main, ["sleep", "rollback", "ctx-1", "rid-9", "-y"])
    assert result.exit_code != 0
    assert '"rollback_summary": null' in result.output
    assert "(partial_rollback)" in result.output
    assert "rollback_summary could not be read" in result.output
    assert "Traceback" not in result.output


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_quota_error_shows_reset_time_and_required_plan(mock_client_cls, mock_config):
    from datetime import UTC, datetime

    from kagura_memory.exceptions import KaguraQuotaError

    mock_config.return_value = {
        "api_key": "key",
        "mcp_url": "https://test.com/mcp",
        "context_id": "ctx-1",
    }
    mock_client = AsyncMock()
    mock_client.remember.side_effect = KaguraQuotaError(
        "remember failed (quota_exceeded): Daily memory limit reached.",
        quota_type="memories_per_day",
        resets_at=datetime(2099, 1, 2, tzinfo=UTC),
        required_plan="pro",
        required_plan_display="L",
    )
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    result = CliRunner().invoke(main, ["remember", "-s", "s", "--content", "c"])
    assert result.exit_code != 0
    assert "Error: remember failed (quota_exceeded): Daily memory limit reached." in result.output
    assert "Resets at: 2099-01-02T00:00:00+00:00" in result.output
    assert "Required plan: L (pro)" in result.output


def test_feature_error_shows_required_plan(monkeypatch):
    from kagura_memory import KaguraFeatureNotAvailableError

    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.create_token.side_effect = KaguraFeatureNotAvailableError(
        "Feature 'resources' not available on M plan.", feature="resources", required_plan="promax"
    )
    monkeypatch.setattr("kagura_memory.cli._get_resource_client", lambda: client)

    result = CliRunner().invoke(main, ["resource", "tokens", "create", "-r", "products"])
    assert result.exit_code != 0
    assert "Feature 'resources' not available on M plan." in result.output
    assert "Required plan: promax" in result.output
    assert "Resets at" not in result.output


def test_guardrails_digest_quota_error_shows_the_gate_lines():
    """The REST guardrails digest renders the gate lines like the shared runners."""
    from kagura_memory.exceptions import KaguraQuotaError

    config = {"api_key": "key", "mcp_url": "https://test.com/mcp", "context_id": "ctx-1"}
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.get_guardrail_digest.side_effect = KaguraQuotaError(
        "Daily MCP quota exceeded.", 86400, quota_type="api_mcp_daily", required_plan="pro"
    )
    with (
        patch("kagura_memory.cli.load_config", return_value=config),
        patch("kagura_memory.cli.MemoryClient") as mock_cls,
    ):
        mock_cls._from_resolved_auth.return_value = client
        result = CliRunner().invoke(main, ["guardrails", "digest", "ctx-1"])
    assert result.exit_code == 1
    assert "Error: Daily MCP quota exceeded." in result.output
    assert "Required plan: pro" in result.output


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_plain_errors_render_without_gate_lines(mock_client_cls, mock_config):
    from kagura_memory.exceptions import KaguraError

    mock_config.return_value = {
        "api_key": "key",
        "mcp_url": "https://test.com/mcp",
        "context_id": "ctx-1",
    }
    mock_client = AsyncMock()
    mock_client.remember.side_effect = KaguraError("remember failed (validation_error): bad")
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    result = CliRunner().invoke(main, ["remember", "-s", "s", "--content", "c"])
    assert result.exit_code != 0
    assert "Error: remember failed (validation_error): bad" in result.output
    assert "Required plan" not in result.output
    assert "Resets at" not in result.output


def test_resource_indexer_status_shows_quota_deferred_run(monkeypatch):
    """`memories_per_day_exceeded` (server v0.68.0+, #1549) renders instead of erroring."""
    import httpx

    from kagura_memory import ResourceClient

    body = indexer_status_dict(skipped_reason="memories_per_day_exceeded")
    client = ResourceClient(api_key="key", base_url="https://test.com")
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=body))
    )
    monkeypatch.setattr("kagura_memory.cli._get_resource_client", lambda: client)

    result = CliRunner().invoke(main, ["resource", "indexer-status", "-r", "products"])
    assert result.exit_code == 0, result.output
    assert '"skipped_reason": "memories_per_day_exceeded"' in result.output


# ============================================================================
# Measurement lane CLI — `kagura measure` (Issue #254)
# ============================================================================


def _measure_mock_client(mock_client_cls, mock_config):
    """Wire a KaguraClient mock whose measurement methods return real models."""
    from kagura_memory import MeasurementResult, MeasurementSeries

    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}
    mock_client = AsyncMock()
    mock_client.record_measurement.return_value = MeasurementResult.model_validate(
        measurement_dict()
    )
    mock_client.recall_series.return_value = MeasurementSeries.model_validate(
        measurement_series_dict()
    )
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client
    return mock_client


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_measure_record(mock_client_cls, mock_config):
    """`kagura measure record` forwards unit/--at and echoes the result JSON."""
    mock_client = _measure_mock_client(mock_client_cls, mock_config)

    result = CliRunner().invoke(
        main,
        [
            "measure",
            "record",
            "ctx-1",
            "weight_kg",
            "71.5",
            "--unit",
            "kg",
            "--at",
            "2026-09-01T07:30:00Z",
        ],
    )
    assert result.exit_code == 0, result.output
    mock_client.record_measurement.assert_awaited_once_with(
        context_id="ctx-1",
        metric="weight_kg",
        value=71.5,
        measured_at="2026-09-01T07:30:00Z",
        unit="kg",
    )
    payload = json.loads(result.output)
    assert payload["measurement_id"] == "cccccccc-dddd-eeee-ffff-000000000000"
    assert payload["value"] == 71.5
    assert payload["unit"] == "kg"


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_measure_record_omits_unset_options(mock_client_cls, mock_config):
    """Without --unit/--at, None is passed so the client omits them from the wire."""
    mock_client = _measure_mock_client(mock_client_cls, mock_config)

    result = CliRunner().invoke(main, ["measure", "record", "ctx-1", "reps", "12"])
    assert result.exit_code == 0, result.output
    mock_client.record_measurement.assert_awaited_once_with(
        context_id="ctx-1", metric="reps", value=12.0, measured_at=None, unit=None
    )


@pytest.mark.parametrize("raw", ["-3.5", "-1e3"])
@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_measure_record_accepts_negative_value(mock_client_cls, mock_config, raw):
    """A negative value is a value, not an unknown option (P&L, temperature deltas)."""
    mock_client = _measure_mock_client(mock_client_cls, mock_config)

    result = CliRunner().invoke(main, ["measure", "record", "ctx-1", "pnl", raw, "--unit", "USD"])
    assert result.exit_code == 0, result.output
    assert mock_client.record_measurement.await_args.kwargs["value"] == float(raw)
    assert mock_client.record_measurement.await_args.kwargs["unit"] == "USD"


@pytest.mark.parametrize(
    ("args", "token"),
    [
        # Trailing: lands as an extra positional, which click itself refuses.
        (["ctx-1", "m", "1", "--unti", "kg"], None),
        # METRIC slot: would otherwise append a junk "--weight" series for good.
        (["ctx-1", "--weight", "71.5"], "--weight"),
        (["ctx-1", "--metric=weight_kg", "71.5"], "--metric=weight_kg"),
        (["ctx-1", "-x", "5"], "-x"),
        # CONTEXT_ID slot.
        (["-c", "dev", "71.5"], "-c"),
    ],
)
@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_measure_record_rejects_unknown_option(mock_client_cls, mock_config, args, token):
    """Tolerating negative numbers must not swallow a mistyped option in any slot."""
    mock_client = _measure_mock_client(mock_client_cls, mock_config)

    result = CliRunner().invoke(main, ["measure", "record", *args])
    assert result.exit_code == 2, result.output
    if token is not None:
        assert f"No such option: {token}" in result.output
    mock_client.record_measurement.assert_not_awaited()


def test_measure_record_rejects_non_numeric_value():
    """A non-numeric value is a click usage error, before any client is built."""
    result = CliRunner().invoke(main, ["measure", "record", "ctx-1", "m", "heavy"])
    assert result.exit_code == 2
    assert "not a valid float" in result.output


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_measure_record_surfaces_client_error(mock_client_cls, mock_config):
    """Client-side validation errors (e.g. NaN) surface as a clean CLI error."""
    mock_client = _measure_mock_client(mock_client_cls, mock_config)
    mock_client.record_measurement.side_effect = ValueError("value must be finite")

    result = CliRunner().invoke(main, ["measure", "record", "ctx-1", "m", "nan"])
    assert result.exit_code == 1
    assert "Error: value must be finite" in result.output


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_measure_series(mock_client_cls, mock_config):
    """`kagura measure series` forwards every filter and echoes the series JSON."""
    mock_client = _measure_mock_client(mock_client_cls, mock_config)

    result = CliRunner().invoke(
        main,
        [
            "measure",
            "series",
            "ctx-1",
            "weight_kg",
            "--period",
            "week",
            "--agg",
            "last",
            "--start",
            "2026-08-01T00:00:00",
            "--end",
            "2026-09-01T00:00:00",
        ],
    )
    assert result.exit_code == 0, result.output
    mock_client.recall_series.assert_awaited_once_with(
        context_id="ctx-1",
        metric="weight_kg",
        period="week",
        agg="last",
        start="2026-08-01T00:00:00",
        end="2026-09-01T00:00:00",
    )
    payload = json.loads(result.output)
    assert payload["count"] == 2
    assert payload["series"][0] == {"bucket": "2026-08-24T00:00:00Z", "value": 72.0, "count": 3}


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_measure_series_defaults_defer_to_server(mock_client_cls, mock_config):
    """With no options, None is passed so the server defaults (day/avg/30d) apply."""
    mock_client = _measure_mock_client(mock_client_cls, mock_config)

    result = CliRunner().invoke(main, ["measure", "series", "ctx-1", "weight_kg"])
    assert result.exit_code == 0, result.output
    mock_client.recall_series.assert_awaited_once_with(
        context_id="ctx-1", metric="weight_kg", period=None, agg=None, start=None, end=None
    )


@pytest.mark.parametrize(
    ("option", "value"), [("--period", "year"), ("--agg", "median")], ids=["period", "agg"]
)
def test_measure_series_rejects_unknown_choice(option, value):
    """--period/--agg are closed choices, rejected before any round-trip."""
    result = CliRunner().invoke(main, ["measure", "series", "ctx-1", "m", option, value])
    assert result.exit_code == 2
    assert "Invalid value" in result.output


@patch("kagura_memory.cli.load_config")
def test_get_kagura_client_missing_api_key(mock_config, monkeypatch, tmp_path):
    """_get_kagura_client surfaces a credential error when no api_key + no OAuth."""
    mock_config.return_value = {"api_key": "", "mcp_url": "https://test.com/mcp"}
    monkeypatch.delenv("KAGURA_API_KEY", raising=False)
    monkeypatch.delenv("KAGURA_PROFILE", raising=False)
    monkeypatch.setattr(
        "kagura_memory.auth.credentials.DEFAULT_CREDENTIALS_PATH",
        tmp_path / "missing-credentials.json",
    )
    monkeypatch.setattr("kagura_memory._auth.load_config", lambda: {"api_key": ""})

    runner = CliRunner()
    result = runner.invoke(main, ["sleep", "rollback", "ctx-1", "rid-9", "-y"])
    assert result.exit_code != 0
    assert "No credentials found" in result.output or "kagura auth login" in result.output


# ============================================================================
# Edge CRUD CLI tests
# ============================================================================


def _edge_obj():
    """Build an Edge model instance for CLI test mocking."""
    from datetime import datetime

    from kagura_memory import Edge

    return Edge(
        source_id="src-uuid",
        target_id="tgt-uuid",
        edge_type="related_to",
        weight=0.5,
        confidence=1.0,
        created_at=datetime(2026, 4, 29, 0, 0, 0),
        last_updated=datetime(2026, 4, 29, 0, 5, 0),
    )


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_edge_list(mock_client_cls, mock_config):
    """edge list should call list_edges and emit JSON."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    mock_client = AsyncMock()
    mock_client.list_edges.return_value = [_edge_obj()]
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    runner = CliRunner()
    result = runner.invoke(main, ["edge", "list", "ctx-1", "mem-1"])
    assert result.exit_code == 0, result.output
    assert "src-uuid" in result.output
    assert "tgt-uuid" in result.output

    mock_client.list_edges.assert_called_once_with(
        context_id="ctx-1",
        memory_id="mem-1",
        min_weight=0.0,
        edge_types=None,
        limit=None,
    )


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_edge_list_with_filters(mock_client_cls, mock_config):
    """edge list should pass --min-weight, --type, --limit through."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    mock_client = AsyncMock()
    mock_client.list_edges.return_value = []
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "edge",
            "list",
            "ctx-1",
            "mem-1",
            "--min-weight",
            "0.5",
            "--type",
            "related_to,depends_on",
            "--limit",
            "10",
        ],
    )
    assert result.exit_code == 0, result.output
    mock_client.list_edges.assert_called_once_with(
        context_id="ctx-1",
        memory_id="mem-1",
        min_weight=0.5,
        edge_types=["related_to", "depends_on"],
        limit=10,
    )


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_edge_create(mock_client_cls, mock_config):
    """edge create should call create_edge with defaults."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    mock_client = AsyncMock()
    mock_client.create_edge.return_value = _edge_obj()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    runner = CliRunner()
    result = runner.invoke(main, ["edge", "create", "ctx-1", "src-uuid", "tgt-uuid"])
    assert result.exit_code == 0, result.output
    assert "related_to" in result.output

    mock_client.create_edge.assert_called_once_with(
        context_id="ctx-1",
        source_id="src-uuid",
        target_id="tgt-uuid",
        edge_type="related_to",
        weight=0.5,
        confidence=1.0,
    )


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_edge_create_with_options(mock_client_cls, mock_config):
    """edge create should pass --type, --weight, --confidence."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    mock_client = AsyncMock()
    mock_client.create_edge.return_value = _edge_obj()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "edge",
            "create",
            "ctx-1",
            "src-uuid",
            "tgt-uuid",
            "--type",
            "depends_on",
            "--weight",
            "0.8",
            "--confidence",
            "0.9",
        ],
    )
    assert result.exit_code == 0, result.output
    mock_client.create_edge.assert_called_once_with(
        context_id="ctx-1",
        source_id="src-uuid",
        target_id="tgt-uuid",
        edge_type="depends_on",
        weight=0.8,
        confidence=0.9,
    )


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_edge_update(mock_client_cls, mock_config):
    """edge update should call update_edge with the provided fields."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    mock_client = AsyncMock()
    mock_client.update_edge.return_value = _edge_obj()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["edge", "update", "ctx-1", "src-uuid", "tgt-uuid", "--weight", "0.9"],
    )
    assert result.exit_code == 0, result.output

    mock_client.update_edge.assert_called_once_with(
        context_id="ctx-1",
        source_id="src-uuid",
        target_id="tgt-uuid",
        weight=0.9,
        edge_type=None,
    )


def test_edge_update_requires_option():
    """edge update should fail without --weight or --type."""
    runner = CliRunner()
    result = runner.invoke(main, ["edge", "update", "ctx-1", "src-uuid", "tgt-uuid"])
    assert result.exit_code != 0
    assert "At least one of --weight or --type" in result.output


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_edge_delete_with_yes_flag(mock_client_cls, mock_config):
    """edge delete with -y should call delete_edge without prompting."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    mock_client = AsyncMock()
    mock_client.delete_edge.return_value = True
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    runner = CliRunner()
    result = runner.invoke(main, ["edge", "delete", "ctx-1", "src-uuid", "tgt-uuid", "-y"])
    assert result.exit_code == 0, result.output
    assert "true" in result.output.lower()

    mock_client.delete_edge.assert_called_once_with(
        context_id="ctx-1",
        source_id="src-uuid",
        target_id="tgt-uuid",
    )


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_edge_delete_prompts_confirmation(mock_client_cls, mock_config):
    """edge delete without -y should prompt and abort on 'n'."""
    runner = CliRunner()
    result = runner.invoke(main, ["edge", "delete", "ctx-1", "src-uuid", "tgt-uuid"], input="n\n")
    assert result.exit_code != 0


@pytest.mark.parametrize(
    "argv",
    [
        ["files", "upload", "{path}", "--importance", "nan"],
        ["ingest", "{path}", "--importance", "NaN"],
        ["context", "search-config", "ctx", "--semantic", "nan"],
        ["context", "search-config", "ctx", "--bm25", "-nan"],
    ],
    ids=["files-upload", "ingest", "semantic", "bm25"],
)
def test_float_range_refuses_nan(argv, tmp_path):
    """click's FloatRange compares NaN false both ways and let it through (#285)."""
    path = tmp_path / "a.txt"
    path.write_text("hi")
    with patch("kagura_memory.cli.load_config") as config:
        result = CliRunner().invoke(main, [a.format(path=path) for a in argv])
    assert result.exit_code == 2, result.output
    assert "nan is not in the range 0.0<=x<=1.0." in result.output
    config.assert_not_called()


def test_resource_tokens_revoke_and_update_accept_string_ids(monkeypatch):
    """Server v0.89.0+ token ids are opaque ``rtok_…`` strings (memory-cloud#1008)."""
    rtok = "rtok_0123456789abcdefABCDEF"
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.update_token.return_value = MagicMock(model_dump_json=lambda indent=None: "{}")
    monkeypatch.setattr("kagura_memory.cli._get_resource_client", lambda: client)

    result = CliRunner().invoke(main, ["resource", "tokens", "revoke", rtok])
    assert result.exit_code == 0, result.output
    client.revoke_token.assert_awaited_once_with(rtok)

    result = CliRunner().invoke(main, ["resource", "tokens", "update", rtok, "-q", "5"])
    assert result.exit_code == 0, result.output
    assert client.update_token.await_args.kwargs["token_id"] == rtok
