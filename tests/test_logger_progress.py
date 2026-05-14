"""Tests for the #108 additions to VerboseLogger and the CLI flag plumbing.

Covers:
- ``output_format`` switch (rich / json / none) and the ``_NULL_LOGGER`` no-op.
- NDJSON schema invariants (``v``, ``ts``, ``stage``, ``kind``).
- Terminal-event contract — entry points emit exactly one ``kind=success`` or
  ``kind=error`` final event even when an unhandled exception propagates.
- CLI ``-v`` / ``--progress`` precedence table (the 8 cells from issue #108).
- ``--help`` text no longer says ``(Phase 2)`` for ``--verbose`` / ``--deep``.
"""

from __future__ import annotations

import io
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner
from rich.console import Console

from kagura_memory.cli import _resolve_progress_logger, main
from kagura_memory.logger import _NULL_LOGGER, VerboseLogger

# ---------------------------------------------------------------------------
# _NULL_LOGGER + output_format=none
# ---------------------------------------------------------------------------


def test_null_logger_is_silent(capsys):
    """Every method on _NULL_LOGGER is a no-op (no stdout/stderr writes)."""
    _NULL_LOGGER.action("act", "details", stage="x")
    _NULL_LOGGER.detail("k", "v", stage="x")
    _NULL_LOGGER.success("ok", stage="x")
    _NULL_LOGGER.warning("warn", stage="x")
    _NULL_LOGGER.error("boom", stage="x")
    _NULL_LOGGER.debug("dbg", {"a": 1}, stage="x")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_output_format_none_explicit_is_also_silent(capsys):
    """A regular VerboseLogger with output_format='none' is also silent."""
    logger = VerboseLogger(level=3, output_format="none")
    logger.action("a", "b")
    logger.error("e")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


# ---------------------------------------------------------------------------
# NDJSON schema invariants
# ---------------------------------------------------------------------------


def _parse_lines(text: str) -> list[dict]:
    """Parse newline-delimited JSON; assert each line is valid JSON."""
    parsed: list[dict] = []
    for line in text.strip().splitlines():
        if not line:
            continue
        parsed.append(json.loads(line))
    return parsed


def test_ndjson_action_emits_required_fields(capsys):
    logger = VerboseLogger(output_format="json")
    logger.action("Fetching", "url=https://example.com", stage="fetch")
    events = _parse_lines(capsys.readouterr().err)
    assert len(events) == 1
    e = events[0]
    assert e["v"] == 1
    assert "ts" in e and e["ts"].endswith("Z")
    assert e["stage"] == "fetch"
    assert e["kind"] == "action"
    assert e["msg"] == "Fetching"
    assert e["detail"] == {"desc": "url=https://example.com"}


def test_ndjson_kind_is_closed_enum(capsys):
    logger = VerboseLogger(output_format="json")
    logger.action("a", stage="s")
    logger.detail("k", 1, stage="s")
    logger.warning("w", stage="s")
    logger.success("ok", stage="s")
    logger.error("err", stage="s")
    logger.debug("d", {"x": 1}, stage="s")
    events = _parse_lines(capsys.readouterr().err)
    kinds = {e["kind"] for e in events}
    assert kinds == {"action", "detail", "warning", "success", "error", "debug"}


def test_ndjson_unknown_stage_when_omitted(capsys):
    """stage=None at the call site → ``"unknown"`` on the wire, per spec."""
    logger = VerboseLogger(output_format="json")
    logger.action("orphan event")
    events = _parse_lines(capsys.readouterr().err)
    assert events[0]["stage"] == "unknown"


def test_ndjson_debug_serializes_non_primitive_safely(capsys):
    """debug() with a non-primitive payload falls back to str() without raising.

    Covers the defensive ``except Exception`` branch around the isinstance
    fallback — a payload whose ``__str__`` is well-behaved (the common case
    when an SDK call passes an arbitrary object) still produces a valid
    NDJSON line.
    """

    class _CustomPayload:
        def __repr__(self) -> str:
            return "_CustomPayload(<x>)"

    logger = VerboseLogger(output_format="json")
    logger.debug("LLM response", _CustomPayload(), stage="summarize")
    events = _parse_lines(capsys.readouterr().err)
    assert events[-1]["kind"] == "debug"
    # The non-primitive is stringified, not dropped.
    assert "_CustomPayload" in events[-1]["detail"]["data"]


def test_ndjson_level_is_ignored(capsys):
    """JSON path emits every event regardless of `level` — consumers filter."""
    logger = VerboseLogger(level=0, output_format="json")
    logger.detail("k", 1, stage="s")  # would be silenced at level<2 on Rich path
    logger.debug("d", {"x": 1}, stage="s")  # likewise level<3
    events = _parse_lines(capsys.readouterr().err)
    assert {e["kind"] for e in events} == {"detail", "debug"}


# ---------------------------------------------------------------------------
# Rich-path level filtering
# ---------------------------------------------------------------------------


def _rich_logger(level: int) -> tuple[VerboseLogger, io.StringIO]:
    """Build a Rich-path logger whose Console writes into a StringIO buffer."""
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, no_color=True)
    return VerboseLogger(level=level, console=console, output_format="rich"), buf


def test_rich_level_1_emits_action_silences_detail():
    logger, buf = _rich_logger(level=1)
    logger.action("Doing thing")
    logger.detail("key", "value")
    out = buf.getvalue()
    assert "Doing thing" in out
    assert "key" not in out  # detail filtered at level 1


def test_rich_level_2_emits_detail():
    logger, buf = _rich_logger(level=2)
    logger.detail("key", "value")
    assert "key" in buf.getvalue()


def test_rich_error_renders_at_level_0():
    """Errors render regardless of level — too important to gate behind -v."""
    logger, buf = _rich_logger(level=0)
    logger.error("boom")
    assert "boom" in buf.getvalue()


# ---------------------------------------------------------------------------
# CLI flag precedence (the 8-cell table from issue #108)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "verbose, progress, expect_logger, expect_format, expect_level",
    [
        # (verbose count, --progress, returns a logger?, output_format, level)
        pytest.param(0, None, False, None, None, id="no-flags-silent"),
        pytest.param(1, None, True, "rich", 1, id="-v-default-rich"),
        pytest.param(2, None, True, "rich", 2, id="-vv-rich-level-2"),
        pytest.param(0, "rich", True, "rich", 1, id="progress-rich-default-level-1"),
        pytest.param(2, "rich", True, "rich", 2, id="-vv-progress-rich-level-2"),
        pytest.param(0, "json", True, "json", 1, id="progress-json-no-verbose"),
        pytest.param(2, "json", True, "json", 2, id="-vv-progress-json-level-ignored"),
        pytest.param(1, "none", False, None, None, id="-v-progress-none-silent-wins"),
    ],
)
def test_cli_progress_precedence_8_cells(
    verbose, progress, expect_logger, expect_format, expect_level
):
    result = _resolve_progress_logger(verbose, progress)
    if not expect_logger:
        assert result is None
        return
    assert result is not None
    assert result.output_format == expect_format
    assert result.level == expect_level


# ---------------------------------------------------------------------------
# --help no longer says (Phase 2)
# ---------------------------------------------------------------------------


def test_process_help_drops_phase2_for_verbose_and_deep():
    """Both --verbose and --deep have shipped — help text was stale."""
    result = CliRunner().invoke(main, ["process", "--help"])
    assert result.exit_code == 0
    assert "(Phase 2)" not in result.output, (
        f"Stale '(Phase 2)' marker still in `kagura process --help`:\n{result.output}"
    )


@pytest.mark.parametrize(
    "command",
    [["ingest", "--help"], ["resource", "import", "--help"], ["files", "upload", "--help"]],
)
def test_progress_flag_documented_in_help(command):
    """--progress is offered with the rich/json/none choices and a hint."""
    result = CliRunner().invoke(main, command)
    assert result.exit_code == 0, result.output
    assert "--progress" in result.output
    assert "json" in result.output


# ---------------------------------------------------------------------------
# Terminal-event contract on unhandled exception (FilesClient.upload smoke)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_files_upload_emits_terminal_error_event_on_exception(capsys):
    """An unexpected exception inside upload() still emits ``kind=error`` last.

    Drives :meth:`FilesClient.upload` through ``--progress=json`` (via the
    in-process logger) with a mocked transport that explodes during reserve.
    The terminal-event contract requires the stderr stream to end with a
    ``kind=error`` event before the exception propagates to the caller.
    """
    from kagura_memory import FilesClient

    client = FilesClient(api_key="test", base_url="https://example.com")
    logger = VerboseLogger(output_format="json")

    with patch.object(client._client, "request", new_callable=AsyncMock) as mock_req:
        mock_req.side_effect = RuntimeError("simulated transport failure")
        with pytest.raises(RuntimeError, match="simulated transport"):
            await client.upload(
                context_id="00000000-0000-0000-0000-000000000001",
                source=b"hello",
                filename="x.txt",
                logger=logger,
            )

    events = _parse_lines(capsys.readouterr().err)
    assert events, "expected at least one NDJSON event before propagation"
    terminal = events[-1]
    assert terminal["kind"] == "error", (
        f"terminal event must be kind=error; got {terminal['kind']}: {terminal}"
    )
    # Partial-state contract — recovering AI consumer can act on this.
    assert "detail" in terminal and "uploaded" in terminal["detail"]
    await client.close()


# ---------------------------------------------------------------------------
# Library default is silent (no logger= → no progress emission)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingestor_emits_terminal_error_event_on_unhandled_exception(capsys):
    """FileIngestor.ingest's try/except wraps the post-fetch body for terminal error.

    If ``_ingest_fetched`` raises (e.g. KaguraLLMError leaks through the
    inner handlers), ``ingest`` still emits a ``kind=error`` final event
    on the JSON path before propagating. Mirrors the FilesClient.upload
    contract verified above.
    """
    from kagura_memory.ingest.ingestor import FileIngestor

    ingestor = MagicMock(spec=FileIngestor)
    # Stub _fetch to succeed so we reach the wrapped _ingest_fetched call.
    fetched = MagicMock()
    fetched.body = b"hello"
    fetched.source_uri = "file:///tmp/x.txt"
    fetched.source_type = "file"

    ingestor._fetch = AsyncMock(return_value=fetched)
    ingestor._ingest_fetched = AsyncMock(side_effect=RuntimeError("simulated llm crash"))
    # Re-use the real method bound to the mock spec.
    ingestor.ingest = FileIngestor.ingest.__get__(ingestor)

    logger = VerboseLogger(output_format="json")
    with pytest.raises(RuntimeError, match="simulated llm crash"):
        await ingestor.ingest(
            "file:///tmp/x.txt",
            context_id="00000000-0000-0000-0000-000000000001",
            logger=logger,
        )

    events = _parse_lines(capsys.readouterr().err)
    assert events, "expected at least one NDJSON event before propagation"
    terminal = events[-1]
    assert terminal["kind"] == "error", f"terminal must be kind=error; got {terminal}"
    assert terminal["stage"] == "complete"


@pytest.mark.asyncio
async def test_library_default_is_silent(capsys):
    """FilesClient.upload() without `logger=` produces zero stderr output."""
    from kagura_memory import FilesClient

    client = FilesClient(api_key="test", base_url="https://example.com")
    reserve_resp = MagicMock()
    reserve_resp.status_code = 201
    reserve_resp.json.return_value = {
        "file_id": "10000000-0000-0000-0000-000000000002",
        "upload_url": "https://r2.example.com/k",
        "expires_at": "2026-05-11T00:05:00Z",
    }
    reserve_resp.raise_for_status = MagicMock()
    reserve_resp.headers = {}
    confirm_resp = MagicMock()
    confirm_resp.status_code = 200
    confirm_resp.json.return_value = {
        "id": "10000000-0000-0000-0000-000000000002",
        "workspace_id": "00000000-0000-0000-0000-000000000001",
        "filename": "x.txt",
        "content_type": "text/plain",
        "size_bytes": 5,
        "sha256": "a" * 64,
        "status": "uploaded",
        "created_at": "2026-05-11T00:00:00Z",
        "uploaded_at": "2026-05-11T00:00:01Z",
    }
    confirm_resp.raise_for_status = MagicMock()
    confirm_resp.headers = {}
    put_resp = MagicMock()
    put_resp.status_code = 200
    put_resp.raise_for_status = MagicMock()
    put_resp.headers = {}

    with (
        patch.object(client._client, "request", new_callable=AsyncMock) as mock_req,
        patch.object(client._upload_client, "put", new_callable=AsyncMock) as mock_put,
    ):
        mock_req.side_effect = [reserve_resp, confirm_resp]
        mock_put.return_value = put_resp
        await client.upload(
            context_id="00000000-0000-0000-0000-000000000001",
            source=b"hello",
            filename="x.txt",
        )

    # No logger= → no stderr emission.
    assert capsys.readouterr().err == ""
    await client.close()
