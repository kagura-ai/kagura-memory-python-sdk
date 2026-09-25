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

from kagura_memory.auth.credentials import reset_state_cache
from kagura_memory.cli import _resolve_progress_logger, main
from kagura_memory.logger import _NULL_LOGGER, VerboseLogger


@pytest.fixture(autouse=True)
def _isolate_credential_state(tmp_path, monkeypatch):
    """Isolate every test from real ``~/.kagura/credentials.json`` and env vars.

    The ``test_resource_import_*`` tests now route through
    ``_get_resource_client`` → ``_resolve_auth``, which consults
    OAuth credentials before the mocked config fallback. Without this
    isolation, a developer's stored OAuth profile or ``KAGURA_PROFILE``
    would pre-empt the test fixtures.
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
    """Parse newline-delimited JSON, skipping non-JSON lines.

    CLI tests that capture stderr may see ``click.ClickException`` output
    (``Error: ...``) interspersed with our NDJSON events. Skip anything
    that doesn't look like a JSON object so the test focuses on the
    progress stream's structural correctness.
    """
    parsed: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
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


def test_ndjson_debug_serializes_non_primitive_via_str(capsys):
    """debug() coerces non-primitive payloads through str() into the NDJSON line.

    The well-behaved branch: a custom class with a normal ``__str__`` /
    ``__repr__`` is rendered into ``detail.data`` rather than dropped.
    The broken-``__str__`` branch is exercised by
    ``test_ndjson_debug_swallows_broken_str_payload`` below.
    """

    class _CustomPayload:
        def __repr__(self) -> str:
            return "_CustomPayload(<x>)"

    logger = VerboseLogger(output_format="json")
    logger.debug("LLM response", _CustomPayload(), stage="summarize")
    events = _parse_lines(capsys.readouterr().err)
    assert events[-1]["kind"] == "debug"
    assert "_CustomPayload" in events[-1]["detail"]["data"]


def test_ndjson_emits_placeholder_when_json_dumps_fails(capsys):
    """``json.dumps`` failures (broken __str__, circular refs) trigger the fallback.

    ``default=str`` only handles non-serializable types whose ``str()`` is
    well-behaved. A class whose ``__str__`` raises makes ``default=str``
    itself raise during encoding — the outer ``try/except`` should catch
    it and emit a minimal placeholder event so the terminal-event
    invariant still holds.
    """

    class _BadObject:
        def __str__(self) -> str:
            raise RuntimeError("nope")

    logger = VerboseLogger(output_format="json")
    logger.detail("key", _BadObject(), stage="x")
    events = _parse_lines(capsys.readouterr().err)
    assert len(events) == 1
    assert events[0]["msg"] == "<event serialization failed>"
    assert events[0]["v"] == 1
    assert events[0]["kind"] == "detail"


def test_ndjson_debug_swallows_broken_str_payload(capsys):
    """A payload whose ``__str__`` raises must not crash the operation.

    Locks the "progress logging must never raise" contract for the
    debug-JSON path: when ``str(data)`` itself fails, debug() emits a
    safe placeholder naming the type instead of propagating.
    """

    class _BadStr:
        def __str__(self) -> str:
            raise RuntimeError("broken __str__")

    logger = VerboseLogger(output_format="json")
    # The call must not raise even though str(_BadStr()) does.
    logger.debug("LLM response", _BadStr(), stage="summarize")
    events = _parse_lines(capsys.readouterr().err)
    assert events[-1]["kind"] == "debug"
    assert "_BadStr" in events[-1]["detail"]["data"]
    assert "raised" in events[-1]["detail"]["data"]


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
async def test_files_upload_emits_terminal_error_event_on_validator_failure(capsys):
    """Pre-flight validators (UUID check, source prep) still get terminal kind=error.

    The fix for Copilot's review pulled ``_validate_context_id`` /
    ``_prepare_source`` / ``_resolve_content_type`` inside the
    ``try/except BaseException`` so a validator failure also emits a
    terminal event before the exception propagates — matching the
    "operation logging never leaves the AI consumer hanging" contract.
    """
    from kagura_memory import FilesClient

    client = FilesClient(api_key="test", base_url="https://example.com")
    logger = VerboseLogger(output_format="json")
    with pytest.raises(ValueError, match="context_id must be a UUID"):
        await client.upload(
            context_id="not-a-uuid",
            source=b"hello",
            filename="x.txt",
            logger=logger,
        )

    events = _parse_lines(capsys.readouterr().err)
    assert events, "validator failure must still emit a terminal event"
    assert events[-1]["kind"] == "error"
    await client.close()


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.ResourceClient")
def test_resource_import_emits_single_terminal_success(mock_rc_cls, mock_config):
    """`kagura resource import --progress=json` emits exactly ONE terminal event.

    Even when the CLI loops over multiple batches internally, the user-facing
    operation is one ``kagura resource import`` invocation and must emit one
    terminal event total (per the AI-consumer "wait for first success/error"
    contract). Verifies the round-3 fix that suppressed per-batch terminal
    events from the SDK and added a single CLI-level closing event.
    """
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    mock_rc = AsyncMock()
    # 250 events at batch size 100 → 3 batches, 3 ingest_events calls.
    mock_rc.ingest_events.return_value = MagicMock(created_count=100, failed_count=0, errors=[])
    mock_rc.__aenter__ = AsyncMock(return_value=mock_rc)
    mock_rc.__aexit__ = AsyncMock(return_value=None)
    mock_rc_cls.from_mcp_url.return_value = mock_rc
    mock_rc_cls._from_resolved_auth.return_value = mock_rc

    jsonl = "\n".join(f'{{"name": "row{i}"}}' for i in range(250))
    # Click 8.2+ separates stdout / stderr on CliRunner.invoke by default
    # so we can parse the NDJSON progress stream independently of the
    # stdout result JSON via result.stderr.
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "resource",
            "import",
            "-r",
            "products",
            "-k",
            "TOKEN",
            "--format",
            "jsonl",
            "--progress",
            "json",
        ],
        input=jsonl,
    )
    assert result.exit_code == 0, result.output

    events = _parse_lines(result.stderr)
    # 1 import_start + 3 import_batch + 1 complete (success) = 5 events.
    terminal = [e for e in events if e["kind"] in ("success", "error")]
    assert len(terminal) == 1, f"expected exactly 1 terminal event, got {len(terminal)}: {terminal}"
    assert terminal[0]["kind"] == "success"
    assert terminal[0]["stage"] == "complete"
    assert terminal[0]["detail"]["total"] == 250


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.ResourceClient")
def test_resource_import_emits_terminal_error_on_batch_failure(mock_rc_cls, mock_config):
    """A mid-loop SDK failure emits exactly ONE terminal kind=error with partial state.

    Locks the round-3 CLI-level ``try/except BaseException`` wrapper around
    the batch loop so that even when ``ingest_events`` blows up, the user
    sees a final terminal event reporting how many events were already
    created/failed before the crash.
    """
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    mock_rc = AsyncMock()
    # First batch succeeds (100 created), second batch raises mid-loop.
    mock_rc.ingest_events.side_effect = [
        MagicMock(created_count=100, failed_count=0, errors=[]),
        RuntimeError("simulated server crash"),
    ]
    mock_rc.__aenter__ = AsyncMock(return_value=mock_rc)
    mock_rc.__aexit__ = AsyncMock(return_value=None)
    mock_rc_cls.from_mcp_url.return_value = mock_rc
    mock_rc_cls._from_resolved_auth.return_value = mock_rc

    jsonl = "\n".join(f'{{"name": "row{i}"}}' for i in range(200))
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "resource",
            "import",
            "-r",
            "products",
            "-k",
            "TOKEN",
            "--format",
            "jsonl",
            "--progress",
            "json",
        ],
        input=jsonl,
    )
    assert result.exit_code != 0

    events = _parse_lines(result.stderr)
    terminal = [e for e in events if e["kind"] in ("success", "error")]
    assert len(terminal) == 1
    assert terminal[0]["kind"] == "error"
    assert terminal[0]["detail"]["created_so_far"] == 100
    assert terminal[0]["detail"]["total_events"] == 200


def test_rich_path_swallows_broken_console_print():
    """A Rich-path ``Console.print`` raising OSError must not crash the caller.

    Locks the ``_safe_rich_print`` wrapper that catches BrokenPipeError /
    OSError so ``kagura ingest -v | head -n 1`` does not crash the
    underlying SDK operation when the consumer closes the pipe early.
    """

    class _BrokenConsole:
        def print(self, *_args: object, **_kwargs: object) -> None:
            raise BrokenPipeError("downstream consumer gone")

    logger = VerboseLogger(level=2, console=_BrokenConsole(), output_format="rich")  # type: ignore[arg-type]
    # Each method exercises the wrapper on a different render path.
    logger.action("a")
    logger.detail("k", "v")
    logger.success("ok")
    logger.warning("warn")
    logger.error("boom")
    logger.debug("dbg", {"x": 1})


def test_rich_detail_swallows_broken_str_value():
    """``detail()`` Rich path defensively stringifies ``value`` before f-string.

    Without the pre-stringification, ``f"...{value}..."`` would call
    ``value.__format__`` / ``__str__`` BEFORE ``_safe_rich_print``'s
    OSError guard could ever run — so a broken ``__str__`` crashes the
    caller. Locks the never-raise contract for ``detail()`` on the Rich
    path, matching ``debug()``'s behavior.
    """

    class _BadStr:
        def __str__(self) -> str:
            raise RuntimeError("broken __str__")

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, no_color=True)
    logger = VerboseLogger(level=2, console=console, output_format="rich")
    logger.detail("key", _BadStr())  # must not raise
    assert "_BadStr" in buf.getvalue()
    assert "raised" in buf.getvalue()


def test_rich_debug_swallows_broken_str_payload():
    """The Rich debug path now falls back safely when ``str(data)`` raises.

    Mirrors the JSON path's ``test_ndjson_debug_swallows_broken_str_payload``
    so the "never raise" contract holds on both render targets — Rich
    used to call ``str()`` eagerly without a guard.
    """

    class _BadStr:
        def __str__(self) -> str:
            raise RuntimeError("broken __str__")

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, no_color=True)
    logger = VerboseLogger(level=3, console=console, output_format="rich")
    logger.debug("LLM response", _BadStr())  # must not raise
    assert "serialization failed" in buf.getvalue()


def test_logger_swallows_broken_stderr_pipe(capsys, monkeypatch):
    """A BrokenPipeError on stderr must not crash the operation being logged.

    Simulates the consumer-pipe-closed-early case (``kagura ingest |
    head -n 1``) where stderr writes start raising mid-stream. The
    ``_emit_json`` path catches OSError and drops the line silently —
    progress logging must never raise per the module docstring.
    """
    import sys

    class _BrokenStream:
        def write(self, data: str) -> int:
            raise BrokenPipeError("downstream consumer gone")

        def flush(self) -> None:
            raise BrokenPipeError("downstream consumer gone")

    logger = VerboseLogger(output_format="json")
    monkeypatch.setattr(sys, "stderr", _BrokenStream())
    # The call must not raise even though every write raises.
    logger.action("orphan", stage="x")


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


# ---------------------------------------------------------------------------
# Bugs found while porting to TypeScript (#285)
# ---------------------------------------------------------------------------


def _import(*extra: str, input: str, fmt: str = "jsonl"):
    return CliRunner().invoke(
        main,
        ["resource", "import", "-r", "products", "-k", "TOKEN", "--format", fmt, *extra],
        input=input,
    )


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.ResourceClient")
def test_resource_import_credential_failure_opens_no_stream(mock_rc_cls, mock_config):
    """``import_start`` came before the client was built, so a credential
    failure left the stream with no terminal event."""
    mock_config.return_value = {}  # no api_key, no profile: the chain fails
    result = _import("--progress", "json", input='{"name": "a"}')
    assert result.exit_code == 1, result.output
    assert _parse_lines(result.stderr) == []
    mock_rc_cls._from_resolved_auth.assert_not_called()


def _wired_resource_client(mock_rc_cls: MagicMock, **ingest: object) -> AsyncMock:
    mock_rc = AsyncMock()
    for key, value in ingest.items():
        setattr(mock_rc.ingest_events, key, value)
    mock_rc.__aenter__ = AsyncMock(return_value=mock_rc)
    mock_rc.__aexit__ = AsyncMock(return_value=None)
    mock_rc_cls._from_resolved_auth.return_value = mock_rc
    return mock_rc


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.ResourceClient")
def test_resource_import_names_an_unmessaged_failure(mock_rc_cls, mock_config):
    """``Import failed: {e}`` printed nothing after the colon for such an error."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}
    _wired_resource_client(mock_rc_cls, side_effect=RuntimeError())
    result = _import("--progress", "json", input='{"name": "a"}')
    assert result.exit_code == 1
    [terminal] = [e for e in _parse_lines(result.stderr) if e["kind"] in ("success", "error")]
    assert terminal["msg"] == "Import failed: RuntimeError"


@pytest.mark.parametrize(
    ("fmt", "rows", "extra", "message"),
    [
        ("csv", "sku,name\na,b,c\n", [], "Row 1: more fields than the header has columns."),
        (
            "csv",
            "sku,name\n,b\n",
            ["--id-column", "sku"],
            "Row 1: doc_id from column 'sku' must be 1-255 characters, got 0.",
        ),
        (
            "jsonl",
            json.dumps({"sku": "x" * 256}),
            ["--id-column", "sku"],
            "Row 1: doc_id from column 'sku' must be 1-255 characters, got 256.",
        ),
    ],
    ids=["extra-cells", "empty-id", "long-id"],
)
@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.ResourceClient")
def test_resource_import_bad_row_is_a_clean_error(
    mock_rc_cls, mock_config, fmt, rows, extra, message
):
    """These rows ended in a pydantic ValidationError traceback."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}
    result = _import(*extra, input=rows, fmt=fmt)
    assert result.exit_code == 1
    assert f"Error: {message}" in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)
    mock_rc_cls._from_resolved_auth.assert_not_called()


_FILE_ID = "10000000-0000-0000-0000-000000000002"


def _uploading(file_obj):
    """A ``FilesClient.upload`` stand-in that reports progress as the real one does."""

    async def upload(*, logger=None, **_):
        logger.action("Reserving upload", stage="reserve")
        logger.success("Upload complete", stage="complete", detail={"file_id": file_obj.id})
        return file_obj

    return upload


def _upload_remember(tmp_path, remember_result):
    from datetime import UTC, datetime

    from kagura_memory.models import FileObject

    file_obj = FileObject(
        id=_FILE_ID,
        workspace_id="00000000-0000-0000-0000-000000000001",
        filename="hello.txt",
        content_type="text/plain",
        size_bytes=2,
        sha256="a" * 64,
        status="uploaded",
        created_at=datetime(2026, 5, 11, tzinfo=UTC),
    )
    path = tmp_path / "hello.txt"
    path.write_text("hi")
    config = {
        "api_key": "key",
        "mcp_url": "https://test.com/mcp",
        "context_id": "00000000-0000-0000-0000-000000000001",
    }
    with (
        patch("kagura_memory.cli.load_config", return_value=config),
        patch("kagura_memory.cli.FilesClient") as files_cls,
        patch("kagura_memory.cli.KaguraClient") as kagura_cls,
    ):
        files = AsyncMock()
        files.upload.side_effect = _uploading(file_obj)
        files.__aenter__ = AsyncMock(return_value=files)
        files.__aexit__ = AsyncMock(return_value=None)
        files_cls._from_resolved_auth.return_value = files
        kagura = AsyncMock()
        if isinstance(remember_result, BaseException):
            kagura.remember.side_effect = remember_result
        else:
            kagura.remember.return_value = remember_result
        kagura.__aenter__ = AsyncMock(return_value=kagura)
        kagura.__aexit__ = AsyncMock(return_value=None)
        kagura_cls.return_value = kagura
        return CliRunner().invoke(
            main, ["files", "upload", str(path), "--remember", "--progress", "json"]
        )


def test_files_upload_remember_failure_ends_the_stream_in_error(tmp_path):
    """The upload's success went out first, so a failed command's stream ended in success."""
    result = _upload_remember(tmp_path, RuntimeError("boom"))
    assert result.exit_code == 1
    events = _parse_lines(result.stderr)
    assert [e["kind"] for e in events] == ["action", "error"]
    assert "creating the linked memory failed: boom" in events[-1]["msg"]
    assert events[-1]["detail"]["reserved_file_id"] == _FILE_ID


def test_files_upload_remember_success_is_the_last_event(tmp_path):
    result = _upload_remember(tmp_path, {"memory_id": "mem-1"})
    assert result.exit_code == 0, result.output
    events = _parse_lines(result.stderr)
    assert [e["kind"] for e in events] == ["action", "success"]
    assert events[-1]["detail"] == {"file_id": _FILE_ID}


@pytest.mark.asyncio
async def test_sdk_terminal_errors_name_an_unmessaged_exception(capsys):
    """``Upload failed: {e}`` / ``Batch ingest failed: {e}`` ended at the colon."""
    from kagura_memory import FilesClient
    from kagura_memory.models import ResourceEventRequest
    from kagura_memory.resource_client import ResourceClient

    logger = VerboseLogger(output_format="json")
    async with FilesClient(api_key="test", base_url="https://example.com") as files:
        with patch.object(files._client, "request", new_callable=AsyncMock) as request:
            request.side_effect = RuntimeError()
            with pytest.raises(RuntimeError):
                await files.upload(
                    context_id="00000000-0000-0000-0000-000000000001",
                    source=b"hi",
                    filename="x.txt",
                    logger=logger,
                )
    async with ResourceClient(api_key="test", base_url="https://example.com") as resources:
        with patch.object(resources._client, "request", new_callable=AsyncMock) as request:
            request.side_effect = RuntimeError()
            with pytest.raises(RuntimeError):
                await resources.ingest_events(
                    "products",
                    "TOKEN",
                    [ResourceEventRequest(op="upsert", doc_id="1")],
                    logger=logger,
                )
    errors = [e["msg"] for e in _parse_lines(capsys.readouterr().err) if e["kind"] == "error"]
    assert errors == ["Upload failed: RuntimeError", "Batch ingest failed: RuntimeError"]


def test_rich_path_prints_caller_text_as_written():
    """File names and messages were read as markup (and emoji codes)."""
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, no_color=True, width=200)
    logger = VerboseLogger(level=2, console=console)
    logger.action("Uploading report[bold].pdf", ":thumbs_up: [/x]")
    logger.detail("k[red]", "v[/y]")
    logger.success("Done :thumbs_up:")
    logger.warning("w [/x]")
    # A `[/x]` raised MarkupError here, inside the caller's except handler.
    logger.error("Upload failed: bad [/x] path C:\\dir\\")
    assert buf.getvalue().splitlines() == [
        "→ Uploading report[bold].pdf :thumbs_up: [/x]",
        "  • k[red]: v[/y]",
        "✓ Done :thumbs_up:",
        "⚠ w [/x]",
        "✗ Upload failed: bad [/x] path C:\\dir\\",
    ]
