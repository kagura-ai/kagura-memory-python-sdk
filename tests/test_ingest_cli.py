"""Tests for the `kagura ingest` CLI subcommand."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from click.testing import CliRunner

from kagura_memory.cli import main
from kagura_memory.models import CostBreakdown, IngestResult

FIXTURE = Path(__file__).parent / "fixtures" / "sample.pdf"


def _mock_client_class() -> MagicMock:
    """Build a MagicMock that replaces KaguraClient in cli.py."""
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls = MagicMock(return_value=mock_client)
    return mock_client_cls


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_ingest_dry_run_emits_estimate_json(
    mock_client_cls: MagicMock, mock_config: MagicMock
) -> None:
    """`kagura ingest --dry-run` writes a CostBreakdown JSON to stdout."""
    mock_config.return_value = {
        "api_key": "test-key",
        "mcp_url": "https://test.com/mcp",
        "context_id": "ctx-uuid",
    }
    mock_client_cls.return_value = MagicMock(spec=type(MagicMock()))

    expected_result = IngestResult(
        is_dry_run=True,
        source_uri=f"file://{FIXTURE.resolve()}",
        source_type="file",
        cost=CostBreakdown(
            is_estimate=True, prompt_tokens=120, completion_tokens=400, text_provider="claude"
        ),
    )

    with patch("kagura_memory.ingest.FileIngestor") as mock_ingestor_cls:
        mock_ingestor = MagicMock()
        mock_ingestor.estimate_cost = AsyncMock(return_value=expected_result)
        mock_ingestor_cls.return_value = mock_ingestor

        runner = CliRunner()
        result = runner.invoke(main, ["ingest", str(FIXTURE), "--dry-run"])

        assert result.exit_code == 0, result.output
        assert '"is_dry_run": true' in result.output
        assert '"is_estimate": true' in result.output
        assert '"prompt_tokens": 120' in result.output
        # Critical: estimate path must NOT call the actual ingest method.
        mock_ingestor.estimate_cost.assert_called_once()


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_ingest_full_path_calls_ingest(mock_client_cls: MagicMock, mock_config: MagicMock) -> None:
    """`kagura ingest <path>` (no --dry-run) calls FileIngestor.ingest."""
    mock_config.return_value = {
        "api_key": "test-key",
        "mcp_url": "https://test.com/mcp",
        "context_id": "ctx-uuid",
    }
    mock_client_cls.return_value = MagicMock(spec=type(MagicMock()))

    expected_result = IngestResult(
        is_dry_run=False,
        source_uri=f"file://{FIXTURE.resolve()}",
        source_type="file",
        overview_id="ov-1",
        section_ids=["sec-1", "sec-2"],
        cost=CostBreakdown(text_provider="claude"),
    )

    with patch("kagura_memory.ingest.FileIngestor") as mock_ingestor_cls:
        mock_ingestor = MagicMock()
        mock_ingestor.ingest = AsyncMock(return_value=expected_result)
        mock_ingestor_cls.return_value = mock_ingestor

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["ingest", str(FIXTURE), "--tags", "pdf,doc", "--importance", "0.8"],
        )

        assert result.exit_code == 0, result.output
        assert '"overview_id": "ov-1"' in result.output
        assert '"section_ids":' in result.output

        # Verify CLI-to-ingestor argument plumbing.
        kwargs = mock_ingestor.ingest.call_args.kwargs
        assert kwargs["context_id"] == "ctx-uuid"
        assert kwargs["tags"] == ["pdf", "doc"]
        assert kwargs["importance"] == 0.8


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_ingest_rejects_invalid_provider_choice(
    mock_client_cls: MagicMock, mock_config: MagicMock
) -> None:
    """Click's Choice gate rejects unknown --vision-provider values."""
    mock_config.return_value = {"api_key": "test", "mcp_url": "https://test.com/mcp"}

    runner = CliRunner()
    result = runner.invoke(main, ["ingest", "x.pdf", "--vision-provider", "bogus"])
    assert result.exit_code != 0
    assert "bogus" in result.output


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_ingest_no_api_key_fails_cleanly(
    mock_client_cls: MagicMock, mock_config: MagicMock
) -> None:
    """Missing api_key surfaces as a click ClickException, not a stack trace."""
    mock_config.return_value = {}

    runner = CliRunner()
    result = runner.invoke(main, ["ingest", "x.pdf"])
    assert result.exit_code != 0
    assert "api_key" in result.output.lower() or "api key" in result.output.lower()


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_ingest_no_context_id_fails_when_not_dry_run(
    mock_client_cls: MagicMock, mock_config: MagicMock
) -> None:
    """Without --dry-run, context_id is required."""
    mock_config.return_value = {"api_key": "test", "mcp_url": "https://test.com/mcp"}

    runner = CliRunner()
    result = runner.invoke(main, ["ingest", "x.pdf"])
    assert result.exit_code != 0
    assert "context_id" in result.output.lower() or "context id" in result.output.lower()
