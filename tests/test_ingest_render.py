"""Tests for the Rich human formatter at ingest/_render.py."""

from __future__ import annotations

import io

import pytest
from rich.console import Console

from kagura_memory.ingest._render import render_dry_run, render_failure, render_result
from kagura_memory.models import CostBreakdown, IngestErrorRecord, IngestResult


def _capture(width: int = 100) -> Console:
    """Build a Rich Console that writes to an in-memory StringIO.

    ``record=True`` lets us read the rendered text via ``export_text()``
    without color codes, which is what we want for assertion on content.
    Width fixed at 100 so column alignment is deterministic across CI.
    """
    return Console(file=io.StringIO(), width=width, record=True, force_terminal=False)


def _result(
    *,
    success: bool = True,
    section_count: int = 3,
    skipped_images: int = 0,
    errors: list[IngestErrorRecord] | None = None,
    warnings: list[str] | None = None,
    is_dry_run: bool = False,
    source_uri: str = "/tmp/report.pdf",
) -> IngestResult:
    """Build an IngestResult fixture with sensible defaults."""
    return IngestResult(
        is_dry_run=is_dry_run,
        source_uri=source_uri,
        source_type="file",
        overview_id="overview-uuid" if success else None,
        section_ids=[f"sec-{i}" for i in range(section_count)] if success else [],
        skipped_images=skipped_images,
        cost=CostBreakdown(
            is_estimate=is_dry_run,
            text_provider="claude",
            vision_provider="gemini",
            est_usd=0.03 if is_dry_run else None,
            prompt_tokens=4200 if is_dry_run else None,
            completion_tokens=1600 if is_dry_run else None,
        ),
        errors=errors or [],
        warnings=warnings or [],
    )


def test_success_path_shows_check_glyph_and_counts() -> None:
    """Clean run: green ✓ headline + Overview/Sections/Skipped rows."""
    console = _capture()
    render_result(_result(section_count=3), console)
    out = console.export_text()
    assert "✓" in out
    assert "Ingested" in out
    assert "/tmp/report.pdf" in out
    assert "Overview:" in out
    assert "overview-uuid" in out
    assert "Sections:" in out
    assert "3 created" in out
    # No "failed" suffix when errors empty.
    assert "failed" not in out


def test_partial_success_shows_warning_glyph_and_failed_count() -> None:
    """overview succeeded, 1 section failed → yellow ⚠ + 'N failed' suffix."""
    console = _capture()
    errors = [
        IngestErrorRecord(
            step="summarize", section_index=2, message="rate limit", exception_type="LLMError"
        )
    ]
    render_result(_result(section_count=2, errors=errors), console)
    out = console.export_text()
    assert "⚠" in out
    assert "completed with warnings" in out
    assert "2 created, 1 failed" in out
    assert "section 2" in out
    assert "rate limit" in out


def test_failure_path_shows_cross_glyph_and_step_reason() -> None:
    """overview_id None → red ✗ headline + Step/Reason rows."""
    console = _capture()
    errors = [IngestErrorRecord(step="extract", message="bad pdf", exception_type="IngestError")]
    render_result(_result(success=False, section_count=0, errors=errors), console)
    out = console.export_text()
    assert "✗" in out
    assert "Failed to ingest" in out
    assert "Step:" in out
    assert "extract" in out
    assert "Reason:" in out
    assert "bad pdf" in out


def test_error_list_truncated_above_three() -> None:
    """L3 error list shows 3 inline + '... N more' hint."""
    console = _capture()
    errors = [
        IngestErrorRecord(step="summarize", section_index=i, message=f"err {i}") for i in range(7)
    ]
    render_result(_result(section_count=10, errors=errors), console)
    out = console.export_text()
    # First three sections appear by index.
    assert "section 0" in out
    assert "section 1" in out
    assert "section 2" in out
    # Fourth+ collapsed.
    assert "section 3" not in out
    assert "4 more" in out


def test_dry_run_uses_arrow_glyph_and_token_estimates() -> None:
    """--dry-run path: blue → glyph + token estimates + 'no LLM calls' footer."""
    console = _capture()
    render_dry_run(_result(is_dry_run=True, section_count=8), console)
    out = console.export_text()
    assert "→" in out
    assert "Dry run" in out
    assert "Tokens:" in out
    assert "prompt=4200" in out
    assert "completion≈1600" in out
    assert "~$0.03" in out
    assert "No LLM calls were made" in out


def test_render_failure_from_exception_uses_cross_glyph() -> None:
    """When an exception escapes, render_failure shows ✗ + type info."""
    console = _capture()
    err = ValueError("ssl handshake timed out")
    render_failure(err, console)
    out = console.export_text()
    assert "✗" in out
    assert "ingest failed" in out
    assert "ssl handshake timed out" in out
    assert "ValueError" in out


def test_rich_markup_in_source_uri_is_escaped() -> None:
    """A bracket-laden source URI must not break Rich markup parsing."""
    console = _capture()
    render_result(_result(source_uri="https://example.com/[evil]/file.pdf"), console)
    out = console.export_text()
    # The bracket sequence appears verbatim; absence here would mean
    # Rich tried to parse [evil] as a style tag and ate it.
    assert "[evil]" in out


@pytest.mark.parametrize("skipped", [0, 1, 5])
def test_skipped_images_row_always_present(skipped: int) -> None:
    """Skipped images row is informative even at 0 — keeps layout stable."""
    console = _capture()
    render_result(_result(skipped_images=skipped), console)
    out = console.export_text()
    assert "Skipped images:" in out
    assert str(skipped) in out
