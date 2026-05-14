"""Render the `kagura ingest` demo screenshot to an SVG asset.

Produces ``assets/ingest-demo.svg`` for embedding in README.md and
release notes. Run after touching the Rich formatter or ingest prompts
so the visual is up to date.

Usage:
    python tools/generate_demo_svg.py

The output is checked into git. Markdown renderers on GitHub and PyPI
both accept SVG via ``<img src="...">`` without further hosting.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from kagura_memory.ingest._render import render_result
from kagura_memory.models import CostBreakdown, IngestResult

OUT = Path(__file__).resolve().parent.parent / "assets" / "ingest-demo.svg"

# A representative successful ingest. Numbers chosen to fit the README's
# "60-second demo" narrative: medium-sized PDF, 8 sections, all clean.
_DEMO = IngestResult(
    is_dry_run=False,
    source_uri="./report.pdf",
    source_type="file",
    overview_id="a1b2c3d4-e5f6-7890-abcd-ef0123456789",
    section_ids=[f"sec-{i:02d}" for i in range(8)],
    skipped_images=0,
    archived_file_id="f8e9d0c1-b2a3-4567-89ab-cdef01234567",
    cost=CostBreakdown(
        is_estimate=False,
        # Provider NAME (matches what the orchestrator writes via
        # ``Provider.name``); the demo previously used a model path like
        # "claude/sonnet-4-6" which is not what the runtime emits.
        text_provider="claude",
        vision_provider="gemini",
        est_usd=0.03,
    ),
)


def main() -> None:
    """Render `_DEMO` to OUT via Console(record=True).save_svg().

    Width fixed at 80 so the output mirrors a typical terminal. The SVG
    title is the command that produced the output — clicking through
    from README will land on something self-explanatory.
    """
    console = Console(record=True, width=80, force_terminal=True)
    render_result(_DEMO, console)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    console.save_svg(str(OUT), title="kagura ingest ./report.pdf")
    print(f"wrote {OUT.relative_to(OUT.parent.parent)} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
