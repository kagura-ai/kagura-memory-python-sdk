"""Rich-based human formatter for ``kagura ingest`` output.

Lives alongside the orchestrator so the visual language stays close to the
shape of :class:`~kagura_memory.models.IngestResult`. The rendering vocabulary
mirrors :class:`~kagura_memory.logger.VerboseLogger` (``✓`` green for success,
``⚠`` yellow for partial / warnings, ``✗`` red for failure, ``→`` blue for
in-progress / dry-run, ``dim`` for labels, ``cyan`` bullets for details).

The three public entry points are stateless and accept the
:class:`rich.console.Console` explicitly so a CLI handler can hand in either
``Console()`` (stdout, auto-terminal-detection) or a stderr-bound console for
error paths.

Why not Click ``style``/``echo``: Click's color helpers do not provide the
fine-grained markup the existing :class:`VerboseLogger` uses, and the
project already depends on Rich (``rich>=13.0.0`` in ``pyproject.toml``).
Adopting Rich here makes ``kagura ingest`` the first CLI command that uses
the project's established design system end-to-end.
"""

from __future__ import annotations

from rich.console import Console

from ..models import IngestErrorRecord, IngestResult

# Width of the right-padded label column. Derived from the longest
# IngestResult-facing label currently in use ("Skipped images:" = 16 chars).
# Centralized so future label additions either fit or push the constant.
_LABEL_WIDTH = 16

# Maximum number of per-step error rows we surface inline. Beyond this,
# the renderer prints a "... N more" hint and refers callers to --json.
# Three keeps the failure section visually contained on an 80-col terminal
# without truncating a typical 1-2 partial-failure run.
_MAX_INLINE_ERRORS = 3

_JSON_HINT = "  [dim]Run with --json for full structured output.[/dim]"


def render_result(result: IngestResult, console: Console) -> None:
    """Render a completed ingestion (success or partial success).

    Routing:
        * ``result.success and not result.errors`` → green ``✓`` summary.
        * ``result.success and result.errors``     → yellow ``⚠`` summary
          (best-effort design: overview was written, some steps failed).
        * ``not result.success``                   → red ``✗`` failure
          (overview not created; details from ``result.errors``).
    """
    if not result.success:
        _render_failure_from_result(result, console)
        return

    glyph = "[bold yellow]⚠[/bold yellow]" if result.errors else "[bold green]✓[/bold green]"
    headline = f"{glyph} [bold]Ingested[/bold] {_escape(result.source_uri)}" + (
        " [dim]— completed with warnings[/dim]" if result.errors else ""
    )
    console.print(headline)
    console.print()

    section_total = len(result.section_ids)
    failed_sections = sum(
        1 for e in result.errors if e.step == "summarize" and e.section_index is not None
    )
    sections_label = f"{section_total} created"
    if failed_sections:
        sections_label += f", {failed_sections} failed"

    _print_kv(console, "Overview:", _escape(result.overview_id or "—"))
    _print_kv(console, "Sections:", sections_label)
    _print_kv(console, "Skipped images:", str(result.skipped_images))
    _print_kv(console, "Cost:", _format_cost(result))

    _print_archive_line(console, result)
    _print_warnings(console, result.warnings)
    _print_error_list(console, result.errors)

    console.print()
    console.print(_JSON_HINT)


def render_dry_run(result: IngestResult, console: Console) -> None:
    """Render a ``--dry-run`` IngestResult (no LLM calls were made)."""
    console.print(f"[bold blue]→[/bold blue] [bold]Dry run[/bold] for {_escape(result.source_uri)}")
    console.print()

    cost = result.cost
    _print_kv(console, "Sections:", str(len(result.section_ids)) if result.section_ids else "—")
    if cost.prompt_tokens is not None or cost.completion_tokens is not None:
        prompt = cost.prompt_tokens if cost.prompt_tokens is not None else "?"
        completion = cost.completion_tokens if cost.completion_tokens is not None else "?"
        _print_kv(console, "Tokens:", f"prompt={prompt}  completion≈{completion}")
    _print_kv(console, "Cost (est):", _format_cost(result))

    _print_warnings(console, result.warnings)
    _print_error_list(console, result.errors)

    console.print()
    console.print("  [dim]No LLM calls were made, no memories were written.[/dim]")
    console.print(_JSON_HINT)


def render_failure(error: BaseException, console: Console) -> None:
    """Render an exception that escaped the orchestrator entirely.

    Used by the CLI when ``ingestor.ingest()`` itself raises (vs.
    returning an IngestResult with ``success=False``). The IngestResult
    path goes through :func:`render_result` which surfaces the same kind
    of information from ``result.errors``.
    """
    console.print(
        f"[bold red]✗[/bold red] [bold red]ingest failed:[/bold red] {_escape(str(error))}"
    )
    console.print()
    console.print(f"  [dim]Type:[/dim] {type(error).__name__}")
    console.print(_JSON_HINT)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _render_failure_from_result(result: IngestResult, console: Console) -> None:
    """Render the ``not result.success`` branch from a full IngestResult."""
    console.print(
        f"[bold red]✗[/bold red] [bold]Failed to ingest[/bold] {_escape(result.source_uri)}"
    )
    console.print()

    # Pick a representative fatal error: prefer non-section steps (the run
    # never got past the orchestration phase) and fall back to the first
    # recorded error of any kind.
    fatal = next(
        (e for e in result.errors if e.section_index is None),
        result.errors[0] if result.errors else None,
    )
    if fatal is not None:
        _print_kv(console, "Step:", fatal.step)
        _print_kv(console, "Reason:", _escape(fatal.message))

    _print_warnings(console, result.warnings)
    if len(result.errors) > 1:
        console.print()
        console.print(f"  [dim]{len(result.errors) - 1} additional error(s) — see --json.[/dim]")

    console.print()
    console.print(_JSON_HINT)


def _print_kv(console: Console, label: str, value: str) -> None:
    """Print one ``  Label:   value`` row with right-padded dim label."""
    console.print(f"  [dim]{label:<{_LABEL_WIDTH}}[/dim] {value}")


def _print_archive_line(console: Console, result: IngestResult) -> None:
    """Surface the archive file_id when one was actually stamped.

    Only prints when ``result.archived_file_id`` is populated — covers
    the three reasons it can be ``None`` (opt-out, no FilesClient, upload
    failed) with a single check, no inference from ``errors``.
    """
    if result.archived_file_id is None:
        return
    console.print()
    console.print(f"  [dim]Original archived:[/dim] file_id={_escape(result.archived_file_id)}")


def _print_warnings(console: Console, warnings: list[str]) -> None:
    """Print up to three warnings as `⚠ message` lines."""
    if not warnings:
        return
    console.print()
    for w in warnings[:_MAX_INLINE_ERRORS]:
        console.print(f"  [bold yellow]⚠[/bold yellow] [yellow]{_escape(w)}[/yellow]")
    if len(warnings) > _MAX_INLINE_ERRORS:
        more = len(warnings) - _MAX_INLINE_ERRORS
        console.print(f"  [dim]... {more} more warning(s).[/dim]")


def _print_error_list(console: Console, errors: list[IngestErrorRecord]) -> None:
    """Print up to three per-step errors with step-prefixed context.

    The first three are shown in full. Anything beyond that is
    summarized as a count with a pointer to --json — keeps the
    visible footprint bounded on long runs.
    """
    if not errors:
        return
    console.print()
    for e in errors[:_MAX_INLINE_ERRORS]:
        subject = e.step if e.section_index is None else f"section {e.section_index}"
        console.print(
            f"  [bold yellow]⚠[/bold yellow] [yellow]{subject}[/yellow]: {_escape(e.message)}"
        )
    if len(errors) > _MAX_INLINE_ERRORS:
        more = len(errors) - _MAX_INLINE_ERRORS
        console.print(f"  [dim]... {more} more (run with --json for full list).[/dim]")


def _format_cost(result: IngestResult) -> str:
    """Render the cost line with provider attribution.

    Uses ``~$X.XX`` when an estimate is available; otherwise falls back
    to provider names alone (the orchestrator does not currently track
    actual USD spend for non-estimate runs).
    """
    cost = result.cost
    parts: list[str] = []
    if cost.est_usd is not None:
        parts.append(f"~${cost.est_usd:.2f}")
    providers: list[str] = []
    if cost.text_provider:
        providers.append(f"text={cost.text_provider}")
    if cost.vision_provider:
        providers.append(f"vision={cost.vision_provider}")
    if providers:
        joined = ", ".join(providers)
        parts.append(f"[dim]({joined})[/dim]")
    return "  ".join(parts) if parts else "—"


def _escape(text: str) -> str:
    """Escape Rich markup so user-supplied strings cannot inject styles.

    Rich uses ``[bold]...[/bold]``-style markup; any source URI or error
    message containing ``[`` would otherwise be misinterpreted. We do not
    use :func:`rich.markup.escape` because it does not handle the common
    case of trailing ``]`` from JSON / paths cleanly — replacing ``[``
    with the bracket-escape sequence is sufficient and reversible.
    """
    return text.replace("[", r"\[")
