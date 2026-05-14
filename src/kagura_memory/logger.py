"""Verbose logging for the Kagura SDK and CLI.

Two consumer classes share one event stream, three render formats:

- **Human operators** read Rich-formatted glyphs/colors on stderr
  (``output_format="rich"``, level-filtered by ``-v`` count).
- **AI agents / scripts** read newline-delimited JSON on stderr
  (``output_format="json"``); they parse the stream with ``jq -c`` or
  any line-buffered JSON reader and wait for the terminal event
  (``kind in ("success", "error")``) to know the operation finished.
- **Library callers** that don't opt in get silent output via
  :data:`_NULL_LOGGER` (``output_format="none"``).

**Stderr-only invariant**: both the Rich and JSON paths write to
``sys.stderr``. Progress must never collide with structured result
output on stdout (CLI consumers pipe stdout through ``jq``).

**Concurrency contract — single-producer per instance**: each logger
instance is meant to be driven by a single async task or thread.
Concurrent writes from multiple producers to the same instance would
interleave NDJSON lines and break line-buffered parsers downstream.
Operation-level dependency injection (each ``ingest()`` / ``upload()``
/ ``ingest_events()`` call gets its own logger) makes this the default
shape; if a future caller wants to fan out across tasks, it must
construct one logger per task and merge streams externally.

**Terminal-event contract**: callers that opt into the JSON path
should emit exactly one terminal event per operation, with either
``kind="success"`` or ``kind="error"`` as the **final** event in the
stream. Mid-stream non-fatal warnings use ``kind="warning"``;
``success`` / ``error`` appear ONLY at the end. AI consumers safely
``wait while kind not in ("success", "error")`` and treat the first
matching event as terminal. The logger itself does not enforce this —
entry points (``FileIngestor.ingest``, ``FilesClient.upload``,
``ResourceClient.ingest_events``) wrap their main body in
``try/except`` so that even an unhandled exception still emits a
terminal ``kind="error"`` before propagating.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from typing import Any, Literal

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

OutputFormat = Literal["rich", "json", "none"]
"""The three render targets for a :class:`VerboseLogger` instance."""

_NDJSON_SCHEMA_VERSION = 1
"""Schema version field embedded in every NDJSON event.

Bumped only when the wire format adds/removes/renames fields in a way
that breaks consumers that already shipped against ``v=1``. Consumers
MUST check ``v`` and refuse to parse unknown values (forward-compat:
additions inside a major version are allowed; removals are not)."""


class VerboseLogger:
    """Emit progress events for long-running SDK operations.

    The six public methods (:meth:`action`, :meth:`detail`,
    :meth:`success`, :meth:`warning`, :meth:`error`, :meth:`debug`)
    are render-format-agnostic — each picks the right path based on
    ``output_format`` and silently drops the call when the level
    filter rejects it. See the module docstring for the concurrency
    and terminal-event contracts.
    """

    def __init__(
        self,
        level: int = 0,
        console: Console | None = None,
        *,
        output_format: OutputFormat = "rich",
    ):
        """Initialize a logger.

        Args:
            level: Rich-path verbosity threshold. ``0`` is silent
                (matches the ``--progress=none`` and ``-v`` not given
                case), ``1`` shows actions/success/warning,
                ``2`` adds details, ``3`` adds debug panels. Ignored
                when ``output_format="json"`` — the JSON path emits
                every event regardless of level (downstream consumers
                filter by ``kind``).
            console: Optional Rich :class:`Console` instance. When
                omitted, a stderr-bound console is created. Tests
                pass a stub here to capture output.
            output_format: ``"rich"`` for Rich glyphs/colors,
                ``"json"`` for newline-delimited JSON, ``"none"`` for
                a NO-OP logger (used by :data:`_NULL_LOGGER`).
        """
        self.level = level
        self.output_format: OutputFormat = output_format
        # stderr-bound so progress never lands on stdout. Rich's
        # auto-detect handles TTY-vs-pipe degradation (no ANSI escapes
        # leak into a redirected stderr).
        self._console = console or Console(stderr=True)

    def _should_log_rich(self, min_level: int) -> bool:
        """True iff this logger should render a Rich event at ``min_level``."""
        return self.output_format == "rich" and self.level >= min_level

    def _emit_json(
        self,
        *,
        kind: str,
        stage: str | None,
        msg: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        """Serialize one NDJSON event to stderr.

        Writes directly to ``sys.stderr`` (not via Rich) so the line
        is exactly one ``\\n``-terminated JSON object with no markup
        injection. ``stage=None`` becomes ``"unknown"`` to keep the
        field required-and-present for downstream parsers. Stage-
        specific structured fields (counts, durations, file_id, ...)
        ride in the ``detail`` dict — the schema deliberately keeps the
        top-level event shape minimal so future fields don't bloat
        ``v=1``; consumers tolerate unknown ``detail`` keys.
        """
        # Capture `now` once: two separate ``datetime.now()`` calls would race
        # across a second boundary and produce an inconsistent ``ts`` field
        # (seconds from the first call, milliseconds from the second).
        now = datetime.now(UTC)
        event: dict[str, Any] = {
            "v": _NDJSON_SCHEMA_VERSION,
            "ts": now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z",
            "stage": stage or "unknown",
            "kind": kind,
        }
        if msg:
            event["msg"] = msg
        if detail:
            event["detail"] = detail
        # Serialization is best-effort. ``default=str`` handles common non-
        # serializable types (Path, UUID, …), but a class with a broken
        # ``__str__`` or a circular reference can still raise. The outer
        # try-except guarantees the "progress logging must never raise"
        # contract — on any serialization failure we emit a minimal
        # placeholder so the terminal-event invariant still holds.
        try:
            line = json.dumps(event, ensure_ascii=False, default=str) + "\n"
        except Exception:  # noqa: BLE001 — best-effort fallback for any json.dumps error
            fallback = {
                "v": _NDJSON_SCHEMA_VERSION,
                "ts": event["ts"],
                "stage": event["stage"],
                "kind": event["kind"],
                "msg": "<event serialization failed>",
            }
            line = json.dumps(fallback) + "\n"
        try:
            sys.stderr.write(line)
            sys.stderr.flush()
        except OSError:
            # Downstream consumer closed stderr early (BrokenPipeError) or
            # the FD is otherwise unwritable. Swallow — the operation being
            # logged must not crash because its observability stream went
            # away. Matches the "progress logging must never raise" contract
            # in the module docstring.
            pass

    def action(self, action: str, details: str = "", *, stage: str | None = None) -> None:
        """Log a major action — ``kind=action``, level ≥ 1."""
        if self.output_format == "none":
            return
        if self.output_format == "json":
            self._emit_json(
                kind="action",
                stage=stage,
                msg=action,
                detail={"desc": details} if details else None,
            )
            return
        if not self._should_log_rich(1):
            return
        self._console.print(f"[bold blue]→[/bold blue] {action}", end="")
        if details:
            self._console.print(f" [dim]{details}[/dim]")
        else:
            self._console.print()

    def detail(self, key: str, value: Any, *, stage: str | None = None) -> None:
        """Log a structured detail — ``kind=detail``, level ≥ 2."""
        if self.output_format == "none":
            return
        if self.output_format == "json":
            self._emit_json(kind="detail", stage=stage, msg=key, detail={"value": value})
            return
        if not self._should_log_rich(2):
            return
        self._console.print(f"  [cyan]•[/cyan] {key}: [yellow]{value}[/yellow]")

    def debug(
        self, title: str, data: Any, syntax: str = "json", *, stage: str | None = None
    ) -> None:
        """Log a debug payload — ``kind=debug``, level ≥ 3."""
        if self.output_format == "none":
            return
        if self.output_format == "json":
            # Serialize non-primitive payloads via str() so the JSON line
            # stays a single object. A class with a broken ``__str__``
            # would otherwise crash the operation; per the "progress
            # logging must never raise" contract, fall back to a safe
            # placeholder that names the type without invoking its
            # stringification. ``_emit_json`` has its own outer safety
            # net for json.dumps failures.
            if isinstance(data, (dict, list, str, int, float, bool, type(None))):
                serialized: Any = data
            else:
                try:
                    serialized = str(data)
                except Exception:  # noqa: BLE001 — never crash the caller on a bad __str__
                    serialized = f"<{type(data).__name__} __str__ raised>"
            self._emit_json(kind="debug", stage=stage, msg=title, detail={"data": serialized})
            return
        if not self._should_log_rich(3):
            return
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data, indent=2, ensure_ascii=False)
        else:
            data_str = str(data)
        if len(data_str) > 2000:
            data_str = data_str[:2000] + "\n... (truncated)"
        self._console.print(
            Panel(
                Syntax(data_str, syntax, theme="monokai", word_wrap=True),
                title=f"[bold magenta]{title}[/bold magenta]",
                border_style="magenta",
                expand=False,
            )
        )

    def success(
        self,
        message: str,
        *,
        stage: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        """Log a terminal success event — ``kind=success``, level ≥ 1.

        Per the terminal-event contract (module docstring), callers
        should emit success **exactly once** as the final event for
        an operation. Mid-stream OK states use :meth:`action` instead.
        """
        if self.output_format == "none":
            return
        if self.output_format == "json":
            self._emit_json(kind="success", stage=stage, msg=message, detail=detail)
            return
        if not self._should_log_rich(1):
            return
        self._console.print(f"[bold green]✓[/bold green] {message}")

    def warning(self, message: str, *, stage: str | None = None) -> None:
        """Log a non-fatal warning — ``kind=warning``, level ≥ 1.

        Mid-stream warnings stay non-terminal (they don't conclude
        the operation). Use :meth:`error` to signal a terminal failure.
        """
        if self.output_format == "none":
            return
        if self.output_format == "json":
            self._emit_json(kind="warning", stage=stage, msg=message)
            return
        if not self._should_log_rich(1):
            return
        self._console.print(f"[bold yellow]⚠[/bold yellow] {message}")

    def error(
        self,
        message: str,
        *,
        stage: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        """Log a terminal error event — ``kind=error``, always shown.

        Per the terminal-event contract, callers should emit error
        **exactly once** as the final event when the operation fails.
        Unlike the other methods, the Rich path renders errors at any
        level (errors are too important to gate behind ``-v``).
        """
        if self.output_format == "none":
            return
        if self.output_format == "json":
            self._emit_json(kind="error", stage=stage, msg=message, detail=detail)
            return
        # Rich path: error always renders, regardless of self.level.
        self._console.print(f"[bold red]✗[/bold red] {message}", style="bold red")


_NULL_LOGGER: VerboseLogger = VerboseLogger(level=0, output_format="none")
"""Module-level NO-OP logger.

Entry points that take ``logger: VerboseLogger | None = None`` should
normalize ``None`` to :data:`_NULL_LOGGER` via :func:`normalize_logger`
so call sites can drop the ``if self.logger:`` guards. The instance is
stateless beyond the output_format check, so a single shared instance
is safe across threads / async tasks (every method returns at the
``"none"`` short-circuit before touching any state)."""


def normalize_logger(logger: VerboseLogger | None) -> VerboseLogger:
    """Return ``logger`` when given, else the module-level :data:`_NULL_LOGGER`.

    The recommended way for SDK entry points (``FileIngestor.ingest``,
    ``FilesClient.upload``, ``ResourceClient.ingest_events``) to accept
    an optional logger without scattering ``logger or _NULL_LOGGER``
    expressions across the codebase. Keeping the normalization in one
    place also means the no-op fallback is documented at the import
    site rather than at every call site.
    """
    return logger if logger is not None else _NULL_LOGGER
