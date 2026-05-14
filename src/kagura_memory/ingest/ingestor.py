"""File ingestion orchestrator.

Stitches together :class:`Fetcher`, :class:`Extractor`, the chunker, and
:class:`Provider` to produce 1 overview memory + N section memories with
``declared_link`` edges atomically created via
:meth:`KaguraClient.remember`'s ``linked_memory_ids`` parameter.

The orchestrator is best-effort: per-section LLM or write failures are
captured in :attr:`IngestResult.errors` and do NOT abort the run. Only
fetch failures and overview-write failures are terminal.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from ..client import KaguraClient
from ..exceptions import KaguraFetchError, KaguraIngestError, KaguraLLMError
from ..files_client import FilesClient
from ..logger import VerboseLogger, normalize_logger
from ..models import CostBreakdown, FileObject, IngestErrorRecord, IngestResult
from ._types import Chunk, ExtractedContent
from .chunker import chunk as do_chunk
from .extractors import get_extractor
from .fetcher import Fetcher, FetchResult
from .providers import get_provider
from .providers.base import Provider

_DEFAULT_OVERVIEW_TOKENS = 400
_DEFAULT_SECTION_TOKENS = 200
_DEFAULT_CONCURRENCY = 4


class FileIngestor:
    """Orchestrate URL/file → Memory Cloud ingestion."""

    def __init__(
        self,
        client: KaguraClient,
        text_provider_name: str = "claude",
        vision_provider_name: str | None = "gemini",
        text_provider: Provider | None = None,
        vision_provider: Provider | None = None,
        concurrency: int = _DEFAULT_CONCURRENCY,
        files_client: FilesClient | None = None,
    ):
        """Construct an ingestor.

        Args:
            client: Authenticated :class:`KaguraClient`. The ingestor does
                not own its lifecycle — caller manages ``async with``.
            text_provider_name: Short name for text summarization. Used
                only when ``text_provider`` is None. Default ``"claude"``
                because Sonnet's summarization quality outperforms cheaper
                models for the section/overview tasks.
            vision_provider_name: Short name for vision. Used only when
                ``vision_provider`` is None. Pass ``None`` to disable
                vision; image content will then be skipped with a warning.
                Default ``"gemini"`` because Gemini 2.5 Flash provides
                strong OCR + layout description at roughly 1/10 the
                per-token cost of Sonnet. Callers concerned about sending
                image bytes to a third-party provider should pass
                ``None`` (or ``--no-vision`` on the CLI).
            text_provider: Pre-built provider override (mainly for tests).
            vision_provider: Pre-built vision provider override (tests).
            concurrency: Maximum concurrent section summarization calls.
            files_client: Optional :class:`FilesClient` used to archive
                the source bytes to the workspace's R2 bucket. When
                supplied AND ``ingest(archive_original=True)``, the
                resulting :class:`FileObject` id is stamped on the
                overview memory's ``details.file_id`` so callers can
                later resolve memory → original bytes via
                :meth:`FilesClient.download_url`. Lifecycle is caller-
                managed, same as ``client``.
        """
        self._client = client
        self._text = text_provider or get_provider(text_provider_name)
        self._vision: Provider | None
        if vision_provider is not None:
            self._vision = vision_provider
        elif vision_provider_name is not None:
            self._vision = get_provider(vision_provider_name)
        else:
            self._vision = None
        self._concurrency = max(1, concurrency)
        self._files = files_client

    # --- Public API ----------------------------------------------------------

    async def ingest(
        self,
        source: str,
        *,
        context_id: str,
        tags: list[str] | None = None,
        importance: float = 0.7,
        max_bytes: int = 100 * 1024 * 1024,
        connect_timeout: float = 10.0,
        read_timeout: float = 60.0,
        allow_http: bool = False,
        allow_system_paths: bool = False,
        archive_original: bool = True,
        logger: VerboseLogger | None = None,
    ) -> IngestResult:
        """Run a full ingestion against ``source``.

        Args:
            archive_original: When True and a :class:`FilesClient` was
                supplied at construction time, the source bytes are
                uploaded to the workspace's object store and the
                resulting ``file_id`` is recorded on the overview
                memory's ``details.file_id``. Set to False to skip
                archival (saves R2 storage; the memory then has no
                back-reference to the original bytes). No effect when
                ``files_client`` is None.
            logger: Optional :class:`VerboseLogger` for progress events
                (Rich for human stderr, NDJSON for AI consumers). When
                omitted, the no-op :data:`_NULL_LOGGER` is used and no
                events are emitted — library default is silent. The
                terminal-event contract (exactly one ``kind=success`` or
                ``kind=error`` as the final event) is honored even when
                an unhandled exception escapes the body.
        """
        log = normalize_logger(logger)
        log.action("Fetching source", source, stage="fetch")
        try:
            fetched = await self._fetch(
                source,
                max_bytes=max_bytes,
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
                allow_http=allow_http,
                allow_system_paths=allow_system_paths,
            )
        except KaguraFetchError as e:
            # Best-effort contract: surface fetch failures via IngestResult
            # so --json output stays machine-readable for scripts.
            log.error(f"Fetch failed: {e}", stage="fetch", detail={"source": source})
            return _fetch_failure_result(source, e, is_dry_run=False, ingestor=self)
        log.detail("Fetched bytes", len(fetched.body), stage="fetch")
        try:
            result = await self._ingest_fetched(
                fetched,
                context_id=context_id,
                tags=tags,
                importance=importance,
                archive_original=archive_original,
                logger=log,
            )
        except BaseException as e:
            # Terminal-event guarantee: emit kind=error before propagating.
            log.error(f"Ingest failed: {e}", stage="complete", detail={"source": source})
            raise
        # Success means the overview memory was created. Per-section errors
        # are best-effort (see IngestResult.success): they ride in
        # ``error_count`` so AI consumers can react, but they do NOT flip the
        # terminal event to ``kind=error``. Only a missing overview does.
        terminal_detail = {
            "overview_id": result.overview_id,
            "section_count": len(result.section_ids),
            "error_count": len(result.errors),
        }
        if result.success:
            log.success("Ingest complete", stage="complete", detail=terminal_detail)
        else:
            log.error(
                "Ingest failed — overview not created",
                stage="complete",
                detail=terminal_detail,
            )
        return result

    async def estimate_cost(
        self,
        source: str,
        *,
        max_bytes: int = 100 * 1024 * 1024,
        connect_timeout: float = 10.0,
        read_timeout: float = 60.0,
        allow_http: bool = False,
        allow_system_paths: bool = False,
    ) -> IngestResult:
        """Local-only token + page count estimate (``--dry-run``).

        Performs the fetch + extract + chunk so the user gets accurate
        section counts, but does NOT call any LLM provider. The returned
        :class:`IngestResult` has ``is_dry_run=True``, no memory IDs, and
        a populated :class:`CostBreakdown` with ``is_estimate=True``.
        Fetch failures are also surfaced as a dry-run IngestResult with a
        ``step="fetch"`` error record, never as a raised exception, so
        ``kagura ingest --dry-run --json`` stays machine-readable.
        """
        try:
            fetched = await self._fetch(
                source,
                max_bytes=max_bytes,
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
                allow_http=allow_http,
                allow_system_paths=allow_system_paths,
            )
        except KaguraFetchError as e:
            return _fetch_failure_result(source, e, is_dry_run=True, ingestor=self)

        try:
            content, chunks = self._extract_and_chunk(fetched)
        except KaguraIngestError as e:
            return IngestResult(
                is_dry_run=True,
                source_uri=fetched.source_uri,
                source_type=fetched.source_type,
                cost=CostBreakdown(is_estimate=True),
                errors=[
                    IngestErrorRecord(
                        step="extract", message=str(e), exception_type=type(e).__name__
                    )
                ],
            )

        # Estimate the prompt token bill the way the orchestrator will
        # actually spend it:
        #   * N section summaries feed each chunk's full text to the model.
        #   * 1 overview summary feeds the SECTION SUMMARIES (not the raw
        #     chunks) — we approximate that input with
        #     ``len(chunks) * _DEFAULT_SECTION_TOKENS`` (the cap each
        #     section summary will fit under).
        section_prompt = sum(self._text.count_tokens(c.text) for c in chunks)
        overview_prompt_est = len(chunks) * _DEFAULT_SECTION_TOKENS
        prompt_tokens = section_prompt + overview_prompt_est
        completion_tokens_est = len(chunks) * _DEFAULT_SECTION_TOKENS + _DEFAULT_OVERVIEW_TOKENS
        cost = CostBreakdown(
            is_estimate=True,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens_est,
            vision_tokens=None,
            est_usd=None,  # USD pricing varies by provider/region; left to caller
            text_provider=self._text.name,
            vision_provider=self._vision.name if self._vision else None,
        )
        warnings: list[str] = []
        if content.images and self._vision is None:
            warnings.append(
                f"{len(content.images)} image(s) detected; pass vision_provider to ingest them"
            )
        return IngestResult(
            is_dry_run=True,
            source_uri=fetched.source_uri,
            source_type=fetched.source_type,
            section_ids=[],
            estimated_section_count=len(chunks),
            skipped_images=len(content.images) if self._vision is None else 0,
            cost=cost,
            warnings=warnings,
        )

    # --- Internals -----------------------------------------------------------

    async def _fetch(
        self,
        source: str,
        *,
        max_bytes: int,
        connect_timeout: float,
        read_timeout: float,
        allow_http: bool,
        allow_system_paths: bool,
    ) -> FetchResult:
        async with Fetcher(
            max_bytes=max_bytes,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            allow_http=allow_http,
            allow_system_paths=allow_system_paths,
        ) as fetcher:
            return await fetcher.fetch(source)

    def _extract_and_chunk(self, fetched: FetchResult) -> tuple[ExtractedContent, list[Chunk]]:
        mime = _infer_mime(fetched)
        extractor = get_extractor(mime)
        content = extractor.extract(fetched.body, source_uri=fetched.source_uri)
        chunks = do_chunk(content, model=getattr(self._text, "text_model", None))
        return content, chunks

    async def _ingest_fetched(
        self,
        fetched: FetchResult,
        *,
        context_id: str,
        tags: list[str] | None,
        importance: float,
        archive_original: bool,
        logger: VerboseLogger | None = None,
    ) -> IngestResult:
        log = normalize_logger(logger)
        errors: list[IngestErrorRecord] = []
        warnings: list[str] = []

        log.action("Extracting and chunking", stage="chunk")
        try:
            content, chunks = self._extract_and_chunk(fetched)
        except (KaguraIngestError, ValueError) as e:
            return IngestResult(
                is_dry_run=False,
                source_uri=fetched.source_uri,
                source_type=fetched.source_type,
                cost=CostBreakdown(
                    text_provider=self._text.name,
                    vision_provider=self._vision.name if self._vision else None,
                ),
                errors=[
                    IngestErrorRecord(
                        step="extract", message=str(e), exception_type=type(e).__name__
                    )
                ],
            )

        skipped_images = 0
        if content.images and self._vision is None:
            skipped_images = len(content.images)
            warnings.append(f"{skipped_images} image(s) skipped — no vision provider configured")

        # Archive runs concurrently with both summarize_sections AND the
        # overview summarization that follows — the archive's result is
        # only needed inside _write_overview to stamp details.file_id.
        # Awaiting it later (vs. between section and overview LLM calls)
        # overlaps the R2 PUT with the overview LLM round-trip too.
        # Failures on either side are recorded as best-effort
        # ``IngestErrorRecord`` entries and do not abort the run.
        archive_task: asyncio.Task[FileObject | None] | None = None
        if archive_original and self._files is not None:
            archive_task = asyncio.create_task(
                self._archive_original(fetched, context_id=context_id, errors=errors)
            )

        log.action(f"Summarizing {len(chunks)} section(s)", stage="summarize")
        section_summaries = await self._summarize_sections(chunks, errors)
        log.detail("Sections summarized", len(section_summaries), stage="summarize")

        # Build the overview from the section summaries (in parallel with
        # the still-pending archive_task, if any).
        log.action("Summarizing overview", stage="summarize")
        try:
            overview_summary = await self._text.summarize_overview(
                [s for s in section_summaries if s], max_tokens=_DEFAULT_OVERVIEW_TOKENS
            )
        except KaguraLLMError as e:
            errors.append(
                IngestErrorRecord(
                    step="summarize", message=f"overview: {e}", exception_type=type(e).__name__
                )
            )
            overview_summary = content.title or fetched.source_uri

        # Now collect the archive result before writing the overview, so
        # archived.id can be stamped on details.file_id.
        archived: FileObject | None = None
        if archive_task is not None:
            archived = await archive_task

        # Write the overview memory first (we need its id for section linking).
        overview_id = await self._write_overview(
            fetched=fetched,
            content=content,
            chunks=chunks,
            overview_summary=overview_summary,
            context_id=context_id,
            tags=tags,
            importance=importance,
            archived=archived,
            errors=errors,
        )

        section_ids: list[str] = []
        if overview_id is not None:
            section_ids = await self._write_sections(
                fetched=fetched,
                chunks=chunks,
                section_summaries=section_summaries,
                overview_id=overview_id,
                context_id=context_id,
                tags=tags,
                importance=importance,
                errors=errors,
            )

        cost = CostBreakdown(
            is_estimate=False,
            text_provider=self._text.name,
            vision_provider=self._vision.name if self._vision else None,
        )
        return IngestResult(
            is_dry_run=False,
            source_uri=fetched.source_uri,
            source_type=fetched.source_type,
            overview_id=overview_id,
            section_ids=section_ids,
            skipped_images=skipped_images,
            archived_file_id=archived.id if archived is not None else None,
            cost=cost,
            warnings=warnings,
            errors=errors,
        )

    async def _summarize_sections(
        self,
        chunks: list[Chunk],
        errors: list[IngestErrorRecord],
    ) -> list[str | None]:
        sem = asyncio.Semaphore(self._concurrency)
        results: list[str | None] = [None] * len(chunks)

        async def run(idx: int, chunk_obj: Chunk) -> None:
            async with sem:
                try:
                    summary = await self._text.summarize(
                        chunk_obj.text, max_tokens=_DEFAULT_SECTION_TOKENS
                    )
                except KaguraLLMError as e:
                    errors.append(
                        IngestErrorRecord(
                            step="summarize",
                            section_index=idx,
                            message=str(e),
                            exception_type=type(e).__name__,
                        )
                    )
                    return
                results[idx] = summary

        await asyncio.gather(*(run(i, c) for i, c in enumerate(chunks)))
        return results

    async def _archive_original(
        self,
        fetched: FetchResult,
        *,
        context_id: str,
        errors: list[IngestErrorRecord],
    ) -> FileObject | None:
        """Upload ``fetched.body`` to the workspace's object store.

        Returns the resulting :class:`FileObject` on success, or ``None``
        on any failure (with an :class:`IngestErrorRecord` of
        ``step="archive"`` appended). The orchestrator still writes
        memories in either case — archival is purely additive context.
        """
        assert self._files is not None  # caller guarded
        filename = _filename_from_source_uri(fetched.source_uri)
        # Forward the fetched / sniffed MIME so signed-URL archives don't
        # land in R2 as application/octet-stream when the filename happens
        # to lack a useful extension. ``None`` lets FilesClient fall back
        # to its own mimetypes.guess_type pipeline.
        content_type = fetched.content_type or None
        try:
            return await self._files.upload(
                context_id=context_id,
                source=fetched.body,
                filename=filename,
                content_type=content_type,
            )
        except Exception as e:  # noqa: BLE001
            errors.append(
                IngestErrorRecord(step="archive", message=str(e), exception_type=type(e).__name__)
            )
            return None

    async def _write_overview(
        self,
        *,
        fetched: FetchResult,
        content: ExtractedContent,
        chunks: list[Chunk],
        overview_summary: str,
        context_id: str,
        tags: list[str] | None,
        importance: float,
        archived: FileObject | None,
        errors: list[IngestErrorRecord],
    ) -> str | None:
        title = content.title or _infer_title(fetched.source_uri)
        details: dict[str, Any] = {
            "format": _infer_format(fetched),
            "source_uri": fetched.source_uri,
            "section_count": len(chunks),
            "extracted_at": datetime.now(UTC).isoformat(),
        }
        if content.page_count is not None:
            details["pages"] = content.page_count
        if archived is not None:
            details["file_id"] = archived.id
            details["sha256"] = archived.sha256
            details["size_bytes"] = archived.size_bytes

        try:
            result = await self._client.remember(
                context_id=context_id,
                summary=_truncate(f"{title}: {overview_summary}", 500),
                content=overview_summary,
                type="document",
                importance=importance,
                tags=tags,
                source_uri=fetched.source_uri,
                source_type=fetched.source_type,
                context_summary=_truncate(
                    f"Document overview ingested from {fetched.source_uri}", 2000
                ),
                details=details,
            )
        except Exception as e:  # noqa: BLE001
            errors.append(
                IngestErrorRecord(
                    step="remember", message=f"overview: {e}", exception_type=type(e).__name__
                )
            )
            return None
        if not isinstance(result, dict) or result.get("status") == "error":
            errors.append(
                IngestErrorRecord(
                    step="remember",
                    message=f"overview write reported error: {result}",
                )
            )
            return None
        memory_id = result.get("memory_id")
        if not memory_id:
            errors.append(
                IngestErrorRecord(
                    step="remember",
                    message=f"overview response missing memory_id: {result}",
                )
            )
            return None
        return str(memory_id)

    async def _write_sections(
        self,
        *,
        fetched: FetchResult,
        chunks: list[Chunk],
        section_summaries: list[str | None],
        overview_id: str,
        context_id: str,
        tags: list[str] | None,
        importance: float,
        errors: list[IngestErrorRecord],
    ) -> list[str]:
        sem = asyncio.Semaphore(self._concurrency)
        results: list[str | None] = [None] * len(chunks)
        section_importance = max(0.0, min(1.0, importance - 0.2))

        async def write_one(idx: int, chunk_obj: Chunk) -> None:
            summary = section_summaries[idx]
            if summary is None:
                # Section summary failed earlier; skip the write.
                return
            details = {
                "parent_id": overview_id,
                "role": "section",
                "section_index": idx,
                "depth": chunk_obj.depth,
            }
            if chunk_obj.anchor:
                details["anchor"] = chunk_obj.anchor
            if chunk_obj.page_range:
                details["page_range"] = list(chunk_obj.page_range)
            heading = chunk_obj.heading or f"section {idx + 1}"

            async with sem:
                try:
                    result = await self._client.remember(
                        context_id=context_id,
                        summary=_truncate(f"{heading}: {summary}", 500),
                        content=chunk_obj.text,
                        type="document_section",
                        importance=section_importance,
                        tags=tags,
                        source_uri=fetched.source_uri,
                        source_type=fetched.source_type,
                        context_summary=_truncate(summary, 2000),
                        details=details,
                        linked_memory_ids=[overview_id],
                    )
                except Exception as e:  # noqa: BLE001
                    errors.append(
                        IngestErrorRecord(
                            step="remember",
                            section_index=idx,
                            message=str(e),
                            exception_type=type(e).__name__,
                        )
                    )
                    return
            if not isinstance(result, dict) or result.get("status") == "error":
                errors.append(
                    IngestErrorRecord(
                        step="remember",
                        section_index=idx,
                        message=f"section write reported error: {result}",
                    )
                )
                return
            memory_id = result.get("memory_id")
            if memory_id:
                results[idx] = str(memory_id)
            else:
                errors.append(
                    IngestErrorRecord(
                        step="remember",
                        section_index=idx,
                        message=f"section response missing memory_id: {result}",
                    )
                )

        await asyncio.gather(*(write_one(i, c) for i, c in enumerate(chunks)))
        return [r for r in results if r is not None]


def _uri_path_lower(source_uri: str) -> str:
    """Lowercased URI path stripped of query / fragment.

    Used for extension-based MIME / format detection so URLs like
    ``https://example.com/report.pdf?token=...`` are recognized as
    PDFs. ``urlsplit().path`` is empty for non-URL strings (bare paths
    without a scheme) — fall back to the full string in that case.
    """
    parsed = urlsplit(source_uri)
    return (parsed.path or source_uri).lower()


def _infer_format(fetched: FetchResult) -> str:
    if fetched.content_type == "application/pdf":
        return "pdf"
    if fetched.content_type.startswith("image/"):
        return "image"
    if _uri_path_lower(fetched.source_uri).endswith(".pdf"):
        return "pdf"
    return fetched.content_type or "unknown"


def _infer_mime(fetched: FetchResult) -> str:
    """Decide which extractor to dispatch based on Content-Type and URI."""
    if fetched.content_type == "application/pdf":
        return "application/pdf"
    if _uri_path_lower(fetched.source_uri).endswith(".pdf"):
        return "application/pdf"
    # Magic-byte sniff for local files (no Content-Type).
    if fetched.body[:5] == b"%PDF-":
        return "application/pdf"
    if fetched.content_type:
        return fetched.content_type
    raise KaguraIngestError(
        f"could not determine MIME for {fetched.source_uri}; supported types: application/pdf"
    )


def _infer_title(source_uri: str) -> str:
    return source_uri.rsplit("/", 1)[-1] or source_uri


def _fetch_failure_result(
    source: str,
    error: KaguraFetchError,
    *,
    is_dry_run: bool,
    ingestor: FileIngestor,
) -> IngestResult:
    """Wrap a fetch failure as an IngestResult instead of raising.

    Lets ``kagura ingest`` (with or without ``--dry-run``) emit JSON for
    fetch-time failures (missing files, blocked URLs, DNS errors, byte
    cap, etc.) so scripts using ``--json`` get the same shape on every
    failure mode. The orchestrator-internal contract is unchanged: the
    raise-based ``Fetcher`` is the source of truth, and this is the
    single conversion point.
    """
    # We have no FetchResult here, so reconstruct the metadata the
    # renderer needs from the original ``source`` argument. URL-shaped
    # inputs are tagged source_type="url"; everything else is "file".
    source_type: Literal["url", "file"] = "url" if "://" in source else "file"
    cost = CostBreakdown(
        is_estimate=is_dry_run,
        text_provider=ingestor._text.name,
        vision_provider=ingestor._vision.name if ingestor._vision else None,
    )
    return IngestResult(
        is_dry_run=is_dry_run,
        source_uri=error.url or source,
        source_type=source_type,
        cost=cost,
        errors=[
            IngestErrorRecord(step="fetch", message=str(error), exception_type=type(error).__name__)
        ],
    )


def _filename_from_source_uri(source_uri: str) -> str:
    """Derive a filename for ``FilesClient.upload`` from a URI.

    ``urlsplit`` drops query / fragment for us so the upload metadata
    does not carry tracking parameters into R2. Works for both
    ``https://...`` and ``file://...``.
    """
    return Path(urlsplit(source_uri).path).name or "ingested-source"


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


__all__ = ["FileIngestor"]
