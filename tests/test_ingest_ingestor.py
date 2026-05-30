"""Tests for the FileIngestor orchestrator."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

# Real PDF extraction depends on pymupdf (an optional extra:
# `pip install kagura-memory[ingest-pdf]`). Without it, the orchestrator's
# extract step raises KaguraIngestError at runtime — but pytest collection
# itself is fine because the import is lazy inside extractors/pdf.py. We
# still skip the module here so the failure surface stays sane on a bare
# `pip install kagura-memory[dev]` install.
pytest.importorskip("pymupdf", reason="pymupdf not installed — install [ingest-pdf] extras")

from kagura_memory.client import KaguraClient  # noqa: E402
from kagura_memory.ingest import FileIngestor  # noqa: E402
from kagura_memory.ingest.ingestor import _OVERVIEW_RESERVED, _SECTION_RESERVED  # noqa: E402
from kagura_memory.ingest.providers.base import Provider  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "sample.pdf"


pytestmark = pytest.mark.skipif(not FIXTURE.exists(), reason="sample.pdf fixture missing")


class FakeProvider:
    """Deterministic in-memory Provider for tests."""

    name = "fake"
    default_text_model = "fake/text"
    default_vision_model: str | None = "fake/vision"
    text_model = "fake/text"
    vision_model: str | None = "fake/vision"

    def __init__(self, has_vision: bool = True) -> None:
        if not has_vision:
            self.vision_model = None
        self.summarize_calls: list[str] = []
        self.overview_calls: list[list[str]] = []
        self.steering_seen: list[str | None] = []

    async def summarize(self, text: str, *, max_tokens: int, steering: str | None = None) -> str:
        self.summarize_calls.append(text)
        self.steering_seen.append(steering)
        return f"[summary len={len(text)}]"

    async def summarize_overview(
        self, section_summaries: list[str], *, max_tokens: int, steering: str | None = None
    ) -> str:
        self.overview_calls.append(list(section_summaries))
        self.steering_seen.append(steering)
        return f"[overview of {len(section_summaries)} sections]"

    async def describe_image(self, image_bytes: bytes, mime: str) -> str:
        return f"[image {mime} len={len(image_bytes)}]"

    def count_tokens(self, text: str, *, for_vision: bool = False) -> int:
        return max(1, len(text) // 4)


def _make_client() -> KaguraClient:
    client = KaguraClient(api_key="test", mcp_url="https://test.com/mcp")
    client._session_id = "test-session"
    return client


@pytest.mark.asyncio
async def test_dry_run_returns_estimate_without_calling_provider(tmp_path: Any) -> None:
    """--dry-run path: never calls summarize / describe_image / remember."""
    client = _make_client()
    provider = FakeProvider()
    ingestor = FileIngestor(
        client=client,
        text_provider=provider,
        vision_provider=provider,
    )
    with patch.object(client, "_call_tool", new_callable=AsyncMock) as remember_mock:
        result = await ingestor.estimate_cost(str(FIXTURE))

    assert result.is_dry_run is True
    assert result.cost.is_estimate is True
    assert result.cost.text_provider == "fake"
    assert result.overview_id is None
    assert result.section_ids == []
    assert provider.summarize_calls == []  # no LLM call
    assert provider.overview_calls == []
    remember_mock.assert_not_called()  # no MCP call
    await client.close()


@pytest.mark.asyncio
async def test_full_ingest_writes_overview_and_sections() -> None:
    """End-to-end: PDF → 1 overview + N sections + linked_memory_ids edges."""
    client = _make_client()
    provider = FakeProvider()
    ingestor = FileIngestor(
        client=client,
        text_provider=provider,
        vision_provider=None,
    )

    # The MCP `remember` tool returns a fresh memory_id per call.
    counter = {"n": 0}

    async def fake_call_tool(tool: str, args: dict[str, Any]) -> dict[str, Any]:
        counter["n"] += 1
        return {"memory_id": f"mem-{counter['n']}", "args": args}

    captured_calls: list[dict[str, Any]] = []

    async def capture_remember(*, context_id: str, **kwargs: Any) -> dict[str, Any]:
        captured_calls.append({"context_id": context_id, **kwargs})
        return await fake_call_tool("remember", kwargs)

    with patch.object(client, "remember", side_effect=capture_remember):
        result = await ingestor.ingest(
            str(FIXTURE),
            context_id="ctx-uuid",
            tags=["pdf", "test"],
        )

    # 3 sections in the fixture → 1 overview + 3 sections.
    assert result.overview_id == "mem-1"
    assert len(result.section_ids) == 3
    assert result.section_ids == ["mem-2", "mem-3", "mem-4"]
    assert result.errors == []

    # Overview call: type=document, no parent details.
    overview_call = captured_calls[0]
    assert overview_call["type"] == "document"
    assert overview_call["details"]["format"] == "pdf"
    assert overview_call["details"]["section_count"] == 3
    assert overview_call["source_type"] == "file"
    assert overview_call["context_id"] == "ctx-uuid"
    assert overview_call["tags"] == ["pdf", "test"]

    # Section calls: type=document_section, parent_id set, linked_memory_ids
    # points at the overview, edge created server-side via declared_link.
    for i, section_call in enumerate(captured_calls[1:]):
        assert section_call["type"] == "document_section"
        assert section_call["details"]["parent_id"] == "mem-1"
        assert section_call["details"]["role"] == "section"
        assert section_call["details"]["section_index"] == i
        assert section_call["linked_memory_ids"] == ["mem-1"]
        assert section_call["context_id"] == "ctx-uuid"

    # Provider was called: 3 section summaries + 1 overview synthesis.
    assert len(provider.summarize_calls) == 3
    assert len(provider.overview_calls) == 1

    await client.close()


@pytest.mark.asyncio
async def test_context_config_steering_flows_to_provider() -> None:
    """End-to-end: context instructions become steering on every summarize call."""
    from kagura_memory.models import ContextDetail, ContextInfo

    client = _make_client()
    provider = FakeProvider()
    # Prime the steering cache so no get_context_info network call is made.
    client._context_info_cache["ctx-uuid"] = ContextInfo(
        context=ContextDetail(id="ctx-uuid", name="billing", summary="ignored — instructions win"),
        instructions="Focus on billing terminology.",
    )
    ingestor = FileIngestor(client=client, text_provider=provider, vision_provider=None)

    async def fake_remember(**kwargs: Any) -> dict[str, Any]:
        return {"memory_id": "mem-x"}

    with patch.object(client, "remember", side_effect=fake_remember):
        await ingestor.ingest(str(FIXTURE), context_id="ctx-uuid")

    # 3 sections + 1 overview, all steered by the resolved instructions.
    assert provider.steering_seen == ["Focus on billing terminology."] * 4

    await client.close()


@pytest.mark.asyncio
async def test_caller_steering_overrides_context_config() -> None:
    """An explicit steering= kwarg wins over context instructions."""
    from kagura_memory.models import ContextDetail, ContextInfo

    client = _make_client()
    provider = FakeProvider()
    client._context_info_cache["ctx-uuid"] = ContextInfo(
        context=ContextDetail(id="ctx-uuid", name="ctx", summary="ctx summary"),
        instructions="context instructions",
    )
    ingestor = FileIngestor(client=client, text_provider=provider, vision_provider=None)

    async def fake_remember(**kwargs: Any) -> dict[str, Any]:
        return {"memory_id": "mem-x"}

    with patch.object(client, "remember", side_effect=fake_remember):
        await ingestor.ingest(str(FIXTURE), context_id="ctx-uuid", steering="caller wins")

    assert set(provider.steering_seen) == {"caller wins"}

    await client.close()


@pytest.mark.asyncio
async def test_section_summarize_failure_collected_not_raised() -> None:
    """A per-section LLM failure is recorded in errors, not raised."""
    from kagura_memory.exceptions import KaguraLLMError

    client = _make_client()
    provider = FakeProvider()

    call_count = {"n": 0}
    original_summarize = provider.summarize

    async def flaky_summarize(text: str, *, max_tokens: int, steering: str | None = None) -> str:
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise KaguraLLMError("simulated LLM failure")
        return await original_summarize(text, max_tokens=max_tokens, steering=steering)

    provider.summarize = flaky_summarize  # type: ignore[method-assign]

    ingestor = FileIngestor(
        client=client,
        text_provider=provider,
    )

    counter = {"n": 0}

    async def fake_remember(**kwargs: Any) -> dict[str, Any]:
        counter["n"] += 1
        return {"memory_id": f"mem-{counter['n']}"}

    with patch.object(client, "remember", side_effect=fake_remember):
        result = await ingestor.ingest(str(FIXTURE), context_id="ctx-uuid")

    # Overview written, 2 of 3 sections written, 1 error captured.
    assert result.overview_id is not None
    assert len(result.section_ids) == 2
    assert any(e.step == "summarize" for e in result.errors)

    await client.close()


@pytest.mark.asyncio
async def test_overview_write_failure_aborts_section_writes() -> None:
    """If the overview write fails, no section writes happen (need parent id)."""
    client = _make_client()
    provider = FakeProvider()
    ingestor = FileIngestor(client=client, text_provider=provider)

    async def failing_remember(**kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("simulated server failure")

    with patch.object(client, "remember", side_effect=failing_remember):
        result = await ingestor.ingest(str(FIXTURE), context_id="ctx-uuid")

    assert result.overview_id is None
    assert result.section_ids == []
    assert any(e.step == "remember" and "overview" in e.message for e in result.errors)

    await client.close()


@pytest.mark.asyncio
async def test_unsupported_format_returns_error_record(tmp_path: Any) -> None:
    """Unrecognized format → extract step records error, no exception escapes."""
    client = _make_client()
    provider = FakeProvider()
    ingestor = FileIngestor(client=client, text_provider=provider)

    # An unknown extension with binary content the sniffer can't classify
    # (not %PDF, not HTML, no registered suffix/Content-Type).
    blob = tmp_path / "data.bin"
    blob.write_bytes(b"\x00\x01\x02\x03\x04not a known format")

    result = await ingestor.ingest(str(blob), context_id="ctx-uuid")
    assert result.overview_id is None
    assert result.errors
    assert result.errors[0].step == "extract"

    await client.close()


def test_provider_protocol_is_satisfied_by_fake() -> None:
    """FakeProvider must satisfy the runtime-checkable Provider Protocol."""
    assert isinstance(FakeProvider(), Provider)


# ---------------------------------------------------------------------------
# Archive integration (FilesClient + archive_original)
# ---------------------------------------------------------------------------


def _make_files_client_mock(
    *,
    file_id: str = "f8e9d0c1",
    sha256: str = "deadbeef",
    size_bytes: int = 1024,
) -> AsyncMock:
    """Build an AsyncMock that mimics FilesClient.upload() returning a FileObject.

    Returns a Mock whose ``upload`` coroutine resolves to a stand-in object
    with the ``id`` / ``sha256`` / ``size_bytes`` attributes the ingestor
    reads. We do not return a real FileObject because constructing one
    requires pydantic-validated datetime fields that the orchestrator
    never inspects.
    """
    from datetime import UTC, datetime

    from kagura_memory.models import FileObject

    fake_file = FileObject(
        id=file_id,
        workspace_id="ctx-uuid",
        filename="sample.pdf",
        content_type="application/pdf",
        size_bytes=size_bytes,
        sha256=sha256,
        status="confirmed",
        created_at=datetime.now(UTC),
    )
    files_client = AsyncMock()
    files_client.upload = AsyncMock(return_value=fake_file)
    files_client.close = AsyncMock(return_value=None)
    return files_client


@pytest.mark.asyncio
async def test_archive_uploads_source_and_stamps_file_id_on_overview() -> None:
    """archive_original=True + files_client → details.file_id is recorded."""
    client = _make_client()
    provider = FakeProvider()
    files_client = _make_files_client_mock(file_id="archived-id-1", sha256="abc123")
    ingestor = FileIngestor(
        client=client,
        text_provider=provider,
        vision_provider=None,
        files_client=files_client,
    )

    counter = {"n": 0}
    captured: list[dict[str, Any]] = []

    async def capture_remember(*, context_id: str, **kwargs: Any) -> dict[str, Any]:
        captured.append({"context_id": context_id, **kwargs})
        counter["n"] += 1
        return {"memory_id": f"mem-{counter['n']}"}

    with patch.object(client, "remember", side_effect=capture_remember):
        result = await ingestor.ingest(
            str(FIXTURE),
            context_id="ctx-uuid",
            archive_original=True,
        )

    files_client.upload.assert_called_once()
    upload_kwargs = files_client.upload.call_args.kwargs
    assert upload_kwargs["context_id"] == "ctx-uuid"
    assert isinstance(upload_kwargs["source"], bytes)
    assert upload_kwargs["filename"] == "sample.pdf"

    overview_call = captured[0]
    assert overview_call["details"]["file_id"] == "archived-id-1"
    assert overview_call["details"]["sha256"] == "abc123"
    assert overview_call["details"]["size_bytes"] == 1024
    assert result.success is True
    assert result.archived_file_id == "archived-id-1"

    await client.close()


@pytest.mark.asyncio
async def test_archive_opt_out_does_not_call_files_client() -> None:
    """archive_original=False → upload is not called even when files_client is set."""
    client = _make_client()
    provider = FakeProvider()
    files_client = _make_files_client_mock()
    ingestor = FileIngestor(
        client=client,
        text_provider=provider,
        vision_provider=None,
        files_client=files_client,
    )

    async def fake_remember(**_: Any) -> dict[str, Any]:
        return {"memory_id": "mem-x"}

    with patch.object(client, "remember", side_effect=fake_remember):
        await ingestor.ingest(str(FIXTURE), context_id="ctx-uuid", archive_original=False)

    files_client.upload.assert_not_called()
    await client.close()


@pytest.mark.asyncio
async def test_archive_failure_recorded_as_error_overview_still_written() -> None:
    """Best-effort: archive upload failure → IngestErrorRecord, ingest continues."""
    client = _make_client()
    provider = FakeProvider()
    files_client = AsyncMock()
    files_client.upload = AsyncMock(side_effect=RuntimeError("R2 boom"))
    ingestor = FileIngestor(
        client=client,
        text_provider=provider,
        vision_provider=None,
        files_client=files_client,
    )

    captured: list[dict[str, Any]] = []

    async def capture_remember(*, context_id: str, **kwargs: Any) -> dict[str, Any]:
        captured.append(kwargs)
        return {"memory_id": "mem-x"}

    with patch.object(client, "remember", side_effect=capture_remember):
        result = await ingestor.ingest(str(FIXTURE), context_id="ctx-uuid", archive_original=True)

    assert result.overview_id == "mem-x"  # overview still succeeded
    assert any(e.step == "archive" for e in result.errors)
    # Overview details should NOT carry a file_id when archive failed.
    assert "file_id" not in captured[0]["details"]

    await client.close()


@pytest.mark.asyncio
async def test_fetch_error_surfaces_as_ingest_result_in_full_path(tmp_path: Any) -> None:
    """A KaguraFetchError during ingest() must become IngestResult, not raise.

    Scripts using --json depend on getting machine-readable output on all
    failure modes, including missing files / blocked URLs / DNS errors.
    """
    client = _make_client()
    provider = FakeProvider()
    ingestor = FileIngestor(
        client=client,
        text_provider=provider,
        vision_provider=None,
    )
    nonexistent = str(tmp_path / "does-not-exist.pdf")

    result = await ingestor.ingest(nonexistent, context_id="ctx-uuid")

    assert result.success is False
    assert result.overview_id is None
    assert any(e.step == "fetch" for e in result.errors)
    assert nonexistent in result.source_uri or "does-not-exist" in result.source_uri
    assert result.source_type == "file"

    await client.close()


@pytest.mark.asyncio
async def test_fetch_error_surfaces_as_ingest_result_in_dry_run(tmp_path: Any) -> None:
    """Same as above but for the --dry-run path (estimate_cost)."""
    client = _make_client()
    provider = FakeProvider()
    ingestor = FileIngestor(
        client=client,
        text_provider=provider,
        vision_provider=None,
    )
    nonexistent = str(tmp_path / "does-not-exist.pdf")

    result = await ingestor.estimate_cost(nonexistent)

    assert result.is_dry_run is True
    assert result.success is False
    assert any(e.step == "fetch" for e in result.errors)

    await client.close()


@pytest.mark.asyncio
async def test_archive_no_files_client_skips_silently() -> None:
    """archive_original=True but no files_client → no upload, no error."""
    client = _make_client()
    provider = FakeProvider()
    ingestor = FileIngestor(
        client=client,
        text_provider=provider,
        vision_provider=None,
        files_client=None,  # explicit None
    )

    async def fake_remember(**_: Any) -> dict[str, Any]:
        return {"memory_id": "mem-x"}

    with patch.object(client, "remember", side_effect=fake_remember):
        result = await ingestor.ingest(str(FIXTURE), context_id="ctx-uuid", archive_original=True)

    assert result.success is True
    assert not any(e.step == "archive" for e in result.errors)
    await client.close()


# ---------------------------------------------------------------------------
# details_extra (#120)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_details_extra_stamped_on_overview_and_sections() -> None:
    """Happy path: details_extra keys appear in every remember() call."""
    client = _make_client()
    provider = FakeProvider()
    ingestor = FileIngestor(
        client=client,
        text_provider=provider,
        vision_provider=None,
    )

    counter = {"n": 0}
    captured: list[dict[str, Any]] = []

    async def capture_remember(*, context_id: str, **kwargs: Any) -> dict[str, Any]:
        captured.append({"context_id": context_id, **kwargs})
        counter["n"] += 1
        return {"memory_id": f"mem-{counter['n']}"}

    with patch.object(client, "remember", side_effect=capture_remember):
        result = await ingestor.ingest(
            str(FIXTURE),
            context_id="ctx-uuid",
            details_extra={"connector_id": "C1", "platform": "slack"},
        )

    assert result.success is True
    # All remember() calls (overview + sections) must carry the extra keys.
    assert len(captured) >= 2
    for call in captured:
        assert call["details"]["connector_id"] == "C1"
        assert call["details"]["platform"] == "slack"

    await client.close()


@pytest.mark.asyncio
async def test_details_extra_none_default_no_extra_keys() -> None:
    """Omitting details_extra (None) produces no extra keys on details — regression guard."""
    client = _make_client()
    provider = FakeProvider()
    ingestor = FileIngestor(
        client=client,
        text_provider=provider,
        vision_provider=None,
    )

    counter = {"n": 0}
    captured: list[dict[str, Any]] = []

    async def capture_remember(*, context_id: str, **kwargs: Any) -> dict[str, Any]:
        captured.append({"context_id": context_id, **kwargs})
        counter["n"] += 1
        return {"memory_id": f"mem-{counter['n']}"}

    with patch.object(client, "remember", side_effect=capture_remember):
        result = await ingestor.ingest(str(FIXTURE), context_id="ctx-uuid")

    assert result.success is True
    overview_details = captured[0]["details"]
    # Every key the SDK stamps must be declared reserved (drift guard).
    assert set(overview_details.keys()) <= _OVERVIEW_RESERVED
    # No extra keys beyond what the SDK stamps.
    assert "connector_id" not in overview_details
    assert "platform" not in overview_details

    # Same drift guard for every section (Copilot #121 review).
    for section_call in captured[1:]:
        section_details = section_call["details"]
        assert set(section_details.keys()) <= _SECTION_RESERVED
        assert "connector_id" not in section_details
        assert "platform" not in section_details

    await client.close()


@pytest.mark.asyncio
async def test_details_extra_overview_reserved_collision_raises() -> None:
    """Passing a key reserved by the overview (file_id) raises ValueError before any remember."""
    client = _make_client()
    provider = FakeProvider()
    ingestor = FileIngestor(
        client=client,
        text_provider=provider,
        vision_provider=None,
    )

    with (
        patch.object(client, "remember", new_callable=AsyncMock) as mock_remember,
        patch.object(ingestor, "_fetch", new_callable=AsyncMock) as mock_fetch,
    ):
        with pytest.raises(ValueError, match="file_id"):
            await ingestor.ingest(
                str(FIXTURE),
                context_id="ctx-uuid",
                details_extra={"file_id": "fake"},
            )
        # "before any fetch or write" contract — both halves verified.
        mock_remember.assert_not_called()
        mock_fetch.assert_not_called()

    await client.close()


@pytest.mark.asyncio
async def test_details_extra_section_reserved_collision_raises() -> None:
    """Passing a key reserved by section writes (parent_id) raises ValueError before any write."""
    client = _make_client()
    provider = FakeProvider()
    ingestor = FileIngestor(
        client=client,
        text_provider=provider,
        vision_provider=None,
    )

    with (
        patch.object(client, "remember", new_callable=AsyncMock) as mock_remember,
        patch.object(ingestor, "_fetch", new_callable=AsyncMock) as mock_fetch,
    ):
        with pytest.raises(ValueError, match="parent_id"):
            await ingestor.ingest(
                str(FIXTURE),
                context_id="ctx-uuid",
                details_extra={"parent_id": "fake"},
            )
        mock_remember.assert_not_called()
        mock_fetch.assert_not_called()

    await client.close()


@pytest.mark.asyncio
async def test_details_extra_multiple_reserved_collisions_sorted() -> None:
    """Multiple reserved key collisions are reported together in sorted order."""
    client = _make_client()
    provider = FakeProvider()
    ingestor = FileIngestor(
        client=client,
        text_provider=provider,
        vision_provider=None,
    )

    with (
        patch.object(client, "remember", new_callable=AsyncMock) as mock_remember,
        patch.object(ingestor, "_fetch", new_callable=AsyncMock) as mock_fetch,
    ):
        with pytest.raises(ValueError, match="file_id") as exc_info:
            await ingestor.ingest(
                str(FIXTURE),
                context_id="ctx-uuid",
                details_extra={"file_id": "x", "parent_id": "y"},
            )
        # Both keys must appear in the error message, in sorted order.
        msg = str(exc_info.value)
        assert "file_id" in msg
        assert "parent_id" in msg
        file_id_pos = msg.index("file_id")
        parent_id_pos = msg.index("parent_id")
        assert file_id_pos < parent_id_pos, "keys must appear in sorted order (file_id < parent_id)"
        mock_remember.assert_not_called()
        mock_fetch.assert_not_called()

    await client.close()


@pytest.mark.asyncio
async def test_details_extra_empty_dict_seals_against_post_call_mutation() -> None:
    """Empty dict still triggers the seal; caller post-call mutation cannot inject reserved keys.

    Regression guard for the TOCTOU window between validation and per-writer
    update() calls. With `if details_extra is not None`, the empty-dict path
    goes through the shallow-copy seal — so when the caller later mutates
    their original reference (here, during the overview remember() call),
    the section writes do NOT see the spoofed key.
    """
    client = _make_client()
    provider = FakeProvider()
    ingestor = FileIngestor(
        client=client,
        text_provider=provider,
        vision_provider=None,
    )

    caller_dict: dict[str, Any] = {}
    counter = {"n": 0}
    captured: list[dict[str, Any]] = []

    async def capture_remember(*, context_id: str, **kwargs: Any) -> dict[str, Any]:
        # Mutate the caller-owned dict mid-ingest. With the seal this has no
        # effect on the SDK's internal copy; without the seal the section
        # writes would pick up the spoofed file_id and corrupt graph state.
        caller_dict["file_id"] = "spoofed-by-caller"
        captured.append({"context_id": context_id, **kwargs})
        counter["n"] += 1
        return {"memory_id": f"mem-{counter['n']}"}

    with patch.object(client, "remember", side_effect=capture_remember):
        result = await ingestor.ingest(
            str(FIXTURE),
            context_id="ctx-uuid",
            details_extra=caller_dict,
        )

    assert result.success is True
    # No captured remember() call may carry the spoofed value.
    for call in captured:
        assert call["details"].get("file_id") != "spoofed-by-caller"

    await client.close()


@pytest.mark.asyncio
async def test_details_extra_non_str_keys_raise_type_error() -> None:
    """Non-str keys raise TypeError at entry, not a cryptic one from sorted() downstream."""
    client = _make_client()
    provider = FakeProvider()
    ingestor = FileIngestor(
        client=client,
        text_provider=provider,
        vision_provider=None,
    )

    with (
        patch.object(client, "remember", new_callable=AsyncMock) as mock_remember,
        patch.object(ingestor, "_fetch", new_callable=AsyncMock) as mock_fetch,
    ):
        with pytest.raises(TypeError, match="details_extra keys must be str"):
            await ingestor.ingest(
                str(FIXTURE),
                context_id="ctx-uuid",
                details_extra={42: "int-key-is-invalid"},  # type: ignore[dict-item]
            )
        mock_remember.assert_not_called()
        mock_fetch.assert_not_called()


# ---------------------------------------------------------------------------
# YouTube transcript source routing (issue #146)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_youtube_url_is_routed_to_transcript_path() -> None:
    """A YouTube URL is detected by host and routed to fetch_youtube()."""
    from kagura_memory.ingest.fetcher import FetchResult

    client = _make_client()
    provider = FakeProvider()
    ingestor = FileIngestor(client=client, text_provider=provider)

    url = "https://youtu.be/dQw4w9WgXcQ"
    md = b"# Test Video\n\n## [00:00]\n\nhello world\n"
    fake_fetch = AsyncMock(
        return_value=FetchResult(
            body=md,
            content_type="text/markdown",
            source_uri=url,
            source_type="url",
            final_url=url,
            bytes_read=len(md),
        )
    )

    with (
        patch("kagura_memory.ingest.ingestor.fetch_youtube", new=fake_fetch),
        patch.object(client, "remember", new_callable=AsyncMock) as mock_remember,
    ):
        mock_remember.return_value = {"memory_id": "mem-1"}
        result = await ingestor.ingest(url, context_id="ctx-uuid")

    fake_fetch.assert_awaited_once()
    assert fake_fetch.await_args is not None
    assert fake_fetch.await_args.args[0] == url
    assert result.source_uri == url
    assert result.source_type == "url"
    # Title flowed from the markdown H1 (TextExtractor promotes it) into the
    # overview memory's summary.
    overview_call = mock_remember.await_args_list[0]
    assert "Test Video" in overview_call.kwargs.get("summary", "")

    await client.close()


@pytest.mark.asyncio
async def test_youtube_missing_dependency_surfaces_as_fetch_error() -> None:
    """Without [ingest-youtube], the youtube path yields a step='fetch' error."""
    client = _make_client()
    provider = FakeProvider()
    ingestor = FileIngestor(client=client, text_provider=provider)

    from kagura_memory.exceptions import KaguraIngestError

    with patch(
        "kagura_memory.ingest.ingestor.fetch_youtube",
        new=AsyncMock(
            side_effect=KaguraIngestError(
                "youtube-transcript-api is not installed. Install with: "
                "pip install 'kagura-memory[ingest-youtube]'"
            )
        ),
    ):
        result = await ingestor.ingest(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ", context_id="ctx-uuid"
        )

    assert result.overview_id is None
    assert result.errors
    assert result.errors[0].step == "fetch"
    assert "ingest-youtube" in result.errors[0].message

    await client.close()


@pytest.mark.asyncio
async def test_http_youtube_url_not_routed_when_allow_http_false() -> None:
    """An http:// YouTube URL with allow_http=False bypasses fetch_youtube.

    is_youtube_url() is host-based and true for http:// too. The _fetch gate
    must NOT route such a URL to the transcript path; it falls through to the
    byte Fetcher, which raises the canonical 'http:// is disabled' error —
    keeping the allow_http contract consistent across source types.
    """
    from kagura_memory.exceptions import KaguraFetchError

    client = _make_client()
    provider = FakeProvider()
    ingestor = FileIngestor(client=client, text_provider=provider)

    url = "http://www.youtube.com/watch?v=dQw4w9WgXcQ"
    with patch("kagura_memory.ingest.ingestor.fetch_youtube", new_callable=AsyncMock) as fake_fetch:
        with pytest.raises(KaguraFetchError) as ei:
            await ingestor._fetch(
                url,
                max_bytes=10_000_000,
                connect_timeout=10.0,
                read_timeout=10.0,
                allow_http=False,
                allow_system_paths=False,
            )
    fake_fetch.assert_not_called()
    assert "http" in str(ei.value).lower()

    await client.close()


@pytest.mark.asyncio
async def test_http_youtube_url_routed_when_allow_http_true() -> None:
    """An http:// YouTube URL with allow_http=True DOES route to fetch_youtube."""
    from kagura_memory.ingest.fetcher import FetchResult

    client = _make_client()
    provider = FakeProvider()
    ingestor = FileIngestor(client=client, text_provider=provider)

    url = "http://www.youtube.com/watch?v=dQw4w9WgXcQ"
    md = b"# Test\n\n## [00:00]\n\nhi\n"
    fake_fetch = AsyncMock(
        return_value=FetchResult(
            body=md,
            content_type="text/markdown",
            source_uri=url,
            source_type="url",
            final_url=url,
            bytes_read=len(md),
        )
    )
    with patch("kagura_memory.ingest.ingestor.fetch_youtube", new=fake_fetch):
        result = await ingestor._fetch(
            url,
            max_bytes=10_000_000,
            connect_timeout=7.0,
            read_timeout=10.0,
            allow_http=True,
            allow_system_paths=False,
        )
    fake_fetch.assert_awaited_once()
    # connect_timeout is threaded into the YouTube path (oEmbed connect bound).
    assert fake_fetch.await_args is not None
    assert fake_fetch.await_args.kwargs.get("connect_timeout") == 7.0
    assert result.source_uri == url

    await client.close()
