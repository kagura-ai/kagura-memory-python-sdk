"""Tests for the FileIngestor orchestrator."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from kagura_memory.client import KaguraClient
from kagura_memory.ingest import FileIngestor
from kagura_memory.ingest.providers.base import Provider

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

    async def summarize(self, text: str, *, max_tokens: int) -> str:
        self.summarize_calls.append(text)
        return f"[summary len={len(text)}]"

    async def summarize_overview(self, section_summaries: list[str], *, max_tokens: int) -> str:
        self.overview_calls.append(list(section_summaries))
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
async def test_section_summarize_failure_collected_not_raised() -> None:
    """A per-section LLM failure is recorded in errors, not raised."""
    from kagura_memory.exceptions import KaguraLLMError

    client = _make_client()
    provider = FakeProvider()

    call_count = {"n": 0}
    original_summarize = provider.summarize

    async def flaky_summarize(text: str, *, max_tokens: int) -> str:
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise KaguraLLMError("simulated LLM failure")
        return await original_summarize(text, max_tokens=max_tokens)

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
    """Non-PDF body → extract step records error, no exception escapes."""
    client = _make_client()
    provider = FakeProvider()
    ingestor = FileIngestor(client=client, text_provider=provider)

    txt = tmp_path / "notes.txt"
    txt.write_bytes(b"not a pdf")

    result = await ingestor.ingest(str(txt), context_id="ctx-uuid")
    assert result.overview_id is None
    assert result.errors
    assert result.errors[0].step == "extract"

    await client.close()


def test_provider_protocol_is_satisfied_by_fake() -> None:
    """FakeProvider must satisfy the runtime-checkable Provider Protocol."""
    assert isinstance(FakeProvider(), Provider)
