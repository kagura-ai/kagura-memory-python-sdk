"""Steering resolution for context-aware ingest summarization (#148).

These exercise ``FileIngestor._resolve_steering`` directly with a primed
context-info cache, so they need neither pymupdf nor a PDF fixture — the
resolution precedence is pure logic over the cached ``ContextInfo``.
"""

from __future__ import annotations

import pytest

from kagura_memory.client import KaguraClient
from kagura_memory.ingest import FileIngestor
from kagura_memory.ingest.ingestor import _STEERING_MAX_CHARS
from kagura_memory.models import ContextDetail, ContextInfo


class _FakeProvider:
    """Minimal Provider stand-in — construction must not touch litellm."""

    name = "fake"
    default_text_model = "fake/text"
    default_vision_model: str | None = None
    text_model = "fake/text"
    vision_model: str | None = None

    async def summarize(self, text: str, *, max_tokens: int, steering: str | None = None) -> str:
        return f"[summary len={len(text)}]"

    async def summarize_overview(
        self, section_summaries: list[str], *, max_tokens: int, steering: str | None = None
    ) -> str:
        return f"[overview {len(section_summaries)}]"

    async def describe_image(self, image_bytes: bytes, mime: str) -> str:
        return "[image]"

    def count_tokens(self, text: str, *, for_vision: bool = False) -> int:
        return max(1, len(text) // 4)


def _ctx_info(*, instructions: str | None, summary: str | None) -> ContextInfo:
    return ContextInfo(
        context=ContextDetail(id="ctx-1", name="ctx", summary=summary),
        instructions=instructions,
    )


def _make_ingestor() -> FileIngestor:
    client = KaguraClient(api_key="test", mcp_url="https://test.com/mcp")
    client._session_id = "test-session"
    return FileIngestor(
        client=client, text_provider=_FakeProvider(), vision_provider=_FakeProvider()
    )


async def _resolve(
    ingestor: FileIngestor, *, cached: ContextInfo | None, caller: str | None
) -> str | None:
    # Prime the per-(client, context_id) cache so no network call is made.
    ingestor._client._context_info_cache["ctx-1"] = cached
    return await ingestor._resolve_steering("ctx-1", caller)


@pytest.mark.asyncio
async def test_precedence_caller_wins_over_context_config() -> None:
    ingestor = _make_ingestor()
    cached = _ctx_info(instructions="from instructions", summary="from summary")
    assert await _resolve(ingestor, cached=cached, caller="from caller") == "from caller"
    await ingestor._client.close()


@pytest.mark.asyncio
async def test_precedence_instructions_win_over_summary() -> None:
    ingestor = _make_ingestor()
    cached = _ctx_info(instructions="from instructions", summary="from summary")
    assert await _resolve(ingestor, cached=cached, caller=None) == "from instructions"
    await ingestor._client.close()


@pytest.mark.asyncio
async def test_precedence_summary_used_when_no_instructions() -> None:
    ingestor = _make_ingestor()
    cached = _ctx_info(instructions=None, summary="from summary")
    assert await _resolve(ingestor, cached=cached, caller=None) == "from summary"
    await ingestor._client.close()


@pytest.mark.asyncio
async def test_precedence_none_when_nothing_available() -> None:
    ingestor = _make_ingestor()
    # No caller, no cached context info at all (fetch failed / empty sentinel).
    assert await _resolve(ingestor, cached=None, caller=None) is None
    # And when the context info exists but carries no usable signal.
    cached = _ctx_info(instructions="   ", summary=None)
    assert await _resolve(ingestor, cached=cached, caller=None) is None
    await ingestor._client.close()


@pytest.mark.asyncio
async def test_whitespace_only_caller_falls_through_to_context_config() -> None:
    ingestor = _make_ingestor()
    cached = _ctx_info(instructions="from instructions", summary=None)
    assert await _resolve(ingestor, cached=cached, caller="   \n ") == "from instructions"
    await ingestor._client.close()


@pytest.mark.asyncio
async def test_resolved_steering_truncated_to_max_chars() -> None:
    ingestor = _make_ingestor()
    long_caller = "x" * (_STEERING_MAX_CHARS + 500)
    resolved = await _resolve(ingestor, cached=None, caller=long_caller)
    assert resolved is not None
    assert len(resolved) == _STEERING_MAX_CHARS

    # Truncation also applies to context-config-sourced steering.
    long_instr = "y" * (_STEERING_MAX_CHARS + 100)
    cached = _ctx_info(instructions=long_instr, summary=None)
    resolved2 = await _resolve(ingestor, cached=cached, caller=None)
    assert resolved2 is not None
    assert len(resolved2) == _STEERING_MAX_CHARS
    await ingestor._client.close()
