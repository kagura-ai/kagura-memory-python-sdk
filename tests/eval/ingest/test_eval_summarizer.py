"""Eval suite for the ingestion summarizer prompts.

These tests call real LLM providers, so they are gated behind
``@pytest.mark.eval`` and individually skipped when the relevant API key
env var is missing.

See ``tests/eval/ingest/README.md`` for the operating contract.
"""

from __future__ import annotations

import os

import pytest

# All eval tests use real LLM calls — keep them out of unit-test runs.
pytestmark = pytest.mark.eval


# ---------------------------------------------------------------------------
# Per-provider parametrization
# ---------------------------------------------------------------------------

_PROVIDERS = [
    pytest.param(
        "claude",
        marks=pytest.mark.skipif(
            not os.getenv("ANTHROPIC_API_KEY"),
            reason="ANTHROPIC_API_KEY not set",
        ),
    ),
    pytest.param(
        "gemini",
        marks=pytest.mark.skipif(
            not os.getenv("GEMINI_API_KEY"),
            reason="GEMINI_API_KEY not set",
        ),
    ),
]


def _provider(name: str):
    """Construct a LiteLLMProvider subclass by short name.

    Lazy import: keeps `litellm` off the import path of plain unit-test
    discovery runs.
    """
    from kagura_memory.ingest.providers import get_provider

    return get_provider(name)


# ---------------------------------------------------------------------------
# Inline fixture content
# ---------------------------------------------------------------------------

# A short English passage chosen for unambiguous summarizability.
_EN_SECTION = (
    "The Apollo program was the third United States human spaceflight program "
    "carried out by NASA. Apollo 11 was the first crewed mission to land on the "
    "Moon, on July 20, 1969. The program ran from 1961 to 1972, accomplishing "
    "six successful crewed lunar landings. Total program cost was about "
    "$25.4 billion at the time."
)

# A short Japanese passage chosen to test the language-preservation contract.
# Content: a brief description of the Apollo program in Japanese.
_JA_SECTION = (
    "アポロ計画は、アメリカ合衆国がNASAを通じて遂行した3度目の有人宇宙飛行計画である。"
    "1969年7月20日にアポロ11号が月面着陸に成功し、人類が初めて月面を歩いた。"
    "計画は1961年から1972年まで実施され、合計6回の有人月面着陸を成功させた。"
    "総予算は当時の金額で約254億ドルにのぼった。"
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_name", _PROVIDERS)
async def test_summary_is_non_empty(provider_name: str) -> None:
    """Contract: summarize() never returns whitespace-only output.

    Empty completions are caught by ``_extract_text`` and raised as
    ``KaguraLLMError``; this test exercises the real-model path that
    the unit test mocks out.
    """
    provider = _provider(provider_name)
    summary = await provider.summarize(_EN_SECTION, max_tokens=200)
    assert summary.strip(), "summarizer returned empty/whitespace output"


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_name", _PROVIDERS)
async def test_summary_respects_length_cap(provider_name: str) -> None:
    """Contract: section summary stays at "2-4 sentences" per system prompt.

    Counts sentence terminators (``.``, ``。``, ``!``, ``?``). The cap is
    intentionally generous — the prompt says 2-4 sentences but a 5-sentence
    response is acceptable; we fail only on 8+ which would indicate the
    length constraint was ignored entirely.
    """
    provider = _provider(provider_name)
    summary = await provider.summarize(_EN_SECTION, max_tokens=200)
    terminators = sum(summary.count(t) for t in (".", "。", "!", "?", "！", "？"))
    assert terminators <= 7, (
        f"summary exceeded the soft length ceiling ({terminators} terminators):\n{summary}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_name", _PROVIDERS)
async def test_summary_preserves_source_language(provider_name: str) -> None:
    """Contract: Japanese input gets a Japanese reply.

    Detects "Japanese-ness" by presence of CJK characters in the response.
    A robust detector (langdetect / fastText) would be more rigorous, but
    CJK-codepoint counting captures the failure mode the prompt is
    defending against (model replies in English on a Japanese input)
    without adding a heavy dependency.
    """
    provider = _provider(provider_name)
    summary = await provider.summarize(_JA_SECTION, max_tokens=200)
    cjk_count = sum(1 for ch in summary if _is_cjk(ch))
    assert cjk_count >= 20, (
        "summary did not preserve the source language "
        f"(only {cjk_count} CJK characters):\n{summary}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_name", _PROVIDERS)
async def test_overview_summarizer_is_non_empty(provider_name: str) -> None:
    """Contract: summarize_overview() also obeys the non-empty rule."""
    provider = _provider(provider_name)
    sections = [_EN_SECTION, _EN_SECTION[:200]]
    overview = await provider.summarize_overview(sections, max_tokens=400)
    assert overview.strip(), "overview summarizer returned empty output"


def _is_cjk(ch: str) -> bool:
    """Rough CJK / hiragana / katakana / Hangul range check.

    Matches the most common ranges used in the test fixture without
    pulling in unicodedata. The ranges cover:
      U+3040..U+309F  Hiragana
      U+30A0..U+30FF  Katakana
      U+4E00..U+9FFF  CJK Unified Ideographs
      U+AC00..U+D7AF  Hangul Syllables
    """
    code = ord(ch)
    return (
        0x3040 <= code <= 0x309F
        or 0x30A0 <= code <= 0x30FF
        or 0x4E00 <= code <= 0x9FFF
        or 0xAC00 <= code <= 0xD7AF
    )
