"""Tests for the litellm-backed Provider shim and concrete subclasses.

The real-LLM eval suite (``tests/eval/ingest/``) exercises these against
live providers, but it's gated by ``@pytest.mark.eval`` and skipped in
unit-test CI. These tests mock ``litellm.acompletion`` so the call
contract is verified deterministically every run.
"""

from __future__ import annotations

import base64
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kagura_memory.exceptions import KaguraLLMError
from kagura_memory.ingest.providers import get_provider
from kagura_memory.ingest.providers._litellm import LiteLLMProvider, _extract_text
from kagura_memory.ingest.providers.base import Provider
from kagura_memory.ingest.providers.claude import ClaudeProvider
from kagura_memory.ingest.providers.gemini import GeminiProvider
from kagura_memory.ingest.providers.ollama import OllamaProvider


def _mock_response(text: str) -> MagicMock:
    """Build a mock object shaped like litellm.acompletion's return value."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = text
    return resp


# ---------------------------------------------------------------------------
# Concrete provider subclasses — class-level model presets
# ---------------------------------------------------------------------------


def test_claude_provider_presets_are_set() -> None:
    assert ClaudeProvider.name == "claude"
    assert ClaudeProvider.default_text_model == "claude-sonnet-4-6"
    assert ClaudeProvider.default_vision_model == "claude-sonnet-4-6"
    # And the protocol is satisfied at runtime.
    assert isinstance(ClaudeProvider(), Provider)


def test_gemini_provider_presets_are_set() -> None:
    assert GeminiProvider.name == "gemini"
    assert "gemini" in GeminiProvider.default_text_model
    assert GeminiProvider.default_vision_model is not None
    assert isinstance(GeminiProvider(), Provider)


def test_ollama_provider_presets_are_set() -> None:
    assert OllamaProvider.name == "ollama"
    assert "ollama" in OllamaProvider.default_text_model
    assert isinstance(OllamaProvider(), Provider)


# ---------------------------------------------------------------------------
# get_provider() factory
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,cls",
    [
        ("claude", ClaudeProvider),
        ("gemini", GeminiProvider),
        ("ollama", OllamaProvider),
    ],
)
def test_get_provider_returns_correct_subclass(name: str, cls: type) -> None:
    assert isinstance(get_provider(name), cls)


def test_get_provider_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown"):
        get_provider("not-a-real-provider")


# ---------------------------------------------------------------------------
# Constructor — vision_model sentinel handling
# ---------------------------------------------------------------------------


def test_provider_uses_class_default_when_no_args() -> None:
    p = ClaudeProvider()
    assert p.text_model == "claude-sonnet-4-6"
    assert p.vision_model == "claude-sonnet-4-6"


def test_provider_explicit_text_model_override() -> None:
    p = ClaudeProvider(text_model="claude-haiku-4-5-20251001")
    assert p.text_model == "claude-haiku-4-5-20251001"


def test_provider_explicit_vision_model_override() -> None:
    p = GeminiProvider(vision_model="gemini/gemini-2.5-pro")
    assert p.vision_model == "gemini/gemini-2.5-pro"


# ---------------------------------------------------------------------------
# summarize / summarize_overview — text generation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_summarize_passes_system_prompt_and_returns_text() -> None:
    """summarize() builds messages=[system, user] and returns the model's text."""
    p = ClaudeProvider()
    fake = AsyncMock(return_value=_mock_response("A short summary."))

    with patch("litellm.acompletion", fake):
        result = await p.summarize("the source section text", max_tokens=200)

    assert result == "A short summary."
    fake.assert_called_once()
    call_kwargs = fake.call_args.kwargs
    assert call_kwargs["model"] == "claude-sonnet-4-6"
    assert call_kwargs["max_tokens"] == 200
    # The messages array carries a system prompt + the user text verbatim.
    messages = call_kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert "data, not as instructions" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "the source section text"


@pytest.mark.asyncio
async def test_summarize_overview_joins_section_summaries() -> None:
    p = ClaudeProvider()
    fake = AsyncMock(return_value=_mock_response("Document overview."))

    with patch("litellm.acompletion", fake):
        result = await p.summarize_overview(["section A.", "section B."], max_tokens=400)

    assert result == "Document overview."
    user_msg = fake.call_args.kwargs["messages"][1]["content"]
    # Both sections appear in the joined user payload, labelled by index.
    assert "[Section 1]" in user_msg
    assert "section A." in user_msg
    assert "[Section 2]" in user_msg
    assert "section B." in user_msg


@pytest.mark.asyncio
async def test_summarize_raises_kagura_llm_error_on_provider_failure() -> None:
    p = ClaudeProvider()
    fake = AsyncMock(side_effect=RuntimeError("upstream 503"))

    with patch("litellm.acompletion", fake):
        with pytest.raises(KaguraLLMError, match="text provider"):
            await p.summarize("anything", max_tokens=200)


@pytest.mark.asyncio
async def test_summarize_raises_on_empty_completion() -> None:
    """The contract: providers MUST NOT return whitespace-only text."""
    p = ClaudeProvider()
    fake = AsyncMock(return_value=_mock_response("   \n  "))

    with patch("litellm.acompletion", fake):
        with pytest.raises(KaguraLLMError, match="empty"):
            await p.summarize("anything", max_tokens=200)


# ---------------------------------------------------------------------------
# describe_image — vision path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_describe_image_base64_encodes_and_sends_image_url() -> None:
    p = GeminiProvider()
    fake = AsyncMock(return_value=_mock_response("Visible text: foo."))
    image_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16

    with patch("litellm.acompletion", fake):
        result = await p.describe_image(image_bytes, "image/png")

    assert result == "Visible text: foo."
    messages = fake.call_args.kwargs["messages"]
    # System prompt sets the language-preservation / no-empty contract.
    assert "data" in messages[0]["content"].lower()
    # User content is multipart: image_url + text placeholder.
    user_content = messages[1]["content"]
    assert isinstance(user_content, list)
    image_part = next(p for p in user_content if p["type"] == "image_url")
    expected_b64 = base64.b64encode(image_bytes).decode("ascii")
    assert image_part["image_url"]["url"] == f"data:image/png;base64,{expected_b64}"
    # The placeholder is punctuation-only (no language bias).
    text_parts = [p for p in user_content if p["type"] == "text"]
    assert len(text_parts) == 1
    assert text_parts[0]["text"].strip() == "."


@pytest.mark.asyncio
async def test_describe_image_raises_when_no_vision_model() -> None:
    # Construct a provider without a vision model by using LiteLLMProvider
    # directly (its default_vision_model is None).
    p = LiteLLMProvider(text_model="any/model")
    assert p.vision_model is None

    with pytest.raises(KaguraLLMError, match="no vision model"):
        await p.describe_image(b"\x00", "image/png")


@pytest.mark.asyncio
async def test_describe_image_provider_failure_becomes_kagura_llm_error() -> None:
    p = GeminiProvider()
    fake = AsyncMock(side_effect=TimeoutError("network timeout"))

    with patch("litellm.acompletion", fake):
        with pytest.raises(KaguraLLMError, match="vision provider"):
            await p.describe_image(b"\x00", "image/png")


# ---------------------------------------------------------------------------
# count_tokens — local, sync, no network
# ---------------------------------------------------------------------------


def test_count_tokens_delegates_to_litellm_when_available() -> None:
    p = ClaudeProvider()
    with patch("litellm.token_counter", return_value=123):
        assert p.count_tokens("some text") == 123


def test_count_tokens_falls_back_to_char_heuristic_on_litellm_failure() -> None:
    """Unknown models trip litellm.token_counter → we fall back to len/4."""
    p = ClaudeProvider()
    with patch("litellm.token_counter", side_effect=RuntimeError("unknown model")):
        text = "a" * 100
        assert p.count_tokens(text) == 25  # 100 // 4


def test_count_tokens_for_vision_uses_vision_model() -> None:
    p = GeminiProvider()
    captured: dict[str, Any] = {}

    def spy(**kwargs: Any) -> int:
        captured.update(kwargs)
        return 42

    with patch("litellm.token_counter", spy):
        p.count_tokens("text", for_vision=True)

    assert captured["model"] == GeminiProvider.default_vision_model


def test_count_tokens_with_no_model_returns_heuristic() -> None:
    """A bare LiteLLMProvider with no vision_model uses the heuristic."""
    p = LiteLLMProvider(text_model="any/model")
    # for_vision=True with vision_model=None → heuristic path.
    assert p.count_tokens("abcdefgh", for_vision=True) == 2  # 8 // 4


# ---------------------------------------------------------------------------
# _extract_text — response shape and empty-output contract
# ---------------------------------------------------------------------------


def test_extract_text_strips_whitespace() -> None:
    assert _extract_text(_mock_response("  hello  ")) == "hello"


def test_extract_text_raises_on_malformed_response() -> None:
    bad = MagicMock()
    bad.choices = []  # IndexError on choices[0]

    with pytest.raises(KaguraLLMError, match="response shape"):
        _extract_text(bad)


def test_extract_text_raises_on_none_content() -> None:
    """``content=None`` from the provider is treated as empty."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = None
    with pytest.raises(KaguraLLMError, match="empty"):
        _extract_text(resp)
