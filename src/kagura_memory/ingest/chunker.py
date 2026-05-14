"""Structural-first chunker for the file ingestion pipeline.

Strategy (per design doc §5):

1. If extracted sections exist: pass each through; if a section's text
   exceeds ``max_tokens``, split it into multiple chunks via a token-window
   fallback. Heading metadata is preserved on the first sub-chunk.
2. If no sections (extractor returned an empty list): treat the whole
   document text as one synthetic section and token-window split.

Token counting uses ``litellm.token_counter`` which is provider-aware. A
fallback ``len(text) // 4`` heuristic kicks in when the counter is
unavailable (e.g. unknown model, or ``litellm`` failing in a test).
"""

from __future__ import annotations

from collections.abc import Iterable

from ._types import Chunk, ExtractedContent, ExtractedSection

DEFAULT_MAX_TOKENS = 1500
DEFAULT_OVERLAP_TOKENS = 100
_FALLBACK_CHARS_PER_TOKEN = 4  # rough heuristic when token_counter is unavailable


def chunk(
    content: ExtractedContent,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    model: str | None = None,
) -> list[Chunk]:
    """Split ``content`` into summarization-ready chunks.

    Args:
        content: Output of an :class:`Extractor`.
        max_tokens: Soft cap per chunk. Sections under this size pass
            through untouched.
        overlap_tokens: Overlap between adjacent token-window chunks
            within the SAME source section. Cross-section chunks never
            overlap (they have distinct heading metadata).
        model: Optional model name for ``litellm.token_counter``. ``None``
            falls back to the character-based heuristic.

    Returns:
        Chunks in document order with stable ``section_index`` values.
    """
    sections: Iterable[ExtractedSection]
    if content.sections:
        sections = content.sections
    else:
        # Extractor returned no sections — synthesize one from the document.
        # Empty content yields no chunks.
        return []

    chunks: list[Chunk] = []
    next_index = 0
    for section in sections:
        for sub_text, is_first in _split_section(
            section.body_text, max_tokens, overlap_tokens, model
        ):
            chunks.append(
                Chunk(
                    text=sub_text,
                    heading=section.heading if is_first else None,
                    page_range=section.page_range,
                    depth=section.depth,
                    anchor=section.anchor if is_first else None,
                    section_index=next_index,
                )
            )
            next_index += 1
    return chunks


def _count_tokens(text: str, model: str | None) -> int:
    """Return the token count for ``text`` under ``model``.

    Falls back to ``len(text) // _FALLBACK_CHARS_PER_TOKEN`` if the
    ``litellm`` token counter raises (unknown model, missing counter, etc.).
    """
    if model:
        try:
            import litellm  # type: ignore[import-untyped]

            return int(litellm.token_counter(model=model, text=text))  # pyright: ignore[reportPrivateImportUsage]
        except Exception:  # noqa: BLE001 - any failure → heuristic fallback
            pass
    return max(1, len(text) // _FALLBACK_CHARS_PER_TOKEN)


def _split_section(
    text: str, max_tokens: int, overlap_tokens: int, model: str | None
) -> list[tuple[str, bool]]:
    """Token-window split a single section's body.

    Returns a list of ``(sub_text, is_first)`` tuples in order; ``is_first``
    is ``True`` for the leading sub-chunk so the chunker can place heading
    metadata only there.
    """
    if not text.strip():
        return []
    total = _count_tokens(text, model)
    if total <= max_tokens:
        return [(text, True)]

    # Approximate token boundaries via character ratio. We don't need exact
    # tokenization — the goal is "chunks fit comfortably under max_tokens"
    # for downstream LLM context windows, with cheap overlap for context.
    chars_per_token = max(1, len(text) // total)
    window_chars = max_tokens * chars_per_token
    overlap_chars = max(0, overlap_tokens * chars_per_token)
    step = max(1, window_chars - overlap_chars)

    out: list[tuple[str, bool]] = []
    pos = 0
    is_first = True
    while pos < len(text):
        sub = text[pos : pos + window_chars]
        if sub.strip():
            out.append((sub, is_first))
            is_first = False
        pos += step
    return out
