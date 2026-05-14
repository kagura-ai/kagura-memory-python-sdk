# Ingestion eval suite

LLM-quality checks for the `kagura ingest` summarizer prompts. These tests
call real LLM providers and are **default-skipped** so unit-test CI stays
cheap and deterministic.

## When to run

- Before merging changes to `src/kagura_memory/ingest/providers/_litellm.py`
  (any prompt edit).
- Before bumping a model default in `src/kagura_memory/ingest/providers/{claude,gemini,ollama}.py`.
- Weekly via the cron-scheduled `eval.yml` workflow (catches model drift
  that ships without an SDK change).

## How to run

```bash
# Requires at least one of:
#   ANTHROPIC_API_KEY=sk-ant-...
#   GEMINI_API_KEY=...
#
# Tests for the missing-key provider are individually skipped — running
# with only ANTHROPIC_API_KEY runs the Claude eval and skips Gemini.

uv run pytest -m eval tests/eval/ingest/
```

## What's covered

Each test runs the actual summarizer (`LiteLLMProvider.summarize`,
`summarize_overview`, `describe_image`) against a small fixture and
asserts deterministic properties:

| Test                                       | Property checked                                                       |
|--------------------------------------------|------------------------------------------------------------------------|
| `test_summary_is_non_empty`                | The summarizer never returns whitespace-only text.                     |
| `test_summary_respects_length_cap`         | Output stays within the "2-4 sentences" ceiling (≤7 terminators).     |
| `test_summary_preserves_source_language`   | A Japanese input gets a Japanese reply (CJK codepoint count ≥ 20).    |
| `test_overview_summarizer_is_non_empty`    | `summarize_overview()` also obeys the non-empty contract.              |

Language detection uses the inline `_is_cjk()` helper in
`test_eval_summarizer.py` — a small Hiragana/Katakana/CJK/Hangul
codepoint check. A heavier library (`langdetect`, `fastText`) would be
more rigorous; the codepoint check captures the failure mode the
prompt is defending against (the model replying in English on a
Japanese input) without adding a runtime dependency.

The eval suite has **no additional optional dependencies** beyond the
ingest extras already required to run the orchestrator
(`pip install kagura-memory[ingest-pdf]`). The real-LLM calls are
made through `litellm`, which is already a core dependency.

## What's intentionally NOT here

- **LLM-as-judge section coverage**: too speculative for v0.16.0; revisit
  when we have golden outputs to compare against.
- **Cost / latency benchmarks**: separate concern (CFO ticket).
- **Vision quality on real images**: pending a vision-source fixture.

## Fixture policy

The eval suite reuses `tests/fixtures/sample.pdf` (English PDF) and
includes inline Japanese test text. Adding a `sample_ja.pdf` fixture is a
follow-up — keep it under 5 KB so the repo doesn't bloat.
