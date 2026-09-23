"""Guard the docs against drifting from memory-cloud behaviour again (#257).

Docstrings, CLI help, skills and examples described server behaviour that
changed between memory-cloud v0.53.0 and v0.76.0 — a fixed 30-day retention,
an immutable embedding model, a ``results`` key on ``load_pinned``. Nothing
raised, so nothing caught it. These tests pin the wording that is known to be
wrong and the pass-through key names a caller has to know about — key names,
not wording, so a docstring can be reworded without breaking CI.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from kagura_memory import AgentsClient, KaguraClient, TagInfo
from kagura_memory.cli import context_search_config

REPO_ROOT = Path(__file__).resolve().parent.parent

# Phrases describing behaviour memory-cloud no longer has.
_STALE_PHRASES = [
    "30-day retention",  # retention is per deployment since v0.66.0
    "recoverable for 30 days",
    "immutable after creation",  # an operator can migrate the model since v0.66.0
    "only ``recall.related_tags`` does",  # recall stopped sending sample_summary in v0.73.0
    'pinned["results"]',  # load_pinned returns ``memories``
]


def _doc_files() -> list[Path]:
    files = sorted((REPO_ROOT / "src" / "kagura_memory").rglob("*.py"))
    files += sorted((REPO_ROOT / "skills").rglob("SKILL.md"))
    files += sorted((REPO_ROOT / "examples").glob("*.py"))
    files += [REPO_ROOT / name for name in ("README.md", "AGENTS.md", "CLAUDE.md")]
    return files


@pytest.mark.parametrize("phrase", _STALE_PHRASES)
def test_no_stale_server_behaviour_in_docs(phrase: str) -> None:
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in _doc_files()
        if phrase in path.read_text(encoding="utf-8")
    ]
    assert not offenders, f"{phrase!r} is stale server behaviour, found in {offenders}"


@pytest.mark.parametrize(
    ("obj", "terms"),
    [
        (KaguraClient.load_pinned, ["memories", "truncated", "total_available"]),
        (KaguraClient.recall, ["tags_normalize", "degraded", "degraded_reason", "tag_suggestions"]),
        (KaguraClient.forget, ["CLEANUP_DELETED_MEMORIES_RETENTION_DAYS", "degraded"]),
        (KaguraClient.remember, ["persistence", "lint"]),
        (KaguraClient.update_memory, ["persistence", "lint", '``""``', "``{}``"]),
        (KaguraClient.create_context, ["invalid_embedding_model", "get_context_info"]),
        (KaguraClient.update_context, ["plan_required", "public_contexts"]),
        (KaguraClient.setup_resource, ["plan_required", "``resources``"]),
        (KaguraClient.merge_contexts, ["pending_embedding"]),
        (KaguraClient.get_agent_bootstrap, ["``trigger``", "include_details"]),
        (AgentsClient.bootstrap, ["``trigger``", "include_details"]),
        (KaguraClient.get_server_info, ["search_defaults", "model_extra"]),
        (KaguraClient.__init__, ["?profile=", "?tools=", "tools/list"]),
        (TagInfo, ["sample_summary"]),
    ],
    ids=lambda v: getattr(v, "__qualname__", "terms"),
)
def test_docstring_documents_current_server_behaviour(obj: object, terms: list[str]) -> None:
    doc = inspect.getdoc(obj) or ""
    missing = [term for term in terms if term not in doc]
    assert not missing, f"{getattr(obj, '__qualname__', obj)} docstring lacks {missing}"


def test_reranker_choices_match_documented_providers() -> None:
    """``--reranker`` and ``update_search_config`` name the server's providers only."""
    option = next(p for p in context_search_config.params if p.name == "reranker")
    choices = set(getattr(option.type, "choices", ()))
    assert choices == {"voyage", "cohere", "self_hosted"}

    doc = inspect.getdoc(KaguraClient.update_search_config) or ""
    for provider in choices:
        assert f'"{provider}"' in doc
    # The retired provider *value*; Ollama as a self_hosted backend is fine.
    assert '"ollama"' not in doc


def test_resource_skill_warns_about_plan_gated_creation() -> None:
    skill = (REPO_ROOT / "skills" / "resource" / "SKILL.md").read_text(encoding="utf-8")
    assert "plan" in skill.lower()
    assert "`resources`" in skill
    assert "`public_contexts`" in skill


def test_merge_example_reports_pending_embedding() -> None:
    """Since server v0.65.0 ``merged`` counts rows, some not searchable yet."""
    example = (REPO_ROOT / "examples" / "client_advanced.py").read_text(encoding="utf-8")
    assert "pending_embedding" in example
