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
import re
from pathlib import Path

import pytest

from kagura_memory import (
    AgentsClient,
    FilesClient,
    KaguraClient,
    ResourceClient,
    TagInfo,
    WorkspaceClient,
)
from kagura_memory._http import mcp_session_expired, sanitize_server_detail
from kagura_memory.auth.device_flow import refresh_access_token, revoke_token
from kagura_memory.cli import context_search_config
from kagura_memory.models import MemoryStatItem, UsageInfo

REPO_ROOT = Path(__file__).resolve().parent.parent

# Phrases describing behaviour memory-cloud no longer has.
_STALE_PHRASES = [
    "30-day retention",  # retention is per deployment since v0.66.0
    "recoverable for 30 days",
    "immutable after creation",  # an operator can migrate the model since v0.66.0
    "only ``recall.related_tags`` does",  # recall stopped sending sample_summary in v0.73.0
    'pinned["results"]',  # load_pinned returns ``memories``
    "`--server` override: decline it",  # plugin 0.77.0 reads kagura-mcp's --guardrails
    "memory-cloud#1669 proposes",  # memory-cloud v0.77.0 shipped it
    # memory-cloud v0.77.0 accepts the Codex, Hermes and OpenClaw registration.
    "not allowlisted, memory-cloud#1657",
    "issues/1657) is open",
    # memory-cloud v0.78.0 shipped #1671: Hermes's device flow works from it.
    "waits on memory-cloud#1671",
    "device flow waits on",
    "device-flow sign-in still",
    # Each harness signed in end to end on a /mcp/w/<workspace-id> URL (#284).
    "not verified yet",
    "has not been verified yet",
    "_HARNESS_OAUTH_UNVERIFIED",
    # memory-cloud v0.135.0 (#2470): a URL without explore gets explore_hints_omitted.
    "present whenever ``include_explore_hints``",
    # #299: memory-cloud v0.82.0 (#1740) re-adopts only server-minted session ids.
    "re-adopts the unknown id as a new session",
    "v0.75.0 skips that session check",
    "v0.75.0 as deployed re-adopts",
    # #299: persistence.detail needs verbose=true since memory-cloud v0.93.0 (#1851).
    "consolidation_archive_min_age_days, detail}",
    # #299 is done: the SDK is checked against memory-cloud v0.141.0.
    "realignment with v0.141.0 is #299",
    "realignment with 0.141.0 is [#299]",
    # Since memory-cloud v0.93.0 a bare URL lists core; ?profile=full adds tools.
    "narrow `tools/list` only",
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


def test_no_doc_says_a_non_uuid_is_a_500_without_the_422() -> None:
    """memory-cloud v0.80.0 (#1693) answers a non-UUID context id with 422 ``VAL-001``.

    The old HTTP 500 (#285) may be named only in the sentence or clause that
    also names the 422 replacing it.
    """
    offenders = [
        f"{path.relative_to(REPO_ROOT)}: {clause}"
        for path in _doc_files()
        for clause in re.split(r"(?<=[.;])\s+", " ".join(path.read_text(encoding="utf-8").split()))
        if "non-UUID" in clause
        and "500" in clause
        and "422" not in clause
        and "VAL-001" not in clause
    ]
    assert not offenders, "a non-UUID is a 422 since memory-cloud v0.80.0: " + " | ".join(offenders)


@pytest.mark.parametrize(
    ("obj", "terms"),
    [
        (
            KaguraClient.load_pinned,
            ["memories", "truncated", "total_available", "max_chars", "context_summary_omitted"],
        ),
        (
            KaguraClient.recall,
            [
                "tags_normalize",
                "degraded",
                "degraded_reason",
                "tag_suggestions",
                # #298 / #295: the v0.108.0+ compact default and what restores the rest.
                "summary_cut",
                '``verbosity="full"``',
                "``content_next_offset``",
                "``folded``",
                "``full_applied``",
                "``context_summary_omitted``",
                "``selector``",
                "``spans``",
                "``explore_hints_omitted``",
                "``?add_tools=explore``",
                "``?profile=full``",
                # #299: supersede_candidate keys and the v0.78.0 annotation.
                "``verified``",
                "``acceptance``",
                "#1683",
            ],
        ),
        (KaguraClient.recall_nearby, ["``location``", "include_details", "details_omitted"]),
        (KaguraClient.recall_upcoming, ["details_omitted", "details_total_chars"]),
        (KaguraClient.forget, ["CLEANUP_DELETED_MEMORIES_RETENTION_DAYS", "degraded"]),
        (KaguraClient.remember, ["persistence", "lint", "``verbose=true``", "missing_fields"]),
        (KaguraClient.update_memory, ["persistence", "lint", '``""``', "``{}``", "#1803"]),
        (KaguraClient.create_context, ["invalid_embedding_model", "get_context_info"]),
        (KaguraClient.update_context, ["plan_required", "public_contexts"]),
        (KaguraClient.setup_resource, ["plan_required", "``resources``"]),
        (KaguraClient.merge_contexts, ["pending_embedding", "#1960"]),
        # #299: the v0.82.0-v0.141.0 permission model and semantics.
        (KaguraClient.delete_context, ["#1960", "validation_error", "permission_denied"]),
        (KaguraClient.create_edge, ["permission_denied", "#2313"]),
        (KaguraClient.update_edge, ["permission_denied", "#2313"]),
        (
            KaguraClient.reference,
            ["``acceptance``", "``verified``", "``not_writable``", "``edge_exists``"],
        ),
        (KaguraClient.get_usage, ["#1946"]),
        (UsageInfo, ["#1946"]),
        (MemoryStatItem, ["access_count", "#1985"]),
        (
            WorkspaceClient,
            ["allowed_context_ids", "context_not_found", "/context-access", "v0.126.0"],
        ),
        (WorkspaceClient.add_member, ["create_invitation", "allowed_context_ids", "v0.126.0"]),
        (WorkspaceClient.mint_member_key, ["#1752"]),
        (FilesClient, ["workspace_id", "v0.126.0"]),
        (FilesClient.upload, ["existing_file", "RES-002", "#1693"]),
        (ResourceClient, ["#1863"]),
        (ResourceClient.list_tokens, ["#1863"]),
        (ResourceClient.update_token, ["KaguraNotFoundError", "#1863"]),
        (ResourceClient.revoke_token, ["KaguraNotFoundError", "#1863"]),
        (ResourceClient.create_token, ["max_resource_tokens", "#1919"]),
        (ResourceClient.setup_resource, ["resource_id_conflict", "#2284"]),
        (KaguraClient.setup_resource, ["resource_id_conflict", "#2284"]),
        (mcp_session_expired, ["``mcp-``", "``DELETE /mcp``", "#1740"]),
        (sanitize_server_detail, ["#1719"]),
        (refresh_access_token, ["invalid_grant", "#1686", "#1738"]),
        (revoke_token, ["client_id", "#1741"]),
        (
            KaguraClient.get_agent_bootstrap,
            [
                "``trigger``",
                "include_details",
                "max_chars",
                "20 since memory-cloud v0.82.0",
                "``instructions``",
                "``context.usage_guide``",
                "#1683",
            ],
        ),
        (
            AgentsClient.bootstrap,
            [
                "``trigger``",
                "include_details",
                "max_chars",
                "20 since memory-cloud v0.82.0",
                "``context.usage_guide``",
            ],
        ),
        (KaguraClient.get_server_info, ["search_defaults", "model_extra", "terms_version"]),
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
