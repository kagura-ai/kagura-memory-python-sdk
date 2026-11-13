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
from kagura_memory.client import MIN_SERVER_VERSION
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
    # #302: content_chunks_enabled / llm_selector_enabled need the context owner.
    "Requires owner or editor permission",
    # #314: memory-cloud v0.142.0 (#2508) puts acceptance on every recall row
    # with a supersede_candidate, not only on reference.
    "Recall rows do not carry",
    "on `reference` only, `acceptance",
    # #313: kagura auth refresh / token refresh under the lock since SDK v0.44.0.
    "`kagura auth refresh`/`token` do not",
    # #321: --merge-details is one details_patch call on memory-cloud v0.140.0+.
    "mention: it reads the memory first",
    # #314: the SDK is checked against memory-cloud v0.142.0.
    "Checked against memory-cloud up to **v0.141.0**",
    # #332: the SDK is checked against memory-cloud v0.155.0.
    "Checked against memory-cloud up to **v0.144.0**",
    # #340: the SDK is checked against memory-cloud v0.161.0.
    "Checked against memory-cloud up to **v0.155.0**",
    # #340: memory-cloud v0.158.0 adds the google / microsoft profiles.
    "The server knows only these two",
    "knows only `full` and `core`",
    "(full | core)",
    "``core`` and ``full``",
    "(``core`` or ``full``)",
    "server v0.73.0+: `full` or `core`)",
    # #338: memory-cloud v0.156.0 accepts batch-level keys on remember_batch items.
    "refused inside an item",
    # #338 review: 0.93.0-0.95.x ignore an item's batch-level key; only 0.96.0+ refuse it.
    "before v0.156.0 refuses every item",
    "before 0.156.0 refuses every item",
    "an older server refuses every such item",
    # #334: the plugin's server_url keeps the configured context= (plugin of v0.149.0).
    "`_plugin_server_url` keeps only `guardrails=off`",
]


def _doc_files() -> list[Path]:
    files = sorted((REPO_ROOT / "src" / "kagura_memory").rglob("*.py"))
    files += sorted((REPO_ROOT / "skills").rglob("SKILL.md"))
    files += sorted((REPO_ROOT / "examples").glob("*.py"))
    files += sorted((REPO_ROOT / "docs").rglob("*.md"))
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
                # #302: the event-time filter keys (memory-cloud v0.127.0).
                "``observed_after``",
                "``observed_before``",
                # #314: acceptance on recall rows from memory-cloud v0.142.0.
                "#2508",
                "v0.142.0",
            ],
        ),
        (KaguraClient.list_memories, ["``observed_at``", "#2330"]),
        # #302: owner-only toggles, not "owner or editor"; the LanceDB refusal.
        (
            KaguraClient.update_search_config,
            ["workspace owner/admin", "invalid_search_config", "#2015", "#2144"],
        ),
        (KaguraClient.recall_nearby, ["``location``", "include_details", "details_omitted"]),
        (KaguraClient.recall_upcoming, ["details_omitted", "details_total_chars"]),
        (KaguraClient.forget, ["CLEANUP_DELETED_MEMORIES_RETENTION_DAYS", "degraded"]),
        (
            KaguraClient.remember,
            [
                "persistence",
                "lint",
                "``verbose=true``",
                "missing_fields",
                # #302: memory-cloud v0.93.0-v0.127.0 write options.
                "duplicate_candidate",
                "tag_normalized",
                "#2330",
                # #314: the v0.142.0 plan headroom notice on a write.
                "``plan_headroom``",
                "#2615",
                "PlanHeadroomNotice",
            ],
        ),
        (
            KaguraClient.update_memory,
            [
                "persistence",
                "lint",
                '``""``',
                "``{}``",
                "#1803",
                # #302: partial edits, optimistic concurrency and the accept.
                "#2585",
                "KaguraVersionConflictError",
                "superseded_memory_id",
                "edge_exists",
                # #314: v0.142.0 plan headroom, recall-row acceptance, guardrail gate.
                "``plan_headroom``",
                "#2615",
                "#2508",
                "``not_writable``",
                "#2614",
            ],
        ),
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
            [
                "``acceptance``",
                "``verified``",
                "``not_writable``",
                "``edge_exists``",
                # #302: versions, continuation reads, and the REST-only flag.
                "#2043",
                "``content_next_offset``",
                "``record_access: false``",
                # #314: recall rows too, the guardrail gate, the author-only edge read.
                "#2508",
                "#2614",
                "``details.tool_trigger``",
            ],
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
        (
            KaguraClient.__init__,
            [
                "?profile=",
                "?tools=",
                "tools/list",
                "?profile=google",
                "unknown_tool",
                "?notices=off",
            ],
        ),
        (KaguraClient.describe_tools, ["?profile=google", "unknown_tool"]),
        (KaguraClient.guide, ["?profile=google", "unknown_tool"]),
        (KaguraClient.get_tool_definitions, ["?profile=google", "unknown_tool"]),
        (KaguraClient.bootstrap, ["notices", "v0.161.0", "?notices=off", "not instructions"]),
        (KaguraClient.remember_batch, ["v0.156.0", "batch_level_argument", "#2753"]),
        (FilesClient.download_url, ["SCAN-001", "SCAN-002", "retry_after", "v0.160.0"]),
        (FilesClient.upload, ["pending", "#565"]),
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


COMPAT_PAGE = REPO_ROOT / "docs" / "compatibility.md"
# README.md is the PyPI long description, where a relative link is dead (#323).
COMPAT_URL = "https://github.com/kagura-ai/kagura-memory-python-sdk/blob/main/docs/compatibility.md"
_MIN_LINE = "**Min memory-cloud:** "
_CEILING = re.compile(r"checked against memory-cloud up to \*\*([\d.]+)\*\*")


def _compat_sections() -> dict[str, str]:
    """docs/compatibility.md's ``## <SDK version>`` sections, in page order (#323)."""
    parts = re.split(r"^## (.+)$", COMPAT_PAGE.read_text(encoding="utf-8"), flags=re.M)
    return dict(zip(parts[1::2], parts[2::2], strict=True))


def _min_line(section: str) -> str:
    """A section's first paragraph, its ``**Min memory-cloud:**`` floor."""
    first = section.strip().split("\n\n", 1)[0]
    assert first.startswith(_MIN_LINE), f"the section does not open with its floor: {first[:60]!r}"
    return first


def _ceiling(section: str) -> str:
    """The ceiling the section's Min memory-cloud line names, never one from its notes."""
    found = _CEILING.search(_min_line(section))
    assert found, f"the Min memory-cloud line names no ceiling: {_min_line(section)[:80]!r}"
    return found.group(1)


def _readme_compat_section() -> str:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    return readme.split("## SDK ↔ memory-cloud Compatibility\n", 1)[1].split("\n## ", 1)[0]


def _readme_compat_rows() -> list[str]:
    return [line for line in _readme_compat_section().splitlines() if re.match(r"\| \[?\d", line)]


def _cells(row: str) -> list[str]:
    return [cell.strip() for cell in re.split(r"(?<!\\)\|", row.strip()[1:-1])]


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))


def _github_anchor(heading: str) -> str:
    """The id GitHub gives a heading: lowercased, punctuation dropped, spaces to ``-``."""
    return re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")


def test_compat_page_headings_are_versions_newest_first() -> None:
    """Each ``## `` heading is one SDK release (or range), once, newest first."""
    headings = re.findall(r"^## (.+)$", COMPAT_PAGE.read_text(encoding="utf-8"), flags=re.M)
    assert len(headings) == len(set(headings)), f"a release has two sections: {headings}"
    for heading in headings:
        assert re.fullmatch(r"\d+\.\d+\.(\d+|x)( – \d+\.\d+\.(\d+|x))?", heading), heading
    starts = [_version(heading.split()[0].replace("x", "0")) for heading in headings]
    assert starts == sorted(starts, reverse=True), f"sections out of order: {headings}"


def test_compat_sections_open_with_their_floor() -> None:
    """Each section's first paragraph is its ``**Min memory-cloud:**`` floor."""
    missing = [h for h, s in _compat_sections().items() if not s.strip().startswith(_MIN_LINE)]
    assert not missing, f"sections that do not open with their floor: {missing}"


@pytest.mark.parametrize("where", ["page", "README"])
def test_compat_docs_name_min_server_version(where: str) -> None:
    """The page intro and the README section give the client's ``MIN_SERVER_VERSION``."""
    if where == "page":
        text = COMPAT_PAGE.read_text(encoding="utf-8").split("\n## ", 1)[0]
    else:
        text = _readme_compat_section()
    named = re.search(r"`MIN_SERVER_VERSION`[^\n]*?\*\*(\d+\.\d+\.\d+)\*\*", text)
    assert named, f"the {where} does not give MIN_SERVER_VERSION"
    assert named.group(1) == MIN_SERVER_VERSION


def test_compat_0_44_section() -> None:
    """#314: the 0.44.0 section names its ceiling (memory-cloud v0.142.0) and its issues."""
    section = _compat_sections()["0.44.0"]
    assert _ceiling(section) == "0.142.0"
    issues = re.findall(r"kagura-memory-python-sdk/issues/(\d+)", section)
    for issue in ("301", "302", "309", "311", "313", "314", "315"):
        assert issue in issues, f"the 0.44.0 section does not link SDK #{issue}"
    for issue in ("2508", "2614", "2615"):
        assert f"memory-cloud/issues/{issue})" in section, f"the section lacks memory-cloud#{issue}"
    assert f"`MIN_SERVER_VERSION` stays **{MIN_SERVER_VERSION}**" in section


def test_compat_0_46_section() -> None:
    """#332: the 0.46.0 section names its ceiling (memory-cloud v0.155.0), issues and floors."""
    section = _compat_sections()["0.46.0"]
    assert _ceiling(section) == "0.155.0"
    floors = _min_line(section)
    for floor in (
        "0.154.0 `WorkspaceClient.list_operation_history`",
        "0.152.0",
        "0.153.0",
        "0.149.0",
    ):
        assert floor in floors, f"the Min memory-cloud line lacks {floor!r}"
    issues = re.findall(r"kagura-memory-python-sdk/issues/(\d+)", section)
    for issue in ("332", "333", "334", "335"):
        assert issue in issues, f"the 0.46.0 section does not link SDK #{issue}"
    for issue in ("2690", "2724", "2728", "2739"):
        assert f"memory-cloud/issues/{issue})" in section, f"the section lacks memory-cloud#{issue}"
    assert "38 REST operations" in section and "no backfill" in section
    assert f"`MIN_SERVER_VERSION` stays **{MIN_SERVER_VERSION}**" in section


def test_compat_0_47_section() -> None:
    """#340: the 0.47.0 section names its ceiling (memory-cloud v0.161.0), issues and floors."""
    section = _compat_sections()["0.47.0"]
    assert _ceiling(section) == "0.161.0"
    floors = _min_line(section)
    for floor in (
        "0.156.0 `dedupe` / `tags_normalize` / `verbose`",
        "0.158.0 the `?profile=google` / `?profile=microsoft` views",
        "0.160.0 `FileObject.scan_status`",
        "0.161.0 `BootstrapResponse.notices`",
    ):
        assert floor in floors, f"the Min memory-cloud line lacks {floor!r}"
    issues = re.findall(r"kagura-memory-python-sdk/issues/(\d+)", section)
    for issue in ("337", "338", "339", "340"):
        assert issue in issues, f"the 0.47.0 section does not link SDK #{issue}"
    for issue in ("565", "2751", "2752", "2753", "2762", "2773", "2778"):
        assert f"memory-cloud/issues/{issue})" in section, f"the section lacks memory-cloud#{issue}"
    for term in ("SCAN-001", "SCAN-002", "batch_level_argument", "?notices=off", "unknown_tool"):
        assert term in section, f"the section lacks {term!r}"
    assert "**Passed through with no SDK change.**" in section
    assert "**Auth.**" in section and "**Not wrapped.**" in section
    assert f"`MIN_SERVER_VERSION` stays **{MIN_SERVER_VERSION}**" in section


def test_readme_compat_rows_have_three_cells() -> None:
    """An unescaped ``|``, even inside a code span, splits a GFM table cell."""
    rows = _readme_compat_rows()
    assert rows, "the README compatibility section has no summary table"
    split = [row for row in rows if len(_cells(row)) != 3]
    assert not split, f"rows with an unescaped pipe: {split}"


def test_readme_compat_summary_matches_the_page() -> None:
    """#323: the README summarises the page's five newest sections, ceiling for ceiling."""
    sections = list(_compat_sections().items())
    rows = _readme_compat_rows()
    assert len(rows) == 5, f"the README summary lists the last five releases: {rows}"
    for row, (heading, section) in zip(rows, sections[:5], strict=True):
        sdk, ceiling, _ = _cells(row)
        assert sdk == f"[{heading}]({COMPAT_URL}#{_github_anchor(heading)})", row
        assert ceiling == _ceiling(section), (
            f"the page's {heading} section says {_ceiling(section)}: {row}"
        )
    newest, section = sections[0]
    sentence = f"SDK {newest} is checked against memory-cloud up to **{_ceiling(section)}**"
    assert sentence in _readme_compat_section()


def test_readme_links_the_docs_page_absolutely() -> None:
    """README.md is the PyPI long description: a relative ``docs/`` link is dead there."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "](docs/" not in readme
    assert f"]({COMPAT_URL})" in _readme_compat_section()


@pytest.mark.parametrize("name", ["CLAUDE.md", "AGENTS.md"])
def test_agent_guides_name_the_compat_ceiling(name: str) -> None:
    """CLAUDE.md / AGENTS.md name the ceiling of the page's newest section (#314: 0.142.0+)."""
    ceiling = _ceiling(next(iter(_compat_sections().values())))
    assert _version(ceiling) >= (0, 142, 0)
    text = (REPO_ROOT / name).read_text(encoding="utf-8")
    assert f"Checked against memory-cloud up to **v{ceiling}**" in text
    assert "docs/compatibility.md" in text
