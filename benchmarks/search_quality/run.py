#!/usr/bin/env python3
"""Search quality benchmark — measures hybrid search precision across 35 categories.

Loads test data from fixtures/, seeds a context, runs queries, and compares
against a saved baseline to detect improvements and regressions.

Usage:
    uv run python benchmarks/search_quality/run.py              # run + compare baseline
    uv run python benchmarks/search_quality/run.py --fresh      # forget all + re-seed + run
    uv run python benchmarks/search_quality/run.py --update-baseline  # save current as baseline
    uv run python benchmarks/search_quality/run.py --cleanup    # delete benchmark context
    uv run python benchmarks/search_quality/run.py --search-mode keyword  # BM25 only
    uv run python benchmarks/search_quality/run.py --rerank     # enable reranking
    uv run python benchmarks/search_quality/run.py --epochs 5   # multi-epoch Hebbian learning
    uv run python benchmarks/search_quality/run.py --explore    # measure explore() precision
    uv run python benchmarks/search_quality/run.py --scale      # include additional_memories.json
"""

import asyncio
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from kagura_memory import KaguraClient
from kagura_memory.config import load_config

console = Console()

EMBEDDING_MODEL = "qwen3-embedding:8b"
CONTEXT_NAME = "bench-crossdomain-search-test"

FIXTURES_DIR = Path(__file__).parent / "fixtures"
BASELINE_PATH = Path(__file__).parent / "baseline.json"
REPORT_PATH = Path(__file__).parent / "report.md"


def load_fixtures(
    *, include_additional: bool = False,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Load target memories, noise memories, and queries from JSON fixtures."""
    with open(FIXTURES_DIR / "target_memories.json") as f:
        targets = json.load(f)
    with open(FIXTURES_DIR / "noise_memories.json") as f:
        noise = json.load(f)
    with open(FIXTURES_DIR / "queries.json") as f:
        queries = json.load(f)
    if include_additional:
        additional_path = FIXTURES_DIR / "additional_memories.json"
        if additional_path.exists():
            with open(additional_path) as f:
                noise = noise + json.load(f)
    return targets, noise, queries


def load_baseline() -> dict | None:
    """Load saved baseline, or None if not found."""
    if BASELINE_PATH.exists():
        with open(BASELINE_PATH) as f:
            return json.load(f)
    return None


def save_baseline(data: dict) -> None:
    """Save current results as the new baseline."""
    with open(BASELINE_PATH, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    console.print(f"[green]Baseline saved to {BASELINE_PATH}[/]")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class QueryResult:
    query: str
    target_idx: int | None
    category: str
    note: str
    top5: list[dict]  # [{summary, score, is_target, is_noise}]
    hit: bool
    hit_rank: int | None
    latency_ms: float
    target_score: float | None
    top1_is_correct: bool


@dataclass
class CategoryStats:
    name: str
    total: int = 0
    top1_correct: int = 0
    top3_hit: int = 0
    top5_hit: int = 0
    avg_target_score: float = 0.0
    avg_latency: float = 0.0
    results: list[QueryResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _match_summary(result_summary: str, memory_summary: str) -> bool:
    """Check if a result matches an expected memory."""
    return memory_summary[:30] in result_summary or result_summary[:30] in memory_summary


def _build_category_stats(
    all_results: list[QueryResult],
) -> dict[str, CategoryStats]:
    categories: dict[str, CategoryStats] = {}
    for qr in all_results:
        if qr.category not in categories:
            categories[qr.category] = CategoryStats(name=qr.category)
        cs = categories[qr.category]
        cs.total += 1
        cs.results.append(qr)
        if qr.top1_is_correct:
            cs.top1_correct += 1
        if qr.hit and qr.hit_rank and qr.hit_rank <= 3:
            cs.top3_hit += 1
        if qr.hit:
            cs.top5_hit += 1
        cs.avg_latency += qr.latency_ms
        if qr.target_score is not None:
            cs.avg_target_score += qr.target_score
    return categories


def _build_results_snapshot(
    all_results: list[QueryResult],
    categories: dict[str, CategoryStats],
) -> dict:
    """Build a JSON-serializable snapshot for baseline comparison."""
    total_scored = sum(1 for r in all_results if r.target_idx is not None)
    total_p1 = sum(1 for r in all_results if r.top1_is_correct)
    total_h3 = sum(
        1 for r in all_results if r.hit and r.hit_rank and r.hit_rank <= 3
    )
    total_h5 = sum(1 for r in all_results if r.hit)

    cat_data = {}
    for cat_name, cs in categories.items():
        scored = sum(1 for r in cs.results if r.target_idx is not None)
        cat_data[cat_name] = {
            "p_at_1": cs.top1_correct,
            "hit_at_3": cs.top3_hit,
            "hit_at_5": cs.top5_hit,
            "total": scored,
            "queries": cs.total,
        }

    known_failures = [
        r.query
        for r in all_results
        if r.target_idx is not None and not r.top1_is_correct
    ]

    return {
        "version": datetime.now(tz=UTC).strftime("%Y-%m-%d"),
        "embedding_model": EMBEDDING_MODEL,
        "total": {
            "p_at_1": total_p1,
            "hit_at_3": total_h3,
            "hit_at_5": total_h5,
            "queries": total_scored,
        },
        "categories": cat_data,
        "known_failures": known_failures,
    }


def _compare_baseline(current: dict, baseline: dict) -> None:
    """Print diff between current results and baseline."""
    console.rule("[bold]Baseline Comparison[/bold]")
    console.print(f"  Baseline: {baseline['version']}")
    console.print(f"  Current:  {current['version']}")
    console.print()

    # Total comparison
    bt, ct = baseline["total"], current["total"]
    for key, label in [
        ("p_at_1", "P@1"),
        ("hit_at_3", "Hit@3"),
        ("hit_at_5", "Hit@5"),
    ]:
        bv, cv = bt[key], ct[key]
        bq, cq = bt["queries"], ct["queries"]
        if cv > bv:
            console.print(
                f"  [green]▲ {label}: {bv}/{bq} → {cv}/{cq} (+{cv - bv})[/]"
            )
        elif cv < bv:
            console.print(
                f"  [red]▼ {label}: {bv}/{bq} → {cv}/{cq} ({cv - bv})[/]"
            )
        else:
            console.print(f"  [dim]= {label}: {cv}/{cq} (no change)[/]")

    console.print()

    # Category-level comparison
    all_cats = set(baseline.get("categories", {})) | set(
        current.get("categories", {})
    )
    changes = []
    for cat in sorted(all_cats):
        bc = baseline.get("categories", {}).get(cat, {})
        cc = current.get("categories", {}).get(cat, {})
        bp1, cp1 = bc.get("p_at_1", 0), cc.get("p_at_1", 0)
        btot, ctot = bc.get("total", 0), cc.get("total", 0)
        if cp1 != bp1:
            changes.append((cat, bp1, btot, cp1, ctot))

    if changes:
        for cat, bp1, btot, cp1, ctot in changes:
            if cp1 > bp1:
                console.print(
                    f"  [green][IMPROVED] {cat}: {bp1}/{btot} → {cp1}/{ctot}[/]"
                )
            else:
                console.print(
                    f"  [red][REGRESSED] {cat}: {bp1}/{btot} → {cp1}/{ctot}[/]"
                )
    else:
        console.print("  [dim]No category-level changes[/]")

    # New passes / new failures
    bf = set(baseline.get("known_failures", []))
    cf = set(current.get("known_failures", []))
    new_passes = bf - cf
    new_failures = cf - bf
    if new_passes:
        console.print()
        for q in sorted(new_passes):
            console.print(f"  [green][NEW PASS] {q}[/]")
    if new_failures:
        console.print()
        for q in sorted(new_failures):
            console.print(f"  [red][NEW FAIL] {q}[/]")


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def _print_category_table(
    all_results: list[QueryResult],
    categories: dict[str, CategoryStats],
) -> None:
    t = Table(title="Results by Category", show_lines=True)
    t.add_column("Category", style="bold")
    t.add_column("Queries", justify="right")
    t.add_column("P@1", justify="right")
    t.add_column("Hit@3", justify="right")
    t.add_column("Hit@5", justify="right")
    t.add_column("Avg Score", justify="right")
    t.add_column("Avg ms", justify="right")

    for cat_name, cs in categories.items():
        scored = sum(1 for r in cs.results if r.target_idx is not None)
        avg_score = cs.avg_target_score / max(scored, 1)
        avg_lat = cs.avg_latency / cs.total
        t.add_row(
            cat_name,
            str(cs.total),
            f"{cs.top1_correct}/{scored}" if scored else "-",
            f"{cs.top3_hit}/{scored}" if scored else "-",
            f"{cs.top5_hit}/{scored}" if scored else "-",
            f"{avg_score:.3f}" if scored else "-",
            f"{avg_lat:.0f}",
        )

    total_scored = sum(1 for r in all_results if r.target_idx is not None)
    total_p1 = sum(1 for r in all_results if r.top1_is_correct)
    total_h3 = sum(
        1 for r in all_results if r.hit and r.hit_rank and r.hit_rank <= 3
    )
    total_h5 = sum(1 for r in all_results if r.hit)
    t.add_row(
        "[bold]TOTAL[/]",
        str(len(all_results)),
        f"[bold]{total_p1}/{total_scored}[/]",
        f"[bold]{total_h3}/{total_scored}[/]",
        f"[bold]{total_h5}/{total_scored}[/]",
        "",
        "",
    )
    console.print(t)


def _print_failures(all_results: list[QueryResult]) -> None:
    failures = [
        r for r in all_results if r.target_idx is not None and not r.top1_is_correct
    ]
    if not failures:
        return
    console.rule("[bold red]Failures (Target not at Rank 1)[/bold red]")
    for f in failures:
        rank_str = f"rank {f.hit_rank}" if f.hit else "NOT FOUND"
        if f.top5:
            top_line = (
                f"Top result: {f.top5[0]['summary']} (score: {f.top5[0]['score']:.3f})"
                + (" [red]NOISE[/]" if f.top5[0].get("is_noise") else "")
            )
        else:
            top_line = "Top result: (no results returned)"
        console.print(
            Panel(
                f"[bold]{f.query}[/]\n"
                f"Category: {f.category} | {f.note}\n"
                f"Target: {rank_str} (score: {f.target_score or 0:.3f})\n"
                + top_line,
                title=f"[red]MISS[/] — {f.category}",
            )
        )


def _save_report(
    all_results: list[QueryResult],
    categories: dict[str, CategoryStats],
    target_memories: list[dict],
    noise_memories: list[dict],
    queries: list[dict],
) -> None:
    all_memories = target_memories + noise_memories
    total_scored = sum(1 for r in all_results if r.target_idx is not None)
    total_p1 = sum(1 for r in all_results if r.top1_is_correct)
    total_h3 = sum(
        1 for r in all_results if r.hit and r.hit_rank and r.hit_rank <= 3
    )
    total_h5 = sum(1 for r in all_results if r.hit)
    now = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M UTC")

    md_lines = [
        "# Japanese Search Quality Test Report",
        "",
        f"> Generated: {now}",
        f"> Embedding: {EMBEDDING_MODEL}",
        f"> Memories: {len(target_memories)} targets + {len(noise_memories)} noise = {len(all_memories)} total",
        f"> Queries: {len(queries)}",
        "",
        "## Summary",
        "",
        f"- **Precision@1**: {total_p1}/{total_scored} ({total_p1 / max(total_scored, 1):.0%})",
        f"- **Hit@3**: {total_h3}/{total_scored} ({total_h3 / max(total_scored, 1):.0%})",
        f"- **Hit@5**: {total_h5}/{total_scored} ({total_h5 / max(total_scored, 1):.0%})",
        "",
        "## Results by Category",
        "",
        "| Category | Queries | P@1 | Hit@3 | Hit@5 |",
        "|----------|---------|-----|-------|-------|",
    ]
    for cat_name, cs in categories.items():
        scored = sum(1 for r in cs.results if r.target_idx is not None)
        if scored:
            md_lines.append(
                f"| {cat_name} | {cs.total} | {cs.top1_correct}/{scored} | {cs.top3_hit}/{scored} | {cs.top5_hit}/{scored} |"
            )

    md_lines += ["", "## Detailed Results", ""]
    for qr in all_results:
        icon = "✓" if qr.top1_is_correct else "△" if qr.hit else "✗"
        md_lines.append(f"### {icon} `{qr.query}` [{qr.category}]")
        md_lines.append("")
        md_lines.append(f"{qr.note}")
        md_lines.append("")
        md_lines.append("| Rank | Score | Summary | Type |")
        md_lines.append("|------|-------|---------|------|")
        for i, t in enumerate(qr.top5):
            label = (
                "**TARGET**" if t["is_target"] else "NOISE" if t["is_noise"] else "other"
            )
            md_lines.append(
                f"| {i + 1} | {t['score']:.3f} | {t['summary']} | {label} |"
            )
        md_lines.append("")

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(md_lines))
    console.print(f"\n[green]Report saved to {REPORT_PATH}[/]")


# ---------------------------------------------------------------------------
# Graph stats & explore evaluation
# ---------------------------------------------------------------------------


def _get_graph_stats() -> dict[str, float]:
    """Query neural_memory_edges from PostgreSQL."""
    r = subprocess.run(
        [
            "docker", "exec", "kagura-postgres", "psql",
            "-U", "kagura", "-d", "kagura", "-t", "-A", "-c",
            "SELECT count(*), round(avg(weight)::numeric, 4), "
            "round(max(weight)::numeric, 4) FROM neural_memory_edges;",
        ],
        capture_output=True,
        text=True,
    )
    parts = r.stdout.strip().split("|")
    if len(parts) == 3 and parts[0].strip():
        return {
            "edges": int(parts[0]),
            "avg_weight": float(parts[1]),
            "max_weight": float(parts[2]),
        }
    return {"edges": 0, "avg_weight": 0.0, "max_weight": 0.0}


async def _run_explore_eval(
    client: KaguraClient,
    context_id: str,
    target_memories: list[dict],
    all_memories: list[dict],
) -> None:
    """Evaluate explore() precision vs recall() for target memories."""
    console.rule("[bold]Explore Evaluation[/bold]")

    # Build tag lookup
    all_tags: dict[str, set[str]] = {}
    for mem in all_memories:
        all_tags[mem["summary"][:30]] = set(mem["tags"])

    # Find target memory IDs via recall
    seed_ids: list[tuple[str, str, set[str]]] = []  # (id, summary, tags)
    for t in target_memories:
        r = await client.recall(context_id=context_id, query=t["summary"][:50], k=1)
        hits = r.get("results", [])
        if hits and _match_summary(hits[0]["summary"], t["summary"]):
            seed_ids.append((hits[0]["memory_id"], t["summary"][:40], set(t["tags"])))

    console.print(f"  Seeds: {len(seed_ids)}/{len(target_memories)} target memories found")

    explore_related = 0
    explore_total = 0
    explore_unique = 0
    recall_related = 0
    recall_total = 0

    for mid, summary, seed_tags in seed_ids:
        # Explore
        explored: list[dict] = []
        try:
            e = await client.explore(
                context_id=context_id, memory_id=mid, depth=2, min_weight=0.0,
            )
            explored = e.get("exploration", {}).get("related_memories", [])
        except Exception:
            pass

        for ex in explored[:5]:
            explore_total += 1
            ex_tags = all_tags.get(ex.get("summary", "")[:30], set())
            if seed_tags & ex_tags:
                explore_related += 1

        # Recall for comparison
        rc = await client.recall(context_id=context_id, query=summary, k=5)
        recall_ids_seen: set[str] = set()
        for h in rc.get("results", [])[:5]:
            if h["memory_id"] == mid:
                continue
            recall_total += 1
            recall_ids_seen.add(h["memory_id"])
            h_tags = all_tags.get(h.get("summary", "")[:30], set())
            if seed_tags & h_tags:
                recall_related += 1

        # Count explore-only hits as unique value (reuse explored)
        for ex in explored[:5]:
            ex_tags = all_tags.get(ex.get("summary", "")[:30], set())
            if seed_tags & ex_tags and ex["memory_id"] not in recall_ids_seen:
                explore_unique += 1

    e_prec = explore_related / max(explore_total, 1)
    r_prec = recall_related / max(recall_total, 1)

    console.print(f"  Explore P@5: {explore_related}/{explore_total} ({e_prec:.1%})")
    console.print(f"  Recall  P@5: {recall_related}/{recall_total} ({r_prec:.1%})")
    console.print(
        f"  Unique value: {explore_unique} "
        f"({explore_unique / max(len(seed_ids) * 5, 1):.1%})"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run_benchmark(
    *,
    fresh: bool = False,
    cleanup: bool = False,
    update_baseline: bool = False,
    search_mode: str = "hybrid",
    use_rerank: bool = False,
    epochs: int = 1,
    explore: bool = False,
    scale: bool = False,
) -> None:
    config = load_config()
    api_key = config.get("api_key", "")
    mcp_url = config.get("mcp_url", "")

    if not api_key or not mcp_url:
        console.print("[red]Error: .kagura.json required[/]")
        sys.exit(1)

    target_memories, noise_memories, queries = load_fixtures(
        include_additional=scale,
    )
    all_memories = target_memories + noise_memories

    async with KaguraClient(api_key=api_key, mcp_url=mcp_url) as client:
        existing = await client.list_contexts()
        ctx_index = {c["name"]: c["id"] for c in existing.get("contexts", [])}

        if cleanup:
            if CONTEXT_NAME in ctx_index:
                ctx_id = ctx_index[CONTEXT_NAME]
                await client.delete_context(context_id=ctx_id)
                console.print(
                    f"[green]Deleted context {CONTEXT_NAME} ({ctx_id[:8]}...)[/]"
                )
            else:
                console.print(f"[yellow]Context {CONTEXT_NAME} not found[/]")
            return

        if fresh and CONTEXT_NAME in ctx_index:
            ctx_id = ctx_index[CONTEXT_NAME]
            await client.forget(context_id=ctx_id, memory_id="all")
            console.print(f"[yellow]Cleared all memories in {CONTEXT_NAME}[/]")

        # Create or reuse context
        if CONTEXT_NAME in ctx_index and not fresh:
            context_id = ctx_index[CONTEXT_NAME]
            console.print(f"Context: {context_id[:8]}... (existing)")
        elif CONTEXT_NAME in ctx_index:
            context_id = ctx_index[CONTEXT_NAME]
            console.print(f"Context: {context_id[:8]}... (cleared)")
        else:
            ctx = await client.create_context(
                name=CONTEXT_NAME,
                display_name="[Bench] Japanese Search Quality",
                summary="Japanese NLP search quality benchmark with noise",
                is_private=True,
                embedding_model=EMBEDDING_MODEL,
            )
            context_id = ctx.get("context_id", "")
            console.print(f"Context: {context_id[:8]}... (created)")

        # Seed memories, skip already registered ones
        existing_summaries: set[str] = set()
        probe = await client.recall(
            context_id=context_id,
            query="データ 料理 IT 引越 走る 敬語 クラウド",
            k=100,
        )
        for r in probe.get("results", []):
            existing_summaries.add(r["summary"][:30])

        to_register = [
            (i, mem)
            for i, mem in enumerate(all_memories)
            if mem["summary"][:30] not in existing_summaries
        ]

        if not to_register:
            console.print(
                f"Memories: all {len(all_memories)} already populated"
            )
        else:
            existing_count = len(all_memories) - len(to_register)
            console.print(
                f"Registering {len(to_register)} memories ({existing_count} existing)..."
            )
            for i, mem in to_register:
                label = "TARGET" if i < len(target_memories) else "NOISE"
                await client.remember(
                    context_id=context_id,
                    summary=mem["summary"],
                    content=mem["content"],
                    type=mem["type"],
                    tags=mem["tags"],
                    importance=mem["importance"],
                )
                console.print(f"  [{label}] {mem['summary'][:50]}...")
            console.print("[green]Done. Waiting for indexing...[/]")
            await asyncio.sleep(2.0)

        # Run queries (multi-epoch if requested)
        noise_prefixes = {nm["summary"][:30] for nm in noise_memories}
        all_results: list[QueryResult] = []
        for epoch in range(1, epochs + 1):
            if epochs > 1:
                console.rule(f"[bold]Epoch {epoch}/{epochs}[/bold]")
            else:
                console.rule("[bold]Running Queries[/bold]")
            all_results.clear()

            for q in queries:
                t0 = time.monotonic()
                results = await client.recall(
                    context_id=context_id,
                    query=q["query"],
                    k=5,
                    search_mode=search_mode if search_mode != "hybrid" else None,
                    use_rerank=use_rerank,
                )
                latency = (time.monotonic() - t0) * 1000
                hits = results.get("results", [])

                top5 = []
                for r in hits:
                    rs = r.get("summary", "")
                    is_target = False
                    is_noise_match = False
                    if q["target_idx"] is not None:
                        is_target = _match_summary(
                            rs, target_memories[q["target_idx"]]["summary"]
                        )
                    is_noise_match = rs[:30] in noise_prefixes
                    top5.append(
                        {
                            "summary": rs[:70],
                            "score": r.get("score", 0.0),
                            "is_target": is_target,
                            "is_noise": is_noise_match,
                        }
                    )

                hit = False
                hit_rank = None
                target_score = None
                for i, t in enumerate(top5):
                    if t["is_target"]:
                        hit = True
                        hit_rank = i + 1
                        target_score = t["score"]
                        break

                qr = QueryResult(
                    query=q["query"],
                    target_idx=q["target_idx"],
                    category=q["category"],
                    note=q["note"],
                    top5=top5,
                    hit=hit,
                    hit_rank=hit_rank,
                    latency_ms=latency,
                    target_score=target_score,
                    top1_is_correct=hit_rank == 1 if hit else False,
                )
                all_results.append(qr)

                # Print live result (only on last epoch or single epoch)
                if epoch == epochs:
                    icon = (
                        "[green]✓[/]"
                        if qr.top1_is_correct
                        else "[yellow]△[/]" if hit else "[red]✗[/]"
                    )
                    console.print(f"\n{icon} [{q['category']:<12}] {q['query']}")
                    console.print(f"  [dim]{q['note']}[/]")
                    for i, t in enumerate(top5):
                        marker = ""
                        if t["is_target"]:
                            marker = " [green]◀ TARGET[/]"
                        elif t["is_noise"]:
                            marker = " [red]◀ NOISE[/]"
                        console.print(
                            f"  {i + 1}. [{t['score']:.3f}] {t['summary']}{marker}"
                        )

            # Epoch summary
            if epochs > 1:
                stats = _get_graph_stats()
                total_scored = sum(
                    1 for r in all_results if r.target_idx is not None
                )
                total_p1 = sum(1 for r in all_results if r.top1_is_correct)
                console.print(
                    f"  P@1={total_p1}/{total_scored} "
                    f"({total_p1 / max(total_scored, 1):.0%})  "
                    f"edges={stats['edges']} avg_w={stats['avg_weight']}"
                )

        # Category summary
        mode_parts = []
        if search_mode != "hybrid":
            mode_parts.append(f"search_mode={search_mode}")
        if use_rerank:
            mode_parts.append("rerank=on")
        mode_label = f" ({', '.join(mode_parts)})" if mode_parts else ""
        console.rule(f"[bold]Category Summary{mode_label}[/bold]")
        categories = _build_category_stats(all_results)
        _print_category_table(all_results, categories)
        _print_failures(all_results)

        # Baseline comparison
        snapshot = _build_results_snapshot(all_results, categories)
        baseline = load_baseline()
        if baseline:
            _compare_baseline(snapshot, baseline)
        else:
            console.print(
                "\n[yellow]No baseline found. Run with --update-baseline to create one.[/]"
            )

        if update_baseline:
            save_baseline(snapshot)

        # Save report
        _save_report(
            all_results, categories, target_memories, noise_memories, queries
        )

        # Explore evaluation
        if explore:
            await _run_explore_eval(
                client, context_id, target_memories, all_memories,
            )


if __name__ == "__main__":
    args = sys.argv[1:]
    mode = "hybrid"
    if "--search-mode" in args:
        idx = args.index("--search-mode")
        if idx + 1 < len(args):
            mode = args[idx + 1]
    num_epochs = 1
    if "--epochs" in args:
        idx = args.index("--epochs")
        if idx + 1 < len(args):
            num_epochs = int(args[idx + 1])
    asyncio.run(
        run_benchmark(
            fresh="--fresh" in args,
            cleanup="--cleanup" in args,
            update_baseline="--update-baseline" in args,
            search_mode=mode,
            use_rerank="--rerank" in args,
            epochs=num_epochs,
            explore="--explore" in args,
            scale="--scale" in args,
        )
    )
