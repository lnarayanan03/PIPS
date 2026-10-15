"""
Run the PIPS benchmark: full prompt vs PageIndex-sliced prompt.

Usage:
    pips-benchmark --mode manual        # one mode (40 Anthropic API calls)
    pips-benchmark --mode all           # all 4 modes (160 API calls)
    pips-benchmark --help

Modes: manual, llm, cross, llm_cross, both (manual + llm), all.
--mode defaults to $BENCHMARK_INDEX_MODE, then "manual".

Outputs (in results/):
    benchmark_results_<mode>.json — raw per-query data
    report_<mode>.txt             — human-readable report
    comparison_summary.txt          — multi-mode summary when applicable
"""

import json
import argparse
import os
from pathlib import Path

from pips.paths import PROJECT_ROOT

# Run from the project root so .env, data/ and results/ resolve wherever this is launched from.
os.chdir(PROJECT_ROOT)

from pips.prompts.enterprise_assistant import (
    PROMPT_SECTIONS,
    FULL_SYSTEM_PROMPT,
    SLICED_BASE_PROMPT,
)
from pips.routing.keyword_tree import PageIndexTree
from pips.routing.llm_keyword_index import (
    apply_keywords,
    load_or_generate_llm_keywords,
)
from pips.routing.cross_encoder_reranker import CrossEncoderReranker
from pips.evaluation.benchmark import (
    MODEL_NAME,
    run_benchmark,
    generate_report,
    count_tokens,
)


def build_tree(sections):
    tree = PageIndexTree()
    tree.build(sections)
    return tree


def save_outputs(results, report: str, suffix: str | None = None) -> None:
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    if suffix:
        json_path = out_dir / f"benchmark_results_{suffix}.json"
        report_path = out_dir / f"report_{suffix}.txt"
    else:
        json_path = out_dir / "benchmark_results.json"
        report_path = out_dir / "report.txt"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    with open(report_path, "w") as f:
        f.write(report)

    print(f"\n✅  Saved: {json_path}")
    print(f"✅  Saved: {report_path}\n")


def output_paths(suffix: str | None = None, partial: bool = False):
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    stem = "benchmark_results"
    report_stem = "report"
    if suffix:
        stem = f"{stem}_{suffix}"
        report_stem = f"{report_stem}_{suffix}"
    if partial:
        stem = f"{stem}_partial"
        report_stem = f"{report_stem}_partial"

    return out_dir / f"{stem}.json", out_dir / f"{report_stem}.txt"


def save_checkpoint(results, suffix: str | None = None) -> None:
    json_path, report_path = output_paths(suffix, partial=True)

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    with open(report_path, "w") as f:
        f.write(generate_report(results))


def summarize_results(label: str, results) -> dict:
    n = len(results)
    if n == 0:
        return {
            "label": label,
            "queries": 0,
            "avg_full_tokens": 0,
            "avg_sliced_tokens": 0,
            "avg_reduction": 0,
            "routing_acc": 0,
            "sliced_score": 0,
        }

    return {
        "label": label,
        "queries": n,
        "avg_full_tokens": sum(r["full"]["tokens"] for r in results) / n,
        "avg_sliced_tokens": sum(r["sliced"]["tokens"] for r in results) / n,
        "avg_reduction": sum(r["token_reduction_pct"] for r in results) / n,
        "routing_acc": sum(1 for r in results if r["routing_correct"]) / n * 100,
        "sliced_score": sum(r["sliced"]["score"] for r in results) / n,
    }


def write_comparison_summary(summaries) -> None:
    if len(summaries) < 2:
        return

    sep = "═" * 104
    lines = [
        sep,
        "     PAGEINDEX MODE COMPARISON SUMMARY",
        sep,
        "",
        f"  {'Mode':<42} {'Rows':<6} {'Full Tok':<10} {'Sliced Tok':<11} {'Reduc%':<9} {'Routing':<9} {'Sliced Score'}",
        "─" * 104,
    ]
    for summary in summaries:
        reduction = f"{summary['avg_reduction']:.1f}%"
        routing = f"{summary['routing_acc']:.0f}%"
        lines.append(
            f"  {summary['label']:<42} "
            f"{summary['queries']:<6} "
            f"{summary['avg_full_tokens']:<10.0f} "
            f"{summary['avg_sliced_tokens']:<11.0f} "
            f"{reduction:<9} "
            f"{routing:<9} "
            f"{summary['sliced_score']:.1f}/10"
        )
    lines.extend([sep])
    text = "\n".join(f"|{line.ljust(104)[:104]}|" for line in lines)
    path = Path("results") / "comparison_summary.txt"
    path.parent.mkdir(exist_ok=True)
    path.write_text(text)
    print(text)
    print(f"\n✅  Saved: {path}\n")


def run_case(
    label: str,
    sections,
    full_prompt: str,
    suffix: str | None,
    reranker=None,
    candidate_k: int | None = None,
):
    print(f"\n🔧  Building PageIndex keyword tree ({label})...")
    tree = build_tree(sections)
    print(tree.describe())

    full_tok = count_tokens(full_prompt)
    total_keywords = sum(len(s["keywords"]) for s in sections.values())
    print(f"\n📊  Full system prompt size  : {full_tok} tokens")
    print(f"    Sections indexed         : {len(sections)}")
    print(f"    Total keyword nodes      : {total_keywords}")
    print(f"    Index source             : {label}")
    if reranker is not None:
        print(f"    Reranker                 : {reranker.model_name}")

    print("\n🚀  Running benchmark (10 queries × 2 runs each + judge calls)...")
    print("    This will make 40 API calls. Estimated time: 2-4 minutes.\n")

    latest_results = []
    results = []

    def checkpoint(rows):
        latest_results.clear()
        latest_results.extend(rows)
        save_checkpoint(rows, suffix)

    try:
        results = run_benchmark(
            full_prompt=full_prompt,
            tree=tree,
            sliced_base_prompt=SLICED_BASE_PROMPT,
            on_row=checkpoint,
            reranker=None if reranker is None else reranker.rerank,
            candidate_k=candidate_k,
        )
    except BaseException:
        print("\n⚠️  Benchmark interrupted before completion. Saving partial results...")
        save_checkpoint(latest_results, suffix)
        json_path, report_path = output_paths(suffix, partial=True)
        print(f"✅  Partial saved: {json_path}")
        print(f"✅  Partial saved: {report_path}\n")
        raise

    report = generate_report(results)
    print(report)
    save_outputs(results, report, suffix)
    return summarize_results(label, results)


MODES = ("manual", "llm", "cross", "llm_cross", "both", "all")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pips-benchmark",
        description="Run the PIPS benchmark. Every mode makes paid Anthropic API calls (40 per mode).",
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        default=os.environ.get("BENCHMARK_INDEX_MODE", "manual").strip().lower(),
        help="which routing mode(s) to benchmark (default: $BENCHMARK_INDEX_MODE or manual)",
    )
    parser.add_argument(
        "--rebuild-llm-index",
        action="store_true",
        default=os.environ.get("REBUILD_LLM_INDEX", "0") == "1",
        help="regenerate data/generated_keywords.json with Claude before running",
    )
    args = parser.parse_args()
    mode = args.mode
    rebuild_llm_index = args.rebuild_llm_index
    summaries = []

    if mode in {"manual", "both", "all"}:
        summaries.append(run_case(
            label="manual keywords",
            sections=PROMPT_SECTIONS,
            full_prompt=FULL_SYSTEM_PROMPT,
            suffix="manual",
        ))

    llm_sections = None
    if mode in {"llm", "llm_cross", "both", "all"}:
        print("\n🧠  Loading LLM-generated keyword index...")
        keyword_map = load_or_generate_llm_keywords(
            PROMPT_SECTIONS,
            model_name=MODEL_NAME,
            rebuild=rebuild_llm_index,
        )
        llm_sections = apply_keywords(PROMPT_SECTIONS, keyword_map)

    if mode in {"llm", "both", "all"}:
        summaries.append(run_case(
            label="LLM-generated keywords",
            sections=llm_sections,
            full_prompt=FULL_SYSTEM_PROMPT,
            suffix="llm",
        ))

    if mode in {"cross", "llm_cross", "all"}:
        print("\n🔁  Loading local cross-encoder reranker...")
        reranker = CrossEncoderReranker(
            os.environ.get(
                "CROSS_ENCODER_MODEL",
                "cross-encoder/ms-marco-MiniLM-L6-v2",
            )
        )

    if mode in {"cross", "all"}:
        summaries.append(run_case(
            label="manual keywords + cross-encoder rerank",
            sections=PROMPT_SECTIONS,
            full_prompt=FULL_SYSTEM_PROMPT,
            suffix="cross",
            reranker=reranker,
            candidate_k=int(os.environ.get("CROSS_ENCODER_CANDIDATE_K", "5")),
        ))

    if mode in {"llm_cross", "all"}:
        summaries.append(run_case(
            label="LLM keywords + cross-encoder rerank",
            sections=llm_sections,
            full_prompt=FULL_SYSTEM_PROMPT,
            suffix="llm_cross",
            reranker=reranker,
            candidate_k=int(os.environ.get("CROSS_ENCODER_CANDIDATE_K", "5")),
        ))

    write_comparison_summary(summaries)


if __name__ == "__main__":
    main()
