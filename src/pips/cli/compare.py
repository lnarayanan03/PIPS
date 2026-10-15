"""Build results/comparison_summary.txt from the saved benchmark results (no API calls).

Usage:
    pips-compare
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from pips.paths import PROJECT_ROOT

# Run from the project root so .env, data/ and results/ resolve wherever this is launched from.
os.chdir(PROJECT_ROOT)


RESULT_FILES = [
    ("Manual keywords", Path("results/benchmark_results_manual.json")),
    ("LLM-generated keywords", Path("results/benchmark_results_llm.json")),
    ("Manual keywords + cross-encoder", Path("results/benchmark_results_cross.json")),
    ("LLM keywords + cross-encoder", Path("results/benchmark_results_llm_cross.json")),
]


def summarize(label: str, path: Path) -> dict | None:
    if not path.exists():
        return None

    rows = json.loads(path.read_text())
    n = len(rows)
    if n == 0:
        return {
            "label": label,
            "rows": 0,
            "avg_full": 0,
            "avg_sliced": 0,
            "reduction": 0,
            "routing": 0,
            "full_score": 0,
            "sliced_score": 0,
            "delta": 0,
        }

    full_score = sum(r["full"]["score"] for r in rows) / n
    sliced_score = sum(r["sliced"]["score"] for r in rows) / n
    return {
        "label": label,
        "rows": n,
        "avg_full": sum(r["full"]["tokens"] for r in rows) / n,
        "avg_sliced": sum(r["sliced"]["tokens"] for r in rows) / n,
        "reduction": sum(r["token_reduction_pct"] for r in rows) / n,
        "routing": sum(1 for r in rows if r["routing_correct"]) / n * 100,
        "full_score": full_score,
        "sliced_score": sliced_score,
        "delta": sliced_score - full_score,
    }


def bordered(lines: list[str], width: int = 118) -> str:
    out = []
    for line in lines:
        stripped = line.strip()
        if stripped and set(stripped) == {"═"}:
            body = "═" * width
        elif stripped and set(stripped) == {"─"}:
            body = "─" * width
        else:
            body = line.ljust(width)[:width]
        out.append(f"|{body}|")
    return "\n".join(out)


def main() -> None:
    summaries = [
        summary
        for label, path in RESULT_FILES
        if (summary := summarize(label, path)) is not None
    ]

    if not summaries:
        raise SystemExit("No benchmark result files found in results/.")

    lines = [
        "═" * 118,
        "     PAGEINDEX PROMPT SLICING — CONSOLIDATED COMPARISON",
        "═" * 118,
        "",
        f"  {'Mode':<34} {'Rows':<5} {'Full Tok':<10} {'Sliced Tok':<11} {'Reduc%':<8} {'Routing':<8} {'Full':<8} {'Sliced':<8} {'Delta'}",
        "─" * 118,
    ]
    for item in summaries:
        reduction = f"{item['reduction']:.1f}%"
        routing = f"{item['routing']:.0f}%"
        lines.append(
            f"  {item['label']:<34} "
            f"{item['rows']:<5} "
            f"{item['avg_full']:<10.0f} "
            f"{item['avg_sliced']:<11.0f} "
            f"{reduction:<8} "
            f"{routing:<8} "
            f"{item['full_score']:<7.1f} "
            f"{item['sliced_score']:<7.1f} "
            f"{item['delta']:+.1f}"
        )
    lines.extend([
        "─" * 118,
        "",
        "Notes:",
        "  Full Tok and Sliced Tok are exact Anthropic input token counts from API usage.",
        "  Routing is the percentage of queries where the expected section appeared in the selected sections.",
        "  Delta is sliced answer score minus full prompt answer score.",
        "═" * 118,
    ])

    text = bordered(lines)
    out_path = Path("results/comparison_summary.txt")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(text)
    print(text)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
