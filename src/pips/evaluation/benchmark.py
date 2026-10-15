"""
Benchmark: Full Prompt vs PageIndex Sliced Prompt

Measures:
  1. Token count (full vs sliced)
  2. Answer quality (LLM-as-judge, 0-10)
  3. Section routing accuracy (did the tree pick the right section?)
  4. Latency

Output: per-query table + aggregate report saved to results/
"""

from __future__ import annotations
import json
import os
import time
from pathlib import Path
from collections.abc import Callable
from typing import Dict, List, Any

from anthropic import Anthropic


def load_env_file(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env_file()
MODEL_NAME = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
client = Anthropic()

# ──────────────────────────────────────────────
# TEST QUERIES — 10 queries, one per domain
# ──────────────────────────────────────────────

TEST_QUERIES: List[Dict[str, str]] = [
    {
        "id": "Q01",
        "query": "My invoice shows a wrong charge and I want a refund for the billing error.",
        "expected_section": "billing_payments",
        "ground_truth": (
            "Billing disputes must be raised within 60 days. "
            "Submit via support ticket or billing@company.com. "
            "Resolution in 5-7 business days."
        ),
    },
    {
        "id": "Q02",
        "query": "I am getting ERR_API_429 error when calling your REST API.",
        "expected_section": "technical_support",
        "ground_truth": (
            "ERR_API_429 means rate limit exceeded. "
            "Check your plan limits. Wait 60 seconds before retrying."
        ),
    },
    {
        "id": "Q03",
        "query": "How do I set up SSO with Okta for my enterprise account?",
        "expected_section": "account_management",
        "ground_truth": (
            "SSO uses SAML 2.0 and is available on Enterprise plan only. "
            "Okta is supported. SCIM provisioning is available."
        ),
    },
    {
        "id": "Q04",
        "query": "Does the Professional plan include Gantt chart and time tracking?",
        "expected_section": "product_features",
        "ground_truth": (
            "Yes. Professional plan includes Gantt chart view and time tracking."
        ),
    },
    {
        "id": "Q05",
        "query": "I want to cancel my subscription and get my money back.",
        "expected_section": "returns_refunds",
        "ground_truth": (
            "30-day money-back guarantee for new subscriptions. "
            "Cancel under Settings > Billing > Cancel Subscription."
        ),
    },
    {
        "id": "Q06",
        "query": "Is your platform HIPAA compliant? We store patient medical data.",
        "expected_section": "security_privacy",
        "ground_truth": (
            "HIPAA compliance module available as Enterprise add-on at $200/month."
        ),
    },
    {
        "id": "Q07",
        "query": "What is the parental leave policy and how many weeks are covered?",
        "expected_section": "hr_policies",
        "ground_truth": (
            "16 weeks fully paid parental leave for birth or adoption, all genders."
        ),
    },
    {
        "id": "Q08",
        "query": "We have 300 users. Is there a volume discount available for us?",
        "expected_section": "sales_pricing",
        "ground_truth": (
            "251-500 users qualify for 15% additional discount on Enterprise plan."
        ),
    },
    {
        "id": "Q09",
        "query": "How do I import my existing boards and projects from Trello?",
        "expected_section": "onboarding",
        "ground_truth": (
            "Use native Trello import via Settings > Import > Trello. Boards are preserved."
        ),
    },
    {
        "id": "Q10",
        "query": "Who owns the data I store in your platform? Does your company have any rights over it?",
        "expected_section": "legal_compliance",
        "ground_truth": (
            "Customer retains full ownership of all their data. "
            "Company has only a limited license for service operation."
        ),
    },
]


# ──────────────────────────────────────────────
# UTILITIES
# ──────────────────────────────────────────────

def count_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def parse_judge_json(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        raw = raw[start:end + 1]

    return json.loads(raw)


def normalize_judge_result(result: Dict[str, Any]) -> Dict[str, Any]:
    score = int(result.get("score", 0))
    score = max(0, min(10, score))
    correct = bool(result.get("correct", score >= 7))
    reasoning = str(result.get("reasoning", "Judge returned incomplete JSON."))
    return {"score": score, "correct": correct, "reasoning": reasoning}


def call_llm(system_prompt: str, user_query: str) -> tuple[str, float, int]:
    t0 = time.time()
    resp = client.messages.create(
        model=MODEL_NAME,
        max_tokens=400,
        system=system_prompt,
        messages=[{"role": "user", "content": user_query}],
    )
    latency = time.time() - t0
    answer = resp.content[0].text.strip()
    return answer, round(latency, 2), resp.usage.input_tokens


def judge_answer(query: str, answer: str, ground_truth: str) -> Dict[str, Any]:
    judge_sys = (
        "You are a strict accuracy evaluator. "
        "Given a user query, the key ground-truth facts, and an AI answer, "
        "determine if the answer correctly captures the ground-truth facts.\n\n"
        "Respond ONLY in valid JSON with exactly these keys:\n"
        '{"score": <int 0-10>, "correct": <true|false>, "reasoning": "<one sentence>"}\n'
        "score >= 7 should set correct=true. No markdown, no extra text."
    )
    judge_user = (
        f"Query: {query}\n"
        f"Ground Truth Facts: {ground_truth}\n"
        f"AI Answer: {answer}"
    )
    try:
        resp = client.messages.create(
            model=MODEL_NAME,
            max_tokens=150,
            system=judge_sys,
            messages=[{"role": "user", "content": judge_user}],
        )
        raw = resp.content[0].text.strip()
        return normalize_judge_result(parse_judge_json(raw))
    except Exception as e:
        return {"score": 0, "correct": False, "reasoning": f"Judge error: {e}"}


# ──────────────────────────────────────────────
# CORE BENCHMARK LOOP
# ──────────────────────────────────────────────

def run_benchmark(
    full_prompt: str,
    tree,
    sliced_base_prompt: str,
    on_row: Callable[[List[Dict[str, Any]]], None] | None = None,
    reranker: Callable[[str, List[tuple[str, str, int]], int], List[tuple[str, str, float]]] | None = None,
    candidate_k: int | None = None,
) -> List[Dict[str, Any]]:
    results = []

    print(f"\n{'─'*82}")
    print(f"  {'ID':<5} {'Full InTok':<11} {'Sliced InTok':<13} {'Reduction':<10} {'Routing':<9} {'Full Acc':<10} {'Sliced Acc'}")
    print(f"{'─'*82}")

    for test in TEST_QUERIES:
        qid = test["id"]
        query = test["query"]
        gt = test["ground_truth"]

        # ── FULL PROMPT RUN ──────────────────
        full_answer, full_latency, full_tokens = call_llm(full_prompt, query)
        full_judge = judge_answer(query, full_answer, gt)

        # ── SLICED PROMPT RUN ────────────────
        sliced_prompt, matched_ids = tree.slice(
            query,
            sliced_base_prompt,
            top_k=2,
            reranker=reranker,
            candidate_k=candidate_k,
        )
        sliced_answer, sliced_latency, sliced_tokens = call_llm(sliced_prompt, query)
        sliced_judge = judge_answer(query, sliced_answer, gt)

        # ── ROUTING ACCURACY ─────────────────
        routing_ok = test["expected_section"] in matched_ids
        reduction_pct = round((1 - sliced_tokens / full_tokens) * 100, 1)

        row: Dict[str, Any] = {
            "id": qid,
            "query": query,
            "expected_section": test["expected_section"],
            "matched_sections": matched_ids,
            "routing_correct": routing_ok,
            "token_reduction_pct": reduction_pct,
            "full": {
                "tokens": full_tokens,
                "latency_s": full_latency,
                "answer": full_answer,
                "score": full_judge["score"],
                "correct": full_judge["correct"],
                "reasoning": full_judge["reasoning"],
            },
            "sliced": {
                "tokens": sliced_tokens,
                "latency_s": sliced_latency,
                "answer": sliced_answer,
                "score": sliced_judge["score"],
                "correct": sliced_judge["correct"],
                "reasoning": sliced_judge["reasoning"],
            },
        }
        results.append(row)
        if on_row is not None:
            on_row(results)

        routing_sym = "✓" if routing_ok else "✗"
        reduction_str = f"{reduction_pct:.1f}%"
        print(
            f"  {qid:<5} {full_tokens:<10} {sliced_tokens:<12} "
            f"{reduction_str:<10}  {routing_sym:<9} "
            f"{full_judge['score']}/10      {sliced_judge['score']}/10"
        )

    print(f"{'─'*82}\n")
    return results


# ──────────────────────────────────────────────
# REPORT GENERATOR
# ──────────────────────────────────────────────

def format_bordered_report(lines: List[str], width: int = 104) -> str:
    formatted = []
    for line in lines:
        stripped = line.strip()
        if stripped and set(stripped) == {"═"}:
            body = "═" * width
        elif stripped and set(stripped) == {"─"}:
            body = "─" * width
        else:
            body = line.ljust(width)[:width]
        formatted.append(f"|{body}|")
    return "\n".join(formatted)


def generate_report(results: List[Dict[str, Any]]) -> str:
    n = len(results)
    if n == 0:
        sep = "═" * 104
        return format_bordered_report([
            sep,
            f"     PAGEINDEX PROMPT SLICING  —  BENCHMARK REPORT  (Anthropic {MODEL_NAME})",
            sep,
            "",
            "",
            "",
            "",
            "AGGREGATE SUMMARY",
            "─" * 40,
            "  Queries tested               : 0",
            "  No completed benchmark rows.",
            "",
            sep,
        ])

    avg_full_tok   = sum(r["full"]["tokens"]   for r in results) / n
    avg_sliced_tok = sum(r["sliced"]["tokens"] for r in results) / n
    avg_reduction  = sum(r["token_reduction_pct"] for r in results) / n
    routing_acc    = sum(1 for r in results if r["routing_correct"]) / n * 100
    full_score     = sum(r["full"]["score"]   for r in results) / n
    sliced_score   = sum(r["sliced"]["score"] for r in results) / n
    full_correct   = sum(1 for r in results if r["full"]["correct"])
    sliced_correct = sum(1 for r in results if r["sliced"]["correct"])
    delta          = sliced_score - full_score

    quality_verdict = (
        "MAINTAINED" if abs(delta) < 0.5
        else ("IMPROVED" if delta > 0 else "DEGRADED")
    )
    novel_verdict = (
        "✅  NOVEL APPROACH VALIDATED — significant token savings with quality maintained."
        if avg_reduction > 50 and quality_verdict in ("MAINTAINED", "IMPROVED")
        else "⚠️  MIXED RESULTS — review routing coverage or section keyword density."
    )

    sep = "═" * 104

    lines = [
        sep,
        f"     PAGEINDEX PROMPT SLICING  —  BENCHMARK REPORT  (Anthropic {MODEL_NAME})",
        sep,
        "",
        "",
        "",
        "",
        "AGGREGATE SUMMARY",
        "─" * 40,
        f"  Queries tested               : {n}",
        f"  Avg tokens (full prompt)     : {avg_full_tok:.0f}",
        f"  Avg tokens (sliced prompt)   : {avg_sliced_tok:.0f}",
        f"  Avg token reduction          : {avg_reduction:.1f}%",
        f"  Section routing accuracy     : {routing_acc:.0f}%",
        "",
        "ANSWER QUALITY",
        "─" * 40,
        f"  Full prompt avg score        : {full_score:.1f}/10  ({full_correct}/{n} correct)",
        f"  Sliced prompt avg score      : {sliced_score:.1f}/10  ({sliced_correct}/{n} correct)",
        f"  Quality delta                : {delta:+.1f}  → {quality_verdict}",
        "",
        "PER-QUERY DETAIL",
        "─" * 104,
        f"  {'ID':<5} {'Full Tok':<10} {'Slc Tok':<10} {'Reduc%':<9} {'Routing':<9} {'FScr':<7} {'SScr':<7} Matched Sections",
        "─" * 104,
    ]

    for r in results:
        routing_sym = "✓" if r["routing_correct"] else "✗"
        sections_str = ", ".join(r["matched_sections"])
        reduction_str = f"{r['token_reduction_pct']:.1f}%"
        lines.append(
            f"  {r['id']:<5} "
            f"{r['full']['tokens']:<10} "
            f"{r['sliced']['tokens']:<10} "
            f"{reduction_str:<9} "
            f"{routing_sym:<9} "
            f"{r['full']['score']}/10   "
            f"{r['sliced']['score']}/10   "
            f"{sections_str}"
        )

    lines += [
        "─" * 104,
        "VERDICT",
        "─" * 104,
        f"  {novel_verdict}",
        sep,
    ]

    return format_bordered_report(lines)
