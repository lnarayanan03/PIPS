"""
PageIndex-style keyword tree.

Philosophy (matching PageIndex's design):
  - No vectors, no embeddings.
  - LLM (or pre-defined rules) builds a keyword→section graph at index time.
  - At query time: tokenise the query, walk the graph, retrieve matching nodes.
  - Scoring: exact phrase match > partial token overlap.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Dict, List, Tuple


# ──────────────────────────────────────────────
# DATA STRUCTURES
# ──────────────────────────────────────────────

@dataclass
class SectionNode:
    section_id: str
    keywords: List[str]
    content: str
    children: List["SectionNode"] = field(default_factory=list)

    # ── scoring ──────────────────────────────
    def match_score(self, query: str) -> int:
        """
        Keyword-graph scoring logic (mirrors PageIndex's node-traversal concept).

        Scoring tiers:
          +3  exact phrase found in query (multi-word keyword fully present)
          +2  every token in the keyword is present individually in the query
          +1  at least one token in the keyword is present in the query
        """
        query_lower = query.lower()
        query_tokens = set(re.findall(r"\b\w+\b", query_lower))

        score = 0
        for keyword in self.keywords:
            kw_lower = keyword.lower()
            kw_tokens = set(re.findall(r"\b\w+\b", kw_lower))

            if kw_lower in query_lower:
                score += 3                       # exact phrase hit
            elif kw_tokens and kw_tokens <= query_tokens:
                score += 2                       # all tokens present
            elif kw_tokens & query_tokens:
                score += 1                       # partial token overlap

        return score


# ──────────────────────────────────────────────
# TREE
# ──────────────────────────────────────────────

class PageIndexTree:
    """
    Flat tree (could be extended to hierarchical).
    Each leaf node = one prompt section.

    build()  → index all sections
    query()  → return top-k sections for a given query (scored)
    slice()  → return assembled sliced prompt string
    """

    def __init__(self) -> None:
        self.nodes: List[SectionNode] = []

    # ── indexing ─────────────────────────────
    def build(self, sections: Dict) -> None:
        """Build the keyword graph from sections dict."""
        self.nodes.clear()
        for section_id, data in sections.items():
            node = SectionNode(
                section_id=section_id,
                keywords=data["keywords"],
                content=data["content"],
            )
            self.nodes.append(node)

    # ── retrieval ────────────────────────────
    def query(
        self,
        user_query: str,
        top_k: int = 2,
        min_score: int = 1,
    ) -> List[Tuple[str, str, int]]:
        """
        Returns list of (section_id, content, score) sorted descending.
        Only sections with score >= min_score are included.
        """
        scored: List[Tuple[str, str, int]] = []
        for node in self.nodes:
            score = node.match_score(user_query)
            if score >= min_score:
                scored.append((node.section_id, node.content, score))

        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:top_k]

    # ── prompt assembly ──────────────────────
    def slice(
        self,
        user_query: str,
        base_prompt: str,
        top_k: int = 2,
        min_score: int = 1,
        reranker: Callable[[str, List[Tuple[str, str, int]], int], List[Tuple[str, str, float]]] | None = None,
        candidate_k: int | None = None,
    ) -> Tuple[str, List[str]]:
        """
        Returns (sliced_prompt_string, list_of_matched_section_ids).
        Falls back to top-2 sections by node order if nothing matches.
        """
        matched = self.query(
            user_query,
            top_k=candidate_k or top_k,
            min_score=min_score,
        )

        if matched:
            if reranker is not None:
                matched = reranker(user_query, matched, top_k)
            sections_content = "\n\n".join(content for _, content, _ in matched)
            matched_ids = [sid for sid, _, _ in matched]
        else:
            # Fallback — return first two nodes (safest default)
            fallback = self.nodes[:2]
            sections_content = "\n\n".join(n.content for n in fallback)
            matched_ids = [n.section_id for n in fallback]

        return base_prompt + sections_content, matched_ids

    # ── debug helpers ────────────────────────
    def describe(self) -> str:
        lines = [f"PageIndexTree ({len(self.nodes)} nodes)"]
        for n in self.nodes:
            lines.append(f"  [{n.section_id}]  keywords={len(n.keywords)}")
        return "\n".join(lines)
