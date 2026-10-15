from __future__ import annotations

from typing import List, Tuple


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, str, int]],
        top_k: int,
    ) -> List[Tuple[str, str, float]]:
        pairs = [(query, content) for _, content, _ in candidates]
        scores = self.model.predict(pairs)
        scored = [
            (section_id, content, float(score))
            for (section_id, content, _), score in zip(candidates, scores)
        ]
        scored.sort(key=lambda item: item[2], reverse=True)
        return scored[:top_k]
