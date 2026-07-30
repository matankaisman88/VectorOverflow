from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from so_rag.config import Settings
from so_rag.hybrid_search import HybridSearchService
from so_rag.models import HybridSearchHit
from so_rag.reranker import CrossEncoderReranker


class GoldenQuery(BaseModel):
    query: str
    relevant_ids: list[int]


class EvalResult(BaseModel):
    query: str
    recall_at_k: float
    reciprocal_rank: float
    retrieved_ids: list[int]


def load_golden_queries(path: Path) -> list[GoldenQuery]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [GoldenQuery.model_validate(q) for q in data["queries"] if q.get("relevant_ids")]


def recall_at_k(retrieved_ids: list[int], relevant_ids: list[int], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hit = len(top_k & set(relevant_ids))
    return hit / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[int], relevant_ids: list[int]) -> float:
    relevant = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def evaluate(
    settings: Settings,
    golden_path: Path,
    *,
    k: int = 10,
    use_reranker: bool = False,
) -> list[EvalResult]:
    queries = load_golden_queries(golden_path)
    hybrid = HybridSearchService(settings)
    reranker = CrossEncoderReranker(settings) if use_reranker else None

    results: list[EvalResult] = []
    for gq in queries:
        hits: list[HybridSearchHit] = hybrid.search(gq.query, top_k=max(k, 30))
        if reranker is not None:
            reranked = reranker.rerank(gq.query, hits, top_k=k)
            retrieved_ids = [s.id for s in reranked]
        else:
            retrieved_ids = [h.id for h in hits][:k]
        results.append(
            EvalResult(
                query=gq.query,
                recall_at_k=recall_at_k(retrieved_ids, gq.relevant_ids, k),
                reciprocal_rank=reciprocal_rank(retrieved_ids, gq.relevant_ids),
                retrieved_ids=retrieved_ids,
            )
        )
    return results


def summarize(results: list[EvalResult]) -> dict[str, float]:
    if not results:
        return {"mean_recall_at_k": 0.0, "mrr": 0.0, "n_queries": 0}
    return {
        "mean_recall_at_k": sum(r.recall_at_k for r in results) / len(results),
        "mrr": sum(r.reciprocal_rank for r in results) / len(results),
        "n_queries": len(results),
    }
