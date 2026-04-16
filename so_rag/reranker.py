from __future__ import annotations

from sentence_transformers import CrossEncoder

from so_rag.config import Settings
from so_rag.models import HybridSearchHit, RerankedSource


class CrossEncoderReranker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = CrossEncoder(settings.rerank_model)

    def rerank(
        self,
        query: str,
        hits: list[HybridSearchHit],
        *,
        top_k: int = 10,
    ) -> list[RerankedSource]:
        if not hits:
            return []
        pairs: list[tuple[str, str]] = [(query, h.document_text) for h in hits]
        scores = self._model.predict(pairs, show_progress_bar=False)
        combined: list[tuple[HybridSearchHit, float]] = list(zip(hits, scores, strict=True))
        combined.sort(key=lambda x: float(x[1]), reverse=True)
        out: list[RerankedSource] = []
        for hit, s in combined[:top_k]:
            out.append(
                RerankedSource(
                    id=hit.id,
                    title=hit.title,
                    tags=hit.tags,
                    document_text=hit.document_text,
                    rerank_score=float(s),
                    rrf_score=hit.score,
                    stackoverflow_url=hit.stackoverflow_url,
                )
            )
        return out
