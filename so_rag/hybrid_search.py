from __future__ import annotations

import logging
import re
from typing import Any

from rank_bm25 import BM25Okapi

from so_rag.config import Settings
from so_rag.indexer import ChromaIndexer
from so_rag.models import HybridSearchHit


def _tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"\W+", text.lower()) if t]


def metadata_matches_stackoverflow_tags(tags_str: str, required_tags: list[str]) -> bool:
    """Hard filter: each required tag must appear as `<tag>` in SO-style tag string."""
    if not required_tags:
        return True
    lowered = (tags_str or "").lower()
    for raw in required_tags:
        tag = raw.strip().lower()
        if not tag:
            continue
        if f"<{tag}>" not in lowered:
            return False
    return True


def reciprocal_rank_fusion_weighted(
    vector_ranked_ids: list[str],
    bm25_ranked_ids: list[str],
    *,
    k: int,
    alpha: float,
) -> dict[str, float]:
    """Weighted RRF: alpha weights the vector list, (1-alpha) the lexical list."""
    scores: dict[str, float] = {}
    for rank, doc_id in enumerate(vector_ranked_ids, start=1):
        contrib = alpha * (1.0 / (k + rank))
        scores[doc_id] = scores.get(doc_id, 0.0) + contrib
    for rank, doc_id in enumerate(bm25_ranked_ids, start=1):
        contrib = (1.0 - alpha) * (1.0 / (k + rank))
        scores[doc_id] = scores.get(doc_id, 0.0) + contrib
    return scores


def _vector_rrf_contribution(rank: int, k: int, alpha: float) -> float:
    return alpha * (1.0 / (k + rank))


def _bm25_rrf_contribution(rank: int, k: int, alpha: float) -> float:
    return (1.0 - alpha) * (1.0 / (k + rank))


class HybridSearchService:
    """
    Lexical BM25 over indexed documents, Chroma vector search, RRF fusion, optional tag filters.
    BM25 corpus is rebuilt from Chroma on first search and cached in-process (see BM25 persistence).
    """

    def __init__(self, settings: Settings, indexer: ChromaIndexer | None = None):
        self.settings = settings
        self.indexer = indexer or ChromaIndexer(settings)
        self.collection = self.indexer.collection
        self.embedding_service = self.indexer.embedding_service
        self._bm25: BM25Okapi | None = None
        self._ids: list[str] = []
        self._metadatas: list[dict[str, Any]] = []
        self._documents: list[str] = []
        self._id_to_index: dict[str, int] = {}

    def _ensure_corpus(self, logger: logging.Logger | logging.LoggerAdapter | None) -> None:
        if self._bm25 is not None:
            return
        data = self.collection.get(include=["documents", "metadatas"])
        self._ids = list(data.get("ids") or [])
        self._documents = list(data.get("documents") or [])
        self._metadatas = list(data.get("metadatas") or [])
        if len(self._ids) != len(self._documents) or len(self._ids) != len(self._metadatas):
            raise RuntimeError("Chroma get() returned mismatched ids/documents/metadatas lengths")
        self._id_to_index = {doc_id: idx for idx, doc_id in enumerate(self._ids)}
        if not self._ids:
            self._bm25 = None
            logging.getLogger(__name__).warning("hybrid_search_empty_collection")
            return
        tokenized = [_tokenize(doc) for doc in self._documents]
        self._bm25 = BM25Okapi(tokenized)
        msg = "hybrid_bm25_corpus_loaded count=%s"
        if logger:
            logger.info(msg, len(self._ids))
        else:
            logging.getLogger(__name__).info(msg, len(self._ids))

    def _allowed_ids(self, tag_filters: list[str]) -> set[str] | None:
        if not tag_filters:
            return None
        allowed: set[str] = set()
        for doc_id, meta in zip(self._ids, self._metadatas, strict=True):
            tags_field = str(meta.get("tags", ""))
            if metadata_matches_stackoverflow_tags(tags_field, tag_filters):
                allowed.add(doc_id)
        return allowed

    def _chroma_where_for_tags(self, tag_filters: list[str]) -> dict[str, Any] | None:
        if not tag_filters:
            return None
        clauses = [{"tags": {"$contains": f"<{t.strip().lower()}>"}} for t in tag_filters if t.strip()]
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def search(
        self,
        query: str,
        *,
        rewritten_query: str | None = None,
        tag_filters: list[str] | None = None,
        top_k: int = 30,
        logger: logging.Logger | logging.LoggerAdapter | None = None,
    ) -> list[HybridSearchHit]:
        self._ensure_corpus(logger)
        search_text = (rewritten_query or query).strip()
        if not search_text or not self._ids or self._bm25 is None:
            return []

        filters = [t.strip() for t in (tag_filters or []) if t.strip()]
        allowed = self._allowed_ids(filters)

        q_tokens = _tokenize(search_text)
        bm25_scores = self._bm25.get_scores(q_tokens)

        bm25_pairs: list[tuple[str, float]] = [
            (self._ids[i], float(bm25_scores[i])) for i in range(len(self._ids))
        ]
        bm25_pairs.sort(key=lambda x: x[1], reverse=True)
        if allowed is not None:
            bm25_pairs = [(i, s) for i, s in bm25_pairs if i in allowed]

        bm25_ranked_ids = [doc_id for doc_id, _ in bm25_pairs]

        embedding = self.embedding_service.embed_texts([search_text])[0]
        fetch_n = min(max(len(self._ids), top_k * 5), 5000) if self._ids else top_k
        where = self._chroma_where_for_tags(filters)

        try:
            vec = self.collection.query(
                query_embeddings=[embedding],
                n_results=min(fetch_n, max(1, len(self._ids))),
                where=where,
                include=["metadatas", "distances", "documents"],
            )
        except Exception:
            vec = self.collection.query(
                query_embeddings=[embedding],
                n_results=min(fetch_n, max(1, len(self._ids))),
                include=["metadatas", "distances", "documents"],
            )
            if allowed is not None:
                ids_list = vec.get("ids", [[]])[0]
                metas = vec.get("metadatas", [[]])[0]
                dists = vec.get("distances", [[]])[0]
                docs = vec.get("documents", [[]])[0]
                kept_ids: list[str] = []
                kept_meta: list[dict[str, Any]] = []
                kept_dist: list[float] = []
                kept_docs: list[str] = []
                for idx, doc_id in enumerate(ids_list):
                    meta = metas[idx] if idx < len(metas) else {}
                    if doc_id in allowed:
                        kept_ids.append(doc_id)
                        kept_meta.append(meta)
                        kept_dist.append(dists[idx] if idx < len(dists) else 0.0)
                        kept_docs.append(docs[idx] if idx < len(docs) else "")
                vec = {
                    "ids": [kept_ids],
                    "metadatas": [kept_meta],
                    "distances": [kept_dist],
                    "documents": [kept_docs],
                }

        vec_ids = vec.get("ids", [[]])[0]
        if allowed is not None:
            vec_ids = [i for i in vec_ids if i in allowed]

        k = self.settings.rrf_k
        alpha = float(self.settings.hybrid_alpha)
        fused = reciprocal_rank_fusion_weighted(vec_ids, bm25_ranked_ids, k=k, alpha=alpha)

        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]

        hits: list[HybridSearchHit] = []
        vec_rank_map = {doc_id: r for r, doc_id in enumerate(vec_ids, start=1)}
        bm25_rank_map = {doc_id: r for r, doc_id in enumerate(bm25_ranked_ids, start=1)}

        for doc_id, fused_score in ranked:
            idx = self._id_to_index.get(doc_id)
            if idx is None:
                continue
            meta = self._metadatas[idx]
            title = str(meta.get("title", ""))
            tags = str(meta.get("tags", ""))
            qid = int(meta.get("id", doc_id))
            vr = vec_rank_map.get(doc_id)
            br = bm25_rank_map.get(doc_id)
            v_comp = _vector_rrf_contribution(vr, k, alpha) if vr is not None else None
            b_comp = _bm25_rrf_contribution(br, k, alpha) if br is not None else None
            hits.append(
                HybridSearchHit(
                    id=qid,
                    title=title,
                    tags=tags,
                    document_text=self._documents[idx],
                    score=float(fused_score),
                    rrf_vector_component=v_comp,
                    rrf_lexical_component=b_comp,
                    vector_rank=vr,
                    bm25_rank=br,
                    stackoverflow_url=f"https://stackoverflow.com/questions/{qid}",
                )
            )

        return hits
