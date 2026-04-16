from __future__ import annotations

import logging
import re
from math import ceil
from typing import Any

from rank_bm25 import BM25Okapi

from so_rag.config import Settings
from so_rag.indexer import ChromaIndexer
from so_rag.models import HybridSearchHit


def _tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"\W+", text.lower()) if t]


def _normalize_tag(tag: str) -> str:
    value = tag.strip().lower()
    if not value:
        return value
    # Lightweight singularization to avoid strict plural mismatches.
    if value.endswith("ies") and len(value) > 3:
        return value[:-3] + "y"
    if value.endswith("sses") and len(value) > 4:
        return value[:-2]
    if value.endswith("s") and not value.endswith("ss") and len(value) > 3:
        return value[:-1]
    return value


def _tag_variants(tag: str) -> set[str]:
    raw = tag.strip().lower()
    normalized = _normalize_tag(raw)
    variants = {v for v in (raw, normalized) if v}
    return variants


def _extract_stackoverflow_tags(tags_str: str) -> set[str]:
    parsed = {m.strip().lower() for m in re.findall(r"<([^>]+)>", tags_str or "") if m.strip()}
    normalized = {_normalize_tag(t) for t in parsed if t}
    return parsed | normalized


def metadata_matches_stackoverflow_tags(tags_str: str, required_tags: list[str]) -> bool:
    """Soft filter: pass when at least one (or majority) requested tags match."""
    if not required_tags:
        return True
    post_tags = _extract_stackoverflow_tags(tags_str)
    required_sets = [_tag_variants(raw) for raw in required_tags if raw.strip()]
    if not required_sets:
        return True
    matched = sum(1 for variants in required_sets if variants & post_tags)
    needed = max(1, ceil(len(required_sets) / 2))
    return matched >= needed


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
        self._corpus_page_size: int = 1000

    def _iter_collection_rows(self) -> tuple[list[str], list[str], list[dict[str, Any]]]:
        """
        Read collection rows in pages to avoid SQLite variable limits on large corpora.
        """
        total = int(self.collection.count())
        if total <= 0:
            return [], [], []

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for offset in range(0, total, self._corpus_page_size):
            page = self.collection.get(
                limit=self._corpus_page_size,
                offset=offset,
                include=["documents", "metadatas"],
            )
            page_ids = list(page.get("ids") or [])
            page_docs = list(page.get("documents") or [])
            page_metas = list(page.get("metadatas") or [])
            if len(page_ids) != len(page_docs) or len(page_ids) != len(page_metas):
                raise RuntimeError("Chroma paged get() returned mismatched ids/documents/metadatas")
            ids.extend(page_ids)
            documents.extend(page_docs)
            metadatas.extend(page_metas)

        return ids, documents, metadatas

    def _ensure_corpus(self, logger: logging.Logger | logging.LoggerAdapter | None) -> None:
        if self._bm25 is not None:
            return
        self._ids, self._documents, self._metadatas = self._iter_collection_rows()
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
        clauses: list[dict[str, Any]] = []
        for tag in tag_filters:
            for variant in _tag_variants(tag):
                clauses.append({"tags": {"$contains": f"<{variant}>"}})
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$or": clauses}

    def search(
        self,
        query: str,
        *,
        rewritten_query: str | None = None,
        tag_filters: list[str] | None = None,
        top_k: int = 30,
        logger: logging.Logger | logging.LoggerAdapter | None = None,
    ) -> list[HybridSearchHit]:
        def _run_search(active_filters: list[str]) -> list[HybridSearchHit]:
            return self._search_impl(
                query=query,
                rewritten_query=rewritten_query,
                tag_filters=active_filters,
                top_k=top_k,
                logger=logger,
            )

        filters = [t.strip() for t in (tag_filters or []) if t.strip()]
        hits = _run_search(filters)
        if hits or not filters:
            return hits
        if logger:
            logger.info("hybrid_tag_filter_fallback_no_results tags=%s", filters)
        return _run_search([])

    def _search_impl(
        self,
        *,
        query: str,
        rewritten_query: str | None,
        tag_filters: list[str] | None,
        top_k: int,
        logger: logging.Logger | logging.LoggerAdapter | None,
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
