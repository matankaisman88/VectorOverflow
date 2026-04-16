from __future__ import annotations

import json
import time
from typing import Any

from openai import OpenAI

from so_rag.config import Settings
from so_rag.generator import generate_answer, preprocess_query
from so_rag.hybrid_search import HybridSearchService
from so_rag.models import QueryPreprocessResult, RAGPipelineResult, RerankedSource
from so_rag.reranker import CrossEncoderReranker


def _fallback_preprocess(user_query: str) -> QueryPreprocessResult:
    return QueryPreprocessResult(technical_tags=[], rewritten_search_query=user_query.strip())


def _fallback_answer_message(sources: list[RerankedSource], warning: str) -> str:
    lines = [warning, "", "Top Stack Overflow links:"]
    for s in sources:
        lines.append(s.stackoverflow_url)
    return "\n".join(lines)


def run_rag_pipeline(
    settings: Settings,
    user_query: str,
    run_id: str,
    logger: Any,
    *,
    hybrid: HybridSearchService | None = None,
    reranker: CrossEncoderReranker | None = None,
    openai_client: OpenAI | None = None,
    hybrid_pool: int = 40,
    rerank_top_k: int = 10,
) -> RAGPipelineResult:
    latency_ms: dict[str, float] = {}
    tags: list[str] = []
    rewritten = user_query.strip()
    llm_error: str | None = None

    t_all = time.perf_counter()
    client = openai_client
    if client is None and settings.openai_api_key:
        client = OpenAI(api_key=settings.openai_api_key)

    t0 = time.perf_counter()
    preprocess_result: QueryPreprocessResult | None = None
    if client:
        try:
            preprocess_result = preprocess_query(
                client,
                settings,
                user_query,
                run_id=run_id,
                logger=logger,
            )
        except Exception as exc:  # noqa: BLE001
            llm_error = f"preprocess_failed: {exc}"
            logger.exception("llm_preprocess_failed run_id=%s", run_id)
            preprocess_result = _fallback_preprocess(user_query)
    else:
        preprocess_result = _fallback_preprocess(user_query)
        llm_error = llm_error or "missing_openai_api_key_for_preprocess"

    assert preprocess_result is not None
    tags = list(preprocess_result.technical_tags)
    rewritten = preprocess_result.rewritten_search_query
    latency_ms["preprocess_ms"] = round((time.perf_counter() - t0) * 1000, 3)

    hybrid_svc = hybrid or HybridSearchService(settings)
    t1 = time.perf_counter()
    hybrid_hits = hybrid_svc.search(
        user_query,
        rewritten_query=rewritten,
        tag_filters=tags,
        top_k=hybrid_pool,
        logger=logger,
    )
    latency_ms["hybrid_ms"] = round((time.perf_counter() - t1) * 1000, 3)

    ranker = reranker or CrossEncoderReranker(settings)
    t2 = time.perf_counter()
    reranked = ranker.rerank(rewritten or user_query, hybrid_hits, top_k=rerank_top_k)
    latency_ms["rerank_ms"] = round((time.perf_counter() - t2) * 1000, 3)

    top_1 = float(reranked[0].rerank_score) if reranked else None
    context_ids = [s.id for s in reranked]

    answer: str
    t3 = time.perf_counter()
    if client and reranked:
        try:
            answer = generate_answer(
                client,
                settings,
                user_query,
                reranked,
                run_id=run_id,
                logger=logger,
            )
        except Exception as exc:  # noqa: BLE001
            llm_error = f"answer_failed: {exc}"
            logger.exception("llm_answer_failed run_id=%s", run_id)
            answer = _fallback_answer_message(
                reranked,
                "Warning: the answer model failed; showing ranked links instead.",
            )
    elif reranked:
        answer = _fallback_answer_message(
            reranked,
            "Warning: OpenAI is not configured or unavailable; showing ranked links only.",
        )
    else:
        answer = "No relevant posts were retrieved for this query."
    latency_ms["generation_ms"] = round((time.perf_counter() - t3) * 1000, 3)
    latency_ms["total_ms"] = round((time.perf_counter() - t_all) * 1000, 3)

    log_payload = {
        "run_id": run_id,
        "latency_per_stage_ms": latency_ms,
        "tags_extracted": tags,
        "top_1_score": top_1,
        "context_source_ids": context_ids,
    }
    logger.info("rag_observability %s", json.dumps(log_payload, ensure_ascii=True))

    return RAGPipelineResult(
        run_id=run_id,
        answer=answer,
        sources=reranked,
        tags_extracted=tags,
        rewritten_query=rewritten,
        latency_ms=latency_ms,
        top_1_score=top_1,
        llm_error=llm_error,
        context_source_ids=context_ids,
    )
