from __future__ import annotations

import json
import logging
from typing import Any

from openai import OpenAI

from so_rag.config import Settings
from so_rag.models import QueryPreprocessResult, RerankedSource


PREPROCESS_SYSTEM = (
    "You extract Stack Overflow style technical tags and rewrite the user question into a short, "
    "keyword-rich search query for retrieval. Respond with JSON only: "
    '{"technical_tags": string[], "rewritten_search_query": string}. '
    "Tags must be lowercase single tokens like python, sql, javascript. "
    "Include at most 5 tags. If unsure, use an empty technical_tags array."
)

ANSWER_SYSTEM = (
    "Answer the question strictly based on the provided Stack Overflow context. "
    "If the information is missing, state that you do not know."
)


def preprocess_query(
    client: OpenAI,
    settings: Settings,
    user_query: str,
    *,
    run_id: str,
    logger: logging.Logger | logging.LoggerAdapter,
) -> QueryPreprocessResult:
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": PREPROCESS_SYSTEM},
            {"role": "user", "content": user_query},
        ],
        response_format={"type": "json_object"},
    )
    raw = (response.choices[0].message.content or "").strip()
    data: dict[str, Any] = json.loads(raw)
    tags = [str(t).strip().lower() for t in data.get("technical_tags", []) if str(t).strip()]
    rewritten = str(data.get("rewritten_search_query", user_query)).strip() or user_query
    result = QueryPreprocessResult(technical_tags=tags[:5], rewritten_search_query=rewritten)
    logger.info(
        "llm_preprocess_complete run_id=%s tags=%s rewritten=%s",
        run_id,
        result.technical_tags,
        result.rewritten_search_query,
    )
    return result


def build_answer_prompt(user_query: str, contexts: list[str]) -> str:
    blocks = "\n\n".join(f"[Excerpt {i+1}]\n{c}" for i, c in enumerate(contexts))
    return f"Question:\n{user_query}\n\nContext:\n{blocks}"


def generate_answer(
    client: OpenAI,
    settings: Settings,
    user_query: str,
    sources: list[RerankedSource],
    *,
    run_id: str,
    logger: logging.Logger | logging.LoggerAdapter,
) -> str:
    contexts = [s.document_text for s in sources]
    user_content = build_answer_prompt(user_query, contexts)
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": user_content},
        ],
    )
    text = (response.choices[0].message.content or "").strip()
    logger.info("llm_answer_complete run_id=%s source_ids=%s", run_id, [s.id for s in sources])
    return text
