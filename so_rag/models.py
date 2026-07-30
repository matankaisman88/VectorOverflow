from __future__ import annotations

import hashlib
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def content_hash(title: str, body: str, tags: str) -> str:
    payload = f"{title}\n{body}\n{tags}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class Status(str, Enum):
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    FAILURE = "FAILURE"


class Post(BaseModel):
    id: int = Field(alias="Id")
    title: str = Field(alias="Title")
    body: str = Field(alias="Body")
    tags: str = Field(alias="Tags", default="")


class PostMetadata(BaseModel):
    id: int
    tags: str = ""
    source: str = "stackoverflow_2013"

    @property
    def stackoverflow_url(self) -> str:
        return f"https://stackoverflow.com/questions/{self.id}"


class DlqRecord(BaseModel):
    original_payload: dict[str, Any]
    error_reason: str
    stage_identity: str
    timestamp: datetime
    run_id: str


class SearchResult(BaseModel):
    id: int
    title: str
    tags: str
    score: float
    stackoverflow_url: str
    body: str = ""


class HybridSearchHit(BaseModel):
    """Single post after hybrid fusion (RRF); `score` is the fused RRF contribution sum."""

    id: int
    title: str
    tags: str
    document_text: str
    score: float
    rrf_vector_component: float | None = None
    rrf_lexical_component: float | None = None
    vector_rank: int | None = None
    bm25_rank: int | None = None
    stackoverflow_url: str


class RerankedSource(BaseModel):
    id: int
    title: str
    tags: str
    document_text: str
    rerank_score: float
    rrf_score: float
    stackoverflow_url: str


class QueryPreprocessResult(BaseModel):
    technical_tags: list[str]
    rewritten_search_query: str


class RAGPipelineResult(BaseModel):
    run_id: str
    answer: str
    sources: list[RerankedSource]
    tags_extracted: list[str]
    rewritten_query: str
    latency_ms: dict[str, float]
    top_1_score: float | None = None
    llm_error: str | None = None
    context_source_ids: list[int]


class IndexingSummary(BaseModel):
    run_id: str
    total_processed: int
    total_indexed: int
    total_rejected: int
    duration_seconds: float
    status: Status
    max_id_seen: int | None = None
    max_last_activity_date: datetime | None = None

