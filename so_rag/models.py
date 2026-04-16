from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


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


class IndexingSummary(BaseModel):
    run_id: str
    total_processed: int
    total_indexed: int
    total_rejected: int
    duration_seconds: float
    status: Status

