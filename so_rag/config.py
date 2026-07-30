from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    embedding_backend: Literal["openai", "sentence_transformers"] = "sentence_transformers"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_threshold: float = -2.0
    hybrid_alpha: float = 0.5
    rrf_k: int = 60
    sentence_transformer_model: str = "all-MiniLM-L6-v2"
    chroma_persist_dir: Path = Path("./data/chroma")
    chroma_collection_prefix: str = "posts"
    # Posts grouped per Chroma get()/upsert() round-trip. Larger = fewer round-trips,
    # more memory held at once (documents + embeddings for the whole batch).
    batch_size: int = 500
    # Texts grouped per embedding backend call. Independent from batch_size:
    # - sentence_transformers: passed as encode(..., batch_size=...) — controls the
    #   actual forward-pass chunk size on CPU/GPU. Raise this toward your GPU's
    #   memory ceiling for throughput; on CPU there's usually little benefit past 64.
    # - openai: caps how many texts go into a single embeddings.create() call, since
    #   the OpenAI embeddings API has request size/token limits. embed_texts() must
    #   internally split `texts` into chunks of this size regardless of how large
    #   the input list is.
    embedding_encode_batch_size: int = 64
    db_type: Literal["mssql"] = "mssql"
    db_server: str = "localhost"
    db_name: str = "StackOverflow2013"
    db_user: str | None = None
    db_password: str | None = None
    db_use_windows_auth: bool = True
    db_port: int = 1433
    db_driver: str = "ODBC Driver 18 for SQL Server"
    db_query: str = (
        # Custom DB_QUERY overrides must also select LastActivityDate for incremental mode,
        # otherwise ingestion falls back to ID-only watermarking.
        "SELECT Id, Title, Body, Tags, LastActivityDate "
        "FROM Posts "
        "WHERE PostTypeId = 1 "
        "ORDER BY Id"
    )
    db_limit: int = 10000
    db_connection_timeout: int = 60
    db_query_timeout: int = 600
    db_fetch_size: int = 500
    watermark_path: Path = Path("./data/watermark.json")
    log_dir: Path = Path("./logs")
    dlq_path: Path = Path("./rejected_records.jsonl")

    def collection_name(self) -> str:
        if self.embedding_backend == "openai":
            model = self.openai_embedding_model.replace("-", "_")
            return f"{self.chroma_collection_prefix}__openai_{model}"
        model = self.sentence_transformer_model.replace("-", "_")
        return f"{self.chroma_collection_prefix}__st_{model}"

