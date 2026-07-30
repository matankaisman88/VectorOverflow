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
    batch_size: int = 100
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

