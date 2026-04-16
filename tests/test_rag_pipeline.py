from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import so_rag.indexer as indexer_module
from so_rag.config import Settings
from so_rag.ingestion import parse_posts_xml
from so_rag.indexer import ChromaIndexer
from so_rag.logging_setup import get_run_logger
from so_rag.models import Status
from so_rag.search import SearchService


def build_settings(tmp_path: Path) -> Settings:
    return Settings(
        embedding_backend="sentence_transformers",
        chroma_persist_dir=tmp_path / "chroma",
        log_dir=tmp_path / "logs",
        dlq_path=tmp_path / "rejected_records.jsonl",
        sentence_transformer_model="all-MiniLM-L6-v2",
        batch_size=100,
    )


class FakeSentenceTransformer:
    def __init__(self, _model_name: str):
        pass

    def encode(self, texts, convert_to_numpy=True):  # noqa: ANN001
        vectors = []
        for text in texts:
            lower = text.lower()
            vectors.append(
                [
                    1.0 if "python" in lower else 0.0,
                    1.0 if "sort" in lower else 0.0,
                    1.0 if "pandas" in lower else 0.0,
                    1.0 if "sum" in lower else 0.0,
                ]
            )
        return np.array(vectors, dtype=np.float32) if convert_to_numpy else vectors


def patch_embedder(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(indexer_module, "SentenceTransformer", FakeSentenceTransformer)


def fixture_payloads(path: str) -> list[dict]:
    return list(parse_posts_xml(Path(path)))


def test_happy_path_10_posts_and_search(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    patch_embedder(monkeypatch)
    settings = build_settings(tmp_path)
    payloads = fixture_payloads("tests/fixtures/posts.xml")
    run_id = "run-happy"
    logger = get_run_logger(run_id, settings.log_dir)
    summary = ChromaIndexer(settings).index_payloads(payloads, run_id, logger)
    assert summary.total_processed == 10
    assert summary.total_indexed == 10
    assert summary.total_rejected == 0
    assert summary.status == Status.SUCCESS

    results = SearchService(settings).search("How can I sort a python list?", top_k=1)
    assert len(results) == 1
    assert results[0].id == 1
    assert results[0].stackoverflow_url.endswith("/1")


def test_deduplication_same_posts_twice(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    patch_embedder(monkeypatch)
    settings = build_settings(tmp_path)
    payloads = fixture_payloads("tests/fixtures/posts.xml")
    logger = get_run_logger("run-dedup", settings.log_dir)
    indexer = ChromaIndexer(settings)
    indexer.index_payloads(payloads, "run-dedup-1", logger)
    first_count = indexer.collection.count()
    indexer.index_payloads(payloads, "run-dedup-2", logger)
    second_count = indexer.collection.count()
    assert first_count == second_count


def test_validation_empty_body_to_dlq(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    patch_embedder(monkeypatch)
    settings = build_settings(tmp_path)
    payloads = fixture_payloads("tests/fixtures/posts_with_invalid.xml")
    logger = get_run_logger("run-invalid", settings.log_dir)
    summary = ChromaIndexer(settings).index_payloads(payloads, "run-invalid", logger)
    assert summary.total_processed == 2
    assert summary.total_indexed == 1
    assert summary.total_rejected >= 1
    assert summary.status == Status.PARTIAL
    assert settings.dlq_path.exists()
    lines = settings.dlq_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines
    record = json.loads(lines[0])
    assert record["stage_identity"] == "validate"


def test_search_metadata_contains_url_and_id(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    patch_embedder(monkeypatch)
    settings = build_settings(tmp_path)
    payloads = fixture_payloads("tests/fixtures/posts.xml")
    logger = get_run_logger("run-metadata", settings.log_dir)
    ChromaIndexer(settings).index_payloads(payloads, "run-metadata", logger)
    result = SearchService(settings).search("pandas sum", top_k=1)[0]
    assert isinstance(result.id, int)
    assert result.stackoverflow_url == f"https://stackoverflow.com/questions/{result.id}"

