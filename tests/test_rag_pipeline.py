from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

import so_rag.indexer as indexer_module
from so_rag.config import Settings
from so_rag.hybrid_search import HybridSearchService, metadata_matches_stackoverflow_tags
from so_rag.ingestion import parse_posts_xml
from so_rag.indexer import ChromaIndexer
from so_rag.logging_setup import get_run_logger
from so_rag.models import Status
from so_rag.orchestrator import run_rag_pipeline
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


def test_metadata_matches_stackoverflow_tags() -> None:
    assert metadata_matches_stackoverflow_tags("<python><list>", ["python"])
    assert not metadata_matches_stackoverflow_tags("<java>", ["python"])


def test_hybrid_rrf_no_duplicate_post_ids(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    patch_embedder(monkeypatch)
    settings = build_settings(tmp_path)
    payloads = fixture_payloads("tests/fixtures/posts.xml")
    logger = get_run_logger("run-hybrid-dedup", settings.log_dir)
    ChromaIndexer(settings).index_payloads(payloads, "run-hybrid-dedup", logger)
    hybrid = HybridSearchService(settings)
    hits = hybrid.search("python sort list", top_k=20, logger=logger)
    assert hits
    assert len(hits) == len({h.id for h in hits})


def test_hybrid_python_tag_filter_excludes_java_post(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    patch_embedder(monkeypatch)
    settings = build_settings(tmp_path)
    payloads = fixture_payloads("tests/fixtures/posts.xml")
    logger = get_run_logger("run-tag-filter", settings.log_dir)
    ChromaIndexer(settings).index_payloads(payloads, "run-tag-filter", logger)
    hybrid = HybridSearchService(settings)
    hits = hybrid.search(
        "anything",
        rewritten_query="list sorting",
        tag_filters=["python"],
        top_k=20,
        logger=logger,
    )
    ids = {h.id for h in hits}
    assert 4 not in ids
    for h in hits:
        assert "<python>" in (h.tags or "").lower()


class _FakeCrossEncoder:
    def __init__(self, *_a, **_kw) -> None:
        pass

    def predict(self, pairs, **_kw):  # noqa: ANN001
        return [float(len(pairs) - i) for i in range(len(pairs))]


def _openai_client_happy_path() -> MagicMock:
    client = MagicMock()

    def _create(**kwargs):  # noqa: ANN001
        resp = MagicMock()
        if kwargs.get("response_format"):
            payload = {
                "technical_tags": ["python"],
                "rewritten_search_query": "python list sort",
            }
            resp.choices = [MagicMock(message=MagicMock(content=json.dumps(payload)))]
        else:
            resp.choices = [MagicMock(message=MagicMock(content="Use list.sort() or sorted()."))]
        return resp

    client.chat.completions.create = _create
    return client


def test_rag_pipeline_happy_path(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    patch_embedder(monkeypatch)
    monkeypatch.setattr("so_rag.reranker.CrossEncoder", _FakeCrossEncoder)
    settings = build_settings(tmp_path)
    settings.openai_api_key = "test-key"
    payloads = fixture_payloads("tests/fixtures/posts.xml")
    logger = get_run_logger("run-rag-happy", settings.log_dir)
    ChromaIndexer(settings).index_payloads(payloads, "run-rag-happy", logger)
    result = run_rag_pipeline(
        settings,
        "Python list sorting",
        "run-rag-happy",
        logger,
        openai_client=_openai_client_happy_path(),
    )
    assert "list" in result.answer.lower() or "sort" in result.answer.lower()
    assert result.tags_extracted == ["python"]
    assert result.sources
    assert result.top_1_score is not None
    assert "hybrid_ms" in result.latency_ms
    assert result.context_source_ids


def test_rag_pipeline_llm_answer_fallback(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    patch_embedder(monkeypatch)
    monkeypatch.setattr("so_rag.reranker.CrossEncoder", _FakeCrossEncoder)
    settings = build_settings(tmp_path)
    settings.openai_api_key = "test-key"
    payloads = fixture_payloads("tests/fixtures/posts.xml")
    logger = get_run_logger("run-rag-fallback", settings.log_dir)
    ChromaIndexer(settings).index_payloads(payloads, "run-rag-fallback", logger)

    client = MagicMock()

    def _create(**kwargs):  # noqa: ANN001
        if kwargs.get("response_format"):
            payload = {
                "technical_tags": ["python"],
                "rewritten_search_query": "python list sort",
            }
            r = MagicMock()
            r.choices = [MagicMock(message=MagicMock(content=json.dumps(payload)))]
            return r
        raise RuntimeError("LLM provider unavailable")

    client.chat.completions.create = _create

    result = run_rag_pipeline(
        settings,
        "Python list sorting",
        "run-rag-fallback",
        logger,
        openai_client=client,
    )
    assert "Warning" in result.answer or "warning" in result.answer.lower()
    assert "stackoverflow.com/questions" in result.answer
    assert result.llm_error is not None

