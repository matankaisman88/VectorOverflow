from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

import so_rag.indexer as indexer_module
from so_rag.config import Settings
from so_rag.indexer import ChromaIndexer, EmbeddingService
from so_rag.logging_setup import get_run_logger
from so_rag.models import Post


def test_openai_embed_texts_chunks_requests() -> None:
    settings = Settings(
        embedding_backend="openai",
        openai_api_key="test-key",
        embedding_encode_batch_size=3,
    )
    service = EmbeddingService(settings)
    call_inputs: list[list[str]] = []

    def fake_create(**kwargs):  # noqa: ANN001
        chunk = list(kwargs["input"])
        call_inputs.append(chunk)
        response = MagicMock()
        response.data = [MagicMock(embedding=[float(i), float(len(text))]) for i, text in enumerate(chunk)]
        return response

    service._openai_client = MagicMock()
    service._openai_client.embeddings.create = fake_create

    texts = [f"t{i}" for i in range(7)]
    embeddings = service.embed_texts(texts)

    assert call_inputs == [texts[0:3], texts[3:6], texts[6:7]]
    assert len(embeddings) == 7
    assert embeddings[0] == [0.0, float(len("t0"))]
    assert embeddings[3] == [0.0, float(len("t3"))]
    assert embeddings[6] == [0.0, float(len("t6"))]


def test_sentence_transformer_encode_uses_embedding_encode_batch_size(monkeypatch) -> None:  # noqa: ANN001
    captured: dict[str, int] = {}

    class TrackingSentenceTransformer:
        def __init__(self, _model_name: str):
            pass

        def encode(self, texts, convert_to_numpy=True, batch_size=32):  # noqa: ANN001
            captured["batch_size"] = batch_size
            return np.zeros((len(texts), 4), dtype=np.float32)

    monkeypatch.setattr(indexer_module, "SentenceTransformer", TrackingSentenceTransformer)

    settings = Settings(embedding_encode_batch_size=64)
    service = EmbeddingService(settings)
    service.embed_texts(["a", "b"])

    assert captured["batch_size"] == 64


def test_process_batch_one_chroma_round_trip_for_small_batch(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(
        indexer_module,
        "SentenceTransformer",
        lambda _model_name: MagicMock(
            encode=lambda texts, batch_size=64, convert_to_numpy=True: np.zeros(  # noqa: ANN001
                (len(texts), 4), dtype=np.float32
            )
        ),
    )

    settings = Settings(
        chroma_persist_dir=tmp_path / "chroma",
        log_dir=tmp_path / "logs",
        dlq_path=tmp_path / "rejected_records.jsonl",
        batch_size=500,
        embedding_encode_batch_size=64,
    )
    indexer = ChromaIndexer(settings)
    logger = get_run_logger("run-chroma-batch", settings.log_dir)

    get_calls = 0
    upsert_calls = 0
    original_get = indexer.collection.get
    original_upsert = indexer.collection.upsert

    def tracked_get(*args, **kwargs):  # noqa: ANN001
        nonlocal get_calls
        get_calls += 1
        return original_get(*args, **kwargs)

    def tracked_upsert(*args, **kwargs):  # noqa: ANN001
        nonlocal upsert_calls
        upsert_calls += 1
        return original_upsert(*args, **kwargs)

    indexer.collection.get = tracked_get
    indexer.collection.upsert = tracked_upsert

    posts = [
        Post.model_validate(
            {
                "Id": i,
                "Title": f"Title {i}",
                "Body": f"Body content for post {i}",
                "Tags": "<python>",
            }
        )
        for i in range(1, 6)
    ]

    indexed_delta, rejected_delta, status, should_stop = indexer._process_batch(
        posts,
        "run-chroma-batch",
        logger,
        indexed_so_far=0,
    )

    assert indexed_delta == 5
    assert rejected_delta == 0
    assert status.value == "SUCCESS"
    assert should_stop is False
    assert get_calls == 1
    assert upsert_calls == 1
