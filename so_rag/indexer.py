from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import chromadb
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from so_rag.config import Settings
from so_rag.ingestion import parse_posts_xml, valid_post_from_payload, write_dlq_record
from so_rag.models import IndexingSummary, Post, Status


class EmbeddingService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._openai_client: OpenAI | None = None
        self._st_model: SentenceTransformer | None = None
        if settings.embedding_backend == "openai":
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required for openai backend")
            self._openai_client = OpenAI(api_key=settings.openai_api_key)
        else:
            self._st_model = SentenceTransformer(settings.sentence_transformer_model)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self.settings.embedding_backend == "openai":
            assert self._openai_client is not None
            response = self._openai_client.embeddings.create(
                model=self.settings.openai_embedding_model,
                input=texts,
            )
            return [item.embedding for item in response.data]
        assert self._st_model is not None
        embeddings = self._st_model.encode(texts, convert_to_numpy=True)
        return [vector.tolist() for vector in embeddings]


class ChromaIndexer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self.settings.chroma_persist_dir))
        self.collection = client.get_or_create_collection(name=self.settings.collection_name())
        self.embedding_service = EmbeddingService(settings)

    def _to_document(self, post: Post) -> str:
        return f"{post.title}\n{post.body}\n{post.tags}".strip()

    def _chunks(self, posts: list[Post], size: int) -> Iterable[list[Post]]:
        for i in range(0, len(posts), size):
            yield posts[i : i + size]

    def index_payloads(self, payloads: Iterable[dict], run_id: str, logger) -> IndexingSummary:
        start = time.perf_counter()
        processed = 0
        indexed = 0
        rejected = 0
        status = Status.SUCCESS
        valid_posts: list[Post] = []

        for payload in payloads:
            processed += 1
            post = valid_post_from_payload(payload, run_id, self.settings.dlq_path)
            if post is None:
                rejected += 1
                status = Status.PARTIAL
                continue
            valid_posts.append(post)

        for batch in self._chunks(valid_posts, self.settings.batch_size):
            ids = [str(p.id) for p in batch]
            try:
                existing = self.collection.get(ids=ids, include=[])
                existing_ids = set(existing.get("ids", []))
                missing_posts = [p for p in batch if str(p.id) not in existing_ids]
                if not missing_posts:
                    continue
                docs = [self._to_document(p) for p in missing_posts]
                embeddings = self.embedding_service.embed_texts(docs)
                self.collection.upsert(
                    ids=[str(p.id) for p in missing_posts],
                    documents=docs,
                    embeddings=embeddings,
                    metadatas=[{"id": p.id, "title": p.title, "tags": p.tags} for p in missing_posts],
                )
                indexed += len(missing_posts)
            except Exception as exc:
                status = Status.PARTIAL if indexed > 0 else Status.FAILURE
                for p in batch:
                    write_dlq_record(
                        self.settings.dlq_path,
                        p.model_dump(),
                        str(exc),
                        "embed",
                        run_id,
                    )
                    rejected += 1
                logger.exception("Failed processing embedding batch")
                if status == Status.FAILURE:
                    break

        summary = IndexingSummary(
            run_id=run_id,
            total_processed=processed,
            total_indexed=indexed,
            total_rejected=rejected,
            duration_seconds=round(time.perf_counter() - start, 4),
            status=status,
        )
        logger.info("indexing_summary=%s", summary.model_dump_json())
        return summary

    def index_posts(self, input_xml: Path, run_id: str, logger) -> IndexingSummary:
        return self.index_payloads(parse_posts_xml(input_xml), run_id, logger)

