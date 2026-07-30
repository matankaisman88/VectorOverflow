from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import chromadb
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from so_rag.config import Settings
from so_rag.ingestion import parse_posts_xml, valid_post_from_payload, write_dlq_record
from so_rag.models import IndexingSummary, Post, Status, content_hash


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

    @staticmethod
    def _update_watermark_trackers(
        payload: dict[str, Any],
        max_id_seen: int | None,
        max_last_activity_date: datetime | None,
    ) -> tuple[int | None, datetime | None]:
        post_id = payload.get("Id")
        if post_id is not None:
            post_id_int = int(post_id)
            max_id_seen = post_id_int if max_id_seen is None else max(max_id_seen, post_id_int)

        last_activity = payload.get("LastActivityDate")
        if last_activity is not None:
            if isinstance(last_activity, str):
                last_activity = datetime.fromisoformat(last_activity)
            if max_last_activity_date is None:
                max_last_activity_date = last_activity
            else:
                max_last_activity_date = max(max_last_activity_date, last_activity)

        return max_id_seen, max_last_activity_date

    def _process_batch(
        self,
        batch: list[Post],
        run_id: str,
        logger,
        indexed_so_far: int,
    ) -> tuple[int, int, Status, bool]:
        """Returns indexed_delta, rejected_delta, status_after_batch, should_stop."""
        if not batch:
            return 0, 0, Status.SUCCESS, False

        ids = [str(p.id) for p in batch]
        try:
            existing = self.collection.get(ids=ids, include=["metadatas"])
            existing_by_id = {
                eid: (meta or {}).get("content_hash")
                for eid, meta in zip(existing.get("ids", []), existing.get("metadatas", []))
            }

            posts_to_upsert: list[Post] = []
            for post in batch:
                new_hash = content_hash(post.title, post.body, post.tags)
                old_hash = existing_by_id.get(str(post.id))
                if str(post.id) not in existing_by_id or old_hash is None or old_hash != new_hash:
                    posts_to_upsert.append(post)

            if not posts_to_upsert:
                return 0, 0, Status.SUCCESS, False

            docs = [self._to_document(p) for p in posts_to_upsert]
            embeddings = self.embedding_service.embed_texts(docs)

            newly_indexed = sum(1 for p in posts_to_upsert if str(p.id) not in existing_by_id)
            re_embedded = len(posts_to_upsert) - newly_indexed
            logger.info("batch_result run_id=%s new=%s updated=%s", run_id, newly_indexed, re_embedded)

            self.collection.upsert(
                ids=[str(p.id) for p in posts_to_upsert],
                documents=docs,
                embeddings=embeddings,
                metadatas=[
                    {
                        "id": p.id,
                        "title": p.title,
                        "tags": p.tags,
                        "content_hash": content_hash(p.title, p.body, p.tags),
                    }
                    for p in posts_to_upsert
                ],
            )
            return len(posts_to_upsert), 0, Status.SUCCESS, False
        except Exception as exc:
            status = Status.PARTIAL if indexed_so_far > 0 else Status.FAILURE
            for post in batch:
                write_dlq_record(
                    self.settings.dlq_path,
                    post.model_dump(),
                    str(exc),
                    "embed",
                    run_id,
                )
            logger.exception("Failed processing embedding batch")
            return 0, len(batch), status, status == Status.FAILURE

    def index_payloads(self, payloads: Iterable[dict], run_id: str, logger) -> IndexingSummary:
        start = time.perf_counter()
        processed = 0
        indexed = 0
        rejected = 0
        status = Status.SUCCESS
        batch_buffer: list[Post] = []
        stop_processing = False
        max_id_seen: int | None = None
        max_last_activity_date: datetime | None = None

        for payload in payloads:
            processed += 1
            if processed % 1000 == 0:
                logger.info("indexing_progress run_id=%s processed=%s", run_id, processed)

            max_id_seen, max_last_activity_date = self._update_watermark_trackers(
                payload, max_id_seen, max_last_activity_date
            )
            post = valid_post_from_payload(payload, run_id, self.settings.dlq_path)
            if post is None:
                rejected += 1
                status = Status.PARTIAL
                continue
            batch_buffer.append(post)
            if len(batch_buffer) >= self.settings.batch_size:
                indexed_delta, rejected_delta, batch_status, should_stop = self._process_batch(
                    batch_buffer,
                    run_id,
                    logger,
                    indexed_so_far=indexed,
                )
                indexed += indexed_delta
                rejected += rejected_delta
                if batch_status != Status.SUCCESS:
                    status = batch_status
                batch_buffer.clear()
                if should_stop:
                    stop_processing = True
                    break

        if not stop_processing and batch_buffer:
            indexed_delta, rejected_delta, batch_status, should_stop = self._process_batch(
                batch_buffer,
                run_id,
                logger,
                indexed_so_far=indexed,
            )
            indexed += indexed_delta
            rejected += rejected_delta
            if batch_status != Status.SUCCESS:
                status = batch_status
            if should_stop:
                status = Status.FAILURE

        summary = IndexingSummary(
            run_id=run_id,
            total_processed=processed,
            total_indexed=indexed,
            total_rejected=rejected,
            duration_seconds=round(time.perf_counter() - start, 4),
            status=status,
            max_id_seen=max_id_seen,
            max_last_activity_date=max_last_activity_date,
        )
        logger.info("indexing_summary=%s", summary.model_dump_json())
        return summary

    def index_posts(self, input_xml: Path, run_id: str, logger) -> IndexingSummary:
        return self.index_payloads(parse_posts_xml(input_xml), run_id, logger)
