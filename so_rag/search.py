from __future__ import annotations

from so_rag.config import Settings
from so_rag.indexer import ChromaIndexer
from so_rag.models import SearchResult


class SearchService:
    def __init__(self, settings: Settings):
        self.indexer = ChromaIndexer(settings)
        self.embedding_service = self.indexer.embedding_service
        self.collection = self.indexer.collection

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        embedding = self.embedding_service.embed_texts([query])[0]
        result = self.collection.query(query_embeddings=[embedding], n_results=top_k)
        ids = result.get("ids", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        output: list[SearchResult] = []
        for idx, doc_id in enumerate(ids):
            metadata = metadatas[idx] if idx < len(metadatas) else {}
            score = distances[idx] if idx < len(distances) else 0.0
            question_id = int(metadata.get("id", doc_id))
            output.append(
                SearchResult(
                    id=question_id,
                    title=str(metadata.get("title", "")),
                    tags=str(metadata.get("tags", "")),
                    score=float(score),
                    stackoverflow_url=f"https://stackoverflow.com/questions/{question_id}",
                )
            )
        return output

