from __future__ import annotations

from so_rag.config import Settings
from so_rag.indexer import ChromaIndexer
from so_rag.models import SearchResult


def _parse_document(document: str, fallback_title: str, fallback_tags: str) -> tuple[str, str]:
    """Split indexed document (title\\nbody\\ntags) back into body text."""
    if not document:
        return "", fallback_tags
    parts = document.split("\n")
    if len(parts) <= 1:
        return document, fallback_tags
    if len(parts) == 2:
        return parts[1], fallback_tags
    title = parts[0]
    tags = parts[-1]
    body = "\n".join(parts[1:-1])
    del title  # title comes from metadata; keep parser symmetric with indexer format
    if not tags:
        tags = fallback_tags
    return body, tags


class SearchService:
    def __init__(self, settings: Settings):
        self.indexer = ChromaIndexer(settings)
        self.embedding_service = self.indexer.embedding_service
        self.collection = self.indexer.collection

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        embedding = self.embedding_service.embed_texts([query])[0]
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["metadatas", "documents", "distances"],
        )
        ids = result.get("ids", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        documents = result.get("documents", [[]])[0]
        distances = result.get("distances", [[]])[0]
        output: list[SearchResult] = []
        for idx, doc_id in enumerate(ids):
            metadata = metadatas[idx] if idx < len(metadatas) else {}
            document = documents[idx] if idx < len(documents) else ""
            score = distances[idx] if idx < len(distances) else 0.0
            question_id = int(metadata.get("id", doc_id))
            title = str(metadata.get("title", ""))
            tags = str(metadata.get("tags", ""))
            body, tags = _parse_document(document, title, tags)
            output.append(
                SearchResult(
                    id=question_id,
                    title=title,
                    tags=tags,
                    score=float(score),
                    stackoverflow_url=f"https://stackoverflow.com/questions/{question_id}",
                    body=body,
                )
            )
        return output

