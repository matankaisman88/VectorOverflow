from __future__ import annotations

from so_rag.models import SearchResult


def format_full_answer(query: str, results: list[SearchResult]) -> str:
    lines = [f"Query: {query}", f"Retrieved {len(results)} posts", ""]
    for rank, item in enumerate(results, start=1):
        lines.extend(
            [
                f"{'=' * 72}",
                f"Result {rank}  |  score={item.score:.4f}  |  id={item.id}",
                f"Title: {item.title}",
                f"Tags:  {item.tags or '(none)'}",
                f"URL:   {item.stackoverflow_url}",
                "",
                "Body:",
                item.body or "(empty)",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"
