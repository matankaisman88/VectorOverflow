from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import typer

from so_rag.config import Settings
from so_rag.ingestion import parse_posts_from_db, parse_posts_xml
from so_rag.indexer import ChromaIndexer
from so_rag.logging_setup import get_run_logger
from so_rag.search import SearchService

app = typer.Typer()


@app.command()
def index(
    from_xml: Path | None = None,
    limit: int | None = None,
) -> None:
    settings = Settings()
    run_id = str(uuid4())
    logger = get_run_logger(run_id, settings.log_dir)
    indexer = ChromaIndexer(settings)
    if from_xml is not None:
        payloads = parse_posts_xml(from_xml)
    else:
        payloads = parse_posts_from_db(settings, limit=limit)
    summary = indexer.index_payloads(payloads, run_id, logger)
    typer.echo(summary.model_dump_json(indent=2))


@app.command()
def search(query: str, top_k: int = 5) -> None:
    print(f"[search] starting query={query!r} top_k={top_k}")
    try:
        print("[search] loading settings...")
        settings = Settings()
        print("[search] creating SearchService...")
        service = SearchService(settings)
        print("[search] running search...")
        results = list(service.search(query=query, top_k=top_k))
        print(f"[search] done. results={len(results)}")

        if not results:
            print("[search] no results found.")
            return

        for item in results:
            typer.echo(item.model_dump_json())
    except Exception as exc:
        print(f"[search] ERROR: {exc!r}")
        raise


if __name__ == "__main__":
    app()

