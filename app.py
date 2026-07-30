from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import typer

from so_rag.config import Settings
from so_rag.ingestion import check_db_connection, parse_posts_from_db, parse_posts_xml
from so_rag.indexer import ChromaIndexer
from so_rag.logging_setup import get_run_logger
from so_rag.answer import format_full_answer
from so_rag.search import SearchService
from so_rag.watermark import load_watermark, reset_watermark, save_watermark

app = typer.Typer()


@app.command("db-check")
def db_check() -> None:
    """Verify SQL Server is reachable and StackOverflow2013 is available."""
    settings = Settings()
    try:
        result = check_db_connection(settings)
        typer.echo(json.dumps(result, indent=2))
    except Exception as exc:
        typer.echo(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        raise typer.Exit(code=1) from exc


def _index_db_batch(
    settings: Settings,
    indexer: ChromaIndexer,
    logger,
    run_id: str,
    *,
    limit: int | None,
    full: bool,
) -> tuple[int, int, bool]:
    """Index one DB batch. Returns (processed, indexed, had_rows)."""
    wm = None if full else load_watermark(settings.watermark_path)
    summary = indexer.index_payloads(
        parse_posts_from_db(settings, limit=limit, watermark=wm),
        run_id,
        logger,
    )
    if summary.total_processed == 0:
        return 0, 0, False
    if summary.max_id_seen is not None:
        save_watermark(settings.watermark_path, summary.max_id_seen, summary.max_last_activity_date)
    return summary.total_processed, summary.total_indexed, True


@app.command()
def index(
    from_xml: Path | None = None,
    limit: int | None = None,
    full: bool = typer.Option(False, help="Ignore watermark and start from the beginning."),
    index_all: bool = typer.Option(False, "--all", help="Index every post in batches until the DB is exhausted."),
) -> None:
    settings = Settings()
    indexer = ChromaIndexer(settings)

    if from_xml is not None:
        run_id = str(uuid4())
        logger = get_run_logger(run_id, settings.log_dir)
        summary = indexer.index_payloads(parse_posts_xml(from_xml), run_id, logger)
        typer.echo(summary.model_dump_json(indent=2))
        return

    if limit is not None and limit > settings.db_limit and not index_all:
        typer.echo(
            f"Error: single-run --limit {limit} exceeds DB_LIMIT ({settings.db_limit}). "
            f"Use: python app.py index --all --full --limit {settings.db_limit}",
            err=True,
        )
        raise typer.Exit(code=1)

    if limit is not None and limit > settings.db_limit and index_all:
        typer.echo(
            f"Warning: --limit {limit} is large; SQL Server may drop the connection. "
            f"Prefer `python app.py index --all --full` (batches of {settings.db_limit}).",
            err=True,
        )

    if full:
        reset_watermark(settings.watermark_path)

    batch_limit = limit if limit is not None else settings.db_limit

    if not index_all:
        run_id = str(uuid4())
        logger = get_run_logger(run_id, settings.log_dir)
        processed, indexed, had_rows = _index_db_batch(
            settings, indexer, logger, run_id, limit=limit, full=full
        )
        if not had_rows:
            typer.echo('{"status":"SUCCESS","total_processed":0,"total_indexed":0,"message":"No new posts."}')
            return
        wm = load_watermark(settings.watermark_path)
        typer.echo(
            json.dumps(
                {
                    "total_processed": processed,
                    "total_indexed": indexed,
                    "watermark_last_id": wm.last_id,
                },
                indent=2,
            )
        )
        return

    total_processed = 0
    total_indexed = 0
    batch_num = 0
    typer.echo(f"Indexing all posts in batches of {batch_limit}...")

    while True:
        batch_num += 1
        run_id = str(uuid4())
        logger = get_run_logger(run_id, settings.log_dir)
        processed, indexed, had_rows = _index_db_batch(
            settings,
            indexer,
            logger,
            run_id,
            limit=batch_limit,
            full=False,
        )
        if not had_rows:
            break

        total_processed += processed
        total_indexed += indexed
        wm = load_watermark(settings.watermark_path)
        typer.echo(
            f"Batch {batch_num}: processed={processed} indexed={indexed} "
            f"watermark_id={wm.last_id} (running total: {total_processed} processed, {total_indexed} indexed)"
        )

        if processed < batch_limit:
            break

    final_count = indexer.collection.count()
    typer.echo(
        json.dumps(
            {
                "status": "SUCCESS",
                "batches": batch_num,
                "total_processed": total_processed,
                "total_indexed": total_indexed,
                "chroma_count": final_count,
                "watermark_last_id": load_watermark(settings.watermark_path).last_id,
            },
            indent=2,
        )
    )


@app.command()
def search(
    query: str,
    top_k: int = 5,
    json_output: bool = typer.Option(False, "--json", help="Print one JSON object per result."),
    verbose: bool = typer.Option(False, help="Print debug progress messages."),
) -> None:
    if verbose:
        print(f"[search] starting query={query!r} top_k={top_k}")
    try:
        if verbose:
            print("[search] loading settings...")
        settings = Settings()
        if verbose:
            print("[search] creating SearchService...")
        service = SearchService(settings)
        if verbose:
            print("[search] running search...")
        results = list(service.search(query=query, top_k=top_k))
        if verbose:
            print(f"[search] done. results={len(results)}")

        if not results:
            typer.echo(f"No results for: {query!r}")
            return

        if json_output:
            for item in results:
                typer.echo(item.model_dump_json())
            return

        typer.echo(format_full_answer(query, results))
    except Exception as exc:
        if verbose:
            print(f"[search] ERROR: {exc!r}")
        raise


@app.command()
def ask(query: str, top_k: int = 3) -> None:
    """Retrieve and print full Stack Overflow post content for a query."""
    settings = Settings()
    service = SearchService(settings)
    results = service.search(query=query, top_k=top_k)
    if not results:
        typer.echo(f"No results for: {query!r}")
        raise typer.Exit(code=0)
    typer.echo(format_full_answer(query, results))


@app.command("eval-retrieval")
def eval_retrieval(
    golden_path: Path = Path("tests/eval/golden_queries.json"),
    k: int = 10,
    rerank: bool = False,
) -> None:
    from so_rag.eval import evaluate, summarize

    settings = Settings()
    results = evaluate(settings, golden_path, k=k, use_reranker=rerank)
    for r in results:
        typer.echo(r.model_dump_json())
    typer.echo(json.dumps(summarize(results), indent=2))


if __name__ == "__main__":
    app()
