# VectorOverflow

Production-oriented Stack Overflow RAG system with hybrid retrieval, re-ranking, and a Streamlit UI.

## Features

- Index Stack Overflow question posts from a local SQL Server database (`StackOverflow2013` by default)
- Optional legacy XML ingestion mode
- Hybrid search: vector retrieval (Chroma) + lexical retrieval (BM25) with RRF fusion
- Cross-encoder re-ranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- LLM query preprocessing (tag extraction + query rewrite) and grounded answer generation
- Streamlit UI with source post and scoring details
- Local Chroma persistence

## Requirements

- Python 3.11+
- SQL Server reachable from this machine
- ODBC driver installed (default: `ODBC Driver 18 for SQL Server`)

## Install

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

## Configuration

Settings are loaded from environment variables or `.env` (via `pydantic-settings`).

```bash
copy .env.example .env
```

Then set `OPENAI_API_KEY` in `.env` for the Streamlit UI and `ask` command. Indexing works locally with the default `EMBEDDING_BACKEND=sentence_transformers`.

Common DB settings:

- `DB_SERVER` (default: `localhost`)
- `DB_NAME` (default: `StackOverflow2013`)
- `DB_USE_WINDOWS_AUTH` (default: `true`)
- `DB_USER` / `DB_PASSWORD` (required when Windows auth is disabled)
- `DB_DRIVER` (default: `ODBC Driver 18 for SQL Server`)
- `DB_QUERY` (default selects question posts from `Posts`)
- `DB_LIMIT` (default: `10000`)

RAG settings:

- `EMBEDDING_BACKEND` (`sentence_transformers` default, or `openai`)
- `OPENAI_API_KEY` (required for Streamlit UI / `ask`; also required if `EMBEDDING_BACKEND=openai`)
- `OPENAI_MODEL` (default: `gpt-4o`)
- `RERANK_MODEL` (default: `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- `RERANK_THRESHOLD` (default: `-2.0`; sources at or below this rerank score are dropped)
- `HYBRID_ALPHA` (default: `0.5`, vector-vs-lexical fusion weight)
- `RRF_K` (default: `60`)
- `CHROMA_PERSIST_DIR` (default: `./data/chroma`)
- `BATCH_SIZE` (default: `500`) — posts per Chroma `get()`/`upsert()` round-trip; raise to reduce Chroma I/O, but uses more RAM per batch.
- `EMBEDDING_ENCODE_BATCH_SIZE` (default: `64`) — texts per embedding backend call; caps OpenAI request size and sets sentence-transformers forward-pass chunk size independently of `BATCH_SIZE`.

## Usage

### Index from DB (default)

```bash
python app.py index
```

Limit DB rows:

```bash
python app.py index --limit 1000
```

Force a re-scan from the beginning with:

```bash
python app.py index --full
```

Index **all** question posts (runs in batches of `DB_LIMIT`, default 10,000, until done):

```bash
python app.py index --all --full
```

This can take a long time and use significant disk space for StackOverflow2013 (millions of posts).
Resume a partial `--all` run without `--full`:

```bash
python app.py index --all
```

If you see `Communication link failure` or `Login timeout expired`:

1. Run `python app.py db-check` — confirms SQL Server + `StackOverflow2013` are reachable.
2. Start **SQL Server (MSSQLSERVER)** in `services.msc` (the default instance hosts this DB).
3. Use `--all` instead of a very large single `--limit` (e.g. 50000).

### Legacy XML indexing

```bash
python app.py index --from-xml tests/fixtures/posts.xml
```

### Search

```bash
python app.py search "How can I sort a python list?"
python app.py search "how to" --top-k 101
python app.py search "python" --json
```

Use `--json` when you need machine-readable output (one JSON object per line).

### Retrieval evaluation

Fill `relevant_ids` in `tests/eval/golden_queries.json` with real indexed post IDs,
then run:

```bash
python app.py eval-retrieval
python app.py eval-retrieval --k 10 --rerank
```

### Streamlit UI

```bash
streamlit run app_ui.py
```

Run from the repo root (same directory as `app.py`). Do not use `so_rag/app_ui.py` directly — Streamlit adds that folder to `sys.path` and `import so_rag` fails.

The UI runs the full pipeline:
1. LLM preprocessing for technical tag extraction and query rewrite
2. Hybrid retrieval (vector + BM25 with metadata tag filtering)
3. Cross-encoder reranking of top candidates
4. LLM answer generation grounded in retrieved Stack Overflow context

If the LLM provider fails, the app falls back to returning top-ranked Stack Overflow links with a warning.

## Tests

```bash
pytest -q
```
