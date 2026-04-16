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
- `OPENAI_API_KEY` (required if `EMBEDDING_BACKEND=openai`)
- `OPENAI_MODEL` (default: `gpt-4o`)
- `RERANK_MODEL` (default: `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- `RERANK_THRESHOLD` (default: `-2.0`; sources at or below this rerank score are dropped)
- `HYBRID_ALPHA` (default: `0.5`, vector-vs-lexical fusion weight)
- `RRF_K` (default: `60`)
- `CHROMA_PERSIST_DIR` (default: `./data/chroma`)

## Usage

### Index from DB (default)

```bash
python app.py index
```

Limit DB rows:

```bash
python app.py index --limit 1000
```

### Legacy XML indexing

```bash
python app.py index --from-xml tests/fixtures/posts.xml
```

### Search

```bash
python app.py search "How can I sort a python list?"
```

### Streamlit UI

```bash
streamlit run so_rag/app_ui.py
```

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
