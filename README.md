# VectorOverflow

Semantic search over Stack Overflow questions using local embeddings and Chroma.

## Features

- Index Stack Overflow question posts from a local SQL Server database (`StackOverflow2013` by default)
- Optional legacy XML ingestion mode
- Semantic query search with top-k results
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

Other settings:

- `EMBEDDING_BACKEND` (`sentence_transformers` default, or `openai`)
- `OPENAI_API_KEY` (required if `EMBEDDING_BACKEND=openai`)
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

## Tests

```bash
pytest -q
```
