from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from lxml import etree, html

from so_rag.config import Settings
from so_rag.models import DlqRecord, Post
from so_rag.watermark import Watermark


def clean_body_html(raw_body: str) -> str:
    if not raw_body or not raw_body.strip():
        return ""
    doc = html.fromstring(raw_body or "")
    code_blocks: list[str] = []
    for node in doc.xpath("//pre|//code"):
        text = " ".join(node.itertext()).strip()
        if text:
            code_blocks.append(f"```{text}```")
        parent = node.getparent()
        if parent is not None:
            parent.remove(node)
    plain_text = " ".join(doc.itertext()).strip()
    parts = [plain_text] if plain_text else []
    parts.extend(code_blocks)
    return "\n".join(parts).strip()


def write_dlq_record(
    dlq_path: Path,
    original_payload: dict[str, Any],
    error_reason: str,
    stage_identity: str,
    run_id: str,
) -> None:
    dlq_path.parent.mkdir(parents=True, exist_ok=True)
    record = DlqRecord(
        original_payload=original_payload,
        error_reason=error_reason,
        stage_identity=stage_identity,
        timestamp=datetime.now(timezone.utc),
        run_id=run_id,
    )
    with dlq_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=True) + "\n")


def parse_posts_xml(input_path: Path) -> Iterator[dict[str, Any]]:
    try:
        context = etree.iterparse(str(input_path), events=("end",), tag="row")
    except Exception as exc:
        raise RuntimeError(f"Failed to open XML file: {input_path}") from exc
    for _, elem in context:
        yield dict(elem.attrib)
        elem.clear()


def _build_mssql_connection_string(settings: Settings) -> str:
    server = settings.db_server
    if settings.db_port != 1433:
        server = f"{server},{settings.db_port}"

    parts = [
        f"DRIVER={{{settings.db_driver}}}",
        f"SERVER={server}",
        f"DATABASE={settings.db_name}",
        f"Connection Timeout={settings.db_connection_timeout}",
        "Encrypt=yes",
        "TrustServerCertificate=yes",
    ]

    if settings.db_use_windows_auth:
        parts.append("Trusted_Connection=yes")
    else:
        if not settings.db_user or not settings.db_password:
            raise ValueError("DB_USER and DB_PASSWORD are required when windows auth is disabled")
        parts.extend([f"UID={settings.db_user}", f"PWD={settings.db_password}"])
    return ";".join(parts)


def _inject_incremental_predicate(base_query: str, watermark: Watermark) -> str:
    query = base_query.strip()
    if not query.upper().startswith("SELECT"):
        raise ValueError("DB_QUERY must start with SELECT")

    predicate = "(Id > ? OR LastActivityDate > ?)"
    upper = query.upper()
    order_by_idx = upper.rfind(" ORDER BY ")
    where_match = re.search(r"\bWHERE\b", query, re.IGNORECASE)

    if order_by_idx != -1:
        before_order = query[:order_by_idx].rstrip()
        after_order = query[order_by_idx:]
        if where_match and where_match.start() < order_by_idx:
            return f"{before_order} AND {predicate} {after_order}"
        return f"{before_order} WHERE {predicate} {after_order}"

    if where_match:
        return f"{query} AND {predicate}"

    return f"{query} WHERE {predicate}"


def _apply_top_limit(query: str, max_rows: int) -> str:
    if max_rows <= 0:
        return query
    upper = query.upper()
    if not upper.startswith("SELECT "):
        raise ValueError("DB_QUERY must start with SELECT")
    return query.replace("SELECT ", f"SELECT TOP {max_rows} ", 1)


def _connect_mssql(settings: Settings, pyodbc: Any, attempts: int = 3):  # noqa: ANN001
    last_exc: Exception | None = None
    conn_string = _build_mssql_connection_string(settings)
    for attempt in range(attempts):
        try:
            return pyodbc.connect(conn_string, timeout=settings.db_connection_timeout)
        except pyodbc.Error as exc:
            last_exc = exc
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
    assert last_exc is not None
    hint = (
        f"Could not connect to SQL Server at {settings.db_server!r} "
        f"(database={settings.db_name!r}). "
        "StackOverflow2013 is usually on the default instance (MSSQLSERVER). "
        "Open services.msc and start 'SQL Server (MSSQLSERVER)', then run: python app.py db-check"
    )
    raise RuntimeError(f"{last_exc}\n\n{hint}") from last_exc


def check_db_connection(settings: Settings) -> dict[str, Any]:
    import pyodbc  # type: ignore

    conn = _connect_mssql(settings, pyodbc, attempts=1)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT DB_NAME(), @@SERVERNAME")
        db_name, server_name = cursor.fetchone()
        cursor.execute("SELECT COUNT(*) FROM Posts WHERE PostTypeId = 1")
        question_posts = int(cursor.fetchone()[0])
        return {
            "ok": True,
            "server": str(server_name),
            "database": str(db_name),
            "question_posts": question_posts,
            "db_server_setting": settings.db_server,
        }
    finally:
        conn.close()


def parse_posts_from_db(
    settings: Settings,
    limit: int | None = None,
    *,
    watermark: Watermark | None = None,
) -> Iterator[dict[str, Any]]:
    try:
        import pyodbc  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("pyodbc is required for DB ingestion") from exc

    query = settings.db_query.strip()
    max_rows = limit if limit is not None else settings.db_limit
    params: tuple[Any, ...] | None = None

    if watermark is not None and watermark.last_id > 0:
        query = _inject_incremental_predicate(query, watermark)
        last_activity = watermark.last_activity_date or datetime(1900, 1, 1, tzinfo=timezone.utc)
        params = (watermark.last_id, last_activity)

    query = _apply_top_limit(query, max_rows)

    conn = _connect_mssql(settings, pyodbc)
    try:
        conn.timeout = settings.db_query_timeout
        cursor = conn.cursor()
        if params is not None:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]
        while True:
            rows = cursor.fetchmany(settings.db_fetch_size)
            if not rows:
                break
            for row in rows:
                payload = dict(zip(columns, row))
                yield {
                    "Id": payload.get("Id"),
                    "Title": payload.get("Title"),
                    "Body": payload.get("Body"),
                    "Tags": payload.get("Tags") or "",
                    "LastActivityDate": payload.get("LastActivityDate"),
                }
    finally:
        conn.close()


def valid_post_from_payload(payload: dict[str, Any], run_id: str, dlq_path: Path) -> Post | None:
    try:
        cleaned_body = clean_body_html(payload.get("Body", ""))
        transformed = {
            "Id": payload.get("Id"),
            "Title": (payload.get("Title") or "").strip(),
            "Body": cleaned_body.strip(),
            "Tags": payload.get("Tags") or "",
        }
        if not transformed["Title"] or not transformed["Body"]:
            write_dlq_record(dlq_path, payload, "Missing Title or Body", "validate", run_id)
            return None
        return Post.model_validate(transformed)
    except Exception as exc:
        write_dlq_record(dlq_path, payload, str(exc), "parse", run_id)
        return None

