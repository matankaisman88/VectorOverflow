from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from lxml import etree, html

from so_rag.config import Settings
from so_rag.models import DlqRecord, Post


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


def parse_posts_from_db(settings: Settings, limit: int | None = None) -> Iterator[dict[str, Any]]:
    try:
        import pyodbc  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("pyodbc is required for DB ingestion") from exc

    query = settings.db_query.strip()
    max_rows = limit if limit is not None else settings.db_limit
    if max_rows > 0:
        upper = query.upper()
        if upper.startswith("SELECT "):
            query = query.replace("SELECT ", f"SELECT TOP {max_rows} ", 1)
        else:
            raise ValueError("DB_QUERY must start with SELECT")

    conn = pyodbc.connect(_build_mssql_connection_string(settings), timeout=30)
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]
        for row in cursor.fetchall():
            payload = dict(zip(columns, row))
            # Keep parity with XML payload key casing expected downstream.
            yield {
                "Id": payload.get("Id"),
                "Title": payload.get("Title"),
                "Body": payload.get("Body"),
                "Tags": payload.get("Tags") or "",
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

