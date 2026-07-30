from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from so_rag.ingestion import _inject_incremental_predicate
from so_rag.watermark import Watermark, load_watermark, save_watermark

BASE_QUERY = (
    "SELECT Id, Title, Body, Tags, LastActivityDate "
    "FROM Posts "
    "WHERE PostTypeId = 1 "
    "ORDER BY Id"
)
WM = Watermark(last_id=100, last_activity_date=datetime(2020, 1, 1, tzinfo=timezone.utc))


def test_inject_incremental_predicate_with_existing_where_and_order_by() -> None:
    result = _inject_incremental_predicate(BASE_QUERY, WM)
    assert "WHERE PostTypeId = 1 AND (Id > ? OR LastActivityDate > ?)" in result
    assert result.endswith("ORDER BY Id")


def test_inject_incremental_predicate_with_where_no_order_by() -> None:
    query = "SELECT Id FROM Posts WHERE PostTypeId = 1"
    result = _inject_incremental_predicate(query, WM)
    assert "WHERE PostTypeId = 1 AND (Id > ? OR LastActivityDate > ?)" in result


def test_inject_incremental_predicate_without_where() -> None:
    query = "SELECT Id FROM Posts ORDER BY Id"
    result = _inject_incremental_predicate(query, WM)
    assert "WHERE (Id > ? OR LastActivityDate > ?)" in result
    assert "ORDER BY Id" in result


def test_inject_incremental_predicate_malformed_query_raises() -> None:
    with pytest.raises(ValueError, match="DB_QUERY must start with SELECT"):
        _inject_incremental_predicate("UPDATE Posts SET Title = 'x'", WM)


def test_load_watermark_missing_file_returns_zero_watermark(tmp_path: Path) -> None:
    wm = load_watermark(tmp_path / "missing.json")
    assert wm.last_id == 0
    assert wm.last_activity_date is None


def test_load_watermark_corrupt_json_returns_zero_watermark(tmp_path: Path) -> None:
    path = tmp_path / "watermark.json"
    path.write_text("{not valid json", encoding="utf-8")
    wm = load_watermark(path)
    assert wm.last_id == 0


def test_save_and_load_watermark_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "watermark.json"
    activity = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
    save_watermark(path, last_id=42, last_activity_date=activity)
    wm = load_watermark(path)
    assert wm.last_id == 42
    assert wm.last_activity_date == activity
    assert wm.updated_at is not None
