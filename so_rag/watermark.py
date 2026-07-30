from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel


class Watermark(BaseModel):
    last_id: int = 0
    last_activity_date: datetime | None = None
    updated_at: datetime | None = None


def load_watermark(path: Path) -> Watermark:
    if not path.exists():
        return Watermark()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return Watermark.model_validate(data)
    except Exception:
        # Corrupt/unreadable watermark must never crash ingestion — fall back to full scan.
        return Watermark()


def save_watermark(path: Path, last_id: int, last_activity_date: datetime | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wm = Watermark(
        last_id=last_id,
        last_activity_date=last_activity_date,
        updated_at=datetime.now(timezone.utc),
    )
    path.write_text(wm.model_dump_json(indent=2), encoding="utf-8")


def reset_watermark(path: Path) -> None:
    if path.exists():
        path.unlink()
