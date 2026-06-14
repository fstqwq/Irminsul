from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from core import Settings, canonical_text, db_connection, json_dumps, text_key, utc_now


ImportMode = Literal["upsert", "insert_only", "sync_source"]


@dataclass(frozen=True)
class ImportRow:
    problem_key: str
    source_key: str
    title: str
    text: str
    url: str
    line_number: int


@dataclass
class ImportStats:
    total: int = 0
    new: int = 0
    overwrite: int = 0
    skip: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)


def source_key_from_problem_id(problem_id: str) -> str:
    cleaned = problem_id.strip()
    for separator in ("/", ":", "_", "-"):
        if separator in cleaned:
            prefix = cleaned.split(separator, 1)[0].strip()
            if prefix:
                return prefix
    return "unknown"


def validate_import_row(raw: dict[str, Any], line_number: int, settings: Settings) -> ImportRow:
    problem_id = str(raw.get("id") or "").strip()
    title = str(raw.get("title") or "").strip()
    text = canonical_text(str(raw.get("text") or ""))
    url = str(raw.get("url") or "").strip()

    if not problem_id:
        raise ValueError("id is required")
    if not title:
        raise ValueError("title is required")
    if not text:
        raise ValueError("text is required")
    if len(title) > settings.limits.field_max_text_chars:
        raise ValueError("title is too long")
    if len(text) > settings.limits.field_max_text_chars:
        raise ValueError("text is too long")

    return ImportRow(
        problem_key=problem_id,
        source_key=source_key_from_problem_id(problem_id),
        title=title,
        text=text,
        url=url,
        line_number=line_number,
    )


def read_import_jsonl(path: Path, settings: Settings) -> tuple[list[ImportRow], list[dict[str, Any]]]:
    rows: list[ImportRow] = []
    errors: list[dict[str, Any]] = []
    with path.open("rb") as f:
        for index, raw_line in enumerate(f, start=1):
            if len(raw_line) > settings.limits.jsonl_max_line_bytes:
                errors.append({"line": index, "error": "line is too large"})
                continue
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("line must be a JSON object")
                rows.append(validate_import_row(payload, index, settings))
            except Exception as exc:
                errors.append({"line": index, "error": str(exc)})
    return rows, errors


def create_import_dry_run(path: Path, mode: ImportMode, settings: Settings) -> dict[str, Any]:
    rows, errors = read_import_jsonl(path, settings)
    stats = ImportStats(total=len(rows), errors=errors)

    with db_connection(settings) as conn:
        existing_keys = {
            row["key"]
            for row in conn.execute(
                "SELECT key FROM problems WHERE key IN (%s)"
                % ",".join("?" for _ in rows),
                [row.problem_key for row in rows],
            )
        } if rows else set()

        seen: set[str] = set()
        for row in rows:
            if row.problem_key in seen:
                stats.errors.append({"line": row.line_number, "error": "duplicate id in upload"})
                continue
            seen.add(row.problem_key)
            if row.problem_key in existing_keys:
                if mode == "insert_only":
                    stats.skip += 1
                else:
                    stats.overwrite += 1
            else:
                stats.new += 1

        job_key = "j:" + uuid.uuid4().hex
        now = utc_now()
        with conn:
            conn.execute(
                """
                INSERT INTO jobs(key, type, status, payload, progress, created_at, updated_at)
                VALUES (?, 'import', 'draft', ?, ?, ?, ?)
                """,
                (
                    job_key,
                    json_dumps({"path": str(path), "mode": mode, "text_keys": [text_key(row.text) for row in rows]}),
                    json_dumps({"stats": stats.__dict__}),
                    now,
                    now,
                ),
            )

    return {"job_key": job_key, "stats": stats.__dict__}
