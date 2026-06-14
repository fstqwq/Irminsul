from __future__ import annotations

import json
import hashlib
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from core import Settings, canonical_text, db_connection, json_dumps, row_to_dict, text_key, utc_now


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _validate_import_mode(mode: str) -> ImportMode:
    if mode not in {"upsert", "insert_only", "sync_source"}:
        raise ValueError("invalid import mode")
    return mode  # type: ignore[return-value]


def create_import_dry_run(path: Path, mode: str, settings: Settings) -> dict[str, Any]:
    import_mode = _validate_import_mode(mode)
    rows, errors = read_import_jsonl(path, settings)
    stats = ImportStats(total=len(rows), errors=errors)
    source_keys = sorted({row.source_key for row in rows})
    if import_mode == "sync_source" and len(source_keys) != 1:
        stats.errors.append({"line": 0, "error": "sync_source requires exactly one source"})

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
                if import_mode == "insert_only":
                    stats.skip += 1
                else:
                    stats.overwrite += 1
            else:
                stats.new += 1

        job_key = "j:" + uuid.uuid4().hex
        now = utc_now()
        stat_result = {
            "total": stats.total,
            "new": stats.new,
            "overwrite": stats.overwrite,
            "skip": stats.skip,
            "errors": stats.errors,
        }
        with conn:
            conn.execute(
                """
                INSERT INTO jobs(key, type, status, payload, progress, created_at, updated_at)
                VALUES (?, 'import', 'draft', ?, ?, ?, ?)
                """,
                (
                    job_key,
                    json_dumps(
                        {
                            "path": str(path),
                            "mode": import_mode,
                            "file_sha256": file_sha256(path),
                            "file_size": path.stat().st_size,
                            "source_keys": source_keys,
                            "text_keys": [text_key(row.text) for row in rows],
                        }
                    ),
                    json_dumps({"stats": stat_result}),
                    now,
                    now,
                ),
            )

    return {"job_key": job_key, "stats": stat_result}


def get_job(settings: Settings, job_key: str) -> dict[str, Any] | None:
    with db_connection(settings) as conn:
        row = conn.execute("SELECT * FROM jobs WHERE key = ?", (job_key,)).fetchone()
        return row_to_dict(row) if row else None


def list_import_jobs(settings: Settings, limit: int = 50) -> list[dict[str, Any]]:
    with db_connection(settings) as conn:
        rows = conn.execute(
            """
            SELECT * FROM jobs
            WHERE type = 'import'
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [row_to_dict(row) for row in rows]


def confirm_import_job(job_key: str, settings: Settings) -> dict[str, Any]:
    with db_connection(settings) as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE key = ? AND type = 'import'",
            (job_key,),
        ).fetchone()
        if row is None:
            raise ValueError("import job not found")
        job = row_to_dict(row)
        if job["status"] != "draft":
            raise ValueError("import job is not draft")
        progress = json.loads(job["progress"])
        dry_run_errors = progress.get("stats", {}).get("errors", [])
        if dry_run_errors:
            raise ValueError("dry run has errors")

        payload = json.loads(job["payload"])
        path = Path(payload["path"])
        if not path.exists():
            raise ValueError("uploaded file is missing")
        if path.stat().st_size != int(payload["file_size"]):
            raise ValueError("uploaded file size changed")
        if file_sha256(path) != payload["file_sha256"]:
            raise ValueError("uploaded file checksum changed")

        now = utc_now()
        with conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'queued', updated_at = ?
                WHERE key = ? AND status = 'draft'
                """,
                (now, job_key),
            )

    return execute_import_job(job_key, settings)


def execute_import_job(job_key: str, settings: Settings) -> dict[str, Any]:
    with db_connection(settings) as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE key = ? AND type = 'import'",
            (job_key,),
        ).fetchone()
        if row is None:
            raise ValueError("import job not found")
        job = row_to_dict(row)
        if job["status"] != "queued":
            raise ValueError("import job is not queued")
        payload = json.loads(job["payload"])
        path = Path(payload["path"])
        mode = _validate_import_mode(str(payload["mode"]))

        now = utc_now()
        with conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'running', progress = ?, updated_at = ?
                WHERE key = ?
                """,
                (json_dumps({"phase": "reading"}), now, job_key),
            )

    rows, errors = read_import_jsonl(path, settings)
    if errors:
        _finish_job(settings, job_key, "failed", {"errors": errors}, "upload validation failed")
        raise ValueError("upload validation failed")

    stats = {"total": len(rows), "new": 0, "overwrite": 0, "skip": 0, "disabled": 0}
    row_keys_by_source: dict[str, set[str]] = {}

    with db_connection(settings) as conn:
        for index, row in enumerate(rows, start=1):
            row_keys_by_source.setdefault(row.source_key, set()).add(row.problem_key)
            now = utc_now()
            problem_text_key = text_key(row.text)
            existing = conn.execute(
                "SELECT key FROM problems WHERE key = ?",
                (row.problem_key,),
            ).fetchone()

            if existing is not None and mode == "insert_only":
                stats["skip"] += 1
            else:
                with conn:
                    conn.execute(
                        """
                        INSERT INTO sources(key, name, enabled, updated_at)
                        VALUES (?, ?, 1, ?)
                        ON CONFLICT(key) DO UPDATE SET updated_at = excluded.updated_at
                        """,
                        (row.source_key, row.source_key, now),
                    )
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO artifacts(
                          key, kind, parent_key, method_key, role, text, data, blob,
                          status, attempts, error, updated_at
                        )
                        VALUES (?, 'problem_text', NULL, NULL, NULL, ?, NULL, NULL,
                                'succeeded', 0, NULL, ?)
                        """,
                        (problem_text_key, row.text, now),
                    )
                    if existing is None:
                        stats["new"] += 1
                        conn.execute(
                            """
                            INSERT INTO problems(
                              key, source_key, title, url, text_key, enabled, deleted, updated_at
                            )
                            VALUES (?, ?, ?, ?, ?, 1, 0, ?)
                            """,
                            (
                                row.problem_key,
                                row.source_key,
                                row.title,
                                row.url,
                                problem_text_key,
                                now,
                            ),
                        )
                    else:
                        stats["overwrite"] += 1
                        conn.execute(
                            """
                            UPDATE problems
                            SET source_key = ?, title = ?, url = ?, text_key = ?,
                                enabled = 1, deleted = 0, updated_at = ?
                            WHERE key = ?
                            """,
                            (
                                row.source_key,
                                row.title,
                                row.url,
                                problem_text_key,
                                now,
                                row.problem_key,
                            ),
                        )

            if index % 100 == 0 or index == len(rows):
                _update_job_progress(
                    settings,
                    job_key,
                    {"phase": "importing", "processed": index, "total": len(rows)},
                )

        if mode == "sync_source":
            for source_key, imported_keys in row_keys_by_source.items():
                placeholders = ",".join("?" for _ in imported_keys)
                params = [source_key, *sorted(imported_keys)]
                with conn:
                    cursor = conn.execute(
                        f"""
                        UPDATE problems
                        SET enabled = 0, updated_at = ?
                        WHERE source_key = ?
                          AND key NOT IN ({placeholders})
                          AND deleted = 0
                          AND enabled = 1
                        """,
                        [utc_now(), *params],
                    )
                    stats["disabled"] += cursor.rowcount

    return _finish_job(settings, job_key, "succeeded", stats, None)


def _update_job_progress(settings: Settings, job_key: str, progress: dict[str, Any]) -> None:
    with db_connection(settings) as conn:
        with conn:
            conn.execute(
                "UPDATE jobs SET progress = ?, updated_at = ? WHERE key = ?",
                (json_dumps(progress), utc_now(), job_key),
            )


def _finish_job(
    settings: Settings,
    job_key: str,
    status: Literal["succeeded", "failed"],
    result: dict[str, Any],
    error: str | None,
) -> dict[str, Any]:
    with db_connection(settings) as conn:
        with conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = ?, result = ?, error = ?, updated_at = ?
                WHERE key = ?
                """,
                (status, json_dumps(result), error, utc_now(), job_key),
            )
    job = get_job(settings, job_key)
    if job is None:
        raise ValueError("import job not found after finish")
    return job
