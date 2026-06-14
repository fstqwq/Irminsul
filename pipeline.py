from __future__ import annotations

import json
import hashlib
import uuid
import shutil
import gc
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np

from core import (
    SCHEMA_VERSION,
    Settings,
    canonical_text,
    db_connection,
    embedding_key,
    index_key,
    json_dumps,
    method_key,
    row_hash,
    row_to_dict,
    rewrite_key,
    text_key,
    utc_now,
)
from search import REWRITE_PROMPT, VIEWS, RewriteResult, embed_texts, normalize_matrix, rewrite_query


ImportMode = Literal["upsert", "insert_only", "sync_source"]


class JobWorker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="yuantiji-job-worker", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.is_set():
            processed = run_next_job(self.settings)
            if not processed:
                self._stop.wait(max(0.05, float(self.settings.jobs.poll_seconds)))


def recover_startup(settings: Settings) -> None:
    with db_connection(settings) as conn:
        with conn:
            conn.execute("UPDATE jobs SET status = 'queued', updated_at = ? WHERE status = 'running'", (utc_now(),))
            conn.execute(
                "UPDATE artifacts SET status = 'pending', updated_at = ? WHERE status = 'running'",
                (utc_now(),),
            )
    building_dir = settings.storage.index_cache_dir / ".building"
    if building_dir.exists():
        shutil.rmtree(building_dir)


def run_next_job(settings: Settings) -> bool:
    with db_connection(settings) as conn:
        row = conn.execute(
            """
            SELECT * FROM jobs
            WHERE status = 'queued'
            ORDER BY created_at
            LIMIT 1
            """
        ).fetchone()
    if row is None:
        return False

    job = row_to_dict(row)
    try:
        if job["type"] == "import":
            execute_import_job(job["key"], settings)
        elif job["type"] == "build_index":
            execute_build_index_job(job["key"], settings)
        elif job["type"] == "cleanup":
            execute_cleanup_job(job["key"], settings)
        else:
            _mark_job_failed(settings, job["key"], f"unsupported job type: {job['type']}")
    except Exception as exc:
        _mark_job_failed(settings, job["key"], str(exc))
    return True


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


def rewrite_method_key(settings: Settings) -> str:
    return method_key(
        "rewrite",
        {
            "model": settings.rewrite_model.model,
            "url": settings.rewrite_model.url,
            "api_key_env": settings.rewrite_model.api_key_env,
            "prompt": REWRITE_PROMPT,
            "views": list(VIEWS),
        },
    )


def embedding_method_key(settings: Settings) -> str:
    return method_key(
        "embedding",
        {
            "model": settings.embedding_model.model,
            "url": settings.embedding_model.url,
            "api_key_env": settings.embedding_model.api_key_env,
            "dtype": "float32",
            "normalized": True,
            "views": list(VIEWS),
        },
    )


def _model_snapshot(settings: Settings, kind: Literal["rewrite", "embedding"]) -> dict[str, Any]:
    model = settings.rewrite_model if kind == "rewrite" else settings.embedding_model
    snapshot: dict[str, Any] = {
        "model": model.model,
        "url": model.url,
        "api_key_env": model.api_key_env,
    }
    if kind == "rewrite":
        snapshot["prompt"] = REWRITE_PROMPT
        snapshot["views"] = list(VIEWS)
    else:
        snapshot["dtype"] = "float32"
        snapshot["normalized"] = True
    return snapshot


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

    job = get_job(settings, job_key)
    if job is None:
        raise ValueError("import job not found after confirm")
    return job


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


def ensure_rewrite_artifact(
    settings: Settings,
    problem_text_key: str,
    selected_method_key: str | None = None,
) -> dict[str, Any]:
    selected_method_key = selected_method_key or rewrite_method_key(settings)
    selected_rewrite_key = rewrite_key(problem_text_key, selected_method_key)

    with db_connection(settings) as conn:
        source = conn.execute(
            "SELECT text FROM artifacts WHERE key = ? AND kind = 'problem_text'",
            (problem_text_key,),
        ).fetchone()
        if source is None:
            raise ValueError("problem text artifact not found")

        now = utc_now()
        with conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO artifacts(
                  key, kind, parent_key, method_key, role, text, data, blob,
                  status, attempts, error, updated_at
                )
                VALUES (?, 'rewrite', ?, ?, 'rewrite', NULL, NULL, NULL,
                        'pending', 0, NULL, ?)
                """,
                (selected_rewrite_key, problem_text_key, selected_method_key, now),
            )
        artifact = row_to_dict(
            conn.execute("SELECT * FROM artifacts WHERE key = ?", (selected_rewrite_key,)).fetchone()
        )
        if artifact["status"] == "succeeded":
            return artifact
        if artifact["status"] == "failed" and artifact["attempts"] >= settings.jobs.rewrite_max_attempts:
            raise ValueError("rewrite artifact exceeded max attempts")

        with conn:
            conn.execute(
                """
                UPDATE artifacts
                SET status = 'running', attempts = attempts + 1, error = NULL, updated_at = ?
                WHERE key = ?
                """,
                (utc_now(), selected_rewrite_key),
            )

    try:
        rewrite = rewrite_query(
            settings.rewrite_model,
            source["text"],
            timeout=settings.request_timeout,
        )
        payload = _rewrite_payload(rewrite, settings)
        with db_connection(settings) as conn:
            with conn:
                conn.execute(
                    """
                    UPDATE artifacts
                    SET status = 'succeeded', data = ?, error = NULL, updated_at = ?
                    WHERE key = ?
                    """,
                    (json_dumps(payload), utc_now(), selected_rewrite_key),
                )
        artifact = get_artifact(settings, selected_rewrite_key)
        if artifact is None:
            raise ValueError("rewrite artifact disappeared")
        return artifact
    except Exception as exc:
        _mark_artifact_failed(settings, selected_rewrite_key, str(exc))
        raise


def ensure_embedding_artifacts(
    settings: Settings,
    selected_rewrite_key: str,
    selected_method_key: str | None = None,
) -> dict[str, dict[str, Any]]:
    selected_method_key = selected_method_key or embedding_method_key(settings)
    with db_connection(settings) as conn:
        rewrite_row = conn.execute(
            "SELECT data FROM artifacts WHERE key = ? AND kind = 'rewrite' AND status = 'succeeded'",
            (selected_rewrite_key,),
        ).fetchone()
        if rewrite_row is None:
            raise ValueError("succeeded rewrite artifact not found")
        rewrite_data = json.loads(rewrite_row["data"])
        view_texts = {view: str(rewrite_data.get(view) or "") for view in VIEWS}
        if not all(view_texts.values()):
            raise ValueError("rewrite artifact does not contain all views")

        now = utc_now()
        with conn:
            for view in VIEWS:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO artifacts(
                      key, kind, parent_key, method_key, role, text, data, blob,
                      status, attempts, error, updated_at
                    )
                    VALUES (?, 'embedding', ?, ?, ?, NULL, NULL, NULL,
                            'pending', 0, NULL, ?)
                    """,
                    (
                        embedding_key(selected_rewrite_key, selected_method_key, view),
                        selected_rewrite_key,
                        selected_method_key,
                        view,
                        now,
                    ),
                )

        artifacts = {
            row["role"]: row_to_dict(row)
            for row in conn.execute(
                """
                SELECT * FROM artifacts
                WHERE kind = 'embedding'
                  AND parent_key = ?
                  AND method_key = ?
                """,
                (selected_rewrite_key, selected_method_key),
            ).fetchall()
        }

    pending_views = [
        view
        for view in VIEWS
        if artifacts[view]["status"] != "succeeded"
        and artifacts[view]["attempts"] < settings.jobs.embedding_max_attempts
    ]
    if not pending_views:
        return {view: artifacts[view] for view in VIEWS}

    for start in range(0, len(pending_views), settings.jobs.embedding_batch_size):
        batch_views = pending_views[start : start + settings.jobs.embedding_batch_size]
        batch_keys = [
            embedding_key(selected_rewrite_key, selected_method_key, view) for view in batch_views
        ]
        _mark_artifacts_running(settings, batch_keys)
        try:
            vectors = embed_texts(
                settings.embedding_model,
                [view_texts[view] for view in batch_views],
                timeout=settings.request_timeout,
            )
            matrix = normalize_matrix(np.asarray(vectors, dtype=np.float32))
            _store_embedding_batch(
                settings,
                selected_rewrite_key,
                selected_method_key,
                batch_views,
                matrix,
            )
        except Exception as exc:
            for key in batch_keys:
                _mark_artifact_failed(settings, key, str(exc))
            raise

    with db_connection(settings) as conn:
        rows = conn.execute(
            """
            SELECT * FROM artifacts
            WHERE kind = 'embedding'
              AND parent_key = ?
              AND method_key = ?
            """,
            (selected_rewrite_key, selected_method_key),
        ).fetchall()
    return {row["role"]: row_to_dict(row) for row in rows}


def get_artifact(settings: Settings, artifact_key: str) -> dict[str, Any] | None:
    with db_connection(settings) as conn:
        row = conn.execute("SELECT * FROM artifacts WHERE key = ?", (artifact_key,)).fetchone()
        return row_to_dict(row) if row else None


def _rewrite_payload(rewrite: RewriteResult, settings: Settings) -> dict[str, Any]:
    clean = rewrite.clean or rewrite.statement
    abstract_zh = rewrite.abstract_zh or rewrite.abstract
    return {
        "clean": clean,
        "statement": rewrite.statement,
        "abstract": rewrite.abstract,
        "abstract_zh": abstract_zh,
        "usage": None,
        "method_snapshot": _model_snapshot(settings, "rewrite"),
    }


def _mark_artifact_failed(settings: Settings, artifact_key: str, error: str) -> None:
    with db_connection(settings) as conn:
        with conn:
            conn.execute(
                """
                UPDATE artifacts
                SET status = 'failed', error = ?, updated_at = ?
                WHERE key = ?
                """,
                (error[:4000], utc_now(), artifact_key),
            )


def _mark_artifacts_running(settings: Settings, artifact_keys: list[str]) -> None:
    if not artifact_keys:
        return
    placeholders = ",".join("?" for _ in artifact_keys)
    with db_connection(settings) as conn:
        with conn:
            conn.execute(
                f"""
                UPDATE artifacts
                SET status = 'running', attempts = attempts + 1, error = NULL, updated_at = ?
                WHERE key IN ({placeholders})
                """,
                [utc_now(), *artifact_keys],
            )


def _store_embedding_batch(
    settings: Settings,
    selected_rewrite_key: str,
    selected_method_key: str,
    views: list[str],
    matrix: np.ndarray,
) -> None:
    now = utc_now()
    with db_connection(settings) as conn:
        with conn:
            for view, vector in zip(views, matrix, strict=True):
                payload = {
                    "dim": int(vector.shape[0]),
                    "dtype": "float32",
                    "normalized": True,
                    "usage": None,
                    "method_snapshot": _model_snapshot(settings, "embedding"),
                }
                conn.execute(
                    """
                    UPDATE artifacts
                    SET status = 'succeeded', data = ?, blob = ?, error = NULL, updated_at = ?
                    WHERE key = ?
                    """,
                    (
                        json_dumps(payload),
                        np.asarray(vector, dtype=np.float32).tobytes(),
                        now,
                        embedding_key(selected_rewrite_key, selected_method_key, view),
                    ),
                )


def create_build_index_job(settings: Settings) -> dict[str, Any]:
    job_key = "j:" + uuid.uuid4().hex
    selected_rewrite_method_key = rewrite_method_key(settings)
    selected_embedding_method_key = embedding_method_key(settings)
    snapshot_dir = settings.storage.index_cache_dir.parent / "builds" / _safe_path_key(job_key)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / "snapshot.jsonl"

    snapshot = _read_enabled_problem_snapshot(settings)
    with snapshot_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in snapshot:
            f.write(json_dumps(row) + "\n")

    now = utc_now()
    payload = {
        "snapshot_path": str(snapshot_path),
        "rewrite_method_key": selected_rewrite_method_key,
        "embedding_method_key": selected_embedding_method_key,
        "problem_count": len(snapshot),
    }
    with db_connection(settings) as conn:
        with conn:
            conn.execute(
                """
                INSERT INTO jobs(key, type, status, payload, progress, created_at, updated_at)
                VALUES (?, 'build_index', 'queued', ?, ?, ?, ?)
                """,
                (
                    job_key,
                    json_dumps(payload),
                    json_dumps({"phase": "queued", "total": len(snapshot)}),
                    now,
                    now,
                ),
            )
    job = get_job(settings, job_key)
    if job is None:
        raise ValueError("build job was not created")
    return job


def execute_build_index_job(job_key: str, settings: Settings) -> dict[str, Any]:
    with db_connection(settings) as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE key = ? AND type = 'build_index'",
            (job_key,),
        ).fetchone()
        if row is None:
            raise ValueError("build job not found")
        job = row_to_dict(row)
        if job["status"] != "queued":
            raise ValueError("build job is not queued")
        payload = json.loads(job["payload"])
        with conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'running', progress = ?, updated_at = ?
                WHERE key = ?
                """,
                (json_dumps({"phase": "artifacts"}), utc_now(), job_key),
            )

    snapshot = _read_snapshot(Path(payload["snapshot_path"]))
    selected_rewrite_method_key = str(payload["rewrite_method_key"])
    selected_embedding_method_key = str(payload["embedding_method_key"])
    prepared: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for position, problem in enumerate(snapshot, start=1):
        try:
            rewrite = ensure_rewrite_artifact(
                settings,
                problem["text_key"],
                selected_rewrite_method_key,
            )
            embeddings = ensure_embedding_artifacts(
                settings,
                rewrite["key"],
                selected_embedding_method_key,
            )
            failed_embeddings = [
                view for view, artifact in embeddings.items() if artifact["status"] != "succeeded"
            ]
            if failed_embeddings:
                failures.append(
                    {
                        "problem_key": problem["key"],
                        "error": f"embedding failed for views: {', '.join(failed_embeddings)}",
                    }
                )
            else:
                prepared.append({"problem": problem, "rewrite": rewrite, "embeddings": embeddings})
        except Exception as exc:
            failures.append({"problem_key": problem["key"], "error": str(exc)})

        _update_job_progress(
            settings,
            job_key,
            {
                "phase": "artifacts",
                "processed": position,
                "total": len(snapshot),
                "failures": len(failures),
            },
        )

    if failures:
        return _finish_job(settings, job_key, "blocked", {"failures": failures}, None)

    rows, row_hashes = _build_index_rows(prepared)
    selected_index_key = index_key(
        row_hashes,
        selected_rewrite_method_key,
        selected_embedding_method_key,
    )
    cache_path = index_cache_path(settings, selected_index_key)
    now = utc_now()
    with db_connection(settings) as conn:
        with conn:
            conn.execute(
                """
                INSERT INTO indexes(key, status, meta, created_at)
                VALUES (?, 'building', ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                  status = 'building',
                  meta = excluded.meta,
                  error = NULL
                """,
                (
                    selected_index_key,
                    json_dumps(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "rewrite_method_key": selected_rewrite_method_key,
                            "embedding_method_key": selected_embedding_method_key,
                            "problem_count": len(prepared),
                            "cache_path": str(cache_path),
                        }
                    ),
                    now,
                ),
            )
            conn.execute("DELETE FROM index_rows WHERE index_key = ?", (selected_index_key,))
            for row in rows:
                conn.execute(
                    """
                    INSERT INTO index_rows(
                      index_key, problem_ord, problem_key, view, embedding_key,
                      title, url, text_key, rewrite_key, row_hash
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        selected_index_key,
                        row["problem_ord"],
                        row["problem_key"],
                        row["view"],
                        row["embedding_key"],
                        row["title"],
                        row["url"],
                        row["text_key"],
                        row["rewrite_key"],
                        row["row_hash"],
                    ),
                )

    try:
        export_index_cache(settings, selected_index_key, prepared, cache_path)
        verify_index_cache(cache_path, selected_index_key)
        with db_connection(settings) as conn:
            with conn:
                conn.execute(
                    """
                    UPDATE indexes
                    SET status = 'built', error = NULL
                    WHERE key = ?
                    """,
                    (selected_index_key,),
                )
        return _finish_job(
            settings,
            job_key,
            "succeeded",
            {"index_key": selected_index_key, "cache_path": str(cache_path)},
            None,
        )
    except Exception as exc:
        with db_connection(settings) as conn:
            with conn:
                conn.execute(
                    "UPDATE indexes SET status = 'failed', error = ? WHERE key = ?",
                    (str(exc), selected_index_key),
                )
        return _finish_job(
            settings,
            job_key,
            "failed",
            {"index_key": selected_index_key},
            str(exc),
        )


def list_indexes(settings: Settings, limit: int = 50) -> list[dict[str, Any]]:
    with db_connection(settings) as conn:
        rows = conn.execute(
            """
            SELECT * FROM indexes
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [row_to_dict(row) for row in rows]


def get_index(settings: Settings, selected_index_key: str) -> dict[str, Any] | None:
    with db_connection(settings) as conn:
        row = conn.execute("SELECT * FROM indexes WHERE key = ?", (selected_index_key,)).fetchone()
        return row_to_dict(row) if row else None


def index_cache_path(settings: Settings, selected_index_key: str) -> Path:
    return settings.storage.index_cache_dir / _safe_path_key(selected_index_key)


def export_index_cache(
    settings: Settings,
    selected_index_key: str,
    prepared: list[dict[str, Any]],
    cache_path: Path,
    force: bool = False,
) -> None:
    building_root = settings.storage.index_cache_dir / ".building"
    temp_path = building_root / _safe_path_key(selected_index_key)
    if temp_path.exists():
        shutil.rmtree(temp_path)
    if cache_path.exists():
        if not force:
            verify_index_cache(cache_path, selected_index_key)
            return
        shutil.rmtree(cache_path)
    temp_path.mkdir(parents=True, exist_ok=True)

    problem_count = len(prepared)
    dim = _embedding_dim(next(iter(prepared[0]["embeddings"].values()))) if prepared else 0
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "index_key": selected_index_key,
        "problem_count": problem_count,
        "views": list(VIEWS),
        "dim": dim,
        "dtype": "float32",
        "files": {
            "problems": "problems.jsonl",
            "views": "views.jsonl",
            **{view: f"{view}.npy" for view in VIEWS},
        },
    }

    matrices = {
        view: np.lib.format.open_memmap(
            temp_path / f"{view}.npy",
            mode="w+",
            dtype=np.float32,
            shape=(problem_count, dim),
        )
        for view in VIEWS
    }

    with (temp_path / "problems.jsonl").open("w", encoding="utf-8", newline="\n") as problems_f, (
        temp_path / "views.jsonl"
    ).open("w", encoding="utf-8", newline="\n") as views_f:
        for problem_ord, item in enumerate(prepared):
            problem = item["problem"]
            rewrite = item["rewrite"]
            rewrite_data = json.loads(rewrite["data"])
            problems_f.write(
                json_dumps(
                    {
                        "ord": problem_ord,
                        "key": problem["key"],
                        "title": problem["title"],
                        "url": problem["url"],
                        "text_key": problem["text_key"],
                        "rewrite_key": rewrite["key"],
                    }
                )
                + "\n"
            )
            views_f.write(
                json_dumps(
                    {
                        "ord": problem_ord,
                        "key": problem["key"],
                        **{view: rewrite_data[view] for view in VIEWS},
                    }
                )
                + "\n"
            )
            for view in VIEWS:
                vector = _artifact_vector(item["embeddings"][view])
                matrices[view][problem_ord, :] = vector

    for matrix in matrices.values():
        matrix.flush()
    del matrix
    matrices.clear()
    gc.collect()
    (temp_path / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.rename(cache_path)


def verify_index_cache(cache_path: Path, expected_index_key: str) -> None:
    manifest_path = cache_path / "manifest.json"
    if not manifest_path.exists():
        raise ValueError("cache manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("index_key") != expected_index_key:
        raise ValueError("cache manifest index key mismatch")
    problem_count = int(manifest["problem_count"])
    dim = int(manifest["dim"])
    for view in VIEWS:
        matrix = np.load(cache_path / f"{view}.npy", mmap_mode="r")
        if tuple(matrix.shape) != (problem_count, dim):
            raise ValueError(f"cache matrix shape mismatch for {view}")
        if matrix.dtype != np.float32:
            raise ValueError(f"cache matrix dtype mismatch for {view}")


def rebuild_index_cache(settings: Settings, selected_index_key: str) -> dict[str, Any]:
    index = get_index(settings, selected_index_key)
    if index is None:
        raise ValueError("index not found")
    if index["status"] == "active":
        raise ValueError("cannot rebuild cache for active index")
    prepared = _prepared_from_index_rows(settings, selected_index_key)
    cache_path = index_cache_path(settings, selected_index_key)
    export_index_cache(settings, selected_index_key, prepared, cache_path, force=True)
    verify_index_cache(cache_path, selected_index_key)
    with db_connection(settings) as conn:
        with conn:
            conn.execute(
                """
                UPDATE indexes
                SET status = 'built', error = NULL
                WHERE key = ?
                """,
                (selected_index_key,),
            )
    refreshed = get_index(settings, selected_index_key)
    if refreshed is None:
        raise ValueError("index not found after cache rebuild")
    return refreshed


def _read_enabled_problem_snapshot(settings: Settings) -> list[dict[str, Any]]:
    with db_connection(settings) as conn:
        rows = conn.execute(
            """
            SELECT key, title, url, text_key
            FROM problems
            WHERE enabled = 1 AND deleted = 0
            ORDER BY key
            """
        ).fetchall()
    return [row_to_dict(row) for row in rows]


def _read_snapshot(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _build_index_rows(prepared: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    row_hashes: list[str] = []
    for problem_ord, item in enumerate(prepared):
        problem = item["problem"]
        rewrite = item["rewrite"]
        embeddings = item["embeddings"]
        for view in VIEWS:
            selected_embedding_key = embeddings[view]["key"]
            selected_row_hash = row_hash(
                problem_ord=problem_ord,
                problem_key=problem["key"],
                view=view,
                row_embedding_key=selected_embedding_key,
                title=problem["title"],
                url=problem["url"],
                row_text_key=problem["text_key"],
                row_rewrite_key=rewrite["key"],
            )
            row_hashes.append(selected_row_hash)
            rows.append(
                {
                    "problem_ord": problem_ord,
                    "problem_key": problem["key"],
                    "view": view,
                    "embedding_key": selected_embedding_key,
                    "title": problem["title"],
                    "url": problem["url"],
                    "text_key": problem["text_key"],
                    "rewrite_key": rewrite["key"],
                    "row_hash": selected_row_hash,
                }
            )
    return rows, row_hashes


def _prepared_from_index_rows(settings: Settings, selected_index_key: str) -> list[dict[str, Any]]:
    with db_connection(settings) as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM index_rows
            WHERE index_key = ?
            ORDER BY problem_ord, view
            """,
            (selected_index_key,),
        ).fetchall()
        artifacts = {
            row["key"]: row_to_dict(row)
            for row in conn.execute(
                """
                SELECT a.*
                FROM artifacts a
                WHERE a.key IN (
                  SELECT rewrite_key FROM index_rows WHERE index_key = ?
                  UNION
                  SELECT embedding_key FROM index_rows WHERE index_key = ?
                )
                """,
                (selected_index_key, selected_index_key),
            ).fetchall()
        }

    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["problem_ord"]), []).append(row_to_dict(row))

    prepared: list[dict[str, Any]] = []
    for problem_ord in sorted(grouped):
        problem_rows = grouped[problem_ord]
        first = problem_rows[0]
        rewrite = artifacts.get(first["rewrite_key"])
        if rewrite is None:
            raise ValueError("rewrite artifact missing for index row")
        embeddings: dict[str, dict[str, Any]] = {}
        for row in problem_rows:
            embedding = artifacts.get(row["embedding_key"])
            if embedding is None:
                raise ValueError("embedding artifact missing for index row")
            embeddings[row["view"]] = embedding
        if set(embeddings) != set(VIEWS):
            raise ValueError("index rows do not contain all views")
        prepared.append(
            {
                "problem": {
                    "key": first["problem_key"],
                    "title": first["title"],
                    "url": first["url"],
                    "text_key": first["text_key"],
                },
                "rewrite": rewrite,
                "embeddings": embeddings,
            }
        )
    return prepared


def _embedding_dim(artifact: dict[str, Any]) -> int:
    data = json.loads(artifact["data"])
    return int(data["dim"])


def _artifact_vector(artifact: dict[str, Any]) -> np.ndarray:
    dim = _embedding_dim(artifact)
    vector = np.frombuffer(artifact["blob"], dtype=np.float32)
    if vector.shape != (dim,):
        raise ValueError("embedding blob dimension mismatch")
    return vector


def _safe_path_key(key: str) -> str:
    return key.replace(":", "_")


def list_problems(
    settings: Settings,
    source_key: str | None = None,
    enabled: bool | None = None,
    deleted: bool | None = None,
    q: str = "",
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    where = ["1 = 1"]
    params: list[Any] = []
    if source_key:
        where.append("source_key = ?")
        params.append(source_key)
    if enabled is not None:
        where.append("enabled = ?")
        params.append(1 if enabled else 0)
    if deleted is not None:
        where.append("deleted = ?")
        params.append(1 if deleted else 0)
    if q.strip():
        where.append("(key LIKE ? OR title LIKE ? OR url LIKE ?)")
        needle = f"%{q.strip()}%"
        params.extend([needle, needle, needle])

    where_sql = " AND ".join(where)
    limit = max(1, min(limit, 500))
    offset = max(0, offset)
    with db_connection(settings) as conn:
        total = conn.execute(
            f"SELECT count(*) FROM problems WHERE {where_sql}",
            params,
        ).fetchone()[0]
        rows = conn.execute(
            f"""
            SELECT key, source_key, title, url, text_key, enabled, deleted, updated_at
            FROM problems
            WHERE {where_sql}
            ORDER BY updated_at DESC, key
            LIMIT ? OFFSET ?
            """,
            [*params, limit, offset],
        ).fetchall()
    return {"total": total, "items": [row_to_dict(row) for row in rows]}


def patch_problem(settings: Settings, problem_key: str, changes: dict[str, Any]) -> dict[str, Any]:
    allowed = {"title", "url", "enabled", "deleted"}
    assignments: list[str] = []
    params: list[Any] = []
    for key, value in changes.items():
        if key not in allowed or value is None:
            continue
        assignments.append(f"{key} = ?")
        params.append(int(value) if key in {"enabled", "deleted"} else str(value))
    if not assignments:
        raise ValueError("no valid problem changes")
    assignments.append("updated_at = ?")
    params.extend([utc_now(), problem_key])
    with db_connection(settings) as conn:
        with conn:
            cursor = conn.execute(
                f"UPDATE problems SET {', '.join(assignments)} WHERE key = ?",
                params,
            )
        if cursor.rowcount == 0:
            raise ValueError("problem not found")
        row = conn.execute("SELECT * FROM problems WHERE key = ?", (problem_key,)).fetchone()
    return row_to_dict(row)


def batch_update_problems(settings: Settings, keys: list[str], action: str) -> dict[str, Any]:
    if not keys:
        raise ValueError("keys are required")
    if action not in {"enable", "disable", "delete", "restore"}:
        raise ValueError("invalid batch action")
    if action == "enable":
        assignments = "enabled = 1"
    elif action == "disable":
        assignments = "enabled = 0"
    elif action == "delete":
        assignments = "deleted = 1"
    else:
        assignments = "deleted = 0"

    placeholders = ",".join("?" for _ in keys)
    with db_connection(settings) as conn:
        with conn:
            cursor = conn.execute(
                f"""
                UPDATE problems
                SET {assignments}, updated_at = ?
                WHERE key IN ({placeholders})
                """,
                [utc_now(), *keys],
            )
    return {"updated": cursor.rowcount}


def list_sources(settings: Settings) -> list[dict[str, Any]]:
    with db_connection(settings) as conn:
        rows = conn.execute(
            """
            SELECT
              s.key, s.name, s.enabled, s.updated_at,
              count(p.key) AS problem_count,
              sum(CASE WHEN p.enabled = 1 AND p.deleted = 0 THEN 1 ELSE 0 END) AS active_count,
              sum(CASE WHEN p.deleted = 1 THEN 1 ELSE 0 END) AS deleted_count
            FROM sources s
            LEFT JOIN problems p ON p.source_key = s.key
            GROUP BY s.key, s.name, s.enabled, s.updated_at
            ORDER BY s.key
            """
        ).fetchall()
    return [row_to_dict(row) for row in rows]


def patch_source(settings: Settings, source_key: str, changes: dict[str, Any]) -> dict[str, Any]:
    allowed = {"name", "enabled"}
    assignments: list[str] = []
    params: list[Any] = []
    for key, value in changes.items():
        if key not in allowed or value is None:
            continue
        assignments.append(f"{key} = ?")
        params.append(int(value) if key == "enabled" else str(value))
    if not assignments:
        raise ValueError("no valid source changes")
    assignments.append("updated_at = ?")
    params.extend([utc_now(), source_key])
    with db_connection(settings) as conn:
        with conn:
            cursor = conn.execute(
                f"UPDATE sources SET {', '.join(assignments)} WHERE key = ?",
                params,
            )
        if cursor.rowcount == 0:
            raise ValueError("source not found")
        row = conn.execute("SELECT * FROM sources WHERE key = ?", (source_key,)).fetchone()
    return row_to_dict(row)


def list_jobs(
    settings: Settings,
    job_type: str | None = None,
    status: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    where = ["1 = 1"]
    params: list[Any] = []
    if job_type:
        where.append("type = ?")
        params.append(job_type)
    if status:
        where.append("status = ?")
        params.append(status)
    limit = max(1, min(limit, 500))
    with db_connection(settings) as conn:
        rows = conn.execute(
            f"""
            SELECT * FROM jobs
            WHERE {' AND '.join(where)}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            [*params, limit],
        ).fetchall()
    return [row_to_dict(row) for row in rows]


def retry_job(settings: Settings, job_key: str) -> dict[str, Any]:
    job = get_job(settings, job_key)
    if job is None:
        raise ValueError("job not found")
    if job["status"] not in {"blocked", "failed"}:
        raise ValueError("job is not retryable")
    with db_connection(settings) as conn:
        with conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'queued', error = NULL, updated_at = ?
                WHERE key = ?
                """,
                (utc_now(), job_key),
            )
    return get_job(settings, job_key) or job


def create_cleanup_job(settings: Settings) -> dict[str, Any]:
    job_key = "j:" + uuid.uuid4().hex
    now = utc_now()
    with db_connection(settings) as conn:
        with conn:
            conn.execute(
                """
                INSERT INTO jobs(key, type, status, payload, progress, created_at, updated_at)
                VALUES (?, 'cleanup', 'queued', ?, ?, ?, ?)
                """,
                (job_key, json_dumps({}), json_dumps({"phase": "queued"}), now, now),
            )
    job = get_job(settings, job_key)
    if job is None:
        raise ValueError("cleanup job was not created")
    return job


def _ensure_inside(root: Path, target: Path) -> Path:
    root_path = root.resolve()
    target_path = target.resolve()
    if target_path == root_path or root_path not in target_path.parents:
        raise ValueError(f"refusing to delete outside configured directory: {target}")
    return target_path


def _remove_file_inside(root: Path, target: Path) -> int:
    _ensure_inside(root, target)
    if target.is_file():
        target.unlink()
        return 1
    return 0


def _remove_dir_inside(root: Path, target: Path) -> int:
    _ensure_inside(root, target)
    if target.is_dir():
        shutil.rmtree(target)
        return 1
    return 0


def _active_import_uploads(settings: Settings) -> set[Path]:
    upload_root = settings.storage.upload_dir
    protected: set[Path] = set()
    with db_connection(settings) as conn:
        rows = conn.execute(
            """
            SELECT payload
            FROM jobs
            WHERE type = 'import' AND status IN ('draft', 'queued', 'running')
            """
        ).fetchall()
    for row in rows:
        try:
            payload = json.loads(row["payload"])
            path = Path(str(payload.get("path", "")))
            protected.add(_ensure_inside(upload_root, path))
        except Exception:
            continue
    return protected


def _cleanup_expired_uploads(settings: Settings) -> int:
    upload_root = settings.storage.upload_dir
    if not upload_root.exists():
        return 0
    protected = _active_import_uploads(settings)
    cutoff = datetime.now(UTC) - timedelta(days=settings.audit.retention_days)
    removed = 0
    for path in upload_root.iterdir():
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved in protected:
            continue
        modified_at = datetime.fromtimestamp(path.stat().st_mtime, UTC)
        if modified_at < cutoff:
            removed += _remove_file_inside(upload_root, path)
    return removed


def _kept_index_cache_names(settings: Settings) -> set[str]:
    keep_keys: set[str] = set()
    with db_connection(settings) as conn:
        active = conn.execute(
            "SELECT value FROM kv WHERE key = 'active_index_key'"
        ).fetchone()
        if active is not None:
            keep_keys.add(str(active["value"]))
        keep_keys.update(
            str(row["key"])
            for row in conn.execute(
                "SELECT key FROM indexes WHERE status = 'active'"
            ).fetchall()
        )
        if settings.index_cache.keep_retired > 0:
            keep_keys.update(
                str(row["key"])
                for row in conn.execute(
                    """
                    SELECT key
                    FROM indexes
                    WHERE status IN ('built', 'retired')
                    ORDER BY COALESCE(activated_at, created_at) DESC
                    LIMIT ?
                    """,
                    (settings.index_cache.keep_retired,),
                ).fetchall()
            )
    return {_safe_path_key(key) for key in keep_keys}


def _cleanup_old_index_caches(settings: Settings) -> int:
    cache_root = settings.storage.index_cache_dir
    if not cache_root.exists():
        return 0
    keep_names = _kept_index_cache_names(settings)
    removed = 0
    for path in cache_root.iterdir():
        if path.name == ".building" or not path.is_dir():
            continue
        if path.name not in keep_names:
            removed += _remove_dir_inside(cache_root, path)
    return removed


def execute_cleanup_job(job_key: str, settings: Settings) -> dict[str, Any]:
    cutoff = utc_now()
    removed_audits = 0
    with db_connection(settings) as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE key = ? AND type = 'cleanup'",
            (job_key,),
        ).fetchone()
        if row is None:
            raise ValueError("cleanup job not found")
        if row["status"] != "queued":
            raise ValueError("cleanup job is not queued")
        with conn:
            conn.execute(
                "UPDATE jobs SET status = 'running', progress = ?, updated_at = ? WHERE key = ?",
                (json_dumps({"phase": "audits"}), utc_now(), job_key),
            )
            cursor = conn.execute(
                """
                DELETE FROM search_audits
                WHERE started_at < datetime('now', ?)
                """,
                (f"-{settings.audit.retention_days} days",),
            )
            removed_audits = cursor.rowcount

    _update_job_progress(settings, job_key, {"phase": "uploads"})
    removed_uploads = _cleanup_expired_uploads(settings)

    _update_job_progress(settings, job_key, {"phase": "cache"})
    building_dir = settings.storage.index_cache_dir / ".building"
    removed_building_dirs = _remove_dir_inside(settings.storage.index_cache_dir, building_dir)
    removed_index_caches = _cleanup_old_index_caches(settings)

    return _finish_job(
        settings,
        job_key,
        "succeeded",
        {
            "removed_audits": removed_audits,
            "removed_uploads": removed_uploads,
            "removed_building_dirs": removed_building_dirs,
            "removed_index_caches": removed_index_caches,
            "cutoff": cutoff,
        },
        None,
    )


def _mark_job_failed(settings: Settings, job_key: str, error: str) -> None:
    with db_connection(settings) as conn:
        with conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'failed', error = ?, updated_at = ?
                WHERE key = ? AND status IN ('queued', 'running')
                """,
                (error[:4000], utc_now(), job_key),
            )


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
    status: Literal["succeeded", "blocked", "failed"],
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
