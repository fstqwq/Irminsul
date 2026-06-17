from __future__ import annotations

import json
import hashlib
import shutil
import uuid
import gc
import threading
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TypeVar

import numpy as np

from core import (
    SCHEMA_VERSION,
    Settings,
    append_job_log,
    cancel_requested,
    canonical_text,
    db_exec,
    db_one,
    db_read_connection,
    db_write_connection,
    embedding_key,
    finish_if_cancel_requested,
    finish_job,
    get_index,
    get_job,
    index_key,
    json_dumps,
    job_progress,
    job_result,
    mark_job_failed,
    method_key,
    row_hash,
    row_to_dict,
    rewrite_key,
    text_key,
    update_job_progress,
    utc_now,
)
from search import REWRITE_PROMPT, VIEWS, RewriteResult, embed_texts, normalize_matrix, read_jsonl_list, rewrite_query


ImportMode = Literal["upsert", "insert_only", "sync_source"]
T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class PipelineJob:
    settings: Settings
    key: str

    def log(
        self,
        level: Literal["info", "warning", "error"],
        message: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        append_job_log(self.settings, self.key, level, message, data)

    def info(self, message: str, data: dict[str, Any] | None = None) -> None:
        self.log("info", message, data)

    def warning(self, message: str, data: dict[str, Any] | None = None) -> None:
        self.log("warning", message, data)

    def error(self, message: str, data: dict[str, Any] | None = None) -> None:
        self.log("error", message, data)

    def progress(self, value: dict[str, Any], result: dict[str, Any] | None = None) -> None:
        update_job_progress(self.settings, self.key, value, result)

    def cancel_requested(self) -> bool:
        return cancel_requested(self.settings, self.key)

    def finish_if_canceled(self, result: dict[str, Any] | None = None) -> dict[str, Any] | None:
        return finish_if_cancel_requested(self.settings, self.key, result)

    def finish(
        self,
        status: Literal["succeeded", "blocked", "failed"],
        result: dict[str, Any],
        error: str | None,
    ) -> dict[str, Any]:
        return finish_job(self.settings, self.key, status, result, error)


def batch_process(
    items: Sequence[T],
    fn: Callable[[T], R],
    max_workers: int = 1,
    cancel: Callable[[], bool] | None = None,
) -> Iterator[tuple[int, T, R | None, Exception | None]]:
    if not items:
        return
    canceled = False
    pool = ThreadPoolExecutor(max_workers=max_workers)
    try:
        pending = {pool.submit(fn, item): (index, item) for index, item in enumerate(items)}
        while pending:
            try:
                future = next(as_completed(pending, timeout=1.0))
            except FuturesTimeoutError:
                if cancel and cancel():
                    canceled = True
                    return
                continue
            index, item = pending.pop(future)
            try:
                yield index, item, future.result(), None
            except Exception as exc:
                yield index, item, None, exc
            if cancel and cancel():
                canceled = True
                return
    finally:
        pool.shutdown(wait=not canceled, cancel_futures=canceled)


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
    with db_write_connection(settings) as conn:
        running_jobs = conn.execute("SELECT * FROM jobs WHERE status = 'running'").fetchall()
        with conn:
            for row in running_jobs:
                job = row_to_dict(row)
                if job_progress(job).get("cancel_requested"):
                    result = job_result(job)
                    result["canceled"] = True
                    conn.execute(
                        """
                        UPDATE jobs
                        SET status = 'failed', result = ?, error = ?, updated_at = ?
                        WHERE key = ?
                        """,
                        (json_dumps(result), "Canceled by admin", utc_now(), job["key"]),
                    )
                else:
                    conn.execute(
                        "UPDATE jobs SET status = 'queued', updated_at = ? WHERE key = ?",
                        (utc_now(), job["key"]),
                    )
            conn.execute(
                "UPDATE artifacts SET status = 'pending', updated_at = ? WHERE status = 'running'",
                (utc_now(),),
            )
    building_dir = settings.storage.index_cache_dir / ".building"
    if building_dir.exists():
        shutil.rmtree(building_dir)


def run_next_job(settings: Settings) -> bool:
    with db_read_connection(settings) as conn:
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
    job_ctx = PipelineJob(settings, job["key"])
    try:
        if job["type"] == "import":
            execute_import_job(job["key"], settings)
        elif job["type"] == "build_index":
            execute_build_index_job(job["key"], settings)
        else:
            mark_job_failed(settings, job["key"], f"unsupported job type: {job['type']}")
    except Exception as exc:
        job_ctx.error("Job failed", {"error": str(exc)})
        mark_job_failed(settings, job["key"], str(exc))
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


@dataclass(frozen=True)
class EmbeddingWorkItem:
    rewrite_key: str
    artifact_key: str
    view: str
    text: str


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
            "model": settings.rewrite_model.identity,
            "prompt": REWRITE_PROMPT,
            "views": list(VIEWS),
        },
    )


def embedding_method_key(settings: Settings) -> str:
    return method_key(
        "embedding",
        {
            "model": settings.embedding_model.identity,
            "dtype": "float32",
            "normalized": True,
            "views": list(VIEWS),
        },
    )


def _model_snapshot(settings: Settings, kind: Literal["rewrite", "embedding"]) -> dict[str, Any]:
    model = settings.rewrite_model if kind == "rewrite" else settings.embedding_model
    snapshot: dict[str, Any] = {
        "model": model.model,
        "identity": model.identity,
        "url": model.url,
        "api_key_env": model.api_key_env,
    }
    if model.provider:
        snapshot["provider"] = model.provider
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


def create_import_dry_run(
    path: Path,
    mode: str,
    settings: Settings,
    filename: str | None = None,
) -> dict[str, Any]:
    import_mode = _validate_import_mode(mode)
    rows, errors = read_import_jsonl(path, settings)
    stats = ImportStats(total=len(rows), errors=errors)
    source_keys = sorted({row.source_key for row in rows})
    if import_mode == "sync_source" and len(source_keys) != 1:
        stats.errors.append({"line": 0, "error": "sync_source requires exactly one source"})

    with db_write_connection(settings) as conn:
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
                            "filename": filename or path.name,
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

    return {"job_key": job_key, "filename": filename or path.name, "stats": stat_result}


def delete_import_draft(job_key: str, settings: Settings) -> dict[str, Any]:
    upload_path: Path | None = None
    with db_write_connection(settings) as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE key = ? AND type = 'import'",
            (job_key,),
        ).fetchone()
        if row is None:
            raise ValueError("import job not found")
        job = row_to_dict(row)
        if job["status"] != "draft":
            raise ValueError("only draft import jobs can be deleted")

        try:
            payload = json.loads(job["payload"] or "{}")
            raw_path = payload.get("path")
            if raw_path:
                candidate = Path(str(raw_path)).resolve()
                upload_root = settings.storage.upload_dir.resolve()
                if upload_root in candidate.parents:
                    upload_path = candidate
        except Exception:
            upload_path = None

        with conn:
            conn.execute("DELETE FROM jobs WHERE key = ?", (job_key,))

    if upload_path is not None:
        try:
            upload_path.unlink(missing_ok=True)
        except OSError:
            pass
    return job


def confirm_import_job(job_key: str, settings: Settings) -> dict[str, Any]:
    with db_write_connection(settings) as conn:
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
    job_ctx = PipelineJob(settings, job_key)
    with db_read_connection(settings) as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE key = ? AND type = 'import'",
            (job_key,),
        ).fetchone()
        if row is None:
            raise ValueError("import job not found")
        job = row_to_dict(row)
        if job["status"] != "queued":
            raise ValueError("import job is not queued")
        if job_progress(job).get("cancel_requested"):
            return job_ctx.finish("failed", {"canceled": True}, "Canceled by admin")
        payload = json.loads(job["payload"])
        path = Path(payload["path"])
        mode = _validate_import_mode(str(payload["mode"]))

    db_exec(
        settings,
        "UPDATE jobs SET status = 'running', progress = ?, updated_at = ? WHERE key = ?",
        (json_dumps({"phase": "reading"}), utc_now(), job_key),
    )

    rows, errors = read_import_jsonl(path, settings)
    if errors:
        job_ctx.finish("failed", {"errors": errors}, "upload validation failed")
        raise ValueError("upload validation failed")

    stats = {"total": len(rows), "new": 0, "overwrite": 0, "skip": 0, "disabled": 0}
    row_keys_by_source: dict[str, set[str]] = {}

    for index, row in enumerate(rows, start=1):
        if canceled := job_ctx.finish_if_canceled({"processed": index - 1, "total": len(rows)}):
            return canceled
        row_keys_by_source.setdefault(row.source_key, set()).add(row.problem_key)
        now = utc_now()
        problem_text_key = text_key(row.text)
        with db_write_connection(settings) as conn:
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
                          status, error, updated_at
                        )
                        VALUES (?, 'problem_text', NULL, NULL, NULL, ?, NULL, NULL,
                                'succeeded', NULL, ?)
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
            job_ctx.progress({"phase": "importing", "processed": index, "total": len(rows)})

    if mode == "sync_source":
        for source_key, imported_keys in row_keys_by_source.items():
            placeholders = ",".join("?" for _ in imported_keys)
            params = [source_key, *sorted(imported_keys)]
            with db_write_connection(settings) as conn:
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

    path.unlink(missing_ok=True)
    return job_ctx.finish("succeeded", stats, None)


def ensure_rewrite_artifact(
    settings: Settings,
    problem_text_key: str,
    selected_method_key: str | None = None,
) -> dict[str, Any]:
    selected_method_key = selected_method_key or rewrite_method_key(settings)
    selected_rewrite_key = rewrite_key(problem_text_key, selected_method_key)

    with db_read_connection(settings) as conn:
        source = conn.execute(
            "SELECT text FROM artifacts WHERE key = ? AND kind = 'problem_text'",
            (problem_text_key,),
        ).fetchone()
        if source is None:
            raise ValueError("problem text artifact not found")
        existing = conn.execute(
            """
            SELECT * FROM artifacts
            WHERE key = ? AND kind = 'rewrite' AND status = 'succeeded'
            """,
            (selected_rewrite_key,),
        ).fetchone()
        if existing is not None:
            return row_to_dict(existing)

    with db_write_connection(settings) as conn:
        now = utc_now()
        with conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO artifacts(
                  key, kind, parent_key, method_key, role, text, data, blob,
                  status, error, updated_at
                )
                VALUES (?, 'rewrite', ?, ?, 'rewrite', NULL, NULL, NULL,
                        'pending', NULL, ?)
                """,
                (selected_rewrite_key, problem_text_key, selected_method_key, now),
            )
            artifact = row_to_dict(
                conn.execute("SELECT * FROM artifacts WHERE key = ?", (selected_rewrite_key,)).fetchone()
            )
            if artifact["status"] == "succeeded":
                return artifact
            conn.execute(
                """
                UPDATE artifacts
                SET status = 'running', error = NULL, updated_at = ?
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
        with db_write_connection(settings) as conn:
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
    bulk = ensure_embedding_artifacts_bulk(settings, [selected_rewrite_key], selected_method_key)
    return bulk[selected_rewrite_key]


def ensure_embedding_artifacts_bulk(
    settings: Settings,
    selected_rewrite_keys: list[str],
    selected_method_key: str | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    selected_method_key = selected_method_key or embedding_method_key(settings)
    artifacts, pending_items = _prepare_embedding_work(
        settings,
        selected_rewrite_keys,
        selected_method_key,
    )
    if not pending_items:
        return artifacts

    batch_size = max(1, settings.jobs.embedding_batch_size)
    for start in range(0, len(pending_items), batch_size):
        batch = pending_items[start : start + batch_size]
        batch_keys = [item.artifact_key for item in batch]
        _mark_artifacts_running(settings, batch_keys)
        try:
            vectors = embed_texts(
                settings.embedding_model,
                [item.text for item in batch],
                timeout=settings.request_timeout,
            )
            matrix = normalize_matrix(np.asarray(vectors, dtype=np.float32))
            _store_embedding_items_batch(settings, batch, matrix)
        except Exception as exc:
            for key in batch_keys:
                _mark_artifact_failed(settings, key, str(exc))
            raise
    return _load_embedding_artifacts(settings, selected_rewrite_keys, selected_method_key)


def _prepare_embedding_work(
    settings: Settings,
    selected_rewrite_keys: list[str],
    selected_method_key: str,
) -> tuple[dict[str, dict[str, dict[str, Any]]], list[EmbeddingWorkItem]]:
    unique_rewrite_keys = list(dict.fromkeys(selected_rewrite_keys))
    view_texts_by_rewrite: dict[str, dict[str, str]] = {}
    now = utc_now()

    with db_write_connection(settings) as conn:
        with conn:
            for selected_rewrite_key in unique_rewrite_keys:
                rewrite_row = conn.execute(
                    """
                    SELECT data
                    FROM artifacts
                    WHERE key = ? AND kind = 'rewrite' AND status = 'succeeded'
                    """,
                    (selected_rewrite_key,),
                ).fetchone()
                if rewrite_row is None:
                    raise ValueError(f"succeeded rewrite artifact not found: {selected_rewrite_key}")
                rewrite_data = json.loads(rewrite_row["data"])
                view_texts = {view: str(rewrite_data.get(view) or "") for view in VIEWS}
                if not all(view_texts.values()):
                    raise ValueError(f"rewrite artifact does not contain all views: {selected_rewrite_key}")
                view_texts_by_rewrite[selected_rewrite_key] = view_texts
                for view in VIEWS:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO artifacts(
                          key, kind, parent_key, method_key, role, text, data, blob,
                          status, error, updated_at
                        )
                        VALUES (?, 'embedding', ?, ?, ?, NULL, NULL, NULL,
                                'pending', NULL, ?)
                        """,
                        (
                            embedding_key(selected_rewrite_key, selected_method_key, view),
                            selected_rewrite_key,
                            selected_method_key,
                            view,
                            now,
                        ),
                    )

        artifacts = _load_embedding_artifacts_from_conn(
            conn,
            unique_rewrite_keys,
            selected_method_key,
        )

    pending_items: list[EmbeddingWorkItem] = []
    for selected_rewrite_key in unique_rewrite_keys:
        rewrite_artifacts = artifacts.get(selected_rewrite_key, {})
        if set(rewrite_artifacts) != set(VIEWS):
            raise ValueError(f"embedding artifacts are incomplete: {selected_rewrite_key}")
        for view in VIEWS:
            artifact = rewrite_artifacts[view]
            if artifact["status"] != "succeeded":
                pending_items.append(
                    EmbeddingWorkItem(
                        rewrite_key=selected_rewrite_key,
                        artifact_key=artifact["key"],
                        view=view,
                        text=view_texts_by_rewrite[selected_rewrite_key][view],
                    )
                )
    return artifacts, pending_items


def _load_embedding_artifacts(
    settings: Settings,
    selected_rewrite_keys: list[str],
    selected_method_key: str,
) -> dict[str, dict[str, dict[str, Any]]]:
    unique_rewrite_keys = list(dict.fromkeys(selected_rewrite_keys))
    with db_read_connection(settings) as conn:
        return _load_embedding_artifacts_from_conn(conn, unique_rewrite_keys, selected_method_key)


def _load_embedding_artifacts_from_conn(
    conn: Any,
    selected_rewrite_keys: list[str],
    selected_method_key: str,
) -> dict[str, dict[str, dict[str, Any]]]:
    artifacts: dict[str, dict[str, dict[str, Any]]] = {
        selected_rewrite_key: {} for selected_rewrite_key in selected_rewrite_keys
    }
    for selected_rewrite_key in selected_rewrite_keys:
        rows = conn.execute(
            """
            SELECT * FROM artifacts
            WHERE kind = 'embedding'
              AND parent_key = ?
              AND method_key = ?
            """,
            (selected_rewrite_key, selected_method_key),
        ).fetchall()
        artifacts[selected_rewrite_key] = {row["role"]: row_to_dict(row) for row in rows}
    return artifacts


def get_artifact(settings: Settings, artifact_key: str) -> dict[str, Any] | None:
    return db_one(settings, "SELECT * FROM artifacts WHERE key = ?", (artifact_key,))


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
    db_exec(
        settings,
        "UPDATE artifacts SET status = 'failed', error = ?, updated_at = ? WHERE key = ?",
        (error[:4000], utc_now(), artifact_key),
    )


def _mark_artifacts_running(settings: Settings, artifact_keys: list[str]) -> None:
    if not artifact_keys:
        return
    placeholders = ",".join("?" for _ in artifact_keys)
    db_exec(
        settings,
        f"UPDATE artifacts SET status = 'running', error = NULL, updated_at = ? WHERE key IN ({placeholders})",
        [utc_now(), *artifact_keys],
    )


def _store_embedding_items_batch(
    settings: Settings,
    items: list[EmbeddingWorkItem],
    matrix: np.ndarray,
) -> None:
    now = utc_now()
    with db_write_connection(settings) as conn:
        with conn:
            for item, vector in zip(items, matrix, strict=True):
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
                        item.artifact_key,
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
        "rewrite_method": _model_snapshot(settings, "rewrite"),
        "embedding_method": _model_snapshot(settings, "embedding"),
        "problem_count": len(snapshot),
    }
    with db_write_connection(settings) as conn:
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


def _record_problem_failure(
    failures: list[dict[str, Any]],
    failed_problem_keys: set[str],
    problem: dict[str, Any],
    phase: str,
    error: str,
) -> None:
    problem_key = str(problem["key"])
    if problem_key in failed_problem_keys:
        return
    failures.append({"problem_key": problem_key, "phase": phase, "error": error})
    failed_problem_keys.add(problem_key)


def _group_snapshot_by_text_key(snapshot: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for problem in snapshot:
        grouped.setdefault(str(problem["text_key"]), []).append(problem)
    return grouped


def _chunks(items: list[str], size: int = 800) -> list[list[str]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _load_succeeded_rewrites_for_text_keys(
    settings: Settings,
    text_keys: list[str],
    selected_method_key: str,
) -> dict[str, dict[str, Any]]:
    if not text_keys:
        return {}

    rewrites: dict[str, dict[str, Any]] = {}
    with db_read_connection(settings) as conn:
        for chunk in _chunks(text_keys):
            placeholders = ",".join("?" for _ in chunk)
            rows = conn.execute(
                f"""
                SELECT *
                FROM artifacts
                WHERE kind = 'rewrite'
                  AND method_key = ?
                  AND status = 'succeeded'
                  AND parent_key IN ({placeholders})
                """,
                [selected_method_key, *chunk],
            ).fetchall()
            for row in rows:
                artifact = row_to_dict(row)
                rewrites[str(artifact["parent_key"])] = artifact
    return rewrites


def _problem_count_for_text_keys(
    text_keys: set[str],
    grouped_by_text_key: dict[str, list[dict[str, Any]]],
) -> int:
    return sum(len(grouped_by_text_key.get(text_key, [])) for text_key in text_keys)


def _ensure_rewrites_for_snapshot(
    settings: Settings,
    job_key: str,
    snapshot: list[dict[str, Any]],
    selected_method_key: str,
    failures: list[dict[str, Any]],
    failed_problem_keys: set[str],
) -> dict[str, dict[str, Any]]:
    job_ctx = PipelineJob(settings, job_key)
    grouped = _group_snapshot_by_text_key(snapshot)
    text_keys = list(grouped)
    rewrite_by_text_key = _load_succeeded_rewrites_for_text_keys(
        settings,
        text_keys,
        selected_method_key,
    )
    items = [(text_key, problems) for text_key, problems in grouped.items() if text_key not in rewrite_by_text_key]
    processed = _problem_count_for_text_keys(set(rewrite_by_text_key), grouped)
    if not text_keys:
        return rewrite_by_text_key

    max_workers = max(1, min(settings.jobs.rewrite_concurrency, max(1, len(items))))
    job_ctx.info(
        "Rewrite phase started",
        {
            "texts": len(text_keys),
            "cached_texts": len(rewrite_by_text_key),
            "pending_texts": len(items),
            "problems": len(snapshot),
            "concurrency": max_workers if items else 0,
        },
    )
    job_ctx.progress({"phase": "rewrite", "processed": processed, "total": len(snapshot), "failures": 0})
    if not items:
        return rewrite_by_text_key

    def rewrite_item(item: tuple[str, list[dict[str, Any]]]) -> dict[str, Any]:
        text_key, _ = item
        return ensure_rewrite_artifact(settings, text_key, selected_method_key)

    for _, (text_key, problems), rewrite, error in batch_process(
        items,
        rewrite_item,
        max_workers=max_workers,
        cancel=job_ctx.cancel_requested,
    ):
        if error is None and rewrite is not None:
            rewrite_by_text_key[text_key] = rewrite
        else:
            message = str(error)
            for problem in problems:
                _record_problem_failure(failures, failed_problem_keys, problem, "rewrite", message)
            job_ctx.error(
                "Rewrite failed",
                {
                    "problem_keys": [problem["key"] for problem in problems],
                    "error": message,
                },
            )

        processed += len(problems)
        job_ctx.progress(
            {
                "phase": "rewrite",
                "processed": processed,
                "total": len(snapshot),
                "failures": len(failures),
            },
            {"failures": failures} if failures else None,
        )

    return rewrite_by_text_key


def execute_build_index_job(job_key: str, settings: Settings) -> dict[str, Any]:
    job_ctx = PipelineJob(settings, job_key)
    with db_read_connection(settings) as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE key = ? AND type = 'build_index'",
            (job_key,),
        ).fetchone()
        if row is None:
            raise ValueError("build job not found")
        job = row_to_dict(row)
        if job["status"] != "queued":
            raise ValueError("build job is not queued")
        if job_progress(job).get("cancel_requested"):
            return job_ctx.finish("failed", {"canceled": True}, "Canceled by admin")
        payload = json.loads(job["payload"])

    db_exec(
        settings,
        "UPDATE jobs SET status = 'running', progress = ?, result = NULL, error = NULL, updated_at = ? "
        "WHERE key = ?",
        (json_dumps({"phase": "rewrite"}), utc_now(), job_key),
    )

    snapshot = read_jsonl_list(Path(payload["snapshot_path"]))
    job_ctx.info("Build index started", {"total": len(snapshot)})
    selected_rewrite_method_key = str(payload["rewrite_method_key"])
    selected_embedding_method_key = str(payload["embedding_method_key"])
    selected_rewrite_method = payload["rewrite_method"]
    selected_embedding_method = payload["embedding_method"]
    prepared: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    failed_problem_keys: set[str] = set()

    rewrite_by_text_key = _ensure_rewrites_for_snapshot(
        settings,
        job_key,
        snapshot,
        selected_rewrite_method_key,
        failures,
        failed_problem_keys,
    )
    current_job = get_job(settings, job_key) or {}
    current_progress = job_progress(current_job)
    if canceled := job_ctx.finish_if_canceled(
        {
            "phase": "rewrite",
            "failures": failures,
            "processed": int(current_progress.get("processed") or 0),
            "total": len(snapshot),
        },
    ):
        return canceled

    if failures:
        job_ctx.warning(
            "Build index blocked after rewrite",
            {"failures": len(failures), "total": len(snapshot)},
        )
        return job_ctx.finish("blocked", {"failures": failures}, None)

    rewrite_keys = list(
        dict.fromkeys(rewrite_by_text_key[problem["text_key"]]["key"] for problem in snapshot)
    )
    problems_by_rewrite_key: dict[str, list[dict[str, Any]]] = {}
    for problem in snapshot:
        rewrite_key_value = rewrite_by_text_key[problem["text_key"]]["key"]
        problems_by_rewrite_key.setdefault(rewrite_key_value, []).append(problem)

    embedding_artifacts, pending_items = _prepare_embedding_work(
        settings,
        rewrite_keys,
        selected_embedding_method_key,
    )
    embedding_counts = {
        rewrite_key_value: sum(
            1
            for artifact in artifacts.values()
            if artifact["status"] == "succeeded"
        )
        for rewrite_key_value, artifacts in embedding_artifacts.items()
    }
    completed_rewrite_keys = {
        rewrite_key_value
        for rewrite_key_value, count in embedding_counts.items()
        if count == len(VIEWS)
    }
    job_ctx.info(
        "Embedding phase started",
        {
            "pending_embeddings": len(pending_items),
            "batch_size": max(1, settings.jobs.embedding_batch_size),
            "concurrency": max(1, settings.jobs.embedding_concurrency),
        },
    )
    job_ctx.progress(
        {
            "phase": "embedding",
            "processed": min(
                len(snapshot),
                sum(len(problems_by_rewrite_key.get(key, [])) for key in completed_rewrite_keys)
                + len(failed_problem_keys),
            ),
            "total": len(snapshot),
            "failures": 0,
            "processed_embeddings": 0,
            "total_embeddings": len(pending_items),
        },
    )

    processed_embeddings = 0
    batch_size = max(1, settings.jobs.embedding_batch_size)
    batches = [pending_items[start : start + batch_size] for start in range(0, len(pending_items), batch_size)]
    max_workers = max(1, min(settings.jobs.embedding_concurrency, len(batches)))

    def embed_batch(batch: list[EmbeddingWorkItem]) -> list[EmbeddingWorkItem]:
        batch_keys = [item.artifact_key for item in batch]
        _mark_artifacts_running(settings, batch_keys)
        vectors = embed_texts(
            settings.embedding_model,
            [item.text for item in batch],
            timeout=settings.request_timeout,
        )
        matrix = normalize_matrix(np.asarray(vectors, dtype=np.float32))
        _store_embedding_items_batch(settings, batch, matrix)
        return batch

    for _, batch, completed_batch, error in batch_process(
        batches,
        embed_batch,
        max_workers=max_workers,
        cancel=job_ctx.cancel_requested,
    ):
        if error is not None:
            message = str(error)
            batch_keys = [item.artifact_key for item in batch]
            for key in batch_keys:
                _mark_artifact_failed(settings, key, message)
            affected_rewrite_keys = sorted({item.rewrite_key for item in batch})
            for rewrite_key_value in affected_rewrite_keys:
                for problem in problems_by_rewrite_key.get(rewrite_key_value, []):
                    _record_problem_failure(failures, failed_problem_keys, problem, "embedding", message)
            job_ctx.error(
                "Embedding batch failed",
                {
                    "rewrite_keys": affected_rewrite_keys,
                    "artifacts": len(batch),
                    "error": message,
                },
            )
        else:
            for item in completed_batch or []:
                embedding_counts[item.rewrite_key] = embedding_counts.get(item.rewrite_key, 0) + 1
                if embedding_counts[item.rewrite_key] == len(VIEWS):
                    completed_rewrite_keys.add(item.rewrite_key)

        processed_embeddings += len(batch)
        job_ctx.progress(
            {
                "phase": "embedding",
                "processed": min(
                    len(snapshot),
                    sum(len(problems_by_rewrite_key.get(key, [])) for key in completed_rewrite_keys)
                    + len(failed_problem_keys),
                ),
                "total": len(snapshot),
                "failures": len(failures),
                "processed_embeddings": processed_embeddings,
                "total_embeddings": len(pending_items),
            },
            {"failures": failures} if failures else None,
        )

    canceled = job_ctx.finish_if_canceled(
        {
            "phase": "embedding",
            "failures": failures,
            "processed": min(
                len(snapshot),
                sum(len(problems_by_rewrite_key.get(key, [])) for key in completed_rewrite_keys)
                + len(failed_problem_keys),
            ),
            "total": len(snapshot),
            "processed_embeddings": processed_embeddings,
            "total_embeddings": len(pending_items),
        },
    )
    if canceled:
        return canceled

    embedding_artifacts = _load_embedding_artifacts(
        settings,
        rewrite_keys,
        selected_embedding_method_key,
    )
    for problem in snapshot:
        rewrite = rewrite_by_text_key.get(problem["text_key"])
        if rewrite is None:
            _record_problem_failure(
                failures, failed_problem_keys, problem, "rewrite", "rewrite artifact was not prepared"
            )
            continue
        embeddings = embedding_artifacts.get(rewrite["key"], {})
        failed_embeddings = [
            view
            for view in VIEWS
            if embeddings.get(view, {}).get("status") != "succeeded"
        ]
        if failed_embeddings:
            _record_problem_failure(
                failures,
                failed_problem_keys,
                problem,
                "embedding",
                f"embedding failed for views: {', '.join(failed_embeddings)}",
            )
            continue
        prepared.append({"problem": problem, "rewrite": rewrite, "embeddings": embeddings})

    if failures:
        job_ctx.warning(
            "Build index blocked",
            {"failures": len(failures), "total": len(snapshot)},
        )
        return job_ctx.finish("blocked", {"failures": failures}, None)

    rows, row_hashes = _build_index_rows(prepared)
    selected_index_key = index_key(
        row_hashes,
        selected_rewrite_method_key,
        selected_embedding_method_key,
    )
    cache_path = index_cache_path(settings, selected_index_key)
    now = utc_now()
    with db_write_connection(settings) as conn:
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
                            "rewrite_method": selected_rewrite_method,
                            "embedding_method": selected_embedding_method,
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
        with db_write_connection(settings) as conn:
            with conn:
                conn.execute(
                    """
                    UPDATE indexes
                    SET status = 'built', error = NULL
                    WHERE key = ?
                    """,
                    (selected_index_key,),
                )
        job_ctx.info(
            "Build index succeeded",
            {"index_key": selected_index_key, "cache_path": str(cache_path)},
        )
        return job_ctx.finish(
            "succeeded",
            {"index_key": selected_index_key, "cache_path": str(cache_path)},
            None,
        )
    except Exception as exc:
        with db_write_connection(settings) as conn:
            with conn:
                conn.execute(
                    "UPDATE indexes SET status = 'failed', error = ? WHERE key = ?",
                    (str(exc), selected_index_key),
                )
        job_ctx.error(
            "Index export failed",
            {"index_key": selected_index_key, "error": str(exc)},
        )
        return job_ctx.finish(
            "failed",
            {"index_key": selected_index_key},
            str(exc),
        )


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
    with db_write_connection(settings) as conn:
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
    with db_read_connection(settings) as conn:
        rows = conn.execute(
            """
            SELECT key, title, url, text_key
            FROM problems
            WHERE enabled = 1 AND deleted = 0
            ORDER BY key
            """
        ).fetchall()
    return [row_to_dict(row) for row in rows]


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
    with db_read_connection(settings) as conn:
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
