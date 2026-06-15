from __future__ import annotations

import json
import os
import gc
import shutil
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from core import db_connection, ensure_database, get_settings, text_key, utc_now
from pipeline import (
    create_build_index_job,
    create_cleanup_job,
    ensure_embedding_artifacts,
    ensure_rewrite_artifact,
    execute_build_index_job,
    execute_cleanup_job,
    index_cache_path,
    rebuild_index_cache,
)
from search import IndexState, RewriteResult, load_index_cache


def _temp_settings(tmp_path: Path):
    base_settings = get_settings()
    storage = replace(
        base_settings.storage,
        db_path=tmp_path / "app.sqlite3",
        upload_dir=tmp_path / "uploads",
        index_cache_dir=tmp_path / "index_cache",
    )
    return replace(base_settings, storage=storage)


def test_rewrite_and_embedding_artifacts_are_reused(monkeypatch, tmp_path: Path) -> None:
    settings = _temp_settings(tmp_path)
    ensure_database(settings)
    problem_text = "Original statement with details."
    problem_text_key = text_key(problem_text)

    with db_connection(settings) as conn:
        with conn:
            conn.execute(
                """
                INSERT INTO artifacts(
                  key, kind, text, status, updated_at
                )
                VALUES (?, 'problem_text', ?, 'succeeded', ?)
                """,
                (problem_text_key, problem_text, utc_now()),
            )

    rewrite_calls = 0

    def fake_rewrite(*args, **kwargs) -> RewriteResult:
        nonlocal rewrite_calls
        rewrite_calls += 1
        return RewriteResult(
            statement="Statement view",
            abstract="Abstract view",
            abstract_zh="Chinese abstract",
            clean="Clean view",
            raw="raw",
        )

    embed_calls: list[list[str]] = []

    def fake_embed(*args, **kwargs) -> np.ndarray:
        texts = args[1]
        embed_calls.append(list(texts))
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )[: len(texts)]

    monkeypatch.setattr("pipeline.rewrite_query", fake_rewrite)
    monkeypatch.setattr("pipeline.embed_texts", fake_embed)

    rewrite = ensure_rewrite_artifact(settings, problem_text_key)
    rewrite_again = ensure_rewrite_artifact(settings, problem_text_key)
    rewrite_data = json.loads(rewrite["data"])

    assert rewrite_calls == 1
    assert rewrite_again["key"] == rewrite["key"]
    assert rewrite_data["clean"] == "Clean view"
    assert rewrite_data["statement"] == "Statement view"
    assert rewrite_data["abstract_zh"] == "Chinese abstract"

    embeddings = ensure_embedding_artifacts(settings, rewrite["key"])
    embeddings_again = ensure_embedding_artifacts(settings, rewrite["key"])

    assert len(embed_calls) == 1
    assert sorted(embeddings) == ["abstract", "abstract_zh", "clean", "statement"]
    assert embeddings_again["clean"]["key"] == embeddings["clean"]["key"]
    for artifact in embeddings.values():
        data = json.loads(artifact["data"])
        assert artifact["status"] == "succeeded"
        assert data["dim"] == 3
        assert len(artifact["blob"]) == 12


def test_build_index_exports_cache(monkeypatch, tmp_path: Path) -> None:
    settings = _temp_settings(tmp_path)
    ensure_database(settings)
    problem_text = "Build this problem."
    problem_text_key = text_key(problem_text)

    with db_connection(settings) as conn:
        with conn:
            conn.execute(
                "INSERT INTO sources(key, name, updated_at) VALUES ('CF', 'CF', ?)",
                (utc_now(),),
            )
            conn.execute(
                """
                INSERT INTO artifacts(key, kind, text, status, updated_at)
                VALUES (?, 'problem_text', ?, 'succeeded', ?)
                """,
                (problem_text_key, problem_text, utc_now()),
            )
            conn.execute(
                """
                INSERT INTO problems(key, source_key, title, url, text_key, updated_at)
                VALUES ('CF/1A', 'CF', 'Build Title', 'https://example.com/1A', ?, ?)
                """,
                (problem_text_key, utc_now()),
            )

    def fake_rewrite(*args, **kwargs) -> RewriteResult:
        return RewriteResult(
            statement="Statement view",
            abstract="Abstract view",
            abstract_zh="Chinese abstract",
            clean="Clean view",
            raw="raw",
        )

    def fake_embed(*args, **kwargs) -> np.ndarray:
        texts = args[1]
        return np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [0.5, 0.5],
            ],
            dtype=np.float32,
        )[: len(texts)]

    monkeypatch.setattr("pipeline.rewrite_query", fake_rewrite)
    monkeypatch.setattr("pipeline.embed_texts", fake_embed)

    job = create_build_index_job(settings)
    finished = execute_build_index_job(job["key"], settings)
    index_key = json.loads(finished["result"])["index_key"]
    cache_path = index_cache_path(settings, index_key)

    assert finished["status"] == "succeeded"
    assert (cache_path / "manifest.json").exists()
    assert np.load(cache_path / "clean.npy").shape == (1, 2)
    loaded = load_index_cache(cache_path, "mmap")
    state = IndexState()
    state.activate(loaded, 1)
    with state.search_snapshot() as snapshot:
        assert snapshot.key == index_key
        assert snapshot.problem_count == 1
    state.current = None
    del snapshot
    del loaded
    gc.collect()

    with db_connection(settings) as conn:
        index = conn.execute("SELECT * FROM indexes WHERE key = ?", (index_key,)).fetchone()
        row_count = conn.execute(
            "SELECT count(*) FROM index_rows WHERE index_key = ?",
            (index_key,),
        ).fetchone()[0]

    assert index["status"] == "built"
    assert row_count == 4

    shutil.rmtree(cache_path)
    rebuilt = rebuild_index_cache(settings, index_key)
    assert rebuilt["status"] == "built"
    assert np.load(cache_path / "statement.npy", mmap_mode="r").shape == (1, 2)


def test_cleanup_removes_expired_operational_files(tmp_path: Path) -> None:
    base_settings = _temp_settings(tmp_path)
    settings = replace(
        base_settings,
        audit=replace(base_settings.audit, retention_days=1),
        index_cache=replace(base_settings.index_cache, keep_retired=1),
    )
    ensure_database(settings)
    settings.storage.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.storage.index_cache_dir.mkdir(parents=True, exist_ok=True)

    old_upload = settings.storage.upload_dir / "old.jsonl"
    protected_upload = settings.storage.upload_dir / "protected.jsonl"
    old_upload.write_text("{}", encoding="utf-8")
    protected_upload.write_text("{}", encoding="utf-8")
    old_timestamp = (datetime.now(UTC) - timedelta(days=2)).timestamp()
    os.utime(old_upload, (old_timestamp, old_timestamp))
    os.utime(protected_upload, (old_timestamp, old_timestamp))

    active_cache = index_cache_path(settings, "i:active")
    recent_cache = index_cache_path(settings, "i:recent")
    stale_cache = index_cache_path(settings, "i:stale")
    building_cache = settings.storage.index_cache_dir / ".building" / "leftover"
    for path in (active_cache, recent_cache, stale_cache, building_cache):
        path.mkdir(parents=True, exist_ok=True)
        (path / "marker").write_text("x", encoding="utf-8")

    now = utc_now()
    with db_connection(settings) as conn:
        with conn:
            conn.execute(
                """
                INSERT INTO search_audits(
                  request_id, started_at, finished_at, status, client_ip, user_agent,
                  query, timings, api_calls, result, cost, error
                )
                VALUES ('old-audit', '2000-01-01T00:00:00Z', '2000-01-01T00:00:00Z',
                        'succeeded', '', '', 'q', '{}', '[]', '{}', '{"microusd":0}', NULL)
                """
            )
            conn.execute(
                "INSERT INTO kv(key, value, updated_at) VALUES ('active_index_key', 'i:active', ?)",
                (now,),
            )
            conn.executemany(
                """
                INSERT INTO indexes(key, status, meta, created_at, activated_at)
                VALUES (?, ?, '{}', ?, ?)
                """,
                [
                    ("i:active", "active", "2026-01-03T00:00:00Z", "2026-01-03T00:00:00Z"),
                    ("i:recent", "built", "2026-01-02T00:00:00Z", None),
                    ("i:stale", "retired", "2026-01-01T00:00:00Z", None),
                ],
            )
            conn.execute(
                """
                INSERT INTO jobs(key, type, status, payload, progress, created_at, updated_at)
                VALUES ('j:protected', 'import', 'draft', ?, '{}', ?, ?)
                """,
                (json.dumps({"path": str(protected_upload)}), now, now),
            )

    job = create_cleanup_job(settings)
    finished = execute_cleanup_job(job["key"], settings)
    result = json.loads(finished["result"])

    assert result["removed_audits"] == 1
    assert result["removed_uploads"] == 1
    assert result["removed_building_dirs"] == 1
    assert result["removed_index_caches"] == 1
    assert not old_upload.exists()
    assert protected_upload.exists()
    assert active_cache.exists()
    assert recent_cache.exists()
    assert not stale_cache.exists()
    assert not (settings.storage.index_cache_dir / ".building").exists()
