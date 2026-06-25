from __future__ import annotations

import json
import gc
import shutil
import threading
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from core import (
    db_exec,
    db_read_connection,
    db_write_connection,
    ensure_database,
    get_settings,
    list_job_logs,
    retry_job,
    text_key,
    utc_now,
)
from pipeline import (
    create_build_index_job,
    embedding_method_key,
    ensure_embedding_artifacts,
    ensure_rewrite_artifact,
    execute_build_index_job,
    index_cache_path,
    rebuild_index_cache,
    rewrite_method_key,
    run_next_job,
    source_key_from_problem_id,
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


def test_source_key_uses_only_slash_separator() -> None:
    assert source_key_from_problem_id("CODEFORCES_GYM/100199A") == "CODEFORCES_GYM"
    assert source_key_from_problem_id(" CODEFORCES/1A ") == "CODEFORCES"
    assert source_key_from_problem_id("CODEFORCES_GYM_100199A") == "unknown"


def test_method_keys_use_model_identity_not_provider(tmp_path: Path) -> None:
    settings = _temp_settings(tmp_path)
    direct_rewrite = replace(
        settings.rewrite_model,
        model="deepseek-v4-flash",
        identity="deepseek-v4-flash",
        url="https://api.deepseek.com/chat/completions",
        api_key_env="DEEPSEEK_API_KEY",
    )
    openrouter_rewrite = replace(
        settings.rewrite_model,
        model="deepseek/deepseek-v4-flash",
        identity="deepseek-v4-flash",
        url="https://openrouter.ai/api/v1/chat/completions",
        api_key_env="OPENROUTER_API_KEY",
    )
    changed_model = replace(openrouter_rewrite, identity="deepseek-v4-flash-next")

    assert rewrite_method_key(replace(settings, rewrite_model=direct_rewrite)) == rewrite_method_key(
        replace(settings, rewrite_model=openrouter_rewrite)
    )
    assert rewrite_method_key(replace(settings, rewrite_model=openrouter_rewrite)) != rewrite_method_key(
        replace(settings, rewrite_model=changed_model)
    )

    direct_embedding = replace(
        settings.embedding_model,
        model="Qwen/Qwen3-Embedding-8B",
        identity="Qwen/Qwen3-Embedding-8B",
        url="https://example.com/embeddings",
        api_key_env="EMBED_KEY_A",
    )
    routed_embedding = replace(
        settings.embedding_model,
        model="Qwen/Qwen3-Embedding-8B",
        identity="Qwen/Qwen3-Embedding-8B",
        url="https://openrouter.ai/api/v1/embeddings",
        api_key_env="OPENROUTER_API_KEY",
    )
    assert embedding_method_key(replace(settings, embedding_model=direct_embedding)) == embedding_method_key(
        replace(settings, embedding_model=routed_embedding)
    )


def test_rewrite_and_embedding_artifacts_are_reused(monkeypatch, tmp_path: Path) -> None:
    settings = _temp_settings(tmp_path)
    ensure_database(settings)
    problem_text = "Original statement with details."
    problem_text_key = text_key(problem_text)

    with db_write_connection(settings) as conn:
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


def test_run_next_job_claims_job_before_execution(monkeypatch, tmp_path: Path) -> None:
    settings = _temp_settings(tmp_path)
    ensure_database(settings)
    now = utc_now()
    with db_write_connection(settings) as conn:
        with conn:
            conn.execute(
                """
                INSERT INTO jobs(key, type, status, payload, progress, created_at, updated_at)
                VALUES ('j:claim', 'import', 'queued', '{}', '{"phase":"queued"}', ?, ?)
                """,
                (now, now),
            )

    calls: list[tuple[str, bool]] = []

    def fake_execute_import_job(job_key, current_settings, *, claimed=False):
        calls.append((job_key, claimed))
        with db_read_connection(current_settings) as conn:
            row = conn.execute("SELECT status, progress FROM jobs WHERE key = ?", (job_key,)).fetchone()
        assert row["status"] == "running"
        assert json.loads(row["progress"])["phase"] == "reading"
        return {}

    monkeypatch.setattr("pipeline.execute_import_job", fake_execute_import_job)

    assert run_next_job(settings) is True
    assert run_next_job(settings) is False
    assert calls == [("j:claim", True)]


def test_run_next_job_does_not_double_claim(monkeypatch, tmp_path: Path) -> None:
    settings = _temp_settings(tmp_path)
    ensure_database(settings)
    now = utc_now()
    with db_write_connection(settings) as conn:
        with conn:
            conn.execute(
                """
                INSERT INTO jobs(key, type, status, payload, progress, created_at, updated_at)
                VALUES ('j:single', 'import', 'queued', '{}', '{"phase":"queued"}', ?, ?)
                """,
                (now, now),
            )

    entered = threading.Event()
    release = threading.Event()
    calls: list[tuple[str, bool]] = []
    results: list[bool] = []

    def fake_execute_import_job(job_key, current_settings, *, claimed=False):
        del current_settings
        calls.append((job_key, claimed))
        entered.set()
        assert release.wait(2)
        return {}

    monkeypatch.setattr("pipeline.execute_import_job", fake_execute_import_job)

    threads = [threading.Thread(target=lambda: results.append(run_next_job(settings))) for _ in range(2)]
    for thread in threads:
        thread.start()
    assert entered.wait(2)
    release.set()
    for thread in threads:
        thread.join()

    assert sorted(results) == [False, True]
    assert calls == [("j:single", True)]


def test_retry_job_requires_compare_and_swap(monkeypatch, tmp_path: Path) -> None:
    settings = _temp_settings(tmp_path)
    ensure_database(settings)
    now = utc_now()
    with db_write_connection(settings) as conn:
        with conn:
            conn.execute(
                """
                INSERT INTO jobs(key, type, status, payload, progress, created_at, updated_at)
                VALUES ('j:retry', 'import', 'blocked', '{}', '{"phase":"blocked"}', ?, ?)
                """,
                (now, now),
            )

    def miss_retry_update(current_settings, sql, params=()):
        del sql, params
        with db_write_connection(current_settings) as conn:
            with conn:
                conn.execute("UPDATE jobs SET status = 'running' WHERE key = 'j:retry'")
        return 0

    monkeypatch.setattr("core.db_exec", miss_retry_update)

    with pytest.raises(ValueError, match="not retryable"):
        retry_job(settings, "j:retry")
    assert list_job_logs(settings, "j:retry") == []


def test_build_index_exports_cache(monkeypatch, tmp_path: Path) -> None:
    settings = _temp_settings(tmp_path)
    ensure_database(settings)
    problem_text = "Build this problem."
    problem_text_key = text_key(problem_text)

    with db_write_connection(settings) as conn:
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

    with db_read_connection(settings) as conn:
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

    def miss_rebuild_update(current_settings, sql, params=()):
        if sql.startswith("UPDATE indexes SET status = 'built'"):
            return 0
        return db_exec(current_settings, sql, params)

    monkeypatch.setattr("pipeline.db_exec", miss_rebuild_update)
    with pytest.raises(ValueError, match="state changed"):
        rebuild_index_cache(settings, index_key)


def test_build_index_rewrites_all_problems_before_embedding(monkeypatch, tmp_path: Path) -> None:
    settings = _temp_settings(tmp_path)
    ensure_database(settings)
    now = utc_now()
    problems = [
        ("CF/1A", "First problem text."),
        ("CF/2A", "Second problem text."),
    ]

    with db_write_connection(settings) as conn:
        with conn:
            conn.execute(
                "INSERT INTO sources(key, name, updated_at) VALUES ('CF', 'CF', ?)",
                (now,),
            )
            for problem_key, text in problems:
                problem_text_key = text_key(text)
                conn.execute(
                    """
                    INSERT INTO artifacts(key, kind, text, status, updated_at)
                    VALUES (?, 'problem_text', ?, 'succeeded', ?)
                    """,
                    (problem_text_key, text, now),
                )
                conn.execute(
                    """
                    INSERT INTO problems(key, source_key, title, url, text_key, updated_at)
                    VALUES (?, 'CF', ?, '', ?, ?)
                    """,
                    (problem_key, problem_key, problem_text_key, now),
                )

    lock = threading.Lock()
    rewrite_calls = 0
    embed_batches: list[int] = []

    def fake_rewrite(*args, **kwargs) -> RewriteResult:
        nonlocal rewrite_calls
        with lock:
            rewrite_calls += 1
        return RewriteResult(
            statement="Statement view",
            abstract="Abstract view",
            abstract_zh="Chinese abstract",
            clean="Clean view",
            raw="raw",
        )

    def fake_embed(*args, **kwargs) -> np.ndarray:
        texts = args[1]
        with lock:
            assert rewrite_calls == len(problems)
            embed_batches.append(len(texts))
        return np.array(
            [[1.0, float(index + 1)] for index in range(len(texts))],
            dtype=np.float32,
        )

    monkeypatch.setattr("pipeline.rewrite_query", fake_rewrite)
    monkeypatch.setattr("pipeline.embed_texts", fake_embed)

    job = create_build_index_job(settings)
    finished = execute_build_index_job(job["key"], settings)

    assert finished["status"] == "succeeded"
    assert rewrite_calls == len(problems)
    assert embed_batches == [len(problems) * 4]
