from __future__ import annotations

import json
import gc
import shutil
import threading
from dataclasses import replace
from pathlib import Path

import numpy as np

from core import db_read_connection, db_write_connection, ensure_database, get_settings, text_key, utc_now
from pipeline import (
    create_build_index_job,
    embedding_method_key,
    ensure_embedding_artifacts,
    ensure_rewrite_artifact,
    execute_build_index_job,
    index_cache_path,
    rebuild_index_cache,
    rewrite_method_key,
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
