from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from core import db_connection, ensure_database, get_settings, text_key, utc_now
from pipeline import ensure_embedding_artifacts, ensure_rewrite_artifact
from search import RewriteResult


def _temp_settings(tmp_path: Path):
    base_settings = get_settings()
    storage = replace(
        base_settings.storage,
        db_path=tmp_path / "app.sqlite3",
        upload_dir=tmp_path / "uploads",
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
                  key, kind, text, status, attempts, updated_at
                )
                VALUES (?, 'problem_text', ?, 'succeeded', 0, ?)
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
