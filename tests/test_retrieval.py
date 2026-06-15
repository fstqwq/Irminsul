from __future__ import annotations

from pathlib import Path

from core import ModelConfig, connect_db, get_settings, migrate
from search import (
    Candidate,
    VIEWS,
    extract_title,
    fuse_scores,
    rerank_candidate_window,
    rerank_documents_with_usage,
)


def test_settings_load() -> None:
    settings = get_settings()

    assert settings.storage.db_path.name == "app.sqlite3"
    assert settings.search.top_per_doc_view == 50
    assert VIEWS == ("clean", "statement", "abstract", "abstract_zh")
    assert (
        settings.audit.pricing["openrouter"]["Qwen/Qwen3-Embedding-8B"][
            "input_price_per_1m_tokens_microusd"
        ]
        == 10000
    )
    assert (
        settings.audit.pricing["deepinfra"]["Qwen/Qwen3-Reranker-8B"][
            "input_price_per_1m_tokens_microusd"
        ]
        == 50000
    )


def test_title_fallbacks() -> None:
    assert extract_title({"title": "  Two Sum  "}, "fallback") == "Two Sum"
    assert (
        extract_title({"text": "**title** : Legs on a Farm\nbody"}, "fallback")
        == "Legs on a Farm"
    )
    assert extract_title({"text": "## Problem Name\nP9064\nbody"}, "fallback") == "P9064"
    assert extract_title({"text": ""}, "fallback") == "fallback"


def test_fuse_scores() -> None:
    candidates = [
        Candidate("a", "A", "", "", "s", "a", 0.2, 1.0),
        Candidate("b", "B", "", "", "s", "a", 0.9, 0.0),
    ]

    fused = fuse_scores(candidates, beta=0.75)

    assert [candidate.problem_id for candidate in fused] == ["a", "b"]
    assert fused[0].final_score is not None
    assert fused[0].final_score > fused[1].final_score


def test_rerank_candidate_window_zero_means_all() -> None:
    candidates = [
        Candidate("a", "A", "", "", "s", "a", 0.9),
        Candidate("b", "B", "", "", "s", "a", 0.8),
        Candidate("c", "C", "", "", "s", "a", 0.7),
    ]

    assert rerank_candidate_window(candidates, 0) == candidates
    assert rerank_candidate_window(candidates, -1) == candidates


def test_rerank_candidate_window_positive_truncates() -> None:
    candidates = [
        Candidate("a", "A", "", "", "s", "a", 0.9),
        Candidate("b", "B", "", "", "s", "a", 0.8),
        Candidate("c", "C", "", "", "s", "a", 0.7),
    ]

    assert [candidate.problem_id for candidate in rerank_candidate_window(candidates, 2)] == ["a", "b"]


def test_reranker_usage_reads_deepinfra_top_level_tokens(monkeypatch) -> None:
    monkeypatch.setenv("RERANK_KEY", "secret")

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "scores": [0.8, 0.2],
                "input_tokens": 1200,
                "inference_status": {"tokens_input": 0},
            }

    def fake_post(*args, **kwargs) -> FakeResponse:
        return FakeResponse()

    monkeypatch.setattr("search.requests.post", fake_post)
    result = rerank_documents_with_usage(
        ModelConfig(
            name="rerank",
            model="Qwen/Qwen3-Reranker-8B",
            url="https://api.deepinfra.com/v1/inference/{model}",
            api_key_env="RERANK_KEY",
        ),
        "query",
        ["doc one", "doc two"],
    )

    assert result.scores == [0.8, 0.2]
    assert result.usage["input_tokens"] == 1200
    assert result.usage["pair_count"] == 2


def test_sqlite_migration(tmp_path: Path) -> None:
    db_path = tmp_path / "app.sqlite3"
    conn = connect_db(db_path)
    try:
        migrate(conn)
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
    finally:
        conn.close()

    assert version == 2
    assert {
        "sources",
        "problems",
        "artifacts",
        "indexes",
        "index_rows",
        "jobs",
        "job_logs",
        "search_audits",
        "kv",
    }.issubset(tables)
