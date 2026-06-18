from __future__ import annotations

from pathlib import Path

from core import ModelConfig, connect_db, get_settings, migrate
from search import (
    Candidate,
    REWRITE_VIEW_MAX_CHARS,
    RewriteFormatError,
    VIEWS,
    fuse_scores,
    parse_rewrite_output,
    rewrite_query_with_usage,
    rerank_candidate_window,
    rerank_documents_with_usage,
)


def test_settings_load() -> None:
    settings = get_settings()

    assert settings.storage.db_path.name == "app.sqlite3"
    assert settings.audit.retention_days == 9999
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


def test_parse_rewrite_output_rejects_oversized_view() -> None:
    oversized = "x" * (REWRITE_VIEW_MAX_CHARS + 1)

    try:
        parse_rewrite_output(
            "\n".join(
                [
                    "[clean]",
                    "Clean",
                    "[statement]",
                    "Statement",
                    "[abstract]",
                    oversized,
                    "[abstract_zh]",
                    "摘要",
                ]
            )
        )
    except RewriteFormatError as exc:
        assert str(exc) == f"abstract_section_too_long:{REWRITE_VIEW_MAX_CHARS + 1}>{REWRITE_VIEW_MAX_CHARS}"
    else:
        raise AssertionError("oversized rewrite view was accepted")


def test_fuse_scores() -> None:
    candidates = [
        Candidate(
            problem_id="a",
            title="A",
            url="",
            clean="",
            statement="s",
            abstract="a",
            abstract_zh="z",
            embedding_score=0.2,
            rerank_score=1.0,
        ),
        Candidate(
            problem_id="b",
            title="B",
            url="",
            clean="",
            statement="s",
            abstract="a",
            abstract_zh="z",
            embedding_score=0.9,
            rerank_score=0.0,
        ),
    ]

    fused = fuse_scores(candidates, beta=0.75)

    assert [candidate.problem_id for candidate in fused] == ["a", "b"]
    assert fused[0].final_score is not None
    assert fused[0].final_score > fused[1].final_score


def test_fuse_scores_tiebreaks_problem_id_ascending() -> None:
    candidates = [
        Candidate("z", "Z", "", "", "s", "a", "z", 0.8, rerank_score=0.7),
        Candidate("a", "A", "", "", "s", "a", "z", 0.8, rerank_score=0.7),
    ]

    fused = fuse_scores(candidates, beta=0.75)

    assert [candidate.problem_id for candidate in fused] == ["a", "z"]


def test_fuse_scores_without_rerank_tiebreaks_problem_id_ascending() -> None:
    candidates = [
        Candidate("z", "Z", "", "", "s", "a", "z", 0.8),
        Candidate("a", "A", "", "", "s", "a", "z", 0.8),
    ]

    fused = fuse_scores(candidates, beta=0.75)

    assert [candidate.problem_id for candidate in fused] == ["a", "z"]


def test_rerank_candidate_window_zero_means_all() -> None:
    candidates = [
        Candidate("a", "A", "", "", "s", "a", "z", 0.9),
        Candidate("b", "B", "", "", "s", "a", "z", 0.8),
        Candidate("c", "C", "", "", "s", "a", "z", 0.7),
    ]

    assert rerank_candidate_window(candidates, 0) == candidates
    assert rerank_candidate_window(candidates, -1) == candidates


def test_rerank_candidate_window_positive_truncates() -> None:
    candidates = [
        Candidate("a", "A", "", "", "s", "a", "z", 0.9),
        Candidate("b", "B", "", "", "s", "a", "z", 0.8),
        Candidate("c", "C", "", "", "s", "a", "z", 0.7),
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


def test_rewrite_request_includes_provider_routing(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "choices": [
                    {
                        "message": {
                            "content": "[clean]\nClean\n\n[statement]\nStatement\n\n[abstract]\nAbstract\n\n[abstract_zh]\n摘要"
                        }
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }

    def fake_post(*args, **kwargs) -> FakeResponse:
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr("search.requests.post", fake_post)
    result = rewrite_query_with_usage(
        ModelConfig(
            name="rewrite",
            model="deepseek/deepseek-v4-flash",
            identity="deepseek-v4-flash",
            url="https://openrouter.ai/api/v1/chat/completions",
            api_key_env="OPENROUTER_API_KEY",
            provider={"order": ["baidu/fp8"], "allow_fallbacks": False},
        ),
        "query",
    )

    assert result.rewrite.statement == "Statement"
    assert captured["json"]["provider"] == {
        "order": ["baidu/fp8"],
        "allow_fallbacks": False,
    }


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
