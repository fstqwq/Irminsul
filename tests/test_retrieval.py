from __future__ import annotations

from pathlib import Path

from core import connect_db, get_settings, migrate
from search import Candidate, VIEWS, extract_title, fuse_scores


def test_settings_load() -> None:
    settings = get_settings()

    assert settings.storage.db_path.name == "app.sqlite3"
    assert settings.search.top_per_doc_view == 50
    assert VIEWS == ("clean", "statement", "abstract", "abstract_zh")


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

    assert version == 1
    assert {
        "sources",
        "problems",
        "artifacts",
        "indexes",
        "index_rows",
        "jobs",
        "search_audits",
        "kv",
    }.issubset(tables)
