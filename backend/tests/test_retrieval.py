from __future__ import annotations

from pathlib import Path

from backend.config import get_settings
from backend.models import Candidate
from backend.retrieval import SearchIndex, extract_title, fuse_scores


def test_settings_and_index_load() -> None:
    settings = get_settings()
    index = SearchIndex(settings.data_dir)

    assert Path(settings.data_dir).exists()
    assert index.corpus_count > 0
    assert index.embedding_shape[0] > 0
    assert index.embedding_shape[1] == 4096


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
    assert fused[0].final_score == 0.8

