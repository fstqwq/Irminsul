from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewriteResult:
    statement: str
    abstract: str
    raw: str


@dataclass(frozen=True)
class Candidate:
    problem_id: str
    title: str
    url: str
    original_text: str
    statement: str
    abstract: str
    embedding_score: float
    rerank_score: float | None = None
    final_score: float | None = None

