from __future__ import annotations

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query_text: str = Field(min_length=1)
    use_rewrite: bool = True
    use_rerank: bool = True
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    beta: float = Field(default=0.75, ge=0.0, le=1.0)
    edited_statement: str | None = None
    edited_abstract: str | None = None


class RewritePayload(BaseModel):
    statement: str
    abstract: str
    raw: str


class CandidatePayload(BaseModel):
    problem_id: str
    title: str
    url: str
    original_text: str
    statement: str
    abstract: str
    embedding_score: float
    rerank_score: float | None = None
    final_score: float | None = None


class ConfigPayload(BaseModel):
    top_retrieval: int
    top_display: int
    default_alpha: float
    default_beta: float
    default_rerank: bool


class HealthPayload(BaseModel):
    ok: bool
    data_dir: str
    corpus_count: int
    embedding_shape: tuple[int, ...]

