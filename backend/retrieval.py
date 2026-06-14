from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .api_clients import embed_texts, normalize_matrix, rerank_documents
from .config import ModelConfig
from .models import Candidate


def read_jsonl_list(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def top_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.empty((0,), dtype=np.int64)
    if k >= scores.shape[0]:
        indices = np.arange(scores.shape[0])
    else:
        indices = np.argpartition(-scores, k - 1)[:k]
    return indices[np.argsort(-scores[indices])]


def _compact_title(title: str, fallback: str) -> str:
    title = " ".join(title.split()).strip()
    if not title:
        return fallback
    return title[:117] + "..." if len(title) > 120 else title


def extract_title(row: dict[str, Any] | None, fallback: str) -> str:
    if not row:
        return fallback
    explicit = row.get("title")
    if isinstance(explicit, str) and explicit.strip():
        return _compact_title(explicit, fallback)

    text = str(row.get("text") or "")
    patterns = (
        r"(?im)^\s*\*\*title\*\*\s*:\s*(.+?)\s*$",
        r"(?im)^\s*##\s*Problem Name\s*$\s*^(.+?)\s*$",
        r"(?im)^\s*#\s+(.+?)\s*$",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return _compact_title(match.group(1), fallback)

    for line in text.splitlines():
        line = line.strip()
        if line:
            return _compact_title(line, fallback)
    return fallback


class SearchIndex:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.corpus_by_id = {row["id"]: row for row in read_jsonl_list(data_dir / "corpus.jsonl")}
        self.rewritten_by_id = {
            row["id"]: row for row in read_jsonl_list(data_dir / "rewritten_corpus.jsonl")
        }
        self.documents = read_jsonl_list(data_dir / "documents.jsonl")
        self.embeddings = np.load(data_dir / "embeddings.npy")
        self._validate()
        self._fused_cache: dict[float, tuple[list[str], np.ndarray]] = {}

    def _validate(self) -> None:
        if len(self.documents) != self.embeddings.shape[0]:
            raise ValueError(
                "documents/embeddings row count mismatch: "
                f"{len(self.documents)} != {self.embeddings.shape[0]}"
            )
        for index, document in enumerate(self.documents):
            if document.get("row") != index:
                raise ValueError(f"documents row mismatch at index {index}")

    @property
    def corpus_count(self) -> int:
        return len(self.rewritten_by_id)

    @property
    def embedding_shape(self) -> tuple[int, ...]:
        return tuple(self.embeddings.shape)

    def title_for_id(self, problem_id: str) -> str:
        rewritten = self.rewritten_by_id.get(problem_id)
        original = self.corpus_by_id.get(problem_id)
        title = extract_title(rewritten, "")
        if title:
            return title
        return extract_title(original, problem_id)

    def fused_corpus(self, alpha: float) -> tuple[list[str], np.ndarray]:
        cache_key = round(float(alpha), 4)
        cached = self._fused_cache.get(cache_key)
        if cached is not None:
            return cached

        rows_by_id: dict[str, dict[str, int]] = defaultdict(dict)
        for document in self.documents:
            rows_by_id[document["id"]][document["kind"]] = int(document["row"])

        ids: list[str] = []
        vectors: list[np.ndarray] = []
        for problem_id, rows_by_kind in rows_by_id.items():
            if "statement" not in rows_by_kind or "abstract" not in rows_by_kind:
                continue
            vector = (
                cache_key * self.embeddings[rows_by_kind["statement"]]
                + (1.0 - cache_key) * self.embeddings[rows_by_kind["abstract"]]
            )
            ids.append(problem_id)
            vectors.append(vector)

        matrix = normalize_matrix(np.array(vectors, dtype=np.float32))
        self._fused_cache[cache_key] = (ids, matrix)
        return ids, matrix

    def retrieve(
        self,
        embedding_model: ModelConfig,
        statement: str,
        abstract: str | None,
        alpha: float,
        timeout: int,
        top_k: int,
        timings: dict[str, float] | None = None,
    ) -> list[Candidate]:
        query_vector = self.embed_query_vector(
            embedding_model,
            statement,
            abstract,
            alpha,
            timeout,
            timings=timings,
        )
        return self.search_with_vector(query_vector, alpha, top_k, timings=timings)

    def embed_query_vector(
        self,
        embedding_model: ModelConfig,
        statement: str,
        abstract: str | None,
        alpha: float,
        timeout: int,
        timings: dict[str, float] | None = None,
    ) -> np.ndarray:
        start = perf_counter()
        if abstract is None:
            query_embeddings = embed_texts(embedding_model, [statement], timeout=timeout)
            query_vector = query_embeddings[0]
            input_count = 1
        else:
            query_embeddings = embed_texts(embedding_model, [statement, abstract], timeout=timeout)
            query_vector = normalize_matrix(
                np.array(
                    [alpha * query_embeddings[0] + (1.0 - alpha) * query_embeddings[1]],
                    dtype=np.float32,
                )
            )[0]
            input_count = 2
        if timings is not None:
            timings["query_embedding"] = perf_counter() - start
            timings["query_embedding_inputs"] = float(input_count)
        return query_vector

    def search_with_vector(
        self,
        query_vector: np.ndarray,
        alpha: float,
        top_k: int,
        timings: dict[str, float] | None = None,
    ) -> list[Candidate]:
        start = perf_counter()
        corpus_ids, corpus_matrix = self.fused_corpus(alpha)
        if timings is not None:
            timings["load_fused_corpus"] = perf_counter() - start

        start = perf_counter()
        scores = corpus_matrix @ query_vector
        ranked_indices = top_indices(scores, min(top_k, len(corpus_ids)))
        if timings is not None:
            timings["local_search"] = perf_counter() - start

        start = perf_counter()
        candidates: list[Candidate] = []
        for index in ranked_indices:
            problem_id = corpus_ids[int(index)]
            rewritten = self.rewritten_by_id[problem_id]
            original = self.corpus_by_id.get(problem_id, {})
            summaries = rewritten["summaries"]
            candidates.append(
                Candidate(
                    problem_id=problem_id,
                    title=self.title_for_id(problem_id),
                    url=rewritten.get("url") or original.get("url", ""),
                    original_text=original.get("text", ""),
                    statement=summaries.get("statement", ""),
                    abstract=summaries.get("abstract", ""),
                    embedding_score=float(scores[int(index)]),
                )
            )
        if timings is not None:
            timings["candidate_build"] = perf_counter() - start
        return candidates

    def rerank(
        self,
        rerank_model: ModelConfig,
        query_statement: str,
        candidates: list[Candidate],
        timeout: int,
    ) -> list[Candidate]:
        scores = rerank_documents(
            rerank_model,
            query_statement,
            [candidate.statement for candidate in candidates],
            timeout=timeout,
        )
        reranked = [
            replace(candidate, rerank_score=score)
            for candidate, score in zip(candidates, scores, strict=True)
        ]
        return sorted(reranked, key=lambda candidate: candidate.rerank_score or 0.0, reverse=True)


def fuse_scores(candidates: list[Candidate], beta: float) -> list[Candidate]:
    fused = []
    for candidate in candidates:
        if candidate.rerank_score is None:
            final_score = candidate.embedding_score
        else:
            final_score = beta * candidate.rerank_score + (1.0 - beta) * candidate.embedding_score
        fused.append(replace(candidate, final_score=final_score))
    return sorted(fused, key=lambda candidate: candidate.final_score or 0.0, reverse=True)

