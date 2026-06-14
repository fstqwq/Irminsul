from __future__ import annotations

import json
import re
import threading
import time
from collections import defaultdict
from collections.abc import Iterator, Generator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import requests

from core import ModelConfig, SearchConfig, Settings


VIEWS = ("clean", "statement", "abstract", "abstract_zh")


@dataclass(frozen=True)
class RewriteResult:
    statement: str
    abstract: str
    raw: str
    clean: str = ""
    abstract_zh: str = ""


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


@dataclass(frozen=True)
class LoadedIndex:
    key: str
    problem_keys: list[str]
    titles: list[str]
    urls: list[str]
    texts: dict[str, list[str]]
    matrices: dict[str, np.ndarray]
    load_mode: str

    @property
    def problem_count(self) -> int:
        return len(self.problem_keys)

    @property
    def embedding_shape(self) -> tuple[int, int] | None:
        if not self.matrices:
            return None
        first = next(iter(self.matrices.values()))
        return tuple(first.shape)  # type: ignore[return-value]


class IndexState:
    def __init__(self) -> None:
        self.current: LoadedIndex | None = None
        self.switching = False
        self.inflight_searches = 0
        self.condition = threading.Condition()

    @contextmanager
    def search_snapshot(self) -> Generator[LoadedIndex, None, None]:
        with self.condition:
            if self.switching:
                raise RuntimeError("index is switching")
            if self.current is None:
                raise RuntimeError("index is not loaded")
            self.inflight_searches += 1
            current = self.current
        try:
            yield current
        finally:
            with self.condition:
                self.inflight_searches -= 1
                self.condition.notify_all()

    def activate(self, new_index: LoadedIndex, drain_timeout_seconds: int) -> LoadedIndex | None:
        deadline = time.monotonic() + drain_timeout_seconds
        with self.condition:
            if self.switching:
                raise RuntimeError("index is already switching")
            self.switching = True
            try:
                while self.inflight_searches > 0:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("timed out waiting for in-flight searches")
                    self.condition.wait(timeout=remaining)
                old_index = self.current
                self.current = new_index
                return old_index
            finally:
                self.switching = False
                self.condition.notify_all()


class RewriteFormatError(ValueError):
    pass


REWRITE_PROMPT = """You rewrite competitive programming statements.

Output exactly four sections and nothing else:

[clean]
Keep the original problem requirements, remove decorative story text when safe, and keep necessary details.

[statement]
Strip off stories, legends, characters, and backgrounds while preserving all requirements. If it is not English, translate it to English. Use TeX delimiters for math.

[abstract]
Make the problem as succinct as possible while still understandable. Prefer the core algorithmic task over symbols.

[abstract_zh]
Write a concise Simplified Chinese abstract of the algorithmic task."""


def read_jsonl_list(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_index_cache(cache_path: Path, load_mode: str = "mmap") -> LoadedIndex:
    manifest_path = cache_path / "manifest.json"
    if not manifest_path.exists():
        raise ValueError("cache manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    index_key = str(manifest["index_key"])
    views = tuple(manifest["views"])
    if views != VIEWS:
        raise ValueError("cache views do not match runtime views")
    problem_count = int(manifest["problem_count"])
    dim = int(manifest["dim"])
    mmap_mode = "r" if load_mode == "mmap" else None

    problems = read_jsonl_list(cache_path / "problems.jsonl")
    view_rows = read_jsonl_list(cache_path / "views.jsonl")
    if len(problems) != problem_count or len(view_rows) != problem_count:
        raise ValueError("cache metadata row count mismatch")

    matrices: dict[str, np.ndarray] = {}
    for view in VIEWS:
        matrix = np.load(cache_path / f"{view}.npy", mmap_mode=mmap_mode)
        if matrix.shape != (problem_count, dim):
            raise ValueError(f"cache matrix shape mismatch for {view}")
        if matrix.dtype != np.float32:
            raise ValueError(f"cache matrix dtype mismatch for {view}")
        matrices[view] = matrix

    return LoadedIndex(
        key=index_key,
        problem_keys=[str(row["key"]) for row in problems],
        titles=[str(row["title"]) for row in problems],
        urls=[str(row["url"]) for row in problems],
        texts={view: [str(row[view]) for row in view_rows] for view in VIEWS},
        matrices=matrices,
        load_mode=load_mode,
    )


def top_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.empty((0,), dtype=np.int64)
    if k >= scores.shape[0]:
        indices = np.arange(scores.shape[0])
    else:
        indices = np.argpartition(-scores, k - 1)[:k]
    return indices[np.argsort(-scores[indices])]


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


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


def _find_section_marker(text: str, name: str) -> re.Match[str] | None:
    pattern = rf"(?im)^[ \t]*(?:\*\*)?\[?{name}\]?(?:\*\*)?[ \t]*:?[ \t]*$"
    return re.search(pattern, text)


def _extract_section(text: str, name: str, next_starts: list[int]) -> str:
    marker = _find_section_marker(text, name)
    if marker is None:
        return ""
    end = min([start for start in next_starts if start > marker.end()] or [len(text)])
    value = text[marker.end() : end].strip()
    if value.startswith("---"):
        value = value[3:].strip()
    if value.endswith("---"):
        value = value[:-3].strip()
    return value


def parse_rewrite_output(text: str) -> RewriteResult:
    markers = {
        name: match
        for name in ("clean", "statement", "abstract", "abstract_zh")
        if (match := _find_section_marker(text, name)) is not None
    }
    if "statement" not in markers:
        raise RewriteFormatError("missing_statement_section")
    if "abstract" not in markers:
        raise RewriteFormatError("missing_abstract_section")
    if markers["abstract"].start() <= markers["statement"].start():
        raise RewriteFormatError("abstract_before_statement")

    starts = [match.start() for match in markers.values()]
    statement = _extract_section(text, "statement", starts)
    abstract = _extract_section(text, "abstract", starts)
    clean = _extract_section(text, "clean", starts) or statement
    abstract_zh = _extract_section(text, "abstract_zh", starts) or abstract

    if not statement:
        raise RewriteFormatError("empty_statement_section")
    if not abstract:
        raise RewriteFormatError("empty_abstract_section")
    return RewriteResult(
        statement=statement,
        abstract=abstract,
        raw=text,
        clean=clean,
        abstract_zh=abstract_zh,
    )


def rewrite_query(model_config: ModelConfig, text: str, timeout: int = 240) -> RewriteResult:
    if not model_config.api_key:
        raise ValueError(f"{model_config.api_key_env} is not configured")
    response = requests.post(
        model_config.resolved_url,
        headers={
            "Authorization": f"Bearer {model_config.api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model_config.model,
            "messages": [
                {"role": "system", "content": REWRITE_PROMPT},
                {"role": "user", "content": text},
            ],
            "temperature": 0,
            "thinking": {"type": "disabled"},
            "stream": False,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    raw = payload["choices"][0]["message"]["content"]
    return parse_rewrite_output(raw)


def embed_texts(model_config: ModelConfig, texts: list[str], timeout: int = 240) -> np.ndarray:
    if not model_config.api_key:
        raise ValueError(f"{model_config.api_key_env} is not configured")
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    response = requests.post(
        model_config.resolved_url,
        headers={
            "Authorization": f"Bearer {model_config.api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model_config.model,
            "input": texts,
            "encoding_format": "float",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    data = sorted(payload["data"], key=lambda item: item["index"])
    vectors = np.array([item["embedding"] for item in data], dtype=np.float32)
    return normalize_matrix(vectors)


def rerank_documents(
    model_config: ModelConfig,
    query: str,
    documents: list[str],
    timeout: int = 240,
) -> list[float]:
    if not model_config.api_key:
        raise ValueError(f"{model_config.api_key_env} is not configured")
    if not documents:
        return []

    response = requests.post(
        model_config.resolved_url,
        headers={
            "Authorization": f"Bearer {model_config.api_key}",
            "Content-Type": "application/json",
        },
        json={"queries": [query], "documents": documents},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    scores = payload.get("scores")
    if not isinstance(scores, list) or len(scores) != len(documents):
        raise ValueError(f"Unexpected reranker response: {payload}")
    return [float(score) for score in scores]


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
            [f"{candidate.original_text}\n\n{candidate.statement}" for candidate in candidates],
            timeout=timeout,
        )
        reranked = [
            replace(candidate, rerank_score=score)
            for candidate, score in zip(candidates, scores, strict=True)
        ]
        return sorted(reranked, key=lambda candidate: candidate.rerank_score or 0.0, reverse=True)


def fuse_scores(
    candidates: list[Candidate],
    beta: float,
    rerank_range_floor: float = 0.1,
    embedding_range_floor: float = 0.05,
) -> list[Candidate]:
    rerank_scores = [candidate.rerank_score for candidate in candidates if candidate.rerank_score is not None]
    if not rerank_scores:
        return sorted(
            [replace(candidate, final_score=candidate.embedding_score) for candidate in candidates],
            key=lambda candidate: (candidate.final_score or 0.0, candidate.embedding_score),
            reverse=True,
        )

    embeddings = [candidate.embedding_score for candidate in candidates]
    rerank_min = min(float(score) for score in rerank_scores)
    rerank_max = max(float(score) for score in rerank_scores)
    embedding_min = min(embeddings)
    embedding_max = max(embeddings)
    rerank_scale = max(rerank_max - rerank_min, rerank_range_floor)
    embedding_scale = max(embedding_max - embedding_min, embedding_range_floor)
    beta = min(max(beta, 0.001), 0.999)
    lambda_value = ((1.0 - beta) / beta) * (rerank_scale / embedding_scale)

    fused: list[Candidate] = []
    for candidate in candidates:
        embedding_score = min(max(candidate.embedding_score, 0.0), 1.0)
        rerank_score = (
            embedding_score
            if candidate.rerank_score is None
            else min(max(float(candidate.rerank_score), 0.0), 1.0)
        )
        final_score = (rerank_score + lambda_value * embedding_score) / (1.0 + lambda_value)
        fused.append(replace(candidate, final_score=final_score))

    return sorted(
        fused,
        key=lambda candidate: (
            candidate.final_score or 0.0,
            candidate.rerank_score if candidate.rerank_score is not None else -1.0,
            candidate.embedding_score,
            candidate.problem_id,
        ),
        reverse=True,
    )


def ndjson_event(event_type: str, **payload: Any) -> str:
    return json.dumps({"type": event_type, **payload}, ensure_ascii=False) + "\n"


def _stage(name: str, state: str, elapsed: float | None = None, detail: str = "") -> str:
    payload: dict[str, Any] = {"name": name, "state": state}
    if elapsed is not None:
        payload["elapsed"] = elapsed
    if detail:
        payload["detail"] = detail
    return ndjson_event("stage", **payload)


def search_events(request: Any, settings: Settings, index: SearchIndex) -> Iterator[str]:
    timings: dict[str, float | str] = {}
    search_config: SearchConfig = settings.search

    try:
        edited_statement = (request.edited_statement or "").strip()
        edited_abstract = (request.edited_abstract or "").strip()
        use_edited_rewrite = bool(edited_statement)

        if use_edited_rewrite:
            statement = edited_statement
            abstract = edited_abstract or None
            timings["rewrite"] = "edited"
            yield _stage("rewrite", "done", detail="edited")
            yield ndjson_event(
                "rewrite",
                statement=statement,
                abstract=abstract or "",
                raw="",
                edited=True,
            )
        elif request.use_rewrite:
            yield _stage("rewrite", "active")
            start = perf_counter()
            rewrite = rewrite_query(
                settings.rewrite_model,
                request.query_text,
                timeout=settings.request_timeout,
            )
            elapsed = perf_counter() - start
            timings["rewrite"] = elapsed
            statement = rewrite.statement
            abstract = rewrite.abstract
            yield _stage("rewrite", "done", elapsed=elapsed)
            yield ndjson_event(
                "rewrite",
                statement=rewrite.statement,
                abstract=rewrite.abstract,
                raw=rewrite.raw,
                edited=False,
            )
        else:
            statement = request.query_text.strip()
            abstract = None
            timings["rewrite"] = "off"
            yield _stage("rewrite", "skip", detail="off")

        if not statement:
            raise ValueError("query text is empty")

        query_timings: dict[str, float] = {}
        input_count = 1 if abstract is None else 2
        yield _stage("embed", "active", detail=f"{input_count} input")
        embed_start = perf_counter()
        query_vector = index.embed_query_vector(
            settings.embedding_model,
            statement,
            abstract,
            alpha=request.alpha,
            timeout=settings.request_timeout,
            timings=query_timings,
        )
        embed_elapsed = perf_counter() - embed_start
        timings["embed"] = embed_elapsed
        yield _stage("embed", "done", elapsed=embed_elapsed, detail=f"{input_count} input")

        yield _stage("search", "active", detail=f"top{search_config.top_retrieval}")
        search_start = perf_counter()
        candidates = index.search_with_vector(
            query_vector,
            alpha=request.alpha,
            top_k=search_config.top_retrieval,
            timings=query_timings,
        )
        search_elapsed = perf_counter() - search_start
        timings["search"] = search_elapsed
        yield _stage(
            "search",
            "done",
            elapsed=search_elapsed,
            detail=f"{len(candidates)} candidates",
        )

        if request.use_rerank:
            rerank_candidates = candidates[: search_config.rerank_top_k]
            yield _stage("rerank", "active", detail=f"{len(rerank_candidates)} candidates")
            rerank_start = perf_counter()
            reranked = index.rerank(
                settings.rerank_model,
                abstract or statement,
                rerank_candidates,
                timeout=settings.request_timeout,
            )
            rerank_by_id = {candidate.problem_id: candidate for candidate in reranked}
            candidates = [rerank_by_id.get(candidate.problem_id, candidate) for candidate in candidates]
            rerank_elapsed = perf_counter() - rerank_start
            timings["rerank"] = rerank_elapsed
            yield _stage("rerank", "done", elapsed=rerank_elapsed)
        else:
            timings["rerank"] = "off"
            yield _stage("rerank", "skip", detail="off")

        fused = fuse_scores(
            candidates,
            request.beta,
            search_config.rerank_range_floor,
            search_config.embedding_range_floor,
        )
        yield ndjson_event(
            "candidates",
            candidates=[asdict(candidate) for candidate in fused],
            timings=timings,
        )
        yield ndjson_event("done")
    except Exception as exc:
        yield ndjson_event("error", message=str(exc))
