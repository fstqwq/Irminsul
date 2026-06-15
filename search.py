from __future__ import annotations

import json
import re
import threading
import time
import uuid
from collections.abc import Iterator, Generator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import requests

from core import ModelConfig, Settings, db_connection, json_dumps, row_to_dict, utc_now


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


@dataclass(frozen=True)
class RewriteCallResult:
    rewrite: RewriteResult
    usage: dict[str, Any]


@dataclass(frozen=True)
class EmbeddingCallResult:
    vectors: np.ndarray
    usage: dict[str, Any]


@dataclass(frozen=True)
class RerankCallResult:
    scores: list[float]
    usage: dict[str, Any]


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


def _public_usage(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): item
        for key, item in value.items()
        if isinstance(item, (str, int, float, bool)) or item is None
    }


def rewrite_query_with_usage(
    model_config: ModelConfig,
    text: str,
    timeout: int = 240,
) -> RewriteCallResult:
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
    return RewriteCallResult(parse_rewrite_output(raw), _public_usage(payload.get("usage")))


def rewrite_query(model_config: ModelConfig, text: str, timeout: int = 240) -> RewriteResult:
    return rewrite_query_with_usage(model_config, text, timeout).rewrite


def embed_texts_with_usage(
    model_config: ModelConfig,
    texts: list[str],
    timeout: int = 240,
) -> EmbeddingCallResult:
    if not model_config.api_key:
        raise ValueError(f"{model_config.api_key_env} is not configured")
    if not texts:
        return EmbeddingCallResult(np.empty((0, 0), dtype=np.float32), {"input_count": 0})

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
    usage = _public_usage(payload.get("usage"))
    usage.setdefault("input_count", len(texts))
    return EmbeddingCallResult(normalize_matrix(vectors), usage)


def embed_texts(model_config: ModelConfig, texts: list[str], timeout: int = 240) -> np.ndarray:
    return embed_texts_with_usage(model_config, texts, timeout).vectors


def rerank_documents_with_usage(
    model_config: ModelConfig,
    query: str,
    documents: list[str],
    timeout: int = 240,
) -> RerankCallResult:
    if not model_config.api_key:
        raise ValueError(f"{model_config.api_key_env} is not configured")
    if not documents:
        return RerankCallResult([], {"pair_count": 0})

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
    usage = _public_usage(payload.get("usage"))
    usage.setdefault("pair_count", len(documents))
    return RerankCallResult([float(score) for score in scores], usage)


def rerank_documents(
    model_config: ModelConfig,
    query: str,
    documents: list[str],
    timeout: int = 240,
) -> list[float]:
    return rerank_documents_with_usage(model_config, query, documents, timeout).scores


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


def query_views_from_rewrite(rewrite: RewriteResult) -> dict[str, str]:
    return {
        "clean": rewrite.clean or rewrite.statement,
        "statement": rewrite.statement,
        "abstract": rewrite.abstract,
        "abstract_zh": rewrite.abstract_zh or rewrite.abstract,
    }


def _numeric(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _model_pricing(
    settings: Settings,
    model_config: ModelConfig,
) -> tuple[str | None, dict[str, Any]]:
    pricing = settings.audit.pricing
    for provider, provider_models in pricing.items():
        if not isinstance(provider_models, dict):
            continue
        model_pricing = provider_models.get(model_config.model)
        if isinstance(model_pricing, dict):
            return str(provider), dict(model_pricing)
    return None, {}


def _estimate_cost_microusd(usage: dict[str, Any], pricing: dict[str, Any]) -> int:
    if not pricing:
        return 0
    input_tokens = _numeric(usage.get("prompt_tokens", usage.get("input_tokens", 0)))
    output_tokens = _numeric(
        usage.get("completion_tokens", usage.get("output_tokens", 0))
    )
    if input_tokens == 0 and output_tokens == 0:
        input_tokens = _numeric(usage.get("total_tokens", 0))

    pair_count = _numeric(usage.get("pair_count", 0))
    cost = 0.0
    cost += (
        input_tokens
        * _numeric(pricing.get("input_price_per_1m_tokens_microusd"))
        / 1_000_000
    )
    cost += (
        output_tokens
        * _numeric(pricing.get("output_price_per_1m_tokens_microusd"))
        / 1_000_000
    )
    cost += pair_count * _numeric(pricing.get("price_per_1k_pairs_microusd")) / 1_000
    cost += (
        pair_count
        * _numeric(pricing.get("price_per_1m_pairs_microusd"))
        / 1_000_000
    )
    cost += pair_count * _numeric(pricing.get("price_per_pair_microusd"))
    return int(round(cost))


def _api_call_entry(
    stage: str,
    model_config: ModelConfig,
    usage: dict[str, Any],
    settings: Settings,
) -> dict[str, Any]:
    provider, pricing = _model_pricing(settings, model_config)
    cost_microusd = _estimate_cost_microusd(usage, pricing)
    return {
        "stage": stage,
        "provider": provider,
        "model": model_config.model,
        "usage": usage,
        "pricing": pricing,
        "cost": {"microusd": cost_microusd},
    }


def _total_cost(api_calls: list[dict[str, Any]]) -> dict[str, int]:
    total = 0
    for call in api_calls:
        cost = call.get("cost")
        if isinstance(cost, dict):
            total += int(_numeric(cost.get("microusd", 0)))
    return {"microusd": total}


def _date_bound(value: str, end_of_day: bool = False) -> str:
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return f"{value}T{'23:59:59' if end_of_day else '00:00:00'}Z"
    return value


def retrieve_loaded_index(
    loaded_index: LoadedIndex,
    query_vectors: np.ndarray,
    top_per_doc_view: int,
) -> list[Candidate]:
    if query_vectors.shape[0] != len(VIEWS):
        raise ValueError("query vector view count mismatch")

    candidates_by_ord: dict[int, float] = {}
    query_matrix = query_vectors.T
    for view in VIEWS:
        matrix = loaded_index.matrices[view]
        scores = matrix @ query_matrix
        best_scores = np.max(scores, axis=1)
        for index in top_indices(best_scores, min(top_per_doc_view, best_scores.shape[0])):
            problem_ord = int(index)
            score = float(best_scores[problem_ord])
            current = candidates_by_ord.get(problem_ord)
            if current is None or score > current:
                candidates_by_ord[problem_ord] = score

    ranked = sorted(candidates_by_ord.items(), key=lambda item: item[1], reverse=True)
    candidates: list[Candidate] = []
    for problem_ord, score in ranked:
        candidates.append(
            Candidate(
                problem_id=loaded_index.problem_keys[problem_ord],
                title=loaded_index.titles[problem_ord],
                url=loaded_index.urls[problem_ord],
                original_text=loaded_index.texts["clean"][problem_ord],
                statement=loaded_index.texts["statement"][problem_ord],
                abstract=loaded_index.texts["abstract"][problem_ord],
                embedding_score=score,
            )
        )
    return candidates


def search_events_loaded(
    request: Any,
    settings: Settings,
    loaded_index: LoadedIndex,
    client_ip: str = "",
    user_agent: str = "",
) -> Iterator[str]:
    timings: dict[str, float | str] = {}
    search_config = settings.search
    request_id = uuid.uuid4().hex
    started_at = utc_now()
    api_calls: list[dict[str, Any]] = []

    try:
        edited_statement = (request.edited_statement or "").strip()
        edited_abstract = (request.edited_abstract or "").strip()

        if edited_statement:
            timings["rewrite"] = "edited"
            rewrite = RewriteResult(
                clean=edited_statement,
                statement=edited_statement,
                abstract=edited_abstract or edited_statement,
                abstract_zh=edited_abstract or edited_statement,
                raw="",
            )
            yield _stage("rewrite", "done", detail="edited")
            yield ndjson_event(
                "rewrite",
                statement=rewrite.statement,
                abstract=rewrite.abstract,
                raw="",
                edited=True,
            )
        elif request.use_rewrite:
            yield _stage("rewrite", "active")
            start = perf_counter()
            rewrite_call = rewrite_query_with_usage(
                settings.rewrite_model,
                request.query_text,
                timeout=settings.request_timeout,
            )
            rewrite = rewrite_call.rewrite
            api_calls.append(
                _api_call_entry(
                    "rewrite",
                    settings.rewrite_model,
                    rewrite_call.usage,
                    settings,
                )
            )
            elapsed = perf_counter() - start
            timings["rewrite"] = elapsed
            yield _stage("rewrite", "done", elapsed=elapsed)
            yield ndjson_event(
                "rewrite",
                statement=rewrite.statement,
                abstract=rewrite.abstract,
                raw=rewrite.raw,
                edited=False,
            )
        else:
            query_text = request.query_text.strip()
            if not query_text:
                raise ValueError("query text is empty")
            timings["rewrite"] = "off"
            rewrite = RewriteResult(
                clean=query_text,
                statement=query_text,
                abstract=query_text,
                abstract_zh=query_text,
                raw="",
            )
            yield _stage("rewrite", "skip", detail="off")

        views = query_views_from_rewrite(rewrite)
        if not views["statement"]:
            raise ValueError("query text is empty")

        yield _stage("embed", "active", detail="4 input")
        embed_start = perf_counter()
        embedding_call = embed_texts_with_usage(
            settings.embedding_model,
            [views[view] for view in VIEWS],
            timeout=settings.request_timeout,
        )
        query_vectors = normalize_matrix(
            np.asarray(embedding_call.vectors, dtype=np.float32)
        )
        api_calls.append(
            _api_call_entry(
                "embedding",
                settings.embedding_model,
                embedding_call.usage,
                settings,
            )
        )
        embed_elapsed = perf_counter() - embed_start
        timings["embed"] = embed_elapsed
        yield _stage("embed", "done", elapsed=embed_elapsed, detail="4 input")

        yield _stage("search", "active", detail=f"top{search_config.top_per_doc_view}")
        search_start = perf_counter()
        candidates = retrieve_loaded_index(
            loaded_index,
            query_vectors,
            search_config.top_per_doc_view,
        )[: search_config.top_retrieval]
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
            rerank_call = rerank_documents_with_usage(
                settings.rerank_model,
                views["abstract_zh"],
                [
                    f"{candidate.original_text}\n\n{candidate.statement}"
                    for candidate in rerank_candidates
                ],
                timeout=settings.request_timeout,
            )
            scores = rerank_call.scores
            api_calls.append(
                _api_call_entry(
                    "rerank",
                    settings.rerank_model,
                    rerank_call.usage,
                    settings,
                )
            )
            reranked = {
                candidate.problem_id: replace(candidate, rerank_score=score)
                for candidate, score in zip(rerank_candidates, scores, strict=True)
            }
            candidates = [reranked.get(candidate.problem_id, candidate) for candidate in candidates]
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
        result_summary = _audit_result_summary(fused)
        write_search_audit(
            settings=settings,
            request_id=request_id,
            started_at=started_at,
            status="succeeded",
            client_ip=client_ip,
            user_agent=user_agent,
            query=request.query_text,
            timings=timings,
            api_calls=api_calls,
            result=result_summary,
            cost=_total_cost(api_calls),
            error=None,
        )
        yield ndjson_event(
            "candidates",
            candidates=[asdict(candidate) for candidate in fused],
            timings=timings,
        )
        yield ndjson_event("done")
    except Exception as exc:
        write_search_audit(
            settings=settings,
            request_id=request_id,
            started_at=started_at,
            status="failed",
            client_ip=client_ip,
            user_agent=user_agent,
            query=getattr(request, "query_text", ""),
            timings=timings,
            api_calls=api_calls,
            result={"candidates": []},
            cost=_total_cost(api_calls),
            error=str(exc),
        )
        yield ndjson_event("error", message=str(exc))


def write_search_audit(
    settings: Settings,
    request_id: str,
    started_at: str,
    status: str,
    client_ip: str,
    user_agent: str,
    query: str,
    timings: dict[str, float | str],
    api_calls: list[dict[str, Any]],
    result: dict[str, Any],
    cost: dict[str, Any],
    error: str | None,
) -> None:
    finished_at = utc_now()
    with db_connection(settings) as conn:
        with conn:
            conn.execute(
                """
                INSERT INTO search_audits(
                  request_id, started_at, finished_at, status, client_ip, user_agent,
                  query, timings, api_calls, result, cost, error
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    started_at,
                    finished_at,
                    status,
                    client_ip,
                    user_agent,
                    query,
                    json_dumps(timings),
                    json_dumps(api_calls),
                    json_dumps(result),
                    json_dumps(cost),
                    error,
                ),
            )


def list_search_audits(
    settings: Settings,
    status: str | None = None,
    q: str = "",
    date_from: str = "",
    date_to: str = "",
    limit: int = 50,
) -> list[dict[str, Any]]:
    where = ["1 = 1"]
    params: list[Any] = []
    if status:
        where.append("status = ?")
        params.append(status)
    if q.strip():
        where.append("query LIKE ?")
        params.append(f"%{q.strip()}%")
    if date_from:
        where.append("started_at >= ?")
        params.append(_date_bound(date_from))
    if date_to:
        where.append("started_at <= ?")
        params.append(_date_bound(date_to, end_of_day=True))
    limit = max(1, min(limit, 500))
    with db_connection(settings) as conn:
        rows = conn.execute(
            f"""
            SELECT * FROM search_audits
            WHERE {' AND '.join(where)}
            ORDER BY started_at DESC
            LIMIT ?
            """,
            [*params, limit],
        ).fetchall()
    return [row_to_dict(row) for row in rows]


def get_search_audit(settings: Settings, request_id: str) -> dict[str, Any] | None:
    with db_connection(settings) as conn:
        row = conn.execute(
            "SELECT * FROM search_audits WHERE request_id = ?",
            (request_id,),
        ).fetchone()
    return row_to_dict(row) if row else None


def _audit_result_summary(candidates: list[Candidate]) -> dict[str, Any]:
    return {
        "candidate_count": len(candidates),
        "top": [
            {
                "problem_id": candidate.problem_id,
                "title": candidate.title,
                "embedding_score": candidate.embedding_score,
                "rerank_score": candidate.rerank_score,
                "final_score": candidate.final_score,
            }
            for candidate in candidates[:20]
        ],
    }


def ndjson_event(event_type: str, **payload: Any) -> str:
    return json.dumps({"type": event_type, **payload}, ensure_ascii=False) + "\n"


def _stage(name: str, state: str, elapsed: float | None = None, detail: str = "") -> str:
    payload: dict[str, Any] = {"name": name, "state": state}
    if elapsed is not None:
        payload["elapsed"] = elapsed
    if detail:
        payload["detail"] = detail
    return ndjson_event("stage", **payload)
