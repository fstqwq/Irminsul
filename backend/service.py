from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict
from time import perf_counter
from typing import Any

from .api_clients import rewrite_query
from .config import Settings
from .retrieval import SearchIndex, fuse_scores
from .schemas import SearchRequest


def ndjson_event(event_type: str, **payload: Any) -> str:
    return json.dumps({"type": event_type, **payload}, ensure_ascii=False) + "\n"


def _stage(name: str, state: str, elapsed: float | None = None, detail: str = "") -> str:
    payload: dict[str, Any] = {"name": name, "state": state}
    if elapsed is not None:
        payload["elapsed"] = elapsed
    if detail:
        payload["detail"] = detail
    return ndjson_event("stage", **payload)


def search_events(
    request: SearchRequest,
    settings: Settings,
    index: SearchIndex,
) -> Iterator[str]:
    timings: dict[str, float | str] = {}

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

        yield _stage("search", "active", detail=f"top{settings.retrieval.top_retrieval}")
        search_start = perf_counter()
        candidates = index.search_with_vector(
            query_vector,
            alpha=request.alpha,
            top_k=settings.retrieval.top_retrieval,
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
            yield _stage("rerank", "active", detail=f"{len(candidates)} candidates")
            rerank_start = perf_counter()
            candidates = index.rerank(
                settings.rerank_model,
                statement,
                candidates,
                timeout=settings.request_timeout,
            )
            rerank_elapsed = perf_counter() - rerank_start
            timings["rerank"] = rerank_elapsed
            yield _stage("rerank", "done", elapsed=rerank_elapsed)
        else:
            timings["rerank"] = "off"
            yield _stage("rerank", "skip", detail="off")

        fused = fuse_scores(candidates, request.beta)
        yield ndjson_event(
            "candidates",
            candidates=[asdict(candidate) for candidate in fused],
            timings=timings,
        )
        yield ndjson_event("done")
    except Exception as exc:
        yield ndjson_event("error", message=str(exc))
