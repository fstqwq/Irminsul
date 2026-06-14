from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import SRC_DIR, Settings, get_settings
from .retrieval import SearchIndex
from .schemas import ConfigPayload, HealthPayload, SearchRequest
from .service import search_events


@lru_cache(maxsize=1)
def settings() -> Settings:
    return get_settings()


@lru_cache(maxsize=1)
def index() -> SearchIndex:
    return SearchIndex(settings().data_dir)


def create_app() -> FastAPI:
    app = FastAPI(title="Yuantiji Retrieval", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health", response_model=HealthPayload)
    def health() -> HealthPayload:
        current_settings = settings()
        current_index = index()
        return HealthPayload(
            ok=True,
            data_dir=str(current_settings.data_dir),
            corpus_count=current_index.corpus_count,
            embedding_shape=current_index.embedding_shape,
        )

    @app.get("/api/config", response_model=ConfigPayload)
    def config() -> ConfigPayload:
        retrieval = settings().retrieval
        return ConfigPayload(
            top_retrieval=retrieval.top_retrieval,
            top_display=retrieval.top_display,
            default_alpha=retrieval.default_alpha,
            default_beta=retrieval.default_beta,
            default_rerank=retrieval.default_rerank,
        )

    @app.post("/api/search")
    def search(request: SearchRequest) -> StreamingResponse:
        return StreamingResponse(
            search_events(request, settings(), index()),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    frontend_dist = SRC_DIR / "frontend" / "dist"
    if frontend_dist.exists():
        app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
    else:
        @app.get("/")
        def missing_frontend() -> JSONResponse:
            return JSONResponse(
                {
                    "message": "Frontend has not been built yet.",
                    "expected_dist": str(Path(frontend_dist)),
                }
            )

    return app


app = create_app()

