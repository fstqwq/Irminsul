from __future__ import annotations

import os
import json
import time
import uuid
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, AsyncIterator
from urllib.parse import urlparse

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from itsdangerous import BadSignature, URLSafeSerializer
from pydantic import BaseModel, Field

from core import SRC_DIR, Settings, ensure_database, get_settings, verify_password
from pipeline import (
    confirm_import_job,
    create_build_index_job,
    create_import_dry_run,
    execute_build_index_job,
    get_index,
    get_job,
    list_import_jobs,
    list_indexes,
)
from search import SearchIndex, search_events


class SearchRequest(BaseModel):
    query_text: str = Field(min_length=1)
    use_rewrite: bool = True
    use_rerank: bool = True
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    beta: float = Field(default=0.75, ge=0.0, le=1.0)
    edited_statement: str | None = None
    edited_abstract: str | None = None


class LoginRequest(BaseModel):
    password: str = Field(min_length=1)


@lru_cache(maxsize=1)
def settings() -> Settings:
    return get_settings()


@lru_cache(maxsize=1)
def legacy_index() -> SearchIndex:
    return SearchIndex(settings().data_dir)


def _admin_secret(current_settings: Settings) -> str:
    secret = os.environ.get(current_settings.admin.signing_secret_env, "").strip()
    if not secret:
        raise HTTPException(status_code=503, detail="Admin signing secret is not configured")
    return secret


def _password_hash(current_settings: Settings) -> str:
    password_hash = os.environ.get(current_settings.admin.password_hash_env, "").strip()
    if not password_hash:
        raise HTTPException(status_code=503, detail="Admin password hash is not configured")
    return password_hash


def _serializer(current_settings: Settings) -> URLSafeSerializer:
    return URLSafeSerializer(_admin_secret(current_settings), salt="yuantiji-admin-session")


def _session_seconds(current_settings: Settings) -> int:
    return max(1, current_settings.admin.session_hours) * 3600


def _host_origin(request: Request) -> str:
    return f"{request.url.scheme}://{request.headers.get('host', request.url.netloc)}"


def _validate_origin(request: Request) -> None:
    if request.method.upper() in {"GET", "HEAD", "OPTIONS"}:
        return
    expected = _host_origin(request)
    origin = request.headers.get("origin")
    if origin and origin != expected:
        raise HTTPException(status_code=403, detail="Invalid request origin")
    referer = request.headers.get("referer")
    if referer:
        parsed = urlparse(referer)
        referer_origin = f"{parsed.scheme}://{parsed.netloc}"
        if referer_origin != expected:
            raise HTTPException(status_code=403, detail="Invalid request referer")


def _read_session(request: Request) -> dict[str, Any]:
    current_settings = settings()
    raw_session = request.cookies.get("admin_session")
    if not raw_session:
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        payload = _serializer(current_settings).loads(raw_session)
    except BadSignature as exc:
        raise HTTPException(status_code=401, detail="Invalid session") from exc
    if not isinstance(payload, dict) or payload.get("sub") != "admin":
        raise HTTPException(status_code=401, detail="Invalid session")
    expires_at = int(payload.get("exp") or 0)
    if expires_at < int(time.time()):
        raise HTTPException(status_code=401, detail="Session expired")
    return payload


def require_admin(request: Request) -> dict[str, Any]:
    payload = _read_session(request)
    _validate_origin(request)
    if request.method.upper() not in {"GET", "HEAD", "OPTIONS"}:
        session_csrf = str(payload.get("csrf") or "")
        cookie_csrf = request.cookies.get("admin_csrf") or ""
        header_csrf = request.headers.get("X-CSRF-Token") or ""
        if not session_csrf or session_csrf != cookie_csrf or session_csrf != header_csrf:
            raise HTTPException(status_code=403, detail="Invalid CSRF token")
    return payload


def _set_session_cookies(response: Response, current_settings: Settings) -> None:
    csrf = os.urandom(24).hex()
    expires_at = int(time.time()) + _session_seconds(current_settings)
    token = _serializer(current_settings).dumps({"sub": "admin", "exp": expires_at, "csrf": csrf})
    max_age = _session_seconds(current_settings)
    response.set_cookie(
        "admin_session",
        token,
        max_age=max_age,
        httponly=True,
        samesite="lax",
    )
    response.set_cookie(
        "admin_csrf",
        csrf,
        max_age=max_age,
        httponly=False,
        samesite="lax",
    )


def _clear_session_cookies(response: Response) -> None:
    response.delete_cookie("admin_session")
    response.delete_cookie("admin_csrf")


def _decode_job(job: dict[str, Any]) -> dict[str, Any]:
    decoded = dict(job)
    for key in ("payload", "progress", "result"):
        value = decoded.get(key)
        if isinstance(value, str) and value:
            try:
                decoded[key] = json.loads(value)
            except ValueError:
                decoded[key] = value
    return decoded


def _decode_index(index: dict[str, Any]) -> dict[str, Any]:
    decoded = dict(index)
    value = decoded.get("meta")
    if isinstance(value, str) and value:
        try:
            decoded["meta"] = json.loads(value)
        except ValueError:
            decoded["meta"] = value
    return decoded


async def _store_upload(file: UploadFile, current_settings: Settings) -> Path:
    current_settings.storage.upload_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "upload.jsonl").suffix or ".jsonl"
    path = current_settings.storage.upload_dir / f"{uuid.uuid4().hex}{suffix}"
    total = 0
    try:
        with path.open("wb") as f:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > current_settings.limits.upload_max_bytes:
                    raise HTTPException(status_code=413, detail="Upload is too large")
                f.write(chunk)
        if total == 0:
            raise HTTPException(status_code=400, detail="Upload is empty")
        return path
    except Exception:
        path.unlink(missing_ok=True)
        raise


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        ensure_database(settings())
        yield

    app = FastAPI(title="Yuantiji", version="0.2.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        current_settings = settings()
        current_index = legacy_index()
        return {
            "ok": True,
            "loaded_index_key": None,
            "problem_count": current_index.corpus_count,
            "embedding_shape": current_index.embedding_shape,
            "views": ["statement", "abstract"],
            "switching": False,
            "data_dir": str(current_settings.data_dir),
            "corpus_count": current_index.corpus_count,
        }

    @app.get("/api/config")
    def config() -> dict[str, Any]:
        search_config = settings().search
        return {
            "top_retrieval": search_config.top_retrieval,
            "top_display": search_config.top_display,
            "default_alpha": search_config.alpha,
            "default_beta": search_config.beta,
            "default_rerank": search_config.default_rerank,
        }

    @app.post("/api/search")
    def search(request: SearchRequest) -> StreamingResponse:
        return StreamingResponse(
            search_events(request, settings(), legacy_index()),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/admin/api/auth/login")
    def login(payload: LoginRequest, response: Response) -> dict[str, bool]:
        current_settings = settings()
        if not verify_password(payload.password, _password_hash(current_settings)):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        _set_session_cookies(response, current_settings)
        return {"ok": True}

    @app.post("/admin/api/auth/logout")
    def logout(response: Response) -> dict[str, bool]:
        _clear_session_cookies(response)
        return {"ok": True}

    @app.get("/admin/api/auth/me")
    def me(session: dict[str, Any] = Depends(require_admin)) -> dict[str, Any]:
        return {"authenticated": True, "sub": session["sub"], "exp": session["exp"]}

    @app.get("/admin/api/dashboard")
    def dashboard(session: dict[str, Any] = Depends(require_admin)) -> dict[str, Any]:
        del session
        current_index = legacy_index()
        return {
            "problem_count": current_index.corpus_count,
            "source_count": 0,
            "active_index_key": None,
            "current_job": None,
            "today_searches": 0,
        }

    @app.post("/admin/api/import/dry-run")
    async def import_dry_run(
        file: UploadFile = File(...),
        mode: str = Form("upsert"),
        session: dict[str, Any] = Depends(require_admin),
    ) -> dict[str, Any]:
        del session
        current_settings = settings()
        path = await _store_upload(file, current_settings)
        try:
            return create_import_dry_run(path, mode, current_settings)
        except ValueError as exc:
            path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/admin/api/import/{job_key}/confirm")
    def import_confirm(
        job_key: str,
        session: dict[str, Any] = Depends(require_admin),
    ) -> dict[str, Any]:
        del session
        try:
            return _decode_job(confirm_import_job(job_key, settings()))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/admin/api/imports")
    def imports(session: dict[str, Any] = Depends(require_admin)) -> dict[str, Any]:
        del session
        return {"items": [_decode_job(job) for job in list_import_jobs(settings())]}

    @app.get("/admin/api/imports/{job_key}")
    def import_detail(
        job_key: str,
        session: dict[str, Any] = Depends(require_admin),
    ) -> dict[str, Any]:
        del session
        job = get_job(settings(), job_key)
        if job is None or job["type"] != "import":
            raise HTTPException(status_code=404, detail="Import job not found")
        return _decode_job(job)

    @app.post("/admin/api/index/build")
    def index_build(session: dict[str, Any] = Depends(require_admin)) -> dict[str, Any]:
        del session
        try:
            job = create_build_index_job(settings())
            return _decode_job(execute_build_index_job(job["key"], settings()))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/admin/api/indexes")
    def indexes(session: dict[str, Any] = Depends(require_admin)) -> dict[str, Any]:
        del session
        return {"items": [_decode_index(index) for index in list_indexes(settings())]}

    @app.get("/admin/api/indexes/{index_key}")
    def index_detail(
        index_key: str,
        session: dict[str, Any] = Depends(require_admin),
    ) -> dict[str, Any]:
        del session
        index = get_index(settings(), index_key)
        if index is None:
            raise HTTPException(status_code=404, detail="Index not found")
        return _decode_index(index)

    @app.get("/admin/api/settings")
    def admin_settings(session: dict[str, Any] = Depends(require_admin)) -> dict[str, Any]:
        del session
        current_settings = settings()
        return {
            "storage": {
                "db_path": str(current_settings.storage.db_path),
                "upload_dir": str(current_settings.storage.upload_dir),
                "index_cache_dir": str(current_settings.storage.index_cache_dir),
            },
            "models": {
                "rewrite": {
                    "model": current_settings.rewrite_model.model,
                    "url": current_settings.rewrite_model.url,
                    "api_key_env": current_settings.rewrite_model.api_key_env,
                },
                "embedding": {
                    "model": current_settings.embedding_model.model,
                    "url": current_settings.embedding_model.url,
                    "api_key_env": current_settings.embedding_model.api_key_env,
                },
                "rerank": {
                    "model": current_settings.rerank_model.model,
                    "url": current_settings.rerank_model.url,
                    "api_key_env": current_settings.rerank_model.api_key_env,
                },
            },
            "search": current_settings.search.__dict__,
            "index_cache": current_settings.index_cache.__dict__,
        }

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
