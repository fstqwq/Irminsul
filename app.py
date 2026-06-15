from __future__ import annotations

import json
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, AsyncIterator
from urllib.parse import urlparse

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from itsdangerous import BadSignature, URLSafeSerializer
from pydantic import BaseModel, Field

from core import (
    SRC_DIR,
    Settings,
    db_connection,
    ensure_database,
    get_settings,
    row_to_dict,
    utc_now,
    verify_password,
)
from pipeline import (
    batch_update_problems,
    cancel_job,
    confirm_import_job,
    create_cleanup_job,
    create_build_index_job,
    create_import_dry_run,
    get_index,
    get_job,
    get_problem,
    index_cache_path,
    JobWorker,
    list_jobs,
    list_import_jobs,
    list_indexes,
    list_job_logs,
    list_problems,
    list_sources,
    patch_problem,
    patch_source,
    recover_startup,
    rebuild_index_cache,
    retry_job,
    verify_index_cache,
)
from search import (
    IndexState,
    get_search_audit,
    list_search_audits,
    load_index_cache,
    search_events_loaded,
)


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


class ProblemPatch(BaseModel):
    title: str | None = None
    url: str | None = None
    enabled: bool | None = None
    deleted: bool | None = None


class ProblemBatchRequest(BaseModel):
    keys: list[str] = Field(min_length=1)


class SourcePatch(BaseModel):
    name: str | None = None
    enabled: bool | None = None


@lru_cache(maxsize=1)
def settings() -> Settings:
    return get_settings()


@lru_cache(maxsize=1)
def index_state() -> IndexState:
    return IndexState()


def _admin_secret(current_settings: Settings) -> str:
    path = current_settings.admin.signing_secret_file
    secret = path.read_text(encoding="utf-8-sig").strip() if path.exists() else ""
    if not secret:
        raise HTTPException(status_code=503, detail="Admin signing secret is not configured")
    return secret


def _password_hash(current_settings: Settings) -> str:
    path = current_settings.admin.password_hash_file
    password_hash = path.read_text(encoding="utf-8-sig").strip() if path.exists() else ""
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
    csrf = secrets.token_hex(24)
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


def _load_active_index(current_settings: Settings, state: IndexState) -> None:
    with db_connection(current_settings) as conn:
        row = conn.execute("SELECT value FROM kv WHERE key = 'active_index_key'").fetchone()
    if row is None:
        return
    selected_index_key = row["value"]
    loaded = load_index_cache(
        index_cache_path(current_settings, selected_index_key),
        current_settings.index_cache.load_mode,
    )
    state.activate(loaded, current_settings.index_cache.activation_drain_timeout_seconds)


def _activate_index(current_settings: Settings, selected_index_key: str, state: IndexState) -> None:
    index = get_index(current_settings, selected_index_key)
    if index is None:
        raise ValueError("index not found")
    if index["status"] not in {"built", "active", "retired"}:
        raise ValueError("index is not built")
    loaded = load_index_cache(
        index_cache_path(current_settings, selected_index_key),
        current_settings.index_cache.load_mode,
    )
    state.activate(loaded, current_settings.index_cache.activation_drain_timeout_seconds)
    now = utc_now()
    with db_connection(current_settings) as conn:
        with conn:
            conn.execute("UPDATE indexes SET status = 'retired' WHERE status = 'active'")
            conn.execute(
                """
                UPDATE indexes
                SET status = 'active', activated_at = ?, error = NULL
                WHERE key = ?
                """,
                (now, selected_index_key),
            )
            conn.execute(
                """
                INSERT INTO kv(key, value, updated_at)
                VALUES ('active_index_key', ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                  value = excluded.value,
                  updated_at = excluded.updated_at
                """,
                (selected_index_key, now),
            )


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


def _decode_job_log(row: dict[str, Any]) -> dict[str, Any]:
    decoded = dict(row)
    value = decoded.get("data")
    if isinstance(value, str) and value:
        try:
            decoded["data"] = json.loads(value)
        except ValueError:
            decoded["data"] = value
    return decoded


def _fallback_job_logs(job: dict[str, Any]) -> list[dict[str, Any]]:
    result = job.get("result")
    if not isinstance(result, dict):
        return []
    logs: list[dict[str, Any]] = []
    if result.get("canceled"):
        logs.append(
            {
                "id": 0,
                "job_key": job.get("key"),
                "level": "warning",
                "message": "Job canceled",
                "data": None,
                "created_at": job.get("updated_at"),
            }
        )
    failures = result.get("failures")
    if isinstance(failures, list):
        for index, failure in enumerate(failures[:500], start=1):
            logs.append(
                {
                    "id": index,
                    "job_key": job.get("key"),
                    "level": "error",
                    "message": "Artifact generation failed",
                    "data": failure,
                    "created_at": job.get("updated_at"),
                }
            )
    return logs


def _decode_index(index: dict[str, Any]) -> dict[str, Any]:
    decoded = dict(index)
    value = decoded.get("meta")
    if isinstance(value, str) and value:
        try:
            decoded["meta"] = json.loads(value)
        except ValueError:
            decoded["meta"] = value
    return decoded


def _decode_audit(audit: dict[str, Any]) -> dict[str, Any]:
    decoded = dict(audit)
    for key in ("timings", "api_calls", "result", "cost"):
        value = decoded.get(key)
        if isinstance(value, str) and value:
            try:
                decoded[key] = json.loads(value)
            except ValueError:
                decoded[key] = value
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
        current_settings = settings()
        ensure_database(current_settings)
        recover_startup(current_settings)
        _load_active_index(current_settings, index_state())
        worker = JobWorker(current_settings)
        worker.start()
        try:
            yield
        finally:
            worker.stop()

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
        state = index_state()
        active_index = state.current
        return {
            "ok": True,
            "loaded_index_key": active_index.key if active_index else None,
            "problem_count": active_index.problem_count if active_index else 0,
            "embedding_shape": active_index.embedding_shape if active_index else None,
            "views": list(active_index.texts.keys()) if active_index else ["statement", "abstract"],
            "switching": state.switching,
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
    def search(payload: SearchRequest, request: Request) -> StreamingResponse:
        state = index_state()
        if state.switching:
            raise HTTPException(status_code=503, detail="Index is switching")
        if state.current is None:
            raise HTTPException(status_code=503, detail="Index is not loaded")
        client_ip = request.client.host if request.client else ""
        user_agent = request.headers.get("user-agent", "")

        def stream() -> Any:
            with state.search_snapshot() as loaded:
                yield from search_events_loaded(
                    payload,
                    settings(),
                    loaded,
                    client_ip=client_ip,
                    user_agent=user_agent,
                )

        return StreamingResponse(
            stream(),
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
        current_settings = settings()
        with db_connection(current_settings) as conn:
            problem_count = conn.execute(
                "SELECT count(*) FROM problems WHERE deleted = 0"
            ).fetchone()[0]
            source_count = conn.execute("SELECT count(*) FROM sources").fetchone()[0]
            active = conn.execute(
                "SELECT value FROM kv WHERE key = 'active_index_key'"
            ).fetchone()
            current_job = conn.execute(
                """
                SELECT * FROM jobs
                WHERE status IN ('queued', 'running')
                ORDER BY created_at
                LIMIT 1
                """
            ).fetchone()
            today_searches = conn.execute(
                "SELECT count(*) FROM search_audits WHERE started_at >= date('now')"
            ).fetchone()[0]
        return {
            "problem_count": problem_count,
            "source_count": source_count,
            "active_index_key": active["value"] if active else None,
            "current_job": _decode_job(row_to_dict(current_job)) if current_job else None,
            "today_searches": today_searches,
        }

    @app.get("/admin/api/problems")
    def problems(
        source_key: str | None = None,
        enabled: bool | None = None,
        deleted: bool | None = None,
        q: str = "",
        limit: int = 50,
        offset: int = 0,
        session: dict[str, Any] = Depends(require_admin),
    ) -> dict[str, Any]:
        del session
        return list_problems(settings(), source_key, enabled, deleted, q, limit, offset)

    @app.post("/admin/api/problems/batch-{action}")
    def problems_batch(
        action: str,
        payload: ProblemBatchRequest,
        session: dict[str, Any] = Depends(require_admin),
    ) -> dict[str, Any]:
        del session
        try:
            return batch_update_problems(settings(), payload.keys, action)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/admin/api/problems/{problem_key:path}")
    def problem_detail(
        problem_key: str,
        session: dict[str, Any] = Depends(require_admin),
    ) -> dict[str, Any]:
        del session
        problem = get_problem(settings(), problem_key)
        if problem is None:
            raise HTTPException(status_code=404, detail="Problem not found")
        return problem

    @app.patch("/admin/api/problems/{problem_key:path}")
    def problem_patch(
        problem_key: str,
        payload: ProblemPatch,
        session: dict[str, Any] = Depends(require_admin),
    ) -> dict[str, Any]:
        del session
        try:
            return patch_problem(settings(), problem_key, payload.model_dump(exclude_unset=True))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/admin/api/sources")
    def sources(session: dict[str, Any] = Depends(require_admin)) -> dict[str, Any]:
        del session
        return {"items": list_sources(settings())}

    @app.patch("/admin/api/sources/{source_key}")
    def source_patch(
        source_key: str,
        payload: SourcePatch,
        session: dict[str, Any] = Depends(require_admin),
    ) -> dict[str, Any]:
        del session
        try:
            return patch_source(settings(), source_key, payload.model_dump(exclude_unset=True))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

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

    @app.get("/admin/api/jobs")
    def jobs(
        type: str | None = None,
        status: str | None = None,
        limit: int = 50,
        session: dict[str, Any] = Depends(require_admin),
    ) -> dict[str, Any]:
        del session
        return {"items": [_decode_job(job) for job in list_jobs(settings(), type, status, limit)]}

    @app.get("/admin/api/jobs/{job_key}")
    def job_detail(
        job_key: str,
        session: dict[str, Any] = Depends(require_admin),
    ) -> dict[str, Any]:
        del session
        job = get_job(settings(), job_key)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        decoded = _decode_job(job)
        logs = [_decode_job_log(row) for row in list_job_logs(settings(), job_key)]
        decoded["logs"] = logs or _fallback_job_logs(decoded)
        return decoded

    @app.post("/admin/api/jobs/{job_key}/retry")
    def job_retry(
        job_key: str,
        session: dict[str, Any] = Depends(require_admin),
    ) -> dict[str, Any]:
        del session
        try:
            return _decode_job(retry_job(settings(), job_key))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/admin/api/jobs/{job_key}/cancel")
    def job_cancel(
        job_key: str,
        session: dict[str, Any] = Depends(require_admin),
    ) -> dict[str, Any]:
        del session
        try:
            return _decode_job(cancel_job(settings(), job_key))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/admin/api/index/build")
    def index_build(session: dict[str, Any] = Depends(require_admin)) -> dict[str, Any]:
        del session
        try:
            return _decode_job(create_build_index_job(settings()))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/admin/api/jobs/cleanup")
    def cleanup_job(session: dict[str, Any] = Depends(require_admin)) -> dict[str, Any]:
        del session
        try:
            return _decode_job(create_cleanup_job(settings()))
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

    @app.post("/admin/api/index/{index_key}/activate")
    def index_activate(
        index_key: str,
        session: dict[str, Any] = Depends(require_admin),
    ) -> dict[str, Any]:
        del session
        try:
            _activate_index(settings(), index_key, index_state())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        index = get_index(settings(), index_key)
        if index is None:
            raise HTTPException(status_code=404, detail="Index not found")
        return _decode_index(index)

    @app.post("/admin/api/index/{index_key}/cache/rebuild")
    def index_cache_rebuild(
        index_key: str,
        session: dict[str, Any] = Depends(require_admin),
    ) -> dict[str, Any]:
        del session
        try:
            return _decode_index(rebuild_index_cache(settings(), index_key))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/admin/api/index/{index_key}/verify")
    def index_verify(
        index_key: str,
        session: dict[str, Any] = Depends(require_admin),
    ) -> dict[str, Any]:
        del session
        try:
            verify_index_cache(index_cache_path(settings(), index_key), index_key)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "index_key": index_key}

    @app.get("/admin/api/audits")
    def audits(
        status: str | None = None,
        q: str = "",
        date_from: str = "",
        date_to: str = "",
        limit: int = 50,
        session: dict[str, Any] = Depends(require_admin),
    ) -> dict[str, Any]:
        del session
        return {
            "items": [
                _decode_audit(audit)
                for audit in list_search_audits(
                    settings(),
                    status=status,
                    q=q,
                    date_from=date_from,
                    date_to=date_to,
                    limit=limit,
                )
            ]
        }

    @app.get("/admin/api/audits/{request_id}")
    def audit_detail(
        request_id: str,
        session: dict[str, Any] = Depends(require_admin),
    ) -> dict[str, Any]:
        del session
        audit = get_search_audit(settings(), request_id)
        if audit is None:
            raise HTTPException(status_code=404, detail="Audit not found")
        return _decode_audit(audit)

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

        @app.get("/admin")
        @app.get("/admin/{path:path}")
        def admin_frontend(path: str = "") -> FileResponse:
            del path
            return FileResponse(frontend_dist / "index.html")

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
