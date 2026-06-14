from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

import app as app_module
from app import create_app
from core import db_connection, ensure_database, get_settings, hash_password, text_key, utc_now
from search import IndexState, RewriteResult


def wait_for_job(client: TestClient, key: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/admin/api/jobs/{key}")
        assert response.status_code == 200
        job = response.json()
        if job["status"] in {"succeeded", "blocked", "failed"}:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job did not finish: {key}")


def test_health_and_config() -> None:
    client = TestClient(create_app())

    health = client.get("/api/health")
    config = client.get("/api/config")

    assert health.status_code == 200
    assert health.json()["ok"] is True
    assert health.json()["problem_count"] > 0
    assert config.status_code == 200
    assert config.json()["top_display"] == 20


def test_search_stream_with_mocks(monkeypatch, tmp_path: Path) -> None:
    import search as search_module

    base_settings = get_settings()
    storage = replace(
        base_settings.storage,
        db_path=tmp_path / "app.sqlite3",
        upload_dir=tmp_path / "uploads",
        index_cache_dir=tmp_path / "index_cache",
    )
    test_settings = replace(base_settings, storage=storage)
    ensure_database(test_settings)

    state = IndexState()
    state.current = search_module.LoadedIndex(
        key="i:test",
        problem_keys=["d_1"],
        titles=["Sample"],
        urls=["https://example.com"],
        texts={
            "clean": ["original"],
            "statement": ["statement"],
            "abstract": ["abstract"],
            "abstract_zh": ["abstract zh"],
        },
        matrices={
            "clean": np.array([[1.0, 0.0]], dtype=np.float32),
            "statement": np.array([[1.0, 0.0]], dtype=np.float32),
            "abstract": np.array([[0.8, 0.2]], dtype=np.float32),
            "abstract_zh": np.array([[0.7, 0.3]], dtype=np.float32),
        },
        load_mode="ram",
    )

    def fake_rewrite(*args, **kwargs):
        from search import RewriteResult

        return RewriteResult(
            statement="rewritten statement",
            abstract="rewritten abstract",
            abstract_zh="rewritten abstract zh",
            clean="rewritten clean",
            raw="raw",
        )

    def fake_embed(*args, **kwargs) -> np.ndarray:
        texts = args[1]
        return np.array([[1.0, 0.0]] * len(texts), dtype=np.float32)

    def fake_rerank(*args, **kwargs) -> list[float]:
        return [0.8]

    monkeypatch.setattr(app_module, "settings", lambda: test_settings)
    monkeypatch.setattr(app_module, "index_state", lambda: state)
    monkeypatch.setenv(test_settings.admin.password_hash_env, hash_password("secret"))
    monkeypatch.setenv(test_settings.admin.signing_secret_env, "test-signing-secret")
    monkeypatch.setattr(search_module, "rewrite_query", fake_rewrite)
    monkeypatch.setattr(search_module, "embed_texts", fake_embed)
    monkeypatch.setattr(search_module, "rerank_documents", fake_rerank)

    with TestClient(create_app()) as client:
        response = client.post(
            "/api/search",
            json={"query_text": "hello", "use_rewrite": True, "use_rerank": True},
        )
        events = [json.loads(line) for line in response.text.splitlines()]

        assert response.status_code == 200
        assert [event["type"] for event in events][-2:] == ["candidates", "done"]
        assert any(event["type"] == "rewrite" for event in events)
        assert events[-2]["candidates"][0]["title"] == "Sample"

        login = client.post("/admin/api/auth/login", json={"password": "secret"})
        csrf = client.cookies.get("admin_csrf")
        assert login.status_code == 200
        assert csrf
        audits = client.get("/admin/api/audits")
        assert audits.status_code == 200
        assert audits.json()["items"][0]["query"] == "hello"
        assert audits.json()["items"][0]["status"] == "succeeded"

    with db_connection(test_settings) as conn:
        audit = conn.execute("SELECT * FROM search_audits").fetchone()

    assert audit["query"] == "hello"
    assert json.loads(audit["result"])["top"][0]["title"] == "Sample"


def test_search_requires_active_index(monkeypatch) -> None:
    monkeypatch.setattr(app_module, "index_state", lambda: IndexState())
    client = TestClient(create_app())

    response = client.post(
        "/api/search",
        json={"query_text": "hello", "use_rewrite": False, "use_rerank": False},
    )

    assert response.status_code == 503


def test_import_dry_run_and_confirm(monkeypatch, tmp_path: Path) -> None:
    base_settings = get_settings()
    storage = replace(
        base_settings.storage,
        db_path=tmp_path / "app.sqlite3",
        upload_dir=tmp_path / "uploads",
    )
    test_settings = replace(
        base_settings,
        storage=storage,
        jobs=replace(base_settings.jobs, poll_seconds=0),
    )
    monkeypatch.setattr(app_module, "settings", lambda: test_settings)
    monkeypatch.setenv(test_settings.admin.password_hash_env, hash_password("secret"))
    monkeypatch.setenv(test_settings.admin.signing_secret_env, "test-signing-secret")

    line = json.dumps(
        {
            "id": "CodeForces/1A",
            "title": "Theatre Square",
            "text": "Calculate paving stones.",
            "url": "https://codeforces.com/problemset/problem/1/A",
        }
    )

    with TestClient(create_app()) as client:
        login = client.post("/admin/api/auth/login", json={"password": "secret"})
        csrf = client.cookies.get("admin_csrf")
        assert login.status_code == 200
        assert csrf

        dry_run = client.post(
            "/admin/api/import/dry-run",
            data={"mode": "upsert"},
            files={"file": ("problems.jsonl", line.encode("utf-8"), "application/jsonl")},
            headers={"X-CSRF-Token": csrf},
        )
        assert dry_run.status_code == 200
        payload = dry_run.json()
        assert payload["stats"]["new"] == 1
        assert payload["stats"]["errors"] == []

        confirm = client.post(
            f"/admin/api/import/{payload['job_key']}/confirm",
            headers={"X-CSRF-Token": csrf},
        )
        assert confirm.status_code == 200
        job = wait_for_job(client, payload["job_key"])
        assert job["status"] == "succeeded"
        assert job["result"]["new"] == 1

        problems = client.get("/admin/api/problems")
        assert problems.status_code == 200
        assert problems.json()["total"] == 1
        assert problems.json()["items"][0]["title"] == "Theatre Square"

        patched_problem = client.patch(
            "/admin/api/problems/CodeForces/1A",
            json={"title": "Updated Theatre Square"},
            headers={"X-CSRF-Token": csrf},
        )
        assert patched_problem.status_code == 200
        assert patched_problem.json()["title"] == "Updated Theatre Square"

        sources = client.get("/admin/api/sources")
        assert sources.status_code == 200
        assert sources.json()["items"][0]["problem_count"] == 1

        patched_source = client.patch(
            "/admin/api/sources/CodeForces",
            json={"name": "CodeForces Archive"},
            headers={"X-CSRF-Token": csrf},
        )
        assert patched_source.status_code == 200
        assert patched_source.json()["name"] == "CodeForces Archive"

        batch = client.post(
            "/admin/api/problems/batch-disable",
            json={"keys": ["CodeForces/1A"]},
            headers={"X-CSRF-Token": csrf},
        )
        assert batch.status_code == 200
        assert batch.json()["updated"] == 1

        jobs = client.get("/admin/api/jobs?type=import")
        assert jobs.status_code == 200
        assert jobs.json()["items"][0]["key"] == payload["job_key"]

    with db_connection(test_settings) as conn:
        source = conn.execute("SELECT * FROM sources WHERE key = 'CodeForces'").fetchone()
        problem = conn.execute("SELECT * FROM problems WHERE key = 'CodeForces/1A'").fetchone()
        artifact = conn.execute(
            "SELECT * FROM artifacts WHERE key = ?",
            (problem["text_key"],),
        ).fetchone()

    assert source is not None
    assert source["name"] == "CodeForces Archive"
    assert problem["title"] == "Updated Theatre Square"
    assert problem["enabled"] == 0
    assert artifact["kind"] == "problem_text"


def test_index_build_activate_and_health(monkeypatch, tmp_path: Path) -> None:
    import pipeline

    base_settings = get_settings()
    storage = replace(
        base_settings.storage,
        db_path=tmp_path / "app.sqlite3",
        upload_dir=tmp_path / "uploads",
        index_cache_dir=tmp_path / "index_cache",
    )
    test_settings = replace(
        base_settings,
        storage=storage,
        jobs=replace(base_settings.jobs, poll_seconds=0),
    )
    ensure_database(test_settings)
    state = IndexState()
    monkeypatch.setattr(app_module, "settings", lambda: test_settings)
    monkeypatch.setattr(app_module, "index_state", lambda: state)
    monkeypatch.setenv(test_settings.admin.password_hash_env, hash_password("secret"))
    monkeypatch.setenv(test_settings.admin.signing_secret_env, "test-signing-secret")

    problem_text = "Build through API."
    problem_text_key = text_key(problem_text)
    with db_connection(test_settings) as conn:
        with conn:
            conn.execute(
                "INSERT INTO sources(key, name, updated_at) VALUES ('CF', 'CF', ?)",
                (utc_now(),),
            )
            conn.execute(
                """
                INSERT INTO artifacts(key, kind, text, status, attempts, updated_at)
                VALUES (?, 'problem_text', ?, 'succeeded', 0, ?)
                """,
                (problem_text_key, problem_text, utc_now()),
            )
            conn.execute(
                """
                INSERT INTO problems(key, source_key, title, url, text_key, updated_at)
                VALUES ('CF/2A', 'CF', 'API Build', 'https://example.com/2A', ?, ?)
                """,
                (problem_text_key, utc_now()),
            )

    def fake_rewrite(*args, **kwargs) -> RewriteResult:
        return RewriteResult(
            statement="Statement view",
            abstract="Abstract view",
            abstract_zh="Chinese abstract",
            clean="Clean view",
            raw="raw",
        )

    def fake_embed(*args, **kwargs) -> np.ndarray:
        texts = args[1]
        return np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [0.5, 0.5],
            ],
            dtype=np.float32,
        )[: len(texts)]

    monkeypatch.setattr(pipeline, "rewrite_query", fake_rewrite)
    monkeypatch.setattr(pipeline, "embed_texts", fake_embed)

    with TestClient(create_app()) as client:
        login = client.post("/admin/api/auth/login", json={"password": "secret"})
        csrf = client.cookies.get("admin_csrf")
        assert login.status_code == 200
        assert csrf

        build = client.post("/admin/api/index/build", headers={"X-CSRF-Token": csrf})
        assert build.status_code == 200
        build_job = wait_for_job(client, build.json()["key"])
        assert build_job["status"] == "succeeded"
        index_key = build_job["result"]["index_key"]

        activate = client.post(
            f"/admin/api/index/{index_key}/activate",
            headers={"X-CSRF-Token": csrf},
        )
        assert activate.status_code == 200
        assert activate.json()["status"] == "active"

        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["loaded_index_key"] == index_key
        assert health.json()["problem_count"] == 1
