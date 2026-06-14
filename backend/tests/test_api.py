from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient

import app as app_module
from app import create_app
from core import db_connection, get_settings, hash_password
from search import Candidate


def test_health_and_config() -> None:
    client = TestClient(create_app())

    health = client.get("/api/health")
    config = client.get("/api/config")

    assert health.status_code == 200
    assert health.json()["ok"] is True
    assert health.json()["problem_count"] > 0
    assert config.status_code == 200
    assert config.json()["top_display"] == 20


def test_search_stream_with_mocks(monkeypatch) -> None:
    import search as search_module

    class FakeIndex:
        def embed_query_vector(self, *args, **kwargs) -> list[float]:
            return [1.0, 0.0]

        def search_with_vector(self, *args, **kwargs) -> list[Candidate]:
            return [
                Candidate(
                    problem_id="d_1",
                    title="Sample",
                    url="https://example.com",
                    original_text="original",
                    statement="statement",
                    abstract="abstract",
                    embedding_score=0.9,
                )
            ]

        def rerank(self, *args, **kwargs) -> list[Candidate]:
            candidate = self.search_with_vector()[0]
            return [Candidate(**{**candidate.__dict__, "rerank_score": 0.8})]

    def fake_index() -> FakeIndex:
        return FakeIndex()

    def fake_rewrite(*args, **kwargs):
        from search import RewriteResult

        return RewriteResult("rewritten statement", "rewritten abstract", "raw")

    app = create_app()
    app.dependency_overrides = {}
    monkeypatch.setattr(app_module, "legacy_index", fake_index)
    monkeypatch.setattr(search_module, "rewrite_query", fake_rewrite)
    client = TestClient(app)

    response = client.post(
        "/api/search",
        json={"query_text": "hello", "use_rewrite": True, "use_rerank": True},
    )
    events = [json.loads(line) for line in response.text.splitlines()]

    assert response.status_code == 200
    assert [event["type"] for event in events][-2:] == ["candidates", "done"]
    assert any(event["type"] == "rewrite" for event in events)
    assert events[-2]["candidates"][0]["title"] == "Sample"


def test_import_dry_run_and_confirm(monkeypatch, tmp_path: Path) -> None:
    base_settings = get_settings()
    storage = replace(
        base_settings.storage,
        db_path=tmp_path / "app.sqlite3",
        upload_dir=tmp_path / "uploads",
    )
    test_settings = replace(base_settings, storage=storage)
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
        job = confirm.json()
        assert job["status"] == "succeeded"
        assert job["result"]["new"] == 1

    with db_connection(test_settings) as conn:
        source = conn.execute("SELECT * FROM sources WHERE key = 'CodeForces'").fetchone()
        problem = conn.execute("SELECT * FROM problems WHERE key = 'CodeForces/1A'").fetchone()
        artifact = conn.execute(
            "SELECT * FROM artifacts WHERE key = ?",
            (problem["text_key"],),
        ).fetchone()

    assert source is not None
    assert problem["title"] == "Theatre Square"
    assert artifact["kind"] == "problem_text"
