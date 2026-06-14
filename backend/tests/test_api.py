from __future__ import annotations

import json

from fastapi.testclient import TestClient

import app as app_module
from app import create_app
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
