"""Endpoint behaviour."""

from __future__ import annotations

import pytest

import app as app_module
from generator import GenerationError
from tests.conftest import FakeGenerator


def test_health_reports_each_dependency(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["index_loaded"] is True
    assert body["reranker_loaded"] is True
    assert body["generator_configured"] is True
    assert body["document_count"] == 5


def test_recommend_returns_structured_recommendations(client):
    response = client.post("/recommend", json={"query": "something vegetarian"})
    assert response.status_code == 200
    body = response.json()
    assert body["recommendations"][0]["dish"] == "Beet Salad"
    assert body["message"]
    assert body["sources"]
    assert {"restaurant", "category", "item", "description", "score"} <= set(
        body["sources"][0]
    )


def test_sources_are_capped_at_rerank_top_n(client):
    body = client.post("/recommend", json={"query": "sushi"}).json()
    assert len(body["sources"]) == 3  # RERANK_TOP_N in conftest


def test_empty_query_is_rejected(client):
    assert client.post("/recommend", json={"query": "   "}).status_code == 422


def test_overlong_query_is_rejected_before_the_model(client, generator):
    response = client.post("/recommend", json={"query": "x" * 101})
    assert response.status_code == 422
    assert generator.calls == []  # never reached the model


def test_missing_query_is_rejected(client):
    assert client.post("/recommend", json={}).status_code == 422


def test_generation_failure_surfaces_upstream_status(client, index):
    application = client.app
    application.dependency_overrides[app_module.get_generator] = lambda: FakeGenerator(
        fail_with=GenerationError("upstream down", status_code=503)
    )
    response = client.post("/recommend", json={"query": "sushi"})
    assert response.status_code == 503


def test_internal_errors_do_not_leak_details(client):
    application = client.app
    application.dependency_overrides[app_module.get_generator] = lambda: FakeGenerator(
        fail_with=GenerationError("secret internal path /srv/keys", status_code=502)
    )
    body = client.post("/recommend", json={"query": "sushi"}).json()
    assert "secret internal path" not in body["detail"]


def test_malformed_model_output_is_rejected(client):
    application = client.app
    bad = FakeGenerator()
    bad.generate = lambda *a, **k: _raw("not json at all")  # type: ignore[assignment]
    application.dependency_overrides[app_module.get_generator] = lambda: bad
    assert client.post("/recommend", json={"query": "sushi"}).status_code == 502


def test_request_id_is_returned(client):
    response = client.get("/health")
    assert response.headers.get("X-Request-ID")


def test_unavailable_index_returns_503(client):
    application = client.app
    application.dependency_overrides.pop(app_module.get_index)
    app_module.state.index = None
    assert client.post("/recommend", json={"query": "sushi"}).status_code == 503


def _raw(text: str):
    from generator import GenerationResult

    return GenerationResult(text=text, model="fake/model")


@pytest.mark.parametrize("path", ["/health"])
def test_health_is_cheap(client, path, generator):
    client.get(path)
    assert generator.calls == []
