"""Shared fixtures.

The default suite runs fully offline: no model downloads, no API calls. The
embedding backend is faked with deterministic hashed vectors so the real
MenuIndex and FAISS code paths are still exercised, and the generator is a stub
injected through FastAPI's dependency overrides.
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pytest

# Must be set before importing the app - Settings validates at import.
os.environ.setdefault("OPENROUTER_API_KEY", "test-key-not-real")
os.environ.setdefault("RETRIEVE_K", "5")
os.environ.setdefault("RERANK_TOP_N", "3")
os.environ.setdefault("MAX_QUERY_LENGTH", "100")
os.environ.setdefault("LOG_LEVEL", "WARNING")

from fastapi.testclient import TestClient  # noqa: E402

import app as app_module  # noqa: E402
from generator import GenerationResult  # noqa: E402
from index_store import MenuDocument, MenuIndex  # noqa: E402
from rerank import PassthroughReranker  # noqa: E402


class FakeEmbeddingBackend:
    """Deterministic pseudo-embeddings - no model, no network."""

    dim = 32

    def _vector(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.lower().encode()).digest()
        raw = np.frombuffer(digest * 4, dtype=np.uint8)[: self.dim]
        vec = raw.astype(np.float32) - 127.5
        return vec / np.linalg.norm(vec)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._vector(t) for t in texts]).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)


@pytest.fixture
def documents() -> list[MenuDocument]:
    return [
        MenuDocument(
            "Zanzibar Café", "GREENS", "Beet Salad", "Kale, goat cheese, beets"
        ),
        MenuDocument(
            "Zanzibar Café", "TORTILLAS", "Fish Tacos", "Cabbage, mango salsa"
        ),
        MenuDocument(
            "Blue Ocean", "SUSHI ROLLS", "California", "Krab, avocado", "Classic Rolls"
        ),
        MenuDocument(
            "Blue Ocean",
            "SUSHI ROLLS",
            "Spicy Toro",
            "Spicy toro, cucumber",
            "Classic Rolls",
        ),
        MenuDocument(
            "Aladdin", "ENTREES", "Falafel Plate", "Chickpea fritters, hummus, salad"
        ),
    ]


@pytest.fixture
def index(documents) -> MenuIndex:
    return MenuIndex.build(documents, FakeEmbeddingBackend())


class FakeGenerator:
    """Returns canned structured output. `fail_with` makes it raise instead."""

    def __init__(self, payload: dict | None = None, fail_with: Exception | None = None):
        self.payload = payload or {
            "recommendations": [
                {
                    "restaurant": "Zanzibar Café",
                    "dish": "Beet Salad",
                    "why": "Earthy and vegetarian, which matches your request.",
                    "dietary_notes": "",
                }
            ],
            "message": "Here is a dish that fits.",
        }
        self.fail_with = fail_with
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, user: str, *, json_schema=None) -> GenerationResult:
        self.calls.append((system, user))
        if self.fail_with:
            raise self.fail_with
        return GenerationResult(
            text=json.dumps(self.payload),
            model="fake/model",
            prompt_tokens=120,
            completion_tokens=45,
            cost_usd=0.0001,
        )


@pytest.fixture
def generator() -> FakeGenerator:
    return FakeGenerator()


@pytest.fixture
def client(index, generator):
    """TestClient with the real routes and fake dependencies.

    Both the dependency overrides (what the routes receive) and app_module.state
    (what /health reports) are populated: a readiness probe is supposed to
    describe real process state, not injected doubles, so it reads `state`
    directly and has to be set up rather than overridden.
    """
    reranker = PassthroughReranker()
    application = app_module.create_app(lifespan_handler=None)
    application.dependency_overrides[app_module.get_index] = lambda: index
    application.dependency_overrides[app_module.get_reranker] = lambda: reranker
    application.dependency_overrides[app_module.get_generator] = lambda: generator
    app_module.limiter.enabled = False

    previous = (
        app_module.state.index,
        app_module.state.reranker,
        app_module.state.generator,
    )
    app_module.state.index = index
    app_module.state.reranker = reranker
    app_module.state.generator = generator

    with TestClient(application) as test_client:
        yield test_client

    application.dependency_overrides.clear()
    (
        app_module.state.index,
        app_module.state.reranker,
        app_module.state.generator,
    ) = previous
