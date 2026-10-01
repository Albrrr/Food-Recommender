"""Dataset parsing, index behaviour, reranking, and prompt grounding."""

from __future__ import annotations

from pathlib import Path

import pytest

from index_store import MenuDocument, MenuIndex, load_menu_documents
from rag import SYSTEM_PROMPT, build_context, run_rag
from rerank import PassthroughReranker
from tests.conftest import FakeEmbeddingBackend, FakeGenerator

DATA = Path(__file__).resolve().parent.parent / "backend" / "DATA.json"


def test_dataset_parses_all_nesting_levels():
    """The dataset nests unevenly; items inside subgroups must not be dropped."""
    documents = load_menu_documents(DATA)
    assert len(documents) > 3200
    assert len({d.restaurant for d in documents}) == 75
    # Items living under a subgroup (SUSHI ROLLS -> Classic Rolls -> California)
    nested = [d for d in documents if d.subgroup]
    assert len(nested) > 200, "subgroup items were dropped by the parser"


def test_every_document_has_a_description():
    assert all(d.description.strip() for d in load_menu_documents(DATA))


def test_document_text_includes_restaurant_and_subgroup():
    doc = MenuDocument("Blue Ocean", "SUSHI", "California", "Krab", "Classic Rolls")
    assert "Blue Ocean" in doc.text
    assert "Classic Rolls" in doc.text


def test_index_returns_k_documents(index):
    assert len(index.search("anything", k=3)) == 3


def test_index_caps_k_at_corpus_size(index):
    assert len(index.search("anything", k=999)) == 5


def test_index_search_is_deterministic(index):
    assert index.search("falafel", k=3) == index.search("falafel", k=3)


def test_index_roundtrips_through_disk(tmp_path, documents):
    backend = FakeEmbeddingBackend()
    MenuIndex.build(documents, backend).save(tmp_path, "fake-model")
    reloaded = MenuIndex.load(tmp_path, backend, "fake-model")
    assert len(reloaded) == len(documents)


def test_index_refuses_a_mismatched_embedding_model(tmp_path, documents):
    from index_store import IndexMismatchError

    backend = FakeEmbeddingBackend()
    MenuIndex.build(documents, backend).save(tmp_path, "model-a")
    with pytest.raises(IndexMismatchError):
        MenuIndex.load(tmp_path, backend, "model-b")


def test_passthrough_reranker_respects_top_n(documents):
    assert len(PassthroughReranker().rerank("q", documents, 2)) == 2


def test_context_numbers_every_document(documents):
    context = build_context(documents)
    assert context.startswith("1. ")
    assert len(context.splitlines()) == len(documents)


def test_prompt_forbids_invention():
    lowered = SYSTEM_PROMPT.lower()
    assert "never invent" in lowered
    assert "empty recommendations list" in lowered


def test_run_rag_passes_retrieved_documents_to_the_model(documents):
    generator = FakeGenerator()
    run_rag("vegetarian", documents, generator)
    _system, user = generator.calls[0]
    assert "Beet Salad" in user
    assert "vegetarian" in user


def test_run_rag_handles_a_declining_model(documents):
    generator = FakeGenerator(
        payload={"recommendations": [], "message": "Nothing here matches that."}
    )
    recommendations, message, _ = run_rag("gold leaf burrito", documents, generator)
    assert recommendations == []
    assert "Nothing here matches" in message


def test_run_rag_strips_fenced_json(documents):
    generator = FakeGenerator()
    generator.generate = lambda *a, **k: _fenced()  # type: ignore[assignment]
    recommendations, _message, _ = run_rag("q", documents, generator)
    assert recommendations[0].dish == "Fish Tacos"


def _fenced():
    from generator import GenerationResult

    return GenerationResult(
        text='```json\n{"recommendations":[{"restaurant":"Z","dish":"Fish Tacos",'
        '"why":"w","dietary_notes":""}],"message":"m"}\n```',
        model="fake/model",
    )


@pytest.mark.integration
def test_real_cross_encoder_reorders_candidates(documents):
    """Opt-in: downloads the reranker model. Run with `pytest -m integration`."""
    from rerank import CrossEncoderReranker

    reranker = CrossEncoderReranker("Xenova/ms-marco-MiniLM-L-6-v2")
    ranked = reranker.rerank("raw fish sushi roll", documents, top_n=2)
    assert {doc.item for doc, _ in ranked} & {"California", "Spicy Toro"}
