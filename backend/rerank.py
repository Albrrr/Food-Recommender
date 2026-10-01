"""Cross-encoder reranking.

The FAISS search is a bi-encoder: queries and documents are embedded
independently, so it can only measure whether two vectors sit near each other,
not whether a dish actually satisfies a request. Menu descriptions are short and
share heavy vocabulary ("grilled", "fresh", "house-made"), which produces
clusters of near-ties where picking a top 3 is close to arbitrary.

A cross-encoder reads the query and document *together* and scores the pair
directly. It is far too slow to run over all 3,347 documents, which is why it
runs second: retrieve RETRIEVE_K cheaply, rerank those down to RERANK_TOP_N
precisely.
"""

from __future__ import annotations

from typing import Protocol

from index_store import MenuDocument


class Reranker(Protocol):
    def rerank(
        self, query: str, documents: list[MenuDocument], top_n: int
    ) -> list[tuple[MenuDocument, float]]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(model_name=model_name)

    def rerank(
        self, query: str, documents: list[MenuDocument], top_n: int
    ) -> list[tuple[MenuDocument, float]]:
        if not documents:
            return []

        # One batched call - scoring pairs individually costs an order of
        # magnitude in latency.
        scores = list(self._model.rerank(query, [d.text for d in documents]))

        ranked = sorted(zip(documents, scores), key=lambda pair: pair[1], reverse=True)
        # Scores are unbounded relevance logits, not probabilities: rank by them,
        # never threshold at an arbitrary cutoff.
        return [(doc, float(score)) for doc, score in ranked[:top_n]]


class PassthroughReranker:
    """No-op reranker, for tests and for RERANK_TOP_N >= RETRIEVE_K setups."""

    def rerank(
        self, query: str, documents: list[MenuDocument], top_n: int
    ) -> list[tuple[MenuDocument, float]]:
        return [(doc, 0.0) for doc in documents[:top_n]]
