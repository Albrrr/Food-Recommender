"""Embedding backends.

Two interchangeable implementations behind one protocol:

  sentence-transformers  Qwen3-Embedding-0.6B (default) - stronger retrieval,
                         pulls torch, ~1.2GB of weights.
  fastembed              ONNX models such as BAAI/bge-base-en-v1.5 - no torch,
                         ~0.2GB, faster to cold-start.

Switch with EMBED_BACKEND / EMBED_MODEL. Changing either invalidates the FAISS
index (the vector dimension changes), so rebuild it with scripts/build_index.py.

Note the query/document asymmetry: instruction-tuned embedding models such as
Qwen3 expect a prompt prefix on the *query* side only. Getting that backwards
degrades retrieval silently, so the two paths are separate methods here rather
than one `encode` with a flag callers can forget.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EmbeddingBackend(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Normalize so inner product equals cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


class SentenceTransformerBackend:
    """Qwen3-Embedding and other sentence-transformers models."""

    def __init__(self, model_name: str, batch_size: int = 16) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._batch_size = batch_size
        self.dim = int(self._model.get_sentence_embedding_dimension())
        # Qwen3-Embedding ships a "query" prompt in its config; other models may not.
        prompts = getattr(self._model, "prompts", None) or {}
        self._query_prompt = "query" if "query" in prompts else None

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=len(texts) > 256,
            convert_to_numpy=True,
        )
        return _l2_normalize(np.asarray(vectors, dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        kwargs = {"convert_to_numpy": True}
        if self._query_prompt:
            kwargs["prompt_name"] = self._query_prompt
        vectors = self._model.encode([text], **kwargs)
        return _l2_normalize(np.asarray(vectors, dtype=np.float32))[0]


class FastEmbedBackend:
    """ONNX models via fastembed - no torch, much smaller image."""

    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)
        spec = next(
            m for m in TextEmbedding.list_supported_models() if m["model"] == model_name
        )
        self.dim = int(spec["dim"])

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vectors = np.asarray(list(self._model.embed(texts)), dtype=np.float32)
        return _l2_normalize(vectors)

    def embed_query(self, text: str) -> np.ndarray:
        vectors = np.asarray(list(self._model.query_embed([text])), dtype=np.float32)
        return _l2_normalize(vectors)[0]


def load_embedding_backend(backend: str, model_name: str) -> EmbeddingBackend:
    if backend == "sentence-transformers":
        return SentenceTransformerBackend(model_name)
    if backend == "fastembed":
        return FastEmbedBackend(model_name)
    raise ValueError(f"Unknown embedding backend: {backend!r}")
