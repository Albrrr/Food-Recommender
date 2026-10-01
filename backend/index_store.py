"""FAISS index over the menu corpus.

Replaces the previous LangChain FAISS wrapper. Two reasons: the index has to be
rebuilt anyway when the embedding model changes, and going direct removes the
langchain/langchain-community dependencies while giving each document real
structured fields instead of one flattened string.

On disk:
    faiss_store/index.faiss     the vectors
    faiss_store/documents.json  the documents, ordered to match vector ids
    faiss_store/meta.json       embedding model + dimension the index was built with

meta.json exists so a mismatch between the index and the configured embedding
model fails loudly at startup. Silently searching an index built by a different
model returns plausible-looking nonsense, which is far worse than an exception.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss

from embeddings import EmbeddingBackend

INDEX_FILE = "index.faiss"
DOCS_FILE = "documents.json"
META_FILE = "meta.json"


@dataclass(frozen=True, slots=True)
class MenuDocument:
    restaurant: str
    category: str
    item: str
    description: str
    subgroup: str = ""

    @property
    def text(self) -> str:
        """The string that gets embedded."""
        group = f", Group: {self.subgroup}" if self.subgroup else ""
        return (
            f"Restaurant: {self.restaurant}, Category: {self.category}{group}, "
            f"Item: {self.item}, Description: {self.description}"
        )

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class IndexMismatchError(RuntimeError):
    """Index on disk was built with a different embedding model."""


def _description_of(value: object) -> str | None:
    """Extract a usable description, or None if this node is not a leaf item."""
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, dict):
        description = value.get("description")
        if isinstance(description, str) and description.strip():
            return description.strip()
        return None
    if isinstance(value, list):
        # Variant lists, e.g. WINGS -> "No Heat" -> ["Naked", "Lemon Pepper", ...]
        options = [str(v).strip() for v in value if str(v).strip()]
        return "Available in: " + ", ".join(options) if options else None
    return None


def _collect(
    restaurant: str,
    category: str,
    node: dict,
    out: list["MenuDocument"],
    subgroup: str = "",
) -> None:
    """Walk a category, recursing into subgroups.

    The dataset nests unevenly: most categories map item -> description, but ~300
    of them insert a subgroup level (SUSHI ROLLS -> Classic Rolls -> California).
    A non-recursive parser silently drops every item inside those - 486 of them,
    about 14% of the corpus.
    """
    for name, value in node.items():
        description = _description_of(value)
        if description is not None:
            out.append(
                MenuDocument(
                    restaurant=restaurant,
                    category=category,
                    item=str(name),
                    description=description,
                    subgroup=subgroup,
                )
            )
        elif isinstance(value, dict):
            _collect(restaurant, category, value, out, subgroup=str(name))


def load_menu_documents(data_path: Path) -> list[MenuDocument]:
    """Flatten DATA.json into leaf menu items."""
    with data_path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    collected: list[MenuDocument] = []
    for restaurant, categories in data.items():
        if not isinstance(categories, dict):
            continue
        for category, items in categories.items():
            if isinstance(items, dict):
                _collect(str(restaurant), str(category), items, collected)

    # Deduplicate. ~85 dishes are listed under more than one category (a side
    # that appears on both the lunch and dinner menu, say). Without this the
    # same dish can occupy several of the RERANK_TOP_N slots and the model is
    # handed the same option three times.
    seen: set[tuple[str, str, str]] = set()
    documents: list[MenuDocument] = []
    for doc in collected:
        key = (
            doc.restaurant.strip().lower(),
            doc.item.strip().lower(),
            doc.description.strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        documents.append(doc)
    return documents


class MenuIndex:
    def __init__(
        self,
        index: faiss.Index,
        documents: list[MenuDocument],
        backend: EmbeddingBackend,
    ) -> None:
        self._index = index
        self._documents = documents
        self._backend = backend

    def __len__(self) -> int:
        return len(self._documents)

    @classmethod
    def build(
        cls, documents: list[MenuDocument], backend: EmbeddingBackend
    ) -> "MenuIndex":
        vectors = backend.embed_documents([d.text for d in documents])
        index = faiss.IndexFlatIP(backend.dim)  # vectors are L2-normalized -> cosine
        index.add(vectors)
        return cls(index, documents, backend)

    def save(self, directory: Path, model_name: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / INDEX_FILE))
        (directory / DOCS_FILE).write_text(
            json.dumps([d.to_dict() for d in self._documents], ensure_ascii=False),
            encoding="utf-8",
        )
        (directory / META_FILE).write_text(
            json.dumps(
                {
                    "embed_model": model_name,
                    "dim": self._backend.dim,
                    "count": len(self._documents),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls, directory: Path, backend: EmbeddingBackend, model_name: str
    ) -> "MenuIndex":
        index_path = directory / INDEX_FILE
        if not index_path.exists():
            raise FileNotFoundError(
                f"No FAISS index at {index_path}. Build one with "
                f"`python scripts/build_index.py`."
            )

        meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
        if meta.get("embed_model") != model_name or meta.get("dim") != backend.dim:
            raise IndexMismatchError(
                f"Index was built with {meta.get('embed_model')!r} "
                f"(dim {meta.get('dim')}) but the app is configured for "
                f"{model_name!r} (dim {backend.dim}). Rebuild with "
                f"`python scripts/build_index.py`."
            )

        raw = json.loads((directory / DOCS_FILE).read_text(encoding="utf-8"))
        documents = [MenuDocument(**d) for d in raw]
        return cls(faiss.read_index(str(index_path)), documents, backend)

    def search(self, query: str, k: int) -> list[tuple[MenuDocument, float]]:
        vector = self._backend.embed_query(query).reshape(1, -1)
        k = min(k, len(self._documents))
        scores, ids = self._index.search(vector, k)
        return [
            (self._documents[i], float(score))
            for i, score in zip(ids[0], scores[0])
            if i != -1
        ]
