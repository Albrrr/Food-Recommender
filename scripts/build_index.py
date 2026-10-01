#!/usr/bin/env python
"""Rebuild the FAISS index from DATA.json.

Run this whenever the dataset or the embedding model changes - the index stores
the model name and vector dimension it was built with, and the app refuses to
start against a mismatched index.

    python scripts/build_index.py
    python scripts/build_index.py --backend fastembed --model BAAI/bge-base-en-v1.5

Deliberately does not import Settings: building an index is an offline job and
should not require an LLM API key.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from embeddings import load_embedding_backend  # noqa: E402
from index_store import MenuIndex, load_menu_documents  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        default=os.environ.get("EMBED_BACKEND", "sentence-transformers"),
        choices=["sentence-transformers", "fastembed"],
    )
    parser.add_argument(
        "--model", default=os.environ.get("EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    )
    parser.add_argument("--data", default=str(BACKEND_DIR / "DATA.json"))
    parser.add_argument("--out", default=str(BACKEND_DIR / "faiss_store"))
    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.out)

    if not data_path.exists():
        print(f"error: dataset not found at {data_path}", file=sys.stderr)
        return 1

    print(f"Loading documents from {data_path.name} ...")
    documents = load_menu_documents(data_path)
    restaurants = {d.restaurant for d in documents}
    print(f"  {len(documents):,} menu items across {len(restaurants)} restaurants")

    print(f"Loading embedding model {args.model} ({args.backend}) ...")
    backend = load_embedding_backend(args.backend, args.model)
    print(f"  dimension: {backend.dim}")

    # Remove the old store rather than writing over it, so a stale file from a
    # previous model cannot be picked up.
    if out_dir.exists():
        print(f"Removing existing index at {out_dir} ...")
        shutil.rmtree(out_dir)

    print(f"Embedding {len(documents):,} documents ...")
    started = time.perf_counter()
    index = MenuIndex.build(documents, backend)
    elapsed = time.perf_counter() - started
    print(f"  done in {elapsed:.1f}s ({len(documents) / elapsed:.0f} docs/s)")

    index.save(out_dir, args.model)
    print(f"\nWrote index to {out_dir}")
    print(f"  documents : {len(index):,}")
    print(f"  dimension : {backend.dim}")
    print(f"  model     : {args.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
