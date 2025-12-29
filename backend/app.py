"""
FastAPI entry point.

- Loads a LangChain FAISS vector store from ./faiss_store (or builds it from DATA.json)
- Loads a Llama 3B Instruct model (see backend/model.py)
- Runs a simple RAG pipeline (see backend/rag.py)
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from model import load_model
from rag import run_rag
from schemas import QueryRequest, QueryResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=(
        ["*"]
        if not os.environ.get("CORS_ORIGINS")
        else [o.strip() for o in os.environ["CORS_ORIGINS"].split(",") if o.strip()]
    ),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vector_store: FAISS | None = None
llm = None
tokenizer = None
startup_error: str | None = None

def _workspace_path() -> Path:
    return Path(__file__).resolve().parent

def load_vector_store() -> FAISS:
    base = _workspace_path()
    store_dir = base / "faiss_store"
    data_json = base / "DATA.json"

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Preferred path: load the already-built store (index.faiss + index.pkl)
    if store_dir.exists():
        return FAISS.load_local(
            str(store_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    # Fallback: build it once from DATA.json and save to store_dir
    if not data_json.exists():
        raise FileNotFoundError(
            f"Missing FAISS store ({store_dir}) and missing dataset ({data_json})."
        )

    from vectorstore import load_vectorstore

    store = load_vectorstore(json_path=str(data_json))
    store.save_local(str(store_dir))
    return store

@app.on_event("startup")
def on_startup():
    global vector_store, llm, tokenizer, startup_error
    try:
        vector_store = load_vector_store()
        llm, tokenizer = load_model()
        startup_error = None
    except Exception as e:
        # Keep the API process alive so /health can report what went wrong.
        startup_error = str(e)
        vector_store = None
        llm = None
        tokenizer = None

@app.get("/health")
async def health_check():
    ok = vector_store is not None and llm is not None and tokenizer is not None
    return {
        "status": "ok" if ok else "not_ready",
        "error": startup_error,
        "has_vector_store": vector_store is not None,
        "has_model": llm is not None,
    }

@app.post("/recommend", response_model=QueryResponse)
async def recommend_food(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")
    if vector_store is None or llm is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail=f"Backend not ready. Startup error: {startup_error}",
        )
    try:
        docs = vector_store.similarity_search(request.query, k=3)
        sources = [d.page_content for d in docs]
        response_text = run_rag(request.query, vector_store, llm, tokenizer, k=3)
        return QueryResponse(response=response_text, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), reload=True)