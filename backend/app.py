"""FastAPI application.

Request path: retrieve RETRIEVE_K candidates from FAISS -> rerank to
RERANK_TOP_N with a cross-encoder -> generate grounded recommendations.

The route is a plain `def`, not `async def`: embedding, reranking and the
upstream call are all blocking, and a blocking call inside an `async def`
handler runs on the event loop and freezes every other request - including
/health. As a sync route, FastAPI runs it in a threadpool instead.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from config import Settings, get_settings
from embeddings import load_embedding_backend
from generator import GenerationError, Generator, OpenRouterGenerator
from index_store import MenuIndex
from observability import (
    RequestContextMiddleware,
    StageTimer,
    configure_logging,
    get_logger,
)
from rag import run_rag
from rerank import CrossEncoderReranker, Reranker
from schemas import HealthResponse, QueryRequest, QueryResponse, Source

BASE_DIR = Path(__file__).resolve().parent
STORE_DIR = BASE_DIR / "faiss_store"

log = get_logger()
limiter = Limiter(key_func=get_remote_address)


class AppState:
    index: MenuIndex | None = None
    reranker: Reranker | None = None
    generator: Generator | None = None
    startup_error: str | None = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    try:
        backend = load_embedding_backend(settings.embed_backend, settings.embed_model)
        state.index = MenuIndex.load(STORE_DIR, backend, settings.embed_model)
        state.reranker = CrossEncoderReranker(settings.rerank_model)
        state.generator = OpenRouterGenerator(settings)
        state.startup_error = None
        log.info(
            "startup.ready",
            documents=len(state.index),
            embed_model=settings.embed_model,
            llm_model=settings.llm_model,
        )
    except Exception as exc:  # keep the process alive so /health can explain
        state.startup_error = f"{type(exc).__name__}: {exc}"
        state.index = state.reranker = state.generator = None
        log.error("startup.failed", error=state.startup_error)
    yield


def create_app(
    settings: Settings | None = None,
    *,
    lifespan_handler=lifespan,
) -> FastAPI:
    """Build the app. Tests pass lifespan_handler=None to skip model loading."""
    settings = settings or get_settings()
    application = FastAPI(
        title="Foodify",
        description="Retrieval-augmented restaurant menu recommendations.",
        version="2.0.0",
        lifespan=lifespan_handler,
    )
    application.state.limiter = limiter
    application.add_exception_handler(RateLimitExceeded, _rate_limit_handler)
    application.add_middleware(RequestContextMiddleware)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    application.include_router(router)

    # Serve the frontend from the same origin as the API. Same-origin means the
    # browser never makes a cross-origin request, so there is no CORS handshake
    # to misconfigure and no backend URL for the frontend to hardcode.
    # Mounted last so /health and /recommend keep precedence over the catch-all.
    frontend_dir = BASE_DIR.parent / "frontend"
    if frontend_dir.is_dir():
        application.mount(
            "/", StaticFiles(directory=frontend_dir, html=True), name="frontend"
        )

    return application


def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=429,
        content={"detail": "Too many requests. Please slow down."},
    )


# --- dependencies (overridable in tests) -------------------------------------


def get_index() -> MenuIndex:
    if state.index is None:
        raise HTTPException(status_code=503, detail="Search index is not available.")
    return state.index


def get_reranker() -> Reranker:
    if state.reranker is None:
        raise HTTPException(status_code=503, detail="Reranker is not available.")
    return state.reranker


def get_generator() -> Generator:
    if state.generator is None:
        raise HTTPException(status_code=503, detail="Generator is not available.")
    return state.generator


# --- routes -------------------------------------------------------------------

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    ready = all((state.index, state.reranker, state.generator))
    return HealthResponse(
        status="ok" if ready else "not_ready",
        index_loaded=state.index is not None,
        reranker_loaded=state.reranker is not None,
        generator_configured=state.generator is not None,
        document_count=len(state.index) if state.index else None,
        embed_model=settings.embed_model,
        llm_model=settings.llm_model,
        error=state.startup_error,
    )


@router.post("/recommend", response_model=QueryResponse)
@limiter.limit(lambda: get_settings().rate_limit)
def recommend(
    request: Request,
    payload: QueryRequest,
    index: MenuIndex = Depends(get_index),
    reranker: Reranker = Depends(get_reranker),
    generator: Generator = Depends(get_generator),
) -> QueryResponse:
    settings = get_settings()
    timer = StageTimer()

    with timer.stage("retrieve"):
        candidates = index.search(payload.query, k=settings.retrieve_k)

    with timer.stage("rerank"):
        reranked = reranker.rerank(
            payload.query, [doc for doc, _ in candidates], settings.rerank_top_n
        )

    documents = [doc for doc, _ in reranked]

    try:
        with timer.stage("generate"):
            recommendations, message, result = run_rag(
                payload.query, documents, generator
            )
    except GenerationError as exc:
        log.warning(
            "recommend.generation_failed",
            error=str(exc),
            status=exc.status_code,
            **timer.finish(),
        )
        raise HTTPException(
            status_code=exc.status_code,
            detail="Could not generate recommendations right now. Please try again.",
        ) from exc

    log.info(
        "recommend.ok",
        query_length=len(payload.query),
        candidates=len(candidates),
        returned=len(recommendations),
        model=result.model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        cost_usd=result.cost_usd,
        **timer.finish(),
    )

    return QueryResponse(
        recommendations=recommendations,
        message=message,
        sources=[
            Source(
                restaurant=doc.restaurant,
                category=doc.category,
                item=doc.item,
                description=doc.description,
                score=score,
            )
            for doc, score in reranked
        ],
    )


app = create_app()
