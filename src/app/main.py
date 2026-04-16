"""
FastAPI application factory.

Startup: initializes singleton services (recommender, rec_service).
Middleware: Prometheus metrics collection, CORS.
Routes: delegates to api/ sub-modules.
Frontend: serves the React SPA from web/dist/.
"""
import os
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from src.app import state
from src.app.api.chat import router as chat_router
from src.app.api.ops import router as ops_router
from src.app.api.recommend import router as recommend_router
from src.app.api.bookshelf import router as bookshelf_router
from src.app.api.admin import router as admin_router
from src.core.recommendation_orchestrator import RecommendationOrchestrator
from src.infra.utils import setup_logger

logger = setup_logger(__name__)

# --- Prometheus metrics ---
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
)

# --- App ---
app = FastAPI(
    title="Book Recommender API",
    description="""Book recommendation system combining RAG (hybrid search + reranking)
and RecSys (7-channel recall + LGBMRanker). See `/docs` for full API reference.""",
    version="2.6.0",
)

app.include_router(chat_router)
app.include_router(ops_router)
app.include_router(recommend_router)
app.include_router(bookshelf_router)
app.include_router(admin_router)


# --- Middleware ---

@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    path = request.url.path
    # Skip metrics/health to avoid high-cardinality noise
    if path in ["/metrics", "/health"]:
        return await call_next(request)
    start = time.perf_counter()
    try:
        response = await call_next(request)
        status = str(response.status_code)
    except Exception:
        status = "500"
        raise
    finally:
        elapsed = time.perf_counter() - start
        REQUEST_COUNT.labels(method=request.method, endpoint=path, status_code=status).inc()
        REQUEST_LATENCY.labels(method=request.method, endpoint=path).observe(elapsed)
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Startup ---

@app.on_event("startup")
async def startup_event():
    from src.core.model_loader import ensure_models_exist
    from src.data.stores.metadata_store import metadata_store as _metadata_store
    from src.services.recommend_service import RecommendationService

    logger.info("Checking/downloading models from HF Hub...")
    ensure_models_exist()

    logger.info("Initializing Recommender Engine...")
    state.recommender = RecommendationOrchestrator(metadata_store_inst=_metadata_store)

    logger.info("Initializing Personalized Rec Service...")
    state.rec_service = RecommendationService()
    try:
        state.rec_service.load_resources()
    except Exception as e:
        logger.error(f"Failed to pre-load resources: {e}")

    logger.info("Engines initialized.")


# --- Core endpoints ---

@app.get("/health", summary="Health check", tags=["Admin"])
async def health_check():
    return {"status": "healthy"}


@app.get(
    "/metrics",
    summary="Prometheus metrics",
    responses={200: {"description": "Prometheus text format"}},
    tags=["Admin"],
)
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# --- Frontend serving (React SPA) ---

if os.path.exists("web/dist/assets"):
    app.mount("/assets", StaticFiles(directory="web/dist/assets"), name="assets")

app.mount("/content", StaticFiles(directory="assets"), name="content")


# Catch-all for client-side routing — MUST be defined last
@app.get("/{full_path:path}", include_in_schema=False)
async def serve_react_app(full_path: str):
    if full_path.startswith(("api", "docs", "openapi", "redoc")):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Not Found")
    if os.path.exists("web/dist/index.html"):
        return FileResponse("web/dist/index.html")
    return {"message": "Backend running. Frontend not built (run `cd web && npm run build`)."}


__all__ = ["app"]
