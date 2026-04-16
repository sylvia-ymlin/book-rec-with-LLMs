"""Recommendation endpoints: semantic search and personalized recommendations."""
from fastapi import APIRouter, HTTPException, Path, Query
from pydantic import BaseModel, Field
from typing import List, Optional

from src.app import state
from src.infra.utils import setup_logger
from src.services.personal_recommend_handler import (
    parse_request_params,
    resolve_seed_from_intent,
    get_ab_diversity_config,
    enrich_personal_results,
)

logger = setup_logger(__name__)

router = APIRouter(tags=["Recommendations"])


# --- Request / Response models ---

class RecommendationRequest(BaseModel):
    """Request body for semantic + RAG-based recommendations."""

    query: str = Field(..., description="Natural language query (e.g. 'a thriller with plot twists')")
    category: str = Field(default="All", description="Filter by category (e.g. Fiction, Romance)")
    user_id: Optional[str] = Field(default="local", description="User identifier for personalization")
    use_agentic: Optional[bool] = Field(
        default=False,
        description="Enable LangGraph workflow: Router → Retrieve → Evaluate → Web Fallback",
    )
    fast: Optional[bool] = Field(
        default=False,
        description="Skip rerank for ~150ms latency (RRF only)",
    )
    async_rerank: Optional[bool] = Field(
        default=False,
        description="Return RRF first, rerank in background; next request gets cached",
    )
    experiment_id: Optional[str] = Field(default=None, description="A/B experiment ID for variant assignment")
    ab_variant: Optional[str] = Field(default=None, description="Force variant: 'control' | 'treatment'")

    model_config = {
        "json_schema_extra": {
            "examples": [{"query": "a romantic comedy set in New York", "category": "Fiction"}]
        }
    }


class FeatureContribution(BaseModel):
    """SHAP-style feature contribution for explainability."""

    feature: str = Field(..., description="Feature name (e.g. 'title_similarity')")
    contribution: float = Field(..., description="Contribution score")
    direction: str = Field(..., description="'positive' or 'negative'")


class BookResponse(BaseModel):
    """Single book in a recommendation response."""

    isbn: str = Field(..., description="ISBN-10 or ISBN-13")
    title: str
    authors: str
    description: str
    thumbnail: str = Field(..., description="Cover image URL or path")
    caption: str = Field(default="", description="One-line summary")
    tags: List[str] = Field(default_factory=list)
    average_rating: float = Field(default=0.0, ge=0, le=5)
    explanations: List[FeatureContribution] = Field(
        default_factory=list, description="Feature contributions"
    )


class RecommendationResponse(BaseModel):
    """Response with list of recommended books."""

    recommendations: List[BookResponse] = Field(..., description="Ordered list of recommended books")


# --- Endpoints ---

@router.post(
    "/recommend",
    response_model=RecommendationResponse,
    summary="Semantic recommendations",
    description="""Generate book recommendations via RAG pipeline:
- **Hybrid search**: BM25 + Dense vector, RRF fusion
- **Rerank**: Cross-Encoder or ONNX (~2x faster)
- **use_agentic=true**: LangGraph Router → Retrieve → Evaluate → Web Fallback
- **fast=true**: Skip rerank (~150ms latency)
- **async_rerank=true**: Return RRF first, rerank in background; next request gets cached
""",
    responses={
        200: {"description": "List of recommended books"},
        500: {"description": "Processing error"},
        503: {"description": "Service not ready"},
    },
)
async def get_recommendations(request: RecommendationRequest):
    if not state.recommender:
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        user_id = request.user_id or "local"
        enable_diversity = True
        if request.experiment_id:
            from src.core.ab_experiments import get_experiment_config, log_experiment
            from src.infra.config import AB_EXPERIMENTS_ENABLED

            if AB_EXPERIMENTS_ENABLED:
                cfg = get_experiment_config(user_id, request.experiment_id, request.ab_variant)
                enable_diversity = cfg.get("enable_diversity_rerank", True)
                variant = "treatment" if enable_diversity else "control"
                log_experiment(request.experiment_id, user_id, variant)

        results = await state.recommender.get_recommendations(
            query=request.query,
            category=request.category,
            user_id=user_id,
            use_agentic=request.use_agentic or False,
            fast=request.fast or False,
            async_rerank=request.async_rerank or False,
            enable_diversity_rerank=enable_diversity,
        )
        return {"recommendations": results}
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/api/recommend/similar/{isbn}",
    response_model=RecommendationResponse,
    summary="Similar books by ISBN",
    description="""Content-based similar books by vector similarity.
- Works for new users (no history required)
- Params: `k` (default 10), `category` (default All)
""",
    responses={
        200: {"description": "List of similar books"},
        500: {"description": "Error"},
        503: {"description": "Service not ready"},
    },
)
def get_similar_books(
    isbn: str = Path(..., description="ISBN of the seed book"),
    k: int = Query(10, ge=1, le=50, description="Number of similar books to return"),
    category: str = Query("All", description="Filter by category"),
):
    if not state.recommender:
        raise HTTPException(status_code=503, detail="Service not ready")
    try:
        results = state.recommender.get_similar_books(isbn=isbn, k=k, category=category)
        return {"recommendations": results}
    except Exception as e:
        logger.error(f"get_similar_books error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/api/recommend/personal",
    response_model=RecommendationResponse,
    summary="Personalized recommendations",
    description="""Personalized recommendations from reading history.
Uses 7-channel recall (ItemCF/UserCF/Swing/Item2Vec/SASRec/YoutubeDNN/Popularity) + LGBMRanker.

| Param | Description |
|-------|-------------|
| user_id | User identifier (default: local) |
| top_k | Number of results (default: 10) |
| recent_isbns | Comma-separated ISBNs from current session (injected into SASRec) |
| intent_query | Zero-shot intent for cold-start users (LLM infers categories → seeds SASRec) |
| experiment_id | A/B experiment ID |
| ab_variant | Force variant: control \\| treatment |
""",
    responses={
        200: {"description": "Personalized recommendations"},
        500: {"description": "Error"},
        503: {"description": "Service not ready"},
    },
)
def personalized_recommendations(
    user_id: str = "local",
    top_k: int = 10,
    limit: Optional[int] = None,
    recent_isbns: Optional[str] = None,
    intent_query: Optional[str] = None,
    experiment_id: Optional[str] = None,
    ab_variant: Optional[str] = None,
):
    effective_user_id, k, real_time_seq = parse_request_params(
        user_id, top_k, limit, recent_isbns, intent_query
    )

    # Zero-shot intent probing for cold-start: seed from query when no recent ISBNs
    if not real_time_seq and state.recommender:
        seed = resolve_seed_from_intent(intent_query or "", effective_user_id, state.recommender)
        if seed:
            real_time_seq = seed

    if not state.rec_service:
        raise HTTPException(status_code=503, detail="Service not ready")

    enable_diversity = get_ab_diversity_config(effective_user_id, experiment_id, ab_variant)

    try:
        recs = state.rec_service.get_recommendations(
            effective_user_id,
            top_k=k,
            real_time_sequence=real_time_seq,
            enable_diversity_rerank=enable_diversity,
        )
        results = enrich_personal_results(
            recs,
            state.recommender.vector_db.get_book_details if state.recommender else lambda _: {},
        )
        return {"recommendations": results}
    except Exception as e:
        logger.error(f"Error in personalized rec: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/api/intent/probe",
    summary="Intent probing",
    description="""Zero-shot intent probing for cold-start users.
LLM infers categories, emotions, and keywords from a free-text query.
- Param: `query` (str)
- Response: `{categories: [...], emotions: [...], keywords: [...]}`
""",
    responses={200: {"description": "Inferred intent"}, 500: {"description": "Error"}},
)
def probe_intent_endpoint(query: str = ""):
    from src.rag.intent_prober import probe_intent

    try:
        return probe_intent(query)
    except Exception as e:
        logger.error(f"Intent probe failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
