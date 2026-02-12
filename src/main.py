"""
Legacy entrypoint module.

The real FastAPI application now lives in `src.app.main`.
This shim keeps `uvicorn src.main:app` and existing tests working.
"""

from src.app.main import app, recommender, rec_service  # noqa: F401

__all__ = ["app", "recommender", "rec_service"]
