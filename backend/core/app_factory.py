"""
Application factory.

Keeping app construction in a `create_app()` function (rather than module
level in `main.py`) makes the app testable: `tests/test_backend` can call
`create_app()` to get a fresh instance with overridden dependencies, without
needing to spin up the real server.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routers.health import router as health_router
from backend.api.routers.explain import router as explain_router
from backend.api.routers.predict import router as predict_router
from backend.api.routers.segment import router as segment_router
from backend.core.exceptions import register_exception_handlers
from backend.database.session import init_db
from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup/shutdown hook.

    Runs once when the app starts (before it accepts traffic) and once when
    it shuts down. Using the `lifespan` context manager instead of the
    deprecated `@app.on_event("startup")` decorator per current FastAPI
    guidance.
    """
    logger.info("Starting %s (env=%s, debug=%s)", settings.APP_NAME, settings.APP_ENV, settings.DEBUG)
    init_db()
    logger.info("Database initialized at %s", settings.DATABASE_URL)
    yield
    logger.info("Shutting down %s", settings.APP_NAME)


def create_app() -> FastAPI:
    """Build and configure the FastAPI application instance."""

    app = FastAPI(
        title=settings.APP_NAME,
        description=(
            "AI-powered brain tumor diagnosis, segmentation, explainability, "
            "medical report generation, and RAG medical assistant."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=_lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)

    # Routers are included with the shared API version prefix so every
    # endpoint is reachable at e.g. /api/v1/health, /api/v1/predict, etc.
    app.include_router(health_router, prefix=settings.API_V1_PREFIX, tags=["Health"])
    app.include_router(predict_router, prefix=settings.API_V1_PREFIX, tags=["Classification"])
    app.include_router(segment_router, prefix=settings.API_V1_PREFIX, tags=["Segmentation"])
    app.include_router(explain_router, prefix=settings.API_V1_PREFIX, tags=["Explainability"])

    return app
