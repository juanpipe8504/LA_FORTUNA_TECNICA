"""FastAPI application factory.

``create_app(container)`` lets tests inject a fake object graph. When no
container is given, the real one is built during the lifespan startup (not at
import time), so ``app = create_app()`` is safe to import anywhere.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.errors import register_exception_handlers
from app.api.middleware import REQUEST_ID_HEADER, register_middleware
from app.api.routes import chat, documents, health, jobs
from app.bootstrap.container import Container, get_container
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging, get_logger

logger = get_logger(__name__)

API_VERSION = "0.1.0"
API_DESCRIPTION = (
    "Multimodal RAG over technical PDFs: upload documents (text, tables, diagrams), "
    "then ask questions and get grounded answers with cited sources and related images."
)
OPENAPI_TAGS = [
    {"name": "documents", "description": "Upload PDFs, list documents, fetch extracted images."},
    {"name": "jobs", "description": "Asynchronous ingestion job status."},
    {"name": "chat", "description": "Grounded question answering."},
    {"name": "health", "description": "Liveness / readiness probes."},
]


def _build_lifespan(owns_container: bool):
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        settings: Settings = app.state.settings
        configure_logging(settings.log_level, settings.log_json)
        if app.state.container is None:
            app.state.container = get_container()
        logger.info("api started", extra={"app_env": settings.app_env.value})
        try:
            yield
        finally:
            if owns_container and app.state.container is not None:
                app.state.container.close()
            logger.info("api stopped")

    return lifespan


def _api_router() -> APIRouter:
    router = APIRouter()
    for module in (documents, jobs, chat, health):
        router.include_router(module.router)
    return router


def create_app(container: Container | None = None) -> FastAPI:
    settings = container.settings if container is not None else get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=API_VERSION,
        description=API_DESCRIPTION,
        openapi_tags=OPENAPI_TAGS,
        lifespan=_build_lifespan(owns_container=container is None),
    )
    app.state.settings = settings
    app.state.container = container

    register_exception_handlers(app)
    register_middleware(app)
    # Added last => outermost, so CORS headers are present on error responses too.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[REQUEST_ID_HEADER, "Location"],
    )

    app.include_router(_api_router(), prefix=settings.api_prefix)
    app.include_router(health.router)  # probes also at root: /health, /health/ready
    return app


app = create_app()
