"""FastAPI application entry point for bioscreen."""

import os

# Prevent OpenMP conflict between torch and FAISS on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app import __version__
from app.api.routes import router
from app.config import get_settings
from app.database.toxin_db import ToxinDatabase
from app.pipeline.embedding import EmbeddingModel


# ── Application state ─────────────────────────────────────────────────────────

class AppState:
    """Holds shared, lazily-initialised resources."""

    toxin_db: ToxinDatabase | None = None
    embedding_model: EmbeddingModel | None = None


state = AppState()


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models and databases on startup; release on shutdown."""
    settings = get_settings()
    logger.info("bioscreen v{} starting up (env={})", __version__, settings.app_env)

    # Load ESM-2 embedding model
    logger.info("Loading ESM-2 model: {}", settings.esm2_model_name)
    state.embedding_model = EmbeddingModel(
        model_name=settings.esm2_model_name,
        device=settings.device,
    )
    state.embedding_model.load()
    logger.info("ESM-2 loaded on device={}", settings.device)

    # Load FAISS toxin database (if it exists)
    state.toxin_db = ToxinDatabase(
        index_path=settings.toxin_db_path,
        meta_path=settings.toxin_meta_path,
    )
    if settings.toxin_db_path.exists():
        state.toxin_db.load()
        logger.info(
            "Toxin DB loaded: {} entries", state.toxin_db.size
        )
    else:
        logger.warning(
            "Toxin DB not found at {}. Run scripts/build_db.py first.",
            settings.toxin_db_path,
        )

    # Attach to app for route access
    app.state.toxin_db = state.toxin_db
    app.state.embedding_model = state.embedding_model

    yield

    logger.info("bioscreen shutting down")


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="bioscreen",
        description=(
            "Structure-based biosecurity screening for AI-designed proteins. "
            "Evaluates function, not just sequence similarity."
        ),
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if not settings.is_production else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api")

    return app


app = create_app()


# ── Dev entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=not settings.is_production,
        log_level=settings.log_level.lower(),
    )
