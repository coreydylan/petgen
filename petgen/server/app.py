"""FastAPI application factory for the PetGen server."""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from petgen.server.deps import get_config, get_pipeline, get_queue, init_deps, reset_deps
from petgen.server.routes import characters, generate, scenes, voice

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize dependencies and start background worker."""
    init_deps()

    config = get_config()
    pipeline = get_pipeline()
    queue = get_queue()

    # Mount static files for generated output
    config.output_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/files", StaticFiles(directory=str(config.output_dir)), name="files")

    # Start background job worker
    worker_task = asyncio.create_task(queue.process_jobs(pipeline))
    logger.info("PetGen server started -- output at %s", config.output_dir)

    yield

    # Shutdown: cancel worker
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass
    reset_deps()
    logger.info("PetGen server shut down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="PetGen API",
        description="AI-powered talking pet video generator",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS -- allow all origins in dev
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    app.include_router(characters.router)
    app.include_router(generate.router)
    app.include_router(voice.router)
    app.include_router(scenes.router)

    @app.get("/api/health")
    async def health():
        return {"status": "ok"}

    return app
