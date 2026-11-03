"""RupeeVision AI — FastAPI app serving the detection API and the static frontend.

Run from the `backend/` folder:
    uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings
from .detector import DetectorService
from .routers import detection, health

logger = logging.getLogger("rupeevision")

# Multipart framing overhead allowed on top of the file size limit.
_MULTIPART_OVERHEAD = 64 * 1024


def create_app(settings: Settings | None = None, service: DetectorService | None = None) -> FastAPI:
    """Build the app. Pass `service` to inject a pre-built (e.g. mocked) detector."""
    settings = settings or (service.settings if service else Settings())
    service = service or DetectorService(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not service.loaded and service.error is None:
            service.load()
            if service.loaded:
                logger.info("Model loaded: %s", settings.model_path)
            else:
                logger.error("Model failed to load: %s", service.error)
        yield

    app = FastAPI(
        title="RupeeVision AI",
        description="Indian ₹50 banknote detection with a YOLOv8 ONNX model on OpenCV DNN.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.detector = service
    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def reject_oversized_uploads(request: Request, call_next):
        # Fail fast before the multipart body is parsed.
        if request.url.path.startswith("/api/detect/"):
            length = request.headers.get("content-length")
            if length and length.isdigit() and int(length) > settings.max_upload_bytes + _MULTIPART_OVERHEAD:
                return JSONResponse(
                    status_code=413,
                    content={"detail": f"Request is larger than "
                                       f"{settings.max_upload_bytes // (1024 * 1024)} MB."})
        return await call_next(request)

    app.include_router(health.router)
    app.include_router(detection.router)

    # Mounted last so /api/* and /docs always take precedence over static files.
    if settings.frontend_dir.is_dir():
        app.mount("/", StaticFiles(directory=settings.frontend_dir, html=True), name="frontend")
    else:
        logger.warning("Frontend folder not found: %s", settings.frontend_dir)

    return app


app = create_app()
