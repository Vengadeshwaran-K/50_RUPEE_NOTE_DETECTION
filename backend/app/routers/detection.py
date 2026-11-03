"""Detection endpoints. Uploads are decoded in memory and never written to disk."""

from __future__ import annotations

import base64

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile, status
from starlette.concurrency import run_in_threadpool

from ..config import TARGET_LABEL, Settings
from ..detector import DetectionResult, DetectorService, InferenceError, ModelNotLoadedError
from ..schemas import BoundingBox, DetectionItem, DetectionResponse

router = APIRouter(prefix="/api/detect", tags=["detection"])

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/pjpeg", "image/png",
                         "application/octet-stream", ""}
MAGIC = {b"\xff\xd8\xff": "JPEG", b"\x89PNG\r\n\x1a\n": "PNG"}

CONF_QUERY = Query(None, gt=0, lt=1, description="Confidence threshold (0-1, exclusive). "
                                                 "Defaults to the server setting.")


async def read_image(upload: UploadFile, settings: Settings) -> np.ndarray:
    content_type = (upload.content_type or "").split(";")[0].strip().lower()
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                            f"Unsupported file type '{content_type}'. Upload a JPG or PNG image.")

    data = await upload.read(settings.max_upload_bytes + 1)
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "The uploaded file is empty.")
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(413,
                            f"File is larger than {settings.max_upload_bytes // (1024 * 1024)} MB.")
    if not any(data.startswith(sig) for sig in MAGIC):
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                            "File content is not a JPG or PNG image.")

    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "The image could not be decoded; it may be corrupt.")

    h, w = image.shape[:2]
    if max(h, w) > settings.max_image_side:
        raise HTTPException(413,
                            f"Image is {w}x{h}; the maximum side length is {settings.max_image_side} px.")
    if min(h, w) < settings.min_image_side:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            f"Image is {w}x{h}; the minimum side length is {settings.min_image_side} px.")
    return image


def to_response(result: DetectionResult, service: DetectorService) -> DetectionResponse:
    names = service.detector.class_names if service.detector else {}
    items = [
        DetectionItem(
            label=TARGET_LABEL,
            class_id=d.class_id,
            class_name=names.get(d.class_id, str(d.class_id)),
            confidence=round(d.confidence, 4),
            bbox=BoundingBox(x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2,
                             width=d.x2 - d.x1, height=d.y2 - d.y1),
        )
        for d in result.detections
    ]
    return DetectionResponse(
        count=len(items),
        label=TARGET_LABEL,
        detections=items,
        inference_ms=round(result.inference_ms, 2),
        image_width=result.image_width,
        image_height=result.image_height,
        conf_threshold=result.conf_threshold,
        annotated_image=base64.b64encode(result.annotated_jpeg).decode("ascii"),
    )


async def run_detection(request: Request, upload: UploadFile, conf: float | None) -> DetectionResponse:
    service: DetectorService = request.app.state.detector
    if not service.loaded:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            f"Model is not loaded: {service.error or 'unknown error'}")
    try:
        image = await read_image(upload, service.settings)
    finally:
        await upload.close()
    try:
        result = await run_in_threadpool(service.detect, image, conf)
    except ModelNotLoadedError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except InferenceError as exc:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(exc)) from exc
    return to_response(result, service)


@router.post("/image", response_model=DetectionResponse)
async def detect_image(request: Request,
                       file: UploadFile = File(..., description="JPG or PNG image"),
                       conf: float | None = CONF_QUERY) -> DetectionResponse:
    """Detect ₹50 notes in an uploaded image."""
    return await run_detection(request, file, conf)


@router.post("/frame", response_model=DetectionResponse)
async def detect_frame(request: Request,
                       file: UploadFile = File(..., description="Single webcam frame (JPG or PNG)"),
                       conf: float | None = CONF_QUERY) -> DetectionResponse:
    """Detect ₹50 notes in a single webcam frame captured by the browser."""
    return await run_detection(request, file, conf)
