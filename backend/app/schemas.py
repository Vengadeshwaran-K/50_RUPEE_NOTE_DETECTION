"""Response models (also drive the Swagger docs at /docs)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])
    model_loaded: bool
    model_error: str | None = None


class ModelInfoResponse(BaseModel):
    target_class: str = Field(examples=["₹50"])
    model_loaded: bool
    model_error: str | None = None
    model_path: str
    class_names: dict[str, str]
    input_width: int | None
    input_height: int | None
    output_shape: list[int] | None
    inference_backend: str
    opencv_version: str
    conf_threshold: float
    iou_threshold: float
    max_detections: int
    max_upload_bytes: int


class BoundingBox(BaseModel):
    """Pixel coordinates in the submitted image (top-left origin)."""
    x1: int
    y1: int
    x2: int
    y2: int
    width: int
    height: int


class DetectionItem(BaseModel):
    label: str = Field(examples=["₹50"])
    class_id: int
    class_name: str = Field(examples=["rupee_50"])
    confidence: float = Field(ge=0, le=1)
    bbox: BoundingBox


class DetectionResponse(BaseModel):
    count: int
    label: str = Field(examples=["₹50"])
    detections: list[DetectionItem]
    inference_ms: float = Field(description="Preprocessing + forward pass + decoding/NMS")
    image_width: int
    image_height: int
    conf_threshold: float
    annotated_image: str = Field(description="Base64-encoded JPEG with boxes drawn")
    annotated_image_mime: str = "image/jpeg"
