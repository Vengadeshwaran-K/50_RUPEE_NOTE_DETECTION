"""Runtime settings for the RupeeVision AI backend.

Every value can be overridden with an environment variable, e.g.

    $env:RUPEEVISION_CONF = "0.35"; uvicorn app.main:app --port 8000
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "model.onnx"
DEFAULT_FRONTEND_DIR = PROJECT_ROOT / "frontend"

TARGET_LABEL = "₹50"


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _env_list(name: str, default: list[str]) -> list[str]:
    value = os.getenv(name)
    return [v.strip() for v in value.split(",") if v.strip()] if value else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    model_path: Path = field(default_factory=lambda: Path(
        os.getenv("RUPEEVISION_MODEL_PATH", str(DEFAULT_MODEL_PATH))))
    frontend_dir: Path = field(default_factory=lambda: Path(
        os.getenv("RUPEEVISION_FRONTEND_DIR", str(DEFAULT_FRONTEND_DIR))))
    conf_threshold: float = field(default_factory=lambda: _env_float("RUPEEVISION_CONF", 0.6))
    iou_threshold: float = field(default_factory=lambda: _env_float("RUPEEVISION_IOU", 0.45))
    backend: str = field(default_factory=lambda: os.getenv("RUPEEVISION_BACKEND", "auto").lower())
    num_threads: int | None = field(default_factory=lambda: (
        _env_int("RUPEEVISION_THREADS", 0) or None))
    max_detections: int = field(default_factory=lambda: _env_int("RUPEEVISION_MAX_DETECTIONS", 300))
    warmup: bool = field(default_factory=lambda: _env_bool("RUPEEVISION_WARMUP", True))
    use_cuda: bool = field(default_factory=lambda: _env_bool("RUPEEVISION_CUDA", False))
    max_upload_bytes: int = field(default_factory=lambda: _env_int(
        "RUPEEVISION_MAX_UPLOAD_BYTES", 10 * 1024 * 1024))
    max_image_side: int = field(default_factory=lambda: _env_int("RUPEEVISION_MAX_IMAGE_SIDE", 8000))
    min_image_side: int = 16
    jpeg_quality: int = field(default_factory=lambda: _env_int("RUPEEVISION_JPEG_QUALITY", 85))
    cors_origins: list[str] = field(default_factory=lambda: _env_list(
        "RUPEEVISION_CORS_ORIGINS", ["http://127.0.0.1:8000", "http://localhost:8000"]))
