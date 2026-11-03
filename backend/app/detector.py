"""Model service: loads the YOLOv8 ONNX model once and runs thread-safe inference.

The pre/post-processing lives in the shared `rupeevision` package (letterbox,
YOLOv8 decoding, NMS), which is also used by the webcam app and `test.py`.
This module only adds lifecycle, locking and per-request confidence
thresholds on top of it.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .config import PROJECT_ROOT, TARGET_LABEL, Settings

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rupeevision.detector import DetectorError, RupeeDetector  # noqa: E402
from rupeevision.onnx_utils import Detection, decode_yolov8, make_blob  # noqa: E402

__all__ = ["DetectorService", "DetectionResult", "ModelNotLoadedError", "InferenceError"]


class ModelNotLoadedError(RuntimeError):
    pass


class InferenceError(RuntimeError):
    pass


@dataclass
class DetectionResult:
    detections: list[Detection]
    inference_ms: float
    image_width: int
    image_height: int
    conf_threshold: float
    annotated_jpeg: bytes


class DetectorService:
    """Owns a single RupeeDetector. `load()` never raises; check `loaded` / `error`."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.detector: RupeeDetector | None = None
        self.error: str | None = None
        # cv2.dnn.Net is not safe to call from several threads at once.
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self.detector is not None

    def load(self) -> None:
        try:
            self.detector = RupeeDetector(
                self.settings.model_path,
                conf_threshold=self.settings.conf_threshold,
                iou_threshold=self.settings.iou_threshold,
                use_cuda=self.settings.use_cuda,
                backend=self.settings.backend,
                num_threads=self.settings.num_threads,
                max_detections=self.settings.max_detections,
            )
            if self.settings.warmup:
                self.detector.warmup()
            self.error = None
        except DetectorError as exc:
            self.detector = None
            self.error = str(exc)

    def info(self) -> dict:
        d = self.detector
        try:
            model_path = str(self.settings.model_path.resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            model_path = str(self.settings.model_path)
        return {
            "target_class": TARGET_LABEL,
            "model_loaded": self.loaded,
            "model_error": self.error,
            "model_path": model_path.replace("\\", "/"),
            "class_names": {str(k): v for k, v in d.class_names.items()} if d else {},
            "input_width": d.input_hw[1] if d else None,
            "input_height": d.input_hw[0] if d else None,
            "output_shape": list(d.output_shape) if d else None,
            "inference_backend": d.backend_name if d else "not loaded",
            "opencv_version": cv2.__version__,
            "conf_threshold": self.settings.conf_threshold,
            "iou_threshold": self.settings.iou_threshold,
            "max_detections": self.settings.max_detections,
            "max_upload_bytes": self.settings.max_upload_bytes,
        }

    def detect(self, frame_bgr: np.ndarray, conf_threshold: float | None = None) -> DetectionResult:
        d = self.detector
        if d is None:
            raise ModelNotLoadedError(self.error or "Model is not loaded")
        conf = self.settings.conf_threshold if conf_threshold is None else conf_threshold

        start = time.perf_counter()
        try:
            blob, lb = make_blob(frame_bgr, d.input_hw)
            if d.backend == "onnxruntime":
                # ONNX Runtime sessions support concurrent run() calls.
                output = d.infer(blob)
            else:
                with self._lock:
                    output = d.infer(blob)
            detections = decode_yolov8(output, d.num_classes, conf, d.iou_threshold, lb,
                                       frame_bgr.shape, max_detections=d.max_detections)
        except (DetectorError, ValueError) as exc:
            raise InferenceError(f"Inference failed: {exc}") from exc
        inference_ms = (time.perf_counter() - start) * 1000

        annotated = d.draw(frame_bgr, detections)
        ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, self.settings.jpeg_quality])
        if not ok:
            raise InferenceError("Could not encode the annotated image as JPEG")

        h, w = frame_bgr.shape[:2]
        return DetectionResult(detections, inference_ms, w, h, conf, encoded.tobytes())
