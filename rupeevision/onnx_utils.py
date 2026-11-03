"""Pre/post-processing helpers for YOLOv8 ONNX detection models.

Verified against models/model.onnx (Ultralytics 8.4.175 export, opset 12):
    input  "images"  : float32 [1, 3, 512, 512], RGB, 0..1
    output "output0" : float32 [1, 5, 5376]  -> rows = cx, cy, w, h, score(rupee_50)

YOLOv8 heads have no objectness column, so the class score is the confidence.
The export has no built-in NMS (end2end=False), so NMS is applied here.
The decoder also accepts the transposed layout [1, N, 4+nc] produced by some
exporters, and rejects anything else with a clear error.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

DEFAULT_PAD_VALUE = 114  # Ultralytics letterbox gray (0..255)
DEFAULT_MAX_DETECTIONS = 300


@dataclass
class ModelInfo:
    input_name: str | None
    input_shape: tuple[int, ...] | None  # (N, C, H, W)
    output_shape: tuple[int, ...] | None
    names: dict[int, str]
    metadata: dict[str, str]


@dataclass
class LetterboxInfo:
    scale: float
    pad_x: float
    pad_y: float


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int


def read_model_info(model_path: Path) -> ModelInfo:
    """Read input/output shapes and Ultralytics metadata from the ONNX file.

    Uses the optional `onnx` package; if it is missing, returns empty info and
    the detector falls back to shapes reported by its inference backend.
    """
    try:
        import onnx
    except ImportError:
        return ModelInfo(None, None, None, {}, {})

    model = onnx.load(str(model_path), load_external_data=False)
    graph = model.graph
    initializers = {i.name for i in graph.initializer}
    inputs = [i for i in graph.input if i.name not in initializers]

    def dims(value_info) -> tuple[int, ...]:
        return tuple(d.dim_value if d.dim_value > 0 else -1
                     for d in value_info.type.tensor_type.shape.dim)

    metadata = {p.key: p.value for p in model.metadata_props}
    names: dict[int, str] = {}
    if "names" in metadata:
        try:
            names = {int(k): str(v) for k, v in ast.literal_eval(metadata["names"]).items()}
        except (ValueError, SyntaxError):
            pass

    return ModelInfo(
        input_name=inputs[0].name if inputs else None,
        input_shape=dims(inputs[0]) if inputs else None,
        output_shape=dims(graph.output[0]) if graph.output else None,
        names=names,
        metadata=metadata,
    )


def _letterbox_params(image_hw: tuple[int, int], new_shape: tuple[int, int],
                      scaleup: bool = True) -> tuple[float, int, int, int, int]:
    """Return (scale, resized_w, resized_h, pad_left, pad_top) for a letterbox."""
    h, w = image_hw
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image size {w}x{h}")
    new_h, new_w = new_shape
    if new_h <= 0 or new_w <= 0:
        raise ValueError(f"Invalid target size {new_w}x{new_h}")

    scale = min(new_h / h, new_w / w)
    if not scaleup:
        scale = min(scale, 1.0)
    resized_w = max(1, int(round(w * scale)))
    resized_h = max(1, int(round(h * scale)))

    pad_x = (new_w - resized_w) / 2
    pad_y = (new_h - resized_h) / 2
    left = int(round(pad_x - 0.1))
    top = int(round(pad_y - 0.1))
    return scale, min(resized_w, new_w), min(resized_h, new_h), max(left, 0), max(top, 0)


def letterbox(image: np.ndarray, new_shape: tuple[int, int],
              color: tuple[int, int, int] = (DEFAULT_PAD_VALUE,) * 3,
              scaleup: bool = True) -> tuple[np.ndarray, LetterboxInfo]:
    """Resize keeping aspect ratio and pad to new_shape (h, w), as Ultralytics does."""
    h, w = image.shape[:2]
    new_h, new_w = new_shape
    scale, resized_w, resized_h, left, top = _letterbox_params((h, w), new_shape, scaleup)

    if (resized_w, resized_h) != (w, h):
        image = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    right = max(0, new_w - resized_w - left)
    bottom = max(0, new_h - resized_h - top)
    padded = cv2.copyMakeBorder(image, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=color)
    return padded, LetterboxInfo(scale=scale, pad_x=left, pad_y=top)


def make_blob(image_bgr: np.ndarray, input_hw: tuple[int, int],
              pad_value: int = DEFAULT_PAD_VALUE,
              scaleup: bool = True) -> tuple[np.ndarray, LetterboxInfo]:
    """BGR frame -> letterboxed NCHW float32 RGB blob scaled to 0..1.

    The image is resized only once (by ``blobFromImage``) and the padding is
    added directly in the normalized float blob. This avoids building an
    intermediate uint8 canvas and a second full-size resize.
    """
    h, w = image_bgr.shape[:2]
    new_h, new_w = input_hw
    scale, resized_w, resized_h, left, top = _letterbox_params((h, w), input_hw, scaleup)

    resized_blob = cv2.dnn.blobFromImage(
        image_bgr, scalefactor=1 / 255.0, size=(resized_w, resized_h),
        swapRB=True, crop=False, ddepth=cv2.CV_32F)
    info = LetterboxInfo(scale=scale, pad_x=left, pad_y=top)

    if (resized_w, resized_h) == (new_w, new_h):
        return resized_blob, info

    blob = np.full((1, 3, new_h, new_w), pad_value / 255.0, np.float32)
    blob[0, :, top:top + resized_h, left:left + resized_w] = resized_blob[0]
    return blob, info


def _nms_indices(boxes_xywh: np.ndarray, scores: np.ndarray, class_ids: np.ndarray,
                 conf_threshold: float, iou_threshold: float,
                 max_detections: int) -> np.ndarray:
    """Class-aware NMS, using OpenCV's batched variant when available."""
    if not len(scores):
        return np.empty(0, dtype=np.int64)

    try:
        indices = cv2.dnn.NMSBoxesBatched(
            boxes_xywh.tolist(), scores.tolist(), class_ids.tolist(),
            conf_threshold, iou_threshold)
    except AttributeError:
        # Older OpenCV: offset boxes per class so classes never suppress each other.
        span = float(max(boxes_xywh[:, 0].max(), boxes_xywh[:, 1].max())) + 1.0
        offset = class_ids.astype(np.float32) * span
        shifted = np.stack([boxes_xywh[:, 0] + offset, boxes_xywh[:, 1],
                            boxes_xywh[:, 2], boxes_xywh[:, 3]], axis=1)
        indices = cv2.dnn.NMSBoxes(
            shifted.tolist(), scores.tolist(), conf_threshold, iou_threshold)

    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if max_detections > 0 and len(indices) > max_detections:
        indices = indices[:max_detections]
    return indices


def decode_yolov8(output: np.ndarray, num_classes: int, conf_threshold: float,
                  iou_threshold: float, letterbox_info: LetterboxInfo,
                  frame_shape: tuple[int, ...],
                  max_detections: int = DEFAULT_MAX_DETECTIONS) -> list[Detection]:
    """Decode raw YOLOv8 output into detections in original-frame pixel coordinates."""
    preds = np.asarray(output)
    if preds.ndim == 3:
        if preds.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, got output shape {preds.shape}")
        preds = preds[0]
    if preds.ndim != 2:
        raise ValueError(f"Unsupported output shape {output.shape}; expected [1, 4+nc, N] or [1, N, 4+nc]")

    channels = 4 + num_classes
    if preds.shape[0] == channels:
        preds = preds.T  # [4+nc, N] -> [N, 4+nc]
    elif preds.shape[1] != channels:
        raise ValueError(
            f"Output shape {output.shape} does not match a YOLOv8 head with {num_classes} class(es) "
            f"(expected one axis of size {channels}). Was the model exported with NMS or a "
            f"different class count?")

    boxes_cxcywh = preds[:, :4]
    class_scores = preds[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(len(class_scores)), class_ids]

    keep = scores >= conf_threshold
    if not np.any(keep):
        return []
    boxes_cxcywh, scores, class_ids = boxes_cxcywh[keep], scores[keep], class_ids[keep]

    # Undo letterbox: model space -> original frame space.
    if letterbox_info.scale <= 0:
        raise ValueError("Invalid letterbox scale")
    cx, cy, bw, bh = boxes_cxcywh.T
    x1 = (cx - bw / 2 - letterbox_info.pad_x) / letterbox_info.scale
    y1 = (cy - bh / 2 - letterbox_info.pad_y) / letterbox_info.scale
    x2 = (cx + bw / 2 - letterbox_info.pad_x) / letterbox_info.scale
    y2 = (cy + bh / 2 - letterbox_info.pad_y) / letterbox_info.scale

    frame_h, frame_w = frame_shape[:2]
    x1 = np.clip(x1, 0, frame_w - 1).astype(np.float32)
    x2 = np.clip(x2, 0, frame_w - 1).astype(np.float32)
    y1 = np.clip(y1, 0, frame_h - 1).astype(np.float32)
    y2 = np.clip(y2, 0, frame_h - 1).astype(np.float32)

    boxes_xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)
    scores = scores.astype(np.float32, copy=False)
    class_ids = class_ids.astype(np.int32, copy=False)

    indices = _nms_indices(boxes_xywh, scores, class_ids,
                           conf_threshold, iou_threshold, max_detections)

    detections = [
        Detection(int(round(x1[i])), int(round(y1[i])), int(round(x2[i])), int(round(y2[i])),
                  float(scores[i]), int(class_ids[i]))
        for i in indices
        if x2[i] > x1[i] and y2[i] > y1[i]
    ]
    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections[:max_detections] if max_detections > 0 else detections
