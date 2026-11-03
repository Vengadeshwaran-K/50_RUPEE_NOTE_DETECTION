"""₹50 banknote detector for YOLOv8 ONNX models.

The inference backend is selectable:

* ``onnxruntime`` - usually the fastest CPU option (optional dependency).
* ``opencv``      - only needs ``opencv-python``; supports CUDA builds.
* ``auto``        - ONNX Runtime when installed, otherwise OpenCV DNN.

Pre/post-processing is shared by the webcam app, ``test.py`` and the FastAPI
service, so every entry point produces identical results.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .onnx_utils import (
    DEFAULT_MAX_DETECTIONS,
    Detection,
    decode_yolov8,
    make_blob,
    read_model_info,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "model.onnx"
DISPLAY_LABEL = "₹50"
FALLBACK_LABEL = "Rs.50"  # used if no font with the ₹ glyph is available

BOX_COLOR_BGR = (0, 200, 0)
TEXT_COLOR_BGR = (0, 0, 0)

BACKENDS = ("auto", "onnxruntime", "opencv")

# Windows fonts that include U+20B9 (₹), plus common Linux/macOS fallbacks.
_FONT_CANDIDATES = [
    "C:/Windows/Fonts/Nirmala.ttc",
    "C:/Windows/Fonts/NirmalaUI.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
]


class DetectorError(RuntimeError):
    pass


class RupeeDetector:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH,
                 conf_threshold: float = 0.6, iou_threshold: float = 0.45,
                 use_cuda: bool = False, backend: str = "auto",
                 num_threads: int | None = None,
                 max_detections: int = DEFAULT_MAX_DETECTIONS):
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise DetectorError(
                f"ONNX model not found: {self.model_path}\n"
                f"Copy your exported best.onnx to {DEFAULT_MODEL_PATH}")
        if not 0.0 < conf_threshold < 1.0:
            raise DetectorError(f"conf_threshold must be in (0, 1), got {conf_threshold}")
        if not 0.0 < iou_threshold < 1.0:
            raise DetectorError(f"iou_threshold must be in (0, 1), got {iou_threshold}")
        if max_detections <= 0:
            raise DetectorError(f"max_detections must be positive, got {max_detections}")

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.num_threads = num_threads
        self.use_cuda = use_cuda

        self.info = read_model_info(self.model_path)
        self.class_names = self.info.names or {0: "rupee_50"}
        self.num_classes = len(self.class_names)

        self.net: cv2.dnn.Net | None = None
        self._ort_session = None
        self._ort_input_name = self.info.input_name
        self._ort_output_name: str | None = None
        self._ort_input_shape: tuple | None = None
        self._backend = "opencv"
        self._device = "CPU"

        self._load_backend(backend, use_cuda, num_threads)
        self.input_hw = self._resolve_input_size()
        self.output_shape = self._probe_output_shape()
        self._font = _load_font()

    # ------------------------------------------------------------ backends

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def backend_name(self) -> str:
        if self._backend == "onnxruntime":
            return f"ONNX Runtime ({self._device})"
        suffix = " (CUDA)" if self._device == "CUDA" else " (CPU)"
        return f"OpenCV DNN{suffix}"

    def _load_backend(self, backend: str, use_cuda: bool, num_threads: int | None) -> None:
        if backend not in BACKENDS:
            raise DetectorError(f"backend must be one of {BACKENDS}, got {backend!r}")

        if backend in ("auto", "onnxruntime"):
            try:
                import onnxruntime as ort
            except ImportError:
                if backend == "onnxruntime":
                    raise DetectorError(
                        "onnxruntime is not installed. Install `onnxruntime` "
                        "(or `onnxruntime-gpu` for CUDA), or use backend='opencv'."
                    ) from None
            else:
                self._init_onnxruntime(ort, use_cuda, num_threads)
                return

        self._init_opencv(use_cuda, num_threads)

    def _init_onnxruntime(self, ort, use_cuda: bool, num_threads: int | None) -> None:
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if num_threads:
            options.intra_op_num_threads = num_threads
            options.inter_op_num_threads = 1

        providers = ["CPUExecutionProvider"]
        if use_cuda and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")

        try:
            self._ort_session = ort.InferenceSession(
                str(self.model_path), sess_options=options, providers=providers)
        except Exception as exc:  # onnxruntime raises several exception types
            raise DetectorError(f"ONNX Runtime could not load {self.model_path}: {exc}") from exc

        input_meta = self._ort_session.get_inputs()[0]
        self._ort_input_name = input_meta.name
        self._ort_input_shape = tuple(input_meta.shape)
        outputs = self._ort_session.get_outputs()
        self._ort_output_name = outputs[0].name if outputs else "output0"
        self._backend = "onnxruntime"
        self._device = "CUDA" if providers[0] == "CUDAExecutionProvider" else "CPU"

    def _init_opencv(self, use_cuda: bool, num_threads: int | None) -> None:
        if num_threads:
            cv2.setNumThreads(num_threads)
        try:
            self.net = cv2.dnn.readNetFromONNX(str(self.model_path))
        except cv2.error as exc:
            raise DetectorError(f"OpenCV could not load {self.model_path}: {exc}") from exc

        if use_cuda:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            self._device = "CUDA"
        self._backend = "opencv"

    # ------------------------------------------------------------- shapes

    def _reported_input_shape(self) -> tuple | None:
        return self.info.input_shape or self._ort_input_shape

    def _resolve_input_size(self) -> tuple[int, int]:
        shape = self._reported_input_shape()
        if shape is None:
            return self._guess_input_size()  # onnx package unavailable
        if len(shape) != 4:
            raise DetectorError(f"Expected NCHW input with 3 channels, model has {shape}")
        if isinstance(shape[1], int) and shape[1] not in (3, -1):
            raise DetectorError(f"Expected NCHW input with 3 channels, model has {shape}")
        h, w = shape[2], shape[3]
        if not isinstance(h, int) or not isinstance(w, int) or h <= 0 or w <= 0:
            raise DetectorError(f"Model has dynamic input size {shape}; re-export with a fixed imgsz")
        if h % 32 or w % 32:
            raise DetectorError(f"Input size {h}x{w} is not a multiple of the YOLOv8 stride (32)")
        return h, w

    def _guess_input_size(self) -> tuple[int, int]:
        """Without model metadata, try common export sizes (512 first: current model)."""
        try:
            logging = cv2.utils.logging
            previous = logging.getLogLevel()
            logging.setLogLevel(logging.LOG_LEVEL_SILENT)
        except (AttributeError, cv2.error):
            logging = None
            previous = None

        try:
            for size in (512, 640, 416, 320, 768, 800, 1024):
                try:
                    self.infer(np.zeros((1, 3, size, size), np.float32))
                    return size, size
                except DetectorError:
                    continue
        finally:
            if logging is not None and previous is not None:
                logging.setLogLevel(previous)
        raise DetectorError("Could not determine the model input size; `pip install onnx` and retry")

    def _probe_output_shape(self) -> tuple[int, ...]:
        dummy = np.zeros((*self.input_hw, 3), dtype=np.uint8)
        blob, _ = make_blob(dummy, self.input_hw)
        out = self.infer(blob)

        expected = 4 + self.num_classes
        if out.ndim != 3 or expected not in out.shape[1:]:
            raise DetectorError(
                f"Unexpected output shape {out.shape}; expected [1, {expected}, N] or [1, N, {expected}] "
                f"for a YOLOv8 detect head with {self.num_classes} class(es)")
        return tuple(out.shape)

    # ---------------------------------------------------------- inference

    def infer(self, blob: np.ndarray) -> np.ndarray:
        """Run one forward pass on a preprocessed NCHW blob."""
        try:
            if self._ort_session is not None:
                return self._ort_session.run(
                    [self._ort_output_name], {self._ort_input_name: blob})[0]
            if self.net is None:
                raise DetectorError("No inference backend is loaded")
            self.net.setInput(blob)
            return self.net.forward()
        except DetectorError:
            raise
        except Exception as exc:  # cv2.error / onnxruntime exception types
            raise DetectorError(f"Inference failed: {exc}") from exc

    def warmup(self, runs: int = 1) -> None:
        """Run a few dummy frames so the first real request is not slowed down."""
        dummy = np.zeros((*self.input_hw, 3), dtype=np.uint8)
        for _ in range(max(1, runs)):
            self.detect(dummy)

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise DetectorError("detect() expects a BGR image of shape (H, W, 3)")

        blob, lb = make_blob(frame_bgr, self.input_hw)
        output = self.infer(blob)
        return decode_yolov8(output, self.num_classes, self.conf_threshold,
                             self.iou_threshold, lb, frame_bgr.shape,
                             max_detections=self.max_detections)

    # ------------------------------------------------------------- drawing

    def draw(self, frame_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """Draw boxes labelled '₹50 0.93' onto a copy of the frame."""
        out = frame_bgr.copy()
        thickness = max(2, round(min(out.shape[:2]) / 300))
        for d in detections:
            cv2.rectangle(out, (d.x1, d.y1), (d.x2, d.y2), BOX_COLOR_BGR, thickness)
        if not detections:
            return out

        label_base = DISPLAY_LABEL if self._font is not None else FALLBACK_LABEL
        labels = [(d, f"{label_base} {d.confidence:.2f}") for d in detections]
        if self._font is None:
            for d, text in labels:
                _draw_label_cv2(out, text, d.x1, d.y1)
            return out
        return _draw_labels_pil(out, labels, self._font)


def _load_font():
    try:
        from PIL import ImageFont
    except ImportError:
        return None
    for path in _FONT_CANDIDATES:
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, 22)
            except OSError:
                continue
    return None


def _draw_label_cv2(img: np.ndarray, text: str, x: int, y: int) -> None:
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    top = y - th - base - 4 if y - th - base - 4 >= 0 else y
    cv2.rectangle(img, (x, top), (x + tw + 6, top + th + base + 4), BOX_COLOR_BGR, -1)
    cv2.putText(img, text, (x + 3, top + th + 2), font, scale, TEXT_COLOR_BGR, thick, cv2.LINE_AA)


def _draw_labels_pil(img_bgr: np.ndarray, labels, font) -> np.ndarray:
    from PIL import Image, ImageDraw

    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    fill = tuple(reversed(BOX_COLOR_BGR))
    text_fill = tuple(reversed(TEXT_COLOR_BGR))
    for d, text in labels:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        tw, th = right - left, bottom - top
        box_top = d.y1 - th - 8 if d.y1 - th - 8 >= 0 else d.y1
        draw.rectangle([d.x1, box_top, d.x1 + tw + 8, box_top + th + 8], fill=fill)
        draw.text((d.x1 + 4 - left, box_top + 4 - top), text, font=font, fill=text_fill)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
