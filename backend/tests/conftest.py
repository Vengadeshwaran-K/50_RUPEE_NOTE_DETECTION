from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import DEFAULT_MODEL_PATH, Settings
from app.detector import DetectionResult, DetectorService, InferenceError
from app.main import create_app
from rupeevision.onnx_utils import Detection

FIXTURES = Path(__file__).parent / "fixtures"


def encode(image: np.ndarray, ext: str = ".jpg") -> bytes:
    ok, buf = cv2.imencode(ext, image)
    assert ok
    return buf.tobytes()


@pytest.fixture
def jpeg_bytes() -> bytes:
    return encode(np.full((240, 320, 3), 127, np.uint8), ".jpg")


@pytest.fixture
def png_bytes() -> bytes:
    return encode(np.full((240, 320, 3), 200, np.uint8), ".png")


class FakeDetectorService(DetectorService):
    """MOCK: stands in for the ONNX model. Returns whatever detections it is given."""

    def __init__(self, detections=None, fail_with: Exception | None = None, settings: Settings | None = None):
        super().__init__(settings or Settings())
        self.detections = detections or []
        self.fail_with = fail_with
        self.calls: list[dict] = []
        self.detector = _FakeRupeeDetector()

    def detect(self, frame_bgr, conf_threshold=None):
        conf = self.settings.conf_threshold if conf_threshold is None else conf_threshold
        self.calls.append({"shape": frame_bgr.shape, "conf": conf})
        if self.fail_with:
            raise self.fail_with
        h, w = frame_bgr.shape[:2]
        return DetectionResult(list(self.detections), 1.5, w, h, conf, encode(frame_bgr))


class _FakeRupeeDetector:
    class_names = {0: "rupee_50"}
    input_hw = (640, 640)
    output_shape = (1, 5, 8400)
    backend = "opencv"
    backend_name = "OpenCV DNN (CPU)"
    max_detections = 300


@pytest.fixture
def make_client():
    clients = []

    def _make(service: DetectorService | None = None, settings: Settings | None = None) -> TestClient:
        app = create_app(settings=settings, service=service)
        client = TestClient(app)
        client.__enter__()
        clients.append(client)
        return client

    yield _make
    for c in clients:
        c.__exit__(None, None, None)


@pytest.fixture
def fake_detection() -> Detection:
    return Detection(x1=10, y1=20, x2=110, y2=80, confidence=0.87654, class_id=0)


@pytest.fixture
def mock_client(make_client, fake_detection):
    return make_client(FakeDetectorService([fake_detection]))


@pytest.fixture
def failing_service():
    return FakeDetectorService(fail_with=InferenceError("Inference failed: boom"))


@pytest.fixture(scope="session")
def real_client():
    if not DEFAULT_MODEL_PATH.is_file():
        pytest.skip(f"real model not found at {DEFAULT_MODEL_PATH}")
    app = create_app(Settings())
    with TestClient(app) as client:
        if not app.state.detector.loaded:
            pytest.fail(f"model failed to load: {app.state.detector.error}")
        yield client
