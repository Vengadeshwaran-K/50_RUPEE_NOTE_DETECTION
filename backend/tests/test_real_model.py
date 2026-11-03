"""REAL inference tests: load models/model.onnx with OpenCV DNN and run it (no mocks).

Fixtures:
  rupee50_train_sample.jpg  - a ₹50 image from the *training* split, so this
                              checks the pipeline end-to-end, not generalisation
                              (see model_improvement/IMPROVEMENT_REPORT.md).
  rupee500_test_sample.jpg  - a ₹500 note from the test split (negative sample).
"""

from __future__ import annotations

import base64

import cv2
import numpy as np
import pytest

from app.config import Settings
from app.detector import DetectorService

from .conftest import FIXTURES

pytestmark = pytest.mark.real_model

POSITIVE = FIXTURES / "rupee50_train_sample.jpg"
NEGATIVE = FIXTURES / "rupee500_test_sample.jpg"


def post_file(client, path, file_path, **params):
    with open(file_path, "rb") as f:
        return client.post(path, files={"file": (file_path.name, f, "image/jpeg")}, params=params)


def test_model_loads_with_expected_shapes(real_client):
    info = real_client.get("/api/model-info").json()
    assert info["model_loaded"] is True
    assert (info["input_width"], info["input_height"]) == (512, 512)
    assert info["output_shape"] == [1, 5, 5376]
    assert info["conf_threshold"] == 0.6
    assert info["class_names"] == {"0": "rupee_50"}


def test_service_detect_direct():
    service = DetectorService(Settings())
    service.load()
    assert service.loaded, service.error
    image = cv2.imread(str(POSITIVE))
    result = service.detect(image)
    assert len(result.detections) >= 1
    assert result.inference_ms > 0


def test_real_positive_image(real_client):
    r = post_file(real_client, "/api/detect/image", POSITIVE)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] >= 1
    h, w = cv2.imread(str(POSITIVE)).shape[:2]
    for det in body["detections"]:
        assert det["label"] == "₹50"
        assert 0.6 <= det["confidence"] <= 1
        b = det["bbox"]
        assert 0 <= b["x1"] < b["x2"] < w and 0 <= b["y1"] < b["y2"] < h
    annotated = cv2.imdecode(np.frombuffer(base64.b64decode(body["annotated_image"]), np.uint8),
                             cv2.IMREAD_COLOR)
    assert annotated.shape[:2] == (h, w)


def test_real_frame_endpoint_matches_image_endpoint(real_client):
    a = post_file(real_client, "/api/detect/image", POSITIVE).json()
    b = post_file(real_client, "/api/detect/frame", POSITIVE).json()
    assert a["detections"] == b["detections"]


def test_real_negative_image_has_no_detections(real_client):
    body = post_file(real_client, "/api/detect/image", NEGATIVE).json()
    assert body["count"] == 0


def test_real_blank_frame_has_no_detections(real_client):
    data = cv2.imencode(".jpg", np.full((480, 640, 3), 127, np.uint8))[1].tobytes()
    body = real_client.post("/api/detect/frame", files={"file": ("f.jpg", data, "image/jpeg")}).json()
    assert body["count"] == 0
    assert body["detections"] == []


def test_real_higher_threshold_filters_detections(real_client):
    body = post_file(real_client, "/api/detect/image", POSITIVE, conf="0.99").json()
    assert body["count"] == 0
    assert body["conf_threshold"] == 0.99
