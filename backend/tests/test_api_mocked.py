"""API tests with a MOCKED detector (no ONNX inference). See test_real_model.py for real inference."""

from __future__ import annotations

import base64

import cv2
import numpy as np
import pytest

from app.config import Settings
from app.detector import DetectorService

from .conftest import FakeDetectorService

ENDPOINTS = ["/api/detect/image", "/api/detect/frame"]


def post(client, path, data: bytes, name="note.jpg", content_type="image/jpeg", **params):
    return client.post(path, files={"file": (name, data, content_type)}, params=params)


def test_health(mock_client):
    r = mock_client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True, "model_error": None}


def test_model_info(mock_client):
    body = mock_client.get("/api/model-info").json()
    assert body["target_class"] == "₹50"
    assert body["model_loaded"] is True
    assert (body["input_width"], body["input_height"]) == (640, 640)
    assert body["inference_backend"].startswith("OpenCV DNN")
    assert body["class_names"] == {"0": "rupee_50"}


@pytest.mark.parametrize("path", ENDPOINTS)
def test_valid_jpeg_upload_response_format(mock_client, jpeg_bytes, path):
    r = post(mock_client, path, jpeg_bytes)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] == 1
    assert body["label"] == "₹50"
    assert body["image_width"] == 320 and body["image_height"] == 240
    assert isinstance(body["inference_ms"], float)
    det = body["detections"][0]
    assert det["label"] == "₹50"
    assert det["class_name"] == "rupee_50"
    assert det["confidence"] == pytest.approx(0.8765)
    assert det["bbox"] == {"x1": 10, "y1": 20, "x2": 110, "y2": 80, "width": 100, "height": 60}
    jpeg = base64.b64decode(body["annotated_image"])
    assert jpeg.startswith(b"\xff\xd8\xff")
    assert cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR).shape == (240, 320, 3)


def test_valid_png_upload(mock_client, png_bytes):
    r = post(mock_client, "/api/detect/image", png_bytes, "note.png", "image/png")
    assert r.status_code == 200, r.text


def test_empty_detections(make_client, jpeg_bytes):
    client = make_client(FakeDetectorService([]))
    body = post(client, "/api/detect/image", jpeg_bytes).json()
    assert body["count"] == 0
    assert body["detections"] == []
    assert body["annotated_image"]


@pytest.mark.parametrize("path", ENDPOINTS)
def test_rejects_non_image_content_type(mock_client, path):
    r = post(mock_client, path, b"hello", "notes.txt", "text/plain")
    assert r.status_code == 415
    assert "JPG or PNG" in r.json()["detail"]


def test_rejects_non_image_bytes_with_image_type(mock_client):
    r = post(mock_client, "/api/detect/image", b"GIF89a....", "fake.jpg", "image/jpeg")
    assert r.status_code == 415


def test_rejects_corrupt_jpeg(mock_client):
    r = post(mock_client, "/api/detect/image", b"\xff\xd8\xff\xe0" + b"\x00" * 100)
    assert r.status_code == 400
    assert "decoded" in r.json()["detail"]


def test_rejects_empty_file(mock_client):
    r = post(mock_client, "/api/detect/image", b"")
    assert r.status_code == 400
    assert "empty" in r.json()["detail"]


def test_rejects_missing_file_field(mock_client):
    r = mock_client.post("/api/detect/image")
    assert r.status_code == 422


def test_rejects_oversized_file(make_client):
    settings = Settings(max_upload_bytes=1000)
    client = make_client(FakeDetectorService(settings=settings))
    big = cv2.imencode(".png", np.random.randint(0, 255, (100, 100, 3), np.uint8))[1].tobytes()
    assert len(big) > 1000
    r = post(client, "/api/detect/image", big, "big.png", "image/png")
    assert r.status_code == 413


def test_rejects_oversized_dimensions(make_client):
    settings = Settings(max_image_side=100)
    client = make_client(FakeDetectorService(settings=settings))
    data = cv2.imencode(".jpg", np.zeros((50, 200, 3), np.uint8))[1].tobytes()
    r = post(client, "/api/detect/image", data)
    assert r.status_code == 413
    assert "200x50" in r.json()["detail"]


def test_rejects_tiny_image(mock_client):
    data = cv2.imencode(".png", np.zeros((8, 8, 3), np.uint8))[1].tobytes()
    assert post(mock_client, "/api/detect/image", data, "t.png", "image/png").status_code == 400


@pytest.mark.parametrize("conf", ["0", "1", "-0.2", "1.5", "abc"])
def test_confidence_threshold_validation(mock_client, jpeg_bytes, conf):
    r = post(mock_client, "/api/detect/image", jpeg_bytes, conf=conf)
    assert r.status_code == 422


def test_confidence_threshold_passed_to_detector(make_client, jpeg_bytes):
    service = FakeDetectorService([])
    client = make_client(service)
    body = post(client, "/api/detect/frame", jpeg_bytes, conf="0.3").json()
    assert service.calls[-1]["conf"] == pytest.approx(0.3)
    assert body["conf_threshold"] == pytest.approx(0.3)
    post(client, "/api/detect/frame", jpeg_bytes)
    assert service.calls[-1]["conf"] == pytest.approx(service.settings.conf_threshold)


def test_inference_error_returns_500(make_client, failing_service, jpeg_bytes):
    client = make_client(failing_service)
    r = post(client, "/api/detect/image", jpeg_bytes)
    assert r.status_code == 500
    assert r.json()["detail"] == "Inference failed: boom"


# --- Missing model: real DetectorService, real load attempt, file absent (not mocked) ---

@pytest.fixture
def missing_model_client(make_client, tmp_path):
    settings = Settings(model_path=tmp_path / "does_not_exist.onnx")
    return make_client(DetectorService(settings))


def test_missing_model_health(missing_model_client):
    body = missing_model_client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False
    assert "not found" in body["model_error"]


def test_missing_model_info(missing_model_client):
    body = missing_model_client.get("/api/model-info").json()
    assert body["model_loaded"] is False
    assert body["input_width"] is None


@pytest.mark.parametrize("path", ENDPOINTS)
def test_missing_model_detect_returns_503(missing_model_client, jpeg_bytes, path):
    r = post(missing_model_client, path, jpeg_bytes)
    assert r.status_code == 503
    assert "Model is not loaded" in r.json()["detail"]


def test_invalid_model_file(make_client, tmp_path):
    bad = tmp_path / "bad.onnx"
    bad.write_bytes(b"not an onnx model")
    client = make_client(DetectorService(Settings(model_path=bad)))
    body = client.get("/api/health").json()
    assert body["model_loaded"] is False
    assert body["model_error"]


# --- Static frontend served alongside the API ---

@pytest.mark.parametrize("path, content_type, marker", [
    ("/", "text/html", "RupeeVision AI"),
    ("/index.html", "text/html", "RupeeVision AI"),
    ("/style.css", "text/css", "--accent"),
    ("/script.js", "javascript", "getUserMedia"),
    ("/assets/logo.svg", "image/svg+xml", "<svg"),
])
def test_static_routes(mock_client, path, content_type, marker):
    r = mock_client.get(path)
    assert r.status_code == 200
    assert content_type in r.headers["content-type"]
    assert marker in r.text


def test_docs_and_openapi_not_shadowed_by_static(mock_client):
    assert mock_client.get("/docs").status_code == 200
    spec = mock_client.get("/openapi.json").json()
    assert {"/api/health", "/api/model-info", "/api/detect/image", "/api/detect/frame"} <= set(spec["paths"])


def test_unknown_routes_404(mock_client):
    assert mock_client.get("/api/does-not-exist").status_code == 404
    assert mock_client.get("/nope.js").status_code == 404


def test_get_on_detect_endpoint_is_not_served(mock_client):
    # POST-only routes fall through to the static mount for GET, which answers 404.
    r = mock_client.get("/api/detect/image")
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("application/json")


def test_cors_allows_local_origin(mock_client):
    r = mock_client.options("/api/health", headers={
        "Origin": "http://localhost:8000", "Access-Control-Request-Method": "GET"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:8000"
