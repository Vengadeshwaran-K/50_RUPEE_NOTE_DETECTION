# RupeeVision AI — ₹50 Banknote Detection

Detect the new Indian **₹50** banknote in images and live webcam frames using a
YOLOv8 ONNX model. Shared Python core + FastAPI backend + a dependency-free
frontend, all in one repository.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?logo=opencv&logoColor=white)

## Repository layout

```text
.
├── rupeevision/          # shared core: ONNX model, preprocessing, decoding, drawing, webcam CLI
├── backend/              # FastAPI service (API + serves the static frontend)
│   ├── app/
│   └── tests/
├── frontend/             # vanilla HTML/CSS/JS UI (no build step)
├── model_improvement/    # training / evaluation experiments
├── models/               # model.onnx + best.pt
├── outputs/              # generated images, snapshots and recordings (git-ignored)
├── test.py               # offline self-checks and image smoke tests
├── pyproject.toml
└── requirements.txt
```

**One repository is recommended for this project.** The frontend is static and
served by the same FastAPI process; keeping both together avoids CORS/version
drift and keeps CI/deployment simple. Split into separate repositories only if
the UI gets its own framework, build pipeline, or deployment team.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r backend/requirements-dev.txt
pip install -e .
```

Run the API (serves the UI at `http://127.0.0.1:8000/`):

```bash
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

Or from inside `backend/`:

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

Run the webcam app:

```bash
python -m rupeevision.webcam
python -m rupeevision.webcam --camera 1 --conf 0.6 --backend onnxruntime --threads 4
```

Run checks:

```bash
python test.py
cd backend && python -m pytest -q
```

## API

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/health` | Liveness + model status |
| `GET` | `/api/model-info` | Input size, classes, backend, limits |
| `POST` | `/api/detect/image` | Upload a JPG/PNG (`file`) |
| `POST` | `/api/detect/frame` | Upload one webcam frame (`file`) |
| `GET` | `/docs` | Swagger UI |

Optional query parameter: `conf` (0–1, exclusive), e.g.
`POST /api/detect/image?conf=0.35`.

## Configuration

Every setting can be overridden with an environment variable (see
`.env.example`):

| Variable | Default | Meaning |
| --- | --- | --- |
| `RUPEEVISION_MODEL_PATH` | `models/model.onnx` | ONNX model path |
| `RUPEEVISION_CONF` | `0.6` | Default confidence threshold |
| `RUPEEVISION_IOU` | `0.45` | NMS IoU threshold |
| `RUPEEVISION_BACKEND` | `auto` | `auto`, `onnxruntime`, `opencv` |
| `RUPEEVISION_THREADS` | `0` (auto) | CPU inference threads |
| `RUPEEVISION_CUDA` | `false` | Prefer CUDA when available |
| `RUPEEVISION_MAX_DETECTIONS` | `300` | Maximum boxes per image |
| `RUPEEVISION_MAX_UPLOAD_BYTES` | `10485760` | Upload size limit |
| `RUPEEVISION_JPEG_QUALITY` | `85` | Annotated image quality |
| `RUPEEVISION_CORS_ORIGINS` | localhost origins | Comma-separated origins |

## Accuracy and performance notes

- **Backend choice:** `auto` uses **ONNX Runtime** when installed (typically
  faster than OpenCV DNN on CPU) and falls back to OpenCV DNN otherwise.
  `pip install onnxruntime` (or `onnxruntime-gpu`).
- **Single-pass preprocessing:** the frame is resized once and padding is added
  in the normalized float blob, avoiding a second full-size resize.
- **Batched NMS:** uses `cv2.dnn.NMSBoxesBatched` when available, with a
  class-aware fallback; detections are capped by `max_detections`.
- **Warm-up:** the backend runs one dummy inference at startup so the first
  request is not slow.
- **Live capture:** the browser requests frames at the model's input width to
  avoid sending pixels the model would only downscale again.
- **Accuracy is model-bound.** Pipeline code only guarantees that inference
  matches training preprocessing. To raise accuracy, improve the dataset and
  retrain (see `model_improvement/`): oversample positives, add geometric
  augmentation, try `yolov8s`, and evaluate at 512 vs 640. Always tune the
  confidence threshold on the validation split with negative (non-₹50) images.

## Docker

```bash
docker build -t rupeevision .
docker run --rm -p 8000:8000 rupeevision
```

## Model files

`models/model.onnx` (~12 MB) and `models/best.pt` (~6 MB) are committed for a
zero-setup clone. If they grow past GitHub's 100 MB limit, move them to a
release asset or Git LFS.

## License

Add your preferred license (for example MIT) before publishing if you want
others to reuse the code.
