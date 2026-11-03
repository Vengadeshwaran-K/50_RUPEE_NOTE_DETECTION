"""Offline checks for the ₹50 detector (no webcam needed).

    python test.py                                  # self-checks only
    python test.py --model model.onnx               # same, for a specific ONNX file
    python test.py --source path/to/image_or_folder # + annotate images into outputs/
    python test.py --source ... --compare-pt        # + compare against models/best.pt (needs ultralytics)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from rupeevision.detector import DEFAULT_MODEL_PATH, PROJECT_ROOT, DetectorError, RupeeDetector
from rupeevision.onnx_utils import LetterboxInfo, decode_yolov8, letterbox

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "test"
PT_PATH = PROJECT_ROOT / "models" / "best.pt"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def check(name: str, condition: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")
    return condition


def self_checks(detector: RupeeDetector) -> bool:
    print("Self-checks:")
    ok = True
    h, w = detector.input_hw
    anchors = sum((h // s) * (w // s) for s in (8, 16, 32))  # YOLOv8 strides
    ok &= check("input size", h % 32 == 0 and w % 32 == 0, str(detector.input_hw))
    ok &= check("output shape", detector.output_shape in ((1, 5, anchors), (1, anchors, 5)),
                str(detector.output_shape))
    ok &= check("class names", detector.class_names == {0: "rupee_50"}, str(detector.class_names))

    # Letterbox maps a known box back to the original frame exactly.
    frame = np.zeros((480, 1280, 3), np.uint8)
    padded, lb = letterbox(frame, (640, 640))
    ok &= check("letterbox output size", padded.shape[:2] == (640, 640), str(padded.shape))
    x1, y1, x2, y2 = 200, 100, 600, 300
    cx = ((x1 + x2) / 2) * lb.scale + lb.pad_x
    cy = ((y1 + y2) / 2) * lb.scale + lb.pad_y
    w, h = (x2 - x1) * lb.scale, (y2 - y1) * lb.scale
    raw = np.zeros((1, 5, 8400), np.float32)
    raw[0, :, 0] = [cx, cy, w, h, 0.9]
    dets = decode_yolov8(raw, 1, 0.5, 0.45, lb, frame.shape)
    ok &= check("box maps back to frame coords",
                len(dets) == 1 and max(abs(dets[0].x1 - x1), abs(dets[0].y1 - y1),
                                       abs(dets[0].x2 - x2), abs(dets[0].y2 - y2)) <= 1,
                str(dets[0]) if dets else "no detection")

    # Transposed layout [1, N, 5] decodes identically.
    dets_t = decode_yolov8(raw.transpose(0, 2, 1), 1, 0.5, 0.45, lb, frame.shape)
    ok &= check("transposed output layout", dets_t == dets)

    # NMS removes a near-duplicate box.
    raw[0, :, 1] = [cx + 2, cy + 2, w, h, 0.8]
    ok &= check("NMS suppresses duplicates", len(decode_yolov8(raw, 1, 0.5, 0.45, lb, frame.shape)) == 1)

    try:
        decode_yolov8(np.zeros((1, 6, 100), np.float32), 1, 0.5, 0.45, LetterboxInfo(1, 0, 0), (640, 640))
        ok &= check("rejects wrong output shape", False)
    except ValueError:
        ok &= check("rejects wrong output shape", True)

    blank = detector.detect(np.full((720, 1280, 3), 127, np.uint8))
    ok &= check("no detections on blank grey frame", len(blank) == 0, f"{len(blank)} found")
    return ok


def collect_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if source.is_dir():
        return sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    raise FileNotFoundError(f"--source not found: {source}")


def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def run_images(detector: RupeeDetector, images: list[Path], compare_pt: bool) -> bool:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pt_model = None
    if compare_pt:
        try:
            from ultralytics import YOLO
        except ImportError:
            print("[error] --compare-pt needs `pip install ultralytics`", file=sys.stderr)
            return False
        if not PT_PATH.is_file():
            print(f"[error] {PT_PATH} not found", file=sys.stderr)
            return False
        pt_model = YOLO(str(PT_PATH))

    print(f"\nRunning on {len(images)} image(s) -> {OUTPUT_DIR}")
    mismatches = 0
    for path in images:
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"  [skip] unreadable: {path.name}")
            continue
        dets = detector.detect(frame)
        cv2.imwrite(str(OUTPUT_DIR / f"{path.stem}_pred.jpg"), detector.draw(frame, dets))
        summary = ", ".join(f"{d.confidence:.2f}" for d in dets) or "none"
        line = f"  {path.name[:60]:60s} notes={len(dets)} conf=[{summary}]"

        if pt_model is not None:
            r = pt_model.predict(frame, imgsz=detector.input_hw[0], conf=detector.conf_threshold,
                                 iou=detector.iou_threshold, verbose=False)[0]
            pt_boxes = r.boxes.xyxy.cpu().numpy().tolist()
            pt_conf = r.boxes.conf.cpu().numpy().tolist()
            match = len(pt_boxes) == len(dets) and all(
                max(iou((d.x1, d.y1, d.x2, d.y2), pb) for pb in pt_boxes) > 0.9
                and min(abs(d.confidence - c) for c in pt_conf) < 0.02
                for d in dets)
            mismatches += not match
            line += f" | pt notes={len(pt_boxes)} {'match' if match else 'MISMATCH'}"
        print(line)

    if pt_model is not None:
        return check("ONNX/OpenCV matches PyTorch model", mismatches == 0,
                     f"{mismatches} mismatch(es) in {len(images)} image(s)")
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH,
                   help="ONNX model (default models/model.onnx; a bare name is also looked up in models/)")
    p.add_argument("--source", type=Path, help="image file or folder of images")
    p.add_argument("--conf", type=float, default=0.6)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--compare-pt", action="store_true", help="cross-check against models/best.pt")
    args = p.parse_args()

    try:
        model = args.model
        if not model.is_file() and (PROJECT_ROOT / "models" / model).is_file():
            model = PROJECT_ROOT / "models" / model
        detector = RupeeDetector(model, conf_threshold=args.conf, iou_threshold=args.iou)
        print(f"Model: {detector.model_path}")
    except DetectorError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    ok = self_checks(detector)
    if args.source:
        try:
            ok &= run_images(detector, collect_images(args.source), args.compare_pt)
        except FileNotFoundError as exc:
            print(f"[error] {exc}", file=sys.stderr)
            return 1
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
