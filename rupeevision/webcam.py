"""Live webcam ₹50 note detection.

Run from the project root:
    python -m rupeevision.webcam
    python -m rupeevision.webcam --camera 1 --conf 0.6 --record

Keys: q / Esc = quit, s = save snapshot to outputs/
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime

import cv2

from .detector import DEFAULT_MODEL_PATH, PROJECT_ROOT, DetectorError, RupeeDetector

OUTPUT_DIR = PROJECT_ROOT / "outputs"
WINDOW_NAME = "ZED Digital - Rs.50 Note Detector (q to quit, s to save)"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Detect the new ₹50 note from a webcam feed.")
    p.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    p.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="path to ONNX model")
    p.add_argument("--conf", type=float, default=0.6, help="confidence threshold (default 0.6)")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (default 0.45)")
    p.add_argument("--width", type=int, default=1280, help="requested capture width")
    p.add_argument("--height", type=int, default=720, help="requested capture height")
    p.add_argument("--record", action="store_true", help="also record annotated video to outputs/")
    p.add_argument("--cuda", action="store_true", help="use CUDA if the backend supports it")
    p.add_argument("--backend", choices=("auto", "onnxruntime", "opencv"), default="auto",
                   help="inference backend (auto prefers onnxruntime when installed)")
    p.add_argument("--threads", type=int, default=None, help="CPU inference threads")
    return p.parse_args(argv)


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    # DirectShow opens much faster than MSMF on most Windows webcams.
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {index}. Check that it is connected, not used by another "
            f"app, and that camera access is allowed in Windows Privacy settings. "
            f"Try --camera 1 if you have more than one camera.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        detector = RupeeDetector(args.model, args.conf, args.iou, use_cuda=args.cuda,
                                 backend=args.backend, num_threads=args.threads)
    except DetectorError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    print(f"[info] model: {detector.model_path}  backend: {detector.backend_name}  "
          f"input: {detector.input_hw}  output: {detector.output_shape}  "
          f"classes: {detector.class_names}")

    try:
        cap = open_camera(args.camera, args.width, args.height)
    except RuntimeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(exist_ok=True)
    writer = None
    fps = 0.0
    last = time.perf_counter()
    failed_reads = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                failed_reads += 1
                if failed_reads > 30:
                    print("[error] Camera stopped returning frames.", file=sys.stderr)
                    return 1
                continue
            failed_reads = 0

            try:
                detections = detector.detect(frame)
            except DetectorError as exc:
                print(f"[warn] {exc}", file=sys.stderr)
                detections = []
            annotated = detector.draw(frame, detections)

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6)) if fps else 1.0 / max(now - last, 1e-6)
            last = now
            cv2.putText(annotated, f"FPS {fps:.1f} | notes: {len(detections)}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            if args.record:
                if writer is None:
                    path = OUTPUT_DIR / f"webcam_{datetime.now():%Y%m%d_%H%M%S}.mp4"
                    h, w = annotated.shape[:2]
                    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
                    print(f"[info] recording to {path}")
                writer.write(annotated)

            cv2.imshow(WINDOW_NAME, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                path = OUTPUT_DIR / f"snapshot_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                cv2.imwrite(str(path), annotated)
                print(f"[info] saved {path}")
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
