"""Evaluate a YOLOv8 model (.pt via Ultralytics, or .onnx via the app's OpenCV DNN pipeline).

    python model_improvement/scripts/evaluate.py --weights models/best.pt --split valid --out model_improvement/baseline/pt_valid
    python model_improvement/scripts/evaluate.py --weights models/model.onnx --split valid --out ...

Writes:
  predictions.json   every box with conf >= 0.01 per image
  thresholds.json    per-threshold TP/FN/FP counts (image level, IoU >= 0.5)
  ultralytics.json   P/R/mAP from Ultralytics val (only for .pt)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

WS = Path(__file__).resolve().parents[1]
ROOT = WS.parent
sys.path.insert(0, str(ROOT))

THRESHOLDS = [0.10, 0.25, 0.40, 0.50, 0.60, 0.70]
PRED_CONF = 0.01
IOU_MATCH = 0.5


def load_gt(label_path: Path, w: int, h: int) -> list[list[float]]:
    boxes = []
    if label_path.exists():
        for line in label_path.read_text().split("\n"):
            if line.strip():
                _, x, y, bw, bh = map(float, line.split())
                boxes.append([(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h])
    return boxes


def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def predict_pt(weights: Path, images: list[Path], imgsz: int):
    from ultralytics import YOLO
    model = YOLO(str(weights))
    out = {}
    for p in images:
        r = model.predict(str(p), imgsz=imgsz, conf=PRED_CONF, iou=0.45, verbose=False, device="cpu")[0]
        out[p.name] = [[*map(float, b), float(c)] for b, c in
                       zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())]
    return out


def predict_onnx(weights: Path, images: list[Path]):
    from rupeevision.detector import RupeeDetector
    det = RupeeDetector(weights, conf_threshold=PRED_CONF, iou_threshold=0.45)
    out = {}
    for p in images:
        out[p.name] = [[d.x1, d.y1, d.x2, d.y2, d.confidence] for d in det.detect(cv2.imread(str(p)))]
    return out, det.input_hw[0]


def threshold_table(preds: dict, gts: dict) -> list[dict]:
    rows = []
    for t in THRESHOLDS:
        tp = fn = fp_pos = fp_neg_boxes = fp_neg_images = 0
        pos_imgs = neg_imgs = 0
        missed, false_alarms = [], []
        for name, gt in gts.items():
            boxes = [b for b in preds.get(name, []) if b[4] >= t]
            if gt:
                pos_imgs += 1
                used = set()
                for g in gt:
                    best = max(((iou(g, b), i) for i, b in enumerate(boxes) if i not in used), default=(0, -1))
                    if best[0] >= IOU_MATCH:
                        tp += 1
                        used.add(best[1])
                    else:
                        fn += 1
                        missed.append(name)
                fp_pos += len(boxes) - len(used)
            else:
                neg_imgs += 1
                if boxes:
                    fp_neg_images += 1
                    fp_neg_boxes += len(boxes)
                    false_alarms.append(f"{name} ({max(b[4] for b in boxes):.2f})")
        n_gt = tp + fn
        prec = tp / (tp + fp_pos + fp_neg_boxes) if (tp + fp_pos + fp_neg_boxes) else None
        rows.append({
            "conf": t, "positives": n_gt, "TP": tp, "FN": fn,
            "recall": round(tp / n_gt, 4) if n_gt else None,
            "precision": round(prec, 4) if prec is not None else None,
            "FP_boxes_on_positive_images": fp_pos,
            "negative_images": neg_imgs, "negative_images_with_FP": fp_neg_images,
            "FP_boxes_on_negatives": fp_neg_boxes,
            "missed": missed, "false_alarms": false_alarms,
        })
    return rows


def score_summary(preds: dict, gts: dict) -> dict:
    """Image-level separation: best matching score on positives vs best score on negatives."""
    pos, neg = [], []
    for name, gt in gts.items():
        boxes = preds.get(name, [])
        if gt:
            matched = [b[4] for b in boxes for g in gt if iou(g, b) >= IOU_MATCH]
            pos.append(max(matched, default=0.0))
        else:
            neg.append(max((b[4] for b in boxes), default=0.0))
    pos_a, neg_a = np.array(pos), np.array(neg)
    auc = float(np.mean([(p > n) + 0.5 * (p == n) for p in pos_a for n in neg_a])) if len(pos_a) and len(neg_a) else None
    return {
        "positive_best_scores": sorted([round(float(x), 4) for x in pos_a], reverse=True),
        "negative_top10_scores": sorted([round(float(x), 4) for x in neg_a], reverse=True)[:10],
        "image_level_auc": round(auc, 4) if auc is not None else None,
        "min_positive": round(float(pos_a.min()), 4) if len(pos_a) else None,
        "max_negative": round(float(neg_a.max()), 4) if len(neg_a) else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=WS / "data" / "v2.yaml")
    ap.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(args.data.read_text())
    split_dir = Path(cfg["path"]) / {"train": cfg["train"], "valid": cfg["val"], "test": cfg["test"]}[args.split]
    images = sorted(p for p in split_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    label_dir = split_dir.parent / "labels"
    gts = {}
    for p in images:
        h, w = cv2.imread(str(p)).shape[:2]
        gts[p.name] = load_gt(label_dir / f"{p.stem}.txt", w, h)

    args.out.mkdir(parents=True, exist_ok=True)
    result = {"weights": str(args.weights), "split": args.split, "n_images": len(images)}
    if args.weights.suffix == ".onnx":
        preds, imgsz = predict_onnx(args.weights, images)
        result["backend"] = "OpenCV DNN (app pipeline)"
    else:
        imgsz = args.imgsz
        preds = predict_pt(args.weights, images, imgsz)
        result["backend"] = "Ultralytics PyTorch"
        from ultralytics import YOLO
        split_key = {"train": "train", "valid": "val", "test": "test"}[args.split]
        m = YOLO(str(args.weights)).val(data=str(args.data), split=split_key, imgsz=imgsz, batch=8,
                                        device="cpu", plots=True, verbose=False,
                                        project=str(args.out.resolve()), name="ultralytics_val", exist_ok=True)
        result["ultralytics"] = {
            "precision": round(float(m.box.mp), 4), "recall": round(float(m.box.mr), 4),
            "mAP50": round(float(m.box.map50), 4), "mAP50_95": round(float(m.box.map), 4),
            "note": "Ultralytics P/R are at the max-F1 confidence, not a fixed threshold",
        }
    result["imgsz"] = imgsz
    result["scores"] = score_summary(preds, gts)
    result["thresholds"] = threshold_table(preds, gts)

    (args.out / "predictions.json").write_text(json.dumps(preds, indent=0))
    (args.out / "summary.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "thresholds"}, indent=1))
    print(f"{'conf':>5} {'TP':>3} {'FN':>3} {'recall':>7} {'negFP_img':>9} {'FPbox':>5}")
    for r in result["thresholds"]:
        print(f"{r['conf']:>5} {r['TP']:>3} {r['FN']:>3} {str(r['recall']):>7} "
              f"{r['negative_images_with_FP']:>4}/{r['negative_images']:<4} "
              f"{r['FP_boxes_on_negatives'] + r['FP_boxes_on_positive_images']:>5}")


if __name__ == "__main__":
    main()
