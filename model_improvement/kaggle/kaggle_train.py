"""RupeeVision ₹50 — controlled YOLOv8 experiments for Kaggle (GPU T4).

Paste this whole file into one notebook cell (or `!python kaggle_train.py`).
Requirements: GPU accelerator ON, Internet ON (to pip install ultralytics and
fetch yolov8n/yolov8s COCO weights), dataset Indian_Rupee_50_Corrected_v2 attached.

It NEVER evaluates on the test split; test is evaluated once, locally, for the
selected candidate only. Output: /kaggle/working/rupee50_experiments.zip
"""

import glob
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.175"], check=True)

import torch  # noqa: E402
import yaml  # noqa: E402
from ultralytics import YOLO  # noqa: E402

# ---------------------------------------------------------------- dataset
cands = [p for p in glob.glob("/kaggle/input/**/Indian_Rupee_50_Corrected_v2/data.yaml", recursive=True)]
assert cands, "Attach the Indian_Rupee_50_Corrected_v2 dataset to this notebook"
DS = Path(cands[0]).parent
OUT = Path("/kaggle/working/rupee50_experiments")
OUT.mkdir(parents=True, exist_ok=True)
print("dataset:", DS)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE",
      f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB" if torch.cuda.is_available() else "")
assert torch.cuda.is_available(), "Turn on the GPU accelerator"

BASE = dict(model="yolov8n.pt", epochs=100, patience=20, imgsz=512, batch=16, optimizer="AdamW",
            lr0=0.002, seed=42, deterministic=True, device=0, workers=2, plots=True, verbose=False)

# Each experiment changes ONE thing relative to the experiment named in "from".
EXPERIMENTS = {
    "exp01_control":      dict(frm=None, args={}),
    "exp02_oversample8":  dict(frm="exp01_control", args={"oversample": 8}),
    "exp03_aug_geo_light": dict(frm="exp02_oversample8",
                                args={"degrees": 10.0, "perspective": 0.0005, "hsv_v": 0.5}),
    "exp04_yolov8s":      dict(frm="exp02_oversample8", args={"model": "yolov8s.pt"}),
    "exp05_imgsz640":     dict(frm="exp02_oversample8", args={"imgsz": 640}),
    # Seed repeats: how much of any difference is run-to-run noise?
    "exp01_control_seed1":     dict(frm="exp01_control", args={"seed": 1}),
    "exp02_oversample8_seed1": dict(frm="exp02_oversample8", args={"seed": 1}),
}


def resolve(name):
    e = EXPERIMENTS[name]
    args = dict(BASE) if e["frm"] is None else resolve(e["frm"])
    args.update(e["args"])
    return args


def make_data_yaml(exp_dir, oversample):
    imgs = sorted((DS / "train" / "images").iterdir())
    lines = []
    for p in imgs:
        lbl = DS / "train" / "labels" / f"{p.stem}.txt"
        pos = lbl.exists() and lbl.read_text().strip() != ""
        lines += [str(p)] * (oversample if pos else 1)
    (exp_dir / "train_list.txt").write_text("\n".join(lines) + "\n")
    cfg = {"train": str(exp_dir / "train_list.txt"), "val": str(DS / "valid" / "images"),
           "nc": 1, "names": {0: "rupee_50"}}
    y = exp_dir / "data.yaml"
    y.write_text(yaml.safe_dump(cfg))
    return y


# ---------------------------------------------------------------- evaluation (validation only)
def load_gt(lbl, w, h):
    out = []
    if lbl.exists():
        for line in lbl.read_text().split("\n"):
            if line.strip():
                _, x, y, bw, bh = map(float, line.split())
                out.append([(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h])
    return out


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0


def evaluate_val(weights, imgsz):
    model = YOLO(str(weights))
    preds, gts = {}, {}
    for p in sorted((DS / "valid" / "images").iterdir()):
        r = model.predict(str(p), imgsz=imgsz, conf=0.01, iou=0.45, verbose=False)[0]
        h, w = r.orig_shape
        gts[p.name] = load_gt(DS / "valid" / "labels" / f"{p.stem}.txt", w, h)
        preds[p.name] = [[*map(float, b), float(c)] for b, c in
                         zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())]
    rows = []
    for t in [0.10, 0.25, 0.40, 0.50, 0.60, 0.70]:
        tp = fn = fp_neg_img = fp_box = 0
        for n, gt in gts.items():
            bx = [b for b in preds[n] if b[4] >= t]
            if gt:
                used = set()
                for g in gt:
                    m = max(((iou(g, b), i) for i, b in enumerate(bx) if i not in used), default=(0, -1))
                    if m[0] >= 0.5:
                        tp += 1
                        used.add(m[1])
                    else:
                        fn += 1
                fp_box += len(bx) - len(used)
            elif bx:
                fp_neg_img += 1
                fp_box += len(bx)
        rows.append({"conf": t, "TP": tp, "FN": fn, "recall": round(tp / (tp + fn), 4),
                     "neg_images_with_FP": fp_neg_img, "FP_boxes": fp_box})
    pos = [max([b[4] for b in preds[n] for g in gt if iou(g, b) >= 0.5], default=0) for n, gt in gts.items() if gt]
    neg = [max([b[4] for b in preds[n]], default=0) for n, gt in gts.items() if not gt]
    m = model.val(data=str(OUT / "_val.yaml"), imgsz=imgsz, batch=16, verbose=False, plots=False)
    return {"thresholds": rows,
            "positive_best_scores": sorted([round(x, 4) for x in pos], reverse=True),
            "negative_top5": sorted([round(x, 4) for x in neg], reverse=True)[:5],
            "ultralytics": {"P": round(float(m.box.mp), 4), "R": round(float(m.box.mr), 4),
                            "mAP50": round(float(m.box.map50), 4), "mAP50_95": round(float(m.box.map), 4)},
            "predictions": preds}


(OUT / "_val.yaml").write_text(yaml.safe_dump({"train": str(DS / "valid" / "images"), "val": str(DS / "valid" / "images"),
                                               "nc": 1, "names": {0: "rupee_50"}}))

# ---------------------------------------------------------------- run
summary = {}
for name in EXPERIMENTS:
    exp_dir = OUT / name
    if (exp_dir / "result.json").exists():
        summary[name] = json.loads((exp_dir / "result.json").read_text())
        continue
    exp_dir.mkdir(exist_ok=True)
    args = resolve(name)
    oversample = args.pop("oversample", 1)
    model_name = args.pop("model")
    data = make_data_yaml(exp_dir, oversample)
    t0 = time.time()
    YOLO(model_name).train(data=str(data), project=str(exp_dir), name="train", exist_ok=True, **args)
    minutes = round((time.time() - t0) / 60, 1)
    ev = evaluate_val(exp_dir / "train" / "weights" / "best.pt", args["imgsz"])
    (exp_dir / "val_predictions.json").write_text(json.dumps(ev.pop("predictions")))
    res = {"from": EXPERIMENTS[name]["frm"], "changes": EXPERIMENTS[name]["args"],
           "args": {**args, "model": model_name, "oversample": oversample}, "train_minutes": minutes, **ev}
    (exp_dir / "result.json").write_text(json.dumps(res, indent=1))
    summary[name] = res
    print(f"\n### {name}: {minutes} min  val pos scores {res['positive_best_scores']}  "
          f"neg top {res['negative_top5']}  {res['ultralytics']}")

(OUT / "summary.json").write_text(json.dumps(summary, indent=1))
print("\n\nexperiment            P      R    mAP50  mAP50-95 | recall@0.25/0.5  negFP@0.25/0.5")
for n, r in summary.items():
    u = r["ultralytics"]
    t = {x["conf"]: x for x in r["thresholds"]}
    print(f"{n:24s} {u['P']:.3f} {u['R']:.3f} {u['mAP50']:.3f} {u['mAP50_95']:.3f} | "
          f"{t[0.25]['recall']:.2f}/{t[0.5]['recall']:.2f}   {t[0.25]['neg_images_with_FP']}/{t[0.5]['neg_images_with_FP']}")

# Package everything except the big optimizer-state copies
pack = Path("/kaggle/working/rupee50_pack")
if pack.exists():
    shutil.rmtree(pack)
for f in OUT.rglob("*"):
    if f.is_file() and not f.name.startswith("last") and f.suffix not in {".cache"}:
        dst = pack / f.relative_to(OUT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)
shutil.make_archive("/kaggle/working/rupee50_experiments", "zip", pack)
print("\nDownload /kaggle/working/rupee50_experiments.zip and put it in "
      "model_improvement/kaggle_results/ on your PC.")
