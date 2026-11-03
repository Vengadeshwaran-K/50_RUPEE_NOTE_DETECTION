"""Run one tracked training experiment.

    python model_improvement/scripts/train.py exp02

Experiment definitions live in model_improvement/experiments.json. Each one
lists only the settings it changes relative to "base"; the merged arguments,
Ultralytics logs/weights and a validation evaluation are saved to
model_improvement/experiments/<name>/.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
ROOT = WS.parent


def build_oversampled_list(data_yaml: Path, factor: int, out: Path) -> Path:
    """Write a train image list where positive images are repeated `factor` times (dataset untouched)."""
    import yaml
    cfg = yaml.safe_load(data_yaml.read_text())
    img_dir = Path(cfg["path"]) / cfg["train"]
    lbl_dir = img_dir.parent / "labels"
    lines = []
    for p in sorted(img_dir.iterdir()):
        lbl = lbl_dir / f"{p.stem}.txt"
        positive = lbl.exists() and lbl.read_text().strip() != ""
        lines += [p.as_posix()] * (factor if positive else 1)
    list_path = out / "train_list.txt"
    list_path.write_text("\n".join(lines) + "\n")
    new_cfg = dict(cfg, train=list_path.as_posix())
    new_yaml = out / "data.yaml"
    new_yaml.write_text(yaml.safe_dump(new_cfg, sort_keys=False))
    return new_yaml


def main() -> None:
    name = sys.argv[1]
    spec = json.loads((WS / "experiments.json").read_text())
    exp = spec["experiments"][name]
    args = {**spec["base"], **exp.get("args", {})}
    out = WS / "experiments" / name
    out.mkdir(parents=True, exist_ok=True)

    data_yaml = Path(args.pop("data"))
    if not data_yaml.is_absolute():
        data_yaml = ROOT / data_yaml
    oversample = args.pop("oversample_positives", 1)
    if oversample > 1:
        data_yaml = build_oversampled_list(data_yaml, oversample, out)
    model = args.pop("model")

    record = {"name": name, "hypothesis": exp.get("hypothesis"), "changes_vs_base": exp.get("args", {}),
              "model": model, "data": str(data_yaml), "oversample_positives": oversample, "train_args": args}
    (out / "experiment.json").write_text(json.dumps(record, indent=1))

    from ultralytics import YOLO
    t0 = time.time()
    YOLO(model).train(data=str(data_yaml), project=str(out), name="train", exist_ok=True, **args)
    record["train_minutes"] = round((time.time() - t0) / 60, 1)
    (out / "experiment.json").write_text(json.dumps(record, indent=1))

    best = out / "train" / "weights" / "best.pt"
    subprocess.run([sys.executable, str(WS / "scripts" / "evaluate.py"), "--weights", str(best),
                    "--split", "valid", "--imgsz", str(args.get("imgsz", 640)),
                    "--out", str(out / "eval_valid")], check=True)


if __name__ == "__main__":
    main()
