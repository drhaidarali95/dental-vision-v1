"""Score the MONAI model on the held-out test split, against the baseline.

  python evaluate.py --datalist data/datalist.json --model runs/student/best.pt \
    --baseline harmony_detector_best.pth --out runs/report_test.json

Two tables:
  monai_4_class  - all four DENTEX classes, MONAI only.
  harmony_wire   - the classes Harmony actually receives, scored the way the
                   wire collapses them (caries + deep caries = "cavity",
                   impacted = "impacted tooth"), MONAI vs baseline side by side.
Run this once, at the end. Do not tune anything on these numbers.
"""
import argparse
import json
from pathlib import Path

import torch

from labels import CLASSES, WIRE_CATEGORY
from metrics import coco_metrics
from model import load_detector, open_film, predict

WIRE_CLASSES = sorted(set(WIRE_CATEGORY.values()))


def to_wire_items(items):
    out = []
    for it in items:
        boxes, labels = [], []
        for box, label in zip(it["box"], it["label"]):
            wire = WIRE_CATEGORY.get(CLASSES[label])
            if wire:
                boxes.append(box)
                labels.append(WIRE_CLASSES.index(wire))
        out.append({"box": boxes, "label": labels})
    return out


def to_wire_preds(preds):
    return [[{**p, "class": WIRE_CATEGORY[p["class"]]} for p in film if p["class"] in WIRE_CATEGORY] for film in preds]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datalist", required=True, type=Path)
    p.add_argument("--model", required=True)
    p.add_argument("--baseline")
    p.add_argument("--split", default="test")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    items = json.loads(args.datalist.read_text())[args.split]
    detector, ckpt = load_detector(args.model, device)
    films = [open_film(it["image"]) for it in items]
    monai_preds = [predict(detector, f, device) for f in films]

    report = {
        "split": args.split,
        "films": len(items),
        "model": {"path": args.model, "epoch": ckpt.get("epoch"), "trained_on": ckpt.get("trained_on")},
        "monai_4_class": coco_metrics(items, monai_preds, CLASSES),
        "harmony_wire": {"classes": WIRE_CLASSES,
                         "monai": coco_metrics(to_wire_items(items), to_wire_preds(monai_preds), WIRE_CLASSES)},
        "note": "DENTEX image-level hold-out. Not an external or clinical validation.",
    }
    if args.baseline:
        from baseline import Baseline

        base = Baseline(args.baseline, device)
        base_preds = [base.predict(f) for f in films]
        report["baseline"] = {"path": args.baseline, "classes": base.names, "epoch": base.epoch}
        report["harmony_wire"]["baseline_faster_rcnn"] = coco_metrics(
            to_wire_items(items), to_wire_preds(base_preds), WIRE_CLASSES)
    else:
        report["baseline"] = {"used": False, "reason": "no baseline checkpoint supplied"}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
