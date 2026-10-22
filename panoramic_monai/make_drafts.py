"""Draft labels for the unlabelled DENTEX panoramics.

  python make_drafts.py --datalist data/datalist.json --teacher runs/teacher/best.pt \
    --baseline harmony_detector_best.pth --out runs/drafts.json

A draft box is kept when either
  - the MONAI teacher is confident on its own (score >= --keep), or
  - the teacher is moderately sure (score >= --agree) AND, for a class the
    Faster R-CNN baseline also knows (caries / impacted), the baseline found
    an overlapping box (IoU >= --iou, score >= --baseline-min).
Films that end up with no boxes are left out rather than taught as clean.

--baseline is optional. Without it only the first rule applies, and the
output records that the baseline was not used.

Drafts are machine labels. They only ever join the training split.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from labels import CLASS_INDEX, CLASSES
from model import load_detector, open_film, predict


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def same_family(a: str, b: str) -> bool:
    """The baseline has one caries class; it may confirm caries or deep caries."""
    caries = {"caries", "deep_caries"}
    return a == b or (a in caries and b in caries)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datalist", required=True, type=Path)
    p.add_argument("--teacher", required=True)
    p.add_argument("--baseline")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--keep", type=float, default=0.6)
    p.add_argument("--agree", type=float, default=0.35)
    p.add_argument("--baseline-min", type=float, default=0.5)
    p.add_argument("--iou", type=float, default=0.4)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = json.loads(args.datalist.read_text())
    films = data["unlabeled"][: args.limit or None]
    teacher, _ = load_detector(args.teacher, device)
    baseline = None
    if args.baseline:
        from baseline import Baseline

        baseline = Baseline(args.baseline, device)
        print(f"baseline classes {baseline.names} -> covers {baseline.covers}")

    items, by_rule = [], Counter()
    for n, film in enumerate(films, 1):
        img = open_film(film["image"])
        t_preds = predict(teacher, img, device)
        b_preds = [b for b in baseline.predict(img) if b["score"] >= args.baseline_min] if baseline else []
        boxes, labels, scores, sources = [], [], [], []
        for t in t_preds:
            rule = None
            if t["score"] >= args.keep:
                rule = "teacher"
            elif t["score"] >= args.agree and baseline and (t["class"] in baseline.covers or
                                                             (t["class"] == "deep_caries" and "caries" in baseline.covers)):
                if any(same_family(t["class"], b["class"]) and iou(t["box"], b["box"]) >= args.iou for b in b_preds):
                    rule = "teacher+baseline"
            if rule:
                boxes.append(t["box"])
                labels.append(CLASS_INDEX[t["class"]])
                scores.append(round(t["score"], 4))
                sources.append(rule)
                by_rule[(t["class"], rule)] += 1
        if boxes:
            items.append({"image": film["image"], "box": boxes, "label": labels, "score": scores, "source": sources})
        if n % 100 == 0:
            print(f"{n}/{len(films)} films, {len(items)} with drafts", flush=True)

    summary = {f"{c} via {r}": k for (c, r), k in sorted(by_rule.items())}
    out = {
        "classes": CLASSES,
        "note": "Machine-made draft labels. Training split only; never val/test; not clinician reviewed.",
        "settings": {k: getattr(args, k) for k in ("keep", "agree", "baseline_min", "iou")},
        "teacher": args.teacher,
        "baseline": {"used": bool(baseline), "path": args.baseline,
                     "classes": baseline.names if baseline else None, "epoch": baseline.epoch if baseline else None},
        "films_scanned": len(films),
        "films_with_drafts": len(items),
        "boxes": summary,
        "items": items,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))
    print(json.dumps({k: v for k, v in out.items() if k != "items"}, indent=2))


if __name__ == "__main__":
    main()
