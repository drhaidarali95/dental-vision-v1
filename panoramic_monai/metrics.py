"""COCO-style box metrics over datalist items, via pycocotools."""
import contextlib
import io

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def coco_metrics(items: list[dict], preds: list[list[dict]], classes: list[str]) -> dict:
    """items: [{box, label}] ground truth; preds: per item [{class, score, box}].

    Returns overall and per-class mAP@0.5 and mAP@0.5:0.95. A class with no
    ground truth boxes reports None rather than a misleading number.
    """
    gt = {"images": [], "annotations": [], "categories": [{"id": i, "name": c} for i, c in enumerate(classes)]}
    dt = []
    for img_id, (item, pred) in enumerate(zip(items, preds)):
        gt["images"].append({"id": img_id})
        for box, label in zip(item["box"], item["label"]):
            x1, y1, x2, y2 = box
            gt["annotations"].append({"id": len(gt["annotations"]) + 1, "image_id": img_id, "category_id": int(label),
                                      "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
        for p in pred:
            if p["class"] not in classes:
                continue
            x1, y1, x2, y2 = p["box"]
            dt.append({"image_id": img_id, "category_id": classes.index(p["class"]),
                       "bbox": [x1, y1, x2 - x1, y2 - y1], "score": p["score"]})
    if not dt:
        return {"mAP_50": 0.0, "mAP_50_95": 0.0, "per_class": {c: None for c in classes}, "predictions": 0}

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = gt
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(dt)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.params.maxDets = [1, 10, 150]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()

    per_class = {}
    precision = ev.eval["precision"]  # [iou, recall, class, area, maxdet]
    for k, name in enumerate(classes):
        has_gt = any(a["category_id"] == k for a in gt["annotations"])
        if not has_gt:
            per_class[name] = None
            continue
        p_all = precision[:, :, k, 0, -1]
        p_50 = precision[0, :, k, 0, -1]
        per_class[name] = {
            "AP_50": float(np.mean(p_50[p_50 > -1])) if (p_50 > -1).any() else 0.0,
            "AP_50_95": float(np.mean(p_all[p_all > -1])) if (p_all > -1).any() else 0.0,
            "gt_boxes": sum(a["category_id"] == k for a in gt["annotations"]),
        }
    return {"mAP_50_95": float(ev.stats[0]), "mAP_50": float(ev.stats[1]),
            "per_class": per_class, "predictions": len(dt)}
