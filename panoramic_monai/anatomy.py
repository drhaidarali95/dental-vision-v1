"""Normal anatomy layer: numbered teeth, and which tooth a finding sits on.

The tooth model (train.py on prepare_teeth.py's datalist) returns boxes for
FDI teeth 11-48. This module turns those into one box per tooth number and
plain-language names, and matches findings to teeth by overlap.
"""
import torch
from PIL import Image

from labels import TEETH, tooth_name
from model import predict


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def read_teeth(detector, img: Image.Image, device: torch.device, min_score: float = 0.4) -> list[dict]:
    """[{tooth: '36', name, confidence, box}], one per tooth number, left to right.

    The detector may give one tooth two neighbouring numbers; the stronger
    box wins, and each number is used at most once.
    """
    kept: list[dict] = []
    for p in sorted(predict(detector, img, device), key=lambda d: -d["score"]):
        if p["score"] < min_score or p["class"] not in TEETH:
            continue
        if any(k["tooth"] == p["class"] or _iou(k["box"], p["box"]) > 0.6 for k in kept):
            continue
        kept.append({"tooth": p["class"], "name": tooth_name(p["class"]),
                     "confidence": round(p["score"], 4), "box": p["box"]})
    return sorted(kept, key=lambda t: (t["box"][0] + t["box"][2]) / 2)


def tooth_for(box: list[float], teeth: list[dict], min_share: float = 0.3) -> str | None:
    """FDI number of the tooth covering most of `box`, if it covers enough of it."""
    area = max(1e-6, (box[2] - box[0]) * (box[3] - box[1]))
    best, share = None, 0.0
    for t in teeth:
        b = t["box"]
        ix = max(0.0, min(box[2], b[2]) - max(box[0], b[0]))
        iy = max(0.0, min(box[3], b[3]) - max(box[1], b[1]))
        s = ix * iy / area
        if s > share:
            best, share = t["tooth"], s
    return best if share >= min_share else None
