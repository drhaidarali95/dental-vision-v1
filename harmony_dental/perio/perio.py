"""Bone level per tooth side from detected landmarks (pure geometry, no AI).

  bone_levels(tooth_box, landmarks) -> {"mesial": {...}, "distal": {...}}
landmarks: list of {"class": "CEJ"|"Root apex"|"Bone crest", "box": [...], "score": s}.
Mesial/distal = the two sides of the tooth box (which is which depends on the tooth's
quadrant; the caller passes `mesial_is_left`).
"""
import math


def _c(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def _inside(p, box, pad=0.15):
    w, h = box[2] - box[0], box[3] - box[1]
    return box[0] - pad * w <= p[0] <= box[2] + pad * w and box[1] - pad * h <= p[1] <= box[3] + pad * h


def bone_levels(tooth_box, landmarks, mesial_is_left=True, min_score=0.3):
    pts = {k: [(_c(l["box"]), l["score"]) for l in landmarks
               if l["class"] == k and l["score"] >= min_score and _inside(_c(l["box"]), tooth_box)]
           for k in ("CEJ", "Root apex", "Bone crest")}
    if not pts["CEJ"] or not pts["Root apex"]:
        return {}
    mid = (tooth_box[0] + tooth_box[2]) / 2
    out = {}
    for side, left in (("mesial", mesial_is_left), ("distal", not mesial_is_left)):
        sel = lambda L: [p for p in L if (p[0][0] < mid) == left]
        cej, crest = sel(pts["CEJ"]), sel(pts["Bone crest"])
        apex = sel(pts["Root apex"]) or pts["Root apex"]
        if not cej or not crest:
            continue
        (c, cs) = max(cej, key=lambda p: p[1])
        (a, as_) = min(apex, key=lambda p: math.dist(p[0], c))
        (b, bs) = min(crest, key=lambda p: math.dist(p[0], c))
        root = math.dist(c, a)
        if root < 1:
            continue
        loss = max(0.0, min(100.0, 100 * math.dist(c, b) / root))
        out[side] = {"loss_pct": round(loss, 1), "cej": c, "crest": b, "apex": a,
                     "confidence": round(min(cs, as_, bs), 3)}
    return out


def stage(max_loss_pct):
    """2017 AAP/EFP radiographic bone-loss staging (suggestion for the dentist)."""
    if max_loss_pct is None:
        return None
    return "I" if max_loss_pct < 15 else "II" if max_loss_pct <= 33 else "III/IV"
