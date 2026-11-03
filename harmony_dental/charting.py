"""Turn raw X-ray detections into the perio charting output (perio/SPEC.md)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "perio"))
from perio import bone_levels, stage  # noqa: E402

RESTORATIONS = {"Crown": "crown", "Filling": "filling", "Implant": "implant", "Root canal treatment": "root canal treatment"}
FINDINGS = {"Caries": "caries", "Deep caries": "deep caries", "Periapical lesion": "periapical lesion", "Badly decayed": "badly decayed"}
LANDMARKS = {"CEJ", "Root apex", "Bone crest"}
ALL_TEETH = [f"{q}{t}" for q in (1, 2, 3, 4) for t in range(1, 9)]


def _overlap(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))


def chart(findings, film="unknown", min_tooth=0.5, model_version=None):
    """findings: [{class, score, box}] from the X-ray part (all scores)."""
    teeth = {}
    for f in findings:
        if f["class"] in ALL_TEETH and f["score"] >= min_tooth:
            if f["class"] not in teeth or f["score"] > teeth[f["class"]]["confidence"]:
                teeth[f["class"]] = {"fdi": f["class"], "present": True, "confidence": round(f["score"], 3),
                                     "box": [round(v, 1) for v in f["box"]], "restorations": [], "findings": []}
    marks = [{"class": f["class"], "box": f["box"], "score": f["score"]} for f in findings if f["class"] in LANDMARKS]
    for t in teeth.values():
        for f in findings:
            if f["score"] < 0.3 or _overlap(t["box"], f["box"]) < 0.5:
                continue
            if f["class"] in RESTORATIONS:
                t["restorations"].append({"kind": RESTORATIONS[f["class"]], "confidence": round(f["score"], 3)})
            elif f["class"] in FINDINGS:
                t["findings"].append({"kind": FINDINGS[f["class"]], "confidence": round(f["score"], 3), "box": [round(v, 1) for v in f["box"]]})
        q = int(t["fdi"][0])
        # On the image the patient's right is on the viewer's left: mesial is toward the midline.
        mesial_is_left = q in (2, 3)
        bl = bone_levels(t["box"], marks, mesial_is_left=mesial_is_left)
        if bl:
            t["bone_level"] = bl
            t["suggested_stage"] = stage(max(s["loss_pct"] for s in bl.values()))
    seen = sorted(teeth)
    span = {q for q in (1, 2, 3, 4) if any(f.startswith(str(q)) for f in seen)}
    missing = [f for f in ALL_TEETH if int(f[0]) in span and f not in teeth] if film == "panoramic" else []
    return {"type": "xray", "film": film, "teeth": [teeth[k] for k in seen], "missing_teeth": missing,
            "model_version": model_version}
