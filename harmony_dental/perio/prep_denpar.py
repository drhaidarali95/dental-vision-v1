"""DenPAR (Zenodo 16645076, CC BY 4.0; 1,000 periapicals) -> detector items for the
perio landmark skill: small boxes for "CEJ", "Root apex" and "Bone crest" points.

The landmarks become extra classes of the one X-ray detector; perio.py turns detected
landmarks into bone-loss % per tooth side. Official Training/Validation/Testing splits kept.

  python prep_denpar.py --root <dir containing Dataset/> --out denpar_items.json
"""
import argparse
import json
from pathlib import Path

HALF = 8  # landmark box half-size in pixels


def pt_box(x, y):
    return [x - HALF, y - HALF, x + HALF, y + HALF]


def crest_points(line, step=40.0):
    """Sample a bone-crest polyline every ~step px (keeps both ends)."""
    import math
    pts = [line[0]]
    for (x1, y1), (x2, y2) in zip(line, line[1:]):
        d = math.hypot(x2 - x1, y2 - y1)
        n = max(1, int(d // step))
        pts += [[x1 + (x2 - x1) * k / n, y1 + (y2 - y1) * k / n] for k in range(1, n + 1)]
    return pts


def load_split(d: Path, source="denpar-perio"):
    items = []
    for kp in sorted((d / "Key Points Annotations").glob("*.json")):
        k = json.loads(kp.read_text())
        img = d / "Images" / k["Image_id"]
        if not img.exists():
            continue
        box, name = [], []
        for x, y in k.get("CEJ_Points", []):
            box.append(pt_box(x, y)); name.append("CEJ")
        for x, y in k.get("Apex_Points", []):
            box.append(pt_box(x, y)); name.append("Root apex")
        bl = d / "Bone Level Annotations" / kp.name
        if bl.exists():
            for line in json.loads(bl.read_text()).get("Bone_Lines", []):
                for x, y in crest_points(line):
                    box.append(pt_box(x, y)); name.append("Bone crest")
        if box:
            items.append({"image": str(img.resolve()), "box": box, "name": name, "source": source})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    ds = next(a.root.rglob("Key Points Annotations")).parent.parent
    out = {"train": load_split(ds / "Training"), "val": load_split(ds / "Validation"), "test": load_split(ds / "Testing")}
    a.out.write_text(json.dumps(out))
    print("DenPAR perio items:", {k: len(v) for k, v in out.items()})


if __name__ == "__main__":
    main()
