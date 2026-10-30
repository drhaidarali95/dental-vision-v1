"""Roboflow/COCO export (train/valid/test folders, each with _annotations.coco.json)
-> datalist.json in the format panoramic_monai/train.py reads.

  python prep_coco.py --root /kaggle/input/dental-periapical-x-rays --out data/datalist.json

The export's own splits are kept (Roboflow augments only its train split, so
valid/test stay original films). "Normal" boxes (healthy teeth) and generic parent labels ("object", "Tooth")
are dropped.
"""
import argparse
import json
from pathlib import Path

SKIP = {"normal", "object", "tooth"}  # healthy-tooth marks and generic parent labels


def load(split_dir: Path, classes: list[str]) -> list[dict]:
    coco = json.loads((split_dir / "_annotations.coco.json").read_text())
    names = {c["id"]: c["name"] for c in coco["categories"]}
    items = {im["id"]: {"image": str((split_dir / im["file_name"]).resolve()), "box": [], "label": []}
             for im in coco["images"] if (split_dir / im["file_name"]).exists()}
    for a in coco["annotations"]:
        it, name = items.get(a["image_id"]), names.get(a["category_id"], "")
        if it is None or name.lower() in SKIP or name not in classes:
            continue
        x, y, w, h = a["bbox"]
        if w > 1 and h > 1:
            it["box"].append([x, y, x + w, y + h])
            it["label"].append(classes.index(name))
    return list(items.values())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    root = next(a.root.rglob("_annotations.coco.json")).parent.parent
    cats = json.loads((root / "train/_annotations.coco.json").read_text())["categories"]
    classes = sorted({c["name"] for c in cats if c["name"].lower() not in SKIP and c.get("supercategory") != "none"})
    if not classes:
        classes = sorted({c["name"] for c in cats if c["name"].lower() not in SKIP})
    out = {"classes": classes, "box_mode": "xyxy_pixels",
           "train": load(root / "train", classes), "val": load(root / "valid", classes),
           "test": load(root / "test", classes), "unlabeled": []}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out))
    for s in ("train", "val", "test"):
        print(s, len(out[s]), "films,", sum(len(i["label"]) for i in out[s]), "boxes")
    print("classes", classes)


if __name__ == "__main__":
    main()
