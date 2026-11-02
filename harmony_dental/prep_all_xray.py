"""Merge every flat X-ray dataset under /kaggle/input into ONE datalist for one model.

Each Roboflow export (COCO json or CSV, with train/valid/test) is read with
periapical_monai/prep_coco.py; class names are unified (e.g. Cavity -> Caries,
RCT -> Root canal treatment, Fillings/Restoration -> Filling). Each source keeps its
own splits, so test films stay unseen. Test films are also written per source
(test_<source>.json) to report a score per film type.

  python prep_all_xray.py --input /kaggle/input --out data/
"""
import argparse
import json
from pathlib import Path

from prep_coco import SKIP

ALIASES = {
    "cavity": "Caries", "caries": "Caries", "dental caries": "Caries",
    "badly decayed": "Badly decayed", "deep caries": "Deep caries",
    "rct": "Root canal treatment", "root canal": "Root canal treatment", "root canal treatment": "Root canal treatment",
    "restoration": "Filling", "fillings": "Filling", "filling": "Filling",
    "crown": "Crown", "implant": "Implant", "impacted tooth": "Impacted tooth", "impacted": "Impacted tooth",
    "periapical lesion": "Periapical lesion",
}


def norm(name: str) -> str | None:
    n = name.strip()
    if n.lower() in SKIP:
        return None
    if len(n) == 2 and n.isdigit():
        return n  # FDI tooth number
    return ALIASES.get(n.lower(), n[:1].upper() + n[1:])


def read_split(split_dir: Path) -> list[dict]:
    items = {}
    coco = split_dir / "_annotations.coco.json"
    if coco.exists():
        c = json.loads(coco.read_text())
        names = {k["id"]: k["name"] for k in c["categories"]}
        for im in c["images"]:
            p = split_dir / im["file_name"]
            if p.exists():
                items[im["id"]] = {"image": str(p.resolve()), "box": [], "name": []}
        for a in c["annotations"]:
            it = items.get(a["image_id"])
            n = norm(names.get(a["category_id"], ""))
            x, y, w, h = a["bbox"]
            if it is not None and n and w > 1 and h > 1:
                it["box"].append([x, y, x + w, y + h]); it["name"].append(n)
    elif (split_dir / "_annotations.csv").exists():
        import csv
        with open(split_dir / "_annotations.csv") as f:
            for r in csv.DictReader(f):
                p = split_dir / r["filename"]
                if not p.exists():
                    continue
                it = items.setdefault(r["filename"], {"image": str(p.resolve()), "box": [], "name": []})
                n = norm(r["class"])
                b = [float(r[k]) for k in ("xmin", "ymin", "xmax", "ymax")]
                if n and b[2] - b[0] > 1 and b[3] - b[1] > 1:
                    it["box"].append(b); it["name"].append(n)
    return list(items.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("/kaggle/input"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--extra", type=Path, help="JSON from prep_extra.py (DENTEX, PAI panoramics)")
    a = ap.parse_args()
    roots = sorted({p.parent.parent for p in a.input.rglob("_annotations.*") if p.parent.name == "train"})
    splits = {"train": [], "val": [], "test": []}
    per_source = {}
    for root in roots:
        src = root.name if root.name not in ("", ".") else root.parent.name
        for s, d in (("train", "train"), ("val", "valid"), ("test", "test")):
            if (root / d).exists():
                its = read_split(root / d)
                for it in its:
                    it["source"] = src
                splits[s] += its
                if s == "test":
                    per_source[src] = its
        print(f"{src}: " + ", ".join(f"{s} {sum(1 for i in splits[s] if i['source'] == src)}" for s in splits), flush=True)
    if a.extra and a.extra.exists():
        ex = json.loads(a.extra.read_text())
        for s in splits:
            splits[s] += ex.get(s, [])
        for it in ex.get("test", []):
            per_source.setdefault(it["source"], []).append(it)
    classes = sorted({n for s in splits.values() for it in s for n in it["name"]})
    def conv(its):
        return [{"image": i["image"], "box": i["box"], "label": [classes.index(n) for n in i["name"]], "source": i["source"]} for i in its]
    a.out.mkdir(parents=True, exist_ok=True)
    out = {"classes": classes, "box_mode": "xyxy_pixels", "train": conv(splits["train"]), "val": conv(splits["val"]),
           "test": conv(splits["test"]), "unlabeled": []}
    (a.out / "datalist.json").write_text(json.dumps(out))
    for src, its in per_source.items():
        (a.out / f"test_{src}.json").write_text(json.dumps({**out, "test": conv(its)}))
    print(len(classes), "classes:", classes, flush=True)


if __name__ == "__main__":
    main()
