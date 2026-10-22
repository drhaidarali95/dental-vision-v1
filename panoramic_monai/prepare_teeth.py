"""Tooth-numbering datalist from DENTEX (FDI 11-48), for the anatomy layer.

  python prepare_teeth.py \
    --source data/raw/training_data/quadrant_enumeration/train_quadrant_enumeration.json:data/raw/training_data/quadrant_enumeration/xrays \
    --shrink 1024 --out data/teeth.json

Every annotation carries the quadrant (category_id_1) and the tooth position
(category_id_2); together they give the FDI number.

Use the quadrant_enumeration set. The disease set labels only the teeth that
have a finding (median 5 per film, vs 29), so training on it would teach the
model that healthy teeth are background.
Films in the findings model's test split are held out here as well, so the
two models can be scored together on films neither has seen.
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

from labels import TEETH, TOOTH_INDEX
from prepare_dentex import shrink, stratified_take


def load(json_path: Path, images_dir: Path) -> list[dict]:
    coco = json.loads(json_path.read_text())
    quad = {c["id"]: c["name"] for c in coco["categories_1"]}
    pos = {c["id"]: c["name"] for c in coco["categories_2"]}
    items = {}
    for im in coco["images"]:
        path = images_dir / im["file_name"]
        if path.exists():
            items[im["id"]] = {"image": str(path.resolve()), "box": [], "label": []}
    for ann in coco["annotations"]:
        item = items.get(ann["image_id"])
        fdi = f"{quad.get(ann.get('category_id_1'))}{pos.get(ann.get('category_id_2'))}"
        if item is None or fdi not in TOOTH_INDEX:
            continue
        x, y, w, h = ann["bbox"]
        if w > 1 and h > 1:
            item["box"].append([x, y, x + w, y + h])
            item["label"].append(TOOTH_INDEX[fdi])
    return [it for it in items.values() if it["box"]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", action="append", required=True, help="annotations.json:images_dir")
    p.add_argument("--exclude-test-of", type=Path, help="findings datalist whose test films stay out")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--test-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=20261021)
    p.add_argument("--shrink", type=int, default=0)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    # DENTEX reuses file names across folders (train_0.png in each), so films
    # are identified by full path, never by name alone.
    items, seen, disease_paths = [], set(), set()
    for spec in args.source:
        js, imgs = spec.split(":", 1)
        is_disease = "categories_3" in Path(js).read_text()
        for it in load(Path(js), Path(imgs)):
            if it["image"] not in seen:
                seen.add(it["image"])
                items.append(it)
                if is_disease:
                    disease_paths.add(it["image"])

    held = []
    if args.exclude_test_of:
        test = json.loads(args.exclude_test_of.read_text())["test"]
        exact = {t["source_image"] for t in test if "source_image" in t}
        # Older findings datalists hold only the shrunk copy's name: match those
        # by name, and only against the disease folder they came from.
        by_name = {Path(t["image"]).stem for t in test if "source_image" not in t}

        def in_findings_test(it):
            return it["image"] in exact or (it["image"] in disease_paths and Path(it["image"]).stem in by_name)

        held = [it for it in items if in_findings_test(it)]
        items = [it for it in items if not in_findings_test(it)]

    rng = random.Random(args.seed)
    test, rest = stratified_take(items, args.test_fraction, rng)
    val, train = stratified_take(rest, args.val_fraction / (1 - args.test_fraction), rng)
    test += held
    if args.shrink:
        folder = args.out.parent / f"teeth_films_{args.shrink}"
        train, val, test = (shrink(x, args.shrink, folder) for x in (train, val, test))

    out = {"classes": TEETH, "task": "teeth", "box_mode": "xyxy_pixels", "seed": args.seed,
           "train": train, "val": val, "test": test, "unlabeled": []}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))
    for split in ("train", "val", "test"):
        n = Counter(l for it in out[split] for l in it["label"])
        print(f"{split}: {len(out[split])} films, {sum(n.values())} teeth, {len(n)}/32 tooth numbers present")


if __name__ == "__main__":
    main()
