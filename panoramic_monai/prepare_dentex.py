"""Turn DENTEX into one datalist JSON for the MONAI detector.

  python prepare_dentex.py \
    --labeled-json  data/raw/training_data/quadrant-enumeration-disease/train_quadrant_enumeration_disease.json \
    --labeled-images data/raw/training_data/quadrant-enumeration-disease/xrays \
    --unlabeled-images data/raw/training_data/unlabelled/xrays \
    --out data/datalist.json

Reads either the official DENTEX layout (categories_3 / category_id_3 hold the
diagnosis) or a plain COCO file. Boxes are written as pixel [x1, y1, x2, y2].

Split: train / val / test by image, stratified so every class appears in val
and test. DENTEX carries no patient ids, so this is an image-level split; pass
--test-json (e.g. the official validation labels) for a cleaner test set.
"""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image

from labels import CLASS_INDEX, CLASSES, canonical

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def load_coco(json_path: Path, images_dir: Path) -> list[dict]:
    coco = json.loads(json_path.read_text())
    if "categories_3" in coco:
        cats, key = coco["categories_3"], "category_id_3"
    else:
        cats, key = coco["categories"], "category_id"
    cat_to_class = {c["id"]: canonical(c["name"]) for c in cats}
    unknown = [c["name"] for c in cats if cat_to_class[c["id"]] is None]
    if unknown:
        print(f"ignoring non-diagnosis categories: {unknown}")

    items = {}
    for im in coco["images"]:
        path = images_dir / im["file_name"]
        if not path.exists():
            print(f"missing image, skipped: {path}")
            continue
        items[im["id"]] = {"image": str(path.resolve()), "box": [], "label": []}
    for ann in coco["annotations"]:
        item = items.get(ann["image_id"])
        name = cat_to_class.get(ann.get(key))
        if item is None or name is None:
            continue
        x, y, w, h = ann["bbox"]
        if w <= 1 or h <= 1:
            continue
        item["box"].append([x, y, x + w, y + h])
        item["label"].append(CLASS_INDEX[name])
    return list(items.values())


def shrink(items: list[dict], long_side: int, folder: Path) -> list[dict]:
    """Save smaller copies (long side = long_side px) and scale boxes to match.

    Makes CPU training practical: full panoramics are ~3000 px wide.
    """
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for it in items:
        src = Path(it["image"])
        # DENTEX reuses file names across its folders (train_0.png in each),
        # so the folder is part of the copy's name.
        tag = hashlib.md5(str(src.parent.resolve()).encode()).hexdigest()[:6]
        dst = folder / f"{tag}_{src.stem}.png"
        with Image.open(src) as img:
            scale = min(1.0, long_side / max(img.size))
            if not dst.exists():
                img.convert("L").resize((round(img.width * scale), round(img.height * scale)), Image.BILINEAR).save(dst)
        new = {**it, "image": str(dst.resolve()), "source_image": it.get("source_image", it["image"])}
        if "box" in it:
            new["box"] = [[v * scale for v in b] for b in it["box"]]
        out.append(new)
    return out


def stratified_take(items: list[dict], fraction: float, rng: random.Random) -> tuple[list, list]:
    """Greedy pick so rare classes reach their share first. Returns (taken, rest)."""
    want = max(1, round(len(items) * fraction))
    totals = Counter(l for it in items for l in set(it["label"]))
    target = {c: max(1, round(totals[c] * fraction)) for c in totals}
    rest = items[:]
    rng.shuffle(rest)
    taken, have = [], Counter()

    def gain(it):
        return sum(max(target[c] - have[c], 0) / target[c] for c in set(it["label"]))

    while rest and len(taken) < want:
        best = max(rest, key=lambda it: (gain(it), rng.random()))
        rest.remove(best)
        taken.append(best)
        have.update(set(best["label"]))
    return taken, rest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labeled-json", required=True, type=Path)
    p.add_argument("--labeled-images", required=True, type=Path)
    p.add_argument("--test-json", type=Path)
    p.add_argument("--test-images", type=Path)
    p.add_argument("--unlabeled-images", type=Path)
    p.add_argument("--val-fraction", type=float, default=0.12)
    p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=20261018)
    p.add_argument("--shrink", type=int, default=0, help="save copies at this long side (e.g. 1024 for CPU)")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    rng = random.Random(args.seed)
    labeled = load_coco(args.labeled_json, args.labeled_images)
    if args.test_json:
        test = load_coco(args.test_json, args.test_images or args.test_json.parent)
        pool = labeled
    else:
        test, pool = stratified_take(labeled, args.test_fraction, rng)
    val, train = stratified_take(pool, args.val_fraction / max(1e-6, 1 - args.test_fraction), rng)

    unlabeled = []
    if args.unlabeled_images:
        used = {it["image"] for it in labeled + test}
        for path in sorted(args.unlabeled_images.rglob("*")):
            if path.suffix.lower() in IMAGE_EXT and str(path.resolve()) not in used:
                unlabeled.append({"image": str(path.resolve())})

    if args.shrink:
        folder = args.out.parent / f"films_{args.shrink}"
        train, val, test, unlabeled = (shrink(x, args.shrink, folder) for x in (train, val, test, unlabeled))

    out = {"classes": CLASSES, "box_mode": "xyxy_pixels", "seed": args.seed,
           "train": train, "val": val, "test": test, "unlabeled": unlabeled}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))

    for split in ("train", "val", "test"):
        counts = Counter(CLASSES[l] for it in out[split] for l in it["label"])
        print(f"{split}: {len(out[split])} images, boxes {dict(counts)}")
        missing = [c for c in CLASSES if counts[c] == 0]
        if split != "train" and missing:
            print(f"  WARNING: {split} has no {missing}; its scores for those classes mean nothing")
    print(f"unlabeled: {len(unlabeled)} images")


if __name__ == "__main__":
    main()
