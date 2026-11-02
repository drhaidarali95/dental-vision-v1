"""Extra X-ray sources that are not Roboflow exports, as items for prep_all_xray.py.

- DENTEX panoramics (Zenodo 7812323, CC BY 4.0): disease boxes (caries, deep caries,
  periapical lesion, impacted) and FDI tooth numbers. Found anywhere under --input
  (e.g. the attached drhaidarali/dentex-data notebook output).
- Periapical lesions on panoramics (HF tirandazdylan/periapical-lesions-pai, CC BY 4.0,
  3,863 films, PAI 3-5): downloaded as parquet when internet is on, or found under --input.

Each source keeps a fixed, seeded split; output: one JSON {"train": [...], "val": [...],
"test": [...]} with items {image, box, name, source}.
"""
import argparse
import io
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labels import TEETH, CLASSES  # noqa: E402
from prepare_dentex import load_coco  # noqa: E402
from prepare_teeth import load as load_teeth  # noqa: E402

NAMES = {"caries": "Caries", "deep_caries": "Deep caries", "periapical_lesion": "Periapical lesion", "impacted": "Impacted tooth"}
PAI_REPO = "https://huggingface.co/datasets/tirandazdylan/periapical-lesions-pai/resolve/main/data/"
PAI_FILES = {"train": ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"],
             "val": ["validation-00000-of-00001.parquet"], "test": ["test-00000-of-00001.parquet"]}


def split(items, seed, val=0.12, test=0.15):
    items = sorted(items, key=lambda i: i["image"])
    random.Random(seed).shuffle(items)
    n = len(items); nt, nv = int(n * test), int(n * val)
    return {"test": items[:nt], "val": items[nt:nt + nv], "train": items[nt + nv:]}


def dentex(inp: Path):
    out = {"train": [], "val": [], "test": []}
    dis = next(inp.rglob("train_quadrant_enumeration_disease.json"), None)
    if dis:
        its = load_coco(dis, dis.parent / "xrays")
        its = [{"image": i["image"], "box": i["box"], "name": [NAMES[CLASSES[l]] for l in i["label"]], "source": "dentex-findings"} for i in its]
        for k, v in split(its, 20261018).items():
            out[k] += v
        print("DENTEX findings:", len(its), flush=True)
    enum = next(inp.rglob("train_quadrant_enumeration.json"), None)
    if enum:
        its = load_teeth(enum, enum.parent / "xrays")
        its = [{"image": i["image"], "box": i["box"], "name": [TEETH[l] for l in i["label"]], "source": "dentex-teeth"} for i in its]
        for k, v in split(its, 20261021).items():
            out[k] += v
        print("DENTEX teeth:", len(its), flush=True)
    return out


def pai(inp: Path, work: Path):
    import pyarrow.parquet as pq
    out = {"train": [], "val": [], "test": []}
    imgdir = work / "pai_images"; imgdir.mkdir(parents=True, exist_ok=True)
    for s, files in PAI_FILES.items():
        for f in files:
            p = next(inp.rglob(f), None)
            if p is None:
                try:
                    import urllib.request
                    p = work / f; urllib.request.urlretrieve(PAI_REPO + f, p)
                except Exception as e:  # offline
                    print("PAI not available:", e, flush=True); return out
            for r in pq.read_table(p).to_pylist():
                ip = imgdir / f"{r['image_id']}.png"
                if not ip.exists():
                    ip.write_bytes(r["image"]["bytes"])
                boxes = [[x, y, x + w, y + h] for x, y, w, h in r["objects"]["bbox"] if w > 1 and h > 1]
                out[s].append({"image": str(ip), "box": boxes, "name": ["Periapical lesion"] * len(boxes), "source": "pai-panoramic"})
    print("PAI:", {k: len(v) for k, v in out.items()}, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("/kaggle/input"))
    ap.add_argument("--work", type=Path, default=Path("/kaggle/tmp/extra"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    total = {"train": [], "val": [], "test": []}
    for part in (dentex(a.input), pai(a.input, a.work)):
        for k in total:
            total[k] += part[k]
    a.out.write_text(json.dumps(total))
    print("extra:", {k: len(v) for k, v in total.items()}, flush=True)


if __name__ == "__main__":
    main()
