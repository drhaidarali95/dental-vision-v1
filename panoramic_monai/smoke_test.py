"""End-to-end smoke test on fake films (CPU, about a minute, no downloads).

  python smoke_test.py

Checks the plumbing only: data prep, one training epoch, draft labels,
evaluation (with a stand-in Faster R-CNN baseline), a tooth model, and the
server's /health, /predict, /anatomy, /annotate layers and non-panoramic refusal.
The fake model learns nothing useful.
"""
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).parent


def fake_film(path: Path, w: int, h: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    arr = (rng.random((h, w)) * 60 + 80).astype(np.uint8)
    x, y = int(w * 0.4), int(h * 0.4)
    arr[y:y + 60, x:x + 50] = 230
    Image.fromarray(arr).save(path)
    return [x, y, 50, 60]


def main():
    tmp = Path(tempfile.mkdtemp())
    labeled, unlabeled = tmp / "labeled", tmp / "unlabeled"
    labeled.mkdir()
    unlabeled.mkdir()
    coco = {"images": [], "annotations": [],
            "categories_3": [{"id": 0, "name": "Impacted"}, {"id": 1, "name": "Caries"},
                             {"id": 2, "name": "Periapical Lesion"}, {"id": 3, "name": "Deep Caries"}]}
    for i in range(16):
        name = f"pano_{i}.png"
        bbox = fake_film(labeled / name, 1400, 640, i)
        coco["images"].append({"id": i, "file_name": name})
        coco["annotations"].append({"id": i, "image_id": i, "bbox": bbox, "category_id_3": i % 4})
    for i in range(3):
        fake_film(unlabeled / f"u_{i}.png", 1400, 640, 100 + i)
    (tmp / "train.json").write_text(json.dumps(coco))

    def run(*args):
        print("$", *args, flush=True)
        subprocess.run([sys.executable, *args], check=True, cwd=HERE)

    # Untrained stand-in for dental-vision-v1's checkpoint (serve_fasterrcnn layout).
    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    frcnn = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
    frcnn.roi_heads.box_predictor = FastRCNNPredictor(frcnn.roi_heads.box_predictor.cls_score.in_features, 5)
    baseline = tmp / "baseline.pth"
    torch.save({"model": frcnn.state_dict(), "epoch": 0,
                "classes": ["__background__", "Cavity", "Fillings", "Implant", "Impacted Tooth"]}, baseline)

    datalist, runs = tmp / "datalist.json", tmp / "runs"
    run("prepare_dentex.py", "--labeled-json", str(tmp / "train.json"), "--labeled-images", str(labeled),
        "--unlabeled-images", str(unlabeled), "--out", str(datalist))
    run("train.py", "--datalist", str(datalist), "--out", str(runs / "teacher"), "--epochs", "1",
        "--batch-size", "2", "--workers", "0", "--no-pretrained", "--val-every", "1")
    model = runs / "teacher" / "best.pt"
    run("make_drafts.py", "--datalist", str(datalist), "--teacher", str(model), "--out", str(runs / "drafts.json"),
        "--keep", "0.0", "--baseline", str(baseline))
    run("train.py", "--datalist", str(datalist), "--drafts", str(runs / "drafts.json"), "--out", str(runs / "student"),
        "--epochs", "1", "--batch-size", "2", "--workers", "0", "--no-pretrained")
    run("evaluate.py", "--datalist", str(datalist), "--model", str(runs / "student" / "best.pt"),
        "--out", str(runs / "report_test.json"), "--baseline", str(baseline))

    # Tooth model: same fake films, every box called tooth 36.
    from labels import TEETH, TOOTH_INDEX

    findings_list = json.loads(datalist.read_text())
    teeth = {"classes": TEETH, "task": "teeth", "unlabeled": [],
             **{k: [{**it, "label": [TOOTH_INDEX["36"]] * len(it["label"])} for it in findings_list[k]]
                for k in ("train", "val", "test")}}
    teeth_list = tmp / "teeth.json"
    teeth_list.write_text(json.dumps(teeth))
    run("train.py", "--datalist", str(teeth_list), "--out", str(runs / "teeth"), "--epochs", "1",
        "--batch-size", "2", "--workers", "0", "--no-pretrained", "--val-every", "1")

    os.environ.update(HARMONY_CHECKPOINT=str(model), HARMONY_KEY="k", MIN_CONF="0.0",
                      PERIAPICAL_LOG=str(tmp / "pa.jsonl"),
                      ANATOMY_CHECKPOINT=str(runs / "teeth" / "best.pt"), ANATOMY_MIN_CONF="0.0")
    sys.path.insert(0, str(HERE))
    from fastapi.testclient import TestClient
    from serve_monai import app

    client = TestClient(app)
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["anatomy"]["available"], health

    def post(img: Image.Image, key="k"):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return client.post("/predict", headers={"Authorization": f"Bearer {key}"},
                           files={"file": ("x.png", buf.getvalue(), "image/png")})

    pano = Image.open(labeled / "pano_0.png")
    assert post(pano, key="wrong").status_code == 401
    r = post(pano)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["image"] == {"width": 1400, "height": 640}
    assert all(f["category"] in ("cavity", "impacted tooth") for f in body["findings"]), body
    for f in body["findings"]:
        b = f["bbox"]
        assert -1 <= b["x1"] < b["x2"] <= 1401 and -1 <= b["y1"] < b["y2"] <= 641, f"box outside film: {b}"
    assert all(f["tooth"] is None or 11 <= f["tooth"] <= 48 for f in body["findings"]), body
    buf = io.BytesIO()
    pano.save(buf, format="PNG")
    for layer in ("findings", "anatomy", "both"):
        marked = client.post(f"/annotate?layer={layer}", headers={"Authorization": "Bearer k"},
                             files={"file": ("x.png", buf.getvalue(), "image/png")})
        assert marked.status_code == 200 and marked.headers["content-type"] == "image/png", layer
        assert Image.open(io.BytesIO(marked.content)).size == (1400, 640)
    anat = client.post("/anatomy", headers={"Authorization": "Bearer k"},
                       files={"file": ("x.png", buf.getvalue(), "image/png")})
    assert anat.status_code == 200, anat.text
    teeth_out = anat.json()["anatomy"]
    assert teeth_out and all(t["structure"] == "tooth" and 11 <= t["tooth"] <= 48 and t["name"] for t in teeth_out)
    assert len({t["tooth"] for t in teeth_out}) == len(teeth_out), "a tooth number was used twice"
    bitewing = Image.fromarray(np.full((900, 1200), 120, np.uint8))
    assert post(bitewing).status_code == 422
    photo = Image.new("RGB", (1600, 700), (200, 80, 60))
    assert post(photo).status_code == 422
    print(f"OK: {len(body['findings'])} wire findings, {len(teeth_out)} teeth; non-panoramic films refused with 422")


if __name__ == "__main__":
    main()
