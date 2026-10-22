"""Modal jobs: download DENTEX, train, make draft labels, retrain, score.

One-time setup (on your own computer, in this panoramic_monai/ folder):

  pip install modal
  modal setup
  modal volume create harmony-monai
  # optional baseline (dental-vision-v1 Faster R-CNN), used for drafts + comparison:
  modal volume put harmony-monai harmony_detector_best.pth /baseline/harmony_detector_best.pth

Run everything (about 10-14 GPU hours on an L4; check Modal's pricing page):

  modal run --detach modal_train.py

Or one step at a time:

  modal run modal_train.py::prepare
  modal run modal_train.py::train --stage teacher
  modal run modal_train.py::drafts
  modal run modal_train.py::train --stage student
  modal run modal_train.py::evaluate

Then fetch the results:

  modal volume get harmony-monai /runs/student/best.pt .
  modal volume get harmony-monai /runs/report_test.json .
"""
import subprocess
from pathlib import Path

import modal

VOL = Path("/vol")
CODE = "/root/ml-monai"
BASELINE = VOL / "baseline" / "harmony_detector_best.pth"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("unzip")
    .pip_install_from_requirements(str(Path(__file__).with_name("requirements.txt")))
    .add_local_dir(str(Path(__file__).parent), CODE, ignore=["data", "runs", "*.pt", "*.pth", "__pycache__"])
)
volume = modal.Volume.from_name("harmony-monai", create_if_missing=True)
app = modal.App("harmony-monai-train", image=image)


def run(*cmd: str) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=CODE)


def baseline_args() -> list[str]:
    if BASELINE.exists():
        return ["--baseline", str(BASELINE)]
    print(f"No baseline at {BASELINE}; continuing without it.")
    return []


@app.function(volumes={VOL: volume}, timeout=6 * 3600, cpu=4, memory=16384)
def prepare():
    """Download DENTEX from Zenodo record 7812323 and build data/datalist.json."""
    import hashlib
    import json
    import urllib.request

    raw = VOL / "dentex"
    raw.mkdir(parents=True, exist_ok=True)
    meta = json.load(urllib.request.urlopen("https://zenodo.org/api/records/7812323"))
    print("License per Zenodo:", meta.get("metadata", {}).get("license"))
    files = {f["key"]: f for f in meta["files"]}
    zip_path = raw / "training_data.zip"
    if not zip_path.exists():
        item = files["training_data.zip"]
        url = item["links"].get("content") or item["links"]["self"]
        md5 = hashlib.md5()
        tmp = zip_path.with_suffix(".part")
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            while chunk := r.read(8 * 1024 * 1024):
                f.write(chunk)
                md5.update(chunk)
        want = str(item.get("checksum", "")).removeprefix("md5:")
        if want and md5.hexdigest() != want:
            tmp.unlink()
            raise RuntimeError(f"MD5 mismatch: {md5.hexdigest()} != {want}")
        tmp.rename(zip_path)
    if not (raw / "training_data").exists():
        subprocess.run(["unzip", "-q", "-o", str(zip_path), "-d", str(raw)], check=True)

    disease_json = next(p for p in raw.rglob("*.json")
                        if "categories_3" in p.read_text(errors="ignore"))
    images = next(d for d in disease_json.parent.rglob("*") if d.is_dir() and d.name.lower() in ("xrays", "images"))
    unlabeled = next((d for d in raw.rglob("*") if d.is_dir() and "unlabel" in d.name.lower()), None)
    cmd = ["python", "prepare_dentex.py", "--labeled-json", str(disease_json), "--labeled-images", str(images),
           "--out", str(VOL / "data" / "datalist.json")]
    if unlabeled:
        cmd += ["--unlabeled-images", str(unlabeled)]
    run(*cmd)
    volume.commit()


@app.function(volumes={VOL: volume}, gpu="L4", timeout=24 * 3600, cpu=8, memory=32768)
def train(stage: str = "teacher", epochs: int = 60):
    if stage not in ("teacher", "student"):
        raise ValueError("stage is 'teacher' or 'student'")
    cmd = ["python", "train.py", "--datalist", str(VOL / "data" / "datalist.json"),
           "--out", str(VOL / "runs" / stage), "--epochs", str(epochs)]
    if stage == "student":
        cmd += ["--drafts", str(VOL / "runs" / "drafts.json")]
    run(*cmd)
    volume.commit()


@app.function(volumes={VOL: volume}, gpu="L4", timeout=12 * 3600, cpu=4, memory=16384)
def drafts():
    run("python", "make_drafts.py", "--datalist", str(VOL / "data" / "datalist.json"),
        "--teacher", str(VOL / "runs" / "teacher" / "best.pt"), "--out", str(VOL / "runs" / "drafts.json"),
        *baseline_args())
    volume.commit()


@app.function(volumes={VOL: volume}, gpu="L4", timeout=3 * 3600, cpu=4, memory=16384)
def evaluate(stage: str = "student"):
    run("python", "evaluate.py", "--datalist", str(VOL / "data" / "datalist.json"),
        "--model", str(VOL / "runs" / stage / "best.pt"), "--out", str(VOL / "runs" / "report_test.json"),
        *baseline_args())
    volume.commit()


@app.local_entrypoint()
def main():
    prepare.remote()
    train.remote(stage="teacher")
    drafts.remote()
    train.remote(stage="student")
    evaluate.remote()
    print("Done. modal volume get harmony-monai /runs/student/best.pt /runs/report_test.json .")
