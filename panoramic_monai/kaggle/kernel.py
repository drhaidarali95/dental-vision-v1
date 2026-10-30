"""Kaggle GPU job: panoramic findings model + tooth-numbering model, end to end.

Runs inside a Kaggle notebook (GPU, internet on). The ml-monai code arrives as
the Kaggle dataset <user>/harmony-ml-monai-code (see push_kaggle.sh). DENTEX
is downloaded from Zenodo (record 7812323, CC BY 4.0) as one resumable file and
only the two labelled folders are kept.

Outputs in /kaggle/working:
  findings/best.pt, findings/report_val.json, report_test.json
  teeth/best.pt,    teeth/report_val.json,    report_teeth_test.json
  run.log
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")
CODE = Path("/kaggle/tmp/ml-monai")
DATA = Path("/kaggle/tmp/dentex")
LONG_SIDE = os.environ.get("HARMONY_LONG_SIDE", "1536")
FINDINGS_EPOCHS = os.environ.get("FINDINGS_EPOCHS", "40")
TEETH_EPOCHS = os.environ.get("TEETH_EPOCHS", "30")


def sh(*cmd, cwd=None):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=cwd)


def code_dir() -> Path:
    found = next(Path("/kaggle/input").rglob("train.py"), None)
    if not found:
        raise SystemExit("ml-monai code dataset is not attached to this notebook")
    return found.parent


def fetch_dentex():
    # One sequential download (Zenodo rate-limits many parallel range requests),
    # resumed with Range headers and backing off on 429/5xx, then extract the two folders.
    import time
    import zipfile

    import requests

    url = "https://zenodo.org/records/7812323/files/training_data.zip?download=1"
    prefixes = ("training_data/quadrant-enumeration-disease/", "training_data/quadrant_enumeration/")
    if all((DATA / p).exists() for p in prefixes):
        print("DENTEX already present", flush=True)
        return
    # Prefer the private cached copy (notebook drhaidarali/dentex-data) when attached.
    cached = next(Path("/kaggle/input").rglob("quadrant-enumeration-disease"), None)
    if cached is not None:
        DATA.mkdir(parents=True, exist_ok=True)
        if (DATA / "training_data").exists():
            shutil.rmtree(DATA / "training_data")
        (DATA / "training_data").symlink_to(cached.parent, target_is_directory=True)
        print(f"DENTEX from cached copy: {cached.parent}", flush=True)
        return
    zpath = Path("/kaggle/tmp/dentex_training_data.zip")
    zpath.parent.mkdir(parents=True, exist_ok=True)
    total = None
    for attempt in range(30):
        have = zpath.stat().st_size if zpath.exists() else 0
        if total is not None and have >= total:
            break
        try:
            headers = {"Range": f"bytes={have}-"} if have else {}
            with requests.get(url, headers=headers, stream=True, timeout=120) as r:
                if r.status_code in (429, 500, 502, 503, 504):
                    wait = int(r.headers.get("Retry-After", 60))
                    print(f"Zenodo said {r.status_code}; waiting {wait}s", flush=True)
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                if have and r.status_code != 206:
                    have = 0  # server ignored Range; start over
                size = int(r.headers.get("Content-Length", 0))
                total = have + size
                last = have
                with open(zpath, "ab" if have else "wb") as f:
                    for chunk in r.iter_content(8 << 20):
                        f.write(chunk)
                        have += len(chunk)
                        if have - last > (1 << 30):
                            last = have
                            print(f"DENTEX {have / 1e9:.1f}/{total / 1e9:.1f} GB", flush=True)
        except requests.RequestException as e:
            print(f"download interrupted ({e}); resuming", flush=True)
            time.sleep(min(300, 15 * (attempt + 1)))
    else:
        raise RuntimeError("could not download DENTEX from Zenodo")
    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist() if n.startswith(prefixes) and not n.endswith("/")]
        for n in names:
            z.extract(n, DATA)
    zpath.unlink()
    print(f"DENTEX ready: {len(names)} files", flush=True)


def main():
    sh(sys.executable, "-m", "pip", "install", "-q", "monai==1.4.0", "pycocotools", "requests")
    if CODE.exists():
        shutil.rmtree(CODE)
    shutil.copytree(code_dir(), CODE)
    fetch_dentex()
    os.environ["HARMONY_LONG_SIDE"] = LONG_SIDE
    disease = DATA / "training_data/quadrant-enumeration-disease"
    enum = DATA / "training_data/quadrant_enumeration"
    data = WORK / "data"
    py = sys.executable

    sh(py, "prepare_dentex.py", "--labeled-json", str(disease / "train_quadrant_enumeration_disease.json"),
       "--labeled-images", str(disease / "xrays"), "--shrink", LONG_SIDE, "--out", str(data / "datalist.json"),
       cwd=CODE)
    sh(py, "train.py", "--datalist", str(data / "datalist.json"), "--out", str(WORK / "findings"),
       "--epochs", FINDINGS_EPOCHS, "--batch-size", "4", "--workers", "4", "--lr", "2e-4", "--val-every", "2", cwd=CODE)
    sh(py, "evaluate.py", "--datalist", str(data / "datalist.json"), "--model", str(WORK / "findings/best.pt"),
       "--out", str(WORK / "report_test.json"), cwd=CODE)

    sh(py, "prepare_teeth.py", "--source", f"{enum / 'train_quadrant_enumeration.json'}:{enum / 'xrays'}",
       "--shrink", LONG_SIDE, "--out", str(data / "teeth.json"), cwd=CODE)
    sh(py, "train.py", "--datalist", str(data / "teeth.json"), "--out", str(WORK / "teeth"),
       "--epochs", TEETH_EPOCHS, "--batch-size", "4", "--workers", "4", "--lr", "2e-4", "--val-every", "2", cwd=CODE)
    # Score the tooth model on its own held-out films (evaluate.py works for any class list).
    sh(py, "-c", (
        "import json,torch,sys; sys.path.insert(0,'.');"
        "from model import load_detector,open_film,predict; from metrics import coco_metrics;"
        f"d=json.load(open('{data / 'teeth.json'}'));"
        f"m,ck=load_detector('{WORK / 'teeth/best.pt'}',torch.device('cuda'),expect=None);"
        "p=[predict(m,open_film(i['image']),torch.device('cuda')) for i in d['test']];"
        f"json.dump({{'films':len(d['test']),**coco_metrics(d['test'],p,m.classes)}},open('{WORK / 'report_teeth_test.json'}','w'),indent=2)"
    ), cwd=CODE)

    # Keep the output small: models and reports only.
    shutil.rmtree(data, ignore_errors=True)
    for run in ("findings", "teeth"):
        for f in ("resume.pt", "last.pt"):
            (WORK / run / f).unlink(missing_ok=True)
    print(json.dumps(json.loads((WORK / "report_test.json").read_text())["monai_4_class"], indent=2))
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
