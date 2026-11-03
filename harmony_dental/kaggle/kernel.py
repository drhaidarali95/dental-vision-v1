"""Kaggle GPU job: the X-ray part of Harmony Dental AI, trained ONCE on all flat
X-rays together (every attached Roboflow-format dataset). Works offline: wheels and
ResNet-50 weights come from the private dataset <user>/harmony-wheels.
Outputs in /kaggle/working: xray/best.pt, history.json, report_val.json,
report_xray_test.json (all), report_test_<source>.json (per film type)."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")
os.environ.setdefault("HARMONY_LONG_SIDE", "1024")
os.environ.setdefault("HARMONY_OVERSAMPLE", "Caries,Badly decayed,Deep caries,Root canal treatment,Periapical lesion")
os.environ.setdefault("HARMONY_OVERSAMPLE_X", "3")
os.environ.setdefault("HARMONY_HOURS", "8")
os.environ.setdefault("HARMONY_REQUIRE_GPU", "1")


def sh(*cmd):
    print("$", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True)


inp = Path("/kaggle/input")
code = next(inp.rglob("prep_all_xray.py")).parent
py = sys.executable
whl = next(inp.rglob("monai-*.whl"), None)
if whl:
    sh(py, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--find-links", whl.parent, "monai", "pycocotools")
else:
    sh(py, "-m", "pip", "install", "-q", "monai==1.4.0", "pycocotools")
w = next(inp.rglob("resnet50-11ad3fa6.pth"), None)
if w:
    d = Path.home() / ".cache/torch/hub/checkpoints"; d.mkdir(parents=True, exist_ok=True); shutil.copy(w, d / w.name)
try:
    import pyarrow  # noqa: F401
except ImportError:
    sh(py, "-m", "pip", "install", "-q", "pyarrow")
DATA = Path("/kaggle/tmp/dentex")
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


try:
    fetch_dentex()
except Exception as e:  # no internet and no cached copy: train without DENTEX
    print("DENTEX not available:", e, flush=True)
# Continue from the previous run when its checkpoint is attached (same data -> same classes).
prev = next(inp.rglob("resume.pt"), None)
if prev and os.environ.get("HARMONY_CONTINUE", "1") == "1":
    (WORK / "xray").mkdir(parents=True, exist_ok=True)
    shutil.copy(prev, WORK / "xray" / "resume.pt")
    print("continuing from", prev, flush=True)
sh(py, code / "prep_extra.py", "--input", Path("/kaggle"), "--out", WORK / "data_extra.json")
sh(py, code / "prep_all_xray.py", "--input", inp, "--out", WORK / "data", "--extra", WORK / "data_extra.json")
sh(py, code / "train.py", "--datalist", WORK / "data/datalist.json", "--out", WORK / "xray", "--epochs", "60", "--batch-size", "4", "--workers", "4", "--val-every", "3")
sh(py, code / "test_report.py", "--datalist", WORK / "data/datalist.json", "--model", WORK / "xray/best.pt", "--out", WORK / "report_xray_test.json")
for t in sorted((WORK / "data").glob("test_*.json")):
    sh(py, code / "test_report.py", "--datalist", t, "--model", WORK / "xray/best.pt", "--out", WORK / f"report_{t.stem}.json")
