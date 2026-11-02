"""Kaggle GPU job: the PHOTO part of Harmony Dental AI (clinical colour photos, caries graded
ICDAS D0-D6). Data: private <user>/harmony-photo-caries (HF Frallex/Caries_Bounding_Box, MIT).
Same detector and training code as the X-ray part; works offline with the wheels/ResNet datasets.
Outputs: photo/best.pt, report_photo_test.json."""
import os, shutil, subprocess, sys
from pathlib import Path

WORK = Path("/kaggle/working")
os.environ.setdefault("HARMONY_LONG_SIDE", "640")
os.environ.setdefault("HARMONY_HOURS", "6")


def sh(*cmd):
    print("$", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True)


inp = Path("/kaggle/input")
code = next(inp.rglob("prep_all_xray.py")).parent
py = sys.executable
whl = next(inp.rglob("monai-*.whl"), None)
sh(py, "-m", "pip", "install", "-q", *(["--no-index", "--no-deps", "--find-links", whl.parent] if whl else []), "monai" if whl else "monai==1.4.0", "pycocotools")
w = next(inp.rglob("resnet50-11ad3fa6.pth"), None)
if w:
    d = Path.home() / ".cache/torch/hub/checkpoints"; d.mkdir(parents=True, exist_ok=True); shutil.copy(w, d / w.name)
photos = next(p for p in inp.iterdir() if "photo" in p.name.lower()) if any("photo" in p.name.lower() for p in inp.iterdir()) else next(inp.rglob("harmony-photo-caries"))
sh(py, code / "prep_all_xray.py", "--input", photos, "--out", WORK / "data")
sh(py, code / "train.py", "--datalist", WORK / "data/datalist.json", "--out", WORK / "photo", "--epochs", "80", "--batch-size", "8")
sh(py, code / "test_report.py", "--datalist", WORK / "data/datalist.json", "--model", WORK / "photo/best.pt", "--out", WORK / "report_photo_test.json")
