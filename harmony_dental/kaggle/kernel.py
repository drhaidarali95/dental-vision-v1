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
os.environ.setdefault("HARMONY_HOURS", "10.5")


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
sh(py, code / "prep_all_xray.py", "--input", inp, "--out", WORK / "data")
sh(py, code / "train.py", "--datalist", WORK / "data/datalist.json", "--out", WORK / "xray", "--epochs", "60", "--batch-size", "4")
sh(py, code / "test_report.py", "--datalist", WORK / "data/datalist.json", "--model", WORK / "xray/best.pt", "--out", WORK / "report_xray_test.json")
for t in sorted((WORK / "data").glob("test_*.json")):
    sh(py, code / "test_report.py", "--datalist", t, "--model", WORK / "xray/best.pt", "--out", WORK / f"report_{t.stem}.json")
