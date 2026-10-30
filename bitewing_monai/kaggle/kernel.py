"""Kaggle GPU job: bitewing tooth-numbering model (FDI 13-48) on the Roboflow
bitewing export mirrored on Kaggle
(sprss96/bitewing-dataset, CC BY 4.0). Code comes from the private
dataset <user>/harmony-periapical-code (panoramic_monai + periapical_monai).

Outputs in /kaggle/working: bitewing/best.pt, history.json, report_val.json,
report_bitewing_test.json, run.log."""
import os
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")
EPOCHS = os.environ.get("BW_EPOCHS", "30")
os.environ["HARMONY_LONG_SIDE"] = os.environ.get("HARMONY_LONG_SIDE", "640")


def sh(*cmd):
    print("$", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True)


code = next(Path("/kaggle/input").rglob("prep_coco.py")).parent
data = next(p for p in Path("/kaggle/input").rglob("_annotations.coco.json") if p.parent.name == "train").parent.parent
py = sys.executable
sh(py, "-m", "pip", "install", "-q", "monai==1.4.0", "pycocotools")
sh(py, code / "prep_coco.py", "--root", data, "--out", WORK / "data/datalist.json")
sh(py, code / "train.py", "--datalist", WORK / "data/datalist.json", "--out", WORK / "bitewing",
   "--epochs", EPOCHS, "--batch-size", "8")
sh(py, code / "test_report.py", "--datalist", WORK / "data/datalist.json",
   "--model", WORK / "bitewing/best.pt", "--out", WORK / "report_bitewing_test.json")
