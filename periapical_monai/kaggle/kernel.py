"""Kaggle GPU job: periapical findings model (caries, badly decayed, RCT, crown,
restoration) on the Roboflow periapical export mirrored on Kaggle
(nadaaglan/dental-periapical-x-rays, CC BY 4.0). Code comes from the private
dataset <user>/harmony-periapical-code (panoramic_monai + periapical_monai).

Outputs in /kaggle/working: periapical/best.pt, history.json, report_val.json,
report_periapical_test.json, run.log."""
import os
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")
EPOCHS = os.environ.get("PA_EPOCHS", "60")
os.environ["HARMONY_LONG_SIDE"] = os.environ.get("HARMONY_LONG_SIDE", "640")


def sh(*cmd):
    print("$", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True)


code = next(Path("/kaggle/input").rglob("prep_coco.py")).parent
data = next(p for p in Path("/kaggle/input").rglob("_annotations.*") if p.parent.name == "train").parent.parent
py = sys.executable
sh(py, "-m", "pip", "install", "-q", "monai==1.4.0", "pycocotools")
sh(py, code / "prep_coco.py", "--root", data, "--out", WORK / "data/datalist.json")
sh(py, code / "train.py", "--datalist", WORK / "data/datalist.json", "--out", WORK / "periapical",
   "--epochs", EPOCHS, "--batch-size", "8")
sh(py, code / "test_report.py", "--datalist", WORK / "data/datalist.json",
   "--model", WORK / "periapical/best.pt", "--out", WORK / "report_periapical_test.json")
