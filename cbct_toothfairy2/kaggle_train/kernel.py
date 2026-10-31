"""Kaggle GPU job: CBCT segmentation v1 on the owner's private ToothFairy2
copy (drhaidarali/toothfairy2, CC BY-SA 4.0). Code: <user>/harmony-cbct-code.
Outputs in /kaggle/working: cbct/best.pt, cbct/history.json, report_cbct_test.json."""
import subprocess
import sys
from pathlib import Path

code = next(Path("/kaggle/input").rglob("train_cbct.py")).parent
data = next(Path("/kaggle/input").rglob("dataset.json"), None)
if data is None or not (data.parent / "labelsTr").exists():
    sys.exit("ToothFairy2 data (dataset.json + labelsTr) not found under /kaggle/input")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "monai==1.4.0", "SimpleITK"], check=True)
subprocess.run([sys.executable, str(code / "train_cbct.py"), "--data", str(data.parent),
                "--work", "/kaggle/tmp/cbct", "--out", "/kaggle/working", "--hours", "7.5"], check=True)
