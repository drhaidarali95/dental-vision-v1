"""Score a trained checkpoint on the datalist's held-out test split -> JSON report."""
import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "panoramic_monai")]  # shared detector code
from model import load_detector  # noqa: E402
from train import evaluate  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--datalist", required=True, type=Path)
p.add_argument("--model", required=True)
p.add_argument("--out", required=True, type=Path)
a = p.parse_args()
data = json.loads(a.datalist.read_text())
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
det, ckpt = load_detector(a.model, dev, expect=None)
metrics, _ = evaluate(det, data["test"], dev)
report = {"split": "test", "films": len(data["test"]), "classes": data["classes"], **metrics,
          "note": "Held-out test split of the source export. Not an external or clinical validation."}
a.out.write_text(json.dumps(report, indent=1))
print(json.dumps({k: report[k] for k in report if k != "per_class"}))
