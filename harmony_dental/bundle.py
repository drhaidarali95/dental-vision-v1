"""Join the X-ray part and the CBCT part into ONE file: harmony-dental.pt (+ .sha256).

  python bundle.py --xray xray/best.pt --cbct cbct-seg-v2.pt --out harmony-dental.pt --version 1
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xray", required=True, type=Path)
    ap.add_argument("--cbct", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--version", default="1")
    ap.add_argument("--scores", type=Path, nargs="*", default=[], help="test report JSONs to embed")
    a = ap.parse_args()
    bundle = {"format": "harmony-dental", "version": a.version,
              "xray": torch.load(a.xray, map_location="cpu", weights_only=False),
              "cbct": torch.load(a.cbct, map_location="cpu", weights_only=False),
              "scores": {p.stem: json.loads(p.read_text()) for p in a.scores}}
    torch.save(bundle, a.out)
    h = hashlib.sha256(a.out.read_bytes()).hexdigest()
    Path(str(a.out) + ".sha256").write_text(f"{h}  {a.out.name}\n")
    print(a.out, h)


if __name__ == "__main__":
    main()
