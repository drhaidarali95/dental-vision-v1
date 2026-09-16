#!/usr/bin/env python3
"""Inspect extracted DENTEX tree and locate COCO-like annotation JSON files."""
import argparse, json, pathlib

p = argparse.ArgumentParser(); p.add_argument("root", nargs="?", default="data")
a = p.parse_args(); root = pathlib.Path(a.root)
for path in root.rglob("*.json"):
    try:
        d = json.loads(path.read_text())
    except Exception:
        continue
    if isinstance(d, dict) and {"images", "annotations", "categories"}.issubset(d):
        print(path)
        print(" images:", len(d["images"]), "annotations:", len(d["annotations"]))
        print(" categories:", [(c.get("id"), c.get("name")) for c in d["categories"]])
