#!/usr/bin/env python3
"""Download DENTEX files from the official Zenodo record.

Usage:
  python scripts/download_dentex.py --record 7812323 --out data/raw

The script queries Zenodo record metadata, prints the record license and files,
and downloads selected archives with streaming + MD5 verification when available.
"""
import argparse, hashlib, json, pathlib, urllib.request


def get_json(url):
    with urllib.request.urlopen(url) as r:
        return json.load(r)


def download(url, dest, expected_md5=None):
    dest.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.md5()
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk); h.update(chunk)
    digest = h.hexdigest()
    if expected_md5 and digest.lower() != expected_md5.lower():
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"MD5 mismatch for {dest.name}: {digest} != {expected_md5}")
    print(f"Downloaded {dest} md5={digest}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--record", default="7812323")
    p.add_argument("--out", default="data/raw")
    p.add_argument("--files", nargs="*", default=["training_data.zip", "validation_data.zip"])
    args = p.parse_args()
    meta = get_json(f"https://zenodo.org/api/records/{args.record}")
    print("Title:", meta.get("metadata", {}).get("title"))
    print("License:", meta.get("metadata", {}).get("license"))
    available = {f["key"]: f for f in meta.get("files", [])}
    print("Available files:", ", ".join(available))
    for name in args.files:
        if name not in available:
            print(f"SKIP: {name} not present in this record")
            continue
        item = available[name]
        checksum = item.get("checksum", "")
        md5 = checksum.split(":", 1)[1] if checksum.startswith("md5:") else None
        url = item.get("links", {}).get("self") or item.get("links", {}).get("content")
        download(url, pathlib.Path(args.out) / name, md5)

if __name__ == "__main__":
    main()
