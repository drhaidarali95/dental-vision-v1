#!/usr/bin/env python3
"""Download DENTEX files from the official Zenodo record."""
import argparse, hashlib, json, pathlib, urllib.request


def get_json(url):
    with urllib.request.urlopen(url) as r:
        return json.load(r)


def normalize_md5(value):
    """Zenodo may expose checksum as md5:<hex>, plain hex, or a dict."""
    if isinstance(value, dict):
        value = value.get("md5") or value.get("checksum") or ""
    value = str(value or "").strip().lower()
    if value.startswith("md5:"):
        value = value.split(":", 1)[1]
    return value if len(value) == 32 and all(c in "0123456789abcdef" for c in value) else None


def download(url, dest, expected_md5=None):
    dest.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.md5()
    try:
        with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
            while True:
                chunk = r.read(4 * 1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                h.update(chunk)
        digest = h.hexdigest()
        if expected_md5 and digest.lower() != expected_md5.lower():
            raise RuntimeError(f"MD5 mismatch for {dest.name}: {digest} != {expected_md5}")
        print(f"Downloaded {dest} md5={digest}")
    except Exception:
        dest.unlink(missing_ok=True)
        raise


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
        expected_md5 = normalize_md5(item.get("checksum"))
        links = item.get("links", {})
        # Prefer the raw content endpoint; `self` may resolve to metadata/API content.
        url = links.get("content") or links.get("self")
        if not url:
            raise RuntimeError(f"No download URL for {name}")
        download(url, pathlib.Path(args.out) / name, expected_md5)

if __name__ == "__main__":
    main()
