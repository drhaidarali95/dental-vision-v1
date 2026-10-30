"""Kaggle notebook: keep one copy of DENTEX (Zenodo 7812323, CC BY 4.0) as this
notebook's output, so training jobs attach it in seconds instead of
downloading 10.9 GB from Zenodo every run. Only the two labelled folders are
kept (/kaggle/working/training_data/...)."""
import time
import zipfile
from pathlib import Path

import requests

URL = "https://zenodo.org/records/7812323/files/training_data.zip?download=1"
KEEP = ("training_data/quadrant-enumeration-disease/", "training_data/quadrant_enumeration/")
zpath = Path("/kaggle/tmp/training_data.zip")
zpath.parent.mkdir(parents=True, exist_ok=True)

total = None
for attempt in range(30):
    have = zpath.stat().st_size if zpath.exists() else 0
    if total and have >= total:
        break
    try:
        headers = {"Range": f"bytes={have}-"} if have else {}
        with requests.get(URL, headers=headers, stream=True, timeout=120) as r:
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(int(r.headers.get("Retry-After", 60)))
                continue
            r.raise_for_status()
            if have and r.status_code != 206:
                have = 0
            total = have + int(r.headers.get("Content-Length", 0))
            last = have
            with open(zpath, "ab" if have else "wb") as f:
                for chunk in r.iter_content(8 << 20):
                    f.write(chunk)
                    have += len(chunk)
                    if have - last > (1 << 30):
                        last = have
                        print(f"DENTEX {have / 1e9:.1f}/{total / 1e9:.1f} GB", flush=True)
    except requests.RequestException as e:
        print(f"interrupted ({e.__class__.__name__}); resuming", flush=True)
        time.sleep(min(300, 15 * (attempt + 1)))
else:
    raise SystemExit("could not download DENTEX")

with zipfile.ZipFile(zpath) as z:
    names = [n for n in z.namelist() if n.startswith(KEEP) and not n.endswith("/")]
    for n in names:
        z.extract(n, "/kaggle/working")
zpath.unlink()
Path("/kaggle/working/SOURCE.txt").write_text(
    "DENTEX training_data.zip from https://zenodo.org/records/7812323 (CC BY 4.0 on Zenodo; "
    f"see REGISTRY.md). Copied {time.strftime('%Y-%m-%d')}. {len(names)} files.\n")
print(f"DENTEX cached: {len(names)} files", flush=True)
