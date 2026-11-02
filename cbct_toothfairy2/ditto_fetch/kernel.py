"""Kaggle CPU job: copy one ditto.ing.unimore.it dataset into a private Kaggle dataset.

Uses Kaggle Secrets DITTO_USER, DITTO_PASS, KAGGLE_API_TOKEN (attach them and run from the
browser: Kaggle passes secrets only to browser-started runs). Copies ToothFairy v1, Maxillo and
Pulpy3D one after another. Nothing secret is printed.
"""
import json, os, re, subprocess, sys, time
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle>=1.7", "requests"], check=True)
import requests

from kaggle_secrets import UserSecretsClient
_sec = UserSecretsClient()
creds = {"ditto_user": _sec.get_secret("DITTO_USER"), "ditto_pass": _sec.get_secret("DITTO_PASS")}
os.environ["KAGGLE_API_TOKEN"] = _sec.get_secret("KAGGLE_API_TOKEN")
JOBS = [{"id": 6, "slug": "toothfairy1-cbct", "file": "ToothFairy_Dataset.zip"},
        {"id": 7, "slug": "maxillo-cbct", "file": "IAN_Maxillo_dataset.zip"},
        {"id": 12, "slug": "pulpy3d-cbct", "file": "Pulpy3D_Dataset.zip"}]
B = "https://ditto.ing.unimore.it"
s = requests.Session(); s.headers["User-Agent"] = "Mozilla/5.0 (harmony-ditto-fetch)"
pg = s.get(B + "/login/", timeout=60)
csrf = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', pg.text).group(1)
r = s.post(B + "/login/", data={"csrfmiddlewaretoken": csrf, "username": creds["ditto_user"], "password": creds["ditto_pass"]},
           headers={"Referer": B + "/login/"}, timeout=60)
if "/login" in r.url or 'name="password"' in r.text:
    sys.exit("ditto login failed")
print("Logged in.", flush=True)
for job in JOBS:
    folder = Path("/kaggle/tmp") / job["slug"]; folder.mkdir(parents=True, exist_ok=True)
    dest = folder / job["file"]
    url = f"{B}/download/dataset/{job['id']}/"
    total = None
    for attempt in range(40):
        have = dest.stat().st_size if dest.exists() else 0
        if total and have >= total:
            break
        try:
            with s.get(url, stream=True, timeout=180, headers={"Range": f"bytes={have}-"} if have else {}) as r:
                r.raise_for_status()
                if "text/html" in r.headers.get("Content-Type", ""):
                    sys.exit("got a web page instead of the file")
                if have and r.status_code != 206:
                    have = 0
                total = have + int(r.headers.get("Content-Length", 0)); last = have
                with open(dest, "ab" if have else "wb") as f:
                    for chunk in r.iter_content(8 << 20):
                        f.write(chunk); have += len(chunk)
                        if have - last > (2 << 30):
                            last = have; print(f"  {have/1e9:.1f} / {total/1e9:.1f} GB", flush=True)
            if have >= total:
                break
        except requests.RequestException as e:
            print("  dropped, resuming:", e.__class__.__name__, flush=True)
        time.sleep(min(300, 15 * (attempt + 1)))
    print("downloaded", dest.stat().st_size / 1e9, "GB", flush=True)
    owner = re.search(r"username:\s*(\S+)", subprocess.run(["kaggle", "config", "view"], capture_output=True, text=True).stdout).group(1)
    (folder / "dataset-metadata.json").write_text(json.dumps({"title": job["slug"], "id": f"{owner}/{job['slug']}", "licenses": [{"name": "other"}]}))
    subprocess.run(["kaggle", "datasets", "create", "-p", str(folder), "-q"], check=True)
    print("Saved privately:", f"{owner}/{job['slug']}", flush=True)
    subprocess.run(["rm", "-rf", str(folder)])
print("All done.", flush=True)
