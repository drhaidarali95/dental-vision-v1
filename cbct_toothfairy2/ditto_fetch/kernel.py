"""Kaggle CPU job: copy one ditto.ing.unimore.it dataset into a private Kaggle dataset.

Reads the owner's credentials from the attached PRIVATE dataset <owner>/harmony-fetch-creds
(creds.json: ditto_user, ditto_pass, kaggle_token). Nothing secret is printed.
The dataset number and slug come from job.json next to this script (written by push.sh).
"""
import json, os, re, subprocess, sys, time
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle>=1.7", "requests"], check=True)
import requests

job = json.loads((Path(__file__).parent / "job.json").read_text()) if (Path(__file__).parent / "job.json").exists() else json.loads(os.environ["DITTO_JOB"])
creds = json.loads(next(Path("/kaggle/input").rglob("creds.json")).read_text())
os.environ["KAGGLE_API_TOKEN"] = creds["kaggle_token"]
B = "https://ditto.ing.unimore.it"
s = requests.Session(); s.headers["User-Agent"] = "Mozilla/5.0 (harmony-ditto-fetch)"
pg = s.get(B + "/login/", timeout=60)
csrf = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', pg.text).group(1)
r = s.post(B + "/login/", data={"csrfmiddlewaretoken": csrf, "username": creds["ditto_user"], "password": creds["ditto_pass"]},
           headers={"Referer": B + "/login/"}, timeout=60)
if "/login" in r.url or 'name="password"' in r.text:
    sys.exit("ditto login failed")
print("Logged in.", flush=True)
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
