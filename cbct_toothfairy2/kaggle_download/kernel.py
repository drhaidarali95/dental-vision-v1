"""Kaggle notebook: fetch ToothFairy2 (CBCT dataset + official checkpoints) into a private Kaggle dataset.

Needs three Kaggle Secrets attached to this notebook (Add-ons -> Secrets):
  DITTO_USER, DITTO_PASS  - the owner's ditto.ing.unimore.it login
  KAGGLE_API_TOKEN        - the owner's Kaggle API token, used to save the files
Nothing secret is printed. Data licence: CC BY-SA 4.0 (cite the papers listed on the site).
"""
import os, re, subprocess, sys, json, time
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle>=1.7", "requests"], check=True)
import requests
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
def secret(name):
    try:
        return secrets.get_secret(name)
    except Exception:
        sys.exit(f"Missing Kaggle Secret {name}: add it under Add-ons -> Secrets and tick it for this notebook.")

user, password, token = secret("DITTO_USER"), secret("DITTO_PASS"), secret("KAGGLE_API_TOKEN")
os.environ["KAGGLE_API_TOKEN"] = token

BASE = "https://ditto.ing.unimore.it"
s = requests.Session()
s.headers["User-Agent"] = "Mozilla/5.0 (harmony-toothfairy2-fetch)"

def get(url, **kw):
    for attempt in range(5):
        try:
            r = s.get(url, timeout=60, **kw)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            print(f"retry {attempt + 1} for {url}: {e}")
            time.sleep(10 * (attempt + 1))
    sys.exit(f"Could not reach {url}")

page = get(f"{BASE}/login/")
csrf = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', page.text).group(1)
r = s.post(f"{BASE}/login/", data={"csrfmiddlewaretoken": csrf, "username": user, "password": password},
           headers={"Referer": f"{BASE}/login/"}, timeout=60)
if "/login" in r.url or 'name="password"' in r.text:
    sys.exit("Login failed: check DITTO_USER / DITTO_PASS (username, not email, if the site asks for one).")
print("Logged in.")

html = get(f"{BASE}/toothfairy2/").text
links = sorted(set(re.findall(r'href="([^"]*(?:download|checkpoint)[^"]*)"', html, re.I)))
print("Download links on the page:", links)
targets = {"ToothFairy2_Dataset.zip": "/download/dataset/4/"}
ckpt = [l for l in links if "checkpoint" in l.lower()]
if ckpt:
    targets["tf2_checkpoints.zip"] = ckpt[0]
else:
    print("WARNING: no Checkpoints link found; fetching the dataset only.")

owner = subprocess.run(["kaggle", "config", "view"], capture_output=True, text=True).stdout
owner = re.search(r"username:\s*(\S+)", owner).group(1)


def download(url, dest):
    # Resumable: on a dropped connection, wait and continue from the bytes we have.
    total = None
    for attempt in range(30):
        have = dest.stat().st_size if dest.exists() else 0
        if total and have >= total:
            return
        try:
            headers = {"Range": f"bytes={have}-"} if have else {}
            with s.get(url, stream=True, timeout=120, headers=headers) as r:
                r.raise_for_status()
                if "text/html" in r.headers.get("Content-Type", ""):
                    sys.exit(f"{dest.name}: got a web page instead of a file (login or link problem).")
                if have and r.status_code != 206:
                    have = 0  # server ignored Range; start over
                total = have + int(r.headers.get("Content-Length", 0))
                last = have
                with open(dest, "ab" if have else "wb") as f:
                    for chunk in r.iter_content(8 << 20):
                        f.write(chunk)
                        have += len(chunk)
                        if have - last > (1 << 30):
                            last = have
                            print(f"  {have / 1e9:.1f} / {total / 1e9:.1f} GB", flush=True)
            if have >= total:
                return
        except requests.RequestException as e:
            print(f"  connection dropped ({e.__class__.__name__}); resuming", flush=True)
        time.sleep(min(300, 15 * (attempt + 1)))
    sys.exit(f"Could not download {dest.name}")


def save_dataset(folder, slug, title):
    # Each download becomes its own private Kaggle dataset, then is deleted locally.
    (folder / "dataset-metadata.json").write_text(json.dumps({
        "title": title, "id": f"{owner}/{slug}", "licenses": [{"name": "CC-BY-SA-4.0"}],
    }))
    exists = subprocess.run(["kaggle", "datasets", "status", f"{owner}/{slug}"], capture_output=True).returncode == 0
    cmd = (["kaggle", "datasets", "version", "-p", str(folder), "-m", "refresh"] if exists
           else ["kaggle", "datasets", "create", "-p", str(folder)])
    print(f"Saving {slug} privately (can take a while) ...", flush=True)
    subprocess.run(cmd, check=True)
    print(f"Saved: https://www.kaggle.com/datasets/{owner}/{slug}", flush=True)


slugs = {"tf2_checkpoints.zip": ("toothfairy2-checkpoints", "toothfairy2-checkpoints"),
         "ToothFairy2_Dataset.zip": ("toothfairy2", "toothfairy2")}
for name in sorted(targets, key=lambda n: n != "tf2_checkpoints.zip"):  # small file first
    slug, title = slugs[name]
    href = targets[name]
    folder = Path("/kaggle/tmp") / slug
    folder.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {name} ...", flush=True)
    download(href if href.startswith("http") else BASE + href, folder / name)
    print(f"  done: {(folder / name).stat().st_size / 1e9:.2f} GB", flush=True)
    save_dataset(folder, slug, title)
    subprocess.run(["rm", "-rf", str(folder)])
print("All done.")
