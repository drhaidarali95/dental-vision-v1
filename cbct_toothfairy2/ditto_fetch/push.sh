#!/usr/bin/env bash
# Start one CPU fetch job per ditto dataset on the main account. Needs <owner>/harmony-fetch-creds.
#   bash push.sh 6 toothfairy1 ToothFairy_Dataset.zip
set -euo pipefail
cd "$(dirname "$0")"
U=$(kaggle config view 2>/dev/null | awk -F': ' '/username/{print $2}' | tr -d ' ')
ID=$1; SLUG=$2; FILE=$3
S=$(mktemp -d)
python3 - "$S/kernel.py" "$ID" "$SLUG" "$FILE" <<'P'
import sys, json
out, i, slug, f = sys.argv[1:]
src = open("kernel.py").read()
src = src.replace('job = json.loads((Path(__file__).parent / "job.json").read_text()) if (Path(__file__).parent / "job.json").exists() else json.loads(os.environ["DITTO_JOB"])',
                  "job = " + json.dumps({"id": int(i), "slug": slug, "file": f}))
open(out, "w").write(src)
P
cat > "$S/kernel-metadata.json" <<META
{"id": "$U/fetch-$SLUG", "title": "fetch-$SLUG", "code_file": "kernel.py", "language": "python", "kernel_type": "script",
 "is_private": true, "enable_gpu": false, "enable_internet": true, "dataset_sources": ["$U/harmony-fetch-creds"],
 "competition_sources": [], "kernel_sources": []}
META
kaggle kernels push -p "$S"; rm -rf "$S"
