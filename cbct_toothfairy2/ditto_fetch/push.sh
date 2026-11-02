#!/usr/bin/env bash
# Upload the fetch notebook to the main account (owner then runs it once from the browser).
set -euo pipefail
cd "$(dirname "$0")"
U=$(kaggle config view 2>/dev/null | awk -F': ' '/username/{print $2}' | tr -d ' ')
S=$(mktemp -d); cp kernel.py "$S/"
cat > "$S/kernel-metadata.json" <<META
{"id": "$U/fetch-ditto-cbct", "title": "fetch-ditto-cbct", "code_file": "kernel.py", "language": "python", "kernel_type": "script",
 "is_private": true, "enable_gpu": false, "enable_internet": true, "dataset_sources": [], "competition_sources": [], "kernel_sources": []}
META
kaggle kernels push -p "$S"; rm -rf "$S"
