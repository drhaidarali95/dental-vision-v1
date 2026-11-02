#!/usr/bin/env bash
# Start the PHOTO part training (from harmony_dental/), with the account's token in KAGGLE_API_TOKEN.
set -euo pipefail
cd "$(dirname "$0")/.."
U=$(kaggle config view 2>/dev/null | awk -F': ' '/username/{print $2}' | tr -d ' ')
S=$(mktemp -d); mkdir -p "$S/code" "$S/kernel"
cp ../panoramic_monai/*.py ../periapical_monai/*.py ./*.py "$S/code/"
echo "{\"title\": \"harmony-dental-code\", \"id\": \"$U/harmony-dental-code\", \"licenses\": [{\"name\": \"other\"}]}" > "$S/code/dataset-metadata.json"
kaggle datasets version -p "$S/code" -m "code $(date -u +%FT%TZ)" -q || kaggle datasets create -p "$S/code" -q
cp photo/kernel.py "$S/kernel/kernel.py"
cat > "$S/kernel/kernel-metadata.json" <<META
{"id": "$U/harmony-dental-photo-train", "title": "harmony-dental-photo-train", "code_file": "kernel.py", "language": "python",
 "kernel_type": "script", "is_private": true, "enable_gpu": true, "enable_internet": false,
 "dataset_sources": ["$U/harmony-dental-code", "$U/harmony-wheels", "$U/harmony-resnet50", "$U/harmony-photo-caries"],
 "competition_sources": [], "kernel_sources": []}
META
sleep 30; kaggle kernels push -p "$S/kernel"; rm -rf "$S"
