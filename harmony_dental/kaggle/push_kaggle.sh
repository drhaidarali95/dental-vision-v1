#!/usr/bin/env bash
# Upload the code and start the ONE X-ray training job (from harmony_dental/).
# Run with the account's token in KAGGLE_API_TOKEN.  ./kaggle/push_kaggle.sh | status | fetch
set -euo pipefail
cd "$(dirname "$0")/.."
U=$(kaggle config view 2>/dev/null | awk -F': ' '/username/{print $2}' | tr -d ' ')
DATASET="$U/harmony-dental-code"; KERNEL="$U/harmony-dental-xray-train"
case "${1:-push}" in
  status) kaggle kernels status "$KERNEL"; exit ;;
  fetch) mkdir -p "${OUT:-runs/kaggle}"; kaggle kernels output "$KERNEL" -p "${OUT:-runs/kaggle}"; exit ;;
esac
# Main account (internet on) also attaches the private DENTEX copy.
if [ "$U" = "drhaidarali" ]; then NET=true; KSRC="\"drhaidarali/dentex-data\""; else NET=false; KSRC=""; fi
S=$(mktemp -d); mkdir -p "$S/code" "$S/kernel"
cp ../panoramic_monai/*.py ../periapical_monai/*.py ./*.py "$S/code/"
echo "{\"title\": \"harmony-dental-code\", \"id\": \"$DATASET\", \"licenses\": [{\"name\": \"other\"}]}" > "$S/code/dataset-metadata.json"
if kaggle datasets status "$DATASET" >/dev/null 2>&1; then kaggle datasets version -p "$S/code" -m "code $(date -u +%FT%TZ)" -q; else kaggle datasets create -p "$S/code" -q; fi
cp kaggle/kernel.py "$S/kernel/kernel.py"
cat > "$S/kernel/kernel-metadata.json" <<META
{"id": "$KERNEL", "title": "harmony-dental-xray-train", "code_file": "kernel.py", "language": "python",
 "kernel_type": "script", "is_private": true, "enable_gpu": true, "enable_internet": $NET,
 "dataset_sources": ["$DATASET", "$U/harmony-wheels", "$U/harmony-resnet50", "nadaaglan/dental-periapical-x-rays", "imtkaggleteam/dental-radiography", "sprss96/bitewing-dataset"],
 "competition_sources": [], "kernel_sources": [$KSRC]}
META
sleep 30
OUT=$(kaggle kernels push -p "$S/kernel" 2>&1); echo "$OUT"; rm -rf "$S"
echo "$OUT" | grep -qiE "error|quota|maximum" && { echo "NOT started" >&2; exit 1; }
echo "Started $KERNEL"
