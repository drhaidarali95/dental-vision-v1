#!/usr/bin/env bash
# Upload panoramic_monai + periapical_monai code privately and start the panoramic restorations GPU job.
#   ./kaggle/push_kaggle.sh          # from pano_restorations_monai/
#   ./kaggle/push_kaggle.sh status | fetch
set -euo pipefail
cd "$(dirname "$0")/.."
USER_NAME=$(kaggle config view 2>/dev/null | awk -F': ' '/username/{print $2}' | tr -d ' ')
DATASET="$USER_NAME/harmony-periapical-code"
KERNEL="$USER_NAME/harmony-pano-restorations-train"
case "${1:-push}" in
  status) kaggle kernels status "$KERNEL"; exit ;;
  fetch) mkdir -p "${OUT:-runs/kaggle}"; kaggle kernels output "$KERNEL" -p "${OUT:-runs/kaggle}"; exit ;;
esac
STAGE=$(mktemp -d); mkdir -p "$STAGE/code" "$STAGE/kernel"
cp ../panoramic_monai/*.py ../periapical_monai/*.py "$STAGE/code/"
echo "{\"title\": \"harmony-periapical-code\", \"id\": \"$DATASET\", \"licenses\": [{\"name\": \"other\"}]}" > "$STAGE/code/dataset-metadata.json"
if kaggle datasets status "$DATASET" >/dev/null 2>&1; then
  kaggle datasets version -p "$STAGE/code" -m "code update $(date -u +%FT%TZ)" -q
else
  kaggle datasets create -p "$STAGE/code" -q
fi
cp kaggle/kernel.py "$STAGE/kernel/kernel.py"
cat > "$STAGE/kernel/kernel-metadata.json" <<META
{"id": "$KERNEL", "title": "harmony-pano-restorations-train", "code_file": "kernel.py", "language": "python",
 "kernel_type": "script", "is_private": true, "enable_gpu": true, "enable_internet": true,
 "dataset_sources": ["$DATASET", "imtkaggleteam/dental-radiography"], "competition_sources": [], "kernel_sources": []}
META
sleep 30
OUTPUT=$(kaggle kernels push -p "$STAGE/kernel" 2>&1); echo "$OUTPUT"
if echo "$OUTPUT" | grep -qiE "error|quota"; then rm -rf "$STAGE"; echo "NOT started: see the message above." >&2; exit 1; fi
rm -rf "$STAGE"
echo "Started $KERNEL"
