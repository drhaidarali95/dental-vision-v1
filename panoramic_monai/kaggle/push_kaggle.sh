#!/usr/bin/env bash
# Upload ml-monai as a private Kaggle dataset and start the GPU training notebook.
#
#   KAGGLE_API_TOKEN=KGAT_... ./kaggle/push_kaggle.sh          # from ml-monai/
#   ./kaggle/push_kaggle.sh status                              # is it running?
#   ./kaggle/push_kaggle.sh fetch                               # download results
#
# Needs: pip install kaggle (CLI 2.x), and a phone-verified Kaggle account
# (Kaggle gives GPUs only to verified accounts). Everything is created PRIVATE.
set -euo pipefail
cd "$(dirname "$0")/.."
USER_NAME=$(kaggle config view 2>/dev/null | awk -F': ' '/username/{print $2}' | tr -d ' ')
[ -n "$USER_NAME" ] || USER_NAME=$(python3 -c "import kaggle; print(kaggle.api.get_config_value('username') or '')" 2>/dev/null || true)
[ -n "$USER_NAME" ] || { echo "Set KAGGLE_USERNAME too if the CLI can't tell who you are."; USER_NAME=${KAGGLE_USERNAME:?}; }
DATASET="$USER_NAME/harmony-ml-monai-code"
KERNEL="$USER_NAME/harmony-ml-monai-train"
OUT=${OUT:-runs/kaggle}

case "${1:-push}" in
  status) kaggle kernels status "$KERNEL"; exit ;;
  fetch) mkdir -p "$OUT"; kaggle kernels output "$KERNEL" -p "$OUT"; ls -la "$OUT"; exit ;;
esac

STAGE=$(mktemp -d)
mkdir -p "$STAGE/code" "$STAGE/kernel"
cp ./*.py requirements.txt "$STAGE/code/"
cat > "$STAGE/code/dataset-metadata.json" <<EOF
{"title": "harmony-ml-monai-code", "id": "$DATASET", "licenses": [{"name": "other"}]}
EOF
if kaggle datasets status "$DATASET" >/dev/null 2>&1; then
  kaggle datasets version -p "$STAGE/code" -m "code update $(date -u +%FT%TZ)" -q
else
  kaggle datasets create -p "$STAGE/code" -q   # private by default
fi

cp kaggle/kernel.py "$STAGE/kernel/kernel.py"
cat > "$STAGE/kernel/kernel-metadata.json" <<EOF
{
  "id": "$KERNEL",
  "title": "harmony-ml-monai-train",
  "code_file": "kernel.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["$DATASET"],
  "competition_sources": [],
  "kernel_sources": ["$USER_NAME/dentex-data"]
}
EOF
# A new dataset version takes a moment to become attachable.
sleep 30
OUTPUT=$(kaggle kernels push -p "$STAGE/kernel" 2>&1); echo "$OUTPUT"
if echo "$OUTPUT" | grep -qiE "error|quota"; then rm -rf "$STAGE"; echo "NOT started: see the message above." >&2; exit 1; fi
rm -rf "$STAGE"
echo "Started $KERNEL. Check with: ./kaggle/push_kaggle.sh status"
