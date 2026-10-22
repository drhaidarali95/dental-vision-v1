#!/usr/bin/env bash
# Upload a trained checkpoint and its test report to a GitHub Release on
# build-harmony. The model file never goes into a commit.
#
#   ./publish_release.sh best.pt report_test.json pano-monai-v1
#
# Needs the GitHub CLI (gh) signed in with write access to the repo.
# The release is created as a DRAFT; publish it on GitHub once you've read
# the report.
set -euo pipefail

MODEL="${1:?path to best.pt}"
REPORT="${2:?path to report_test.json}"
TAG="${3:-pano-monai-v1}"
REPO="${REPO:-drhaidarali95/dental-vision-v1}"

SHA=$(sha256sum "$MODEL" | cut -d' ' -f1)
NOTES=$(mktemp)
{
  echo "MONAI RetinaNet panoramic detector (ml-monai/)."
  echo
  echo "- Model file: $(basename "$MODEL")"
  echo "- SHA-256: $SHA"
  echo "- Test report: $(basename "$REPORT") (DENTEX hold-out; not a clinical validation)"
  echo
  echo "AI detections for clinician review only."
} > "$NOTES"

gh release create "$TAG" "$MODEL" "$REPORT" --repo "$REPO" --draft \
  --title "Panoramic detector $TAG" --notes-file "$NOTES"
rm -f "$NOTES"
echo "Draft release $TAG created on $REPO (sha256 $SHA)."
