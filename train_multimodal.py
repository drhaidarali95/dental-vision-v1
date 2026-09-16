"""Unified Dental Vision training launcher.

The product is one multimodal diagnostic system, while each modality keeps an
appropriate specialist encoder/model. This launcher coordinates those branches
and reserves patient-level fusion for evidence representations rather than raw
pixels/voxels/meshes.
"""
import argparse
import json
from pathlib import Path


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)

    required = ["xray", "cbct", "photo", "surface_scan", "clinical_record", "fusion", "implant_planning"]
    missing = [k for k in required if k not in cfg.get("modules", {})]
    if missing:
        raise SystemExit(f"Missing multimodal modules: {missing}")

    print("Dental Vision multimodal training plan")
    for name in required:
        module = cfg["modules"][name]
        print(f"- {name}: enabled={module.get('enabled', True)}")

    if args.dry_run:
        print("DRY RUN OK: configuration is structurally complete.")
        return

    raise SystemExit(
        "Training adapters must be invoked only after dataset paths, licenses, "
        "patient-level splits, and modality-specific evaluation targets are verified."
    )


if __name__ == "__main__":
    main()
