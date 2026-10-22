# Dental Vision V1

Training workspace for an AI-assisted dental radiograph evidence extraction system.

## V1 targets
- tooth localization / numbering
- caries and deep caries
- periapical lesions
- impacted teeth

## Architecture
Radiograph -> vision evidence extraction -> structured findings JSON -> downstream clinician-reviewed diagnostic reasoning.

## Status
Repository initialized. Dataset licensing and provenance must be verified before training or commercial deployment. No clinical performance claims should be made without held-out validation.

## Layout
- `REGISTRY.md` - every dataset, model and outside code, with license and commercial status. Read first.
- `ROADMAP.md` - the planned models, one per record type, and their data.
- `AGENTS.md` - rules for keeping this package portable and sellable.
- `panoramic_monai/` - MONAI panoramic findings + tooth-numbering models (Kaggle GPU training).
- `cbct_toothfairy2/` - CBCT work; `kaggle_download/` fetches ToothFairy2 into a private Kaggle dataset.
- `ml/`, `train.py`, `evaluate.py`, `src/`, `notebooks/` - original v1 Faster R-CNN detector.
