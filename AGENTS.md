# Rules for this repository

This repository is the owner's standalone dental AI package. It is kept
separate from the Harmony app so it can be moved, licensed or sold on its own.

- One home: all model code, training jobs and data download scripts live here,
  one folder per record type (`panoramic_monai/`, `cbct_toothfairy2/`, ...).
  Harmony only calls the served models over the `/predict` contract.
- `REGISTRY.md` lists every dataset, model and piece of outside code with its
  exact source, license and commercial status. Update it in the same commit.
- Commercial-safe data only. NonCommercial (NC) licenses are never used.
  Anything unclear is marked VERIFY until settled.
- Never commit datasets, scans or model weights. They go to private Kaggle
  datasets or GitHub Releases, recorded in `REGISTRY.md`.
- Secrets (Kaggle token, dataset logins) live only in Kaggle Secrets or
  environment variables, never in code.
- Always report held-out test scores. Never claim or imply 100% accuracy or
  clinical validation.
