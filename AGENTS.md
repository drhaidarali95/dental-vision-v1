# Rules for this repository

This repository is the owner's standalone dental AI package. It is kept
separate from the Harmony app so it can be moved, licensed or sold on its own.

- Owner's goal: fully trained, high-capability diagnostic models that are the
  owner's own asset, usable in any project (not only Harmony), clonable,
  storable anywhere, improvable, licensable and sellable. Every model must be
  self-contained: weights + the code to load and run it + its test report +
  checksums, with nothing Harmony-specific required to use it.
- Robustness is a requirement: train with image-quality degradation (low
  resolution, blur, noise, exposure, compression) so old, scanned and
  low-resolution records still read, and report scores on degraded films too.
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
- Timing in every message, status and report is a specific clock time with date and time zone (e.g. "Oct 30, 11:40 UTC"), never only a duration like "about 5 hours". Give both when useful: "finishes about Oct 30, 11:40 UTC (~5 h)".
- Always report held-out test scores. Never claim or imply 100% accuracy or
  clinical validation.
