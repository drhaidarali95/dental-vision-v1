# Live status (updated automatically every 2 hours)

Last update: 2026-10-30 05:15 UTC

## Now
- Done Oct 30, 05:05 UTC: DENTEX private copy; ToothFairy2 checkpoints (4.6 GB) and scans (26.1 GB) saved privately.
- Running since Oct 30, 05:10 UTC: panoramic v2 retrain (image-quality training). Expected done about Oct 30, 10:40 UTC.

## Queue (runs in this order, one GPU job at a time)
1. Panoramic v2 (robust) - findings + tooth numbering.
2. Periapical: fetch Zenodo 13772918 (CC BY 4.0) and DenPAR (Zenodo 16645076) into private Kaggle, inspect, train.
2b. Panoramic extra classes: add Kaggle "Dental Radiography" (CC BY-SA 4.0) for fillings + implants.
2c. Bitewing: Kaggle sprss96/bitewing-dataset (Roboflow mirror, CC BY 4.0) + ivinlyju/bitewing + shubhamskg/bitewing-datasets (VERIFY provenance). No key needed. Roboflow originals optional later.
2d. More periapical/panoramic: MMDental (CC0, 1,000 RVG+OPG, 21 classes), nadaaglan periapical (CC BY 4.0), HF periapical-lesions-pai (CC BY 4.0), HF Caries_Bounding_Box (MIT).
2e. Occlusal: no labelled occlusal set exists. Plan: one shared intraoral model (periapical + bitewing) that also reads occlusal films, reported with lower confidence and tested separately once any occlusal films are available.
3. One shared 2D X-ray model: panoramic + periapical (+ bitewing/occlusal when data exists).
4. CBCT: train on ToothFairy2 (data saved Oct 30, 05:05 UTC).

## Blocked (needs data, not effort)
- Occlusal: no labelled public set found on Kaggle, Hugging Face, Zenodo, Figshare, Mendeley, Roboflow or Scientific Data. Covered for now by the shared intraoral model (2e); real occlusal films needed to test it.

## Stopping rule
Each model retrains in rounds. A round is kept only if its held-out test score
beats the best so far. Training on a model stops after 2 rounds in a row
without improvement (more rounds on the same data would only memorise it), and
the best version is saved. Scores are always reported honestly; no model is
ever called 100% accurate.

## Scores so far (held-out test, 1.0 = perfect)
| Model | Version | Score | Notes |
|---|---|---|---|
| Panoramic findings | v1 | 0.54 | impacted 0.92, caries 0.47, deep caries 0.41, periapical 0.39 |
| Panoramic tooth numbers | v1 | 0.96 | |
