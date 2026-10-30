# Live status (updated automatically every 2 hours)

Last update: 2026-10-30 09:40 UTC

## Now
- Done Oct 30, 05:05 UTC: DENTEX private copy; ToothFairy2 checkpoints (4.6 GB) and scans (26.1 GB) saved privately.
- Done Oct 30, 08:28 UTC: panoramic v2 retrain. Clean test: findings 0.544 (v1 0.544), teeth 0.956 (v1 0.956). Degraded films: v2 0.546 vs v1 0.540. Kept v2 (near-tie; counts as round 1 without meaningful gain under the stopping rule). Saved privately as harmony-pano-models-v2.
- Running since Oct 30, 08:34 UTC: periapical model (1,882 periapical X-rays; caries, badly decayed, root canal, crown, restoration). Expected done about Oct 30, 10:30 UTC.

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
| Panoramic findings | v2 | 0.54 (degraded 0.55) | deep caries 0.43, periapical 0.40, impacted 0.92 |
| Panoramic tooth numbers | v2 | 0.96 | |

## Watchdog log
- Oct 30, 06:53 UTC: OK. Panoramic v2 retrain running (started 05:10, expected done ~10:40). No data/weights committed; registry, queue and dashboard on plan.
- Oct 30, 09:53 UTC: OK. Periapical training running (started 08:34, expected ~10:30). Panoramic v2 saved with checksums and scores. No data/weights committed.
