# Live status (updated automatically every 2 hours)

Last update: 2026-10-30 18:40 UTC

## Now
- Done Oct 30, 05:05 UTC: DENTEX private copy; ToothFairy2 checkpoints (4.6 GB) and scans (26.1 GB) saved privately.
- Done Oct 30, 08:28 UTC: panoramic v2 retrain. Clean test: findings 0.544 (v1 0.544), teeth 0.956 (v1 0.956). Degraded films: v2 0.546 vs v1 0.540. Kept v2 (near-tie; counts as round 1 without meaningful gain under the stopping rule). Saved privately as harmony-pano-models-v2.
- Done Oct 30, ~10:40 UTC: periapical model v1. Held-out 54 films 0.517 (crown 0.87, restoration 0.75, caries 0.38, badly decayed 0.33, root canal 0.25). Saved privately as harmony-periapical-models-v1.
- Done Oct 30, ~18:20 UTC: bitewing tooth numbering v1. Held-out 831 bitewings 0.80. Saved privately as harmony-bitewing-models-v1.
- PAUSED Oct 30, 18:30 UTC: Kaggle's free weekly GPU allowance (30 h, shared with the owner's other Kaggle projects) is used up. Next job ready to go: panoramic restorations (cavity, fillings, impacted tooth, implant; pano_restorations_monai). The autopilot retries every tick and starts it as soon as the allowance resets (weekly; expected around Oct 31, 00:00 UTC).

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
| Bitewing tooth numbers | v1 | 0.80 | 831 test bitewings; weakest teeth 33, 23, 18, 28 |
| Periapical findings | v1 | 0.52 | crown 0.87, restoration 0.75, caries 0.38, badly decayed 0.33, root canal 0.25 |

## Watchdog log
- Oct 30, 06:53 UTC: OK. Panoramic v2 retrain running (started 05:10, expected done ~10:40). No data/weights committed; registry, queue and dashboard on plan.
- Oct 30, 09:53 UTC: OK. Periapical training running (started 08:34, expected ~10:30). Panoramic v2 saved with checksums and scores. No data/weights committed.
- Oct 30, 12:53 UTC: OK. Bitewing tooth numbering running (started 12:36, expected ~15:40). Periapical v1 saved with checksum and scores. No data/weights committed.
- Oct 30, 15:54 UTC: OK. Bitewing training slightly past its estimate (3 h 18 min vs ~3 h), well under the 2x stuck limit (18:44 UTC). Autopilot healthy (last tick 14:28). No data/weights committed.
- Oct 30, 18:53 UTC: OK. No job running: paused on the weekly Kaggle GPU limit, restorations job ready; autopilot retries each tick. Bitewing v1 saved with checksum and scores. No data/weights committed.
