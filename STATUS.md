# Live status (updated automatically every 2 hours)

Last update: 2026-10-30 04:30 UTC

## Now
- DENTEX private copy (`dentex-data`): downloading.
- Next: robust panoramic retrain (v2, image-quality training), ~5 GPU hours.

## Queue (runs in this order, one GPU job at a time)
1. Panoramic v2 (robust) - findings + tooth numbering.
2. Periapical: fetch Zenodo 13772918 (CC BY 4.0) into private Kaggle, inspect, train.
3. One shared 2D X-ray model: panoramic + periapical (+ bitewing/occlusal when data exists).
4. CBCT: train on ToothFairy2 once its download is done (needs the owner's one click).

## Blocked (needs data, not effort)
- Bitewing and occlusal X-rays: no commercial-safe public data. Needs the owner's own anonymised films.
- CBCT: ToothFairy2 download must be started once from the owner's browser.

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
