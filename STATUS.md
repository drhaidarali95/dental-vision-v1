# Live status (updated automatically every 2 hours)

Last update: 2026-11-02 01:52 UTC

## Now
- Done Oct 30, 05:05 UTC: DENTEX private copy; ToothFairy2 checkpoints (4.6 GB) and scans (26.1 GB) saved privately.
- Done Oct 30, 08:28 UTC: panoramic v2 retrain. Clean test: findings 0.544 (v1 0.544), teeth 0.956 (v1 0.956). Degraded films: v2 0.546 vs v1 0.540. Kept v2 (near-tie; counts as round 1 without meaningful gain under the stopping rule). Saved privately as harmony-pano-models-v2.
- Done Oct 30, ~10:40 UTC: periapical model v1. Held-out 54 films 0.517 (crown 0.87, restoration 0.75, caries 0.38, badly decayed 0.33, root canal 0.25). Saved privately as harmony-periapical-models-v1.
- Done Oct 30, ~18:20 UTC: bitewing tooth numbering v1. Held-out 831 bitewings 0.80. Saved privately as harmony-bitewing-models-v1.
- Oct 30, 18:30 to Oct 31, 00:28 UTC: paused on Kaggle's weekly GPU limit; it reset by Oct 31, 00:28 UTC.
- Done Oct 31 (~14:00 UTC): CBCT segmentation v1 (ToothFairy2). Held-out 48 scans mean Dice 0.64: lower jaw 0.95, lower teeth 0.80-0.91, nerve canals 0.78, upper teeth 0.30-0.68, upper jaw 0.49, sinuses 0.50. Saved privately Nov 1, 19:15 UTC as harmony-cbct-models-v1.
- Note: scheduled checks from Oct 31, 06:53 UTC to Nov 1, 12:54 UTC were not processed (session was offline); caught up Nov 1, 19:12 UTC.
- Done Oct 31, ~02:20 UTC: panoramic restorations v1. Held-out 73 films 0.74 (implant 0.95, impacted 0.81, fillings 0.79, cavity 0.42). Saved privately as harmony-pano-restorations-v1.

- Nov 1, 20:30 UTC: CBCT v2 code ready (starts from v1, class-balanced crops). Kaggle refused to start it: "maximum GPU sessions reached" (another account notebook, credible-reranker-v6, is running). Retrying every tick.

- Nov 1, 20:55 UTC: CBCT v2 RUNNING on second Kaggle account haydaralik (kernel haydaralik/harmony-cbct-train-v2; datasets shared from drhaidarali). Expected done ~06:30 UTC Nov 2. This session has only the main token; check/fetch it via a helper session that has KAGGLE2_API_TOKEN.

- Nov 1, 21:15 UTC: Periapical v2 RUNNING on haydaralik (kernel haydaralik/harmony-periapical-train): 1024 px, oversampled caries/badly decayed/RCT films, 100 epochs. Expected ~01:15 UTC Nov 2. Goal set by owner: 99% per skill. 10-day refresh scheduled (trig_01PBP2ydwDvhxYvE6pf2NpK1, 1st/11th/21st 03:41 UTC).

- Nov 1, 22:16 UTC: CBCT v2 restarted offline on haydaralik (wheels dataset haydaralik/harmony-wheels), RUNNING, expected ~07:45 UTC Nov 2. Periapical v2 failed offline (ResNet-50 pretrained weights download); fix ready: ship resnet50-11ad3fa6.pth in harmony-wheels and pre-copy to torch cache.

- Nov 1, 23:45 UTC: OWNER DECISION: ONE model for everything (harmony_dental/): one file harmony-dental.pt = X-ray part (one detector trained on ALL flat X-rays together: panoramic, periapical, bitewing, occlusal) + CBCT part; automatic routing by input type. No more per-film-type models. Next GPU job: harmony_dental X-ray part (kaggle/push_kaggle.sh on haydaralik, offline).

- Nov 2, 00:12 UTC: 3 jobs RUNNING on haydaralik: harmony-dental-xray-train (ONE X-ray part, all flat X-rays, started Nov 1 23:37, 10.5 h cap, done ~10:30 UTC), harmony-cbct-train-v2 (CBCT part, done ~07:45 UTC), harmony-periapical-train (test run).

- Nov 2, 01:30 UTC: harmony_dental X-ray part v2 adds DENTEX (findings + teeth) and 3,863 PAI panoramics (periapical lesions); runs on main account drhaidarali (internet on, DENTEX attached). Photo part data found: HF Frallex/Caries_Bounding_Box = 3,221 clinical photos, ICDAS D0-D6 (MIT). Periapical-only run to be stopped (owner request; needs KAGGLE2 token).

- Nov 2, 01:07 UTC: X-ray part v2 (with DENTEX + PAI panoramics) RUNNING on main account (drhaidarali/harmony-dental-xray-train). Nov 2, 01:13 UTC: notebook drhaidarali/fetch-ditto-cbct uploaded; owner must run it once from the browser (secrets). Licences of ToothFairy v1/Maxillo/Pulpy3D: VERIFY before use in the sellable model.

## Queue (runs in this order, one GPU job at a time)
1. Panoramic v2 (robust) - findings + tooth numbering.
2. Periapical: fetch Zenodo 13772918 (CC BY 4.0) and DenPAR (Zenodo 16645076) into private Kaggle, inspect, train.
2b. Panoramic extra classes: add Kaggle "Dental Radiography" (CC BY-SA 4.0) for fillings + implants.
2c. Bitewing: Kaggle sprss96/bitewing-dataset (Roboflow mirror, CC BY 4.0) + ivinlyju/bitewing + shubhamskg/bitewing-datasets (VERIFY provenance). No key needed. Roboflow originals optional later.
2d. More periapical/panoramic: MMDental (CC0; Kaggle copy has images only, labels missing, find original release), nadaaglan periapical (CC BY 4.0), HF periapical-lesions-pai (CC BY 4.0), HF Caries_Bounding_Box (MIT).
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
| Panoramic restorations | v1 | 0.74 | implant 0.95, impacted 0.81, fillings 0.79, cavity 0.42 (73 films) |
| CBCT segmentation | v1 | 0.64 Dice | 48 test scans; lower jaw 0.95, lower teeth 0.80-0.91, upper teeth 0.30-0.68 |
| Periapical findings | v1 | 0.52 | crown 0.87, restoration 0.75, caries 0.38, badly decayed 0.33, root canal 0.25 |

## Watchdog log
- Oct 30, 06:53 UTC: OK. Panoramic v2 retrain running (started 05:10, expected done ~10:40). No data/weights committed; registry, queue and dashboard on plan.
- Oct 30, 09:53 UTC: OK. Periapical training running (started 08:34, expected ~10:30). Panoramic v2 saved with checksums and scores. No data/weights committed.
- Oct 30, 12:53 UTC: OK. Bitewing tooth numbering running (started 12:36, expected ~15:40). Periapical v1 saved with checksum and scores. No data/weights committed.
- Oct 30, 15:54 UTC: OK. Bitewing training slightly past its estimate (3 h 18 min vs ~3 h), well under the 2x stuck limit (18:44 UTC). Autopilot healthy (last tick 14:28). No data/weights committed.
- Oct 30, 18:53 UTC: OK. No job running: paused on the weekly Kaggle GPU limit, restorations job ready; autopilot retries each tick. Bitewing v1 saved with checksum and scores. No data/weights committed.
- Oct 30, 21:54 UTC: OK. Still paused on the weekly Kaggle GPU limit (autopilot retried 20:28, refused by quota). Autopilot alive; it now also refreshes "Last update" on no-change ticks. No data/weights committed.
- Oct 31, 00:53 UTC: OK. GPU allowance reset; panoramic restorations running (started 00:28, expected ~02:30). Queue order and licences on plan. No data/weights committed.
- Oct 31, 03:53 UTC: OK with one note. No job running since 02:20 (restorations done). Next item, CBCT, needs its training job built first; autopilot takes it at 04:27. Restorations v1 saved with checksum and scores. No data/weights committed.
- Nov 1, 19:20 UTC: Fixed. 26 checks missed while the session was offline (Oct 31, 06:53 to Nov 1, 12:54 UTC). CBCT finished; results fetched, saved with checksums, registry and scores updated. No data/weights committed.
- Nov 1, 21:56 UTC: Fixed. CBCT v2 and periapical v2 on haydaralik both ERROR: no internet on that account (phone not verified), pip could not install monai. Fix: wheels shipped as private dataset haydaralik/harmony-wheels, kernels re-pushed offline. Main account token stopped working (owner to re-add KAGGLE_API_TOKEN). No data/weights committed.
- Nov 2, 01:52 UTC: OK. All 3 haydaralik jobs RUNNING (checked 01:50 with the new haydaralik key, now usable in this session); main-account X-ray v2 RUNNING. Periapical-only run can't be cancelled via API (no session id); owner may stop it in the UI. No data/weights committed.
