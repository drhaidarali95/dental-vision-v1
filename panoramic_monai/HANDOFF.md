# panoramic_monai hand-off (for the next session)

Read this first when continuing the Harmony detector work.

## Decisions already made by the owner
- Build detectors for every dental record type (panoramic, periapical,
  bitewing, clinical photos, CBCT, 3D intraoral scans), one model per type.
- **Commercial-safe data only** (Harmony is a paid product). Non-commercial
  datasets (CC BY-NC / NC-ND) are skipped: the Mendeley bitewing set and
  Teeth3DS 3D scans.
- GPU: **Kaggle** (free). The owner added `KAGGLE_API_TOKEN` to the cloud
  environment's variables. Do not use Modal or Google Cloud.
- Harmony changes are allowed but live on branch `imaging-anatomy-toggle`
  until the owner approves a merge (merging syncs to Lovable). No database
  changes.
- Never claim or imply 100% accuracy; always report held-out test scores.

## State
- Branch `monai-panoramic-detector`: all ml-monai code (findings model,
  tooth-numbering model, server with /predict, /anatomy, /annotate layers,
  coloured outlines, smoke test passing).
- Branch `imaging-anatomy-toggle`: Harmony Findings / Anatomy / Both switch,
  FDI -> Universal tooth numbers; all 589 tests pass. Not merged.
- No trained model exists yet. CPU training in the session kept dying when
  the session paused (reached 3 of 20 epochs, val mAP@0.5 0.23 at epoch 2).

## Next steps
1. `cd ml-monai && pip install kaggle && ./kaggle/push_kaggle.sh`
   (if the username can't be detected, `kaggle config view` shows it; export
   `KAGGLE_USERNAME`). This uploads the code privately and starts one Kaggle
   GPU run: DENTEX download, findings model (40 epochs, 1536 px), test report,
   tooth model (30 epochs), tooth test report. Expect ~5-7 GPU hours.
2. `./kaggle/push_kaggle.sh status` until complete; then `fetch`.
3. Show the owner real test films (findings, anatomy, both) with
   `render.py`, and the honest scores from `report_test.json` and
   `report_teeth_test.json`.
4. `publish_release.sh` for both `best.pt` files (draft GitHub Release on
   build-harmony; never commit model files).
5. Next record types (commercial-safe data found so far, all CC BY 4.0 on
   Zenodo; inspect formats before writing prep code):
   - Periapical lesions: zenodo 13772918
   - Clinical photos, caries: zenodo 14827784 (Zenodo says CC BY 4.0, the
     paper says CC BY-NC-ND: tell the owner before training on it)
   - Clinical photos, gums (healthy/gingivitis/periodontitis): zenodo 22047269
   - CBCT: ToothFairy2 (CC BY-SA, 480 scans, FDI teeth) needs the owner's
     free grand-challenge.org sign-up; asked, not yet confirmed.
   - Bitewing and 3D intraoral scans: no commercial-safe public set found yet.
