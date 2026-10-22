# Harmony panoramic detector (MONAI RetinaNet)

A second practice detector, for **panoramic films only**. It speaks the same
`/predict` and `/health` contract as `ml/serve_fasterrcnn.py`, so Harmony can
point at it without code changes (`src/lib/detector-contract.ts`).

Output is an unverified AI detection for clinician review. It is not
clinically validated and never a diagnosis.

## What it does

| Model learns (DENTEX) | Sent to Harmony as | Notes |
|---|---|---|
| caries | `cavity` | |
| deep caries | `cavity` | Harmony has no deep-caries category yet |
| impacted | `impacted tooth` | |
| periapical lesion | not sent | written to the server's own log (`PERIAPICAL_LOG`) |

Non-panoramic films get **HTTP 422**. Harmony treats any error from the
detector as "unavailable" and reads the film with its current (hosted) reader,
so bitewings, periapicals and photos keep working as today. Harmony sends only
the image, so the server decides by shape and colour: wide (aspect 1.6–3.2),
at least 1000 px across, and greyscale. Tune with `PANO_MIN_ASPECT`,
`PANO_MAX_ASPECT`, `PANO_MIN_WIDTH`.

Note: if one film in a multi-film read is refused, Harmony falls back for the
whole read. Send panoramics on their own to get this model's reading.

## Files

| File | Purpose |
|---|---|
| `labels.py` | The four classes and how they map to Harmony's wire names |
| `model.py` | RetinaNet build/load/save, film loading, prediction, panoramic check |
| `prepare_dentex.py` | DENTEX → `datalist.json` (train / val / test / unlabelled) |
| `train.py` | Training; keeps the best checkpoint by val mAP@0.5 |
| `baseline.py` | Loads dental-vision-v1's Faster R-CNN read-only |
| `make_drafts.py` | Draft labels for the unlabelled DENTEX panoramics |
| `evaluate.py` | Test-split report, MONAI vs the Faster R-CNN baseline |
| `serve_monai.py` | The `/health` + `/predict` service, plus `/annotate` |
| `render.py` | Draws each finding as a coloured, named outline on the film |
| `modal_train.py` | Modal jobs for the whole training pipeline |
| `modal_serve.py` | Modal deployment of the service |
| `publish_release.sh` | Puts the model file in a GitHub Release (never a commit) |
| `smoke_test.py` | CPU plumbing test on fake films |

## Pipeline

1. **Data.** DENTEX (Zenodo 7812323): about 705 fully labelled panoramics
   plus about 1,571 unlabelled ones. The split is by image (DENTEX has no
   patient ids), stratified so every class is in val and test.
2. **Teacher.** Train RetinaNet on the expert labels only.
3. **Draft labels.** Run the teacher over the unlabelled films. Keep a box
   when the teacher is confident (≥ 0.6), or moderately sure (≥ 0.35) *and*
   dental-vision-v1's Faster R-CNN found the same thing (IoU ≥ 0.4). The
   baseline only knows cavity and impacted tooth, so it can confirm caries,
   deep caries and impacted, never periapical lesions. Films with no kept
   boxes are dropped.
4. **Student.** Retrain on expert + draft labels. Drafts join training only.
5. **Score.** One run on the test split: all four classes, plus a side-by-side
   with the baseline on exactly what Harmony receives (`cavity`,
   `impacted tooth`).

The Faster R-CNN checkpoint is optional. Without it, step 3 uses only the
confident-teacher rule and step 5 reports MONAI alone; both outputs say so.

## Run it on Modal

```
cd panoramic_monai
pip install modal && modal setup
modal volume create harmony-monai
# optional baseline:
modal volume put harmony-monai harmony_detector_best.pth /baseline/harmony_detector_best.pth
modal run --detach modal_train.py
```

Then download `best.pt` and `report_test.json`
(`modal volume get harmony-monai /runs/student/best.pt .` and the same for
`/runs/report_test.json`), read the report, and publish:

```
./publish_release.sh best.pt report_test.json pano-monai-v1
```

That creates a **draft** GitHub Release on build-harmony holding the model
file and its SHA-256.

## Serve it

```
modal volume create harmony-pano
modal volume put harmony-pano best.pt /best.pt
modal secret create harmony-pano-key HARMONY_KEY=<long random key>
modal deploy modal_serve.py
```

Or anywhere with a GPU:

```
pip install -r requirements.txt
HARMONY_CHECKPOINT=best.pt HARMONY_KEY=<secret> uvicorn serve_monai:app --port 8002
```

Give Harmony the HTTPS address, key and `MODEL_VERSION`
(default `harmony_pano_monai_v1`) the same way as the existing detector.

## Before trusting it

- **Licence.** The DENTEX licence is reported as CC BY-SA 4.0 on GitHub and
  CC BY-NC-SA 4.0 on Hugging Face. Settle this before any commercial use (see
  dental-vision-v1's `DATASET.md`).
- **Draft labels are partial.** A finding the teacher missed counts as
  background in a draft film. The thresholds keep drafts conservative; check
  that the student beats the teacher on val before using it.
- **Scores are not clinical validation.** The DENTEX hold-out comes from the
  same source as training. Validate on panoramics from a different clinic or
  machine, and on Harmony's own clinician-confirmed films.
- **Periapical lesions** are logged, not shown. The log holds the film's
  SHA-256, image size and boxes only, no image data.

## Coloured outlines

`POST /annotate` (same key and upload as `/predict`) returns the film as a PNG:
each finding filled and outlined in its colour (caries yellow, deep caries
orange, periapical lesion magenta, impacted tooth blue) with its name and
confidence written beside it. Offline: `python render.py film.png --model best.pt --out marked.png`.

The model detects boxes, not shapes. Each outline is traced inside its box
from the film's own brightness (lesions darker, impacted teeth brighter), so
it is approximate, and the picture's footer says so. Harmony itself keeps
drawing boxes only.

## Smoke test

```
pip install -r requirements.txt
python smoke_test.py
```

Checks the plumbing only: data prep, a one-epoch train, drafts with a
stand-in baseline, evaluation, and the server's 200 / 401 / 422 responses and
box bounds.
