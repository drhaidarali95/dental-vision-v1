# Using these models anywhere

Each trained model is a standalone asset. Nothing here depends on Harmony.

## Get everything (two steps)

1. Code: `git clone https://github.com/drhaidarali95/dental-vision-v1`
2. Weights (private, owner's Kaggle account):
   `kaggle datasets download drhaidarali/harmony-pano-models-v1 --unzip -p models/`
   Check them: `sha256sum -c models/SHA256SUMS.txt`

`REGISTRY.md` lists every model, where its weights are, its checksum, its
training data and license, and its test scores.

## Run it

```
cd panoramic_monai
pip install -r requirements.txt
python render.py film.png --model ../models/pano-findings-v1.pt --out marked.png   # picture
```

Or as a web service any app can call (`/health`, `/predict`, `/anatomy`, `/annotate`):

```
HARMONY_CHECKPOINT=../models/pano-findings-v1.pt \
ANATOMY_CHECKPOINT=../models/pano-teeth-v1.pt \
HARMONY_KEY=<a random secret callers must send> \
uvicorn serve_monai:app --host 0.0.0.0 --port 8000
```

Or in your own Python code: `model.load_detector(path, device)` then
`model.predict(detector, image, device)` returns `[{class, score, box}]`.

## Move, back up, license or sell

Copy three things together: this repository, the weights dataset, and
`REGISTRY.md` (provenance and licenses). Settle every **VERIFY** row in the
registry before a sale.

All outputs are AI detections for clinician review, not diagnoses.
