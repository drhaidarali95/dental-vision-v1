# Harmony detector service

Trained classes: Cavity, Fillings, Implant, Impacted Tooth.

Place the trained checkpoint at models/harmony_detector_best.pth or set MODEL_PATH.
Set HARMONY_MODEL_API_KEY in production.

Run: uvicorn ml.serve:app --host 0.0.0.0 --port 8000
Health: GET /health
Inference: POST /predict as multipart field file, optionally Authorization: Bearer <key>.

This model is AI-assisted and requires clinician review. Current held-out evaluation used 73 images and IoU/score threshold 0.50; it is not a clinical validation study.
