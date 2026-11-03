"""Harmony Dental AI model server.

  HARMONY_MODEL=harmony-dental.pt uvicorn serve:app --host 0.0.0.0 --port 8080

POST /analyze (multipart file=<image or scan>, optional film=periapical|bitewing|panoramic|occlusal)
 -> X-ray: perio charting JSON (perio/SPEC.md) + all findings; CBCT: structures + volumes.
GET /health -> model version and parts. Output is for dentist review only.
"""
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile

from charting import chart
from harmony_model import HarmonyDental

app = FastAPI(title="Harmony Dental AI")
MODEL = HarmonyDental(os.environ.get("HARMONY_MODEL", "harmony-dental.pt"))


@app.get("/health")
def health():
    return {"ok": True, "version": MODEL.version, "scores": MODEL.scores}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), film: str = Form("unknown")):
    suffix = "".join(Path(file.filename or "x").suffixes) or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        path = tmp.name
    try:
        if MODEL.is_volume(path):
            r = MODEL.read_cbct(path)
            r.pop("labels", None)
            return r
        raw = MODEL.read_xray(path, min_score=0.05)["findings"]
        out = chart(raw, film=film, model_version=MODEL.version)
        out["all_findings"] = [f for f in raw if f["score"] >= 0.3]
        out["note"] = "AI output for dentist review; not a diagnosis."
        return out
    finally:
        os.unlink(path)
