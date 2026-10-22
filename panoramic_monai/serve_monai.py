"""Harmony panoramic detector service (MONAI RetinaNet).

Same wire contract as ml/serve_fasterrcnn.py and src/lib/detector-contract.ts:

  GET  /health   -> model status (no auth, no weights exposed)
  POST /predict  -> Authorization: Bearer <key>, multipart field "file"
  POST /anatomy  -> same input; numbered teeth (FDI 11-48) with names. Needs
                    ANATOMY_CHECKPOINT (the tooth model); 503 without it.
  POST /annotate?layer=findings|anatomy|both -> the film as PNG, findings
                    outlined in colour and named, and/or teeth numbered.

With ANATOMY_CHECKPOINT set, each /predict finding also carries "tooth" (FDI
number of the tooth it overlaps most), or null when no tooth covers it.

  HARMONY_CHECKPOINT=/path/best.pt HARMONY_KEY=<secret> \
  uvicorn serve_monai:app --host 0.0.0.0 --port 8002

Panoramic films only. Anything else gets HTTP 422, which makes Harmony fall
back to its current reader. Caries and deep caries go out as "cavity",
impacted teeth as "impacted tooth". Periapical lesions are NOT sent; they are
written to this server's own log (PERIAPICAL_LOG, JSON lines, no image data).

Output is an AI detection for clinician review only.
"""
import hashlib
import io
import json
import logging
import os
import time

import torch
from fastapi import FastAPI, File, Header, HTTPException, Query, Response, UploadFile

from anatomy import read_teeth, tooth_for
from labels import LOG_ONLY, TEETH, WIRE_CATEGORY
from model import load_detector, open_film, panoramic_check, predict

CHECKPOINT = os.environ["HARMONY_CHECKPOINT"]
ANATOMY_CHECKPOINT = os.environ.get("ANATOMY_CHECKPOINT", "").strip()
ANATOMY_VERSION = os.environ.get("ANATOMY_VERSION", "harmony_teeth_monai_v1")
ANATOMY_MIN_CONF = float(os.environ.get("ANATOMY_MIN_CONF", "0.4"))
KEY = os.environ["HARMONY_KEY"]
VERSION = os.environ.get("MODEL_VERSION", "harmony_pano_monai_v1")
THRESHOLD = float(os.environ.get("MIN_CONF", "0.5"))
PERIAPICAL_LOG = os.environ.get("PERIAPICAL_LOG", "periapical_findings.jsonl")
PANO_MIN_ASPECT = float(os.environ.get("PANO_MIN_ASPECT", "1.6"))
PANO_MAX_ASPECT = float(os.environ.get("PANO_MAX_ASPECT", "3.2"))
PANO_MIN_WIDTH = int(os.environ.get("PANO_MIN_WIDTH", "1000"))
MAX_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "40")) * 1024 * 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log = logging.getLogger("serve_monai")
logging.basicConfig(level=logging.INFO)
detector, ckpt = load_detector(CHECKPOINT, DEVICE)
teeth_model = load_detector(ANATOMY_CHECKPOINT, DEVICE, expect=TEETH)[0] if ANATOMY_CHECKPOINT else None
LOADED_AT = time.time()

app = FastAPI()


def log_periapical(film_hash: str, width: int, height: int, items: list[dict]) -> None:
    """Server-side record only. Harmony never receives these."""
    record = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "model": VERSION,
              "film_sha256": film_hash, "image": {"width": width, "height": height}, "findings": items}
    log.info("periapical_lesion %s", json.dumps(record))
    try:
        with open(PERIAPICAL_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as error:
        log.warning("could not write %s: %s", PERIAPICAL_LOG, error)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": VERSION,
        "architecture": "monai_retinanet_resnet50_fpn",
        "classes": sorted(set(WIRE_CATEGORY.values())),
        "modality": "panoramic",
        "anatomy": {"available": teeth_model is not None, "model": ANATOMY_VERSION if teeth_model else None},
        "epoch": ckpt.get("epoch"),
        "threshold": THRESHOLD,
        "device": str(DEVICE),
        "weights_modified": False,
        "loaded_seconds_ago": round(time.time() - LOADED_AT),
    }


async def _read_panoramic(file: UploadFile, authorization: str):
    if authorization != f"Bearer {KEY}":
        raise HTTPException(401, "bad key")
    raw = await file.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "file too large")
    try:
        img = open_film(io.BytesIO(raw))
    except Exception:
        raise HTTPException(400, "not an image")
    reason = panoramic_check(img, PANO_MIN_ASPECT, PANO_MAX_ASPECT, PANO_MIN_WIDTH)
    if reason:
        raise HTTPException(422, f"This model reads panoramic films only ({reason}).")
    return raw, img


def _teeth(img) -> list[dict]:
    return read_teeth(teeth_model, img, DEVICE, ANATOMY_MIN_CONF) if teeth_model else []


@app.post("/anatomy")
async def anatomy_route(file: UploadFile = File(...), authorization: str = Header("")):
    _, img = await _read_panoramic(file, authorization)
    if teeth_model is None:
        raise HTTPException(503, "The anatomy model is not loaded on this server.")
    W, H = img.size
    started = time.time()
    structures = [{
        "structure": "tooth",
        "tooth": int(t["tooth"]),
        "name": t["name"],
        "description": f"Tooth {t['tooth']}: {t['name'].lower()}.",
        "confidence": t["confidence"],
        "bbox": dict(zip(("x1", "y1", "x2", "y2"), t["box"])),
        "source": ANATOMY_VERSION,
    } for t in _teeth(img)]
    return {
        "model": ANATOMY_VERSION,
        "image": {"width": W, "height": H},
        "inference_ms": round((time.time() - started) * 1000),
        "anatomy": structures,
    }


@app.post("/annotate")
async def annotate_route(file: UploadFile = File(...), authorization: str = Header(""),
                         layer: str = Query("findings", pattern="^(findings|anatomy|both)$")):
    from render import draw, draw_teeth

    _, img = await _read_panoramic(file, authorization)
    if layer != "findings" and teeth_model is None:
        raise HTTPException(503, "The anatomy model is not loaded on this server.")
    # Findings first: their outlines are traced from the untouched film.
    out = draw(img, predict(detector, img, DEVICE), THRESHOLD) if layer != "anatomy" else img
    if layer != "findings":
        out = draw_teeth(out, _teeth(img), footer=layer == "anatomy")
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")


@app.post("/predict")
async def predict_route(file: UploadFile = File(...), authorization: str = Header("")):
    raw, img = await _read_panoramic(file, authorization)
    W, H = img.size
    started = time.time()
    findings, periapical = [], []
    teeth = _teeth(img)
    for p in predict(detector, img, DEVICE):
        if p["score"] < THRESHOLD:
            continue
        x1, y1, x2, y2 = p["box"]
        box = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        if p["class"] in LOG_ONLY:
            periapical.append({"class": p["class"], "confidence": round(p["score"], 4), "bbox": box})
            continue
        findings.append({
            "category": WIRE_CATEGORY[p["class"]],
            "confidence": round(p["score"], 4),
            "bbox": box,
            "tooth": int(t) if (t := tooth_for(p["box"], teeth)) else None,
            "source": VERSION,
            "clinician_review_required": True,
        })
    if periapical:
        log_periapical(hashlib.sha256(raw).hexdigest(), W, H, periapical)
    return {
        "model": VERSION,
        "threshold": THRESHOLD,
        "image": {"width": W, "height": H},
        "inference_ms": round((time.time() - started) * 1000),
        "findings": findings,
    }
