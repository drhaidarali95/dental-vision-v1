import io, os, pathlib, gc
from typing import Any
from urllib.request import Request, urlopen

import torch
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import pil_to_tensor

MODEL_PATH = os.getenv("MODEL_PATH", "/tmp/harmony_detector_best.pth")
MODEL_ZIP_URL = os.getenv("MODEL_ZIP_URL", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
API_KEY = os.getenv("HARMONY_MODEL_API_KEY", "")
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.50"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    if not MODEL_ZIP_URL:
        raise RuntimeError("Model missing and MODEL_ZIP_URL is not configured")
    pathlib.Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    tmp = MODEL_PATH + ".download"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    req = Request(MODEL_ZIP_URL, headers=headers)
    with urlopen(req, timeout=300) as src, open(tmp, "wb") as dst:
        while chunk := src.read(1024 * 1024):
            dst.write(chunk)
    os.replace(tmp, MODEL_PATH)

ensure_model()
# Load checkpoint on CPU, construct the exact trained architecture, then release
# the duplicate checkpoint/state-dict memory before serving requests.
checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
classes = checkpoint.get("classes", ["__background__", "Cavity", "Fillings", "Implant", "Impacted Tooth"])
model_epoch = checkpoint.get("epoch")
validation_loss = checkpoint.get("val_loss")
model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(classes))
model.load_state_dict(checkpoint["model"])
del checkpoint
gc.collect()
model.to(DEVICE).eval()

app = FastAPI(title="Harmony Dental Detector", version="1.0.1")

def authorize(authorization: str | None):
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.get("/health")
def health():
    return {"ok": True, "device": str(DEVICE), "classes": classes, "model_epoch": model_epoch, "validation_loss": validation_loss}

@app.post("/predict")
async def predict(file: UploadFile = File(...), authorization: str | None = Header(default=None)) -> dict[str, Any]:
    authorize(authorization)
    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image") from exc
    tensor = pil_to_tensor(image).float().div(255.0).to(DEVICE)
    with torch.inference_mode():
        out = model([tensor])[0]
    findings = []
    for box, label, score in zip(out["boxes"].cpu(), out["labels"].cpu(), out["scores"].cpu()):
        confidence = float(score)
        if confidence < SCORE_THRESHOLD:
            continue
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        findings.append({"category": classes[int(label)], "confidence": confidence, "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}, "source": "harmony_detector_v1", "clinician_review_required": True})
    return {"model": "harmony_detector_v1", "threshold": SCORE_THRESHOLD, "image": {"width": image.width, "height": image.height}, "findings": findings, "disclaimer": "AI-assisted detection; clinician review required."}
