"""Modal deployment for the MONAI panoramic detector.

Runs serve_monai.py on a T4 GPU. The checkpoint is copied byte-for-byte into
a PRIVATE Modal volume. Only /health and a bearer-protected /predict are
reachable over HTTPS. Periapical-lesion records go to the same volume.

One-time setup (on your own computer, in this panoramic_monai/ folder):

  modal volume create harmony-pano
  modal volume put harmony-pano best.pt /best.pt
  modal secret create harmony-pano-key HARMONY_KEY=<long random key>
  modal deploy modal_serve.py

Read the periapical log later with:

  modal volume get harmony-pano /logs/periapical_findings.jsonl .
"""
from pathlib import Path

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(Path(__file__).with_name("requirements.txt")))
    .env({
        "HARMONY_CHECKPOINT": "/model/best.pt",
        "MODEL_VERSION": "harmony_pano_monai_v1",
        "MIN_CONF": "0.5",
        "PERIAPICAL_LOG": "/model/logs/periapical_findings.jsonl",
    })
    .add_local_python_source("serve_monai", "model", "labels", "render")
)

volume = modal.Volume.from_name("harmony-pano")
app = modal.App("harmony-pano-detector", image=image)


@app.function(
    gpu="T4",
    memory=6144,
    volumes={"/model": volume},
    secrets=[modal.Secret.from_name("harmony-pano-key")],
    scaledown_window=300,
    timeout=120,
)
@modal.asgi_app()
def web():
    import os

    os.makedirs("/model/logs", exist_ok=True)
    from serve_monai import app as fastapi_app

    @fastapi_app.on_event("shutdown")
    def _flush_log():
        volume.commit()

    return fastapi_app
