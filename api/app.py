"""
app.py
------
FastAPI inference service for the Cats vs Dogs classifier.

Endpoints:
    GET  /health        - Liveness check (M2)
    POST /predict       - Image upload → label + confidence (M2)
    GET  /metrics       - Request counts and latency stats (M5)
    GET  /performance   - Latest batch accuracy from simulate_requests (M5)

Run locally:
    uvicorn api.app:app --reload --port 8000
"""

import os
import sys
import time
import json
import logging
from io import BytesIO

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from src.predict import run_inference

# Allow importing from src/ when running inside the container (src is on PYTHONPATH)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── In-memory counters (M5 basic monitoring) ───────────────────────────────────
_stats = {
    "total_requests": 0,
    "cat_count": 0,
    "dog_count": 0,
    "total_latency_ms": 0.0,
    "errors": 0,
}

# Path where simulate_requests.py writes its results
PERFORMANCE_LOG = os.path.join(os.path.dirname(__file__), "..", "performance_log.json")

MODEL_PATH = os.environ.get("MODEL_PATH", "model.h5")

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Cats vs Dogs Classifier",
    description="Binary image classification API — MLOps Assignment 2",
    version="1.0.0",
)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", summary="Health check")
def health():
    """Returns service status. Used by Kubernetes liveness probe and smoke tests."""
    return {"status": "ok", "model_path": MODEL_PATH}


@app.post("/predict", summary="Classify an image as cat or dog")
async def predict(file: UploadFile = File(...)):
    """
    Accepts a JPEG/PNG image file and returns:
    - label       : "cat" or "dog"
    - confidence  : probability of the predicted class
    - probabilities: {"cat": float, "dog": float}
    """
    _stats["total_requests"] += 1
    start = time.time()

    # Validate file type
    if file.content_type not in ("image/jpeg", "image/png"):
        _stats["errors"] += 1
        raise HTTPException(status_code=400, detail="Only JPEG and PNG images are supported.")

    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
    except Exception:
        _stats["errors"] += 1
        raise HTTPException(status_code=400, detail="Could not read the uploaded image.")

    try:
        result = run_inference(image, model_path=MODEL_PATH)
    except FileNotFoundError as e:
        _stats["errors"] += 1
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        _stats["errors"] += 1
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail="Inference failed.")

    elapsed_ms = (time.time() - start) * 1000
    _stats["total_latency_ms"] += elapsed_ms

    if result["label"] == "cat":
        _stats["cat_count"] += 1
    else:
        _stats["dog_count"] += 1

    logger.info(
        f"predict | file={file.filename} | label={result['label']} "
        f"| confidence={result['confidence']} | latency={elapsed_ms:.1f}ms"
    )

    return JSONResponse(content=result)


@app.get("/metrics", summary="Basic request metrics")
def metrics():
    """
    Returns in-app counters:
    - total_requests, cat_count, dog_count, errors
    - avg_latency_ms
    """
    total = _stats["total_requests"]
    avg_latency = (
        round(_stats["total_latency_ms"] / total, 2) if total > 0 else 0.0
    )
    return {
        "total_requests": total,
        "cat_count": _stats["cat_count"],
        "dog_count": _stats["dog_count"],
        "errors": _stats["errors"],
        "avg_latency_ms": avg_latency,
    }


@app.get("/performance", summary="Latest batch performance from simulate_requests")
def performance():
    """
    Returns the most recent post-deployment performance log written by
    scripts/simulate_requests.py. Returns 404 if no log exists yet.
    """
    if not os.path.exists(PERFORMANCE_LOG):
        raise HTTPException(
            status_code=404,
            detail="No performance log found. Run scripts/simulate_requests.py first.",
        )
    with open(PERFORMANCE_LOG) as f:
        data = json.load(f)
    return data
