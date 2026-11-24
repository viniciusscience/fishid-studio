from __future__ import annotations

import hashlib
import io
import json
import os
import time
from pathlib import Path
from typing import Any, List

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps

from storage import get_storage_config, save_contribution, validate_image_quality

# =============================================================================
# Configurações editáveis do Backend (API Keras - FishID Studio)
# =============================================================================
MODEL_PATH: str = os.getenv("MODEL_PATH", "models/modelo.keras").strip()
NORMALIZE_DIV_255: bool = os.getenv("NORMALIZE_DIV_255", "true").lower() == "true"
PRESERVE_ASPECT_RATIO: bool = os.getenv("PRESERVE_ASPECT_RATIO", "true").lower() == "true"
DEFAULT_INPUT_SIZE: tuple[int, int] = (320, 320)
MAX_BATCH_SIZE: int = int(os.getenv("MAX_BATCH_SIZE", "32"))
MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "15"))

# Catálogo oficial das 10 espécies em ordem alfabética (mesma ordem do class_indices.json)
SPECIES_CATALOG: list[dict[str, str]] = [
    {"scientific": "Coryphaena hippurus", "common_pt": "Dourado-do-mar", "group": "Pelágico"},
    {"scientific": "Istiophorus platypterus", "common_pt": "Agulhão-vela (Peixe-vela)", "group": "Peixe de bico"},
    {"scientific": "Kajikia albida", "common_pt": "Agulhão-branco (Marlim-branco)", "group": "Peixe de bico"},
    {"scientific": "Katsuwonus pelamis", "common_pt": "Bonito-listrado", "group": "Atum"},
    {"scientific": "Makaira nigricans", "common_pt": "Agulhão-negro (Marlim-azul)", "group": "Peixe de bico"},
    {"scientific": "Tetrapturus pfluegeri", "common_pt": "Agulhão-verde (Bico-longo)", "group": "Peixe de bico"},
    {"scientific": "Thunnus alalunga", "common_pt": "Albacora-branca", "group": "Atum"},
    {"scientific": "Thunnus albacares", "common_pt": "Albacora-laje", "group": "Atum"},
    {"scientific": "Thunnus obesus", "common_pt": "Albacora-bandolim", "group": "Atum"},
    {"scientific": "Xiphias gladius", "common_pt": "Espadarte (Meca)", "group": "Peixe de bico"},
]

CLASS_NAMES: list[str] = [item["scientific"] for item in SPECIES_CATALOG]
SCIENTIFIC_TO_PT: dict[str, str] = {item["scientific"]: item["common_pt"] for item in SPECIES_CATALOG}

app = FastAPI(
    title="FishID Studio API",
    description="Backend FastAPI + Keras para classificação de 10 espécies de grandes peixes migratórios e coleta de dados.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _load_image(file_bytes: bytes) -> Image.Image:
    """Load image bytes, apply EXIF auto‑orientation and convert to RGB."""
    pil_img = Image.open(io.BytesIO(file_bytes))
    pil_img = ImageOps.exif_transpose(pil_img)
    pil_img = pil_img.convert("RGB")
    return pil_img

def _image_to_array(pil_img: Image.Image) -> np.ndarray:
    arr = np.asarray(pil_img)
    if PRESERVE_ASPECT_RATIO:
        # Simple resize preserving aspect ratio – pad with black if needed.
        h, w = arr.shape[:2]
        target_h, target_w = DEFAULT_INPUT_SIZE
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(arr, (new_w, new_h))
        canvas = np.zeros((target_h, target_w, 3), dtype=resized.dtype)
        y0 = (target_h - new_h) // 2
        x0 = (target_w - new_w) // 2
        canvas[y0:y0+new_h, x0:x0+new_w] = resized
        arr = canvas
    else:
        arr = cv2.resize(arr, DEFAULT_INPUT_SIZE)
    if NORMALIZE_DIV_255:
        arr = arr.astype("float32") / 255.0
    return arr

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.get("/info")
def info() -> dict[str, Any]:
    return {
        "model_path": MODEL_PATH,
        "species_catalog": SPECIES_CATALOG,
        "demo_mode": os.getenv("DEMO_MODE", "true").lower() == "true",
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...), hint_species: str | None = Form(None)) -> dict[str, Any]:
    content = await file.read()
    pil_img = _load_image(content)
    arr = _image_to_array(pil_img)
    # Dummy model inference – deterministic demo probabilities.
    probs = np.full(len(CLASS_NAMES), 1 / len(CLASS_NAMES), dtype=float)
    top_idx = int(np.argmax(probs))
    response = {
        "display_name": CLASS_NAMES[top_idx],
        "top_species": CLASS_NAMES[top_idx],
        "scientific_name": CLASS_NAMES[top_idx],
        "confidence": float(probs[top_idx]),
        "probabilities": {name: float(p) for name, p in zip(CLASS_NAMES, probs)},
    }
    if hint_species:
        response["hint_species"] = hint_species
    return response

@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)) -> list[dict[str, Any]]:
    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=400, detail=f"Batch size exceeds maximum of {MAX_BATCH_SIZE}")
    total_size_mb = sum(len(await f.read()) for f in files) / (1024 * 1024)
    if total_size_mb > MAX_UPLOAD_MB:
        raise HTTPException(status_code=413, detail="Payload exceeds maximum allowed size")
    # Reset file pointers after size check.
    for f in files:
        await f.seek(0)
    results: list[dict[str, Any]] = []
    for f in files:
        content = await f.read()
        pil_img = _load_image(content)
        arr = _image_to_array(pil_img)
        probs = np.full(len(CLASS_NAMES), 1 / len(CLASS_NAMES), dtype=float)
        top_idx = int(np.argmax(probs))
        results.append({
            "filename": f.filename,
            "display_name": CLASS_NAMES[top_idx],
            "confidence": float(probs[top_idx]),
        })
    return results

@app.post("/contribute")
async def contribute(
    file: UploadFile = File(...),
    species: str = Form(...),
    location: str = Form(None),
) -> dict[str, Any]:
    content = await file.read()
    pil_img = _load_image(content)
    # Validate quality before storing.
    ok, msg, metrics = validate_image_quality(pil_img)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    # Re‑encode JPEG without EXIF.
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", exif=b"")
    img_bytes = buf.getvalue()
    storage_cfg = get_storage_config()
    saved_path = save_contribution(img_bytes, species, location, storage_cfg)
    return {"saved_path": str(saved_path), "quality": metrics}

@app.get("/contributions/stats")
def contributions_stats() -> dict[str, Any]:
    cfg = get_storage_config()
    manifest_path = Path(cfg["base_path"]).joinpath("manifest.jsonl")
    if not manifest_path.is_file():
        return {"total_samples": 0}
    total = 0
    confirmed = 0
    needs_review = 0
    by_species: dict[str, int] = {}
    latest_ts = None
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            status = entry.get("status", "needs_review")
            if status == "confirmed":
                confirmed += 1
            else:
                needs_review += 1
            sp = entry.get("species", "unknown")
            by_species[sp] = by_species.get(sp, 0) + 1
            ts = entry.get("timestamp")
            if ts:
                latest_ts = max(latest_ts or ts, ts)
    return {
        "total_samples": total,
        "confirmed_count": confirmed,
        "needs_review_count": needs_review,
        "by_species": by_species,
        "latest_timestamp_utc": latest_ts,
    }
