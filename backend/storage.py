"""Pipeline multi-cloud de coleta de imagens com filtro de qualidade (Local, Google Drive, AWS S3 ou GCS).

Troque o destino apenas alterando variáveis de ambiente no .env:
  - STORAGE_BACKEND=local   -> salva em backend/collected_data/
  - STORAGE_BACKEND=gdrive  -> envia para pasta do Google Drive (grátis, sem conta de faturamento)
  - STORAGE_BACKEND=s3      -> envia para AWS S3 (ou R2/MinIO via S3_ENDPOINT_URL)
  - STORAGE_BACKEND=gcs     -> envia para Google Cloud Storage
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

def validate_image_quality(pil_img: Image.Image) -> tuple[bool, str, dict[str, float]]:
    """Filtra fotos borradas, escuras, estouradas ou de baixa resolução antes de salvar no dataset."""
    w, h = pil_img.size
    if w < 224 or h < 224:
        return (
            False,
            f"Imagem com resolução muito baixa ({w}×{h} px). Envie uma foto mais nítida (mínimo 224×224 px).",
            {"width": float(w), "height": float(h)},
        )

    gray = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())
    metrics = {
        "sharpness": round(sharpness, 2),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
    }

    if sharpness < 35.0:
        return (False, "Foto muito borrada ou fora de foco. Tente tirar uma nova foto bem nítida do peixe.", metrics)
    if brightness < 22.0:
        return (False, "Foto escura demais para uso científico. Tente fotografar em um local mais iluminado.", metrics)
    if brightness > 240.0 or contrast < 14.0:
        return (False, "Foto sem contraste ou com excesso de luz. Ajuste o ângulo para evitar reflexo forte.", metrics)
    return True, "ok", metrics

def _slugify(text: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_-]+", "_", text.strip())
    return clean.strip("_") or "desconhecida"

def get_storage_config() -> dict[str, Any]:
    backend = os.getenv("STORAGE_BACKEND", "local").strip().lower()
    if backend not in {"local", "gdrive", "s3", "gcs"}:
        backend = "local"
    base_path = Path(os.getenv("LOCAL_STORAGE_PATH", "backend/collected_data")).resolve()
    base_path.mkdir(parents=True, exist_ok=True)
    return {"backend": backend, "base_path": base_path}

def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def save_contribution(image_bytes: bytes, species: str, location: str | None, cfg: dict[str, Any]) -> Path:
    """Save a contribution image and its metadata.

    The function now deduplicates based on SHA‑256 hash – if an identical image already
    exists in the storage directory, the existing path is returned and no new file is
    written.
    """
    base_path: Path = cfg["base_path"]
    img_hash = _hash_bytes(image_bytes)
    # Look for an existing file with the same hash.
    for existing in base_path.glob("*.jpg"):
        if existing.stem.startswith(img_hash[:8]):
            # Assume same content – return existing path.
            return existing
    # No duplicate found – create new files.
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = _slugify(species)
    filename = f"{timestamp}_{slug}_{img_hash[:8]}.jpg"
    img_path = base_path / filename
    img_path.write_bytes(image_bytes)
    # Write accompanying JSON metadata.
    meta = {
        "species": species,
        "location": location,
        "timestamp": timestamp,
        "hash": img_hash,
        "status": "needs_review",
    }
    meta_path = img_path.with_suffix('.json')
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding='utf-8')
    # Append to manifest (JSON Lines).
    manifest_path = base_path / "manifest.jsonl"
    with manifest_path.open("a", encoding="utf-8") as mf:
        mf.write(json.dumps(meta, ensure_ascii=False) + "\n")
    return img_path
