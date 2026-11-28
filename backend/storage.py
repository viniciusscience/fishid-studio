"""Pipeline multi-cloud de coleta de imagens com filtro de qualidade (Local, Google Drive, AWS S3 ou GCS).

Troque o destino apenas alterando variáveis de ambiente no .env:
  - STORAGE_BACKEND=local   -> salva em backend/collected_data/
  - STORAGE_BACKEND=gdrive  -> envia para pasta do Google Drive (grátis, sem conta de faturamento)
  - STORAGE_BACKEND=s3      -> envia para AWS S3 (ou R2/MinIO via S3_ENDPOINT_URL)
  - STORAGE_BACKEND=gcs     -> envia para Google Cloud Storage
"""
from __future__ import annotations

import hashlib
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
    min_dim = int(os.getenv("FISHID_MIN_IMAGE_DIM", "224"))
    min_sharpness = float(os.getenv("FISHID_MIN_SHARPNESS", "35.0"))
    min_brightness = float(os.getenv("FISHID_MIN_BRIGHTNESS", "22.0"))
    max_brightness = float(os.getenv("FISHID_MAX_BRIGHTNESS", "240.0"))
    min_contrast = float(os.getenv("FISHID_MIN_CONTRAST", "14.0"))
    w, h = pil_img.size
    if w < min_dim or h < min_dim:
        return (
            False,
            f"Imagem com resolução muito baixa ({w}×{h} px). Envie uma foto mais nítida (mínimo {min_dim}×{min_dim} px).",
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

    if sharpness < min_sharpness:
        return (
            False,
            "Foto muito borrada ou fora de foco. Tente tirar uma nova foto bem nítida do peixe.",
            metrics,
        )
    if brightness < min_brightness:
        return (
            False,
            "Foto escura demais para uso científico. Tente fotografar em um local mais iluminado.",
            metrics,
        )
    if brightness > max_brightness or contrast < min_contrast:
        return (
            False,
            "Foto sem contraste ou com excesso de luz. Ajuste o ângulo para evitar reflexo forte.",
            metrics,
        )
    return True, "ok", metrics


def _slugify(text: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_-]+", "_", text.strip())
    return clean.strip("_") or "desconhecida"


def get_storage_config() -> dict[str, Any]:
    backend = os.getenv("STORAGE_BACKEND", "local").strip().lower()
    if backend not in {"local", "gdrive", "s3", "gcs"}:
        backend = "local"
    return {
        "backend": backend,
        "bucket": os.getenv("STORAGE_BUCKET", "").strip(),
        "prefix": os.getenv("STORAGE_PREFIX", "fishid_collected").strip().strip("/"),
        "local_dir": os.getenv("LOCAL_STORAGE_DIR", "collected_data").strip(),
        "gdrive_folder_id": os.getenv("GDRIVE_UPLOAD_FOLDER_ID", "").strip(),
        "aws_region": os.getenv("AWS_REGION", "sa-east-1").strip(),
        "s3_endpoint_url": os.getenv("S3_ENDPOINT_URL", "").strip() or None,
        "gcs_project_id": os.getenv("GCS_PROJECT_ID", "").strip() or None,
    }


def _find_duplicate_in_manifest(manifest_path: Path, image_sha256: str) -> dict[str, Any] | None:
    if not manifest_path.is_file():
        return None
    with manifest_path.open("r", encoding="utf-8") as mf:
        for line in mf:
            raw = line.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and record.get("sha256") == image_sha256:
                return record
    return None


def get_contribution_stats() -> dict[str, Any]:
    """Lê manifest.jsonl e agrega total de amostras, confirmadas vs. revisão e contagem por espécie."""
    cfg = get_storage_config()
    manifest_path = Path(cfg["local_dir"]) / cfg["prefix"] / "manifest.jsonl"

    total_samples = 0
    confirmed = 0
    needs_review = 0
    species_counts: dict[str, int] = {}

    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as mf:
            for line in mf:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                total_samples += 1
                if record.get("user_confirmed"):
                    confirmed += 1
                else:
                    needs_review += 1
                species = str(
                    record.get("user_suggested_species")
                    or record.get("predicted_scientific")
                    or "nova_especie"
                ).strip() or "nova_especie"
                species_counts[species] = species_counts.get(species, 0) + 1

    ordered_species = dict(sorted(species_counts.items(), key=lambda kv: (-kv[1], kv[0])))
    return {
        "total_samples": total_samples,
        "confirmed": confirmed,
        "needs_review": needs_review,
        "by_species": ordered_species,
    }


def save_contribution(
    image_bytes: bytes,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Armazena imagem + metadados JSON no destino configurado por variável de ambiente."""
    cfg = get_storage_config()
    image_sha256 = hashlib.sha256(image_bytes).hexdigest()

    if cfg["backend"] == "local":
        base_dir = Path(cfg["local_dir"])
        manifest_path = base_dir / cfg["prefix"] / "manifest.jsonl"
        existing = _find_duplicate_in_manifest(manifest_path, image_sha256)
        if existing is not None:
            existing_key = str(existing.get("image_key", ""))
            existing_location = str(base_dir / existing_key) if existing_key else str(manifest_path)
            return {
                "saved": False,
                "duplicate": True,
                "backend": "local",
                "sample_id": str(existing.get("sample_id", "")),
                "location": existing_location,
                "sha256": image_sha256,
            }

    sample_id = uuid.uuid4().hex[:12]
    ts = datetime.now(timezone.utc)
    date_part = ts.strftime("%Y%m%d")

    target_species = (
        metadata.get("user_suggested_species")
        or metadata.get("predicted_scientific")
        or "nova_especie"
    )
    folder_species = _slugify(str(target_species))
    review_bucket = "confirmed" if metadata.get("user_confirmed") else "needs_review"

    rel_img_key = f"{cfg['prefix']}/{review_bucket}/{folder_species}/{date_part}_{sample_id}.jpg"
    rel_meta_key = f"{cfg['prefix']}/{review_bucket}/{folder_species}/{date_part}_{sample_id}.json"

    enriched_meta = {
        "sample_id": sample_id,
        "timestamp_utc": ts.isoformat(),
        "storage_backend": cfg["backend"],
        "image_key": rel_img_key,
        "sha256": image_sha256,
        **metadata,
    }
    meta_bytes = json.dumps(enriched_meta, ensure_ascii=False, indent=2).encode("utf-8")

    # 1. AWS S3 / Compatível (R2, MinIO)
    if cfg["backend"] == "s3":
        if not cfg["bucket"]:
            raise RuntimeError("STORAGE_BACKEND=s3 exige STORAGE_BUCKET definido no .env.")
        import boto3

        client = boto3.client(
            "s3",
            region_name=cfg["aws_region"],
            endpoint_url=cfg["s3_endpoint_url"],
        )
        client.put_object(
            Bucket=cfg["bucket"],
            Key=rel_img_key,
            Body=image_bytes,
            ContentType="image/jpeg",
        )
        client.put_object(
            Bucket=cfg["bucket"],
            Key=rel_meta_key,
            Body=meta_bytes,
            ContentType="application/json; charset=utf-8",
        )
        return {
            "saved": True,
            "duplicate": False,
            "backend": "s3",
            "sample_id": sample_id,
            "location": f"s3://{cfg['bucket']}/{rel_img_key}",
            "sha256": image_sha256,
        }

    # 2. Google Cloud Storage (GCS)
    if cfg["backend"] == "gcs":
        if not cfg["bucket"]:
            raise RuntimeError("STORAGE_BACKEND=gcs exige STORAGE_BUCKET definido no .env.")
        from google.cloud import storage

        client = storage.Client(project=cfg["gcs_project_id"])
        bucket = client.bucket(cfg["bucket"])
        bucket.blob(rel_img_key).upload_from_string(image_bytes, content_type="image/jpeg")
        bucket.blob(rel_meta_key).upload_from_string(meta_bytes, content_type="application/json")
        return {
            "saved": True,
            "duplicate": False,
            "backend": "gcs",
            "sample_id": sample_id,
            "location": f"gs://{cfg['bucket']}/{rel_img_key}",
            "sha256": image_sha256,
        }

    # 3. Armazenamento local (padrão para Docker / VPS)
    base_dir = Path(cfg["local_dir"])
    img_path = base_dir / rel_img_key
    meta_path = base_dir / rel_meta_key
    img_path.parent.mkdir(parents=True, exist_ok=True)
    img_path.write_bytes(image_bytes)
    meta_path.write_bytes(meta_bytes)

    manifest_path = base_dir / cfg["prefix"] / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as mf:
        mf.write(json.dumps(enriched_meta, ensure_ascii=False) + "\n")

    return {
        "saved": True,
        "duplicate": False,
        "backend": "local",
        "sample_id": sample_id,
        "location": str(img_path),
        "sha256": image_sha256,
    }
