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
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import ExitStack, suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from filelock import FileLock
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
        return (
            False,
            "Foto muito borrada ou fora de foco. Tente tirar uma nova foto bem nítida do peixe.",
            metrics,
        )
    if brightness < 22.0:
        return (
            False,
            "Foto escura demais para uso científico. Tente fotografar em um local mais iluminado.",
            metrics,
        )
    if brightness > 240.0 or contrast < 14.0:
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


def _target_species(metadata: dict[str, Any]) -> str:
    return str(metadata.get("user_suggested_species") or metadata.get("predicted_scientific") or "nova_especie")


def _manifest_rows(path: Path) -> Iterator[dict[str, Any]]:
    try:
        manifest = path.open("rb")
    except FileNotFoundError:
        return
    with manifest:
        for line in manifest:
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if isinstance(entry, dict):
                yield entry


def _append_manifest(path: Path, entry: dict[str, Any]) -> None:
    with path.open("a+b") as manifest:
        offset = manifest.tell()
        separator = b""
        if offset:
            manifest.seek(-1, os.SEEK_END)
            if manifest.read(1) != b"\n":
                separator = b"\n"
        row = json.dumps(entry, ensure_ascii=False).encode("utf-8")
        try:
            manifest.write(separator + row + b"\n")
            manifest.flush()
        except OSError:
            manifest.truncate(offset)
            raise


def _discard_cloud_object(delete: Callable[..., Any], **kwargs: Any) -> None:
    # O rollback em nuvem depende da permissão de exclusão do bucket.
    with suppress(Exception):
        delete(**kwargs)


def get_contribution_stats() -> dict[str, Any]:
    cfg = get_storage_config()
    manifest_path = Path(cfg["local_dir"]) / cfg["prefix"] / "manifest.jsonl"
    confirmed = needs_review = 0
    by_species: Counter[str] = Counter()
    for entry in _manifest_rows(manifest_path):
        by_species[_target_species(entry)] += 1
        if entry.get("user_confirmed"):
            confirmed += 1
        else:
            needs_review += 1
    return {
        "total_samples": confirmed + needs_review,
        "confirmed": confirmed,
        "needs_review": needs_review,
        "by_species": dict(by_species),
    }


def save_contribution(image_bytes: bytes, metadata: dict[str, Any]) -> dict[str, Any]:
    """Armazena uma única cópia de cada JPEG e mantém um manifesto local."""
    cfg = get_storage_config()
    base_dir = Path(cfg["local_dir"])
    manifest_path = base_dir / cfg["prefix"] / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    with FileLock(f"{manifest_path}.lock"):
        for entry in _manifest_rows(manifest_path):
            stored_hash = entry.get("sha256")
            # Contribuições locais antigas registram o caminho, mas não o hash.
            if not stored_hash and entry.get("image_key"):
                try:
                    stored_hash = hashlib.sha256((base_dir / entry["image_key"]).read_bytes()).hexdigest()
                except FileNotFoundError:
                    continue
            if stored_hash == image_hash:
                return {
                    "saved": True,
                    "duplicate": True,
                    "backend": entry.get("storage_backend", "local") if entry.get("location") else "local",
                    "sample_id": entry["sample_id"],
                    "location": entry.get("location") or str(base_dir / entry["image_key"]),
                }

        with ExitStack() as cleanup:
            entry = _store_contribution(cfg, image_bytes, {**metadata, "sha256": image_hash}, cleanup)
            _append_manifest(manifest_path, entry)
            cleanup.pop_all()
        return {
            "saved": True,
            "backend": entry["storage_backend"],
            "sample_id": entry["sample_id"],
            "location": entry["location"],
        }


def _store_contribution(
    cfg: dict[str, Any],
    image_bytes: bytes,
    metadata: dict[str, Any],
    cleanup: ExitStack,
) -> dict[str, Any]:
    sample_id = uuid.uuid4().hex[:12]
    ts = datetime.now(timezone.utc)
    date_part = ts.strftime("%Y%m%d")

    folder_species = _slugify(_target_species(metadata))
    review_bucket = "confirmed" if metadata.get("user_confirmed") else "needs_review"

    prefix = f"{cfg['prefix']}/" if cfg["prefix"] else ""
    sample_key = f"{prefix}{review_bucket}/{folder_species}/{date_part}_{sample_id}"
    rel_img_key, rel_meta_key = f"{sample_key}.jpg", f"{sample_key}.json"
    backend = cfg["backend"] if cfg["backend"] in {"s3", "gcs"} else "local"
    base_dir = Path(cfg["local_dir"])
    img_path = base_dir / rel_img_key
    scheme = "gs" if backend == "gcs" else backend
    location = str(img_path) if backend == "local" else f"{scheme}://{cfg['bucket']}/{rel_img_key}"

    enriched_meta = {
        **metadata,
        "sample_id": sample_id,
        "timestamp_utc": ts.isoformat(),
        "storage_backend": backend,
        "image_key": rel_img_key,
        "location": location,
    }
    meta_bytes = json.dumps(enriched_meta, ensure_ascii=False, indent=2).encode("utf-8")

    # 1. AWS S3 / Compatível (R2, MinIO)
    if backend == "s3":
        if not cfg["bucket"]:
            raise RuntimeError("STORAGE_BACKEND=s3 exige STORAGE_BUCKET definido no .env.")
        import boto3

        client = boto3.client(
            "s3",
            region_name=cfg["aws_region"],
            endpoint_url=cfg["s3_endpoint_url"],
        )
        cleanup.callback(_discard_cloud_object, client.delete_object, Bucket=cfg["bucket"], Key=rel_img_key)
        client.put_object(
            Bucket=cfg["bucket"],
            Key=rel_img_key,
            Body=image_bytes,
            ContentType="image/jpeg",
        )
        cleanup.callback(_discard_cloud_object, client.delete_object, Bucket=cfg["bucket"], Key=rel_meta_key)
        client.put_object(
            Bucket=cfg["bucket"],
            Key=rel_meta_key,
            Body=meta_bytes,
            ContentType="application/json; charset=utf-8",
        )

    # 2. Google Cloud Storage (GCS)
    elif backend == "gcs":
        if not cfg["bucket"]:
            raise RuntimeError("STORAGE_BACKEND=gcs exige STORAGE_BUCKET definido no .env.")
        from google.cloud import storage

        client = storage.Client(project=cfg["gcs_project_id"])
        bucket = client.bucket(cfg["bucket"])
        image_blob = bucket.blob(rel_img_key)
        metadata_blob = bucket.blob(rel_meta_key)
        cleanup.callback(_discard_cloud_object, image_blob.delete)
        image_blob.upload_from_string(image_bytes, content_type="image/jpeg")
        cleanup.callback(_discard_cloud_object, metadata_blob.delete)
        metadata_blob.upload_from_string(meta_bytes, content_type="application/json")

    # 3. Armazenamento local (padrão para Docker / VPS)
    else:
        img_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path = base_dir / rel_meta_key
        cleanup.callback(img_path.unlink, missing_ok=True)
        img_path.write_bytes(image_bytes)
        cleanup.callback(meta_path.unlink, missing_ok=True)
        meta_path.write_bytes(meta_bytes)
    return enriched_meta
