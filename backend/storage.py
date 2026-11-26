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
import tempfile
import threading
import uuid
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


_CONTRIBUTION_LOCK = threading.RLock()


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


def _manifest_path(cfg: dict[str, Any]) -> Path:
    return Path(cfg["local_dir"]) / cfg["prefix"] / "manifest.jsonl"


@contextmanager
def _locked_manifest(manifest: Path):
    """Serialize writers/readers on one host (threads and worker processes)."""
    with _CONTRIBUTION_LOCK:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with (manifest.parent / ".manifest.lock").open("a+b") as lock:
            if os.name == "nt":
                import msvcrt

                if lock.seek(0, os.SEEK_END) == 0:
                    lock.write(b"\0")
                    lock.flush()
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if os.name == "nt":
                    lock.seek(0)
                    msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _manifest_records(manifest: Path) -> list[dict[str, Any]]:
    if not manifest.exists():
        return []
    records = []
    with manifest.open(encoding="utf-8") as source:
        for line in source:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # Older append-only manifests may contain a truncated last line.
                continue
            if (isinstance(record, dict) and isinstance(record.get("sample_id"), str)
                    and isinstance(record.get("image_key"), str)):
                records.append(record)
    return records


def _append_manifest(manifest: Path, record: dict[str, Any]) -> None:
    """Publish a complete JSONL file; a failed write leaves the old index intact."""
    previous = manifest.read_bytes() if manifest.exists() else b""
    if previous and not previous.endswith(b"\n"):
        previous += b"\n"
    row = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=manifest.parent, prefix=".manifest-", delete=False) as out:
            temporary = Path(out.name)
            out.write(previous)
            out.write(row)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, manifest)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _destination(cfg: dict[str, Any]) -> str:
    backend = cfg["backend"]
    if backend == "s3":
        parts = [backend, cfg["bucket"], cfg["prefix"], cfg["s3_endpoint_url"]]
    elif backend == "gcs":
        parts = [backend, cfg["bucket"], cfg["prefix"], cfg["gcs_project_id"]]
    else:
        parts = ["local", str((Path(cfg["local_dir"]) / cfg["prefix"]).resolve())]
    return json.dumps(parts, ensure_ascii=False)


def _target_species(metadata: dict[str, Any]) -> str:
    return str(metadata.get("user_suggested_species") or metadata.get("predicted_scientific") or "nova_especie")


def get_contribution_stats() -> dict[str, Any]:
    """Count the current destination in this installation's local manifest.

    This does not list remote buckets or contributions indexed by another host.
    Legacy rows without destination/hash remain included for compatibility;
    malformed rows are skipped. The manifest is the source of truth, not a
    check that previously saved objects still exist.
    """
    cfg = get_storage_config()
    manifest = _manifest_path(cfg)
    destination = _destination(cfg)
    stats: dict[str, Any] = {"total": 0, "confirmed": 0, "needs_review": 0, "by_species": {}}
    if not manifest.exists():
        return stats
    with _locked_manifest(manifest):
        records = _manifest_records(manifest)
    for record in records:
        if record.get("storage_destination") not in (None, destination):
            continue
        review = "confirmed" if record.get("user_confirmed") else "needs_review"
        species = _target_species(record)
        stats["total"] += 1
        stats[review] += 1
        stats["by_species"][species] = stats["by_species"].get(species, 0) + 1
    return stats


def save_contribution(image_bytes: bytes, metadata: dict[str, Any]) -> dict[str, Any]:
    """Save once per SHA256/destination; duplicates retain the first metadata.

    The local manifest indexes successful saves for every backend. Its lock
    coordinates workers sharing this index, not independent hosts. Cloud
    rollback is best effort if an upload or index write fails.
    """
    cfg = get_storage_config()
    backend = cfg["backend"] if cfg["backend"] in {"s3", "gcs"} else "local"
    if backend in {"s3", "gcs"} and not cfg["bucket"]:
        raise RuntimeError(f"STORAGE_BACKEND={backend} exige STORAGE_BUCKET definido no .env.")
    digest = hashlib.sha256(image_bytes).hexdigest()
    destination = _destination(cfg)
    manifest = _manifest_path(cfg)
    with _locked_manifest(manifest):
        for record in _manifest_records(manifest):
            if (record.get("image_sha256") == digest
                    and record.get("storage_destination") == destination
                    and isinstance(record.get("location"), str)):
                return {"saved": True, "duplicate": True, "backend": backend,
                        "sample_id": record["sample_id"], "location": record["location"]}

        sample_id = uuid.uuid4().hex[:12]
        ts = datetime.now(timezone.utc)
        review = "confirmed" if metadata.get("user_confirmed") else "needs_review"
        stem = f"{review}/{_slugify(_target_species(metadata))}/{ts:%Y%m%d}_{sample_id}"
        rel_img_key = "/".join(part for part in (cfg["prefix"], stem + ".jpg") if part)
        rel_meta_key = "/".join(part for part in (cfg["prefix"], stem + ".json") if part)
        img_path = Path(cfg["local_dir"]) / rel_img_key
        location = (f"s3://{cfg['bucket']}/{rel_img_key}" if backend == "s3" else
                    f"gs://{cfg['bucket']}/{rel_img_key}" if backend == "gcs" else str(img_path))
        # Internal identity fields always win over caller-supplied metadata.
        enriched_meta = {**metadata, "sample_id": sample_id, "timestamp_utc": ts.isoformat(),
                         "storage_backend": backend, "storage_destination": destination,
                         "image_key": rel_img_key, "image_sha256": digest, "location": location}
        meta_bytes = json.dumps(enriched_meta, ensure_ascii=False, indent=2).encode("utf-8")
        cleanup = []
        try:
            if backend == "s3":
                import boto3

                client = boto3.client("s3", region_name=cfg["aws_region"], endpoint_url=cfg["s3_endpoint_url"])
                for key, body, mime in ((rel_img_key, image_bytes, "image/jpeg"),
                                        (rel_meta_key, meta_bytes, "application/json; charset=utf-8")):
                    cleanup.append(lambda key=key: client.delete_object(Bucket=cfg["bucket"], Key=key))
                    client.put_object(Bucket=cfg["bucket"], Key=key, Body=body, ContentType=mime)
            elif backend == "gcs":
                from google.cloud import storage

                bucket = storage.Client(project=cfg["gcs_project_id"]).bucket(cfg["bucket"])
                for key, body, mime in ((rel_img_key, image_bytes, "image/jpeg"),
                                        (rel_meta_key, meta_bytes, "application/json")):
                    blob = bucket.blob(key)
                    cleanup.append(blob.delete)
                    blob.upload_from_string(body, content_type=mime)
            else:
                meta_path = Path(cfg["local_dir"]) / rel_meta_key
                img_path.parent.mkdir(parents=True, exist_ok=True)
                for path, body in ((img_path, image_bytes), (meta_path, meta_bytes)):
                    cleanup.append(lambda path=path: path.unlink(missing_ok=True))
                    path.write_bytes(body)
            _append_manifest(manifest, enriched_meta)
        except Exception:
            for undo in reversed(cleanup):
                with suppress(Exception):
                    undo()
            raise
        return {"saved": True, "duplicate": False, "backend": backend,
                "sample_id": sample_id, "location": location}
