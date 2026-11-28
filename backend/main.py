from __future__ import annotations

import hashlib
import io
import os
import time
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
from pydantic import BaseModel

from storage import (
    get_contribution_stats,
    get_storage_config,
    save_contribution,
    validate_image_quality,
)

# =============================================================================
# Configurações editáveis do Backend (API Keras - FishID Studio)
# =============================================================================
MODEL_PATH: str = os.getenv("MODEL_PATH", "models/modelo.keras").strip()
NORMALIZE_DIV_255: bool = os.getenv("NORMALIZE_DIV_255", "true").lower() == "true"
PRESERVE_ASPECT_RATIO: bool = os.getenv("PRESERVE_ASPECT_RATIO", "true").lower() == "true"
DEFAULT_INPUT_SIZE: tuple[int, int] = (320, 320)
MAX_BATCH_FILES: int = 32
MAX_BATCH_FILE_BYTES: int = 15 * 1024 * 1024  # 15 MB per file

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

_MODEL_CACHE: dict[str, Any] | None = None
_MODEL_LOAD_ERROR: str | None = None
_MODEL_CHECKED_MTIME: float | None = None


def _parse_input_size(input_shape: Any) -> tuple[int, int]:
    if isinstance(input_shape, list) and input_shape:
        input_shape = input_shape[0]
    if isinstance(input_shape, (tuple, list)) and len(input_shape) >= 3:
        h, w = input_shape[1], input_shape[2]
        if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
            return (h, w)
    return DEFAULT_INPUT_SIZE


def _parse_num_classes(output_shape: Any) -> int | None:
    if isinstance(output_shape, list) and output_shape:
        output_shape = output_shape[0]
    if isinstance(output_shape, (tuple, list)) and len(output_shape) >= 2:
        last = output_shape[-1]
        if isinstance(last, int) and last > 0:
            return last
    return None


def _display_name(scientific: str) -> str:
    common = SCIENTIFIC_TO_PT.get(scientific)
    return f"{common} ({scientific})" if common else scientific


def get_model_state() -> tuple[dict[str, Any] | None, str | None]:
    global _MODEL_CACHE, _MODEL_LOAD_ERROR, _MODEL_CHECKED_MTIME

    path = Path(MODEL_PATH)
    if not path.is_file():
        _MODEL_CACHE = None
        _MODEL_LOAD_ERROR = None
        _MODEL_CHECKED_MTIME = None
        return None, None

    mtime = path.stat().st_mtime
    if _MODEL_CACHE is not None and _MODEL_CHECKED_MTIME == mtime:
        return _MODEL_CACHE, None

    try:
        import keras

        model = keras.models.load_model(str(path), compile=False)
        input_shape = model.input_shape
        output_shape = model.output_shape
        trainable_params = int(sum(np.prod(w.shape) for w in model.trainable_weights))
        non_trainable_params = int(sum(np.prod(w.shape) for w in model.non_trainable_weights))

        _MODEL_CACHE = {
            "model": model,
            "model_name": getattr(model, "name", path.stem),
            "input_shape": list(input_shape) if isinstance(input_shape, tuple) else input_shape,
            "input_size": _parse_input_size(input_shape),
            "output_shape": list(output_shape) if isinstance(output_shape, tuple) else output_shape,
            "num_classes": _parse_num_classes(output_shape),
            "total_params": int(model.count_params()),
            "trainable_params": trainable_params,
            "non_trainable_params": non_trainable_params,
        }
        _MODEL_LOAD_ERROR = None
        _MODEL_CHECKED_MTIME = mtime
        return _MODEL_CACHE, None
    except Exception as exc:
        _MODEL_CACHE = None
        _MODEL_LOAD_ERROR = str(exc)
        _MODEL_CHECKED_MTIME = mtime
        return None, _MODEL_LOAD_ERROR


def get_effective_class_names(model_bundle: dict[str, Any] | None) -> list[str]:
    names = list(CLASS_NAMES)
    if model_bundle and model_bundle.get("num_classes"):
        n_out = int(model_bundle["num_classes"])
        if len(names) > n_out:
            return names[:n_out]
        if len(names) < n_out:
            names.extend([f"Espécie {idx + 1}" for idx in range(len(names), n_out)])
    return names


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)
    exp_vals = np.exp(shifted)
    total = np.sum(exp_vals)
    if total <= 0:
        return np.full_like(x, 1.0 / max(len(x), 1), dtype=np.float64)
    return exp_vals / total


def predict_rgb_array(
    image_rgb: np.ndarray,
    model_bundle: dict[str, Any] | None,
    hint_species: str | None = None,
) -> dict[str, Any]:
    class_names = get_effective_class_names(model_bundle)

    if model_bundle is None:
        thumb = cv2.resize(image_rgb, (16, 16), interpolation=cv2.INTER_AREA)
        digest = hashlib.sha256(thumb.tobytes()).digest()
        seed = int.from_bytes(digest[:4], "big")
        rng = np.random.default_rng(seed)

        logits = rng.normal(loc=0.0, scale=0.72, size=len(class_names))
        if hint_species and hint_species in class_names:
            winner_idx = class_names.index(hint_species)
        else:
            winner_idx = int(seed % len(class_names))
        logits[winner_idx] += float(rng.uniform(2.6, 4.0))
        probs = _softmax(logits)
        latency_ms = float(rng.uniform(18.5, 39.0))

        ordered = sorted(
            [
                {
                    "scientific": sci,
                    "common_pt": SCIENTIFIC_TO_PT.get(sci, sci),
                    "species": _display_name(sci),
                    "probability": float(prob),
                }
                for sci, prob in zip(class_names, probs)
            ],
            key=lambda item: item["probability"],
            reverse=True,
        )
        top = ordered[0]
        return {
            "mode": "demo",
            "top_species": top["common_pt"],
            "scientific_name": top["scientific"],
            "display_name": top["species"],
            "confidence": top["probability"],
            "latency_ms": round(latency_ms, 1),
            "probabilities": ordered,
        }

    model = model_bundle["model"]
    target_h, target_w = model_bundle["input_size"]

    if PRESERVE_ASPECT_RATIO:
        h, w = image_rgb.shape[:2]
        scale = min(target_w / max(w, 1), target_h / max(h, 1))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        scaled = cv2.resize(image_rgb, (new_w, new_h), interpolation=interp)
        pad_top = (target_h - new_h) // 2
        pad_bot = target_h - new_h - pad_top
        pad_left = (target_w - new_w) // 2
        pad_right = target_w - new_w - pad_left
        resized = cv2.copyMakeBorder(
            scaled, pad_top, pad_bot, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
    else:
        resized = cv2.resize(image_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    batch = resized.astype(np.float32)
    if NORMALIZE_DIV_255:
        batch = batch / 255.0
    batch = np.expand_dims(batch, axis=0)

    t0 = time.perf_counter()
    raw_output = model(batch, training=False)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    preds = np.asarray(raw_output, dtype=np.float64).reshape(-1)
    if np.any(preds < 0.0) or not np.isclose(np.sum(preds), 1.0, atol=1e-2):
        probs = _softmax(preds)
    else:
        total = float(np.sum(preds))
        probs = preds / total if total > 0 else np.full_like(preds, 1.0 / len(preds))

    n = min(len(probs), len(class_names))
    ordered = sorted(
        [
            {
                "scientific": sci,
                "common_pt": SCIENTIFIC_TO_PT.get(sci, sci),
                "species": _display_name(sci),
                "probability": float(prob),
            }
            for sci, prob in zip(class_names[:n], probs[:n])
        ],
        key=lambda item: item["probability"],
        reverse=True,
    )
    top = ordered[0]
    return {
        "mode": "real",
        "top_species": top["common_pt"],
        "scientific_name": top["scientific"],
        "display_name": top["species"],
        "confidence": top["probability"],
        "latency_ms": round(latency_ms, 1),
        "probabilities": ordered,
    }


class HealthResponse(BaseModel):
    status: Literal["ok"]


class CatalogSpecies(BaseModel):
    scientific: str
    common_pt: str
    group: str


class ModelInfoResponse(BaseModel):
    mode: Literal["demo", "real"]
    status: str
    load_error: str | None
    model_path: str
    input_size: list[int]
    total_params: int | None
    trainable_params: int | None
    normalize_div_255: bool
    preserve_aspect_ratio: bool
    species_catalog: list[CatalogSpecies]
    class_names: list[str]
    storage_backend: str
    storage_bucket: str


class SpeciesProbability(BaseModel):
    scientific: str
    common_pt: str
    species: str
    probability: float


class PredictionResponse(BaseModel):
    mode: Literal["demo", "real"]
    top_species: str
    scientific_name: str
    display_name: str
    confidence: float
    latency_ms: float
    probabilities: list[SpeciesProbability]


class BatchPredictionItem(PredictionResponse):
    filename: str


class BatchPredictionResponse(BaseModel):
    count: int
    results: list[BatchPredictionItem]


class ContributionResponse(BaseModel):
    saved: bool
    duplicate: bool = False
    backend: str
    sample_id: str
    location: str
    sha256: str = ""


class ContributionStatsResponse(BaseModel):
    total_samples: int
    confirmed: int
    needs_review: int
    by_species: dict[str, int]


def _open_rgb_image(raw_bytes: bytes) -> Image.Image:
    """Abre imagem aplicando correção de orientação EXIF antes da conversão para RGB."""
    with Image.open(io.BytesIO(raw_bytes)) as opened:
        transposed = ImageOps.exif_transpose(opened)
        return transposed.convert("RGB")


def _encode_clean_jpeg(pil_img: Image.Image, quality: int = 92) -> bytes:
    """Re-codifica somente os pixels RGB em JPEG sem tags EXIF (GPS/dispositivo)."""
    pixel_only = Image.fromarray(np.asarray(pil_img, dtype=np.uint8), mode="RGB")
    buf = io.BytesIO()
    pixel_only.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


# Document the success bodies independently of response serialization.
@app.get(
    "/health",
    tags=["System"],
    summary="Check API health",
    description="Checks whether the API is responding.",
    response_model=None,
    responses={200: {"model": HealthResponse}},
)
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/species")
def list_species(group: str | None = None, q: str | None = None) -> dict[str, Any]:
    group_filter = group.casefold() if group is not None else None
    query = (q or "").casefold()
    species = []
    for item in SPECIES_CATALOG:
        if group_filter is not None and item["group"].casefold() != group_filter:
            continue
        if query not in item["scientific"].casefold() and query not in item["common_pt"].casefold():
            continue
        species.append({**item, "display_name": _display_name(item["scientific"])})
    return {"count": len(species), "species": species}


@app.get(
    "/info",
    tags=["Model"],
    summary="Get model and catalog information",
    description="Returns the current inference mode, model configuration, species catalog, and storage settings.",
    response_model=None,
    responses={200: {"model": ModelInfoResponse}},
)
def model_info() -> dict[str, Any]:
    bundle, err = get_model_state()
    storage_cfg = get_storage_config()
    if bundle is None:
        return {
            "mode": "demo",
            "status": "Modo Demonstração (Simulado)" if not err else "Erro ao carregar (.keras)",
            "load_error": err,
            "model_path": MODEL_PATH,
            "input_size": list(DEFAULT_INPUT_SIZE),
            "total_params": None,
            "trainable_params": None,
            "normalize_div_255": NORMALIZE_DIV_255,
            "preserve_aspect_ratio": PRESERVE_ASPECT_RATIO,
            "species_catalog": SPECIES_CATALOG,
            "class_names": [_display_name(c) for c in CLASS_NAMES],
            "storage_backend": storage_cfg["backend"],
            "storage_bucket": storage_cfg["bucket"] or storage_cfg["local_dir"],
        }

    return {
        "mode": "real",
        "status": f"Modelo ativo ({bundle['model_name']})",
        "load_error": None,
        "model_path": MODEL_PATH,
        "input_size": list(bundle["input_size"]),
        "total_params": bundle["total_params"],
        "trainable_params": bundle["trainable_params"],
        "normalize_div_255": NORMALIZE_DIV_255,
        "preserve_aspect_ratio": PRESERVE_ASPECT_RATIO,
        "species_catalog": SPECIES_CATALOG,
        "class_names": [_display_name(c) for c in get_effective_class_names(bundle)],
        "storage_backend": storage_cfg["backend"],
        "storage_bucket": storage_cfg["bucket"] or storage_cfg["local_dir"],
    }


@app.post(
    "/predict",
    tags=["Inference"],
    summary="Classify a fish image",
    description="Accepts an image upload and returns the top species and ranked probabilities. Without a loaded model, the API returns a demonstration result.",
    response_model=None,
    responses={200: {"model": PredictionResponse}},
)
async def predict_endpoint(
    file: UploadFile = File(...),
    hint_species: str | None = Form(default=None),
) -> dict[str, Any]:
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Arquivo de imagem vazio.")
    try:
        pil_img = _open_rgb_image(raw_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Imagem inválida: {exc}") from exc

    bundle, _ = get_model_state()
    return predict_rgb_array(np.asarray(pil_img), bundle, hint_species=hint_species)


@app.post(
    "/predict/batch",
    tags=["Inference"],
    summary="Classify multiple fish images in a batch",
    description="Accepts up to 32 image uploads (max 15 MB per file) and returns ranked species predictions for each image.",
    response_model=None,
    responses={200: {"model": BatchPredictionResponse}},
)
async def predict_batch_endpoint(
    files: list[UploadFile] = File(...),
    hint_species: str | None = Form(default=None),
) -> dict[str, Any]:
    if not files:
        raise HTTPException(status_code=400, detail="Nenhum arquivo de imagem enviado.")
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Lote excede o máximo permitido de {MAX_BATCH_FILES} imagens.",
        )

    bundle, _ = get_model_state()
    results: list[dict[str, Any]] = []
    for idx, upload in enumerate(files):
        fname = upload.filename or f"image_{idx}"
        raw_bytes = await upload.read(MAX_BATCH_FILE_BYTES + 1)
        if len(raw_bytes) > MAX_BATCH_FILE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Arquivo '{fname}' excede o limite máximo de 15 MB.",
            )
        if not raw_bytes:
            raise HTTPException(status_code=400, detail=f"Arquivo de imagem vazio: {fname}")
        try:
            pil_img = _open_rgb_image(raw_bytes)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Imagem inválida ({fname}): {exc}") from exc
        pred = predict_rgb_array(np.asarray(pil_img), bundle, hint_species=hint_species)
        results.append({"filename": fname, **pred})

    return {"count": len(results), "results": results}


@app.post(
    "/contribute",
    tags=["Citizen Science"],
    summary="Submit a research image",
    description="Checks the uploaded image quality, normalizes orientation, strips EXIF metadata, and saves the image and submission metadata to the configured storage backend.",
    response_model=None,
    responses={200: {"model": ContributionResponse}},
)
async def contribute_endpoint(
    file: UploadFile = File(...),
    predicted_scientific: str = Form(default=""),
    predicted_common_pt: str = Form(default=""),
    confidence: float = Form(default=0.0),
    user_confirmed: bool = Form(default=False),
    user_suggested_species: str = Form(default=""),
    source: str = Form(default="upload"),
    notes: str = Form(default=""),
) -> dict[str, Any]:
    """Salva foto enviada por usuário + metadados no storage configurado (Local, AWS S3 ou GCS)."""
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Arquivo de imagem vazio.")
    try:
        pil_img = _open_rgb_image(raw_bytes)
        ok_quality, reason, quality_metrics = validate_image_quality(pil_img)
        if not ok_quality:
            raise HTTPException(status_code=422, detail=reason)
        clean_jpeg = _encode_clean_jpeg(pil_img, quality=92)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Imagem inválida: {exc}") from exc

    metadata = {
        "predicted_scientific": predicted_scientific,
        "predicted_common_pt": predicted_common_pt,
        "confidence": round(float(confidence), 6),
        "user_confirmed": bool(user_confirmed),
        "user_suggested_species": user_suggested_species.strip(),
        "source": source,
        "notes": notes.strip(),
        "image_size": {"width": pil_img.size[0], "height": pil_img.size[1]},
        "quality_metrics": quality_metrics,
        "consent_research": True,
    }
    try:
        return save_contribution(clean_jpeg, metadata)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erro ao salvar contribuição: {exc}") from exc


@app.get(
    "/contributions/stats",
    tags=["Citizen Science"],
    summary="Get citizen-science contribution statistics",
    description="Reads manifest.jsonl and returns total samples, confirmed vs. needs_review counts, and per-species counts.",
    response_model=None,
    responses={200: {"model": ContributionStatsResponse}},
)
def contributions_stats_endpoint() -> dict[str, Any]:
    return get_contribution_stats()
