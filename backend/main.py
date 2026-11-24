from __future__ import annotations

import hashlib
import io
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
from starlette.concurrency import run_in_threadpool

from storage import get_contribution_stats, get_storage_config, save_contribution, validate_image_quality

# =============================================================================
# Configurações editáveis do Backend (API Keras - FishID Studio)
# =============================================================================
MODEL_PATH: str = os.getenv("MODEL_PATH", "models/modelo.keras").strip()
NORMALIZE_DIV_255: bool = os.getenv("NORMALIZE_DIV_255", "true").lower() == "true"
PRESERVE_ASPECT_RATIO: bool = os.getenv("PRESERVE_ASPECT_RATIO", "true").lower() == "true"
DEFAULT_INPUT_SIZE: tuple[int, int] = (320, 320)
MAX_BATCH_FILES = 32
MAX_BATCH_FILE_BYTES = 15 * 1024 * 1024

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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/info")
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


async def _read_image(file: UploadFile, max_bytes: int = -1) -> Image.Image:
    raw_bytes = await file.read(max_bytes + 1 if max_bytes >= 0 else -1)
    if 0 <= max_bytes < len(raw_bytes):
        raise HTTPException(status_code=413, detail="Cada arquivo pode ter no máximo 15 MiB.")
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Arquivo de imagem vazio.")
    try:
        with Image.open(io.BytesIO(raw_bytes)) as image:
            return ImageOps.exif_transpose(image).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Imagem inválida: {exc}") from exc


@app.post("/predict")
async def predict_endpoint(
    file: UploadFile = File(...),
    hint_species: str | None = Form(default=None),
) -> dict[str, Any]:
    pil_img = await _read_image(file)
    bundle, _ = get_model_state()
    return predict_rgb_array(np.asarray(pil_img), bundle, hint_species=hint_species)


@app.post("/predict/batch")
async def predict_batch_endpoint(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    """Classifica os recortes na ordem em que foram enviados."""
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(status_code=413, detail="O lote pode ter no máximo 32 arquivos.")
    if any(file.size is not None and file.size > MAX_BATCH_FILE_BYTES for file in files):
        raise HTTPException(status_code=413, detail="Cada arquivo pode ter no máximo 15 MiB.")

    bundle, _ = get_model_state()
    predictions = []
    for file in files:
        pil_img = await _read_image(file, MAX_BATCH_FILE_BYTES)
        predictions.append(predict_rgb_array(np.asarray(pil_img), bundle))
    return {"predictions": predictions}


@app.post("/contribute")
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
    pil_img = await _read_image(file)
    ok_quality, reason, quality_metrics = validate_image_quality(pil_img)
    if not ok_quality:
        raise HTTPException(status_code=422, detail=reason)
    pil_img.info.clear()
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=92)

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
        return await run_in_threadpool(save_contribution, buf.getvalue(), metadata)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erro ao salvar contribuição: {exc}") from exc


@app.get("/contributions/stats")
def contributions_stats() -> dict[str, Any]:
    return get_contribution_stats()
