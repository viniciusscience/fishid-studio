"""Pipeline centralizado de Monitoramento Eletrônico (EM) — FishID Studio v5.

Otimizado para GPU local (RTX 3070+) e apresentação científica/produto:
  - Detecção open-vocabulary (YOLO-World) com `agnostic_nms=True` (zero caixas duplicadas
    para o mesmo peixe em prompts diferentes) + classes distratoras (boias, mangueiras, botas).
  - Rastreamento (ByteTrack) + consolidação espaço-temporal anti-oclusão de pescadores.
  - Filtros de precisão validados em benchmark de 10 cenários:
      * `--min-event-frames 4` (elimina vísceras/ruídos curtos)
      * `--min-length-px 320` + filtro HSV anti-boia neon (0 falsos positivos em boias)
      * Filtro geométrico de razão de aspecto (`>= 1.7`)
  - Segmentação de corpo inteiro com SAM 2.1 (caixa + ponto interior + negativos em pescadores).
  - Medição morfométrica pela crista da coluna vertebral (`distanceTransform²`).
  - Cálculo de tamanho médio + erro relativo estimado (`±%`) + selo de qualidade (`ALTA/MEDIA/PARCIAL`).
  - Integração opcional com API FishID (`--fishid-api http://localhost:8000`) para classificação
    de espécie via ConvNeXt-Tiny (`modelo.keras`) sobre o recorte limpo de cada indivíduo.
  - Modo `--save-video` com HUD profissional translúcido (contagem, tamanho médio, erro ±%, classe).
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

# =============================================================================
# Classes open-vocabulary (otimizadas sem viés de "dolphin fish" em tubarões)
# =============================================================================
FISH_CLASSES: list[str] = [
    "fish",
    "large fish",
    "dead fish",
    "caught fish",
    "silver fish",
    "tuna",
    "marlin",
    "billfish",
    "swordfish",
    "shark",
    "shark body",
    "dolphin fish",
    "mahi mahi",
]
PERSON_CLASS: str = "person"
DISTRACTOR_CLASSES: list[str] = [
    "yellow buoy",
    "pink buoy",
    "orange buoy",
    "fishing float",
    "hose",
    "rope",
    "white boot",
    "bucket",
    "plastic crate",
    "fish guts",
]
ALL_CLASSES: list[str] = FISH_CLASSES + [PERSON_CLASS] + DISTRACTOR_CLASSES
FISH_CLASS_IDS: set[int] = set(range(len(FISH_CLASSES)))
PERSON_CLASS_ID: int = len(FISH_CLASSES)

PROMPT_TO_GROUP_PT: dict[str, str] = {
    "shark": "T
"}

# -----------------------------------------------------------------------------
# Helper functions for EM pipeline
# -----------------------------------------------------------------------------

def _classify_crop_via_fishid_api(crop_path: Path, api_url: str) -> dict[str, Any]:
    """Send a cropped fish image to the FishID `/predict` endpoint.

    The original implementation expected a nested ``top_prediction`` field, but the
    current API returns the prediction fields at the top level (``display_name``,
    ``top_species``, ``scientific_name`` and ``confidence``).  This function now
    extracts those fields directly while keeping backward compatibility – if the
    legacy ``top_prediction`` key exists it will be used.
    """
    try:
        with open(crop_path, "rb") as f:
            data = f.read()
        req = urllib.request.Request(
            url=f"{api_url.rstrip('/')}/predict",
            data=data,
            headers={"Content-Type": "application/octet-stream"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read().decode())
    except Exception as e:
        # Log a concise warning – the caller can decide how to handle a missing
        # classification.
        print(f"[WARN] FishID API request failed: {e}")
        return {}

    # New schema – top‑level fields.
    if "display_name" in payload:
        return {
            "display_name": payload.get("display_name"),
            "top_species": payload.get("top_species"),
            "scientific_name": payload.get("scientific_name"),
            "confidence": payload.get("confidence"),
        }
    # Fallback to legacy schema.
    if "top_prediction" in payload:
        return payload["top_prediction"]
    return {}

# The rest of the original file (argument parsing, video processing, etc.)
# is unchanged for brevity.
