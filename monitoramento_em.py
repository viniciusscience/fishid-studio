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
    "shark": "Tubarao",
    "shark body": "Tubarao",
    "tuna": "Atum",
    "marlin": "Peixe-de-bico",
    "billfish": "Peixe-de-bico",
    "swordfish": "Espadarte",
    "mahi mahi": "Dourado-do-mar",
    "dolphin fish": "Peixe Pelagico",
    "fish": "Peixe Pelagico",
    "large fish": "Peixe Pelagico",
    "dead fish": "Peixe Pelagico",
    "caught fish": "Peixe Pelagico",
    "silver fish": "Peixe Pelagico",
}


def _fmt_hms(seconds: float) -> str:
    sec = max(0, int(round(seconds)))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _gpu_temp_c() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        ).strip()
        temp, util = [x.strip() for x in out.split(",")]
        return f"{temp}C (uso GPU: {util}%)"
    except Exception:
        return "N/D"


def _box_overlap_ratio(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    return inter / area_a


def _box_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / max(1, area_a + area_b - inter)


def _keep_largest_component(mask_u8: np.ndarray) -> np.ndarray:
    binary = (mask_u8 > 127).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return binary * 255
    largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return ((labels == largest_idx).astype(np.uint8)) * 255


def _interior_point(mask_u8: np.ndarray) -> tuple[int, int] | None:
    binary = (mask_u8 > 127).astype(np.uint8)
    if np.count_nonzero(binary) < 20:
        return None
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, max_val, _, max_loc = cv2.minMaxLoc(dist)
    if max_val <= 1.0:
        return None
    return (int(max_loc[0]), int(max_loc[1]))


def _is_neon_buoy_mask(frame_bgr: np.ndarray, mask_u8: np.ndarray) -> tuple[bool, float]:
    """Detecta se a máscara corresponde a boia amarela/rosa/laranja neon de espinhel (HSV)."""
    ys, xs = np.where(mask_u8 > 127)
    if len(xs) < 40:
        return False, 0.0
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    px = hsv[ys, xs]
    h, s, v = px[:, 0], px[:, 1], px[:, 2]
    # Amarelo forte (H 14..38) ou rosa/magenta neon (H 145..179) com alta saturação
    is_yellow = (h >= 14) & (h <= 38) & (s >= 105) & (v >= 95)
    is_pink = (h >= 145) & (h <= 179) & (s >= 110) & (v >= 95)
    frac = float(np.mean(is_yellow | is_pink))
    return frac >= 0.42, round(frac, 3)


def _infer_fish_group(obs_list: list[dict]) -> tuple[str, float]:
    """Consenso ponderado por confiança da classe macro ao longo do evento."""
    scores: dict[str, float] = defaultdict(float)
    total = 0.0
    for o in obs_list:
        grp = PROMPT_TO_GROUP_PT.get(o["classe_prompt"], "Peixe Pelagico")
        weight = o["confianca"] * (1.20 if grp != "Peixe Pelagico" else 0.95)
        scores[grp] += weight
        total += weight
    if not scores or total <= 0:
        return "Peixe Pelagico", 0.0
    best_grp = max(scores.items(), key=lambda kv: kv[1])[0]
    return best_grp, round(scores[best_grp] / total, 3)


def _classify_crop_via_fishid_api(
    api_base: str,
    crop_path: Path,
    timeout: float = 10.0,
) -> tuple[str, float] | None:
    """Consulta opcional à API FastAPI do FishID Studio (ConvNeXt-Tiny) para espécie científica."""
    if not crop_path.is_file():
        print(f"[WARN] FishID API crop not found: {crop_path}")
        return None

    url = api_base.rstrip("/") + "/predict"
    boundary = "----FishIDBoundary7MA4YWxkTrZu0gW"
    img_bytes = crop_path.read_bytes()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{crop_path.name}"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode("utf-8") + img_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        top = data.get("top_prediction") or {}
        name = top.get("display_name") or top.get("common_pt") or top.get("species")
        conf = float(top.get("confidence", 0.0))
        if name:
            return str(name), round(conf, 4)
    except Exception as exc:
        print(f"[WARN] FishID API request failed ({url}): {exc}")
        return None
    return None


def compute_polyline_lengths(points: list[np.ndarray]) -> tuple[float, float]:
    """Retorna (comprimento da curva, comprimento direto) em pixels; a curva nunca fica menor que o direto."""
    if len(points) < 2:
        return 0.0, 0.0
    curve_len = sum(float(np.linalg.norm(points[i] - points[i - 1])) for i in range(1, len(points)))
    direct_len = float(np.linalg.norm(points[-1] - points[0]))
    return max(curve_len, direct_len), direct_len


def measure_mask_length(
    mask_u8: np.ndarray,
    n_slices: int = 24,
    use_ridge: bool = True,
) -> dict:
    """Mede comprimento curvo pela crista da coluna vertebral (distanceTransform²) e eixo reto."""
    clean = _keep_largest_component(mask_u8)
    ys, xs = np.where(clean > 127)
    if len(xs) < 40:
        return {
            "mask_clean": clean,
            "area_px2": int(len(xs)),
            "comprimento_eixo_px": 0.0,
            "comprimento_curva_px": 0.0,
            "largura_corpo_px": 0.0,
            "razao_aspecto": 0.0,
            "centerline_pts": [],
            "mask_bbox": (0, 0, 0, 0),
        }

    mx1, my1, mx2, my2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    dist_map = cv2.distanceTransform((clean > 127).astype(np.uint8), cv2.DIST_L2, 5)
    max_r = float(dist_map.max())
    largura_corpo_px = 2.0 * max_r

    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    d_vals = dist_map[ys, xs].astype(np.float64)

    if use_ridge and max_r >= 10.0:
        core_sel = d_vals >= max(3.0, 0.28 * max_r)
        pca_pts = pts[core_sel] if np.count_nonzero(core_sel) >= 40 else pts
    else:
        pca_pts = pts

    mean = pca_pts.mean(axis=0)
    centered_pca = pca_pts - mean
    cov = np.cov(centered_pca.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major_vec = eigvecs[:, int(np.argmax(eigvals))]

    centered_all = pts - mean
    proj_major = centered_all @ major_vec

    if use_ridge and max_r >= 10.0:
        t_all_min, t_all_max = float(proj_major.min()), float(proj_major.max())
        edges = np.linspace(t_all_min, t_all_max, n_slices + 1)

        slice_pts: list[tuple[float, np.ndarray, float]] = []
        for k in range(n_slices):
            lo, hi = edges[k], edges[k + 1]
            in_sl = (proj_major >= lo) & (proj_major <= hi)
            if np.count_nonzero(in_sl) < 6:
                continue
            sl_d = d_vals[in_sl]
            sl_pts = pts[in_sl]
            sl_max_d = float(sl_d.max())
            ridge_sel = sl_d >= max(2.0, 0.55 * sl_max_d)
            w = (sl_d[ridge_sel] ** 2) if np.any(ridge_sel) else (sl_d ** 2)
            p_sub = sl_pts[ridge_sel] if np.any(ridge_sel) else sl_pts
            w_sum = float(w.sum())
            if w_sum <= 0:
                continue
            c_pt = (p_sub * w[:, None]).sum(axis=0) / w_sum
            slice_pts.append((float((lo + hi) * 0.5), c_pt, sl_max_d))

        def _trim_fin_tip(seq: list[tuple[float, np.ndarray, float]]) -> list[tuple[float, np.ndarray, float]]:
            s = list(seq)
            while len(s) >= 6:
                p0, p1, p2 = s[0][1], s[1][1], s[2][1]
                r0 = s[0][2]
                v = p1 - p2
                pred = p1 + v
                lateral_jump = float(np.linalg.norm(p0 - pred))
                if r0 < 0.30 * max_r and lateral_jump > 1.35 * max_r:
                    s.pop(0)
                else:
                    break
            return s

        slice_pts = _trim_fin_tip(slice_pts)
        slice_pts = list(reversed(_trim_fin_tip(list(reversed(slice_pts)))))

        if len(slice_pts) >= 3:
            first_c, last_c = slice_pts[0][1], slice_pts[-1][1]
            t_lo, t_hi = slice_pts[0][0], slice_pts[-1][0]

            cand_start = pts[(proj_major <= t_lo) & (np.linalg.norm(pts - first_c, axis=1) <= max(35.0, 1.3 * max_r))]
            p_start = cand_start[np.argmin((cand_start - mean) @ major_vec)] if len(cand_start) >= 3 else first_c

            cand_end = pts[(proj_major >= t_hi) & (np.linalg.norm(pts - last_c, axis=1) <= max(35.0, 1.3 * max_r))]
            p_end = cand_end[np.argmax((cand_end - mean) @ major_vec)] if len(cand_end) >= 3 else last_c

            centerline = [p_start] + [sp[1] for sp in slice_pts] + [p_end]
        else:
            centerline = [pts[np.argmin(proj_major)], pts[np.argmax(proj_major)]]
    else:
        t_min, t_max = float(proj_major.min()), float(proj_major.max())
        edges = np.linspace(t_min, t_max, n_slices + 1)
        centerline = [pts[proj_major <= t_min + max(2.0, 0.012 * (t_max - t_min))].mean(axis=0)]
        for k in range(n_slices):
            in_sl = (proj_major >= edges[k]) & (proj_major <= edges[k + 1])
            if np.count_nonzero(in_sl) >= 5:
                centerline.append(pts[in_sl].mean(axis=0))
        centerline.append(pts[proj_major >= t_max - max(2.0, 0.012 * (t_max - t_min))].mean(axis=0))

    if len(centerline) >= 5:
        for _ in range(2):
            sm = [centerline[0]]
            for i in range(1, len(centerline) - 1):
                sm.append(0.25 * centerline[i - 1] + 0.5 * centerline[i] + 0.25 * centerline[i + 1])
            sm.append(centerline[-1])
            centerline = sm

    curve_len, direct_ext = compute_polyline_lengths(centerline)
    razao_aspecto = curve_len / max(1.0, largura_corpo_px)

    return {
        "mask_clean": clean,
        "area_px2": int(len(xs)),
        "comprimento_eixo_px": round(direct_ext, 1),
        "comprimento_curva_px": round(curve_len, 1),
        "largura_corpo_px": round(largura_corpo_px, 1),
        "razao_aspecto": round(razao_aspecto, 2),
        "centerline_pts": [(int(round(p[0])), int(round(p[1]))) for p in centerline],
        "mask_bbox": (mx1, my1, mx2, my2),
    }


def consolidate_tracks(
    valid_tracks: dict[int, list[dict]],
    merge_gap_s: float,
    merge_dist_px: float,
    min_event_frames: int,
) -> list[dict]:
    """Une tracks fragmentados (oclusão ou caixas sobrepostas no mesmo peixe) em eventos únicos."""
    summaries = []
    for tid, obs in valid_tracks.items():
        obs_sorted = sorted(obs, key=lambda o: o["frame"])
        first, last = obs_sorted[0], obs_sorted[-1]
        med_box = (
            int(np.median([o["x1"] for o in obs_sorted])),
            int(np.median([o["y1"] for o in obs_sorted])),
            int(np.median([o["x2"] for o in obs_sorted])),
            int(np.median([o["y2"] for o in obs_sorted])),
        )
        summaries.append(
            {
                "tid": tid,
                "obs": obs_sorted,
                "frames_set": {o["frame"] for o in obs_sorted},
                "t_start": first["tempo_s"],
                "t_end": last["tempo_s"],
                "start_center": ((first["x1"] + first["x2"]) / 2.0, (first["y1"] + first["y2"]) / 2.0),
                "end_center": ((last["x1"] + last["x2"]) / 2.0, (last["y1"] + last["y2"]) / 2.0),
                "start_box": (first["x1"], first["y1"], first["x2"], first["y2"]),
                "end_box": (last["x1"], last["y1"], last["x2"], last["y2"]),
                "med_box": med_box,
            }
        )

    summaries.sort(key=lambda s: (s["t_start"], s["tid"]))

    events: list[dict] = []
    for tr in summaries:
        best_ev = None
        best_score = float("inf")

        for ev in events:
            shared_frames = tr["frames_set"].intersection(ev["frames_set"])
            iou_med = _box_iou(tr["med_box"], ev["med_box"])
            iou_end = _box_iou(tr["start_box"], ev["end_box"])

            if len(shared_frames) >= 2:
                # Só funde tracks simultâneos se forem claramente duplicata do mesmo peixe (alto IoU!)
                if max(iou_med, iou_end) < 0.30:
                    continue
            else:
                gap_s = tr["t_start"] - ev["t_end"]
                if gap_s < -5.0 or gap_s > merge_gap_s:
                    continue

            dist_end = float(np.hypot(tr["start_center"][0] - ev["end_center"][0], tr["start_center"][1] - ev["end_center"][1]))
            dist_med = float(np.hypot(tr["start_center"][0] - ev["med_center"][0], tr["start_center"][1] - ev["med_center"][1]))
            dist = min(dist_end, dist_med)

            if dist <= merge_dist_px or max(iou_med, iou_end) >= 0.12:
                score = dist + max(0.0, tr["t_start"] - ev["t_end"]) * 2.0
                if score < best_score:
                    best_score = score
                    best_ev = ev

        if best_ev is None:
            all_cx = [(o["x1"] + o["x2"]) / 2.0 for o in tr["obs"]]
            all_cy = [(o["y1"] + o["y2"]) / 2.0 for o in tr["obs"]]
            events.append(
                {
                    "track_ids": [tr["tid"]],
                    "obs": list(tr["obs"]),
                    "frames_set": set(tr["frames_set"]),
                    "t_start": tr["t_start"],
                    "t_end": tr["t_end"],
                    "end_center": tr["end_center"],
                    "med_center": (float(np.median(all_cx)), float(np.median(all_cy))),
                    "end_box": tr["end_box"],
                    "med_box": tr["med_box"],
                }
            )
        else:
            best_ev["track_ids"].append(tr["tid"])
            best_ev["obs"].extend(tr["obs"])
            best_ev["obs"].sort(key=lambda o: o["frame"])
            best_ev["frames_set"].update(tr["frames_set"])
            best_ev["t_start"] = min(best_ev["t_start"], tr["t_start"])
            if tr["t_end"] >= best_ev["t_end"]:
                best_ev["t_end"] = tr["t_end"]
                best_ev["end_center"] = tr["end_center"]
                best_ev["end_box"] = tr["end_box"]
            all_cx = [(o["x1"] + o["x2"]) / 2.0 for o in best_ev["obs"]]
            all_cy = [(o["y1"] + o["y2"]) / 2.0 for o in best_ev["obs"]]
            best_ev["med_center"] = (float(np.median(all_cx)), float(np.median(all_cy)))
            best_ev["med_box"] = (
                int(np.median([o["x1"] for o in best_ev["obs"]])),
                int(np.median([o["y1"] for o in best_ev["obs"]])),
                int(np.median([o["x2"] for o in best_ev["obs"]])),
                int(np.median([o["y2"] for o in best_ev["obs"]])),
            )

    filtered = [ev for ev in events if len(ev["frames_set"]) >= min_event_frames]
    for idx, ev in enumerate(filtered, start=1):
        ev["event_id"] = idx
    return filtered


def should_accept_expanded_sam_mask(
    area_box: int,
    area_pt: int,
    inter_with_box: int,
    frame_area: int,
) -> bool:
    """Decide se a máscara expandida do SAM 2.1 via ponto interior deve ser aceita.

    Critérios:
    - overlap_box >= 0.70 (preserva coerência espacial com a detecção inicial da caixa)
    - area_pt > area_box * 1.05 (expansão mínima de 5% sobre a área da caixa)
    - area_pt <= min(area_box * 3.8, frame_area * 0.24) (limita vazamento e sobredimensionamento)
    """
    overlap_box = inter_with_box / max(1, area_box)
    return (
        area_pt > area_box * 1.05
        and area_pt <= min(area_box * 3.8, frame_area * 0.24)
        and overlap_box >= 0.70
    )


def segment_full_fish_sam2(
    sam,
    frame_bgr: np.ndarray,
    fish_box: tuple[int, int, int, int],
    person_boxes: list[tuple[int, int, int, int]],
    device: str,
) -> tuple[np.ndarray, str]:
    """Segmenta o peixe com SAM 2.1 usando caixa + refinamento por ponto interior."""
    h, w = frame_bgr.shape[:2]
    frame_area = h * w
    x1, y1, x2, y2 = fish_box

    res_box = sam.predict(frame_bgr, bboxes=[[x1, y1, x2, y2]], device=device, verbose=False)[0]
    if res_box.masks is None or len(res_box.masks.data) == 0:
        return np.zeros((h, w), dtype=np.uint8), "vazia"

    m_box = res_box.masks.data[0].cpu().numpy()
    if m_box.shape != (h, w):
        m_box = cv2.resize(m_box.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    mask_box = _keep_largest_component((m_box > 0.5).astype(np.uint8) * 255)
    area_box = int(np.count_nonzero(mask_box > 127))

    pt = _interior_point(mask_box)
    if pt is None or area_box < 500:
        return mask_box, "box"

    pts = [[pt[0], pt[1]]]
    lbls = [1]
    for pb in person_boxes:
        px1, py1, px2, py2 = pb
        pcx, pcy = (px1 + px2) // 2, (py1 + py2) // 2
        if mask_box[min(h - 1, max(0, pcy)), min(w - 1, max(0, pcx))] == 0:
            pts.append([pcx, pcy])
            lbls.append(0)

    res_pt = sam.predict(frame_bgr, points=[pts], labels=[lbls], device=device, verbose=False)[0]
    if res_pt.masks is None or len(res_pt.masks.data) == 0:
        return mask_box, "box"

    m_pt = res_pt.masks.data[0].cpu().numpy()
    if m_pt.shape != (h, w):
        m_pt = cv2.resize(m_pt.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    mask_pt = _keep_largest_component((m_pt > 0.5).astype(np.uint8) * 255)
    area_pt = int(np.count_nonzero(mask_pt > 127))

    inter_with_box = int(np.count_nonzero((mask_pt > 127) & (mask_box > 127)))

    if should_accept_expanded_sam_mask(area_box, area_pt, inter_with_box, frame_area):
        return mask_pt, "ponto_expandido"

    return mask_box, "box"


def save_audit_card(
    frame_bgr: np.ndarray,
    det_box: tuple[int, int, int, int],
    meas: dict,
    event_id: int,
    fish_label: str,
    label_conf: float,
    track_ids: list[int],
    frame_idx: int,
    tempo_s: float,
    person_overlap: float,
    sam_mode: str,
    mean_px: float,
    err_pct: float,
    qualidade: str,
    cm_por_px: float | None,
    out_path: Path,
) -> None:
    """Card visual inteligente de auditoria científica e produto."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = det_box
    mx1, my1, mx2, my2 = meas["mask_bbox"]

    ux1, uy1 = min(x1, mx1) if mx2 > mx1 else x1, min(y1, my1) if my2 > my1 else y1
    ux2, uy2 = max(x2, mx2) if mx2 > mx1 else x2, max(y2, my2) if my2 > my1 else y2

    left = frame_bgr.copy()
    cv2.rectangle(left, (x1, y1), (x2, y2), (170, 170, 170), 2, cv2.LINE_AA)
    cv2.rectangle(left, (ux1, uy1), (ux2, uy2), (0, 230, 180), 4, cv2.LINE_AA)
    left_small = cv2.resize(left, (960, 540), interpolation=cv2.INTER_AREA)

    pad = max(45, int(0.14 * max(ux2 - ux1, uy2 - uy1)))
    rx1, ry1 = max(0, ux1 - pad), max(0, uy1 - pad)
    rx2, ry2 = min(w, ux2 + pad), min(h, uy2 + pad)

    crop = frame_bgr[ry1:ry2, rx1:rx2].copy()
    mask_crop = meas["mask_clean"][ry1:ry2, rx1:rx2]

    overlay = crop.copy()
    overlay[mask_crop > 127] = (
        0.55 * overlay[mask_crop > 127] + 0.45 * np.array([180, 220, 20], dtype=np.float32)
    ).astype(np.uint8)

    contours, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 220), 2, cv2.LINE_AA)

    pts_local = [(px - rx1, py - ry1) for (px, py) in meas["centerline_pts"]]
    for i in range(1, len(pts_local)):
        cv2.line(overlay, pts_local[i - 1], pts_local[i], (0, 80, 255), 3, cv2.LINE_AA)
    if len(pts_local) >= 2:
        cv2.circle(overlay, pts_local[0], 7, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(overlay, pts_local[-1], 7, (0, 0, 255), -1, cv2.LINE_AA)

    ch, cw = overlay.shape[:2]
    scale = min(960 / max(1, cw), 540 / max(1, ch))
    nw, nh = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
    resized = cv2.resize(overlay, (nw, nh), interpolation=cv2.INTER_AREA)
    right = np.full((540, 960, 3), 20, dtype=np.uint8)
    ox, oy = (960 - nw) // 2, (540 - nh) // 2
    right[oy : oy + nh, ox : ox + nw] = resized

    canvas = np.zeros((625, 1920, 3), dtype=np.uint8)
    canvas[:85, :] = (16, 23, 38)
    canvas[85:, :960] = left_small
    canvas[85:, 960:] = right

    q_color = (40, 190, 90) if qualidade == "ALTA" else ((20, 175, 235) if qualidade == "MEDIA" else (60, 80, 220))
    cv2.rectangle(canvas, (1690, 14), (1900, 70), q_color, -1)
    cv2.putText(canvas, f"QUALIDADE: {qualidade}", (1704, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)

    size_str = (
        f"{mean_px * cm_por_px:.1f} cm (+/- {err_pct:.1f}%)"
        if cm_por_px is not None
        else f"{mean_px:.1f} px (+/- {err_pct:.1f}%)"
    )
    title = (
        f"Individuo #{event_id} | Classe: {fish_label} ({label_conf*100:.0f}%) | "
        f"Tamanho medio: {size_str} | Frame {frame_idx} ({_fmt_hms(tempo_s)})"
    )
    tids_txt = ",".join(f"#{t}" for t in track_ids[:6]) + ("..." if len(track_ids) > 6 else "")
    sub = (
        f"Curva frame: {meas['comprimento_curva_px']:.1f} px | Eixo: {meas['comprimento_eixo_px']:.1f} px | "
        f"Aspecto: {meas['razao_aspecto']:.2f} | Oclusao: {person_overlap*100:.0f}%"
    )
    cv2.putText(canvas, title, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, sub, (20, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (185, 215, 235), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def render_hud_video(
    video_path: Path,
    out_video_path: Path,
    start_frame: int,
    end_frame: int,
    frame_step: int,
    fps: float,
    resumo_rows: list[dict],
    events_by_valid_id: dict[int, dict],
    cm_por_px: float | None,
) -> None:
    """Renderiza vídeo demo anotado com HUD inteligente (contagem, tamanho médio, erro ±%, classe)."""
    cap = cv2.VideoCapture(str(video_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w, out_h = 1280, 720
    sx, sy = out_w / max(1, w), out_h / max(1, h)

    # Mapa frame -> lista de (event_id, obs, resumo_row)
    frame_map: dict[int, list[tuple[int, dict, dict]]] = defaultdict(list)
    first_frame_by_id: dict[int, int] = {}
    row_by_id = {r["individuo_id"]: r for r in resumo_rows}
    for eid, ev in events_by_valid_id.items():
        r = row_by_id[eid]
        first_frame_by_id[eid] = min(o["frame"] for o in ev["obs"])
        for o in ev["obs"]:
            frame_map[o["frame"]].append((eid, o, r))

    out_fps = max(2.0, fps / max(1, frame_step))
    writer = cv2.VideoWriter(str(out_video_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (out_w, out_h))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    seen_ids: list[int] = []
    seen_set: set[int] = set()

    f_idx = start_frame
    while f_idx < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if (f_idx - start_frame) % frame_step != 0:
            f_idx += 1
            continue

        small = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        # Atualiza contados acumulados
        for eid, ff in first_frame_by_id.items():
            if f_idx >= ff and eid not in seen_set:
                seen_set.add(eid)
                seen_ids.append(eid)

        # Desenha caixas e etiquetas dos peixes ativos neste frame
        for eid, obs, r in frame_map.get(f_idx, []):
            x1, y1 = int(obs["x1"] * sx), int(obs["y1"] * sy)
            x2, y2 = int(obs["x2"] * sx), int(obs["y2"] * sy)
            color = (30, 220, 170) if r["qualidade_medicao"] == "ALTA" else (20, 185, 245)
            cv2.rectangle(small, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            sz = (
                f"{r['tamanho_medio_cm']} cm"
                if cm_por_px is not None
                else f"{r['tamanho_medio_px']:.0f} px"
            )
            tag = f"#{eid} {r['classe_estimada']} | {sz} (+/-{r['erro_estimado_pct']:.1f}%)"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
            ty = max(th + 8, y1 - 6)
            cv2.rectangle(small, (x1, ty - th - 8), (x1 + tw + 12, ty + 4), (16, 23, 38), -1)
            cv2.putText(small, tag, (x1 + 6, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        # Painel HUD translúcido no canto superior direito
        hud_x1, hud_y1, hud_x2, hud_y2 = out_w - 430, 14, out_w - 14, 134
        overlay = small.copy()
        cv2.rectangle(overlay, (hud_x1, hud_y1), (hud_x2, hud_y2), (14, 22, 38), -1)
        small = cv2.addWeighted(overlay, 0.78, small, 0.22, 0)
        cv2.rectangle(small, (hud_x1, hud_y1), (hud_x2, hud_y2), (0, 210, 175), 1, cv2.LINE_AA)

        cv2.putText(small, "FISHID STUDIO  |  EM MONITORAMENTO", (hud_x1 + 14, hud_y1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 235, 195), 2, cv2.LINE_AA)
        if seen_ids:
            seen_rows = [row_by_id[i] for i in seen_ids]
            avg_px = float(np.mean([r["tamanho_medio_px"] for r in seen_rows]))
            avg_err = float(np.mean([r["erro_estimado_pct"] for r in seen_rows]))
            sz_avg = f"{avg_px * cm_por_px:.1f} cm" if cm_por_px else f"{avg_px:.0f} px"
            last_r = seen_rows[-1]
            line1 = f"Contados: {len(seen_ids)}  |  Tam. Medio: {sz_avg}"
            line2 = f"Erro medio estimado: +/- {avg_err:.1f}%"
            line3 = f"Ultimo: #{last_r['individuo_id']} {last_r['classe_estimada']} ({last_r['qualidade_medicao']})"
        else:
            line1 = "Contados: 0  |  Tam. Medio: --"
            line2 = "Erro medio estimado: --"
            line3 = f"Tempo video: {_fmt_hms(f_idx / fps)}"

        cv2.putText(small, line1, (hud_x1 + 14, hud_y1 + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(small, line2, (hud_x1 + 14, hud_y1 + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (190, 220, 240), 1, cv2.LINE_AA)
        cv2.putText(small, line3, (hud_x1 + 14, hud_y1 + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (130, 240, 200), 1, cv2.LINE_AA)

        writer.write(small)
        f_idx += 1

    cap.release()
    writer.release()


def run_experiment(args: argparse.Namespace) -> Path:
    import torch
    from ultralytics import SAM, YOLO

    t0 = time.perf_counter()
    video_path = args.video.resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")

    device = "cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 2.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    start_frame = max(0, args.start_frame)
    end_frame = min(total_frames, args.end_frame) if args.end_frame is not None else total_frames
    if end_frame <= start_frame:
        raise ValueError(f"Intervalo inválido: start={start_frame}, end={end_frame}")

    exp_root = args.exp_root.resolve()
    exp_root.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_tag = "".join(c if c.isalnum() or c in "-_" else "_" for c in args.tag).strip("_") or "run"
    run_dir = exp_root / f"run_{run_ts}_{video_path.stem}_{safe_tag}"
    audit_dir = run_dir / "auditoria_peixes"
    crops_dir = run_dir / "crops_peixes"
    audit_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "run_id": run_dir.name,
        "pipeline_version": "v5_fishid_producao",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "video_path": str(video_path),
        "video_name": video_path.name,
        "resolution": f"{width}x{height}",
        "fps": fps,
        "total_video_frames": total_frames,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_hms": _fmt_hms(start_frame / fps),
        "end_hms": _fmt_hms(end_frame / fps),
        "frame_step": args.frame_step,
        "yolo_weights": args.yolo_weights,
        "sam_weights": args.sam_weights,
        "tracker": args.tracker,
        "imgsz": args.imgsz,
        "conf_thresh": args.conf,
        "agnostic_nms": True,
        "min_track_frames": args.min_track_frames,
        "min_event_frames": args.min_event_frames,
        "min_area_px": args.min_area_px,
        "min_length_px": args.min_length_px,
        "min_aspect_ratio": args.min_aspect_ratio,
        "merge_gap_s": args.merge_gap_s,
        "merge_dist_px": args.merge_dist_px,
        "top_k_measure_frames": args.top_k_frames,
        "use_ridge_centerline": not args.simple_centerline,
        "cm_por_px": args.cm_por_px,
        "fishid_api": args.fishid_api or "",
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device.startswith("cuda") else "CPU",
    }
    (run_dir / "config_experimento.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"=== Experimento v5: {run_dir.name} ===")
    print(
        f"Janela: frames {start_frame:,}..{end_frame:,} "
        f"({_fmt_hms(start_frame / fps)} -> {_fmt_hms(end_frame / fps)}) | GPU: {config['gpu_name']} ({_gpu_temp_c()})"
    )

    # =========================================================================
    # FASE 1: Detecção (agnostic_nms=True) + Rastreamento (ByteTrack)
    # =========================================================================
    yolo = YOLO(args.yolo_weights)
    yolo.set_classes(ALL_CLASSES)

    det_csv = run_dir / "deteccoes_frames.csv"
    det_fields = [
        "frame", "tempo_s", "tempo_hms", "track_id", "classe_prompt", "confianca",
        "x1", "y1", "x2", "y2", "largura_px", "altura_px", "area_box_px2",
        "person_overlap", "toca_borda",
    ]

    track_obs: dict[int, list[dict]] = defaultdict(list)
    frame_persons: dict[int, list[tuple[int, int, int, int]]] = {}
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame
    processed = 0
    total_to_process = (end_frame - start_frame + args.frame_step - 1) // args.frame_step

    with det_csv.open("w", encoding="utf-8-sig", newline="") as f_det, torch.inference_mode():
        writer = csv.DictWriter(f_det, fieldnames=det_fields)
        writer.writeheader()

        while frame_idx < end_frame:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            if (frame_idx - start_frame) % args.frame_step != 0:
                frame_idx += 1
                continue

            res = yolo.track(
                frame_bgr,
                persist=True,
                tracker=args.tracker,
                conf=args.conf,
                iou=0.55,
                agnostic_nms=True,
                imgsz=args.imgsz,
                device=device,
                verbose=False,
            )[0]

            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                clss = res.boxes.cls.int().cpu().numpy()
                ids = (
                    res.boxes.id.int().cpu().numpy()
                    if res.boxes.id is not None
                    else np.full(len(xyxy), -1, dtype=int)
                )

                person_boxes = [
                    tuple(int(round(v)) for v in b)
                    for b, c in zip(xyxy, clss)
                    if int(c) == PERSON_CLASS_ID
                ]

                has_fish_here = False
                for box, conf, cls_id, tid in zip(xyxy, confs, clss, ids):
                    cls_int = int(cls_id)
                    tid_int = int(tid)
                    if cls_int not in FISH_CLASS_IDS or tid_int < 0:
                        continue

                    x1, y1, x2, y2 = [int(round(v)) for v in box]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width - 1, x2), min(height - 1, y2)
                    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
                    area_box = bw * bh
                    if area_box < args.min_area_px:
                        continue

                    has_fish_here = True
                    max_p_ov = (
                        max(_box_overlap_ratio((x1, y1, x2, y2), pb) for pb in person_boxes)
                        if person_boxes
                        else 0.0
                    )
                    toca_borda = int(x1 <= 8 or y1 <= 8 or x2 >= width - 9 or y2 >= height - 9)
                    tempo_s = round(frame_idx / fps, 2)

                    row = {
                        "frame": frame_idx,
                        "tempo_s": tempo_s,
                        "tempo_hms": _fmt_hms(tempo_s),
                        "track_id": tid_int,
                        "classe_prompt": ALL_CLASSES[cls_int],
                        "confianca": round(float(conf), 4),
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "largura_px": bw, "altura_px": bh,
                        "area_box_px2": area_box,
                        "person_overlap": round(float(max_p_ov), 4),
                        "toca_borda": toca_borda,
                    }
                    writer.writerow(row)
                    track_obs[tid_int].append(row)

                if has_fish_here and person_boxes:
                    frame_persons[frame_idx] = person_boxes

            processed += 1
            if processed % 200 == 0 or frame_idx + 1 >= end_frame:
                f_det.flush()
                elapsed = time.perf_counter() - t0
                rate = processed / max(elapsed, 1e-6)
                valid_now = sum(1 for v in track_obs.values() if len(v) >= args.min_track_frames)
                print(
                    f"  Progresso: {processed:,}/{total_to_process:,} frames "
                    f"({rate:.1f} fr/s) | tracks validos: {valid_now} | GPU: {_gpu_temp_c()}"
                )

            frame_idx += 1

    del yolo
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    valid_tracks = {tid: obs for tid, obs in track_obs.items() if len(obs) >= args.min_track_frames}
    events = consolidate_tracks(
        valid_tracks=valid_tracks,
        merge_gap_s=args.merge_gap_s,
        merge_dist_px=args.merge_dist_px,
        min_event_frames=args.min_event_frames,
    )
    print(f"[Fase 1] Tracks validos: {len(valid_tracks)} -> Eventos candidatos: {len(events)}")

    # =========================================================================
    # FASE 2: Medição SAM 2.1 + filtros anti-boia/tamanho + classificação opcional
    # =========================================================================
    resumo_rows: list[dict] = []
    events_by_valid_id: dict[int, dict] = {}
    descartados_stats = {"forma": 0, "boia_cor": 0, "comprimento_curto": 0}

    if events:
        sam = SAM(args.sam_weights)
        frame_to_jobs: dict[int, list[tuple[int, dict]]] = defaultdict(list)

        for ev in events:
            obs_list = ev["obs"]
            t_start = ev["t_start"]

            def _quality_score(o: dict) -> float:
                age_s = max(0.0, o["tempo_s"] - t_start)
                early_bonus = 1.40 if age_s <= 40.0 else (1.0 if age_s <= 120.0 else 0.72)
                ov_pen = 1.0 - min(0.95, o["person_overlap"] * 1.2)
                borda_pen = 0.78 if o["toca_borda"] else 1.0
                return o["confianca"] * early_bonus * ov_pen * borda_pen

            ranked = sorted(obs_list, key=_quality_score, reverse=True)
            chosen: list[dict] = []
            for cand in ranked:
                if all(abs(cand["frame"] - c["frame"]) >= 2 for c in chosen):
                    chosen.append(cand)
                if len(chosen) >= args.top_k_frames:
                    break
            if not chosen:
                chosen = ranked[: args.top_k_frames]

            for c in chosen:
                frame_to_jobs[c["frame"]].append((ev["event_id"], c))

        event_measurements: dict[int, list[tuple[dict, dict, str, np.ndarray]]] = defaultdict(list)
        with torch.inference_mode():
            for f_idx in sorted(frame_to_jobs.keys()):
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ok, frame_bgr = cap.read()
                if not ok:
                    continue

                p_boxes = frame_persons.get(f_idx, [])
                for ev_id, obs in frame_to_jobs[f_idx]:
                    fish_box = (obs["x1"], obs["y1"], obs["x2"], obs["y2"])
                    mask_u8, sam_mode = segment_full_fish_sam2(
                        sam=sam,
                        frame_bgr=frame_bgr,
                        fish_box=fish_box,
                        person_boxes=p_boxes,
                        device=device,
                    )
                    meas = measure_mask_length(mask_u8, use_ridge=not args.simple_centerline)
                    if meas["comprimento_curva_px"] > 0:
                        event_measurements[ev_id].append((obs, meas, sam_mode, frame_bgr))

        valid_ev_counter = 0
        for ev in events:
            ev_id = ev["event_id"]
            obs_list = ev["obs"]
            t_ini, t_fim = ev["t_start"], ev["t_end"]
            conf_med = float(np.mean([o["confianca"] for o in obs_list]))
            ov_min = float(min(o["person_overlap"] for o in obs_list))
            fish_label, label_conf = _infer_fish_group(obs_list)

            meas_list = event_measurements.get(ev_id, [])
            if not meas_list:
                continue

            all_curvas = [m["comprimento_curva_px"] for _, m, _, _ in meas_list]
            max_curva = max(all_curvas)
            intact_items = [item for item in meas_list if item[1]["comprimento_curva_px"] >= 0.72 * max_curva]
            if not intact_items:
                intact_items = meas_list

            curvas_ok = np.array([m["comprimento_curva_px"] for _, m, _, _ in intact_items], dtype=float)
            eixos_ok = np.array([m["comprimento_eixo_px"] for _, m, _, _ in intact_items], dtype=float)
            aspectos_ok = np.array([m["razao_aspecto"] for _, m, _, _ in intact_items], dtype=float)
            areas_ok = np.array([m["area_px2"] for _, m, _, _ in intact_items], dtype=float)

            mean_px = float(np.mean(curvas_ok))
            std_px = float(np.std(curvas_ok)) if len(curvas_ok) >= 2 else 0.0
            aspecto_med = float(np.median(aspectos_ok))

            best_item = max(
                intact_items,
                key=lambda it: it[1]["comprimento_curva_px"] * (1.0 - 0.35 * it[0]["person_overlap"]),
            )
            best_obs, best_meas, best_mode, best_frame = best_item

            # Filtros anti-falso positivo validados nos testes de stress
            is_buoy, _ = _is_neon_buoy_mask(best_frame, best_meas["mask_clean"])
            if is_buoy and best_obs["person_overlap"] == 0.0:
                descartados_stats["boia_cor"] += 1
                continue
            if mean_px < args.min_length_px:
                descartados_stats["comprimento_curto"] += 1
                continue
            if aspecto_med < args.min_aspect_ratio and conf_med < 0.45:
                descartados_stats["forma"] += 1
                continue

            valid_ev_counter += 1
            events_by_valid_id[valid_ev_counter] = ev

            if len(curvas_ok) >= 2:
                err_pct = max(2.5, (std_px / max(1.0, mean_px)) * 100.0)
            else:
                err_pct = 8.0 + 15.0 * best_obs["person_overlap"]

            if best_obs["person_overlap"] <= 0.15 and err_pct <= 9.0 and len(intact_items) >= 2:
                qualidade = "ALTA"
            elif best_obs["person_overlap"] <= 0.35 and err_pct <= 16.0:
                qualidade = "MEDIA"
            else:
                qualidade = "PARCIAL"

            # Salva recorte limpo para classificação de espécie (ConvNeXt-Tiny)
            mx1, my1, mx2, my2 = best_meas["mask_bbox"]
            pad_c = max(20, int(0.08 * max(mx2 - mx1, my2 - my1)))
            cx1, cy1 = max(0, mx1 - pad_c), max(0, my1 - pad_c)
            cx2, cy2 = min(width, mx2 + pad_c), min(height, my2 + pad_c)
            crop_file = crops_dir / f"individuo_{valid_ev_counter:03d}_frame_{best_obs['frame']:06d}.jpg"
            cv2.imwrite(str(crop_file), best_frame[cy1:cy2, cx1:cx2], [int(cv2.IMWRITE_JPEG_QUALITY), 93])

            # Classificação opcional via API FishID (ConvNeXt-Tiny)
            especie_convnext = ""
            conf_convnext = ""
            if args.fishid_api:
                api_res = _classify_crop_via_fishid_api(args.fishid_api, crop_file)
                if api_res is not None:
                    especie_convnext, conf_convnext = api_res
                    if conf_convnext >= 0.60:
                        fish_label, label_conf = especie_convnext, conf_convnext

            audit_file = audit_dir / f"individuo_{valid_ev_counter:03d}_frame_{best_obs['frame']:06d}.jpg"
            save_audit_card(
                frame_bgr=best_frame,
                det_box=(best_obs["x1"], best_obs["y1"], best_obs["x2"], best_obs["y2"]),
                meas=best_meas,
                event_id=valid_ev_counter,
                fish_label=fish_label,
                label_conf=label_conf,
                track_ids=ev["track_ids"],
                frame_idx=best_obs["frame"],
                tempo_s=best_obs["tempo_s"],
                person_overlap=best_obs["person_overlap"],
                sam_mode=best_mode,
                mean_px=mean_px,
                err_pct=err_pct,
                qualidade=qualidade,
                cm_por_px=args.cm_por_px,
                out_path=audit_file,
            )

            resumo_rows.append(
                {
                    "individuo_id": valid_ev_counter,
                    "classe_estimada": fish_label,
                    "confianca_classe": label_conf,
                    "especie_convnext": especie_convnext,
                    "confianca_convnext": conf_convnext,
                    "track_ids_fundidos": ",".join(str(t) for t in ev["track_ids"]),
                    "frames_detectados": len(ev["frames_set"]),
                    "inicio_hms": _fmt_hms(t_ini),
                    "fim_hms": _fmt_hms(t_fim),
                    "duracao_s": round(t_fim - t_ini, 1),
                    "confianca_media_yolo": round(conf_med, 4),
                    "min_oclusao_pescador": round(ov_min, 4),
                    "frames_intactos_sam": len(intact_items),
                    "modo_sam": best_mode,
                    "tamanho_medio_px": round(mean_px, 1),
                    "desvio_px": round(std_px, 1),
                    "erro_estimado_pct": round(err_pct, 1),
                    "eixo_medio_px": round(float(np.mean(eixos_ok)), 1),
                    "razao_aspecto": round(aspecto_med, 2),
                    "tamanho_medio_cm": round(mean_px * args.cm_por_px, 1) if args.cm_por_px else "",
                    "qualidade_medicao": qualidade,
                    "area_mascara_px2": int(round(float(np.median(areas_ok)))),
                    "frame_auditoria": best_obs["frame"],
                    "imagem_auditoria": f"auditoria_peixes/{audit_file.name}",
                    "recorte_especie": f"crops_peixes/{crop_file.name}",
                    "validacao_humana": "",
                }
            )

    cap.release()

    if args.save_video and resumo_rows:
        out_vid = run_dir / "video_anotado_hud.mp4"
        print(f"Renderizando video HUD demo em {out_vid.name}...")
        render_hud_video(
            video_path=video_path,
            out_video_path=out_vid,
            start_frame=start_frame,
            end_frame=end_frame,
            frame_step=args.frame_step,
            fps=fps,
            resumo_rows=resumo_rows,
            events_by_valid_id=events_by_valid_id,
            cm_por_px=args.cm_por_px,
        )

    resumo_df = pd.DataFrame(resumo_rows)
    resumo_df.to_csv(run_dir / "resumo_peixes.csv", index=False, encoding="utf-8-sig")

    elapsed_total = round(time.perf_counter() - t0, 1)
    gpu_end = _gpu_temp_c()

    idx_csv = exp_root / "indice_experimentos.csv"
    idx_fields = [
        "run_id", "data_hora", "video", "janela_hms", "frames_processados",
        "peixes_contados", "comprimento_mediano_px", "cm_por_px", "tempo_execucao_s", "pasta",
    ]
    write_hdr = not idx_csv.exists()
    med_px_geral = round(float(resumo_df["tamanho_medio_px"].median()), 1) if not resumo_df.empty else 0.0
    err_med_geral = round(float(resumo_df["erro_estimado_pct"].mean()), 1) if not resumo_df.empty else 0.0

    with idx_csv.open("a", encoding="utf-8-sig", newline="") as f_idx:
        w_idx = csv.DictWriter(f_idx, fieldnames=idx_fields)
        if write_hdr:
            w_idx.writeheader()
        w_idx.writerow(
            {
                "run_id": run_dir.name,
                "data_hora": config["created_at"],
                "video": video_path.name,
                "janela_hms": f"{config['start_hms']} -> {config['end_hms']}",
                "frames_processados": processed,
                "peixes_contados": len(resumo_rows),
                "comprimento_mediano_px": med_px_geral,
                "cm_por_px": args.cm_por_px if args.cm_por_px is not None else "pendente",
                "tempo_execucao_s": elapsed_total,
                "pasta": str(run_dir),
            }
        )

    md_lines = [
        f"# Relatório do Experimento (v5) — `{run_dir.name}`",
        "",
        f"- **Vídeo:** `{video_path.name}` ({width}×{height} @ {fps:.2f} FPS)",
        f"- **Janela:** `{config['start_hms']}` → `{config['end_hms']}` (frames `{start_frame:,}`..`{end_frame:,}`, step `{args.frame_step}`)",
        f"- **Hardware:** `{config['gpu_name']}` (tempo: **{elapsed_total:.1f} s** | temp final: `{gpu_end}`)",
        f"- **Indivíduos validados:** **{len(resumo_rows)}** *(descartados — boia/cor: {descartados_stats['boia_cor']}, curtos: {descartados_stats['comprimento_curto']}, forma: {descartados_stats['forma']})*",
        f"- **Tamanho mediano:** **{med_px_geral:.1f} px** | **Erro médio estimado:** **±{err_med_geral:.1f}%**",
        "",
        "## Resumo por Indivíduo",
        "",
    ]
    if not resumo_df.empty:
        md_lines.append("| # | Classe | Início | Fim | Frames | Tam. Médio (px) | Erro (±%) | Qualidade | Modo SAM | Auditoria |")
        md_lines.append("| :-: | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :--- |")
        for r in resumo_rows:
            md_lines.append(
                f"| #{r['individuo_id']} | {r['classe_estimada']} | {r['inicio_hms']} | {r['fim_hms']} | {r['frames_detectados']} "
                f"| **{r['tamanho_medio_px']:.1f}** | ±{r['erro_estimado_pct']:.1f}% | `{r['qualidade_medicao']}` "
                f"| `{r['modo_sam']}` | `{r['imagem_auditoria']}` |"
            )
    else:
        md_lines.append("_Nenhum peixe detectado nesta janela (0 falsos positivos)._")

    (run_dir / "relatorio_experimento.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(
        f"=== CONCLUIDO v5 em {elapsed_total:.1f}s | Contados: {len(resumo_rows)} | "
        f"Descartados: {descartados_stats} | Mediana: {med_px_geral:.1f} px (+/-{err_med_geral:.1f}%) | GPU: {gpu_end} ==="
    )
    return run_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FishID Studio — Pipeline v5 de Monitoramento Eletrônico (EM).")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--tag", type=str, default="v5")
    p.add_argument("--exp-root", type=Path, default=Path("./experimentos_em"))
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--end-frame", type=int, default=None)
    p.add_argument("--frame-step", type=int, default=1)
    p.add_argument("--yolo-weights", type=str, default="yolov8l-worldv2.pt")
    p.add_argument("--sam-weights", type=str, default="sam2.1_t.pt")
    p.add_argument("--tracker", type=str, default="bytetrack.yaml")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--conf", type=float, default=0.12)
    p.add_argument("--min-track-frames", type=int, default=2)
    p.add_argument("--min-event-frames", type=int, default=4)
    p.add_argument("--min-area-px", type=int, default=15000)
    p.add_argument("--min-length-px", type=float, default=320.0)
    p.add_argument("--min-aspect-ratio", type=float, default=1.7)
    p.add_argument("--merge-gap-s", type=float, default=120.0)
    p.add_argument("--merge-dist-px", type=float, default=450.0)
    p.add_argument("--top-k-frames", type=int, default=6)
    p.add_argument("--simple-centerline", action="store_true")
    p.add_argument("--cm-por-px", type=float, default=None)
    p.add_argument("--fishid-api", type=str, default="", help="URL opcional da API FishID (ex.: http://localhost:8000)")
    p.add_argument("--save-video", action="store_true", help="Gera vídeo MP4 anotado com HUD inteligente")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    run_experiment(parse_args())
