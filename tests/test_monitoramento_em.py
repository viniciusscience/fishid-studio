import json
import sys
import urllib.error
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import monitoramento_em  # noqa: E402
from monitoramento_em import (  # noqa: E402
    _classify_crop_via_fishid_api,
    _is_neon_buoy_mask,
    consolidate_tracks,
    measure_mask_length,
)


class _DummyHTTPResponse:
    def __init__(self, payload: dict):
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _write_dummy_crop(tmp_path: Path) -> Path:
    crop_path = tmp_path / "crop.jpg"
    img = np.full((64, 128, 3), 140, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    crop_path.write_bytes(buf.tobytes())
    return crop_path


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            {
                "mode": "real",
                "display_name": "Albacora-laje (Thunnus albacares)",
                "top_species": "Albacora-laje",
                "scientific_name": "Thunnus albacares",
                "confidence": 0.91234,
            },
            ("Albacora-laje (Thunnus albacares)", 0.9123),
        ),
        (
            {"top_species": "Espadarte (Meca)", "scientific_name": "Xiphias gladius", "confidence": 0.77},
            ("Espadarte (Meca)", 0.77),
        ),
        (
            {"scientific_name": "Coryphaena hippurus", "confidence": 0.85},
            ("Coryphaena hippurus", 0.85),
        ),
        (
            {"top_prediction": {"display_name": "Bonito-listrado (Katsuwonus pelamis)", "confidence": 0.64}},
            ("Bonito-listrado (Katsuwonus pelamis)", 0.64),
        ),
    ],
)
def test_classify_crop_via_fishid_api_parses_root_and_fallback_fields(
    tmp_path, monkeypatch, payload, expected
):
    crop_path = _write_dummy_crop(tmp_path)
    monkeypatch.setattr(
        monitoramento_em.urllib.request,
        "urlopen",
        lambda req, timeout=10: _DummyHTTPResponse(payload),
    )
    assert _classify_crop_via_fishid_api("http://127.0.0.1:8000", crop_path) == expected


def test_classify_crop_matches_live_backend_predict_response(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    backend_dir = str(Path(__file__).resolve().parents[1] / "backend")
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    import main

    crop_path = _write_dummy_crop(tmp_path)
    api_resp = TestClient(main.app).post(
        "/predict",
        files={"file": (crop_path.name, crop_path.read_bytes(), "image/jpeg")},
        data={"hint_species": "Coryphaena hippurus"},
    )
    assert api_resp.status_code == 200
    payload = api_resp.json()

    monkeypatch.setattr(
        monitoramento_em.urllib.request,
        "urlopen",
        lambda req, timeout=10: _DummyHTTPResponse(payload),
    )
    result = _classify_crop_via_fishid_api("http://127.0.0.1:8000", crop_path)
    assert result == (payload["display_name"], round(payload["confidence"], 4))


def test_classify_crop_via_fishid_api_warns_single_line_when_offline(tmp_path, monkeypatch, capsys):
    crop_path = _write_dummy_crop(tmp_path)

    def _raise_offline(req, timeout=10):
        raise urllib.error.URLError("connection refused\nextra detail")

    monkeypatch.setattr(monitoramento_em.urllib.request, "urlopen", _raise_offline)
    result = _classify_crop_via_fishid_api("http://127.0.0.1:9999/", crop_path)
    assert result is None

    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert len(lines) == 1
    assert "[WARN]" in lines[0]
    assert "http://127.0.0.1:9999/predict" in lines[0]


def test_measure_mask_length_handles_empty_straight_and_curved_masks():
    empty = np.zeros((200, 400), dtype=np.uint8)
    res_empty = measure_mask_length(empty)
    assert res_empty["comprimento_curva_px"] == 0.0
    assert res_empty["comprimento_eixo_px"] == 0.0
    assert res_empty["centerline_pts"] == []

    # Straight horizontal fish silhouette (~240 px long, ~40 px wide)
    straight = np.zeros((200, 400), dtype=np.uint8)
    cv2.ellipse(straight, center=(200, 100), axes=(120, 20), angle=0, startAngle=0, endAngle=360, color=255, thickness=-1)
    res_straight = measure_mask_length(straight, use_ridge=True)
    assert 200.0 <= res_straight["comprimento_curva_px"] <= 260.0
    assert res_straight["comprimento_curva_px"] >= res_straight["comprimento_eixo_px"]
    assert res_straight["razao_aspecto"] > 4.0
    assert len(res_straight["centerline_pts"]) >= 2

    # Curved arc fish mask: curved length should exceed straight endpoint distance
    curved = np.zeros((300, 400), dtype=np.uint8)
    angles = np.linspace(np.deg2rad(-70), np.deg2rad(70), 40)
    arc_pts = np.column_stack(
        [200 + 110 * np.sin(angles), 220 - 110 * np.cos(angles)]
    ).astype(np.int32)
    cv2.polylines(curved, [arc_pts], isClosed=False, color=255, thickness=28)
    res_curved = measure_mask_length(curved, use_ridge=True)
    assert res_curved["comprimento_curva_px"] > res_curved["comprimento_eixo_px"] * 1.10


def test_is_neon_buoy_mask_distinguishes_yellow_pink_buoys_from_silver_fish():
    mask = np.zeros((120, 120), dtype=np.uint8)
    cv2.circle(mask, (60, 60), 30, 255, -1)

    # Neon yellow buoy (BGR: high G+R, low B)
    yellow_frame = np.zeros((120, 120, 3), dtype=np.uint8)
    yellow_frame[:] = (10, 225, 240)
    is_yellow_buoy, yellow_frac = _is_neon_buoy_mask(yellow_frame, mask)
    assert is_yellow_buoy is True
    assert yellow_frac >= 0.90

    # Neon pink/magenta buoy (BGR: high B+R, low G)
    pink_frame = np.zeros((120, 120, 3), dtype=np.uint8)
    pink_frame[:] = (180, 20, 240)
    is_pink_buoy, pink_frac = _is_neon_buoy_mask(pink_frame, mask)
    assert is_pink_buoy is True
    assert pink_frac >= 0.90

    # Silver/grey pelagic fish (low saturation)
    silver_frame = np.full((120, 120, 3), (175, 170, 165), dtype=np.uint8)
    is_fish_buoy, fish_frac = _is_neon_buoy_mask(silver_frame, mask)
    assert is_fish_buoy is False
    assert fish_frac == 0.0

    # Tiny mask (<40 px) always returns False
    tiny_mask = np.zeros((120, 120), dtype=np.uint8)
    tiny_mask[10:14, 10:14] = 255
    assert _is_neon_buoy_mask(yellow_frame, tiny_mask) == (False, 0.0)


def _make_obs(frame: int, tempo_s: float, box: tuple[int, int, int, int]) -> dict:
    x1, y1, x2, y2 = box
    return {
        "frame": frame,
        "tempo_s": tempo_s,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "classe_prompt": "tuna",
        "confianca": 0.85,
        "person_overlap": 0.0,
        "toca_borda": 0,
    }


def test_consolidate_tracks_merges_fragmented_and_duplicate_tracks_and_filters_short():
    valid_tracks = {
        # Fragment 1 and 2 of the same fish separated by a short gap & small spatial shift -> merged
        1: [_make_obs(10, 5.0, (100, 100, 300, 180)), _make_obs(11, 5.5, (104, 100, 304, 180))],
        2: [_make_obs(20, 10.0, (110, 102, 310, 182)), _make_obs(21, 10.5, (112, 102, 312, 182))],
        # Simultaneous high-IoU duplicate on same frames as track 1 -> merged into same event
        3: [_make_obs(10, 5.0, (102, 102, 298, 178)), _make_obs(11, 5.5, (105, 102, 302, 178))],
        # Distinct fish far away on overlapping frames -> separate event
        4: [
            _make_obs(10, 5.0, (700, 500, 900, 580)),
            _make_obs(11, 5.5, (702, 500, 902, 580)),
            _make_obs(12, 6.0, (704, 500, 904, 580)),
            _make_obs(13, 6.5, (706, 500, 906, 580)),
        ],
        # Too-short noise track (2 frames < min_event_frames=4) -> dropped
        5: [_make_obs(50, 25.0, (400, 400, 480, 440)), _make_obs(51, 25.5, (402, 400, 482, 440))],
    }

    events = consolidate_tracks(
        valid_tracks=valid_tracks,
        merge_gap_s=20.0,
        merge_dist_px=150.0,
        min_event_frames=4,
    )

    assert len(events) == 2
    assert [ev["event_id"] for ev in events] == [1, 2]
    assert sorted(events[0]["track_ids"]) == [1, 2, 3]
    assert events[0]["frames_set"] == {10, 11, 20, 21}
    assert events[1]["track_ids"] == [4]
    assert events[1]["frames_set"] == {10, 11, 12, 13}
