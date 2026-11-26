"""CPU-only regression coverage for EM geometry, track merging and API fallback."""

import http.client
import io
import json
import sys
import urllib.error
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import monitoramento_em as em  # noqa: E402


@pytest.fixture
def crop(tmp_path):
    path = tmp_path / "fish.jpg"
    path.write_bytes(b"test-crop-bytes")
    return path


def mock_prediction(monkeypatch, payload):
    """Replace the network boundary; no server, model or image inference runs."""
    requests = []

    def urlopen(request, timeout):
        requests.append((request, timeout))
        return io.BytesIO(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(em.urllib.request, "urlopen", urlopen)
    return requests


def test_api_uses_root_fields_and_posts_the_crop(monkeypatch, crop, capsys):
    requests = mock_prediction(monkeypatch, {
        "display_name": "Dourado-do-mar (Coryphaena hippurus)",
        "top_species": "Dourado-do-mar",
        "scientific_name": "Coryphaena hippurus",
        "confidence": 0.876543,
        "top_prediction": {"display_name": "Wrong legacy value", "confidence": 0.1},
    })
    assert em._classify_crop_via_fishid_api("http://fishid.test/", crop) == (
        "Dourado-do-mar (Coryphaena hippurus)", 0.8765,
    )
    request, timeout = requests[0]
    assert request.full_url == "http://fishid.test/predict"
    assert request.get_method() == "POST"
    assert timeout == 10
    assert b'name="file"; filename="fish.jpg"' in request.data
    assert crop.read_bytes() in request.data
    assert "multipart/form-data; boundary=" in request.get_header("Content-type")
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("payload, expected", [
    ({"display_name": " ", "top_species": "Atum", "scientific_name": "Thunnus albacares", "confidence": 0.7}, ("Atum", 0.7)),
    ({"scientific_name": "Thunnus albacares", "confidence": 1.0}, ("Thunnus albacares", 1.0)),
    ({"display_name": " Atum ", "confidence": 0}, ("Atum", 0.0)),
])
def test_api_label_fallbacks_and_probability_boundaries(monkeypatch, crop, payload, expected):
    mock_prediction(monkeypatch, payload)
    assert em._classify_crop_via_fishid_api("http://fishid.test", crop) == expected


@pytest.mark.parametrize("payload", [
    [], None, {},
    {"display_name": ["Atum"], "confidence": 0.8},
    {"display_name": "Atum"},
    {"display_name": "Atum", "confidence": None},
    {"display_name": "Atum", "confidence": True},
    {"display_name": "Atum", "confidence": "not-a-number"},
    {"display_name": "Atum", "confidence": float("nan")},
    {"display_name": "Atum", "confidence": float("inf")},
    {"display_name": "Atum", "confidence": -0.01},
    {"display_name": "Atum", "confidence": 1.01},
])
def test_api_invalid_prediction_returns_none_with_one_warning(monkeypatch, crop, capsys, payload):
    mock_prediction(monkeypatch, payload)
    assert em._classify_crop_via_fishid_api("http://fishid.test", crop) is None
    output = capsys.readouterr()
    assert len(output.out.splitlines()) == 1
    assert output.out.startswith("[WARN]")
    assert output.err == ""


@pytest.mark.parametrize("error", [
    urllib.error.URLError("connection refused\nextra error detail"),
    TimeoutError("connection timed out"),
    urllib.error.HTTPError("http://fishid.test/predict", 503, "Unavailable", {}, None),
    http.client.IncompleteRead(b"partial", 100),
])
def test_api_unavailable_is_optional_and_warns_once(monkeypatch, crop, capsys, error):
    def unavailable(*args, **kwargs):
        raise error

    monkeypatch.setattr(em.urllib.request, "urlopen", unavailable)
    assert em._classify_crop_via_fishid_api("http://fishid.test", crop) is None
    output = capsys.readouterr()
    assert len(output.out.splitlines()) == 1
    assert output.out.startswith("[WARN]")
    assert output.err == ""


def test_api_invalid_json_is_optional(monkeypatch, crop, capsys):
    monkeypatch.setattr(em.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"<html>unavailable</html>"))
    assert em._classify_crop_via_fishid_api("http://fishid.test", crop) is None
    assert capsys.readouterr().out.startswith("[WARN]")


def test_api_missing_crop_does_not_call_server(monkeypatch, tmp_path, capsys):
    requests = mock_prediction(monkeypatch, {"display_name": "Atum", "confidence": 0.8})
    assert em._classify_crop_via_fishid_api("http://fishid.test", tmp_path / "missing.jpg") is None
    assert requests == []
    assert capsys.readouterr().out.startswith("[WARN]")


def test_measure_empty_mask_has_zero_dimensions():
    mask = np.zeros((80, 120), dtype=np.uint8)
    result = em.measure_mask_length(mask)
    assert result["area_px2"] == 0
    assert result["comprimento_eixo_px"] == result["comprimento_curva_px"] == 0.0
    assert result["largura_corpo_px"] == result["razao_aspecto"] == 0.0
    assert result["mask_bbox"] == (0, 0, 0, 0)
    assert result["centerline_pts"] == []
    np.testing.assert_array_equal(result["mask_clean"], mask)


def test_measure_rectangle_matches_known_geometry_and_does_not_mutate_input():
    mask = np.zeros((100, 260), dtype=np.uint8)
    mask[40:60, 30:230] = 255
    before = mask.copy()
    result = em.measure_mask_length(mask, use_ridge=False)
    assert result["area_px2"] == 200 * 20
    assert result["mask_bbox"] == (30, 40, 229, 59)
    assert result["comprimento_eixo_px"] == pytest.approx(198, abs=3)
    assert result["comprimento_curva_px"] == pytest.approx(result["comprimento_eixo_px"], abs=0.2)
    assert result["largura_corpo_px"] == pytest.approx(20, abs=1)
    assert result["razao_aspecto"] > 9.5
    np.testing.assert_array_equal(mask, before)


def test_measure_discards_detached_smaller_component():
    mask = np.zeros((140, 280), dtype=np.uint8)
    mask[50:80, 40:240] = 255
    mask[5:10, 5:10] = 255
    result = em.measure_mask_length(mask)
    assert result["area_px2"] == 200 * 30
    assert result["mask_bbox"] == (40, 50, 239, 79)
    assert not result["mask_clean"][5:10, 5:10].any()


@pytest.mark.parametrize("use_ridge", [True, False])
def test_measure_is_rotation_invariant_for_elongated_ellipse(use_ridge):
    mask = np.zeros((160, 300), dtype=np.uint8)
    cv2.ellipse(mask, (150, 80), (100, 20), 0, 0, 360, 255, -1)
    horizontal = em.measure_mask_length(mask, use_ridge=use_ridge)
    vertical = em.measure_mask_length(np.rot90(mask).copy(), use_ridge=use_ridge)
    assert horizontal["comprimento_eixo_px"] == pytest.approx(200, abs=5)
    assert horizontal["comprimento_curva_px"] >= horizontal["comprimento_eixo_px"]
    assert vertical["comprimento_curva_px"] == pytest.approx(horizontal["comprimento_curva_px"], abs=2)
    assert vertical["largura_corpo_px"] == pytest.approx(horizontal["largura_corpo_px"], abs=1)
    assert vertical["area_px2"] == horizontal["area_px2"]


def hsv_frame(hue, saturation=255, value=255, shape=(10, 10)):
    hsv = np.empty((*shape, 3), dtype=np.uint8)
    hsv[:] = (hue, saturation, value)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


@pytest.mark.parametrize("hue", [25, 160])
def test_neon_mask_detects_yellow_and_pink(hue):
    assert em._is_neon_buoy_mask(hsv_frame(hue), np.full((10, 10), 255, np.uint8)) == (True, 1.0)


@pytest.mark.parametrize("hue, saturation, value", [(90, 255, 255), (25, 30, 255), (25, 255, 50)])
def test_neon_mask_rejects_blue_gray_and_dark_pixels(hue, saturation, value):
    frame = hsv_frame(hue, saturation, value)
    assert em._is_neon_buoy_mask(frame, np.full((10, 10), 255, np.uint8)) == (False, 0.0)


def test_neon_fraction_uses_only_selected_pixels_and_includes_threshold():
    frame = hsv_frame(90, shape=(10, 20))
    frame[:, 10:] = hsv_frame(25)
    frame.reshape(-1, 3)[:42] = hsv_frame(25, shape=(1, 42)).reshape(-1, 3)
    mask = np.zeros((10, 20), np.uint8)
    mask.reshape(-1)[:100] = 255
    # Keep precisely 42 neon pixels inside the 100-pixel mask; the rest of the
    # image may be neon without influencing this fraction.
    frame.reshape(-1, 3)[42:100] = hsv_frame(90, shape=(1, 58)).reshape(-1, 3)
    assert em._is_neon_buoy_mask(frame, mask) == (True, 0.42)
    frame.reshape(-1, 3)[41] = hsv_frame(90, shape=(1, 1))[0, 0]
    assert em._is_neon_buoy_mask(frame, mask) == (False, 0.41)


def test_neon_mask_ignores_tiny_regions():
    mask = np.zeros((10, 10), np.uint8)
    mask.reshape(-1)[:39] = 255
    assert em._is_neon_buoy_mask(hsv_frame(25), mask) == (False, 0.0)


def observation(frame, *, x=10, y=20, time=None):
    return {
        "frame": frame, "tempo_s": frame / 10 if time is None else time,
        "x1": x, "y1": y, "x2": x + 100, "y2": y + 30,
    }


def test_consolidate_empty_input_returns_no_events():
    assert em.consolidate_tracks({}, merge_gap_s=2, merge_dist_px=20, min_event_frames=2) == []


def test_consolidate_merges_fragmented_track_and_orders_observations():
    tracks = {
        9: [observation(13, x=15), observation(12, x=15)],
        3: [observation(11), observation(10)],
    }
    events = em.consolidate_tracks(tracks, merge_gap_s=1, merge_dist_px=20, min_event_frames=4)
    assert len(events) == 1
    event = events[0]
    assert event["track_ids"] == [3, 9]
    assert event["frames_set"] == {10, 11, 12, 13}
    assert [obs["frame"] for obs in event["obs"]] == [10, 11, 12, 13]
    assert event["t_start"] == 1.0
    assert event["t_end"] == 1.3
    assert event["event_id"] == 1
    # Input lists retain their original ordering.
    assert [obs["frame"] for obs in tracks[9]] == [13, 12]


def test_consolidate_rejects_a_long_temporal_gap_despite_same_position():
    tracks = {1: [observation(0), observation(1)], 2: [observation(100), observation(101)]}
    events = em.consolidate_tracks(tracks, merge_gap_s=1, merge_dist_px=20, min_event_frames=2)
    assert [event["track_ids"] for event in events] == [[1], [2]]
    assert [event["event_id"] for event in events] == [1, 2]


def test_consolidate_keeps_two_simultaneous_fish_separate():
    tracks = {
        1: [observation(10), observation(11)],
        2: [observation(10, x=400), observation(11, x=400)],
    }
    # Even a permissive distance threshold cannot merge nonoverlapping fish
    # observed in the same two frames.
    events = em.consolidate_tracks(tracks, merge_gap_s=2, merge_dist_px=1000, min_event_frames=2)
    assert [event["track_ids"] for event in events] == [[1], [2]]


def test_consolidate_merges_duplicate_boxes_but_counts_unique_frames():
    tracks = {
        1: [observation(10), observation(11)],
        2: [observation(10, x=12), observation(11, x=12)],
    }
    merged = em.consolidate_tracks(tracks, merge_gap_s=2, merge_dist_px=20, min_event_frames=2)
    assert len(merged) == 1
    assert merged[0]["track_ids"] == [1, 2]
    assert merged[0]["frames_set"] == {10, 11}
    assert em.consolidate_tracks(tracks, merge_gap_s=2, merge_dist_px=20, min_event_frames=3) == []


def test_consolidate_rejects_distant_nonoverlapping_fragment():
    tracks = {
        1: [observation(10), observation(11)],
        2: [observation(12, x=400), observation(13, x=400)],
    }
    events = em.consolidate_tracks(tracks, merge_gap_s=2, merge_dist_px=20, min_event_frames=2)
    assert [event["track_ids"] for event in events] == [[1], [2]]
