import io
import json
import urllib.request
from urllib.error import URLError

import numpy as np
import pytest

from monitoramento_em import (
    _classify_crop_via_fishid_api,
    _is_neon_buoy_mask,
    consolidate_tracks,
    measure_mask_length,
)


@pytest.mark.parametrize("name_field", ["display_name", "top_species", "scientific_name"])
def test_fishid_prediction_reads_root_fields(monkeypatch, tmp_path, name_field):
    crop = tmp_path / "fish.jpg"
    crop.write_bytes(b"test crop")
    prediction = {name_field: "Thunnus albacares", "confidence": 0.912345}

    def urlopen(request, timeout):
        assert request.full_url == "http://fishid.test/predict"
        assert request.get_method() == "POST"
        assert b'name="file"; filename="fish.jpg"' in request.data
        assert timeout == 10
        return io.BytesIO(json.dumps(prediction).encode())

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    assert _classify_crop_via_fishid_api("http://fishid.test/", crop) == ("Thunnus albacares", 0.9123)


def test_fishid_offline_warning(monkeypatch, tmp_path, capsys):
    crop = tmp_path / "fish.jpg"
    crop.write_bytes(b"test crop")

    def offline(*args, **kwargs):
        raise URLError("connection refused\nsecond line")

    monkeypatch.setattr(urllib.request, "urlopen", offline)
    assert _classify_crop_via_fishid_api("http://fishid.test", crop) is None
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1 and lines[0].startswith("[WARN]")
    assert "connection refused" in lines[0]


def test_measure_mask_length():
    mask = np.zeros((60, 120), dtype=np.uint8)
    assert measure_mask_length(mask)["comprimento_curva_px"] == 0
    mask[20:40, 10:110] = 255
    measurement = measure_mask_length(mask, use_ridge=False)
    assert measurement["mask_bbox"] == (10, 20, 109, 39)
    # Each endpoint averages the three outermost columns of the mask.
    assert measurement["comprimento_eixo_px"] == pytest.approx(97)
    assert measurement["comprimento_curva_px"] == pytest.approx(97)
    assert measurement["largura_corpo_px"] == pytest.approx(20)


def test_neon_buoy_mask():
    frame = np.full((24, 24, 3), 100, dtype=np.uint8)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    assert _is_neon_buoy_mask(frame, mask) == (False, 0.0)
    mask[5:15, 5:15] = 255
    assert _is_neon_buoy_mask(frame, mask) == (False, 0.0)
    frame[5:15, 5:15] = (0, 255, 255)
    assert _is_neon_buoy_mask(frame, mask) == (True, 1.0)


def test_consolidate_tracks_merges_fragments_and_filters_short_tracks():
    def observations(x, frames):
        return [
            {"frame": frame, "tempo_s": frame / 10, "x1": x, "x2": x + 30, "y1": 0, "y2": 10}
            for frame in frames
        ]

    tracks = {
        3: observations(400, [4, 3]),
        2: observations(5, [4, 3]),
        1: observations(0, [2, 1]),
        4: observations(400, [100]),
    }
    events = consolidate_tracks(tracks, merge_gap_s=0.2, merge_dist_px=20, min_event_frames=2)
    assert [event["track_ids"] for event in events] == [[1, 2], [3]]
    assert [event["event_id"] for event in events] == [1, 2]
    assert [obs["frame"] for obs in events[0]["obs"]] == [1, 2, 3, 4]
