import importlib.util
import io
import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import main as api  # noqa: E402


@pytest.fixture
def monitor():
    spec = importlib.util.spec_from_file_location("fishid_monitor_for_tests", ROOT / "monitoramento_em.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Only the inference path uses PyTorch. Keep this request test CPU-only.
    with patch.dict(sys.modules, {"torch": ModuleType("torch")}):
        spec.loader.exec_module(module)
    return module


@pytest.fixture
def crop(tmp_path):
    crop_path = tmp_path / "sample.jpg"
    Image.new("RGB", (32, 32), (30, 120, 210)).save(crop_path)
    return crop_path


def test_monitor_reads_the_backend_prediction_response(monitor, crop, monkeypatch):
    from urllib.request import urlopen

    monkeypatch.setattr(api, "get_model_state", lambda: (None, None))

    def send(request, timeout):
        assert request.get_method() == "POST"
        assert request.full_url == "http://testserver/predict"
        assert timeout == 10
        response = client.post(
            "/predict", content=request.data,
            headers={"Content-Type": request.get_header("Content-type")},
        )
        response.raise_for_status()
        payload.update(response.json())
        return io.BytesIO(response.content)

    payload = {}
    with TestClient(api.app) as client, patch.object(monitor.urllib.request, "urlopen", send):
        result = monitor._classify_crop_via_fishid_api("http://testserver/", crop)

    assert monitor.urllib.request.urlopen is urlopen
    assert payload["mode"] == "demo"
    assert "top_prediction" not in payload
    assert result == (payload["display_name"], round(payload["confidence"], 4))


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"top_species": "Dourado", "confidence": 0.875}, ("Dourado", 0.875)),
        ({"scientific_name": "Xiphias gladius", "confidence": 0.125}, ("Xiphias gladius", 0.125)),
        ({"top_prediction": {"common_pt": "Atum", "confidence": 0.725}}, ("Atum", 0.725)),
        ({"confidence": 0.9}, None),
    ],
)
def test_monitor_prediction_name_fallbacks(monitor, crop, payload, expected):
    with patch.object(monitor.urllib.request, "urlopen", return_value=io.BytesIO(json.dumps(payload).encode())):
        assert monitor._classify_crop_via_fishid_api("http://testserver", crop) == expected
