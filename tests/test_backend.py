"""API regression coverage with demo/fake inference and isolated local storage."""

import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import main  # noqa: E402


def image_bytes(seed=1, *, exif=None, size=(320, 256)):
    pixels = np.random.default_rng(seed).integers(32, 224, size=(size[1], size[0], 3), dtype=np.uint8)
    buffer = io.BytesIO()
    options = {"exif": exif} if exif is not None else {}
    Image.fromarray(pixels).save(buffer, format="JPEG", quality=95, **options)
    return buffer.getvalue()


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("LOCAL_STORAGE_DIR", str(tmp_path / "contributions"))
    monkeypatch.setenv("STORAGE_PREFIX", "samples")
    monkeypatch.setattr(main, "MODEL_PATH", str(tmp_path / "no-model.keras"))
    monkeypatch.setattr(main, "_MODEL_CACHE", None)
    monkeypatch.setattr(main, "_MODEL_LOAD_ERROR", None)
    monkeypatch.setattr(main, "_MODEL_CHECKED_MTIME", None)
    # Keep quality tests independent of environment settings on a developer's host.
    for key in (
        "FISHID_MIN_IMAGE_DIM", "FISHID_MIN_SHARPNESS", "FISHID_MIN_BRIGHTNESS",
        "FISHID_MAX_BRIGHTNESS", "FISHID_MIN_CONTRAST",
    ):
        monkeypatch.delenv(key, raising=False)
    with TestClient(main.app) as api:
        yield api


def assert_prediction(value, mode="demo"):
    assert value["mode"] == mode
    assert 0 <= value["confidence"] <= 1
    assert value["latency_ms"] >= 0
    probabilities = value["probabilities"]
    assert probabilities
    assert sum(row["probability"] for row in probabilities) == pytest.approx(1)
    assert [row["probability"] for row in probabilities] == sorted(
        [row["probability"] for row in probabilities], reverse=True,
    )
    assert value["scientific_name"] == probabilities[0]["scientific"]
    assert value["top_species"] == probabilities[0]["common_pt"]
    assert value["display_name"] == probabilities[0]["species"]


def test_health_and_info_work_without_a_model(client):
    assert client.get("/health").json() == {"status": "ok"}
    response = client.get("/info")
    assert response.status_code == 200
    info = response.json()
    assert info["mode"] == "demo"
    assert info["load_error"] is None
    assert info["total_params"] is None
    assert info["input_size"] == [320, 320]
    assert len(info["species_catalog"]) == len(info["class_names"]) == 10
    assert info["storage_backend"] == "local"


def test_predict_returns_consistent_public_fields_and_hint(client):
    payload = image_bytes()
    response = client.post(
        "/predict", files={"file": ("fish.jpg", payload, "image/jpeg")},
        data={"hint_species": "Xiphias gladius"},
    )
    assert response.status_code == 200
    prediction = response.json()
    assert_prediction(prediction)
    assert prediction["scientific_name"] == "Xiphias gladius"


@pytest.mark.parametrize("raw", [b"", b"not an image"])
@pytest.mark.parametrize("endpoint", ["/predict", "/contribute"])
def test_empty_and_invalid_images_are_rejected(client, endpoint, raw):
    response = client.post(endpoint, files={"file": ("bad.jpg", raw, "image/jpeg")})
    assert response.status_code == 400
    assert response.json()["detail"]


@pytest.mark.parametrize("endpoint", ["/predict", "/contribute", "/predict/batch"])
def test_missing_file_is_rejected(client, endpoint):
    assert client.post(endpoint).status_code == 422


@pytest.mark.parametrize("endpoint", ["/predict", "/contribute", "/predict/batch"])
def test_per_file_size_limit_rejects_before_decoding(client, monkeypatch, endpoint):
    monkeypatch.setattr(main, "MAX_IMAGE_BYTES", 1024)
    field = "files" if endpoint.endswith("/batch") else "file"
    response = client.post(endpoint, files={field: ("large.jpg", b"x" * 1025, "image/jpeg")})
    assert response.status_code == 413


def test_image_at_the_byte_limit_is_accepted(client, monkeypatch):
    payload = image_bytes()
    monkeypatch.setattr(main, "MAX_IMAGE_BYTES", len(payload))
    response = client.post("/predict", files={"file": ("fish.jpg", payload, "image/jpeg")})
    assert response.status_code == 200
    assert_prediction(response.json())


def test_fake_model_path_is_used_for_predict_and_info(client, monkeypatch):
    calls = []

    class FakeModel:
        def __call__(self, batch, training=False):
            calls.append(batch)
            assert training is False
            return np.array([[0.2, 0.8]])

    bundle = {
        "model": FakeModel(), "input_size": (8, 12), "num_classes": 2,
        "model_name": "test-model", "total_params": 100, "trainable_params": 80,
    }
    monkeypatch.setattr(main, "get_model_state", lambda: (bundle, None))
    monkeypatch.setattr(main, "NORMALIZE_DIV_255", True)
    response = client.post("/predict", files={"file": ("fish.jpg", image_bytes(), "image/jpeg")})
    assert response.status_code == 200
    assert_prediction(response.json(), mode="real")
    assert response.json()["confidence"] == pytest.approx(0.8)
    assert len(calls) == 1
    assert calls[0].shape == (1, 8, 12, 3)
    assert calls[0].dtype == np.float32
    assert 0 <= calls[0].min() <= calls[0].max() <= 1
    info = client.get("/info").json()
    assert info["mode"] == "real"
    assert info["input_size"] == [8, 12]
    assert info["total_params"] == 100


def test_batch_preserves_input_order_and_matches_single_predictions(client):
    payloads = [image_bytes(seed) for seed in (2, 9, 5)]
    expected = [
        client.post("/predict", files={"file": ("same-name.jpg", payload, "image/jpeg")}).json()
        for payload in payloads
    ]
    response = client.post(
        "/predict/batch", files=[("files", ("same-name.jpg", payload, "image/jpeg")) for payload in payloads],
    )
    assert response.status_code == 200
    result = response.json()
    assert result["count"] == 3
    assert result["results"] == [
        {"index": index, "filename": "same-name.jpg", **prediction}
        for index, prediction in enumerate(expected)
    ]


def test_batch_accepts_32_files_and_rejects_33(client):
    payload = image_bytes(size=(16, 16))
    uploads = [("files", (f"fish-{index}.jpg", payload, "image/jpeg")) for index in range(33)]
    allowed = client.post("/predict/batch", files=uploads[:32])
    assert allowed.status_code == 200
    assert allowed.json()["count"] == 32
    assert client.post("/predict/batch", files=uploads).status_code == 413


def test_batch_identifies_bad_file_and_does_not_return_partial_results(client):
    response = client.post(
        "/predict/batch", files=[
            ("files", ("valid.jpg", image_bytes(), "image/jpeg")),
            ("files", ("broken.jpg", b"invalid", "image/jpeg")),
        ],
    )
    assert response.status_code == 400
    assert response.json()["detail"]["index"] == 1
    assert response.json()["detail"]["filename"] == "broken.jpg"
    assert "results" not in response.json()


@pytest.fixture
def camera_jpeg():
    exif = Image.Exif()
    exif[274] = 6  # Rotate 90 degrees clockwise for display.
    exif[271] = "Private camera maker"
    exif[272] = "Private camera model"
    exif[34853] = {1: "N", 2: (10.0, 20.0, 30.0), 3: "E", 4: (40.0, 50.0, 0.0)}
    payload = image_bytes(exif=exif)
    with Image.open(io.BytesIO(payload)) as image:
        assert image.getexif().get_ifd(34853)  # Ensure the input really has GPS data.
        expected_pixels = np.rot90(np.asarray(image.convert("RGB")), k=3)
    return payload, expected_pixels


@pytest.mark.parametrize("endpoint", ["/predict", "/predict/batch"])
def test_inference_honors_exif_orientation(client, monkeypatch, camera_jpeg, endpoint):
    payload, expected = camera_jpeg
    captured = []
    original_predict = main.predict_rgb_array

    def observe(image_rgb, bundle, hint_species=None):
        captured.append(image_rgb)
        return original_predict(image_rgb, bundle, hint_species)

    monkeypatch.setattr(main, "predict_rgb_array", observe)
    field = "files" if endpoint.endswith("/batch") else "file"
    response = client.post(endpoint, files={field: ("camera.jpg", payload, "image/jpeg")})
    assert response.status_code == 200
    assert len(captured) == 1
    np.testing.assert_array_equal(captured[0], expected)


def test_contribute_stores_oriented_jpeg_without_gps_or_device_exif(client, camera_jpeg):
    payload, expected = camera_jpeg
    response = client.post(
        "/contribute", files={"file": ("camera.jpg", payload, "image/jpeg")},
        data={
            "predicted_scientific": "Coryphaena hippurus", "user_confirmed": "true",
            "confidence": "0.8", "notes": " field test ",
        },
    )
    assert response.status_code == 200
    result = response.json()
    assert result["saved"] is True
    location = Path(result["location"])
    with Image.open(location) as saved:
        assert saved.format == "JPEG"
        assert saved.mode == "RGB"
        assert saved.size == (expected.shape[1], expected.shape[0])
        assert not saved.getexif()
        assert "exif" not in saved.info
    metadata = json.loads(location.with_suffix(".json").read_text())
    assert metadata["image_size"] == {"width": expected.shape[1], "height": expected.shape[0]}
    assert metadata["user_confirmed"] is True
    assert metadata["notes"] == "field test"


def test_low_resolution_contribution_is_not_saved(client, tmp_path):
    response = client.post(
        "/contribute", files={"file": ("tiny.jpg", image_bytes(size=(32, 32)), "image/jpeg")},
    )
    assert response.status_code == 422
    assert not list((tmp_path / "contributions").rglob("*.jpg"))


def test_stats_count_confirmed_review_and_species_from_saved_contributions(client):
    empty = client.get("/contributions/stats")
    assert empty.status_code == 200
    assert empty.json() == {"total": 0, "confirmed": 0, "needs_review": 0, "by_species": {}}
    for seed, confirmed, species in ((10, "true", "Coryphaena hippurus"), (11, "false", "Xiphias gladius")):
        response = client.post(
            "/contribute", files={"file": ("fish.jpg", image_bytes(seed), "image/jpeg")},
            data={"predicted_scientific": species, "user_confirmed": confirmed},
        )
        assert response.status_code == 200
    stats = client.get("/contributions/stats")
    assert stats.status_code == 200
    assert stats.json() == {
        "total": 2, "confirmed": 1, "needs_review": 1,
        "by_species": {"Coryphaena hippurus": 1, "Xiphias gladius": 1},
    }


def test_storage_error_is_reported_without_success(client, monkeypatch):
    def fail_save(*args):
        raise OSError("test storage unavailable")

    monkeypatch.setattr(main, "save_contribution", fail_save)
    response = client.post("/contribute", files={"file": ("fish.jpg", image_bytes(), "image/jpeg")})
    assert response.status_code == 500
    assert "saved" not in response.json()


def test_new_endpoints_are_documented(client):
    schema = client.get("/openapi.json").json()
    for path, method, fields in (
        ("/predict/batch", "post", {"count", "results"}),
        ("/contributions/stats", "get", {"total", "confirmed", "needs_review", "by_species"}),
    ):
        operation = schema["paths"][path][method]
        assert operation["summary"]
        assert operation["description"]
        reference = operation["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
        model = schema["components"]["schemas"][reference.rsplit("/", 1)[-1]]
        assert fields <= model["properties"].keys()
