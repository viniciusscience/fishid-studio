import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageOps


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "backend"))
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("STORAGE_PREFIX", "samples")
    monkeypatch.setenv("LOCAL_STORAGE_DIR", str(tmp_path))
    backend = import_module("main")
    monkeypatch.setattr(backend, "MODEL_PATH", str(tmp_path / "missing.keras"))
    with TestClient(backend.app) as test_client:
        yield test_client


def jpeg(orientation=1, device="Test phone", seed=0):
    rgb = np.random.default_rng(seed).integers(0, 256, (240, 320, 3), dtype=np.uint8)
    image = Image.fromarray(rgb)
    exif = image.getexif()
    exif[274] = orientation
    exif[272] = device
    exif[34853] = {1: "N", 3: "W"}
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", exif=exif)
    return buffer.getvalue()


def upload(raw, field="file"):
    return (field, ("sample.jpg", raw, "image/jpeg"))


def test_health_and_info(client):
    assert client.get("/health").json() == {"status": "ok"}
    response = client.get("/info")
    assert response.status_code == 200
    info = response.json()
    assert info["mode"] == "demo"
    assert len(info["species_catalog"]) == len(info["class_names"]) == 10
    assert info["storage_backend"] == "local"


def test_predict_and_batch(client):
    photos = [jpeg(orientation=6), jpeg(seed=1)]
    single = client.post("/predict", files=[upload(photos[0])])
    assert single.status_code == 200
    prediction = single.json()
    assert prediction["display_name"] and prediction["top_species"] and prediction["scientific_name"]
    assert 0 <= prediction["confidence"] <= 1
    assert len(prediction["probabilities"]) == 10
    with Image.open(io.BytesIO(photos[0])) as image:
        oriented = io.BytesIO()
        ImageOps.exif_transpose(image).save(oriented, "PNG")
    assert client.post("/predict", files=[upload(oriented.getvalue())]).json() == prediction

    batch = client.post("/predict/batch", files=[upload(photo, "files") for photo in photos])
    assert batch.status_code == 200
    second = client.post("/predict", files=[upload(photos[1])]).json()
    assert second != prediction
    assert batch.json()["predictions"] == [prediction, second]


def test_batch_limits(client):
    photo = jpeg()
    max_bytes = 15 * 1024 * 1024
    largest = photo.ljust(max_bytes, b"\x00")
    files = [upload(photo, "files")] * 31
    accepted = client.post("/predict/batch", files=[*files, upload(largest, "files")])
    assert accepted.status_code == 200
    assert len(accepted.json()["predictions"]) == 32
    assert client.post("/predict/batch", files=files + files[:2]).status_code == 413
    assert client.post("/predict/batch", files=[upload(largest + b"\x00", "files")]).status_code == 413
    assert client.post("/predict/batch").status_code == 422


@pytest.mark.parametrize("contents", [b"", b"invalid image"])
@pytest.mark.parametrize(("route", "field"), [("/predict", "file"), ("/predict/batch", "files"), ("/contribute", "file")])
def test_invalid_image(client, contents, route, field):
    assert client.post(route, files=[upload(contents, field)]).status_code == 400


def test_contribute_privacy_deduplication_and_stats(client, tmp_path):
    stats = "/contributions/stats"
    assert client.get(stats).json() == {"total_samples": 0, "confirmed": 0, "needs_review": 0, "by_species": {}}
    labels = {
        "predicted_scientific": "Thunnus obesus",
        "user_suggested_species": "Thunnus albacares",
        "user_confirmed": "true",
    }
    contribution = client.post("/contribute", files=[upload(jpeg(orientation=6))], data=labels)
    assert contribution.status_code == 200
    original = contribution.json()
    duplicate = client.post("/contribute", files=[upload(jpeg(orientation=6, device="Another phone"))])
    assert duplicate.status_code == 200
    assert duplicate.json() == {**original, "duplicate": True}
    stored = Path(original["location"])
    with Image.open(stored) as image:
        assert image.size == (240, 320)
        assert not image.getexif()
    manifest = tmp_path / "samples" / "manifest.jsonl"
    entry = json.loads(manifest.read_text(encoding="utf-8"))
    assert entry["sha256"] == hashlib.sha256(stored.read_bytes()).hexdigest()
    assert entry["image_size"] == {"width": 240, "height": 320}
    second = client.post("/contribute", files=[upload(jpeg(seed=1))], data={"predicted_scientific": "Thunnus obesus"})
    assert second.status_code == 200
    assert len(list(tmp_path.rglob("*.jpg"))) == 2
    assert client.get(stats).json() == {
        "total_samples": 2, "confirmed": 1, "needs_review": 1,
        "by_species": {"Thunnus albacares": 1, "Thunnus obesus": 1},
    }


def test_legacy_manifest_and_concurrent_retry(client, tmp_path):
    manifest = tmp_path / "samples" / "manifest.jsonl"
    manifest.parent.mkdir()
    legacy = {"sample_id": "legacy", "predicted_scientific": "Xiphias gladius", "user_confirmed": True}
    manifest.write_text("invalid line\n" + json.dumps(legacy), encoding="utf-8")
    photo = jpeg()
    with ThreadPoolExecutor(2) as executor:
        results = list(executor.map(lambda _: client.post("/contribute", files=[upload(photo)]), range(2)))
    assert [result.status_code for result in results] == [200, 200]
    assert results[0].json()["sample_id"] == results[1].json()["sample_id"]
    assert sum(result.json().get("duplicate", False) for result in results) == 1
    assert client.get("/contributions/stats").json() == {
        "total_samples": 2, "confirmed": 1, "needs_review": 1,
        "by_species": {"Xiphias gladius": 1, "nova_especie": 1},
    }


def test_contribution_retry_after_failed_manifest_write(client, monkeypatch, tmp_path):
    storage = import_module("storage")

    def fail(*args):
        raise OSError("manifest write failed")

    photo = jpeg()
    with monkeypatch.context() as patch:
        patch.setattr(storage, "_append_manifest", fail)
        assert client.post("/contribute", files=[upload(photo)]).status_code == 500
    assert not list(tmp_path.rglob("*.jpg"))
    assert not list(tmp_path.rglob("*.json"))
    assert client.post("/contribute", files=[upload(photo)]).status_code == 200
    assert len(list(tmp_path.rglob("*.jpg"))) == 1
    assert client.get("/contributions/stats").json()["total_samples"] == 1


@pytest.mark.parametrize(("size", "accepted"), [((128, 240), False), ((240, 240), True)])
def test_image_quality(size, accepted, client):
    storage = import_module("storage")
    with Image.open(io.BytesIO(jpeg())) as image:
        cropped = image.crop((0, 0, *size))
        assert storage.validate_image_quality(cropped)[0] is accepted
        if not accepted:
            buffer = io.BytesIO()
            cropped.save(buffer, "JPEG")
            assert client.post("/contribute", files=[upload(buffer.getvalue())]).status_code == 422
