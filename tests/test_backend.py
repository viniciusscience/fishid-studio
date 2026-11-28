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
from storage import validate_image_quality  # noqa: E402


def _make_sharp_image(width: int = 320, height: int = 240, seed: int = 0) -> Image.Image:
    """Create a deterministic textured RGB image that passes default quality checks."""
    rng = np.random.default_rng(seed)
    grid = (np.indices((height, width)).sum(axis=0) % 2 * 60 + 85).astype(np.int16)
    noise = rng.integers(-12, 13, size=(height, width, 3), dtype=np.int16)
    rgb = np.clip(grid[:, :, None] + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _encode_jpeg(img: Image.Image, exif: Image.Exif | None = None) -> bytes:
    buf = io.BytesIO()
    if exif is not None:
        img.save(buf, format="JPEG", quality=92, exif=exif)
    else:
        img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


@pytest.fixture
def isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("LOCAL_STORAGE_DIR", str(tmp_path))
    monkeypatch.setenv("STORAGE_PREFIX", "fishid_test")
    for suffix in ("MIN_IMAGE_DIM", "MIN_SHARPNESS", "MIN_BRIGHTNESS", "MAX_BRIGHTNESS", "MIN_CONTRAST"):
        monkeypatch.delenv(f"FISHID_{suffix}", raising=False)
    return tmp_path


def test_health_and_info_endpoints(isolated_storage):
    client = TestClient(main.app)

    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json() == {"status": "ok"}

    info_resp = client.get("/info")
    assert info_resp.status_code == 200
    info = info_resp.json()
    assert info["mode"] in {"demo", "real"}
    assert len(info["species_catalog"]) == 10
    assert len(info["class_names"]) == 10
    assert info["storage_backend"] == "local"


def test_predict_endpoint_returns_ranked_probabilities():
    client = TestClient(main.app)
    img_bytes = _encode_jpeg(_make_sharp_image(300, 300, seed=7))

    resp = client.post(
        "/predict",
        files={"file": ("fish.jpg", img_bytes, "image/jpeg")},
        data={"hint_species": "Thunnus albacares"},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["mode"] == "demo"
    assert payload["scientific_name"] == "Thunnus albacares"
    assert payload["top_species"] == "Albacora-laje"
    assert payload["display_name"] == "Albacora-laje (Thunnus albacares)"
    assert 0.0 < payload["confidence"] <= 1.0
    assert len(payload["probabilities"]) == 10
    assert sum(item["probability"] for item in payload["probabilities"]) == pytest.approx(1.0, abs=1e-4)

    empty_resp = client.post("/predict", files={"file": ("empty.jpg", b"", "image/jpeg")})
    assert empty_resp.status_code == 400

    bad_resp = client.post("/predict", files={"file": ("bad.jpg", b"not-an-image", "image/jpeg")})
    assert bad_resp.status_code == 400


def test_predict_batch_endpoint_classifies_multiple_crops():
    client = TestClient(main.app)
    files = [
        ("files", ("crop_1.jpg", _encode_jpeg(_make_sharp_image(256, 256, seed=1)), "image/jpeg")),
        ("files", ("crop_2.jpg", _encode_jpeg(_make_sharp_image(256, 256, seed=2)), "image/jpeg")),
    ]
    resp = client.post("/predict/batch", files=files)
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert [item["filename"] for item in body["results"]] == ["crop_1.jpg", "crop_2.jpg"]
    for item in body["results"]:
        assert item["display_name"]
        assert item["scientific_name"]
        assert 0.0 < item["confidence"] <= 1.0


def test_predict_batch_enforces_max_files_and_max_file_size(monkeypatch):
    client = TestClient(main.app)
    tiny_jpg = _encode_jpeg(_make_sharp_image(64, 64, seed=3))

    too_many = [("files", (f"f_{i}.jpg", tiny_jpg, "image/jpeg")) for i in range(main.MAX_BATCH_FILES + 1)]
    resp_many = client.post("/predict/batch", files=too_many)
    assert resp_many.status_code == 400

    monkeypatch.setattr(main, "MAX_BATCH_FILE_BYTES", 1024)
    oversized = b"x" * 1025
    resp_big = client.post(
        "/predict/batch",
        files=[("files", ("huge.jpg", oversized, "image/jpeg"))],
    )
    assert resp_big.status_code == 413


def test_contribute_applies_exif_transpose_strips_exif_and_deduplicates(isolated_storage):
    client = TestClient(main.app)

    # Landscape 360x240 raw pixels tagged with EXIF Orientation=6 (90 deg CW -> portrait 240x360)
    # plus synthetic camera Make/Model and GPSInfo tags that must be stripped.
    img = _make_sharp_image(width=360, height=240, seed=42)
    exif = Image.Exif()
    exif[274] = 6  # Orientation: Rotate 90 CW
    exif[271] = "SecretPhoneMaker"  # Make
    exif[272] = "SuperSecretModel99"  # Model
    exif[34853] = {1: "S", 2: (26.0, 54.0, 0.0), 3: "W", 4: (48.0, 39.0, 0.0)}  # GPSInfo
    raw_with_exif = _encode_jpeg(img, exif=exif)
    assert b"SecretPhoneMaker" in raw_with_exif

    form_data = {
        "predicted_scientific": "Coryphaena hippurus",
        "predicted_common_pt": "Dourado-do-mar",
        "confidence": "0.93",
        "user_confirmed": "true",
        "user_suggested_species": "Coryphaena hippurus",
        "source": "upload",
    }

    first = client.post(
        "/contribute",
        files={"file": ("portrait_phone.jpg", raw_with_exif, "image/jpeg")},
        data=form_data,
    )
    assert first.status_code == 200
    first_body = first.json()
    assert first_body["saved"] is True
    assert first_body["duplicate"] is False
    assert first_body["sha256"]

    saved_path = Path(first_body["location"])
    assert saved_path.is_file()
    saved_bytes = saved_path.read_bytes()
    assert b"SecretPhoneMaker" not in saved_bytes
    assert b"SuperSecretModel99" not in saved_bytes

    with Image.open(saved_path) as saved_img:
        assert saved_img.size == (240, 360)
        assert len(saved_img.getexif()) == 0

    meta_path = saved_path.with_suffix(".json")
    saved_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert saved_meta["image_size"] == {"width": 240, "height": 360}
    assert saved_meta["sha256"] == first_body["sha256"]

    # Re-uploading the exact same photo must be deduplicated without creating a second file or manifest entry.
    second = client.post(
        "/contribute",
        files={"file": ("portrait_phone_dup.jpg", raw_with_exif, "image/jpeg")},
        data=form_data,
    )
    assert second.status_code == 200
    second_body = second.json()
    assert second_body["saved"] is False
    assert second_body["duplicate"] is True
    assert second_body["sample_id"] == first_body["sample_id"]
    assert second_body["sha256"] == first_body["sha256"]

    manifest_path = isolated_storage / "fishid_test" / "manifest.jsonl"
    manifest_lines = [ln for ln in manifest_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(manifest_lines) == 1
    saved_jpegs = list((isolated_storage / "fishid_test").rglob("*.jpg"))
    assert len(saved_jpegs) == 1


def test_contributions_stats_endpoint_reads_manifest(isolated_storage):
    client = TestClient(main.app)

    empty_resp = client.get("/contributions/stats")
    assert empty_resp.status_code == 200
    assert empty_resp.json() == {
        "total_samples": 0,
        "confirmed": 0,
        "needs_review": 0,
        "by_species": {},
    }

    uploads = [
        (10, True, "Thunnus albacares"),
        (11, True, "Thunnus albacares"),
        (12, False, "Xiphias gladius"),
    ]
    for seed, confirmed, species in uploads:
        resp = client.post(
            "/contribute",
            files={"file": (f"sample_{seed}.jpg", _encode_jpeg(_make_sharp_image(260, 260, seed=seed)), "image/jpeg")},
            data={
                "predicted_scientific": species,
                "user_confirmed": "true" if confirmed else "false",
                "user_suggested_species": species,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["saved"] is True

    # Append a corrupted line to ensure /contributions/stats stays resilient.
    manifest_path = isolated_storage / "fishid_test" / "manifest.jsonl"
    with manifest_path.open("a", encoding="utf-8") as mf:
        mf.write("{malformed-json\n")

    stats_resp = client.get("/contributions/stats")
    assert stats_resp.status_code == 200
    assert stats_resp.json() == {
        "total_samples": 3,
        "confirmed": 2,
        "needs_review": 1,
        "by_species": {
            "Thunnus albacares": 2,
            "Xiphias gladius": 1,
        },
    }


def test_validate_image_quality_accepts_sharp_and_rejects_poor_photos(isolated_storage):
    sharp = _make_sharp_image(256, 256, seed=99)
    ok, reason, metrics = validate_image_quality(sharp)
    assert ok is True
    assert reason == "ok"
    assert metrics["sharpness"] >= 35.0

    low_res = _make_sharp_image(120, 120, seed=99)
    ok_low, msg_low, _ = validate_image_quality(low_res)
    assert ok_low is False
    assert "resolução" in msg_low

    blurry = Image.fromarray(np.full((256, 256, 3), 120, dtype=np.uint8), mode="RGB")
    ok_blur, msg_blur, _ = validate_image_quality(blurry)
    assert ok_blur is False
    assert "borrada" in msg_blur

    dark_pixels = (np.indices((256, 256)).sum(axis=0) % 2 * 20).astype(np.uint8)
    dark = Image.fromarray(dark_pixels).convert("RGB")
    ok_dark, msg_dark, _ = validate_image_quality(dark)
    assert ok_dark is False
    assert "escura" in msg_dark

    bright_pixels = (242 + (np.indices((256, 256)).sum(axis=0) % 2 * 12)).astype(np.uint8)
    bright = Image.fromarray(bright_pixels).convert("RGB")
    ok_bright, msg_bright, _ = validate_image_quality(bright)
    assert ok_bright is False
    assert "excesso de luz" in msg_bright
