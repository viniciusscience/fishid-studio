import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.main import app, get_storage_config

client = TestClient(app)

def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

def test_info_endpoint():
    r = client.get("/info")
    assert r.status_code == 200
    data = r.json()
    assert "species_catalog" in data
    assert len(data["species_catalog"]) == 10
    assert data.get("demo_mode", True) is True

def test_predict_success(tmp_path: Path):
    # Create a simple 300x300 white image.
    from PIL import Image
    img = Image.new("RGB", (300, 300), color="white")
    buf = Path(tmp_path / "test.jpg")
    img.save(buf, format="JPEG")
    with buf.open("rb") as f:
        files = {"file": ("test.jpg", f, "image/jpeg")}
        r = client.post("/predict", files=files)
    assert r.status_code == 200
    resp = r.json()
    assert "display_name" in resp
    assert "confidence" in resp
    assert 0.0 <= resp["confidence"] <= 1.0

def test_predict_batch_limits(tmp_path: Path):
    from PIL import Image
    files = []
    for i in range(33):  # exceed MAX_BATCH_SIZE (32)
        img = Image.new("RGB", (250, 250), color="white")
        path = tmp_path / f"img{i}.jpg"
        img.save(path, format="JPEG")
        files.append(("files", (f"img{i}.jpg", path.open("rb"), "image/jpeg")))
    r = client.post("/predict/batch", files=files)
    assert r.status_code == 400
    # Now test size limit (create a large dummy file >15MiB)
    large_path = tmp_path / "large.jpg"
    large_path.write_bytes(b"0" * (15 * 1024 * 1024 + 1))
    r = client.post("/predict/batch", files={"files": ("large.jpg", large_path.open("rb"), "image/jpeg")})
    assert r.status_code == 413

def test_contribute_and_deduplication(tmp_path: Path, monkeypatch):
    # Override storage config to use a temporary directory.
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("LOCAL_STORAGE_PATH", str(tmp_path))
    from backend.storage import get_storage_config
    cfg = get_storage_config()
    # Create a valid image.
    from PIL import Image
    img = Image.new("RGB", (300, 300), color="white")
    buf = Path(tmp_path / "valid.jpg")
    img.save(buf, format="JPEG")
    with buf.open("rb") as f:
        files = {"file": ("valid.jpg", f, "image/jpeg"), "species": (None, "Thunnus albacares")}
        r = client.post("/contribute", files=files)
    assert r.status_code == 200
    saved_path = Path(r.json()["saved_path"]).resolve()
    assert saved_path.is_file()
    # Submit the same image again – should return the same path (deduplication).
    with buf.open("rb") as f:
        files = {"file": ("valid.jpg", f, "image/jpeg"), "species": (None, "Thunnus albacares")}
        r2 = client.post("/contribute", files=files)
    assert r2.status_code == 200
    assert Path(r2.json()["saved_path"]).resolve() == saved_path

def test_contributions_stats(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("LOCAL_STORAGE_PATH", str(tmp_path))
    # Ensure manifest exists with two entries.
    from backend.storage import save_contribution, get_storage_config
    cfg = get_storage_config()
    img_bytes = Image.new("RGB", (300, 300), color="white").tobytes()
    # Save two contributions.
    save_contribution(img_bytes, "Thunnus albacares", None, cfg)
    save_contribution(img_bytes, "Xiphias gladius", None, cfg)  # different species
    r = client.get("/contributions/stats")
    assert r.status_code == 200
    stats = r.json()
    assert stats["total_samples"] >= 2
    assert stats["by_species"]["Thunnus albacares"] >= 1
    assert stats["by_species"]["Xiphias gladius"] >= 1
