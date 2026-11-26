"""Contribution indexing tests; cloud clients are in-memory stubs only."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
from PIL import Image
import pytest

BACKEND = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND))
import storage  # noqa: E402


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("LOCAL_STORAGE_DIR", str(tmp_path))
    monkeypatch.setenv("STORAGE_PREFIX", "dataset")
    monkeypatch.setenv("STORAGE_BUCKET", "fish-tests")
    monkeypatch.delenv("S3_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("GCS_PROJECT_ID", raising=False)
    return tmp_path


def manifest(store):
    return store / "dataset" / "manifest.jsonl"


def records(store):
    return [json.loads(line) for line in manifest(store).read_text().splitlines()]


def test_duplicate_retains_first_sample_and_metadata(store):
    first_meta = {"user_suggested_species": "Thunnus obesus", "user_confirmed": True, "note": "first"}
    first = storage.save_contribution(b"identical-jpeg", first_meta)
    duplicate = storage.save_contribution(b"identical-jpeg", {"user_suggested_species": "Other", "note": "changed"})
    assert first == {**duplicate, "duplicate": False}
    assert first["saved"] and duplicate["duplicate"]
    assert len(list(store.rglob("*.jpg"))) == 1
    row, = records(store)
    assert row["note"] == "first"
    assert row["user_suggested_species"] == "Thunnus obesus"
    assert row["image_sha256"] == hashlib.sha256(b"identical-jpeg").hexdigest()
    assert json.loads(Path(first["location"]).with_suffix(".json").read_text()) == row
    assert storage.get_contribution_stats() == {
        "total": 1, "confirmed": 1, "needs_review": 0, "by_species": {"Thunnus obesus": 1}}


def test_internal_identity_cannot_be_overridden(store):
    fake = {"sample_id": "forged", "timestamp_utc": "yesterday", "storage_backend": "s3",
            "image_key": "../../other.jpg", "image_sha256": "fake", "storage_destination": "fake",
            "location": "/outside.jpg", "user_suggested_species": "../../A fish"}
    result = storage.save_contribution(b"photo", fake)
    row, = records(store)
    assert result["sample_id"] != fake["sample_id"]
    assert row["timestamp_utc"] != "yesterday"
    assert row["storage_backend"] == result["backend"] == "local"
    assert row["image_sha256"] == hashlib.sha256(b"photo").hexdigest()
    assert row["storage_destination"] != "fake"
    assert Path(result["location"]).is_relative_to(store)
    assert ".." not in row["image_key"]
    assert row["location"] == result["location"]
    assert fake["sample_id"] == "forged"  # Caller data was not mutated.


def test_different_images_count_by_review_and_species(store):
    storage.save_contribution(b"one", {"predicted_scientific": "A", "user_confirmed": True})
    storage.save_contribution(b"two", {"predicted_scientific": "A", "user_suggested_species": "B"})
    storage.save_contribution(b"three", {"predicted_scientific": "A"})
    storage.save_contribution(b"four", {})
    assert storage.get_contribution_stats() == {
        "total": 4, "confirmed": 1, "needs_review": 3,
        "by_species": {"A": 2, "B": 1, "nova_especie": 1}}


def test_empty_stats_and_legacy_manifest(store):
    assert storage.get_contribution_stats() == {"total": 0, "confirmed": 0, "needs_review": 0, "by_species": {}}
    assert not manifest(store).exists()
    manifest(store).parent.mkdir()
    legacy = {"sample_id": "old", "image_key": "old.jpg", "predicted_scientific": "Legacy", "user_confirmed": True}
    manifest(store).write_text(json.dumps(legacy) + '\n{}\n[]\nnot-json\n{"truncated":', encoding="utf-8")
    assert storage.get_contribution_stats() == {
        "total": 1, "confirmed": 1, "needs_review": 0, "by_species": {"Legacy": 1}}
    saved = storage.save_contribution(b"new", {"predicted_scientific": "New"})
    assert saved["saved"] and not saved["duplicate"]
    assert storage.get_contribution_stats()["total"] == 2


def test_prefix_or_local_directory_changes_are_separate_destinations(store, monkeypatch):
    first = storage.save_contribution(b"photo", {})
    monkeypatch.setenv("STORAGE_PREFIX", "other")
    second = storage.save_contribution(b"photo", {})
    assert not second["duplicate"] and second["location"] != first["location"]
    monkeypatch.setenv("LOCAL_STORAGE_DIR", str(store / "another-root"))
    third = storage.save_contribution(b"photo", {})
    assert not third["duplicate"] and third["location"] != second["location"]
    assert storage.get_contribution_stats()["total"] == 1


def test_empty_prefix_stays_inside_local_directory(store, monkeypatch):
    monkeypatch.setenv("STORAGE_PREFIX", "")
    result = storage.save_contribution(b"photo", {})
    assert Path(result["location"]).is_relative_to(store)
    assert (store / "manifest.jsonl").exists()


def test_concurrent_threads_save_one_sample(store):
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: storage.save_contribution(b"same-photo", {}), range(16)))
    assert sum(not result["duplicate"] for result in results) == 1
    assert len({result["sample_id"] for result in results}) == 1
    assert len(records(store)) == 1
    assert len(list(store.rglob("*.jpg"))) == 1


def test_concurrent_processes_save_one_sample(store):
    # Children invoke storage only, using the temporary local destination above.
    script = "import json,storage; print(json.dumps(storage.save_contribution(b'one-photo', {})))"
    env = dict(os.environ, PYTHONPATH=str(BACKEND))
    children = [subprocess.Popen([sys.executable, "-c", script], env=env, text=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(4)]
    results = []
    try:
        for child in children:
            out, err = child.communicate(timeout=30)
            assert child.returncode == 0, err
            results.append(json.loads(out))
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.communicate()
    assert sum(not result["duplicate"] for result in results) == 1
    assert len({result["sample_id"] for result in results}) == 1
    assert len(records(store)) == 1


def test_local_metadata_write_failure_leaves_no_sample_or_manifest(store, monkeypatch):
    original = Path.write_bytes
    def fail_metadata(path, body):
        if path.suffix == ".json":
            raise OSError("metadata disk error")
        return original(path, body)
    monkeypatch.setattr(Path, "write_bytes", fail_metadata)
    with pytest.raises(OSError, match="metadata disk error"):
        storage.save_contribution(b"photo", {})
    assert not manifest(store).exists()
    assert not list(store.rglob("*.jpg"))
    assert not list(store.rglob("*.json"))


def test_manifest_publish_failure_keeps_old_index_and_removes_new_files(store, monkeypatch):
    first = storage.save_contribution(b"first", {})
    before = manifest(store).read_bytes()
    def fail_replace(*args):
        raise OSError("cannot publish index")
    monkeypatch.setattr(storage.os, "replace", fail_replace)
    with pytest.raises(OSError, match="cannot publish index"):
        storage.save_contribution(b"second", {})
    assert manifest(store).read_bytes() == before
    assert [str(path) for path in store.rglob("*.jpg")] == [first["location"]]
    assert not list(store.rglob(".manifest-*"))


class CloudStub:
    def __init__(self):
        self.objects = {}
        self.writes = []
        self.deleted = []
        self.fail_at = None

    def put(self, bucket, key, body, content_type):
        self.writes.append((bucket, key, content_type))
        if len(self.writes) == self.fail_at:
            raise RuntimeError("simulated upload failure")
        self.objects[bucket, key] = body

    def delete(self, bucket, key):
        self.deleted.append((bucket, key))
        self.objects.pop((bucket, key), None)

    def put_object(self, **kw):
        self.put(kw["Bucket"], kw["Key"], kw["Body"], kw["ContentType"])

    def delete_object(self, **kw):
        self.delete(kw["Bucket"], kw["Key"])

    def bucket(self, name):
        def blob(key):
            return types.SimpleNamespace(
                upload_from_string=lambda body, content_type: self.put(name, key, body, content_type),
                delete=lambda: self.delete(name, key))
        return types.SimpleNamespace(blob=blob)


@pytest.fixture(params=["s3", "gcs"])
def cloud(store, monkeypatch, request):
    fake = CloudStub()
    monkeypatch.setenv("STORAGE_BACKEND", request.param)
    boto = types.ModuleType("boto3")
    boto.client = lambda *args, **kwargs: fake
    monkeypatch.setitem(sys.modules, "boto3", boto)
    google = types.ModuleType("google")
    cloud_module = types.ModuleType("google.cloud")
    storage_module = types.ModuleType("google.cloud.storage")
    storage_module.Client = lambda **kwargs: fake
    google.cloud = cloud_module
    cloud_module.storage = storage_module
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.cloud", cloud_module)
    monkeypatch.setitem(sys.modules, "google.cloud.storage", storage_module)
    return fake, request.param


def test_cloud_saves_index_and_skips_duplicate_uploads(store, cloud):
    fake, backend = cloud
    first = storage.save_contribution(b"cloud-photo", {"predicted_scientific": "Cloud fish"})
    duplicate = storage.save_contribution(b"cloud-photo", {"predicted_scientific": "Changed"})
    assert first["backend"] == backend
    assert first["saved"] and not first["duplicate"]
    assert duplicate["duplicate"] and duplicate["sample_id"] == first["sample_id"]
    assert len(fake.writes) == len(fake.objects) == 2
    assert len(records(store)) == 1
    assert not list(store.rglob("*.jpg"))
    assert storage.get_contribution_stats() == {
        "total": 1, "confirmed": 0, "needs_review": 1, "by_species": {"Cloud fish": 1}}
    meta, = [json.loads(body) for (bucket, key), body in fake.objects.items() if key.endswith(".json")]
    assert meta == records(store)[0]


def test_cloud_bucket_change_does_not_deduplicate_or_mix_stats(store, cloud, monkeypatch):
    fake, backend = cloud
    storage.save_contribution(b"photo", {"predicted_scientific": "First bucket"})
    monkeypatch.setenv("STORAGE_BUCKET", "another-bucket")
    second = storage.save_contribution(b"photo", {"predicted_scientific": "Second bucket"})
    assert not second["duplicate"] and len(fake.objects) == 4
    assert len(records(store)) == 2
    assert storage.get_contribution_stats()["by_species"] == {"Second bucket": 1}
    monkeypatch.setenv("STORAGE_BUCKET", "fish-tests")
    assert storage.get_contribution_stats()["by_species"] == {"First bucket": 1}


def test_s3_endpoint_or_gcs_project_is_part_of_destination(store, cloud, monkeypatch):
    fake, backend = cloud
    storage.save_contribution(b"photo", {})
    variable = "S3_ENDPOINT_URL" if backend == "s3" else "GCS_PROJECT_ID"
    monkeypatch.setenv(variable, "https://example.invalid" if backend == "s3" else "other-project")
    result = storage.save_contribution(b"photo", {})
    assert not result["duplicate"] and len(fake.writes) == 4
    assert storage.get_contribution_stats()["total"] == 1


def test_cloud_upload_failure_rolls_back_and_does_not_poison_retry(store, cloud):
    fake, _ = cloud
    fake.fail_at = 2
    with pytest.raises(RuntimeError, match="simulated upload failure"):
        storage.save_contribution(b"photo", {})
    assert fake.objects == {}
    assert not manifest(store).exists()
    fake.fail_at = None
    result = storage.save_contribution(b"photo", {})
    assert result["saved"] and not result["duplicate"]
    assert len(records(store)) == 1


def test_cloud_manifest_failure_rolls_back_uploads(store, cloud, monkeypatch):
    fake, _ = cloud
    def fail_replace(*args):
        raise OSError("manifest full")
    monkeypatch.setattr(storage.os, "replace", fail_replace)
    with pytest.raises(OSError, match="manifest full"):
        storage.save_contribution(b"photo", {})
    assert fake.objects == {}
    assert not manifest(store).exists()


def test_missing_cloud_bucket_does_not_write_an_index(store, cloud, monkeypatch):
    fake, _ = cloud
    monkeypatch.setenv("STORAGE_BUCKET", "")
    with pytest.raises(RuntimeError, match="STORAGE_BUCKET"):
        storage.save_contribution(b"photo", {})
    assert not fake.writes and not manifest(store).exists()


@pytest.fixture
def quality_defaults(monkeypatch):
    for name in ("MIN_IMAGE_DIM", "MIN_SHARPNESS", "MIN_BRIGHTNESS", "MAX_BRIGHTNESS", "MIN_CONTRAST"):
        monkeypatch.delenv("FISHID_" + name, raising=False)


def checkerboard(low, high):
    gray = np.where(np.indices((224, 224)).sum(axis=0) % 2, high, low).astype(np.uint8)
    return Image.fromarray(gray).convert("RGB")


@pytest.mark.parametrize(("photo", "message"), [
    (Image.new("RGB", (223, 224), "gray"), "resolução"),
    (Image.new("RGB", (224, 224), "gray"), "borrada"),
    (checkerboard(0, 40), "escura"),
    (checkerboard(245, 255), "excesso de luz"),
    (checkerboard(109, 111), "sem contraste"),
])
def test_quality_rejects_each_documented_failure(quality_defaults, photo, message):
    accepted, reason, metrics = storage.validate_image_quality(photo)
    assert not accepted and message in reason and metrics


def test_quality_threshold_boundaries_are_inclusive(quality_defaults, monkeypatch):
    photo = checkerboard(80, 140)
    accepted, _, metrics = storage.validate_image_quality(photo)
    assert accepted
    monkeypatch.setenv("FISHID_MIN_IMAGE_DIM", "224")
    monkeypatch.setenv("FISHID_MIN_SHARPNESS", str(metrics["sharpness"]))
    monkeypatch.setenv("FISHID_MIN_BRIGHTNESS", "110")
    monkeypatch.setenv("FISHID_MAX_BRIGHTNESS", "110")
    monkeypatch.setenv("FISHID_MIN_CONTRAST", "30")
    assert storage.validate_image_quality(photo)[0]
