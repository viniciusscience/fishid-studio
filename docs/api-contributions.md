# Batch predictions and contribution storage

`POST /predict` still returns its prediction fields at the JSON root:
`display_name`, `top_species`, `scientific_name`, `confidence`, and
`probabilities`. The electronic-monitoring client reads these fields directly.
If the API is unavailable or its response is invalid, the optional classification
returns no result and emits one `[WARN]` line for that request.

## Batch prediction

Send repeated `files` multipart fields to `POST /predict/batch`:

```sh
curl -F 'files=@fish-a.jpg' -F 'files=@fish-b.jpg' http://localhost:8000/predict/batch
```

The response contains `count` and `results`. Each result includes a zero-based
`index`, the uploaded `filename`, and the same prediction fields as `/predict`.
Results keep upload order. An optional `hint_species` form field has the same
meaning as in single-image prediction (a hint used in demo mode).

- At most **32 files**, each at most **15 MiB (15 × 1024 × 1024 bytes)**.
- Exceeding either limit returns **413**. The per-image limit also applies to
  `/predict` and `/contribute`.
- An empty or invalid image returns **400**. A batch error identifies its index
  and filename; the API does not return partial successful results.
- Images are decoded one at a time. Camera EXIF orientation is applied before RGB
  conversion, for single predictions, batches, and contributions.
- Without a model file, responses explicitly use `mode: "demo"`; these simulated
  predictions must not be interpreted as measured model accuracy.

## Contributions, privacy and duplicates

`POST /contribute` applies the existing image-quality checks to the oriented RGB
image. It saves a fresh JPEG containing only pixels, without original EXIF, GPS,
camera/device tags or other source image metadata. The caller's explicit form
metadata (such as notes and species labels) is still saved.

The response keeps `saved`, `backend`, `sample_id` and `location`, and adds
`duplicate`. SHA-256 is calculated over the sanitized JPEG bytes in
`save_contribution`. Repeating an upload with the same stored bytes and destination
returns the first sample's identity with `duplicate: true`; it creates no second
image, sidecar or manifest row. The **first submission's labels and confirmation
status remain unchanged**, so re-uploading is not a label-edit operation.

The index is `LOCAL_STORAGE_DIR/STORAGE_PREFIX/manifest.jsonl`. New successful
local, S3 and GCS saves are all indexed there. Readers/writers sharing that local
index coordinate through a thread lock and file lock, and manifest updates are
atomic. Failed saves are not indexed; cleanup of partially uploaded cloud objects
is best effort. Keep the index on persistent storage, including for cloud-backed
deployments.

Deduplication is scoped to the configured destination and this shared index.
It is not a distributed/global cloud index or a visual-similarity check. Existing
rows without a content hash are not retroactively deduplicated. Removing an object
outside the application does not automatically remove or repair its index row.

## Contribution statistics

```sh
curl http://localhost:8000/contributions/stats
```

Example response shape:

```json
{"total":2,"confirmed":1,"needs_review":1,"by_species":{"Thunnus albacares":2}}
```

Statistics count valid rows in this instance's manifest for the current storage
destination, not all objects in a cloud bucket. Species uses the suggested label,
then the predicted scientific name, then `nova_especie`. Old rows without a
destination remain included for compatibility because their destination cannot
be reconstructed reliably. Malformed/truncated rows are skipped. Duplicate
uploads do not increase counts.

## Development verification

From the repository root with Python 3.11:

```sh
python -m pip install -r requirements-test.txt
ruff check .
python -m pytest -v
```

The CPU test dependencies do not require TensorFlow, trained weights, GPU access
or cloud credentials. Tests use the demo path or a small fake model, temporary
storage, and S3/GCS client doubles. The training notebook is excluded from Ruff;
Python services, scripts and tests use its standard E4/E7/E9/F checks. CI runs
syntax, lint, smoke and pytest checks on every push and pull request.
