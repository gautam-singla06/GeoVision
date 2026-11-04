# GeoVision

GeoVision is a small image-retrieval prototype for street-image geolocation. It embeds an uploaded image with pretrained ResNet-18, finds visually similar geotagged reference images, and returns the coordinates of the best match. It is not a GPS estimator or a production geolocation system.

## Current status

Implemented:

- Wikimedia Commons acquisition with per-image provenance and licence fields.
- Coordinate validation, 224×224 preprocessing, ResNet-18 embeddings, NumPy cosine-similarity index, CLI prediction, Streamlit UI, and held-out retrieval evaluation.
- Smoke tests for configuration, distance/similarity math, splitting, and (when PyTorch is installed) embedding shape.

Not yet completed in this checkout:

- No reference images, `data/raw/all_metadata.json`, or `data/index/` files are present, so the UI and prediction command cannot return a result yet.
- The pipeline could not be executed here because Windows has only the Microsoft Store `python` alias and no installed Python interpreter. Consequently, no dataset download, model-weight download, tests, index, or predictions were claimed as completed.
- The supplied images have not been inspected yet. Wikimedia geosearch returns geographically tagged Commons files, but they can include landmarks, indoor photos, artwork, or other non-street imagery; this is not a curated street-view dataset.

## Windows setup

1. Install Python 3.12 from [python.org](https://www.python.org/downloads/) and select **Add python.exe to PATH**. Close and reopen PowerShell, then verify:

   ```powershell
   python --version
   # Python 3.12.x
   ```

2. From the project directory, create and activate an environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` for that shell and activate again.

3. Install the CPU build of PyTorch, then the remaining dependencies:

   ```powershell
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```

## Run the small reproducible prototype

The default in `src/config.py` is **3 images per region** across eight regions (at most 24 thumbnails). It is deliberately small; change `NUM_IMAGES_PER_REGION` only after the baseline works.

```powershell
python src\test_pipeline.py
python src\download_data.py
python src\build_index.py
python src\evaluate.py
streamlit run src\app.py
```

For a command-line query:

```powershell
python src\predict.py path\to\photo.jpg
```

The first index build fetches the official pretrained ResNet-18 weights through torchvision (about 45 MB). No API token or payment is required for the Wikimedia download. The downloader saves each image's source page title, thumbnail URL, coordinates, and `LicenseShortName` in `data/raw/<region>/metadata.json`; `all_metadata.json` is the combined GPS metadata. Retain this provenance when sharing the data.

## Dataset and attribution

Reference images are queried from the [Wikimedia Commons API](https://commons.wikimedia.org/w/api.php) using geosearch around Paris, London, New York, Tokyo, Sydney, Cairo, Rio de Janeiro, and Mumbai. The API requires no credentials. Each accepted item must have both latitude and longitude; files without either coordinate are skipped. The exact licence is recorded per file because Commons licences vary (commonly Creative Commons, but not guaranteed to be a single licence). Use the saved metadata and the original Commons file page to meet the licence and attribution terms for every downloaded image.

## Limitations

- Similarity is not geographic accuracy. A high score only means that a query resembles a reference image.
- The tiny sample and eight city regions are suitable only for an end-to-end demonstration.
- The evaluation holds out images from the same Wikimedia collection, so it does not measure performance on independent real-world street photos.
- This retrieval baseline does not train or fine-tune a geolocation model.

## Project files

- `src/download_data.py` — token-free Commons downloader and provenance writer.
- `src/build_index.py` — ResNet-18 embedding extraction and index creation.
- `src/predict.py` — nearest-neighbour prediction.
- `src/evaluate.py` — same-source held-out evaluation.
- `src/app.py` — Streamlit interface.
- `src/test_pipeline.py` — smoke tests.
