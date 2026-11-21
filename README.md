# GeoVision

GeoVision is a lightweight, CPU-only image retrieval prototype for estimating where an outdoor street image may have been taken. It embeds images with pretrained ResNet-18, ranks GPS-labelled Mapillary reference thumbnails by cosine similarity, and reports the coordinate of the nearest reference. It is a retrieval baseline, not a trained geolocation model.

## Problem and architecture

```text
Mapillary Graph API → raw thumbnails + GPS/provenance JSON → processed 224×224 images
                                                        ↓
                         ResNet-18 embeddings → NumPy cosine index
                                                        ↓
                         Streamlit upload → top-5 matches + map
```

The default collection region is downtown Pittsburgh, Pennsylvania. Collection is an explicit command, capped at 20 images by default; the CLI allows a maximum of 100 and the downloader enforces a 500 MB storage ceiling. The pipeline has separate `data/raw/mapillary`, `data/processed/mapillary`, and `data/index` outputs. Downloads resume using image IDs in metadata. The existing unrelated files under `data/raw/` are not used by this Mapillary pipeline.

## Technology

Python 3.12, PyTorch CPU, torchvision ResNet-18 ImageNet weights, Pillow, NumPy, Requests, python-dotenv, and Streamlit. Embeddings are L2-normalized and searched with a matrix-vector cosine product, which is suitable for this small index and avoids an additional vector database.

## Data source, permissions, and quality

Images and coordinates come from the authenticated [Mapillary Graph API](https://www.mapillary.com/developer/api-documentation). The collector stores image ID, GPS, capture time when available, sequence ID, thumbnail URL, and Mapillary attribution. Consult the [Mapillary terms](https://www.mapillary.com/terms) and API documentation for current access, attribution, and reuse requirements. API access does not grant rights beyond those terms; retain attribution and review applicable image rights before redistribution.

The API query returns public geotagged imagery, not a curated guarantee of street scenes. The collector rejects very small, nearly uniform, exact/near-duplicate images, but cannot reliably detect portraits, indoor scenes, unrelated objects, or every unusable image. Inspect the small validation set before treating it as a reference dataset. The saved thumbnail URLs are API-provided URLs and are fetched only when the user runs the collection command.

## Setup

Install Python 3.12, then from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `MAPILLARY_CLIENT_TOKEN` to your Mapillary client token. `.env` is ignored by Git. The token is read locally and is not printed by the downloader. Missing credentials produce a clear error. Do not commit credentials or the downloaded dataset.

## Collect data

Collection is opt-in and starts with at most 20 images:

```powershell
python src\download_data.py --limit 20
```

After reviewing the validation imagery, a collection can be expanded up to 100 images:

```powershell
python src\download_data.py --limit 100
```

To specify another bounding box, pass `--bbox west,south,east,north`. The default bounds cover downtown Pittsburgh. Resume by repeating the same command; previously saved IDs are skipped. Collection may return fewer images if the API provides too few candidates or filters reject them. No data is collected just by starting the app.

## Build index and run

The first index build may download the official torchvision ResNet-18 weights (about 45 MB); no image dataset is downloaded by this step.

```powershell
python src\build_index.py
streamlit run src\app.py
```

The UI shows the best-match coordinates on a map and up to five matches with similarity values. A similarity score measures visual embedding similarity only; it is not a probability or accuracy estimate. The predicted coordinate is copied from the closest reference GPS point. When ground truth is available programmatically, `predict(..., ground_truth=(lat, lon))` adds its great-circle distance.

## Evaluation

```powershell
python src\evaluate.py --held-out-frac 0.2
```

Evaluation splits images before indexing queries, so a query image is never in the reference index. It reports mean and median top-1 geographic error and top-1/top-5 region-label retrieval accuracy, and saves the result to `data/index/evaluation.json`. Top-k accuracy means the query's region label appears among the nearest references; it is not exact-image accuracy. A small one-region collection has limited value for region classification, and even held-out results from the same collection do not establish performance on novel street photos. The command reports an error rather than inventing metrics when there are insufficient held-out examples.

## Tests

Run the local unit suite with:

```powershell
python src\test_pipeline.py
```

Tests are designed to run without downloading Mapillary images or pretrained weights.

## Limitations and next steps

- ResNet-18 image similarity is a weak geographic signal and is not fine-tuned for geolocation.
- Downtown Pittsburgh imagery can be sparse; map coverage and visual quality depend on Mapillary availability.
- Automatic quality checks cannot fully filter scene categories; manual review is needed.
- Similarity is not calibrated confidence. The best match can be geographically wrong.
- Improve the baseline with reviewed street-only data, spatially separated evaluation, richer geospatial retrieval, and a calibrated confidence model.
