"""Shared paths and conservative defaults for the GeoVision prototype."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
MAPILLARY_DIR = RAW_DIR / "mapillary"

# Approximate downtown Pittsburgh bounds (west, south, east, north).
REGION = {"name": "Downtown Pittsburgh", "bbox": [-80.025, 40.425, -79.975, 40.455]}
DATASET_LIMIT = 20  # validation cap; configurable up to 100 via CLI
MAX_DATASET_IMAGES = 100
MAX_STORAGE_BYTES = 500 * 1024 * 1024
IMAGE_SIZE = 224
MODEL_NAME = "ResNet-18 (ImageNet pretrained)"
EMBEDDING_DIM = 512
API_DELAY = 0.2

# Backwards-compatible name for existing callers/tests.
REGIONS = [{"name": "Downtown_Pittsburgh", "lat": 40.4406, "lon": -79.9959}]
NUM_IMAGES_PER_REGION = DATASET_LIMIT
GEOSEARCH_RADIUS = 2500
