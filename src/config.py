"""
config.py — Central configuration for GeoVision.

All paths and tuneable settings live here. Change NUM_IMAGES_PER_REGION
to download more or fewer images, or add new regions to REGIONS.
"""
from pathlib import Path
import os

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_DIR  = BASE_DIR / "data"
RAW_DIR   = DATA_DIR / "raw"
INDEX_DIR = DATA_DIR / "index"

# ---------------------------------------------------------------------------
# Download settings
# ---------------------------------------------------------------------------
# Number of images to attempt per city region.
# Keep the default intentionally small and reproducible: 3 × 8 regions means
# at most 24 thumbnails. Increase only after validating the baseline.
NUM_IMAGES_PER_REGION = 3

# Pixel size of downloaded thumbnail (square).
IMAGE_SIZE = 224

# Search radius around each region centre (metres).  Max: 10 000.
GEOSEARCH_RADIUS = 10_000

# Polite delay between Wikimedia API calls (seconds).
API_DELAY = 0.25

# ---------------------------------------------------------------------------
# Feature-extraction model
# ---------------------------------------------------------------------------
# resnet18 is the lightest torchvision model; no GPU needed.
MODEL_NAME = "resnet18"
EMBEDDING_DIM = 512  # penultimate layer output size for resnet18

# ---------------------------------------------------------------------------
# Geographic regions used to build the reference database.
# Spread across multiple continents so predictions are globally diverse.
# ---------------------------------------------------------------------------
REGIONS = [
    {
        "name": "New_Delhi",
        "lat": 28.6139,
        "lon": 77.2090
    }
]
