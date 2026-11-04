"""
download_data.py — Download geotagged images from Wikimedia Commons.

Usage:
    python src/download_data.py

No API token required.  Images are Creative Commons licensed.
A provenance log (metadata.json) is saved alongside each region's images
to satisfy CC-BY / CC-BY-SA attribution requirements.
"""

import sys
import os
import json
import time
import io
import requests
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Allow running as "python src/download_data.py" from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    RAW_DIR, REGIONS, NUM_IMAGES_PER_REGION,
    IMAGE_SIZE, GEOSEARCH_RADIUS, API_DELAY,
)

WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"

# Wikimedia requires a descriptive User-Agent for scripts that make API calls.
HEADERS = {
    "User-Agent": (
        "GeoVisionProject/0.1 "
        "(educational ML project; "
        "https://github.com/user/GeoVision)"
    )
}


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def geosearch(lat: float, lon: float, radius: int = GEOSEARCH_RADIUS,
              limit: int = 500) -> list[dict]:
    """
    Return up to `limit` geotagged files from Wikimedia Commons
    within `radius` metres of (lat, lon).
    """
    params = {
        "action":      "query",
        "list":        "geosearch",
        "gscoord":     f"{lat}|{lon}",
        "gsradius":    radius,
        "gslimit":     min(limit, 500),
        "gsnamespace": 6,       # Namespace 6 = File:
        "format":      "json",
    }
    resp = requests.get(WIKIMEDIA_API, params=params,
                        headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json().get("query", {}).get("geosearch", [])


def get_image_info(page_title: str,
                   thumb_width: int = IMAGE_SIZE) -> tuple[str | None, float | None, float | None, str]:
    """
    Return (thumbnail_url, lat, lon, license_str) for a Wikimedia file.
    Returns (None, None, None, '') if the page has no usable image or coords.
    """
    params = {
        "action":  "query",
        "titles":  page_title,
        "prop":    "imageinfo|coordinates",
        "iiprop":  "url|mime|extmetadata",
        "iiurlwidth": thumb_width,
        "format":  "json",
    }
    resp = requests.get(WIKIMEDIA_API, params=params,
                        headers=HEADERS, timeout=30)
    resp.raise_for_status()
    pages = resp.json().get("query", {}).get("pages", {})

    for page in pages.values():
        ii = (page.get("imageinfo") or [{}])[0]
        url  = ii.get("thumburl") or ii.get("url", "")
        mime = ii.get("mime", "")
        if not url or not mime.startswith("image/"):
            return None, None, None, ""

        # License from extmetadata
        meta   = ii.get("extmetadata", {})
        lic    = meta.get("LicenseShortName", {}).get("value", "Unknown")

        # Coordinates
        coords = page.get("coordinates", [])
        lat    = coords[0]["lat"] if coords else None
        lon    = coords[0]["lon"] if coords else None

        return url, lat, lon, lic

    return None, None, None, ""


# ---------------------------------------------------------------------------
# Per-region downloader
# ---------------------------------------------------------------------------

def download_region(region: dict, num_images: int = NUM_IMAGES_PER_REGION) -> list[dict]:
    """
    Download images for one region.
    Returns list of metadata dicts (one per successfully downloaded image).
    """
    out_dir = RAW_DIR / region["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.json"

    # Resume: load any previously saved metadata
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = []

    already_saved = {m["filename"] for m in metadata}

    # Fetch candidate list
    candidates = geosearch(region["lat"], region["lon"], limit=num_images * 3)
    downloaded = len(already_saved)

    for item in tqdm(candidates, desc=f"  {region['name']}", leave=False):
        if downloaded >= num_images:
            break

        title = item.get("title", "")
        if not title:
            continue

        # Build a filesystem-safe filename
        safe = (title.replace("File:", "")
                     .replace("/", "_")
                     .replace(" ", "_"))[:80]
        filename = safe + ".jpg"

        if filename in already_saved:
            downloaded += 1
            continue

        try:
            url, lat, lon, license_str = get_image_info(title)
            time.sleep(API_DELAY)

            # A record is useful to the index only when it has a complete
            # coordinate pair. Commons search candidates normally have both,
            # but retain this guard for incomplete file metadata.
            if not url or lat is None or lon is None:
                continue

            # Download thumbnail bytes
            img_resp = requests.get(url, headers=HEADERS, timeout=30)
            img_resp.raise_for_status()

            # Open, convert, resize, save
            img = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
            save_path = out_dir / filename
            img.save(save_path, "JPEG", quality=85)

            metadata.append({
                "filename": filename,
                "path":     str(save_path),
                "lat":      lat,
                "lon":      lon,
                "region":   region["name"],
                "source":   "Wikimedia Commons",
                "license":  license_str,
                "title":    title,
                "url":      url,
            })
            already_saved.add(filename)
            downloaded += 1
            time.sleep(API_DELAY)

        except Exception:
            # Skip any image that fails (network glitch, corrupt file, etc.)
            continue

    # Persist metadata for this region
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return metadata


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(" GeoVision — Downloading reference images")
    print(" Source : Wikimedia Commons (no API token required)")
    print(f" Target : {NUM_IMAGES_PER_REGION} images × {len(REGIONS)} regions")
    print(f" Output : {RAW_DIR}")
    print("=" * 60)

    all_metadata: list[dict] = []

    for region in REGIONS:
        print(f"\n[Region] {region['name']}")
        meta = download_region(region)
        all_metadata.extend(meta)
        print(f"  ✓  {len(meta)} images saved")

    # Combined metadata file
    combined_path = RAW_DIR / "all_metadata.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f" Done. Total images downloaded: {len(all_metadata)}")
    print(f" Combined metadata : {combined_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

