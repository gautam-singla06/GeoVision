"""Opt-in Mapillary Graph API collector for public geotagged images.

Run: python src/download_data.py [--limit 20] [--lat LAT --lng LNG --radius METRES]
No collection happens on import. Keep source attribution and Mapillary IDs.
"""
import argparse
import io
import json
import os
import sys
import time
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import requests
from PIL import Image, ImageStat

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import API_DELAY, DATASET_LIMIT, MAPILLARY_DIR, MAX_DATASET_IMAGES, MAX_STORAGE_BYTES, REGION

API_URL = "https://graph.mapillary.com/images"
FIELDS = "id,geometry,thumb_1024_url,captured_at,sequence,compass_angle,is_pano"
REQUEST_TIMEOUT_SECONDS = 30


def get_token():
    """Read the credential locally without ever displaying it."""
    token = os.getenv("MAPILLARY_CLIENT_TOKEN")
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if not token and env_file.exists():
        try:
            from dotenv import dotenv_values
            token = (dotenv_values(env_file).get("MAPILLARY_CLIENT_TOKEN") or "").strip()
        except ImportError:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("MAPILLARY_CLIENT_TOKEN="):
                    token = line.split("=", 1)[1].strip().strip("\"'")
                    break
    if not token or token == "your_mapillary_client_token_here":
        raise RuntimeError("Mapillary credentials are missing. Set MAPILLARY_CLIENT_TOKEN in the root .env file.")
    return token


def region_search(region: dict) -> tuple[float, float, int]:
    """Derive search coordinates from REGION, falling back to its bounds."""
    west, south, east, north = region["bbox"]
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("REGION bbox must be west,south,east,north within valid geographic bounds")
    lat, lng = region.get("lat", (south + north) / 2), region.get("lng", (west + east) / 2)
    if not (-90 <= lat <= 90 and -180 <= lng <= 180):
        raise ValueError("REGION lat/lng must be valid geographic coordinates")

    def distance_metres(lat_a, lng_a, lat_b, lng_b):
        d_lat, d_lng = radians(lat_b - lat_a), radians(lng_b - lng_a)
        a = sin(d_lat / 2) ** 2 + cos(radians(lat_a)) * cos(radians(lat_b)) * sin(d_lng / 2) ** 2
        return 6_371_000 * 2 * asin(sqrt(a))

    radius = region.get("radius")
    if radius is None:
        radius = max(distance_metres(lat, lng, corner_lat, corner_lng)
                     for corner_lat in (south, north) for corner_lng in (west, east))
    if not isinstance(radius, (int, float)) or radius <= 0:
        raise ValueError("REGION radius must be a positive number of metres")
    return lat, lng, max(1, int(radius + 0.999))


def quality_ok(image: Image.Image) -> bool:
    if image.width < 160 or image.height < 100:
        return False
    return ImageStat.Stat(image.convert("L")).stddev[0] >= 12


def dhash(image: Image.Image) -> int:
    pixels = image.convert("L").resize((9, 8))
    values = list(pixels.get_flattened_data() if hasattr(pixels, "get_flattened_data") else pixels.getdata())
    return sum((values[y * 9 + x] > values[y * 9 + x + 1]) << (y * 8 + x)
               for y in range(8) for x in range(8))


def fetch_candidates(token: str, lat: float, lng: float, radius: int, limit: int) -> list[dict]:
    """Fetch candidates with the working lat/lng/radius Mapillary query."""
    if not (-90 <= lat <= 90 and -180 <= lng <= 180 and radius > 0):
        raise ValueError("search coordinates or radius are invalid")
    params = {"fields": FIELDS, "lat": lat, "lng": lng, "radius": radius,
              "limit": min(200, max(limit * 4, limit))}
    try:
        response = requests.get(API_URL, params=params, headers={"Authorization": f"OAuth {token}"},
                                timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        status = getattr(getattr(exc, "response", None), "status_code", "network error")
        raise RuntimeError(f"Mapillary request failed ({status}). Check API access and try again.") from None
    except ValueError:
        raise RuntimeError("Mapillary returned an unreadable response.") from None
    results = payload.get("data", []) if isinstance(payload, dict) else []
    return results if isinstance(results, list) else []


def download_dataset(limit: int = DATASET_LIMIT, lat: float | None = None,
                     lng: float | None = None, radius: int | None = None) -> list[dict]:
    if not 1 <= limit <= MAX_DATASET_IMAGES:
        raise ValueError(f"limit must be between 1 and {MAX_DATASET_IMAGES}")
    default_lat, default_lng, default_radius = region_search(REGION)
    lat = default_lat if lat is None else lat
    lng = default_lng if lng is None else lng
    radius = default_radius if radius is None else radius
    token = get_token()
    MAPILLARY_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = MAPILLARY_DIR / "metadata.json"
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else []
    except (OSError, ValueError):
        raise RuntimeError("Existing Mapillary metadata is unreadable; move it aside before collecting.") from None
    if not isinstance(metadata, list):
        raise RuntimeError("Existing Mapillary metadata must be a JSON list.")
    seen_ids = {str(r["image_id"]) for r in metadata if isinstance(r, dict) and r.get("image_id")}
    seen_hashes = []
    for record in metadata:
        try:
            if isinstance(record, dict) and record.get("dhash"):
                seen_hashes.append(int(record["dhash"], 16))
        except (TypeError, ValueError):
            continue
    candidates = fetch_candidates(token, lat, lng, radius, limit)
    current_size = sum(p.stat().st_size for p in MAPILLARY_DIR.glob("*.jpg"))
    for item in candidates:
        if len(metadata) >= limit or current_size >= MAX_STORAGE_BYTES:
            break
        image_id = str(item.get("id", ""))
        url = item.get("thumb_1024_url")
        point = item.get("geometry", {}).get("coordinates", [])
        if not image_id or image_id in seen_ids or not isinstance(url, str) or not url.startswith("https://") or len(point) < 2:
            continue
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            raw = response.content
            if current_size + len(raw) > MAX_STORAGE_BYTES:
                continue
            with Image.open(io.BytesIO(raw)) as source:
                source.load()
                image = source.convert("RGB")
            if not quality_ok(image):
                continue
            image_hash = dhash(image)
            if any((image_hash ^ old).bit_count() <= 4 for old in seen_hashes):
                continue
            destination = MAPILLARY_DIR / f"{image_id}.jpg"
            temporary = destination.with_suffix(".jpg.part")
            image.save(temporary, "JPEG", quality=88, optimize=True)
            temporary.replace(destination)
            size = destination.stat().st_size
            current_size += size
            metadata.append({"image_id": image_id, "filename": destination.name, "path": str(destination),
                             "lat": float(point[1]), "lon": float(point[0]), "region": REGION["name"],
                             "source": "Mapillary", "attribution": "© Mapillary", "captured_at": item.get("captured_at"),
                             "sequence_id": item.get("sequence"), "compass_angle": item.get("compass_angle"),
                             "is_pano": item.get("is_pano"), "thumbnail_url": url,
                             "dhash": f"{image_hash:016x}", "bytes": size})
            seen_ids.add(image_id)
            seen_hashes.append(image_hash)
            meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            time.sleep(API_DELAY)
        except (requests.RequestException, OSError, ValueError):
            continue
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=DATASET_LIMIT)
    parser.add_argument("--lat", type=float, help="search latitude (defaults to REGION configuration)")
    parser.add_argument("--lng", type=float, help="search longitude (defaults to REGION configuration)")
    parser.add_argument("--radius", type=int, help="search radius in metres (defaults to REGION configuration)")
    args = parser.parse_args()
    if (args.lat is None) != (args.lng is None):
        parser.error("--lat and --lng must be provided together")
    records = download_dataset(args.limit, args.lat, args.lng, args.radius)
    print(f"Mapillary images available: {len(records)} (metadata: {MAPILLARY_DIR / 'metadata.json'})")
    print("Review image quality and licensing terms before redistribution.")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
