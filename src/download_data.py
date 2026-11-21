"""Opt-in Mapillary Graph API collector for public geotagged images.

Run: python src/download_data.py [--limit 20] [--bbox west,south,east,north]
No collection happens on import. Keep source attribution and Mapillary IDs.
"""
import argparse
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

import requests
from PIL import Image, ImageStat

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import API_DELAY, DATASET_LIMIT, MAPILLARY_DIR, MAX_DATASET_IMAGES, MAX_STORAGE_BYTES, REGION

API_URL = "https://graph.mapillary.com/images"
FIELDS = "id,geometry,thumb_1024_url,captured_at,sequence,compass_angle,is_pano"


def get_token():
    """Read token from environment or root .env without ever displaying it."""
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


def quality_ok(image: Image.Image) -> bool:
    """Reject tiny, blank, or near-uniform thumbnails; cannot semantically identify portraits."""
    if image.width < 160 or image.height < 100:
        return False
    gray = image.convert("L")
    return ImageStat.Stat(gray).stddev[0] >= 12


def dhash(image: Image.Image) -> int:
    pixels = image.convert("L").resize((9, 8))
    values = list(pixels.get_flattened_data() if hasattr(pixels, "get_flattened_data") else pixels.getdata())
    return sum((values[y * 9 + x] > values[y * 9 + x + 1]) << (y * 8 + x)
               for y in range(8) for x in range(8))


def fetch_candidates(token: str, bbox: list[float], limit: int) -> list[dict]:
    params = {"fields": FIELDS, "bbox": ",".join(map(str, bbox)), "limit": min(200, max(limit * 4, limit))}
    try:
        response = requests.get(
            API_URL,
            params=params,
            headers={"Authorization": f"OAuth {token}"},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        # Don't include exception text: request URLs can contain the access token.
        raise RuntimeError(f"Mapillary request failed ({getattr(exc.response, 'status_code', 'network error')}). Check API access and try again.") from None
    except ValueError:
        raise RuntimeError("Mapillary returned an unreadable response.") from None
    return data.get("data", [])


def download_dataset(limit: int = DATASET_LIMIT, bbox: list[float] | None = None) -> list[dict]:
    if not 1 <= limit <= MAX_DATASET_IMAGES:
        raise ValueError(f"limit must be between 1 and {MAX_DATASET_IMAGES}")
    token = get_token()
    bbox = bbox or REGION["bbox"]
    MAPILLARY_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = MAPILLARY_DIR / "metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else []
    seen_ids = {str(r["image_id"]) for r in metadata}
    seen_hashes = [int(r["dhash"], 16) for r in metadata if r.get("dhash")]
    candidates = fetch_candidates(token, bbox, limit)
    current_size = sum(p.stat().st_size for p in MAPILLARY_DIR.glob("*.jpg"))
    for item in candidates:
        if len(metadata) >= limit or current_size >= MAX_STORAGE_BYTES:
            break
        image_id = str(item.get("id", ""))
        url = item.get("thumb_1024_url")
        point = item.get("geometry", {}).get("coordinates", [])
        if not image_id or image_id in seen_ids or not url or len(point) < 2:
            continue
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            raw = response.content
            if current_size + len(raw) > MAX_STORAGE_BYTES:
                continue
            image = Image.open(io.BytesIO(raw)).convert("RGB")
            if not quality_ok(image):
                continue
            phash = dhash(image)
            if any((phash ^ old).bit_count() <= 4 for old in seen_hashes):
                continue
            filename = f"{image_id}.jpg"
            image.save(MAPILLARY_DIR / filename, "JPEG", quality=88, optimize=True)
            size = (MAPILLARY_DIR / filename).stat().st_size
            current_size += size
            record = {"image_id": image_id, "filename": filename, "path": str(MAPILLARY_DIR / filename),
                      "lat": float(point[1]), "lon": float(point[0]), "region": REGION["name"],
                      "source": "Mapillary", "attribution": "© Mapillary", "captured_at": item.get("captured_at"),
                      "sequence_id": item.get("sequence"), "compass_angle": item.get("compass_angle"),
                      "is_pano": item.get("is_pano"), "thumbnail_url": url, "dhash": f"{phash:016x}", "bytes": size}
            metadata.append(record)
            seen_ids.add(image_id)
            seen_hashes.append(phash)
            meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            time.sleep(API_DELAY)
        except (requests.RequestException, OSError, ValueError):
            continue
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=DATASET_LIMIT)
    parser.add_argument("--bbox", help="west,south,east,north")
    args = parser.parse_args()
    bbox = [float(v) for v in args.bbox.split(",")] if args.bbox else None
    records = download_dataset(args.limit, bbox)
    print(f"Mapillary images available: {len(records)} (metadata: {MAPILLARY_DIR / 'metadata.json'})")
    print("Review image quality and licensing terms before redistribution.")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
