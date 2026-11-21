"""Dataset audit helpers shared by the research dashboard and experiment scripts."""
import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageStat

from config import DATA_DIR, MAPILLARY_DIR


def _dhash(path: Path):
    try:
        with Image.open(path) as image:
            pixels = image.convert("L").resize((9, 8))
            values = list(pixels.getdata())
        value = sum((values[y * 9 + x] > values[y * 9 + x + 1]) << (y * 8 + x) for y in range(8) for x in range(8))
        return f"{value:016x}"
    except (OSError, ValueError):
        return None


def mapillary_records():
    """Return only locally present, GPS-valid Mapillary records eligible for research."""
    path = MAPILLARY_DIR / "metadata.json"
    if not path.exists():
        return []
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    valid = []
    for row in records if isinstance(records, list) else []:
        try:
            image, lat, lon = Path(row["path"]), float(row["lat"]), float(row["lon"])
            if row.get("source") == "Mapillary" and image.is_file() and -90 <= lat <= 90 and -180 <= lon <= 180:
                valid.append(row)
        except (KeyError, TypeError, ValueError):
            continue
    return valid


def legacy_records():
    """Read existing non-Mapillary rows for audit only; never train on them."""
    path = DATA_DIR / "index" / "metadata.json"
    try:
        rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    except (OSError, ValueError):
        rows = []
    return [row for row in rows if isinstance(row, dict) and row.get("source") != "Mapillary"]


def audit_records(records):
    """Compute real local file and metadata statistics without adding records."""
    rows, hashes = [], []
    for row in records:
        item, path = dict(row), Path(row.get("path", ""))
        item.update(file_exists=path.is_file(), width=None, height=None, quality_stddev=None)
        if path.is_file():
            try:
                with Image.open(path) as image:
                    image.load()
                    item["width"], item["height"] = image.size
                    item["quality_stddev"] = round(ImageStat.Stat(image.convert("L")).stddev[0], 3)
                item["dhash"] = _dhash(path)
                hashes.append(item["dhash"])
            except (OSError, ValueError):
                item["file_exists"] = False
        rows.append(item)
    frame = pd.DataFrame(rows)
    summary = {"records": len(rows), "readable_images": int(frame.get("file_exists", pd.Series(dtype=bool)).sum()),
               "regions": int(frame["region"].nunique()) if "region" in frame else 0,
               "gps_complete": int(frame[["lat", "lon"]].notna().all(axis=1).sum()) if {"lat", "lon"} <= set(frame) else 0,
               "exact_dhash_duplicates": len(hashes) - len(set(hashes)),
               "missing_values": int(frame.isna().sum().sum()) if not frame.empty else 0}
    return frame, summary


def spatial_group(row):
    """Keep one capture sequence (or approximately 110 m grid cell) in one split."""
    if row.get("sequence_id"):
        return f"sequence:{row['sequence_id']}"
    return f"cell:{round(float(row['lat']), 3)}:{round(float(row['lon']), 3)}"
