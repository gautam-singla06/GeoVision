"""Nearest-neighbour image retrieval; similarity is not location confidence."""
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import INDEX_DIR


def haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(min(1.0, math.sqrt(a)))


_cache = {}


def load_index():
    emb_path, meta_path = INDEX_DIR / "embeddings.npy", INDEX_DIR / "metadata.json"
    if not emb_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Mapillary index not found. Run python src/build_index.py first.")
    stamp = (emb_path.stat().st_mtime_ns, meta_path.stat().st_mtime_ns)
    if _cache.get("stamp") != stamp:
        _cache["emb"] = np.load(emb_path, mmap_mode="r")
        _cache["meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
        if _cache["emb"].shape[0] != len(_cache["meta"]):
            raise ValueError("Index metadata and embedding row counts do not match; rebuild the index.")
        if any(row.get("source") != "Mapillary" for row in _cache["meta"]):
            raise ValueError("This index is not a Mapillary index. Run python src/build_index.py to rebuild it.")
        _cache["stamp"] = stamp
    return _cache["emb"], _cache["meta"]


def load_model():
    if "model" not in _cache:
        from build_index import load_model as load
        _cache["model"], _cache["transform"], _cache["torch"] = load()
    return _cache["model"], _cache["transform"], _cache["torch"]


def predict(image_path, top_k=5, ground_truth=None):
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Query image not found: {path}")
    emb, metadata = load_index()
    model, transform, torch = load_model()
    with Image.open(path) as im:
        tensor = transform(im.convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        vec = model(tensor).squeeze(0).cpu().numpy()
    norm = np.linalg.norm(vec)
    if norm <= 1e-8:
        raise ValueError("Query image is blank or could not be embedded.")
    sims = np.asarray(emb @ (vec / norm), dtype=np.float32)
    inds = np.argsort(sims)[::-1][:max(1, min(int(top_k), len(metadata)))]
    results = []
    for rank, idx in enumerate(inds, 1):
        record = metadata[int(idx)]
        match = {"rank": rank, "similarity": float(sims[idx]), "lat": float(record["lat"]),
                 "lon": float(record["lon"]), "region": record.get("region", ""),
                 "ref_path": record["path"], "image_id": record.get("image_id", ""),
                 "attribution": record.get("attribution", "© Mapillary"),
                 "thumbnail_url": record.get("thumbnail_url", "")}
        if ground_truth is not None:
            match["distance_km"] = haversine_km(*ground_truth, match["lat"], match["lon"])
        results.append(match)
    return results


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python src/predict.py <image_path>")
    print(json.dumps(predict(sys.argv[1]), indent=2))
