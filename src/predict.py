"""
predict.py — Query the index with a new image and return the nearest match.

Usage (command-line):
    python src/predict.py path/to/your_photo.jpg

Returns:
    Predicted lat/lon, similarity score, and path to the matched reference image.
"""

import sys
import json
import math
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import INDEX_DIR, MODEL_NAME


# ---------------------------------------------------------------------------
# Haversine distance (km) between two lat/lon pairs
# ---------------------------------------------------------------------------
def haversine_km(lat1: float, lon1: float,
                 lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlam  = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2)**2
    return 2 * R * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Index loader (cached across calls in the same process)
# ---------------------------------------------------------------------------
_cache: dict = {}

def load_index() -> tuple[np.ndarray, list[dict]]:
    if "emb" not in _cache:
        emb_path  = INDEX_DIR / "embeddings.npy"
        meta_path = INDEX_DIR / "metadata.json"
        if not emb_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                "Index not found. Run  python src/build_index.py  first."
            )
        _cache["emb"]  = np.load(str(emb_path))
        with open(meta_path, encoding="utf-8") as f:
            _cache["meta"] = json.load(f)
    return _cache["emb"], _cache["meta"]


# ---------------------------------------------------------------------------
# Model loader (cached)
# ---------------------------------------------------------------------------
def load_model():
    if "model" not in _cache:
        import torch
        import torchvision.models as models
        import torchvision.transforms as T

        weights = models.ResNet18_Weights.DEFAULT
        m = models.resnet18(weights=weights)
        m = torch.nn.Sequential(*list(m.children())[:-1])
        m.eval()

        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        _cache["model"]     = m
        _cache["transform"] = transform
        _cache["torch"]     = torch

    return _cache["model"], _cache["transform"], _cache["torch"]


# ---------------------------------------------------------------------------
# Core prediction function
# ---------------------------------------------------------------------------
def predict(image_path: str | Path, top_k: int = 5) -> list[dict]:
    """
    Embed `image_path` and return the top-k nearest reference images.

    Each result dict contains:
        rank            int          1 = closest
        similarity      float        cosine similarity (0 – 1, higher = better match)
        lat, lon        float        GPS of matched reference image
        region          str          city/region label
        ref_path        str          path to matched reference image file
        title           str          Wikimedia Commons file title

    Important note on accuracy:
        Similarity score reflects visual similarity, NOT geographic accuracy.
        Geographic error is only meaningful when the true location is known.
    """
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Query image not found: {img_path}")

    model, transform, torch = load_model()
    emb_matrix, meta        = load_index()

    # Embed query
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        feat = model(tensor).squeeze().numpy()
    norm = np.linalg.norm(feat)
    if norm < 1e-8:
        raise ValueError("Query image produced a zero embedding (possibly blank or corrupt).")
    query_vec = (feat / norm).astype(np.float32)

    # Cosine similarity (emb_matrix rows are already L2-normalised)
    similarities = emb_matrix @ query_vec      # (N,)
    top_indices  = np.argsort(similarities)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        record = meta[idx]
        results.append({
            "rank":       rank,
            "similarity": float(similarities[idx]),
            "lat":        record["lat"],
            "lon":        record["lon"],
            "region":     record["region"],
            "ref_path":   record["path"],
            "title":      record.get("title", ""),
            "license":    record.get("license", ""),
        })
    return results


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <image_path>")
        sys.exit(1)

    query_path = sys.argv[1]
    print(f"\nQuery image : {query_path}")
    print("Loading index and model …\n")

    results = predict(query_path, top_k=5)

    print(f"{'Rank':<6} {'Similarity':>10}  {'Region':<20}  {'Lat':>10}  {'Lon':>10}")
    print("-" * 65)
    for r in results:
        print(f"{r['rank']:<6} {r['similarity']:>10.4f}  "
              f"{r['region']:<20}  {r['lat']:>10.4f}  {r['lon']:>10.4f}")

    best = results[0]
    print(f"\n▶  Best prediction: {best['region']}")
    print(f"   Coordinates    : {best['lat']:.5f}, {best['lon']:.5f}")
    print(f"   Similarity     : {best['similarity']:.4f}")
    print(f"\nNote: similarity ≠ geographic accuracy.")
    print(f"      A high score means the query looks visually like the matched image,")
    print(f"      not necessarily that it was taken at those exact coordinates.")


if __name__ == "__main__":
    main()

