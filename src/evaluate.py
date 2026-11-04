"""
evaluate.py — Held-out evaluation of the image-retrieval geolocation system.

Usage:
    python src/evaluate.py [--held_out_frac 0.2] [--top_k 1]

What this script does:
  1. Loads the full metadata from data/raw/all_metadata.json.
  2. Splits images into a 80% reference set and 20% held-out query set
     (stratified by region so every region is represented in both).
  3. Builds an in-memory index from the reference set.
  4. For each held-out image, predicts coordinates using cosine similarity.
  5. Computes and reports:
       - Mean / Median geographic error (km)      — when GPS is available
       - Recall@K — fraction of queries where top-K includes the correct region

Important honesty notice printed at runtime:
  Geographic error is only meaningful here because the held-out images come
  from the same source (Wikimedia Commons) as the reference database.
  Real-world accuracy on arbitrary street photos will be lower.
"""

import sys
import json
import math
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RAW_DIR, MODEL_NAME

# ---- lazy torch ----
def _load_torch():
    import torch
    import torchvision.models as models
    import torchvision.transforms as T
    weights = models.ResNet18_Weights.DEFAULT
    model   = models.resnet18(weights=weights)
    model   = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return model, transform, torch


def embed(model, transform, torch, path: Path) -> np.ndarray | None:
    try:
        img    = Image.open(path).convert("RGB")
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            vec = model(tensor).squeeze().numpy()
        n = np.linalg.norm(vec)
        return (vec / n).astype(np.float32) if n > 1e-8 else None
    except Exception:
        return None


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R  = 6371.0
    p1 = math.radians(lat1);  p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a  = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def stratified_split(meta: list[dict], held_frac: float):
    """Split by region so every region appears in both sets."""
    from collections import defaultdict
    import random
    random.seed(42)

    by_region: dict[str, list[dict]] = defaultdict(list)
    for m in meta:
        by_region[m["region"]].append(m)

    ref, query = [], []
    for region_items in by_region.values():
        random.shuffle(region_items)
        n_held = max(1, round(len(region_items) * held_frac))
        query.extend(region_items[:n_held])
        ref.extend(region_items[n_held:])
    return ref, query


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--held_out_frac", type=float, default=0.20,
                        help="Fraction of images to hold out as queries (default 0.20)")
    parser.add_argument("--top_k", type=int, default=1,
                        help="Predict using the top-K nearest neighbour (default 1)")
    args = parser.parse_args()

    meta_path = RAW_DIR / "all_metadata.json"
    if not meta_path.exists():
        print("[ERROR] data/raw/all_metadata.json not found. Run download_data.py first.")
        sys.exit(1)

    with open(meta_path, encoding="utf-8") as f:
        all_meta = json.load(f)

    # Keep only records whose image file exists
    all_meta = [m for m in all_meta if Path(m["path"]).exists()]
    print(f"Total usable images: {len(all_meta)}")

    ref_meta, query_meta = stratified_split(all_meta, args.held_out_frac)
    print(f"Reference set : {len(ref_meta)} images")
    print(f"Query set     : {len(query_meta)} images  (held-out)\n")

    if len(ref_meta) < 5:
        print("[ERROR] Not enough reference images. Download more images first.")
        sys.exit(1)

    model, transform, torch = _load_torch()
    print(f"Model: {MODEL_NAME} (CPU)\n")

    # Build reference embeddings
    ref_embs  = []
    ref_valid = []
    for m in tqdm(ref_meta, desc="Embedding reference set"):
        v = embed(model, transform, torch, Path(m["path"]))
        if v is not None:
            ref_embs.append(v)
            ref_valid.append(m)

    ref_matrix = np.stack(ref_embs, axis=0)  # (N_ref, D)

    # Evaluate on query set
    geo_errors_km: list[float] = []
    region_hits_k: list[bool]  = []

    for m in tqdm(query_meta, desc="Evaluating query set"):
        v = embed(model, transform, torch, Path(m["path"]))
        if v is None:
            continue

        sims    = ref_matrix @ v
        top_idx = np.argsort(sims)[::-1][: args.top_k]

        # Geographic error from top-1
        best = ref_valid[top_idx[0]]
        err  = haversine_km(m["lat"], m["lon"], best["lat"], best["lon"])
        geo_errors_km.append(err)

        # Recall@K: is the true region among the top-K regions?
        top_regions = {ref_valid[i]["region"] for i in top_idx}
        region_hits_k.append(m["region"] in top_regions)

    # ---------- Report ----------
    print("\n" + "=" * 60)
    print(" GeoVision — Evaluation Results")
    print("=" * 60)
    print(
        "\n⚠️  Honesty note:\n"
        "  These numbers measure retrieval within the Wikimedia Commons\n"
        "  reference DB, not real-world geolocation accuracy.\n"
        "  Held-out images come from the same distribution as the reference\n"
        "  set — expect higher accuracy than on truly novel street photos.\n"
    )

    if geo_errors_km:
        errs = np.array(geo_errors_km)
        print(f"Geographic error  (km)  — {len(errs)} queries")
        print(f"  Mean   : {errs.mean():.1f} km")
        print(f"  Median : {np.median(errs):.1f} km")
        print(f"  ≤25 km : {(errs <= 25).mean()*100:.1f}%")
        print(f"  ≤750 km: {(errs <= 750).mean()*100:.1f}%")
        print(f"  ≤2500 km:{(errs <= 2500).mean()*100:.1f}%")
        print()

    if region_hits_k:
        recall = sum(region_hits_k) / len(region_hits_k) * 100
        print(f"Region Recall@{args.top_k}: {recall:.1f}%  "
              f"({sum(region_hits_k)}/{len(region_hits_k)} correct)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

