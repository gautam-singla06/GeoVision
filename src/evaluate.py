"""Held-out retrieval evaluation using non-overlapping reference/query images."""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MAPILLARY_DIR, MODEL_NAME
from build_index import extract_embedding, load_model
from predict import haversine_km


def stratified_split(meta, held_frac=0.2, seed=42):
    if not 0 < held_frac < 1:
        raise ValueError("held_frac must be between 0 and 1")
    groups = {}
    for item in meta:
        groups.setdefault(item.get("region", ""), []).append(item)
    rng = random.Random(seed)
    reference, query = [], []
    for group in groups.values():
        rng.shuffle(group)
        if len(group) < 2:
            reference.extend(group)
            continue
        count = min(len(group) - 1, max(1, round(len(group) * held_frac)))
        query.extend(group[:count])
        reference.extend(group[count:])
    return reference, query


def evaluate_records(records, held_frac=0.2, top_k=5, seed=42):
    ref_records, query_records = stratified_split(records, held_frac, seed)
    if not query_records:
        raise ValueError("Evaluation needs at least two valid images in one region.")
    model, transform, torch = load_model()
    ref_vectors, valid_refs = [], []
    for row in tqdm(ref_records, desc="Embedding reference split"):
        vec = extract_embedding(model, transform, torch, Path(row["path"]))
        if vec is not None:
            ref_vectors.append(vec)
            valid_refs.append(row)
    if not valid_refs:
        raise ValueError("No usable reference embeddings.")
    matrix = np.stack(ref_vectors)
    errors, region1, region5 = [], [], []
    evaluated = 0
    for row in tqdm(query_records, desc="Evaluating held-out split"):
        vec = extract_embedding(model, transform, torch, Path(row["path"]))
        if vec is None:
            continue
        sims = matrix @ vec
        order = np.argsort(sims)[::-1][:max(5, top_k)]
        matches = [valid_refs[int(i)] for i in order]
        errors.append(haversine_km(float(row["lat"]), float(row["lon"]),
                                   float(matches[0]["lat"]), float(matches[0]["lon"])))
        region1.append(matches[0].get("region") == row.get("region"))
        region5.append(any(m.get("region") == row.get("region") for m in matches[:5]))
        evaluated += 1
    if not errors:
        raise ValueError("No held-out images could be embedded.")
    return {"model": MODEL_NAME, "reference_count": len(valid_refs), "query_count": evaluated,
            "median_error_km": float(np.median(errors)), "mean_error_km": float(np.mean(errors)),
            "top1_region_accuracy": float(np.mean(region1)), "top5_region_accuracy": float(np.mean(region5)),
            "definition": "Top-k accuracy means the query's region label appears among k nearest references; it is not exact-image retrieval accuracy.",
            "limitation": "Held-out images share the source and region collection; results do not establish real-world accuracy."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--held-out-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    path = MAPILLARY_DIR / "metadata.json"
    if not path.exists():
        raise FileNotFoundError("Mapillary dataset not found. Run python src/download_data.py first.")
    records = [r for r in json.loads(path.read_text(encoding="utf-8")) if Path(r["path"]).is_file()]
    result = evaluate_records(records, args.held_out_frac, seed=args.seed)
    output = Path(__file__).resolve().parent.parent / "data" / "index" / "evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
