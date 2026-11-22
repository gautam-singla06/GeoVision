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


def sequence_aware_split(meta, held_frac=0.2, seed=42):
    """Split complete Mapillary sequences by region, never individual images.

    Rows without a sequence ID are excluded because assigning them would make
    independence impossible to establish.  The returned report makes this
    explicit to callers and saved evaluation artifacts.
    """
    if not 0 < held_frac < 1:
        raise ValueError("held_frac must be between 0 and 1")
    regions, missing_sequence = {}, []
    for item in meta:
        sequence_id = item.get("sequence_id")
        if not sequence_id:
            missing_sequence.append(item)
            continue
        regions.setdefault(item.get("region", ""), {}).setdefault(str(sequence_id), []).append(item)
    rng = random.Random(seed)
    reference, query = [], []
    reference_sequences, query_sequences = set(), set()
    for sequences in regions.values():
        ids = list(sequences)
        rng.shuffle(ids)
        if len(ids) < 2:
            reference.extend(sequences[ids[0]])
            reference_sequences.add(ids[0])
            continue
        count = min(len(ids) - 1, max(1, round(len(ids) * held_frac)))
        for sequence_id in ids[:count]:
            query.extend(sequences[sequence_id])
            query_sequences.add(sequence_id)
        for sequence_id in ids[count:]:
            reference.extend(sequences[sequence_id])
            reference_sequences.add(sequence_id)
    report = {"missing_sequence_id_records": len(missing_sequence),
              "eligible_sequence_count": len(reference_sequences | query_sequences),
              "reference_sequence_count": len(reference_sequences),
              "query_sequence_count": len(query_sequences)}
    return reference, query, report


def stratified_split(meta, held_frac=0.2, seed=42):
    """Compatibility wrapper for callers that only need the two split lists."""
    reference, query, _ = sequence_aware_split(meta, held_frac, seed)
    return reference, query


def evaluate_records(records, held_frac=0.2, top_k=5, seed=42):
    ref_records, query_records, split_report = sequence_aware_split(records, held_frac, seed)
    if not query_records:
        raise ValueError("Evaluation needs at least two independent sequences in a region; records without sequence_id are excluded.")
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
    errors, region1, region5, predictions = [], [], [], []
    evaluated = 0
    for row in tqdm(query_records, desc="Evaluating held-out split"):
        vec = extract_embedding(model, transform, torch, Path(row["path"]))
        if vec is None:
            continue
        sims = matrix @ vec
        order = np.argsort(sims)[::-1][:max(5, top_k)]
        matches = [valid_refs[int(i)] for i in order]
        error_km = haversine_km(float(row["lat"]), float(row["lon"]),
                                float(matches[0]["lat"]), float(matches[0]["lon"]))
        errors.append(error_km)
        region1.append(matches[0].get("region") == row.get("region"))
        region5.append(any(m.get("region") == row.get("region") for m in matches[:5]))
        predictions.append({"image_id": row.get("image_id", Path(row["path"]).stem),
                            "sequence_id": row.get("sequence_id"), "region": row.get("region"),
                            "actual_lat": float(row["lat"]), "actual_lon": float(row["lon"]),
                            "predicted_lat": float(matches[0]["lat"]), "predicted_lon": float(matches[0]["lon"]),
                            "error_km": error_km, "top1_region_correct": bool(region1[-1]),
                            "top5_region_correct": bool(region5[-1])})
        evaluated += 1
    if not errors:
        raise ValueError("No held-out images could be embedded.")
    return {"model": MODEL_NAME, "reference_count": len(valid_refs), "query_count": evaluated,
            "reference_sequence_count": split_report["reference_sequence_count"],
            "query_sequence_count": split_report["query_sequence_count"],
            "unique_sequence_count": split_report["eligible_sequence_count"],
            "missing_sequence_id_records": split_report["missing_sequence_id_records"],
            "median_error_km": float(np.median(errors)), "mean_error_km": float(np.mean(errors)),
            "top1_region_accuracy": float(np.mean(region1)), "top5_region_accuracy": float(np.mean(region5)),
            "per_query_predictions": predictions,
            "definition": "Top-k accuracy means the query's region label appears among k nearest references; it is not exact-image retrieval accuracy.",
            "limitation": "Results use held-out Mapillary sequences within Downtown Pittsburgh. They do not establish accuracy in other locations, cities, or independent imagery collections."}


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
