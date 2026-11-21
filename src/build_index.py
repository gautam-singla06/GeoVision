"""Create a CPU ResNet-18 cosine index from Mapillary reference images."""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import EMBEDDING_DIM, INDEX_DIR, MAPILLARY_DIR, MODEL_NAME, PROCESSED_DIR


def load_metadata():
    path = MAPILLARY_DIR / "metadata.json"
    if not path.exists():
        raise FileNotFoundError("No Mapillary metadata found. Run python src/download_data.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


def load_model():
    import torch
    from torchvision.models import ResNet18_Weights, resnet18
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    transform = weights.transforms()
    return model, transform, torch


def extract_embedding(model, transform, torch, img_path: Path) -> np.ndarray | None:
    try:
        with Image.open(img_path) as im:
            tensor = transform(im.convert("RGB")).unsqueeze(0)
        with torch.inference_mode():
            vector = model(tensor).squeeze(0).cpu().numpy()
        norm = np.linalg.norm(vector)
        return (vector / norm).astype(np.float32) if norm > 1e-8 else None
    except (OSError, ValueError, RuntimeError):
        return None


def build_index(records=None):
    records = load_metadata() if records is None else records
    valid = [r for r in records if Path(r["path"]).is_file()]
    if not valid:
        raise ValueError("No readable Mapillary images found; collect images first.")
    model, transform, torch = load_model()
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    embeddings, kept = [], []
    for record in tqdm(valid, desc="Embedding reference images"):
        processed_dir = PROCESSED_DIR / "mapillary"
        processed_dir.mkdir(parents=True, exist_ok=True)
        processed_path = processed_dir / f"{record.get('image_id', Path(record['path']).stem)}.jpg"
        try:
            with Image.open(record["path"]) as im:
                im.convert("RGB").resize((224, 224), Image.Resampling.LANCZOS).save(processed_path, "JPEG", quality=88)
        except OSError:
            continue
        vector = extract_embedding(model, transform, torch, processed_path)
        if vector is not None:
            embeddings.append(vector)
            kept.append({**record, "raw_path": record["path"], "path": str(processed_path)})
    if not embeddings:
        raise ValueError("No embeddings could be produced from the available images.")
    matrix = np.stack(embeddings)
    if matrix.shape[1] != EMBEDDING_DIM:
        raise ValueError(f"Unexpected embedding dimension: {matrix.shape[1]}")
    np.save(INDEX_DIR / "embeddings.npy", matrix)
    (INDEX_DIR / "metadata.json").write_text(json.dumps(kept, indent=2), encoding="utf-8")
    stats = {"image_count": len(kept), "embedding_dim": int(matrix.shape[1]), "model": MODEL_NAME,
             "regions": sorted({r["region"] for r in kept}), "source": "Mapillary"}
    (INDEX_DIR / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


if __name__ == "__main__":
    try:
        print(json.dumps(build_index(), indent=2))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
