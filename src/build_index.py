"""
build_index.py — Extract CNN embeddings and build a searchable index.

Usage:
    python src/build_index.py

Run this AFTER download_data.py has populated data/raw/.
It will produce two files in data/index/:
    embeddings.npy   — (N, 512) float32 embedding matrix
    metadata.json    — list of N dicts with filename / lat / lon / region
"""

import sys
import json
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RAW_DIR, INDEX_DIR, MODEL_NAME, EMBEDDING_DIM

# ---------------------------------------------------------------------------
# Lazy torch import so the script gives a clean error if torch is missing
# ---------------------------------------------------------------------------
def _load_model():
    import torch
    import torchvision.models as models
    import torchvision.transforms as T

    # Load pretrained ResNet-18 (downloads weights ~45 MB on first run)
    weights_enum = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights_enum)

    # Remove the final classification head → keep feature extractor only
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()

    # Standard ImageNet normalisation
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    return model, transform, torch


def extract_embedding(model, transform, torch, img_path: Path) -> np.ndarray | None:
    """Return a 1-D normalised L2 embedding for one image file, or None on error."""
    try:
        img = Image.open(img_path).convert("RGB")
        tensor = transform(img).unsqueeze(0)   # (1, 3, 224, 224)
        with torch.no_grad():
            feat = model(tensor)               # (1, 512, 1, 1)
        vec = feat.squeeze().numpy()           # (512,)
        norm = np.linalg.norm(vec)
        if norm < 1e-8:
            return None
        return (vec / norm).astype(np.float32)
    except Exception:
        return None


def main() -> None:
    # ---- collect all metadata records ----
    combined_meta_path = RAW_DIR / "all_metadata.json"
    if not combined_meta_path.exists():
        # Fall back to per-region files
        all_meta: list[dict] = []
        for region_dir in sorted(RAW_DIR.iterdir()):
            mp = region_dir / "metadata.json"
            if mp.exists():
                with open(mp, encoding="utf-8") as f:
                    all_meta.extend(json.load(f))
    else:
        with open(combined_meta_path, encoding="utf-8") as f:
            all_meta = json.load(f)

    if not all_meta:
        print("[ERROR] No images found in data/raw/. Run download_data.py first.")
        sys.exit(1)

    print(f"Found {len(all_meta)} images. Loading model …")
    model, transform, torch = _load_model()
    print(f"Model: {MODEL_NAME}  (CPU, no GPU needed)")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    embeddings: list[np.ndarray] = []
    kept_meta:  list[dict]       = []

    for record in tqdm(all_meta, desc="Extracting features"):
        img_path = Path(record["path"])
        if not img_path.exists():
            continue
        vec = extract_embedding(model, transform, torch, img_path)
        if vec is not None:
            embeddings.append(vec)
            kept_meta.append(record)

    if not embeddings:
        print("[ERROR] No embeddings produced. Check that images exist in data/raw/.")
        sys.exit(1)

    emb_matrix = np.stack(embeddings, axis=0)  # (N, 512)

    # ---- save ----
    emb_path  = INDEX_DIR / "embeddings.npy"
    meta_path = INDEX_DIR / "metadata.json"

    np.save(str(emb_path), emb_matrix)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(kept_meta, f, indent=2, ensure_ascii=False)

    print(f"\n✓  Index built:  {emb_matrix.shape[0]} images  ×  {emb_matrix.shape[1]} dims")
    print(f"   Embeddings  → {emb_path}")
    print(f"   Metadata    → {meta_path}")


if __name__ == "__main__":
    main()

