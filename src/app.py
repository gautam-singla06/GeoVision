"""Local Streamlit interface for Mapillary-backed GeoVision retrieval."""
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import INDEX_DIR, MAPILLARY_DIR, MODEL_NAME, REGION
from predict import load_index, predict

st.set_page_config(page_title="GeoVision", page_icon="🌍", layout="wide")
st.title("🌍 GeoVision")
st.caption("Street image location retrieval with GPS-labelled Mapillary imagery")
st.markdown("GeoVision compares an uploaded image with a small local reference index using pretrained ResNet-18 embeddings. The closest reference coordinate is a retrieval estimate; similarity is visual resemblance, not a calibrated probability of geographic correctness.")

with st.sidebar:
    st.header("Project")
    st.write("Reference region: Downtown Pittsburgh, Pennsylvania")
    st.write(f"Model: {MODEL_NAME} · CPU inference")
    st.caption("Imagery attribution: © Mapillary. Follow Mapillary terms and image-specific rights for reuse.")

meta_path = MAPILLARY_DIR / "metadata.json"
if meta_path.exists():
    records = json.loads(meta_path.read_text(encoding="utf-8"))
    total_bytes = sum(r.get("bytes", 0) for r in records)
    c1, c2, c3 = st.columns(3)
    c1.metric("Reference images", len(records))
    c2.metric("Dataset size", f"{total_bytes / (1024 * 1024):.1f} MB")
    c3.metric("Region", REGION["name"])
else:
    records = []
    st.info("No Mapillary dataset is collected yet. Run `python src/download_data.py --limit 20` to opt in to a small validation collection.")

eval_path = INDEX_DIR / "evaluation.json"
if eval_path.exists():
    st.subheader("Held-out evaluation")
    st.json(json.loads(eval_path.read_text(encoding="utf-8")))

st.subheader("Locate an image")
index_meta_path = INDEX_DIR / "metadata.json"
index_ready = False
if (INDEX_DIR / "embeddings.npy").exists() and index_meta_path.exists():
    try:
        index_rows = json.loads(index_meta_path.read_text(encoding="utf-8"))
        index_ready = bool(index_rows) and all(row.get("source") == "Mapillary" for row in index_rows)
    except (OSError, ValueError):
        index_ready = False
if not index_ready:
    st.info("A Mapillary reference index is not ready. Collect data, then run `python src/build_index.py`.")
uploaded = st.file_uploader("Upload a street-level image", type=["jpg", "jpeg", "png", "webp"])
if uploaded:
    try:
        image = Image.open(uploaded).convert("RGB")
        left, right = st.columns(2)
        left.image(image, caption="Query image", use_container_width=True)
        if not index_ready:
            st.error("The reference index is not ready. Run `python src/build_index.py` after collecting images.")
        else:
            suffix = Path(uploaded.name).suffix or ".jpg"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                image.save(tmp, format="JPEG")
                query_path = tmp.name
            try:
                with st.spinner("Extracting image features and searching references…"):
                    matches = predict(query_path, top_k=5)
            finally:
                Path(query_path).unlink(missing_ok=True)
            best = matches[0]
            right.metric("Predicted latitude", f"{best['lat']:.6f}°")
            right.metric("Predicted longitude", f"{best['lon']:.6f}°")
            right.metric("Top match similarity", f"{best['similarity']:.4f}")
            st.caption("Similarity scores measure embedding similarity only. They are not prediction confidence or accuracy.")
            st.map(pd.DataFrame([{"lat": best["lat"], "lon": best["lon"]}]), zoom=13)
            st.subheader("Top 5 reference matches")
            cols = st.columns(min(5, len(matches)))
            for col, match in zip(cols, matches):
                with col:
                    ref = Path(match["ref_path"])
                    if ref.exists():
                        st.image(str(ref), use_container_width=True)
                    st.markdown(f"**#{match['rank']} · {match['similarity']:.4f}**")
                    st.caption(f"{match['lat']:.5f}, {match['lon']:.5f}\n\nID: {match.get('image_id', '')}\n\n© Mapillary")
    except (OSError, ValueError, RuntimeError, FileNotFoundError) as exc:
        st.error(f"Could not process this image: {exc}")

with st.expander("How this prototype works and its limits"):
    st.write("ResNet-18 produces a 512-dimensional visual embedding. GeoVision ranks reference images by cosine similarity and uses the best match's GPS coordinate as its prediction. A proper held-out evaluation is available with `python src/evaluate.py`; with a small or single-location sample, reported metrics may not be meaningful.")
