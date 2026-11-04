"""
app.py — Streamlit web interface for GeoVision.

Run with:
    streamlit run src/app.py

Upload any street-level photograph; the app will:
  1. Extract a visual embedding using a pretrained ResNet-18.
  2. Search the reference database for the visually closest images.
  3. Show the predicted coordinates and matched reference image.
  4. Display the predicted location on an interactive map.
"""

import sys
import math
from pathlib import Path
import streamlit as st
import numpy as np
from PIL import Image

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import predict, load_index, load_model, haversine_km
from config import INDEX_DIR, BASE_DIR

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GeoVision",
    page_icon="🌍",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def google_maps_url(lat: float, lon: float) -> str:
    return f"https://www.google.com/maps?q={lat},{lon}"


def check_index_ready() -> bool:
    return (INDEX_DIR / "embeddings.npy").exists() and \
           (INDEX_DIR / "metadata.json").exists()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/"
             "No_image_available_600_x_450.svg/300px-No_image_available_600_x_450.svg.png",
             width=60)
    st.title("GeoVision 🌍")
    st.markdown(
        "**AI-based street-image geolocation**\n\n"
        "Upload a street photo → the model predicts where in the world it was taken."
    )
    st.divider()
    st.markdown("### How it works")
    st.markdown(
        "1. A pretrained **ResNet-18** extracts a visual fingerprint (embedding) "
        "from your photo.\n"
        "2. That fingerprint is compared against a reference database of "
        "geo-tagged images downloaded from Wikimedia Commons.\n"
        "3. The location of the most visually similar reference image is "
        "returned as the prediction."
    )
    st.divider()
    st.markdown(
        "⚠️ **Accuracy note**\n\n"
        "Similarity score ≠ geographic accuracy. "
        "A high score means the query *looks like* the matched image — "
        "not that it was taken at those exact coordinates."
    )
    top_k = st.slider("Top-K matches to show", min_value=1, max_value=10, value=3)

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("GeoVision 🌍")
st.subheader("AI-Based Street-Level Image Geolocation")

# Index check
if not check_index_ready():
    st.error(
        "⚠️ No index found. Please run these commands first:\n\n"
        "```\n"
        "python src/download_data.py\n"
        "python src/build_index.py\n"
        "```"
    )
    st.stop()

# Load index stats
emb_matrix, meta = load_index()
st.info(
    f"📚 Reference database: **{len(meta)} images** from "
    f"{len(set(m['region'] for m in meta))} regions  |  "
    f"Embedding size: {emb_matrix.shape[1]} dims  |  "
    f"Model: ResNet-18 (CPU)"
)

st.divider()

# File upload
uploaded = st.file_uploader(
    "Upload a street-level photograph",
    type=["jpg", "jpeg", "png", "webp"],
    help="Any outdoor/street photo works. The model is best at landmark-rich scenes.",
)

if uploaded is not None:
    # Save upload to a temporary path
    tmp_dir  = BASE_DIR / "data" / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / "query.jpg"

    query_img = Image.open(uploaded).convert("RGB")
    query_img.save(tmp_path, "JPEG")

    # Layout
    col_query, col_match = st.columns(2)

    with col_query:
        st.markdown("#### Your uploaded photo")
        st.image(query_img, use_container_width=True)

    # Run prediction
    with st.spinner("Analysing image …"):
        try:
            results = predict(tmp_path, top_k=top_k)
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()

    best = results[0]

    with col_match:
        st.markdown(f"#### Best match  —  {best['region']}")
        ref_path = Path(best["ref_path"])
        if ref_path.exists():
            st.image(str(ref_path), use_container_width=True)
        else:
            st.warning("Reference image file not found on disk.")
        st.caption(
            f"Wikimedia: {best.get('title', 'N/A')}  |  "
            f"License: {best.get('license', 'Unknown')}"
        )

    st.divider()

    # Prediction summary
    st.markdown("### 📍 Predicted Location")
    pred_lat = best["lat"]
    pred_lon = best["lon"]

    m1, m2, m3 = st.columns(3)
    m1.metric("Latitude",   f"{pred_lat:.5f}°")
    m2.metric("Longitude",  f"{pred_lon:.5f}°")
    m3.metric("Similarity", f"{best['similarity']:.4f}")

    maps_url = google_maps_url(pred_lat, pred_lon)
    st.markdown(f"[📌 Open in Google Maps]({maps_url})")

    # Map
    st.markdown("### 🗺️ Location Map")
    import pandas as pd
    map_data = pd.DataFrame({"lat": [pred_lat], "lon": [pred_lon]})
    st.map(map_data, zoom=4)

    # All top-K results table
    if len(results) > 1:
        st.divider()
        st.markdown("### 🏆 All Top Matches")
        import pandas as pd
        rows = []
        for r in results:
            rows.append({
                "Rank":       r["rank"],
                "Region":     r["region"],
                "Similarity": f"{r['similarity']:.4f}",
                "Latitude":   f"{r['lat']:.5f}",
                "Longitude":  f"{r['lon']:.5f}",
                "Maps Link":  google_maps_url(r["lat"], r["lon"]),
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

    # Attribution
    st.divider()
    st.caption(
        "Reference images sourced from Wikimedia Commons under Creative Commons licences. "
        "Predictions are for educational purposes only."
    )

