"""GeoVision geospatial AI research dashboard."""
import json
import math
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image
try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    px = go = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import INDEX_DIR, MODEL_NAME
from predict import predict
from research import audit_records, legacy_records, mapillary_records

st.set_page_config(page_title="GeoVision | Geospatial Intelligence", page_icon="\U0001F4CD", layout="wide", initial_sidebar_state="expanded")
PAGES = {
    "01 · Project Overview": "Overview", "02 · Dataset Collection": "Dataset Collection",
    "03 · Dataset Exploration": "Dataset Explorer", "04 · Cleaning & Preprocessing": "Preprocessing",
    "05 · Feature Extraction": "Feature Extraction", "06 · Model Architecture": "Architecture",
    "07 · Model Training": "Model Training", "08 · Decision Tree": "Decision Tree",
    "09 · Random Forest": "Random Forest", "10 · Hyperparameter Tuning": "Tuning",
    "11 · Model Comparison": "Model Comparison", "12 · Evaluation Metrics": "Evaluation",
    "13 · Error Analysis": "Error Analysis", "14 · Actual vs Predicted GPS": "Actual vs Predicted",
    "15 · Feature Importance": "Feature Importance", "16 · Results & Conclusion": "Conclusion",
}
with st.sidebar:
    st.markdown("<div style='padding:8px 4px 20px'><div class='eyebrow'>GEOSPATIAL AI</div><h2 style='margin:4px 0'>GeoVision</h2><p>Research workspace · CPU inference</p></div>", unsafe_allow_html=True)
    selection = st.radio("WORKSPACE", list(PAGES), label_visibility="collapsed")
    page = PAGES[selection]
    light_theme = st.toggle("Light theme", value=False, help="Switch between charcoal and light surfaces.")
    st.markdown("<hr style='border-color:#e2eae8'>", unsafe_allow_html=True)
    st.caption(f"Backbone · {MODEL_NAME}")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
:root{--canvas:#15191F;--card:#202630;--elevated:#2B3139;--accent:#FCD535;--ink:#EAECEF;--secondary:#A0A8B3;--border:#343B45;--success:#0ECB81;--error:#F6465D;--info:#3B82F6}
.stApp{background:var(--canvas)!important;color:var(--ink)!important;font-family:Inter,Arial,sans-serif!important;font-feature-settings:"ss01"}
[data-testid="stHeader"]{background:var(--canvas)!important}
[data-testid="stSidebar"]{background:#202630!important;border-right:1px solid var(--border)!important}
[data-testid="stSidebar"] *{color:var(--ink)!important}
h1,h2,h3,h4{color:var(--ink)!important;font-family:Inter,Arial,sans-serif!important;font-weight:600!important;letter-spacing:-.025em!important}
p,li,label,[data-testid="stCaptionContainer"],.stCaption{color:var(--secondary)!important}
.eyebrow{color:var(--accent)!important;font-size:.72rem;font-weight:600;letter-spacing:.12em;text-transform:uppercase}
[data-testid="stMetric"]{background:var(--card)!important;border:1px solid var(--border)!important;border-radius:8px!important;box-shadow:none!important;padding:16px 18px!important}
[data-testid="stMetricLabel"]{color:var(--secondary)!important}
[data-testid="stMetricValue"]{color:var(--accent)!important;font-family:'JetBrains Mono',monospace!important;font-feature-settings:'tnum'!important;letter-spacing:-.025em}
.hero{background:#202630!important;border:1px solid var(--border)!important;border-radius:12px!important;box-shadow:none!important;padding:28px 32px;margin:8px 0 22px}
.hero .eyebrow{color:var(--accent)!important}.hero h1{color:var(--ink)!important}.hero p{color:var(--secondary)!important}
.card{background:var(--card)!important;border:1px solid var(--border)!important;border-radius:8px!important;box-shadow:none!important;padding:18px 20px;margin:5px 0;min-height:106px}
.card h3{color:var(--ink)!important;font-size:1rem!important;margin:0 0 8px}.card p{color:var(--secondary)!important;line-height:1.55;margin:0}
.tag{display:inline-block;background:#343b45!important;color:var(--accent)!important;border:1px solid #4a525c!important;border-radius:6px!important;padding:5px 9px;font-size:.77rem;font-weight:600}
div[data-testid="stFileUploader"]{background:var(--card)!important;border:1px dashed #59616b!important;border-radius:8px!important}
div.stButton>button{background:var(--accent)!important;color:#181a20!important;border:0!important;border-radius:6px!important;font-weight:600!important;min-height:40px!important;transition:filter .15s ease}
div.stButton>button:hover{filter:brightness(1.06)}
div[data-baseweb="input"] input,div[data-baseweb="select"]>div,textarea{background:#2B3139!important;color:#EAECEF!important;border-color:#343B45!important}
[data-baseweb="popover"],[data-baseweb="menu"]{background:#202630!important;color:#EAECEF!important}
[data-testid="stTextInput"] input,[data-testid="stNumberInput"] input{color:#EAECEF!important}
[data-testid="stDataFrame"]{border:1px solid var(--border)!important;border-radius:8px!important;overflow:hidden}
[data-testid="stAlert"]{background:#202630!important;color:#EAECEF!important;border:1px solid var(--border)!important}
details{background:var(--card)!important;border:1px solid var(--border)!important;border-radius:8px!important}
.stMarkdown a{color:#7ca7ff!important}.plotly .main-svg{font-family:Inter,'JetBrains Mono',sans-serif!important}
@media(max-width:768px){.hero{padding:20px!important}.hero h1{font-size:1.8rem!important}.card{padding:15px!important}[data-testid="stMetric"]{padding:12px!important}}
</style>
""", unsafe_allow_html=True)
DARK_MODE = not light_theme
PLOT = {"paper": "#15191F" if DARK_MODE else "#ffffff", "surface": "#202630" if DARK_MODE else "#ffffff",
        "text": "#EAECEF" if DARK_MODE else "#0d253d", "muted": "#A0A8B3" if DARK_MODE else "#64748d",
        "grid": "#343B45" if DARK_MODE else "#e3e8ee", "title": "#FCD535" if DARK_MODE else "#1c1e54",
        "hover": "#2B3139" if DARK_MODE else "#1c1e54", "accent": "#FCD535" if DARK_MODE else "#533afd",
        "blue": "#3B82F6", "success": "#0ECB81", "error": "#F6465D"}
if light_theme:
    st.markdown("""<style>
    .stApp{background:#f6f9fc!important;color:#0d253d!important}
    [data-testid="stHeader"],[data-testid="stSidebar"]{background:#fff!important;color:#0d253d!important}
    [data-testid="stSidebar"] *{color:#0d253d!important}
    [data-testid="stMetric"],.card,details,[data-testid="stAlert"]{background:#fff!important;color:#0d253d!important;border-color:#e3e8ee!important;box-shadow:0 1px 3px #0000000c!important}
    [data-testid="stMetricLabel"],.card p,p,li,label,[data-testid="stCaptionContainer"],.stCaption{color:#566477!important}
    [data-testid="stMetricValue"],h1,h2,h3,h4,.card h3{color:#0d253d!important}
    .eyebrow{color:#533afd!important}.hero{background:#202630!important;border-color:#343b45!important}.hero .eyebrow{color:#fcd535!important}.hero h1,.hero p{color:#eaecef!important}
    .tag{background:#eef0ff!important;color:#3730a3!important;border-color:#dce1ff!important}
    div.stButton>button{background:#533afd!important;color:#fff!important}
    div[data-baseweb="input"] input,div[data-baseweb="select"]>div,textarea{background:#fff!important;color:#0d253d!important;border-color:#cbd5e1!important}
    [data-baseweb="popover"],[data-baseweb="menu"]{background:#fff!important;color:#0d253d!important}
    div[data-testid="stFileUploader"]{background:#fff!important;border-color:#a0a8b3!important}
    [data-testid="stAlert"]{background:#eef3ff!important;color:#273951!important;border-color:#dbe5ff!important}
    .stMarkdown a{color:#3b5bcc!important}
    </style>""",unsafe_allow_html=True)

@st.cache_data(show_spinner="Auditing reference imagery…")
def load_audit():
    current_records = mapillary_records()
    current_frame, current_summary = audit_records(current_records)
    return current_records, current_frame, current_summary

records, frame, summary = load_audit()
legacy_count = len(legacy_records())

def saved(name):
    path = INDEX_DIR / name
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except (OSError, ValueError):
        return None

def notebook_results():
    path=Path(__file__).resolve().parent/"notebook_experiments.json"
    try: return json.loads(path.read_text(encoding="utf-8"))
    except (OSError,ValueError): return None

def heading(kicker, title, description):
    st.markdown(f"<div class='eyebrow'>{kicker}</div><h1 style='margin:2px 0 4px'>{title}</h1><p style='margin-top:0'>{description}</p>", unsafe_allow_html=True)

def card(title, text):
    st.markdown(f"<div class='card'><h3>{title}</h3><p>{text}</p></div>", unsafe_allow_html=True)

def metric_row(items):
    cols = st.columns(len(items))
    for col, (label, value, help_text) in zip(cols, items):
        col.metric(label, value, help=help_text)

def style_plot(fig, height=360):
    """Apply explicit high-contrast theme, labels, axes and hover styling."""
    fig.update_layout(
        template=go.layout.Template(layout=go.Layout(colorway=[PLOT["accent"], PLOT["blue"], PLOT["success"], PLOT["error"]])),
        paper_bgcolor=PLOT["paper"], plot_bgcolor=PLOT["surface"],
        font=dict(family="Inter, JetBrains Mono, sans-serif", size=13, color=PLOT["text"]),
        title=dict(font=dict(size=19, color=PLOT["title"]), x=0.02, xanchor="left"),
        legend=dict(font=dict(color=PLOT["text"], size=12), bgcolor=PLOT["surface"],
                    bordercolor=PLOT["grid"], borderwidth=1),
        hoverlabel=dict(bgcolor=PLOT["hover"], bordercolor=PLOT["grid"],
                        font=dict(color="#ffffff", family="Inter, sans-serif", size=12)),
        margin=dict(l=68, r=28, t=64, b=62), height=height,
        autosize=True,
    )
    fig.update_xaxes(title_font=dict(color=PLOT["text"], size=13), tickfont=dict(color=PLOT["muted"], size=12),
                     gridcolor=PLOT["grid"], zerolinecolor=PLOT["grid"], automargin=True)
    fig.update_yaxes(title_font=dict(color=PLOT["text"], size=13), tickfont=dict(color=PLOT["muted"], size=12),
                     gridcolor=PLOT["grid"], zerolinecolor=PLOT["grid"], automargin=True)
    return fig

def compare_chart(labels, values, title, y_title, percent=False):
    if px is not None:
        chart_data = pd.DataFrame({"Metric": labels, "Value": values})
        fig = px.bar(chart_data, x="Metric", y="Value", color="Metric",
                     color_discrete_sequence=[PLOT["accent"], PLOT["blue"]], title=title,
                     hover_data={"Value": ":.3f" if not percent else ".1%"})
        fig.update_layout(yaxis_title=y_title, showlegend=False)
        if percent: fig.update_yaxes(tickformat=".0%", range=[0, 1.05])
        style_plot(fig,320)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("Interactive charts require Plotly. Install the project requirements to enable them.")
        st.bar_chart(pd.DataFrame({"Metric": labels, "Value": values}).set_index("Metric"))

if page == "Overview":
    st.markdown("<div class='hero'><div class='eyebrow'>LOCATION INTELLIGENCE · RESEARCH PREVIEW</div><h1>Find the place behind the picture.</h1><p>GeoVision compares street-level imagery using visual embeddings and GPS-labelled references, then maps the closest visual matches.</p><span class='tag'>Downtown Pittsburgh · single-region corpus</span></div>", unsafe_allow_html=True)
    metric_row([("Reference images", f"{summary['records']:,}", "Valid local Mapillary images with GPS"), ("Coverage", f"{summary['regions']} region", "All available reference imagery is from Downtown Pittsburgh"), ("GPS coverage", f"{summary['gps_complete']:,} / {summary['records']:,}", "Eligible records with latitude and longitude"), ("Sequences", f"{(saved('evaluation.json') or {}).get('unique_sequence_count', '—')}", "Sequences represented in the saved held-out evaluation")])
    st.subheader("Research workflow")
    cols=st.columns(4)
    for col,title,body in zip(cols,["01 · Collect","02 · Embed","03 · Retrieve","04 · Evaluate"],["GPS-labelled Mapillary frames with source metadata.","CPU ResNet-18 converts each frame into a normalized feature vector.","Cosine similarity ranks reference frames for an uploaded image.","Held-out sequences measure GPS distance and region retrieval."]):
        with col: card(title,body)
    st.info("Scope: 454 images from one region support a local retrieval demonstration. The 100% region retrieval score only reflects this single-region label and does not mean 100% geolocation accuracy.")
elif page == "Dataset Collection":
    heading("DATA ORIGIN", "Dataset collection", "A local Mapillary corpus with GPS and capture sequence provenance.")
    metric_row([("Verified images",summary["readable_images"],"Readable image files with valid coordinates"),("GPS complete",summary["gps_complete"],"Records with latitude and longitude"),("Capture sequences",frame["sequence_id"].nunique() if "sequence_id" in frame else 0,"Unique Mapillary sequence IDs"),("Regions",summary["regions"],"Current collection is Downtown Pittsburgh")])
    st.markdown("Images were collected through the Mapillary Graph API. The Colab notebook also contains an unexecuted Wikimedia Paris API demo; it has no saved outputs and is not treated as a measured experiment.")
elif page == "Preprocessing":
    heading("DATA QUALITY", "Cleaning & preprocessing", "Measured properties from the current local image records.")
    metric_row([("Readable",summary["readable_images"],"Pillow image decode succeeds"),("GPS complete",summary["gps_complete"],"Valid GPS fields"),("Repeated dHash",summary["exact_dhash_duplicates"],"Exact perceptual hash duplicates"),("Missing values",summary["missing_values"],"Nulls across audited metadata and image audit columns")])
    if not frame.empty and px is not None:
        dims=frame.dropna(subset=["width","height"])
        plot=px.scatter(dims,x="width",y="height",color="region",hover_data=["image_id","sequence_id"],title="Reference image dimensions",labels={"width":"Width (pixels)","height":"Height (pixels)"})
        style_plot(plot,400); st.plotly_chart(plot,use_container_width=True)
    with st.expander("Preprocessing method"):
        st.write("The existing index builder converts images to RGB, resizes to 224 × 224, applies ImageNet ResNet preprocessing, and L2-normalizes embeddings. Evaluation holds full capture sequences apart.")
elif page == "Feature Extraction":
    heading("VISUAL REPRESENTATION", "Feature extraction", "Actual saved CPU embedding index statistics.")
    stats=saved("stats.json") or {}
    metric_row([("Indexed images",stats.get("image_count",0),"Rows in saved embedding index"),("Dimensions",stats.get("embedding_dim",0),"ResNet-18 feature vector width"),("Backbone",stats.get("model",MODEL_NAME),"Pretrained feature extractor"),("Runtime", "CPU", "No dedicated GPU required")])
    st.write("Embeddings are visual feature vectors used for cosine-similarity retrieval. Individual dimensions are not interpretable concepts.")
elif page == "Architecture":
    heading("MODEL DESIGN", "ResNet-18 retrieval architecture", "The implemented system is a feature extractor and nearest-reference retriever.")
    for c,t,b in zip(st.columns(3),["Image","ResNet-18","Cosine retrieval"],["RGB input resized and ImageNet-normalized.","512-dimensional L2-normalized representation.","Rank local references; report the nearest reference GPS."]):
        with c: card(t,b)
    st.caption("The Mapillary retrieval system does not train ResNet-18 end-to-end for coordinates.")
elif page == "Dataset Explorer":
    heading("REFERENCE CORPUS", "Dataset Explorer", "Inspect the verified local image records, their coverage and capture sequences.")
    metric_row([("Images",f"{summary['readable_images']:,}","Readable local files"),("Regions",summary["regions"],"Distinct region labels"),("Sequences",frame["sequence_id"].nunique() if "sequence_id" in frame else 0,"Distinct capture sequences"),("Duplicate hashes",summary["exact_dhash_duplicates"],"Exact dHash repeats")])
    if frame.empty: st.warning("No valid reference image records were found locally.")
    else:
        q=st.text_input("Search records",placeholder="Filter by image ID, region, sequence, or filename")
        view=frame.copy()
        if q:
            mask=view.astype(str).apply(lambda col: col.str.contains(q,case=False,na=False)).any(axis=1); view=view[mask]
        left,right=st.columns([1.15,.85])
        with left:
            st.dataframe(view[[c for c in ["image_id","filename","region","sequence_id","lat","lon","width","height","quality_stddev"] if c in view]],use_container_width=True,height=400,hide_index=True)
            if "region" in frame: compare_chart(frame["region"].value_counts().index.tolist(),frame["region"].value_counts().values.tolist(),"Images by region","Image count")
            if "sequence_id" in frame:
                counts=frame["sequence_id"].value_counts().sort_values(ascending=False).head(15)
                compare_chart(counts.index.tolist(),counts.values.tolist(),"Largest capture sequences","Image count")
        with right:
            st.subheader("Image preview")
            options=view.index.tolist()
            if options:
                idx=st.selectbox("Select a record",options,format_func=lambda i: str(view.loc[i].get("image_id",view.loc[i].get("filename",i))))
                row=view.loc[idx]; st.image(row["path"],use_container_width=True)
                st.caption(f"{row.get('lat','')}, {row.get('lon','')} · {row.get('sequence_id','Sequence unavailable')} · © Mapillary")
            st.subheader("GPS distribution")
            map_rows=view.dropna(subset=["lat","lon"])
            if px is not None and not map_rows.empty:
                center={"lat":float(map_rows.lat.mean()),"lon":float(map_rows.lon.mean())}
                geo=px.scatter_map(map_rows,lat="lat",lon="lon",color="region",hover_data=[c for c in ["image_id","sequence_id"] if c in map_rows],zoom=13,center=center,map_style="carto-darkmatter" if DARK_MODE else "carto-positron",title="Reference image GPS",color_discrete_sequence=[PLOT["accent"]])
                geo.update_traces(marker=dict(size=5,opacity=.52),cluster=dict(enabled=True,maxzoom=14))
                geo.update_layout(map=dict(zoom=13,center=center),margin=dict(l=0,r=0,t=52,b=0),height=330,showlegend=False,paper_bgcolor=PLOT["paper"],plot_bgcolor=PLOT["surface"],font=dict(color=PLOT["text"]),title=dict(font=dict(color=PLOT["title"],size=17)))
                st.plotly_chart(geo,use_container_width=True)
            if "quality_stddev" in view:
                qdf=view["quality_stddev"].dropna().reset_index(drop=True).rename("Grayscale standard deviation").to_frame()
                quality=px.line(qdf,x=qdf.index,y="Grayscale standard deviation",title="Image grayscale variation")
                quality.update_layout(xaxis_title="Audited image index",yaxis_title="Grayscale standard deviation (0 to 255)")
                style_plot(quality,280); st.plotly_chart(quality,use_container_width=True)
elif page == "Evaluation":
    heading("HELD-OUT SEQUENCES", "Evaluation & error analysis", "Measured retrieval results from the saved evaluation artifact.")
    ev=saved("evaluation.json")
    if not ev: st.warning("No evaluation results are available. Run the sequence-aware evaluation after collecting eligible data.")
    else:
        metric_row([("Mean GPS error",f"{ev['mean_error_km']:.3f} km","Mean great-circle distance from query GPS to the top reference GPS"),("Median GPS error",f"{ev['median_error_km']:.3f} km","Median great-circle distance"),("Reference set",f"{ev['reference_count']} images",f"{ev.get('reference_sequence_count','—')} sequences"),("Query set",f"{ev['query_count']} images",f"{ev.get('query_sequence_count','—')} held-out sequences")])
        st.subheader("Region-label retrieval")
        a,b=st.columns(2); a.metric("Top-1 region accuracy",f"{ev['top1_region_accuracy']:.0%}"); b.metric("Top-5 region accuracy",f"{ev['top5_region_accuracy']:.0%}")
        st.caption("Both scores are region-label retrieval accuracy. Since the dataset contains only Downtown Pittsburgh, 100% does not indicate correct GPS localization.")
        charts=st.columns(2)
        with charts[0]: compare_chart(["Mean error","Median error"],[ev["mean_error_km"],ev["median_error_km"]],"GPS error summary","Great-circle error (km)")
        with charts[1]: compare_chart(["Top-1 region","Top-5 region"],[ev["top1_region_accuracy"],ev["top5_region_accuracy"]],"Region-label retrieval","Accuracy",percent=True)
        st.markdown("<div class='card'><h3>Metric guide</h3><p><b>Mean GPS error</b> averages query-to-match great-circle distances. <b>Median GPS error</b> is the midpoint and is less affected by large outliers. <b>Top-k region accuracy</b> checks whether the correct region label appears in the first k results; it does not measure coordinate accuracy.</p></div>",unsafe_allow_html=True)
        st.markdown(f"<div class='card'><h3>Evaluation scope</h3><p>{ev.get('limitation','Results are specific to this evaluation split.')}</p><p style='margin-top:8px'>{ev.get('definition','')}</p><p style='margin-top:8px'>Records missing sequence IDs and excluded from the split: {ev.get('missing_sequence_id_records','not recorded')}.</p></div>",unsafe_allow_html=True)
        st.caption("The current evaluation artifact also contains paired held-out coordinates and query-level GPS errors.")
elif page == "Live Prediction":
    heading("IMAGE RETRIEVAL", "Live prediction", "Upload a street-level image to retrieve its closest visual references.")
    ready=bool(records) and (INDEX_DIR/"embeddings.npy").exists() and (INDEX_DIR/"metadata.json").exists()
    uploaded=st.file_uploader("Choose a street image",type=["jpg","jpeg","png","webp"],label_visibility="collapsed")
    if not ready: st.warning("Prediction index is unavailable for the verified Mapillary reference corpus.")
    if uploaded and ready:
        try:
            image=Image.open(uploaded).convert("RGB")
            l,r=st.columns([.9,1.1]); l.image(image,caption="Uploaded image",use_container_width=True)
            with tempfile.NamedTemporaryFile(suffix=".jpg",delete=False) as f: image.save(f,"JPEG"); query_path=f.name
            try:
                with st.spinner("Extracting features and ranking references…"): matches=predict(query_path,top_k=5)
            finally: Path(query_path).unlink(missing_ok=True)
            best=matches[0]
            with r:
                st.markdown("<div class='eyebrow'>RETRIEVED LOCATION</div>",unsafe_allow_html=True)
                metric_row([("Latitude",f"{best['lat']:.5f}°","Coordinate copied from the closest reference"),("Longitude",f"{best['lon']:.5f}°","Coordinate copied from the closest reference")])
                if px is not None:
                    match_frame=pd.DataFrame(matches)
                    center={"lat":float(match_frame.lat.mean()),"lon":float(match_frame.lon.mean())}
                    match_map=px.scatter_map(match_frame,lat="lat",lon="lon",color="rank",size_max=8,hover_name="image_id",hover_data=["similarity","region"],zoom=14,center=center,map_style="carto-darkmatter" if DARK_MODE else "carto-positron",title="Top retrieved reference GPS",color_continuous_scale=[[0,PLOT["accent"]],[1,PLOT["blue"]]])
                    match_map.update_traces(marker=dict(size=5,opacity=.82))
                    match_map.update_layout(map=dict(zoom=14,center=center),margin=dict(l=0,r=0,t=48,b=0),height=300,font=dict(color=PLOT["text"]),paper_bgcolor=PLOT["paper"],plot_bgcolor=PLOT["surface"],title=dict(font=dict(color=PLOT["title"],size=16)),coloraxis_colorbar=dict(title=dict(text="Match rank",font=dict(color=PLOT["text"])),tickfont=dict(color=PLOT["text"])))
                    st.plotly_chart(match_map,use_container_width=True)
                st.caption("Similarity ranks visual resemblance. It is not a probability or confidence estimate.")
            st.subheader("Top reference matches")
            cols=st.columns(len(matches))
            for col,m in zip(cols,matches):
                with col:
                    p=Path(m["ref_path"])
                    if p.is_file(): st.image(str(p),use_container_width=True)
                    st.markdown(f"**Match {m['rank']}** · {m['similarity']:.3f}")
                    st.caption(f"{m['lat']:.5f}, {m['lon']:.5f}\n\n{m.get('image_id','')} · © Mapillary")
        except (OSError,ValueError,RuntimeError,FileNotFoundError): st.error("Image prediction could not be completed. Check that the uploaded file is readable and the local index is current.")
elif page == "Actual vs Predicted":
    heading("GEOGRAPHIC COMPARISON", "Actual vs predicted", "Held-out sequence GPS compared with the nearest-reference prediction.")
    model_view=st.selectbox("Experiment",["ResNet-18 nearest-reference","Decision Tree regression","Random Forest regression"])
    ev=saved("evaluation.json"); rows=(ev or {}).get("per_query_predictions",[]) if model_view == "ResNet-18 nearest-reference" else []
    if rows:
        errors=pd.DataFrame(rows)
        metric_row([("Held-out queries",len(errors),"Per-query predictions saved by sequence-aware evaluation"),("Mean error",f"{errors.error_km.mean():.3f} km","Mean great-circle distance"),("Median error",f"{errors.error_km.median():.3f} km","Median great-circle distance")])
        region_filter=st.multiselect("Filter region",sorted(errors.region.dropna().unique()),default=sorted(errors.region.dropna().unique()))
        errors=errors[errors.region.isin(region_filter)]
        if px is not None and go is not None and not errors.empty:
            fig=go.Figure()
            link_lat=[]; link_lon=[]
            for row in errors.itertuples():
                link_lat.extend([row.actual_lat,row.predicted_lat,None])
                link_lon.extend([row.actual_lon,row.predicted_lon,None])
            fig.add_trace(go.Scattermap(lat=link_lat,lon=link_lon,mode="lines",line=dict(width=1,color=PLOT["muted"]),opacity=.32,hoverinfo="skip",showlegend=False))
            fig.add_trace(go.Scattermap(lat=errors.actual_lat,lon=errors.actual_lon,mode="markers",name="Actual GPS",marker=dict(size=5,color=PLOT["accent"],opacity=.78),cluster=dict(enabled=True,maxzoom=14),customdata=errors[["image_id","error_km"]],hovertemplate="Actual<br>%{lat:.5f}, %{lon:.5f}<br>ID %{customdata[0]}<extra></extra>"))
            fig.add_trace(go.Scattermap(lat=errors.predicted_lat,lon=errors.predicted_lon,mode="markers",name="Predicted GPS",marker=dict(size=5,color=PLOT["blue"],opacity=.78),cluster=dict(enabled=True,maxzoom=14),customdata=errors[["image_id","error_km"]],hovertemplate="Predicted<br>%{lat:.5f}, %{lon:.5f}<br>Error %{customdata[1]:.3f} km<extra></extra>"))
            all_lat=pd.concat([errors.actual_lat,errors.predicted_lat])
            all_lon=pd.concat([errors.actual_lon,errors.predicted_lon])
            center={"lat":float(all_lat.mean()),"lon":float(all_lon.mean())}
            span_lat=float(all_lat.max()-all_lat.min())*111_000
            span_lon=float(all_lon.max()-all_lon.min())*111_000*math.cos(math.radians(center["lat"]))
            zoom=min(16,max(11,15-math.log2(max(span_lat,span_lon,250)/500)))
            fig.update_layout(map=dict(style="carto-darkmatter" if DARK_MODE else "carto-positron",center=center,zoom=zoom),legend=dict(orientation="h",y=1.02))
            style_plot(fig,560)
            st.plotly_chart(fig,use_container_width=True)
            left,right=st.columns(2)
            with left:
                scat=px.scatter(errors,x="actual_lat",y="predicted_lat",color="error_km",hover_data=["image_id","sequence_id","actual_lon","predicted_lon"],color_continuous_scale=[[0,PLOT["accent"]],[1,PLOT["error"]]],title="Actual vs predicted latitude",labels={"actual_lat":"Actual latitude (°)","predicted_lat":"Predicted latitude (°)","error_km":"GPS error (km)"})
                scat.update_layout(coloraxis_colorbar=dict(title=dict(text="GPS error (km)",font=dict(color=PLOT["text"])),tickfont=dict(color=PLOT["text"])))
                style_plot(scat,350); st.plotly_chart(scat,use_container_width=True)
            with right:
                hist=px.histogram(errors,x="error_km",nbins=20,color_discrete_sequence=[PLOT["blue"]],title="Held-out GPS error distribution",hover_data=["image_id"])
                hist.update_layout(xaxis_title="Great-circle error (km)",yaxis_title="Query count"); style_plot(hist,350); st.plotly_chart(hist,use_container_width=True)
            st.dataframe(errors[["image_id","sequence_id","actual_lat","actual_lon","predicted_lat","predicted_lon","error_km"]].sort_values("error_km",ascending=False),use_container_width=True,hide_index=True)
        elif errors.empty: st.info("No rows match this filter.")
        else: st.info("Install Plotly to view interactive coordinate maps and error charts.")
    else:
        st.info(f"Per-query coordinates are not saved for {model_view}. The notebook contains aggregate regression results, but no query-level Decision Tree or Random Forest coordinates were persisted.")
elif page == "Error Analysis":
    heading("HELD-OUT ERROR", "Error analysis", "Distribution and individual outliers from genuine per-query results.")
    evaluation=saved("evaluation.json"); rows=(evaluation or {}).get("per_query_predictions",[])
    if rows and px is not None:
        err=pd.DataFrame(rows)
        maximum=st.slider("Maximum displayed error (km)",float(err.error_km.min()),float(err.error_km.max()),float(err.error_km.max()))
        subset=err[err.error_km<=maximum]
        fig=px.histogram(subset,x="error_km",color="region",nbins=24,title="Held-out great-circle GPS error",hover_data=["image_id","sequence_id"])
        fig.update_layout(xaxis_title="GPS error (km)",yaxis_title="Query count"); style_plot(fig,400); st.plotly_chart(fig,use_container_width=True)
        st.dataframe(subset.sort_values("error_km",ascending=False)[["image_id","sequence_id","error_km"]],use_container_width=True,hide_index=True)
    else: st.info("Per-query GPS errors are unavailable; no distribution is estimated from aggregate metrics.")
elif page == "Methodology":
    heading("RESEARCH DESIGN", "Methodology", "A compact CPU retrieval pipeline for GPS-labelled street imagery.")
    c=st.columns(3)
    with c[0]: card("1 · Validate","Unreadable images and invalid GPS are excluded; image dimensions and dHash are audited.")
    with c[1]: card("2 · Represent","ImageNet-pretrained ResNet-18 produces normalized 512-dimensional embeddings.")
    with c[2]: card("3 · Retrieve","Cosine similarity ranks references; the top reference GPS is returned as a retrieval estimate.")
    with st.expander("Evaluation split and metric definitions"):
        st.write("Evaluation assigns entire Mapillary sequences to either the reference or query set. Reported distance is great-circle error to the top reference. Region accuracy is a separate label metric and is not GPS accuracy.")
    with st.expander("Dataset source and provenance"):
        st.markdown("[Mapillary Graph API](https://www.mapillary.com/developer/api-documentation) · preserve attribution and comply with current image terms.")
elif page in ("Model Training", "Decision Tree", "Random Forest", "Tuning", "Model Comparison", "Feature Importance"):
    title={"Model Training":"Model training","Decision Tree":"Decision Tree experiment","Random Forest":"Random Forest experiment","Tuning":"Hyperparameter tuning","Model Comparison":"Model comparison","Feature Importance":"Feature importance"}[page]
    heading("SAVED EXPERIMENTS",title,"Only existing model files and recorded experiment metrics are shown.")
    experiment=saved("regression_experiments.json")
    notebook=notebook_results() or {}
    notebook_models={row["name"]:row for row in notebook.get("experiments",[])}
    dt_result=notebook_models.get("Decision Tree")
    rf_baseline=notebook_models.get("Random Forest baseline")
    rf_tuned=notebook_models.get("Tuned Random Forest")
    provenance=notebook.get("provenance",{})
    saved_rf_path=Path(__file__).resolve().parent.parent/"models"/"evaluation.json"
    try: saved_rf=json.loads(saved_rf_path.read_text(encoding="utf-8")) if saved_rf_path.exists() else None
    except (OSError,ValueError): saved_rf=None
    if experiment:
        st.success(f"Embedding regression experiment · {experiment.get('samples','—')} samples · {experiment.get('spatial_groups','—')} spatial groups")
        st.dataframe(pd.DataFrame(experiment.get("models",{})).T,use_container_width=True)
    if saved_rf and page in ("Model Training","Random Forest","Tuning","Model Comparison"):
        st.subheader("Saved tuned Random Forest result")
        st.caption("Existing model artifact; shown separately from ResNet nearest-neighbour evaluation.")
        metric_row([("Mean GPS error",f"{saved_rf['mean_error_m']:.2f} m","Saved experiment metric"),("Median GPS error",f"{saved_rf['median_error_m']:.2f} m","Saved experiment metric"),("Within 500 m",f"{saved_rf['within_500m_percent']:.2f}%","Saved experiment threshold result")])
        st.markdown("<div class='card'><h3>Recorded configuration</h3><p>"+" · ".join(f"{k.replace('_',' ')}: {v}" for k,v in saved_rf.get("best_parameters",{}).items())+"</p></div>",unsafe_allow_html=True)
        if page in ("Random Forest","Model Training"):
            artifact_path=Path(__file__).resolve().parent.parent/"models"/"best_random_forest.pkl"
            if artifact_path.exists():
                try:
                    import joblib
                    fitted_model=joblib.load(artifact_path)
                    estimator=getattr(fitted_model,"estimators_",[None])[0]
                    params=estimator.get_params() if hasattr(estimator,"get_params") else {}
                    if params:
                        st.markdown("#### Fitted estimator configuration")
                        st.dataframe(pd.DataFrame([{"Parameter":k,"Value":str(v)} for k,v in params.items() if k in ("n_estimators","max_depth","min_samples_split","min_samples_leaf","max_features","criterion","random_state","n_jobs","bootstrap")]),use_container_width=True,hide_index=True)
                except (OSError,ValueError,AttributeError,EOFError):
                    st.caption("The fitted estimator configuration could not be read from the local model artifact.")
        st.caption("Training/validation progress curves and the dataset/split provenance for this legacy Random Forest result were not found in the audited artifacts.")
        if px is not None and page in ("Tuning","Model Comparison"):
            metrics=pd.DataFrame({"Threshold":["100 m","250 m","500 m","1 km"],"Queries within threshold":[saved_rf["within_100m_percent"],saved_rf["within_250m_percent"],saved_rf["within_500m_percent"],saved_rf["within_1km_percent"]]})
            fig=px.bar(metrics,x="Threshold",y="Queries within threshold",color="Threshold",title="Saved Random Forest distance thresholds",hover_data={"Queries within threshold":":.2f"})
            fig.update_layout(showlegend=False,yaxis_title="Within threshold (%)",hovermode="x unified"); style_plot(fig,320); st.plotly_chart(fig,use_container_width=True)
    if page in ("Model Training","Decision Tree","Random Forest"):
        metric_row([("Training images",provenance.get("split",{}).get("training_images","—"),"Sequence-grouped notebook split"),("Testing images",provenance.get("split",{}).get("testing_images","—"),"Held-out test split"),("Test fraction",f"{provenance.get('split',{}).get('test_fraction',0)*100:.0f}%","GroupShuffleSplit by sequence_id"),("Input features",512,"ResNet-18 embedding dimensions")])
        st.caption("The notebook contains scikit-learn experiments rather than epoch-based neural-network training. Training logs and train/validation curves were not saved.")
    if page == "Decision Tree" and dt_result:
        st.subheader("Executed notebook experiment · Decision Tree regressor")
        st.markdown(f"<div class='card'><h3>{dt_result['estimator']}</h3><p>Targets: latitude and longitude · Dataset: {provenance.get('dataset_images')} Mapillary images · GroupShuffleSplit random_state={provenance.get('split',{}).get('random_state')}</p><p>Parameters: "+" · ".join(f"{key}={value}" for key,value in dt_result["parameters"].items())+"</p></div>",unsafe_allow_html=True)
        metric_row([("Mean GPS error",f"{dt_result['gps_mean_error_m']:.2f} m","Mean Haversine distance on 168 held-out queries"),("Median GPS error",f"{dt_result['gps_median_error_m']:.2f} m","Median Haversine distance"),("GPS RMSE",f"{dt_result['gps_rmse_m']:.2f} m","Root mean square of per-query Haversine distance"),("Within 250 m",f"{dt_result['within_250m_percent']:.2f}%","Share of held-out queries within the threshold")])
        if px is not None:
            thresholds=pd.DataFrame({"Distance threshold (m)":[100,250,500,1000],"Queries within threshold (%)":[dt_result["within_100m_percent"],dt_result["within_250m_percent"],dt_result["within_500m_percent"],dt_result["within_1000m_percent"]]})
            chart=px.bar(thresholds,x="Distance threshold (m)",y="Queries within threshold (%)",title="Decision Tree GPS error thresholds",hover_data={"Queries within threshold (%)":":.2f"},color_discrete_sequence=[PLOT["accent"]])
            style_plot(chart,350); st.plotly_chart(chart,use_container_width=True)
        st.info("The notebook reports aggregate test metrics but did not save fitted Decision Tree weights or per-query coordinates. Actual-versus-predicted maps, individual error distributions, feature importance, and a tree diagram are therefore unavailable for this model. This is regression, so a confusion matrix does not apply.")
    if page == "Random Forest":
        st.subheader("Baseline and tuned Random Forest notebook results")
        rows=[r for r in [rf_baseline,rf_tuned] if r]
        if rows:
            st.dataframe(pd.DataFrame([{ "Experiment":r["name"],"Mean GPS error (m)":r["gps_mean_error_m"],"Median GPS error (m)":r["gps_median_error_m"],"GPS RMSE (m)":r["gps_rmse_m"],"Within 250 m (%)":r["within_250m_percent"]} for r in rows]),use_container_width=True,hide_index=True)
            st.markdown("#### Recorded model configurations")
            config_cols=st.columns(len(rows))
            for col,result in zip(config_cols,rows):
                with col:
                    card(result["name"],result["estimator"]+" · "+" · ".join(f"{k}={v}" for k,v in result["parameters"].items()))
            st.caption("Both results are from the notebook's 286/168 sequence-grouped split. Tuned model settings are the saved GridSearchCV best parameters.")
    if page == "Tuning":
        tuning=provenance.get("hyperparameter_search",{})
        if tuning:
            metric_row([("CV GPS error",f"{tuning['best_cv_gps_error_m']:.2f} m","Best 3-fold GroupKFold score on training data"),("CV folds",3,"Grouped by capture sequence"),("Candidate settings",tuning["candidate_combinations"],"GridSearchCV combinations"),("Total fits",tuning["total_fits"],"12 combinations × 3 folds")])
            st.subheader("Search space and selected values")
            st.dataframe(pd.DataFrame([{"Parameter":key,"Searched values":", ".join("None" if value is None else str(value) for value in values),"Selected":str(tuning["best_parameters"][key])} for key,values in tuning["parameters"].items()]),use_container_width=True,hide_index=True)
            st.caption("Per-candidate scores were not persisted, so no candidate performance curve is shown.")
    if page == "Model Comparison":
        comparison=[r for r in [dt_result,rf_baseline,rf_tuned] if r]
        retrieval=saved("evaluation.json")
        if retrieval: comparison.append({"name":"ResNet-18 nearest-reference","gps_mean_error_m":retrieval["mean_error_km"]*1000,"gps_median_error_m":retrieval["median_error_km"]*1000,"gps_rmse_m":None,"within_250m_percent":None})
        if comparison:
            comp=pd.DataFrame(comparison)
            selected=st.multiselect("Experiments",comp.name.tolist(),default=comp.name.tolist(),key="comparison_models")
            comp=comp[comp.name.isin(selected)]
            if px is not None:
                fig=px.bar(comp,x="name",y="gps_mean_error_m",color="name",title="Mean GPS error by recorded experiment",hover_data=["gps_median_error_m","gps_rmse_m"],color_discrete_sequence=[PLOT["accent"],PLOT["success"],PLOT["blue"],PLOT["error"]])
                fig.update_layout(xaxis_title="Experiment",yaxis_title="Mean Haversine error (m)",showlegend=False); style_plot(fig,390); st.plotly_chart(fig,use_container_width=True)
                st.dataframe(comp[[c for c in ["name","gps_mean_error_m","gps_median_error_m","gps_rmse_m","within_250m_percent"] if c in comp]],use_container_width=True,hide_index=True)
            st.caption("Decision Tree and Random Forest use notebook regression predictions on its 168-query group split. ResNet retrieval uses a separate held-out split and nearest-reference method; compare as distinct experiments, not as a controlled head-to-head test.")
    if page == "Feature Importance":
        model_path=Path(__file__).resolve().parent.parent/"models"/"best_random_forest.pkl"
        if model_path.exists():
            try:
                import joblib
                fitted=joblib.load(model_path)
                importances=[est.feature_importances_ for est in getattr(fitted,"estimators_",[]) if hasattr(est,"feature_importances_")]
                if importances and px is not None:
                    scores=pd.Series(pd.concat([pd.Series(v) for v in importances],axis=1).mean(axis=1)).nlargest(20).sort_values()
                    imp=pd.DataFrame({"Embedding dimension":[f"Dimension {i}" for i in scores.index],"Importance":scores.values})
                    fig=px.bar(imp,x="Importance",y="Embedding dimension",orientation="h",title="Top Random Forest embedding importances",hover_data={"Importance":":.5f"})
                    fig.update_layout(xaxis_title="Mean impurity importance",yaxis_title="ResNet embedding dimension"); style_plot(fig,560); st.plotly_chart(fig,use_container_width=True)
                    st.caption("These are embedding dimensions, not human-interpretable visual concepts.")
                else: st.info("The saved model does not expose feature importances for this estimator.")
            except (OSError,ValueError,AttributeError,EOFError): st.info("The saved model artifact could not provide readable feature importances.")
        else: st.info("No fitted Random Forest artifact is available for feature importance.")
    if not experiment and not saved_rf and not notebook_models: card("Experiment status","No model experiment artifacts are available. Training and comparisons will appear after a genuine experiment is saved.")
elif page == "Conclusion":
    heading("RESEARCH SUMMARY", "Conclusion & next steps", "A transparent retrieval prototype with region-specific evidence.")
    card("What the evidence supports","Visual retrieval can return GPS-labelled reference images and report held-out sequence distances within the available Downtown Pittsburgh corpus.")
    st.subheader("Limitations")
    st.write("The 454-image corpus represents one region. Region-label scores cannot establish performance across cities; visual similarity can select a geographically distant frame; aggregate evaluation data does not support error-distribution or paired map plots.")
    st.subheader("Next research steps")
    st.write("Expand reviewed street imagery across independent regions, retain sequence provenance, store per-query predictions, then evaluate with geographically separated test sets.")
