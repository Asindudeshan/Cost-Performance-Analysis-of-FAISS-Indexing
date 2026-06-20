import streamlit as st
import os
import time
import cv2
import faiss
import torch
import numpy as np
import speech_recognition as sr
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from db_manager import get_image_path_by_id
import psutil
from unittest.mock import patch, MagicMock
# --- PAGE CONFIG ---
st.set_page_config(page_title="Multimodal FAISS Cost-Performance Benchmark", layout="wide")

# --- PREMIUM GLASSMORPHISM CUSTOM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* Background Gradient */
    .stApp {
        background: radial-gradient(circle at 50% 50%, #0d0d11 0%, #050508 100%);
        color: #f4f4f5;
    }
    
    /* Header Gradient styling */
    .header-title {
        background: linear-gradient(135deg, #f43f5e 0%, #8b5cf6 50%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3.2rem;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 5px;
        text-shadow: 0px 4px 30px rgba(139, 92, 246, 0.25);
    }
    
    .header-subtitle {
        text-align: center;
        color: #a1a1aa;
        font-size: 1.1rem;
        font-weight: 300;
        margin-bottom: 30px;
    }
    
    /* Premium Glassmorphic Container */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.02) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 24px !important;
        padding: 25px !important;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5) !important;
        margin-bottom: 20px !important;
    }
    
    /* Dynamic Cost Metrics Card */
    .metric-card {
        background: rgba(139, 92, 246, 0.06);
        border: 1px solid rgba(139, 92, 246, 0.15);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 8px 24px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        background: rgba(139, 92, 246, 0.1);
        border: 1px solid rgba(139, 92, 246, 0.3);
        box-shadow: 0 12px 30px rgba(139, 92, 246, 0.2);
    }
    
    .metric-title {
        font-size: 0.95rem;
        color: #a1a1aa;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 5px;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #ffffff;
    }
    
    .metric-sub {
        font-size: 0.85rem;
        color: #8b5cf6;
        margin-top: 3px;
    }
    
    /* Image Grid and Animation effects */
    [data-testid="stImage"] {
        border-radius: 12px;
        overflow: hidden;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    [data-testid="stImage"]:hover {
        transform: scale(1.04) translateY(-5px);
        box-shadow: 0 15px 30px rgba(139, 92, 246, 0.3);
        border: 1px solid rgba(139, 92, 246, 0.2);
    }
    
    /* Hide the header anchor links (ugly link icons) */
    .header-anchor, [data-testid="stHeaderActionElements"], a.header-anchor {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# --- CONFIG & PATHS ---
IMAGE_DIR = "images"
DB_PATH = "image_database.db"
IVF_INDEX_PATH = "ivf_index.faiss"
HNSW_INDEX_PATH = "hnsw_index.faiss"
DIMENSION = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- HELPER FUNCTIONS ---
@st.cache_resource
def load_resources():
    model_id = "openai/clip-vit-base-patch32"
    try:
        model = CLIPModel.from_pretrained(model_id).to(DEVICE)
        processor = CLIPProcessor.from_pretrained(model_id)
    except:
        model = CLIPModel.from_pretrained(model_id, local_files_only=True).to(DEVICE)
        processor = CLIPProcessor.from_pretrained(model_id, local_files_only=True)
    
    # Load indexes
    ivf_idx = faiss.read_index(IVF_INDEX_PATH) if os.path.exists(IVF_INDEX_PATH) else None
    hnsw_idx = faiss.read_index(HNSW_INDEX_PATH) if os.path.exists(HNSW_INDEX_PATH) else None
    
    return model, processor, ivf_idx, hnsw_idx

# Load resources
with st.spinner("Initializing CLIP Models & Index Libraries..."):
    model, processor, ivf_index, hnsw_index = load_resources()

# Check for index files
if ivf_index is None or hnsw_index is None:
    st.error("❌ FAISS Index files (`ivf_index.faiss`, `hnsw_index.faiss`) not found! Please build them first using `build_indexes.py`.")
    st.stop()

# --- INPUT PROCESSING UTILITIES ---

def robust_extract_features(outputs):
    """Robustly extracts the raw feature tensor from CLIP model outputs."""
    if isinstance(outputs, torch.Tensor):
        return outputs
    elif hasattr(outputs, "image_embeds"):
        return outputs.image_embeds
    elif hasattr(outputs, "text_embeds"):
        return outputs.text_embeds
    elif hasattr(outputs, "pooler_output"):
        return outputs.pooler_output
    elif hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state[:, 0, :]
    else:
        return outputs[0] if hasattr(outputs, "__getitem__") else outputs

def extract_video_query_vector(video_path, num_frames=5):
    """Extracts uniformly spaced keyframes from a query video and computes their average CLIP vector."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        return None
        
    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    frames_rgb = []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    
    if not frames_rgb:
        return None
        
    pil_images = [Image.fromarray(f) for f in frames_rgb]
    inputs = processor(images=pil_images, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
        features = robust_extract_features(outputs)
        features = features / features.norm(dim=-1, keepdim=True)
        # Average the embeddings of all frames to get a stable query vector
        avg_vec = torch.mean(features, dim=0, keepdim=True)
        avg_vec = avg_vec / avg_vec.norm(dim=-1, keepdim=True)
        return avg_vec.cpu().numpy().astype('float32')

def transcribe_voice(audio_file):
    """Converts uploaded WAV voice query to text using SpeechRecognition."""
    r = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = r.record(source)
    try:
        text = r.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        st.warning("⚠️ Speech Recognition could not understand the audio.")
        return None
    except sr.RequestError as e:
        st.error(f"❌ Speech Recognition API error: {e}")
        return None

def extract_text_query_vector(text_query):
    """Encodes a text query to a CLIP vector."""
    inputs = processor(text=[text_query], return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
        features = robust_extract_features(outputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype('float32')

def extract_image_query_vector(image):
    """Encodes a PIL image to a CLIP vector."""
    inputs = processor(images=[image], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
        features = robust_extract_features(outputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype('float32')

def get_ahash(image_path):
    try:
        img = Image.open(image_path).convert('L').resize((8, 8), Image.Resampling.LANCZOS)
        pixels = list(img.getdata())
        avg = sum(pixels) / 64.0
        return ''.join(['1' if p > avg else '0' for p in pixels])
    except Exception:
        return None

def filter_duplicates(ids, distances, k_val=5, threshold=8):
    unique_ids = []
    unique_dists = []
    seen_hashes = []
    
    for idx, img_id in enumerate(ids):
        img_path = get_image_path_by_id(int(img_id))
        if not img_path or not os.path.exists(img_path):
            continue
        
        ahash = get_ahash(img_path)
        if ahash is None:
            continue
            
        # Check Hamming distance against all seen hashes
        is_duplicate = False
        for seen_hash in seen_hashes:
            dist = sum(c1 != c2 for c1, c2 in zip(ahash, seen_hash))
            if dist <= threshold:
                is_duplicate = True
                break
                
        if not is_duplicate:
            unique_ids.append(img_id)
            unique_dists.append(distances[idx])
            seen_hashes.append(ahash)
            if len(unique_ids) >= k_val:
                break
                
    return np.array(unique_ids), np.array(unique_dists)


# --- MAIN DASHBOARD INTERFACE ---

st.markdown("<div class='header-title'>✦ MULTIMODAL FAISS SEARCH BENCHMARK ✦</div>", unsafe_allow_html=True)
st.markdown("<div class='header-subtitle'>Cost-Performance Comparison: HNSW Graph Index vs. Inverted File (IVF) Index</div>", unsafe_allow_html=True)

# ── Top-level tabs ───────────────────────────────────────────────────────────
tab_benchmark, tab_simulator, tab_monitor = st.tabs([
    "📊 Benchmark & Search",
    "🖥️ Virtual Resource Simulator",
    "🔴 Live System Monitor"
])

with tab_benchmark:

    # Main columns: Left sidebar for options & inputs, Right for benchmarking
    left_col, right_col = st.columns([1, 2])

    with left_col:
        with st.container(border=True):
            st.subheader("1. Setup Parameters")
            # Removed interactive slider by request. Use fixed k for retrieval.
            k_val = 5

            st.markdown("---")
            st.subheader("💡 Research Addition")
            enable_hybrid = st.checkbox("Enable Hybrid Resource-Aware Search", value=False)
            if enable_hybrid:
                st.info("🤖 **Auto-Routing Active:** System automatically balances retrieval speed and memory footprint based on real-time resource availability.")
            ram_threshold = 85.0

            st.markdown("---")
            st.subheader("2. Query Modality")
            modality = st.radio("Select Input Modality", ["📷 Image"], horizontal=True)

            query_vector = None

            # Render input forms depending on selection
            if modality == "📷 Image":
                uploaded_img = st.file_uploader("Upload query image...", type=["jpg", "png", "jpeg"])
                if uploaded_img:
                    img = Image.open(uploaded_img).convert("RGB")
                    st.image(img, caption="Query Image", width=150)
                    with st.spinner("Extracting image vector..."):
                        query_vector = extract_image_query_vector(img)

            if query_vector is not None:
                with st.expander("👁️ View Generated Vector (Raw Numbers)", expanded=False):
                    st.write(f"**Vector Shape:** `{query_vector.shape}` (e.g. 1 image, 512 dimensions)")
                    st.write("**Mathematical Representation (First 50 values):**")
                    st.code(str(query_vector[0][:50]) + " ...")

    # Search & Benchmark display in right column
    with right_col:
        if query_vector is not None:

            # ── REAL-TIME SYSTEM PERFORMANCE SNAPSHOT ────────────────────────
            with st.container(border=True):
                st.subheader("📡 Real-Time System Performance")
                st.caption("Captured at the moment of search")

                _vm   = psutil.virtual_memory()
                _swap = psutil.swap_memory()
                _cpu  = psutil.cpu_percent(interval=0.3)
                try:
                    _io1 = psutil.disk_io_counters()
                    import time as _t; _t.sleep(0.1)
                    _io2 = psutil.disk_io_counters()
                    _busy = min(((_io2.read_time + _io2.write_time) - (_io1.read_time + _io1.write_time)), 100.0)
                except Exception:
                    _busy = 0.0

                _ram_pct    = _vm.percent
                _free_mb    = _vm.available / (1024**2)
                _swap_pct   = _swap.percent

                _rt_col1, _rt_col2, _rt_col3, _rt_col4, _rt_col5 = st.columns(5)
                for _col, _label, _val, _unit, _max in [
                    (_rt_col1, "💾 RAM",      _ram_pct,  "%",   100),
                    (_rt_col2, "⚡ CPU",      _cpu,      "%",   100),
                    (_rt_col3, "🔄 Swap",     _swap_pct, "%",   100),
                    (_rt_col4, "📦 Free RAM", _free_mb,  "MB", 8192),
                    (_rt_col5, "💿 Disk I/O", _busy,     "%",   100),
                ]:
                    with _col:
                        _pct_for_bar = (_val / _max)
                        _color = "#f43f5e" if _pct_for_bar > 0.80 else ("#f59e0b" if _pct_for_bar > 0.60 else "#22c55e")
                        st.markdown(f"""
                        <div class='metric-card' style='border-top:4px solid {_color};'>
                            <div class='metric-title'>{_label}</div>
                            <div class='metric-value' style='color:{_color};font-size:1.4rem;'>
                                {"%.0f" % _val}{_unit}
                            </div>
                            <div class='metric-sub'>{"%.1f%%" % (_pct_for_bar*100)} used</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.progress(min(_pct_for_bar, 1.0))

            # ── SEARCH PANEL ─────────────────────────────────────────────────
            with st.container(border=True):

                # --- HYBRID ROUTING LOGIC ---
                run_ivf = True
                run_hnsw = True

                if enable_hybrid:
                    st.subheader("🧠 Hybrid Resource-Aware Search (Research Algorithm)")

                    # Show RAM, CPU, Swap side by side
                    _h1, _h2, _h3 = st.columns(3)
                    with _h1:
                        _h_ram_color = "#f43f5e" if _ram_pct > ram_threshold else "#22c55e"
                        st.markdown(f"""
                        <div class='metric-card' style='border-top:4px solid {_h_ram_color};'>
                            <div class='metric-title'>💾 RAM Usage</div>
                            <div class='metric-value' style='color:{_h_ram_color};font-size:1.5rem;'>{_ram_pct:.1f}%</div>
                            <div class='metric-sub'>Threshold: {ram_threshold}%</div>
                        </div>""", unsafe_allow_html=True)
                        st.progress(_ram_pct / 100.0)
                    with _h2:
                        _h_cpu_color = "#f43f5e" if _cpu > 80.0 else ("#f59e0b" if _cpu > 50.0 else "#22c55e")
                        st.markdown(f"""
                        <div class='metric-card' style='border-top:4px solid {_h_cpu_color};'>
                            <div class='metric-title'>⚡ CPU Usage</div>
                            <div class='metric-value' style='color:{_h_cpu_color};font-size:1.5rem;'>{_cpu:.1f}%</div>
                            <div class='metric-sub'>Threshold: 80%</div>
                        </div>""", unsafe_allow_html=True)
                        st.progress(_cpu / 100.0)
                    with _h3:
                        _h_swap_color = "#f43f5e" if _swap_pct > 50.0 else ("#f59e0b" if _swap_pct > 30.0 else "#22c55e")
                        st.markdown(f"""
                        <div class='metric-card' style='border-top:4px solid {_h_swap_color};'>
                            <div class='metric-title'>🔄 Swap Usage</div>
                            <div class='metric-value' style='color:{_h_swap_color};font-size:1.5rem;'>{_swap_pct:.1f}%</div>
                            <div class='metric-sub'>Threshold: 50%</div>
                        </div>""", unsafe_allow_html=True)
                        st.progress(_swap_pct / 100.0)

                    # Routing decision — always show all 3 values
                    _any_pressure = (_ram_pct > ram_threshold) or (_cpu > 80.0) or (_swap_pct > 50.0)

                    _ram_ok   = "🟢" if _ram_pct  <= ram_threshold else "🔴"
                    _cpu_ok   = "🟢" if _cpu       <= 80.0          else "🔴"
                    _swap_ok  = "🟢" if _swap_pct  <= 50.0          else "🔴"

                    _status_line = (
                        f"{_ram_ok} RAM: **{_ram_pct:.1f}%** (limit {ram_threshold}%)  |  "
                        f"{_cpu_ok} CPU: **{_cpu:.1f}%** (limit 80%)  |  "
                        f"{_swap_ok} Swap: **{_swap_pct:.1f}%** (limit 50%)"
                    )
                    st.markdown(_status_line)

                    if _any_pressure:
                        _reasons = []
                        if _ram_pct  > ram_threshold: _reasons.append(f"RAM {_ram_pct:.1f}%")
                        if _cpu      > 80.0:          _reasons.append(f"CPU {_cpu:.1f}%")
                        if _swap_pct > 50.0:          _reasons.append(f"Swap {_swap_pct:.1f}%")
                        st.warning(f"⚠️ **Pressure Detected — {' + '.join(_reasons)} exceeded limits** → Routing to **IVF Index**.")
                        run_hnsw = False
                    else:
                        st.success("✅ **RAM, CPU & Swap all healthy** → Routing to **HNSW Index** for maximum recall.")
                        run_ivf = False
                else:
                    st.subheader("📊 FAISS Cost-Performance Comparison")

                num_vectors = ivf_index.ntotal
                ivf_size = os.path.getsize(IVF_INDEX_PATH) / 1024 if os.path.exists(IVF_INDEX_PATH) else 0
                hnsw_size = os.path.getsize(HNSW_INDEX_PATH) / 1024 if os.path.exists(HNSW_INDEX_PATH) else 0

                ivf_actual = faiss.downcast_index(ivf_index.index) if hasattr(ivf_index, 'index') else ivf_index
                hnsw_actual = faiss.downcast_index(hnsw_index.index) if hasattr(hnsw_index, 'index') else hnsw_index

                if query_vector.shape[1] > ivf_index.d:
                    query_vector = np.ascontiguousarray(query_vector[:, :ivf_index.d])
                    faiss.normalize_L2(query_vector)
                elif query_vector.shape[1] < ivf_index.d:
                    st.error(f"Query vector dimension ({query_vector.shape[1]}) is smaller than index dimension ({ivf_index.d}).")
                    st.stop()

                if hasattr(hnsw_actual, 'hnsw'):
                    hnsw_actual.hnsw.efSearch = 512
                _, exact_ids = hnsw_index.search(query_vector, k_val)
                k_search = min(30, ivf_index.ntotal)

                if run_ivf:
                    if hasattr(ivf_actual, 'nprobe'):
                        ivf_actual.nprobe = 10
                    t0 = time.time()
                    raw_ivf_dist, raw_ivf_ids = ivf_index.search(query_vector, k_search)
                    ivf_time = (time.time() - t0) * 1000
                    ivf_ids_filtered, ivf_dist_filtered = filter_duplicates(raw_ivf_ids[0], raw_ivf_dist[0], k_val=k_val, threshold=8)
                    ivf_ids = [ivf_ids_filtered]
                    ivf_dist = [ivf_dist_filtered]
                    ivf_recall = len(np.intersect1d(ivf_ids[0], exact_ids[0])) / k_val * 100

                if run_hnsw:
                    if hasattr(hnsw_actual, 'hnsw'):
                        hnsw_actual.hnsw.efSearch = 64
                    t0 = time.time()
                    raw_hnsw_dist, raw_hnsw_ids = hnsw_index.search(query_vector, k_search)
                    hnsw_time = (time.time() - t0) * 1000
                    hnsw_ids_filtered, hnsw_dist_filtered = filter_duplicates(raw_hnsw_ids[0], raw_hnsw_dist[0], k_val=k_val, threshold=8)
                    hnsw_ids = [hnsw_ids_filtered]
                    hnsw_dist = [hnsw_dist_filtered]
                    hnsw_recall = len(np.intersect1d(hnsw_ids[0], exact_ids[0])) / k_val * 100

                if enable_hybrid:
                    if run_ivf:
                        st.markdown("""
                        <div class='metric-card' style='border-top: 4px solid #f43f5e;'>
                            <div class='metric-title'>IVF Inverted Index (Routed)</div>
                            <div class='metric-value'>%.4f ms</div>
                            <div class='metric-sub'>Recall: %d%% | Storage: %.1f KB</div>
                        </div>
                        """ % (ivf_time, ivf_recall, ivf_size), unsafe_allow_html=True)
                    elif run_hnsw:
                        st.markdown("""
                        <div class='metric-card' style='border-top: 4px solid #3b82f6;'>
                            <div class='metric-title'>HNSW Graph Index (Routed)</div>
                            <div class='metric-value'>%.4f ms</div>
                            <div class='metric-sub'>Recall: %d%% | Storage: %.1f KB</div>
                        </div>
                        """ % (hnsw_time, hnsw_recall, hnsw_size), unsafe_allow_html=True)
                else:
                    m_col1, m_col2 = st.columns(2)
                    with m_col1:
                        st.markdown("""
                        <div class='metric-card' style='border-top: 4px solid #f43f5e;'>
                            <div class='metric-title'>IVF Inverted Index</div>
                            <div class='metric-value'>%.4f ms</div>
                            <div class='metric-sub'>Recall: %d%% | Storage: %.1f KB</div>
                        </div>
                        """ % (ivf_time, ivf_recall, ivf_size), unsafe_allow_html=True)
                    with m_col2:
                        st.markdown("""
                        <div class='metric-card' style='border-top: 4px solid #3b82f6;'>
                            <div class='metric-title'>HNSW Graph Index</div>
                            <div class='metric-value'>%.4f ms</div>
                            <div class='metric-sub'>Recall: %d%% | Storage: %.1f KB</div>
                        </div>
                        """ % (hnsw_time, hnsw_recall, hnsw_size), unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    speedup = ... # wait, let's keep lines 526-530 unchanged, just indent them
                    speedup = ivf_time / hnsw_time if hnsw_time < ivf_time else hnsw_time / ivf_time
                    faster_index = "HNSW" if hnsw_time < ivf_time else "IVF"
                    st.info("💡 **Cost-Performance Insight:** **%s** was **%.2fx** faster for this query than the other method." % (faster_index, speedup))
                    storage_mult = hnsw_size / ivf_size if ivf_size > 0 else 0
                    st.warning("⚠️ **Storage Cost Insight:** HNSW graph index occupies **%.1fx** more storage than IVF index (due to spatial proximity links)." % storage_mult)

            st.subheader("🖼️ Retrieval Output" + (" (Routed Index)" if enable_hybrid else ": IVF vs. HNSW"))
            if not enable_hybrid:
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.markdown("<h4 style='color:#f43f5e;'>IVF Results</h4>", unsafe_allow_html=True)
                    for idx in range(len(ivf_ids[0])):
                        img_id = int(ivf_ids[0][idx])
                        dist = ivf_dist[0][idx]
                        img_path = get_image_path_by_id(img_id)
                        if img_path and os.path.exists(img_path):
                            st.image(img_path, caption=f"ID: {img_id} (Dist: {dist:.4f})", use_container_width=True)
                with res_col2:
                    st.markdown("<h4 style='color:#3b82f6;'>HNSW Results</h4>", unsafe_allow_html=True)
                    for idx in range(len(hnsw_ids[0])):
                        img_id = int(hnsw_ids[0][idx])
                        dist = hnsw_dist[0][idx]
                        img_path = get_image_path_by_id(img_id)
                        if img_path and os.path.exists(img_path):
                            st.image(img_path, caption=f"ID: {img_id} (Dist: {dist:.4f})", use_container_width=True)
            else:
                if run_ivf:
                    st.markdown("<h4 style='color:#f43f5e;'>IVF Results</h4>", unsafe_allow_html=True)
                    for idx in range(len(ivf_ids[0])):
                        img_id = int(ivf_ids[0][idx])
                        dist = ivf_dist[0][idx]
                        img_path = get_image_path_by_id(img_id)
                        if img_path and os.path.exists(img_path):
                            st.image(img_path, caption=f"ID: {img_id} (Dist: {dist:.4f})", width=250)
                elif run_hnsw:
                    st.markdown("<h4 style='color:#3b82f6;'>HNSW Results</h4>", unsafe_allow_html=True)
                    for idx in range(len(hnsw_ids[0])):
                        img_id = int(hnsw_ids[0][idx])
                        dist = hnsw_dist[0][idx]
                        img_path = get_image_path_by_id(img_id)
                        if img_path and os.path.exists(img_path):
                            st.image(img_path, caption=f"ID: {img_id} (Dist: {dist:.4f})", width=250)

        else:
            with st.container(border=True):
                st.subheader("💡 Awaiting Input Query")
                st.write("Please configure search parameters, upload a query image, and the HNSW vs. IVF cost-performance benchmark will run automatically!")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Virtual Resource Simulator
# ─────────────────────────────────────────────────────────────────────────────
with tab_simulator:
    with st.container(border=True):
        st.subheader("🖥️ Virtual Resource Simulator")
        st.write("""
        Simulate **any system-resource condition** (RAM, CPU, Swap, Free RAM, Disk I/O)
        and instantly see which FAISS index the router would choose — **no real hardware pressure needed!**
        """)

    # ── Sliders ───────────────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("⚙️ Set Simulated Resource Values")
        sim_col1, sim_col2 = st.columns(2)

        with sim_col1:
            sim_ram   = st.slider("💾 RAM Usage (%)",        0.0, 100.0, 50.0, 0.5, key="sim_ram")
            sim_swap  = st.slider("🔄 Swap Usage (%)",       0.0, 100.0, 10.0, 0.5, key="sim_swap")
            sim_disk  = st.slider("💿 Disk I/O (%)",         0.0, 100.0, 10.0, 0.5, key="sim_disk")
        with sim_col2:
            sim_cpu   = st.slider("⚡ CPU Usage (%)",        0.0, 100.0, 30.0, 0.5, key="sim_cpu")
            sim_free  = st.slider("📦 Free RAM (MB)",        0.0, 8192.0, 2048.0, 64.0, key="sim_free")

    # ── Threshold settings ────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("🎯 Threshold Configuration")
        th_col1, th_col2, th_col3 = st.columns(3)
        with th_col1:
            thr_ram  = st.number_input("RAM threshold (%)",  0.0, 100.0, 95.0, key="thr_ram")
            thr_cpu  = st.number_input("CPU threshold (%)",  0.0, 100.0, 80.0, key="thr_cpu")
        with th_col2:
            thr_swap = st.number_input("Swap threshold (%)", 0.0, 100.0, 50.0, key="thr_swap")
            thr_free = st.number_input("Min Free RAM (MB)",  0.0, 8192.0, 512.0, key="thr_free")
        with th_col3:
            thr_disk = st.number_input("Disk I/O threshold (%)", 0.0, 100.0, 80.0, key="thr_disk")

    # ── Live metric dashboard ─────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("📡 Live System Snapshot (Simulated)")
        dash_cols = st.columns(5)
        metrics = [
            ("💾 RAM",     sim_ram,  thr_ram,  "%"),
            ("⚡ CPU",     sim_cpu,  thr_cpu,  "%"),
            ("🔄 Swap",    sim_swap, thr_swap, "%"),
            ("📦 Free RAM",sim_free, thr_free, "MB"),
            ("💿 Disk I/O",sim_disk, thr_disk, "%"),
        ]
        violations = []
        for col, (label, val, thr, unit) in zip(dash_cols, metrics):
            with col:
                breached = (val >= thr) if unit == "%" else (val < thr)
                color = "#f43f5e" if breached else "#22c55e"
                badge = "⚠️" if breached else "✅"
                st.markdown(f"""
                <div class='metric-card' style='border-top: 4px solid {color};'>
                    <div class='metric-title'>{label}</div>
                    <div class='metric-value' style='color:{color};font-size:1.5rem;'>{val:.1f}{unit}</div>
                    <div class='metric-sub'>Threshold: {thr}{unit} {badge}</div>
                </div>
                """, unsafe_allow_html=True)
                if breached:
                    violations.append(f"{label}: {val:.1f}{unit} {'≥' if unit=='%' else '<'} {thr}{unit}")

    # ── Routing Decision ──────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("🤖 Router Decision")

        if violations:
            decision = "IVF"
            st.error(f"🔴 **RESOURCE PRESSURE DETECTED**")
            for v in violations:
                st.markdown(f"- `{v}`")
            st.markdown("""
            <div class='metric-card' style='border-top:4px solid #f43f5e; margin-top:15px;'>
                <div class='metric-title'>Selected Index</div>
                <div class='metric-value' style='color:#f43f5e;'>IVF Index</div>
                <div class='metric-sub'>Safe Mode — Low Memory Footprint</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            decision = "HNSW"
            st.success("🟢 **ALL METRICS HEALTHY**")
            st.markdown("""
            <div class='metric-card' style='border-top:4px solid #22c55e; margin-top:15px;'>
                <div class='metric-title'>Selected Index</div>
                <div class='metric-value' style='color:#22c55e;'>HNSW Index</div>
                <div class='metric-sub'>High Recall Mode</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Scenario Quick-Load buttons ───────────────────────────────────────────
    with st.container(border=True):
        st.subheader("⚡ Quick Scenario Tests")
        st.write("Use the buttons below to load preset scenarios — then adjust sliders to fine-tune.")
        st.markdown("""
        | Scenario | RAM | CPU | Swap | Free RAM | Disk | Expected |
        |---|---|---|---|---|---|---|
        | ✅ All Healthy | 50% | 30% | 10% | 2048 MB | 5% | **HNSW** |
        | 🔴 RAM Critical | 96% | 30% | 10% | 2048 MB | 5% | **IVF** |
        | 🔴 CPU Spike | 50% | 85% | 10% | 2048 MB | 5% | **IVF** |
        | 🔴 Swap High | 50% | 30% | 55% | 2048 MB | 5% | **IVF** |
        | 🔴 Low Free RAM | 50% | 30% | 10% | 256 MB | 5% | **IVF** |
        | 🔴 Disk Busy | 50% | 30% | 10% | 2048 MB | 85% | **IVF** |
        | 🔴 All Critical | 98% | 95% | 70% | 128 MB | 95% | **IVF** |
        | ✅ Edge / Boundary | 94.9% | 79.9% | 49.9% | 512.1 MB | 79.9% | **HNSW** |
        """)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — LIVE SYSTEM MONITOR
# Reads REAL hardware stats via psutil — works on any machine
# ═══════════════════════════════════════════════════════════════════════════
with tab_monitor:
    import platform, socket

    # ── Machine Identity ──────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("🖥️ Machine Identity")
        _mc1, _mc2, _mc3, _mc4 = st.columns(4)
        _mc1.metric("Hostname",  socket.gethostname())
        _mc2.metric("OS",        f"{platform.system()} {platform.release()}")
        _mc3.metric("CPU Cores", f"{psutil.cpu_count(logical=False)}P / {psutil.cpu_count()}L")
        _mc4.metric("Total RAM", f"{psutil.virtual_memory().total/(1024**3):.1f} GB")

    # ── Auto-Refresh control ──────────────────────────────────────────────
    with st.container(border=True):
        _rc1, _rc2, _rc3 = st.columns([2, 1, 1])
        with _rc1:
            auto_refresh = st.toggle("🔄 Auto-Refresh every 3 sec", value=False, key="mon_refresh")
        with _rc2:
            st.button("⟳ Refresh Now", key="mon_manual")
        with _rc3:
            st.caption(f"Last updated: {time.strftime('%H:%M:%S')}")

    # ── Collect live stats ────────────────────────────────────────────────
    _lvm      = psutil.virtual_memory()
    _lswap    = psutil.swap_memory()
    _lcpu_all = psutil.cpu_percent(interval=0.5, percpu=True)
    _lcpu_avg = sum(_lcpu_all) / len(_lcpu_all)
    _lfree_mb = _lvm.available / (1024**2)
    _lused_mb = _lvm.used      / (1024**2)
    _ltot_mb  = _lvm.total     / (1024**2)

    try:
        _lio1 = psutil.disk_io_counters()
        time.sleep(0.1)
        _lio2 = psutil.disk_io_counters()
        _ldisk_busy  = min(((_lio2.read_time + _lio2.write_time) - (_lio1.read_time + _lio1.write_time)), 100.0)
        _lread_mbs   = (_lio2.read_bytes  - _lio1.read_bytes)  / (1024**2) * 10
        _lwrite_mbs  = (_lio2.write_bytes - _lio1.write_bytes) / (1024**2) * 10
    except Exception:
        _ldisk_busy = _lread_mbs = _lwrite_mbs = 0.0

    try:
        _lnet1 = psutil.net_io_counters()
        time.sleep(0.1)
        _lnet2 = psutil.net_io_counters()
        _lnet_send = (_lnet2.bytes_sent - _lnet1.bytes_sent) / 1024 * 10
        _lnet_recv = (_lnet2.bytes_recv - _lnet1.bytes_recv) / 1024 * 10
    except Exception:
        _lnet_send = _lnet_recv = 0.0

    # ── 5 Main metric cards ───────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("📊 Live Resource Overview")
        _lv_cols = st.columns(5)
        for _col, (_lbl, _val, _unit, _max, _desc) in zip(_lv_cols, [
            ("💾 RAM",       _lvm.percent,  "%",   100,       "Physical memory"),
            ("⚡ CPU",       _lcpu_avg,     "%",   100,       "All-core average"),
            ("🔄 Swap",      _lswap.percent,"%",   100,       "Virtual memory"),
            ("📦 Free RAM",  _lfree_mb,     "MB",  _ltot_mb,  "Available"),
            ("💿 Disk I/O",  _ldisk_busy,   "%",   100,       "Disk busy time"),
        ]):
            with _col:
                _ratio = min(_val / _max, 1.0)
                _bar_ratio = _ratio
                if _lbl == "📦 Free RAM":
                    _ratio = 1.0 - _ratio   # invert: low free RAM = high pressure
                _color = "#f43f5e" if _ratio > 0.80 else ("#f59e0b" if _ratio > 0.50 else "#22c55e")
                st.markdown(f"""
                <div class='metric-card' style='border-top:4px solid {_color};'>
                    <div class='metric-title'>{_lbl}</div>
                    <div class='metric-value' style='color:{_color};font-size:1.35rem;'>{"%.1f" % _val}{_unit}</div>
                    <div class='metric-sub'>{_desc}</div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(_bar_ratio)

    # ── Per-Core CPU ──────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("⚡ CPU — Per Core")
        _cc_cols = st.columns(min(len(_lcpu_all), 8))
        for _i, (_ccc, _cpct) in enumerate(zip(_cc_cols, _lcpu_all[:8])):
            with _ccc:
                _cc = "#f43f5e" if _cpct > 80 else ("#f59e0b" if _cpct > 50 else "#22c55e")
                st.markdown(f"""
                <div class='metric-card' style='border-top:3px solid {_cc};padding:12px;'>
                    <div class='metric-title' style='font-size:0.75rem;'>Core {_i}</div>
                    <div class='metric-value' style='color:{_cc};font-size:1.2rem;'>{_cpct:.0f}%</div>
                </div>""", unsafe_allow_html=True)
                st.progress(_cpct / 100.0)
        if len(_lcpu_all) > 8:
            st.caption(f"... {len(_lcpu_all)-8} more cores (showing first 8)")

    # ── Memory + Disk + Network ───────────────────────────────────────────
    with st.container(border=True):
        _mem_c1, _mem_c2, _mem_c3 = st.columns(3)
        _mem_c1.metric("Used RAM",  f"{_lused_mb/1024:.2f} GB", delta=f"{_lvm.percent:.1f}% used")
        _mem_c2.metric("Free RAM",  f"{_lfree_mb/1024:.2f} GB", delta=f"{100-_lvm.percent:.1f}% free")
        _mem_c3.metric("Swap Used", f"{_lswap.used/(1024**3):.2f} GB", delta=f"{_lswap.percent:.1f}%")
        st.progress(_lvm.percent / 100.0)

        _di_c1, _di_c2, _di_c3 = st.columns(3)
        _di_c1.metric("Disk Read",    f"{_lread_mbs:.2f} MB/s")
        _di_c2.metric("Disk Write",   f"{_lwrite_mbs:.2f} MB/s")
        _di_c3.metric("Net ↑↓",       f"{_lnet_send:.1f} / {_lnet_recv:.1f} KB/s")

    # ── Live Router Decision ──────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("🤖 Live Router Decision — This Machine")
        _viol = []
        if _lvm.percent   >= 95.0: _viol.append(f"RAM {_lvm.percent:.1f}% >= 95%")
        if _lcpu_avg      >= 80.0: _viol.append(f"CPU {_lcpu_avg:.1f}% >= 80%")
        if _lswap.percent >= 50.0: _viol.append(f"Swap {_lswap.percent:.1f}% >= 50%")
        if _lfree_mb      <  512:  _viol.append(f"Free RAM {_lfree_mb:.0f} MB < 512 MB")
        if _ldisk_busy    >= 80.0: _viol.append(f"Disk I/O {_ldisk_busy:.1f}% >= 80%")

        if _viol:
            st.error("🔴 Resource Pressure on THIS machine!")
            for v in _viol: st.markdown(f"- `{v}`")
            st.markdown("""
            <div class='metric-card' style='border-top:4px solid #f43f5e;'>
                <div class='metric-title'>Router selects</div>
                <div class='metric-value' style='color:#f43f5e;'>IVF Index</div>
                <div class='metric-sub'>Safe Mode</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.success("🟢 All metrics healthy on THIS machine!")
            st.markdown("""
            <div class='metric-card' style='border-top:4px solid #22c55e;'>
                <div class='metric-title'>Router selects</div>
                <div class='metric-value' style='color:#22c55e;'>HNSW Index</div>
                <div class='metric-sub'>High Recall Mode</div>
            </div>""", unsafe_allow_html=True)

    # ── Auto-refresh trigger ──────────────────────────────────────────────
    if auto_refresh:
        time.sleep(3)
        st.rerun()
