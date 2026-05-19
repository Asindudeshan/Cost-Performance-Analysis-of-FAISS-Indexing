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
    .glass-panel {
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 24px;
        padding: 30px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5);
        margin-bottom: 20px;
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

# --- MAIN DASHBOARD INTERFACE ---

st.markdown("<div class='header-title'>✦ MULTIMODAL FAISS SEARCH BENCHMARK ✦</div>", unsafe_allow_html=True)
st.markdown("<div class='header-subtitle'>Cost-Performance Comparison: HNSW Graph Index vs. Inverted File (IVF) Index</div>", unsafe_allow_html=True)

# Main columns: Left sidebar for options & inputs, Right for benchmarking
left_col, right_col = st.columns([1, 2])

with left_col:
    st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
    st.subheader("1. Setup Parameters")
    k_val = st.slider("Results to Retrieve (k)", min_value=1, max_value=10, value=5)
    
    st.markdown("---")
    st.subheader("💡 Research Addition")
    enable_hybrid = st.toggle("Enable Hybrid Resource-Aware Search", value=False)
    if enable_hybrid:
        st.info("🤖 **Auto-Routing Active:** System automatically routes traffic if RAM > 85%")
    ram_threshold = 85.0
        
    st.markdown("---")
    st.subheader("2. Query Modality")
    modality = st.radio("Select Input Modality", ["📷 Image", "🎥 Video", "🎙️ Voice"], horizontal=True)
    
    query_vector = None
    transcription_text = None
    
    # Render input forms depending on selection
    if modality == "📷 Image":
        uploaded_img = st.file_uploader("Upload query image...", type=["jpg", "png", "jpeg"])
        if uploaded_img:
            img = Image.open(uploaded_img).convert("RGB")
            st.image(img, caption="Query Image", width=150)
            with st.spinner("Extracting image vector..."):
                query_vector = extract_image_query_vector(img)
                
    elif modality == "🎥 Video":
        uploaded_vid = st.file_uploader("Upload query video...", type=["mp4", "avi"])
        if uploaded_vid:
            with open("temp_query_v.mp4", "wb") as f:
                f.write(uploaded_vid.read())
            st.video("temp_query_v.mp4")
            with st.spinner("Processing video frames..."):
                query_vector = extract_video_query_vector("temp_query_v.mp4")
                
    elif modality == "🎙️ Voice":
        st.info("🎙️ Please upload a short audio description of the image/object in WAV format (e.g. 'a photo of a yellow flower').")
        uploaded_aud = st.file_uploader("Upload WAV audio...", type=["wav"])
        if uploaded_aud:
            with open("temp_query_a.wav", "wb") as f:
                f.write(uploaded_aud.read())
            st.audio("temp_query_a.wav")
            
            with st.spinner("Transcribing Voice to Text..."):
                transcription_text = transcribe_voice("temp_query_a.wav")
                
            if transcription_text:
                st.success(f"🗣️ Transcribed: **\"{transcription_text}\"**")
                with st.spinner("Encoding transcription..."):
                    query_vector = extract_text_query_vector(transcription_text)
                    
    if query_vector is not None:
        with st.expander("👁️ View Generated Vector (Raw Numbers)", expanded=False):
            st.write(f"**Vector Shape:** `{query_vector.shape}` (e.g. 1 image, 512 dimensions)")
            st.write("**Mathematical Representation (First 50 values):**")
            st.code(str(query_vector[0][:50]) + " ...")
                    
    st.markdown("</div>", unsafe_allow_html=True)

# Search & Benchmark display in right column
with right_col:
    if query_vector is not None:
        st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
        
        # --- HYBRID ROUTING LOGIC ---
        run_ivf = True
        run_hnsw = True
        
        if enable_hybrid:
            st.subheader("🧠 Hybrid Resource-Aware Search (Research Algorithm)")
            current_ram = psutil.virtual_memory().percent
            st.markdown(f"**Current System RAM Usage:** `{current_ram}%`")
            st.progress(current_ram / 100.0)
            
            if current_ram > ram_threshold:
                st.warning(f"⚠️ **Memory Critical (>{ram_threshold}%).** Routing query to **IVF Index** to save memory.")
                run_hnsw = False
            else:
                st.success(f"✅ **Memory Healthy (<{ram_threshold}%).** Routing query to **HNSW Index** for maximum accuracy.")
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

        # Approximation of exact search (Ground Truth) to calculate Recall
        if hasattr(hnsw_actual, 'hnsw'):
            hnsw_actual.hnsw.efSearch = 512
        _, exact_ids = hnsw_index.search(query_vector, k_val)
        
        # Execute IVF if routed
        if run_ivf:
            if hasattr(ivf_actual, 'nprobe'):
                ivf_actual.nprobe = 10
            t0 = time.time()
            ivf_dist, ivf_ids = ivf_index.search(query_vector, k_val)
            ivf_time = (time.time() - t0) * 1000
            ivf_recall = len(np.intersect1d(ivf_ids[0], exact_ids[0])) / k_val * 100
        
        # Execute HNSW if routed
        if run_hnsw:
            if hasattr(hnsw_actual, 'hnsw'):
                hnsw_actual.hnsw.efSearch = 64
            t0 = time.time()
            hnsw_dist, hnsw_ids = hnsw_index.search(query_vector, k_val)
            hnsw_time = (time.time() - t0) * 1000
            hnsw_recall = len(np.intersect1d(hnsw_ids[0], exact_ids[0])) / k_val * 100
            
        # --- DISPLAY METRICS ---
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
            speedup = ivf_time / hnsw_time if hnsw_time < ivf_time else hnsw_time / ivf_time
            faster_index = "HNSW" if hnsw_time < ivf_time else "IVF"
            st.info("💡 **Cost-Performance Insight:** **%s** was **%.2fx** faster for this query than the other method." % (faster_index, speedup))
            storage_mult = hnsw_size / ivf_size if ivf_size > 0 else 0
            st.warning("⚠️ **Storage Cost Insight:** HNSW graph index occupies **%.1fx** more storage than IVF index (due to spatial proximity links)." % storage_mult)
            
        st.markdown("</div>", unsafe_allow_html=True)
        
        # --- DISPLAY SIDE-BY-SIDE SEARCH MATCHES ---
        st.subheader("🖼️ Retrieval Output" + (" (Routed Index)" if enable_hybrid else ": IVF vs. HNSW"))
        if not enable_hybrid:
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.markdown("<h4 style='color:#f43f5e;'>IVF Results</h4>", unsafe_allow_html=True)
                for idx in range(k_val):
                    img_id = int(ivf_ids[0][idx])
                    dist = ivf_dist[0][idx]
                    img_path = get_image_path_by_id(img_id)
                    if img_path and os.path.exists(img_path):
                        st.image(img_path, caption=f"ID: {img_id} (Dist: {dist:.4f})", use_container_width=True)
            with res_col2:
                st.markdown("<h4 style='color:#3b82f6;'>HNSW Results</h4>", unsafe_allow_html=True)
                for idx in range(k_val):
                    img_id = int(hnsw_ids[0][idx])
                    dist = hnsw_dist[0][idx]
                    img_path = get_image_path_by_id(img_id)
                    if img_path and os.path.exists(img_path):
                        st.image(img_path, caption=f"ID: {img_id} (Dist: {dist:.4f})", use_container_width=True)
        else:
            if run_ivf:
                st.markdown("<h4 style='color:#f43f5e;'>IVF Results</h4>", unsafe_allow_html=True)
                for idx in range(k_val):
                    img_id = int(ivf_ids[0][idx])
                    dist = ivf_dist[0][idx]
                    img_path = get_image_path_by_id(img_id)
                    if img_path and os.path.exists(img_path):
                        st.image(img_path, caption=f"ID: {img_id} (Dist: {dist:.4f})", width=250)
            elif run_hnsw:
                st.markdown("<h4 style='color:#3b82f6;'>HNSW Results</h4>", unsafe_allow_html=True)
                for idx in range(k_val):
                    img_id = int(hnsw_ids[0][idx])
                    dist = hnsw_dist[0][idx]
                    img_path = get_image_path_by_id(img_id)
                    if img_path and os.path.exists(img_path):
                        st.image(img_path, caption=f"ID: {img_id} (Dist: {dist:.4f})", width=250)
                    
    else:
        st.markdown("<div class='glass-panel' style='text-align:center;'>", unsafe_allow_html=True)
        st.subheader("💡 Awaiting Input Query")
        st.write("Please configure search parameters, upload a query (Image, Video, or Voice WAV), and the HNSW vs. IVF cost-performance benchmark will run automatically!")
        st.markdown("</div>", unsafe_allow_html=True)
