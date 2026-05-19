import streamlit as st
import os
import time
import cv2
import faiss
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# --- PAGE CONFIG ---
st.set_page_config(page_title="FAISS Research Dashboard", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif !important; }
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgba(20, 20, 30, 1) 0%, rgba(10, 10, 15, 1) 100%);
        color: #e4e4e7;
    }
    .header-text {
        background: linear-gradient(90deg, #f43f5e, #8b5cf6, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        text-align: center;
        margin-bottom: 10px;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 25px;
        margin: 10px 0;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
        border-top: 4px solid #8b5cf6;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- CONSTANTS ---
MODEL_ID = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- HELPER FUNCTIONS ---
@st.cache_resource
def load_clip():
    try:
        model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE)
        processor = CLIPProcessor.from_pretrained(MODEL_ID)
    except:
        model = CLIPModel.from_pretrained(MODEL_ID, local_files_only=True).to(DEVICE)
        processor = CLIPProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    return model, processor

def get_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        if len(frames) % 10 == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if len(frames) >= 150: break # Limit for performance
    cap.release()
    return frames

def get_vectors(frames, model, processor):
    all_vectors = []
    pil_frames = [Image.fromarray(f) for f in frames]
    for i in range(0, len(pil_frames), 16):
        batch = pil_frames[i:i+16]
        inputs = processor(images=batch, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            if hasattr(outputs, "image_embeds"):
                features = outputs.image_embeds
            else:
                features = outputs
            features = features / features.norm(dim=-1, keepdim=True)
            all_vectors.append(features.cpu().numpy())
    return np.vstack(all_vectors).astype('float32')

# --- MAIN UI ---
st.markdown("<div class='header-text'>✦ RESEARCH PERFORMANCE DASHBOARD ✦</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #a1a1aa;'>Real-time Benchmarking: HNSW vs IVF for Video Retrieval</p>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='glass-card'><h3>1. Select Research Video</h3>", unsafe_allow_html=True)
    video_file = st.file_uploader("Upload a video file...", type=["mp4", "avi", "mov"])
    if video_file:
        with open("temp_v.mp4", "wb") as f: f.write(video_file.read())
        v_path = "temp_v.mp4"
        st.video(v_path)
    else:
        v_path = "tik.mp4" if os.path.exists("tik.mp4") else None
        if v_path:
            st.info("Using default video: tik.mp4")
            st.video(v_path)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='glass-card'><h3>2. Select Query Image</h3>", unsafe_allow_html=True)
    image_file = st.file_uploader("Upload an image to search...", type=["jpg", "png", "jpeg"])
    if image_file:
        query_img = Image.open(image_file).convert("RGB")
        st.image(query_img, width=250)
    else:
        query_img = None
    st.markdown("</div>", unsafe_allow_html=True)

if v_path and query_img:
    st.markdown("---")
    with st.spinner("🚀 Analyzing Video & Benchmarking..."):
        model, processor = load_clip()
        frames = get_video_frames(v_path)
        vectors = get_vectors(frames, model, processor)
        dim = vectors.shape[1]

        # --- IVF BENCHMARK ---
        t_ivf_b = time.time()
        nlist = int(np.sqrt(len(vectors))) if len(vectors) > 10 else 5
        quantizer = faiss.IndexFlatL2(dim)
        ivf = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        ivf.train(vectors)
        ivf.add(vectors)
        ivf_build = (time.time() - t_ivf_b) * 1000

        # --- HNSW BENCHMARK ---
        t_hnsw_b = time.time()
        hnsw = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        hnsw.add(vectors)
        hnsw_build = (time.time() - t_hnsw_b) * 1000

        # --- SEARCH ---
        inputs = processor(images=[query_img], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            q_feat = model.get_image_features(**inputs)
            q_feat = q_feat / q_feat.norm(dim=-1, keepdim=True)
            q_vec = q_feat.cpu().numpy().astype('float32')

        t_ivf_s = time.time()
        ivf.nprobe = 5
        ivf_d, ivf_i = ivf.search(q_vec, 3)
        ivf_search = (time.time() - t_ivf_s) * 1000

        t_hnsw_s = time.time()
        hnsw.hnsw.efSearch = 64
        hnsw_d, hnsw_i = hnsw.search(q_vec, 3)
        hnsw_search = (time.time() - t_hnsw_s) * 1000

    # --- DISPLAY RESULTS ---
    st.markdown("<h2 style='text-align: center;'>📊 Performance Comparison</h2>", unsafe_allow_html=True)
    
    m1, m2 = st.columns(2)
    with m1:
        st.markdown(f"<div class='metric-card'><h4>IVF Index</h4><p>Build: <b>{ivf_build:.2f}ms</b></p><p>Search: <b>{ivf_search:.4f}ms</b></p></div>", unsafe_allow_html=True)
        st.markdown("#### IVF Best Matches")
        r_cols = st.columns(3)
        for i in range(3): r_cols[i].image(frames[ivf_i[0][i]])

    with m2:
        st.markdown(f"<div class='metric-card' style='border-top-color: #3b82f6;'><h4>HNSW Index</h4><p>Build: <b>{hnsw_build:.2f}ms</b></p><p>Search: <b>{hnsw_search:.4f}ms</b></p></div>", unsafe_allow_html=True)
        st.markdown("#### HNSW Best Matches")
        r_cols = st.columns(3)
        for i in range(3): r_cols[i].image(frames[hnsw_i[0][i]])

    st.info(f"💡 Research Insight: {'HNSW' if hnsw_search < ivf_search else 'IVF'} was {(ivf_search/hnsw_search if hnsw_search < ivf_search else hnsw_search/ivf_search):.1f}x faster for this search.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #52525b;'>Advanced FAISS Benchmarking Tool</p>", unsafe_allow_html=True)
