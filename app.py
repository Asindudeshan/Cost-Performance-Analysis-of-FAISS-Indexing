import streamlit as st
import os
import time
import faiss
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from db_manager import get_image_path_by_id

# Setup Page
st.set_page_config(page_title="FAISS Image Search", layout="wide")

# Inject Premium Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    /* Global Typography */
    html, body, [class*="css"]  {
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* Main Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #09090b 0%, #18181b 50%, #27272a 100%);
        color: #f4f4f5;
    }
    
    /* Header styling with Gradient */
    h1 {
        background: -webkit-linear-gradient(45deg, #f43f5e, #8b5cf6, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        font-size: 3.5rem !important;
        text-align: center;
        padding-bottom: 20px;
        text-shadow: 0px 4px 20px rgba(139, 92, 246, 0.2);
    }
    
    /* Subheaders */
    h2, h3 {
        font-weight: 600 !important;
        color: #e4e4e7 !important;
        letter-spacing: 0.5px;
    }
    
    /* Sidebar Glassmorphism */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.02) !important;
        backdrop-filter: blur(12px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Image hover animations */
    [data-testid="stImage"] {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 10px 25px rgba(0,0,0,0.4);
        transition: transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.4s ease;
        border: 1px solid rgba(255,255,255,0.05);
    }
    
    [data-testid="stImage"]:hover {
        transform: translateY(-8px) scale(1.03);
        box-shadow: 0 20px 35px rgba(139, 92, 246, 0.4);
        border: 1px solid rgba(139, 92, 246, 0.3);
        z-index: 10;
    }
    
    /* Info boxes (Time metrics) - Glassy violet */
    div[data-testid="stInfo"] {
        background: rgba(139, 92, 246, 0.1) !important;
        border: 1px solid rgba(139, 92, 246, 0.3) !important;
        border-radius: 12px !important;
        color: #f4f4f5 !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    /* File uploader */
    [data-testid="stFileUploadDropzone"] {
        background: rgba(255, 255, 255, 0.03);
        border: 2px dashed rgba(139, 92, 246, 0.6);
        border-radius: 20px;
        transition: all 0.3s ease;
        padding: 30px;
    }
    
    [data-testid="stFileUploadDropzone"]:hover {
        background: rgba(139, 92, 246, 0.1);
        border-color: #f43f5e;
        transform: scale(1.01);
    }
    
    /* Sliders Customization */
    .stSlider > div[data-baseweb="slider"] > div > div > div {
        background: linear-gradient(90deg, #f43f5e, #8b5cf6, #3b82f6) !important;
    }
    .stSlider > div[data-baseweb="slider"] > div > div > div[role="slider"] {
        background-color: #ffffff !important;
        border: 3px solid #8b5cf6 !important;
        box-shadow: 0 0 10px rgba(139, 92, 246, 0.5);
    }
    
    hr {
        border-color: rgba(255,255,255,0.08);
        margin: 40px 0;
    }
    
    /* Caption styling */
    .caption {
        font-family: 'Outfit', sans-serif;
        text-align: center;
        padding: 8px;
        background: rgba(0,0,0,0.5);
        border-radius: 8px;
        margin-top: 5px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("✦ Visual Image Search ✦")
st.markdown("<h3 style='text-align: center; color: #a1a1aa; font-weight: 300; margin-bottom: 40px;'>High-Performance Vector Retrieval with HNSW & IVF</h3>", unsafe_allow_html=True)

# Paths
IMAGE_DIR = "images"

@st.cache_resource
def load_indexes():
    ivf_index = faiss.read_index("ivf_index.faiss")
    hnsw_index = faiss.read_index("hnsw_index.faiss")
    return ivf_index, hnsw_index

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor, device

# Load Resources
with st.spinner("Loading AI Models and FAISS Indexes..."):
    try:
        ivf_index, hnsw_index = load_indexes()
        model, processor, device = load_model()
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        st.stop()

st.sidebar.header("Settings")
k_neighbors = st.sidebar.slider("Number of Results (k)", min_value=1, max_value=20, value=5)
ef_search = st.sidebar.slider("HNSW efSearch", min_value=10, max_value=100, value=32)
nprobe = st.sidebar.slider("IVF nprobe", min_value=1, max_value=50, value=10)

# Apply FAISS settings
try:
    # Use downcast_index to get the actual implementation (HNSW, IVF, etc.)
    # even if wrapped in IndexIDMap
    hnsw_actual = faiss.downcast_index(hnsw_index.index) if hasattr(hnsw_index, 'index') else hnsw_index
    ivf_actual = faiss.downcast_index(ivf_index.index) if hasattr(ivf_index, 'index') else ivf_index
    
    if hasattr(hnsw_actual, 'hnsw'):
        hnsw_actual.hnsw.efSearch = ef_search
    
    if hasattr(ivf_actual, 'nprobe'):
        ivf_actual.nprobe = nprobe
except Exception as e:
    st.warning(f"Note: Could not set search parameters: {e}")

# Search Options
search_mode = st.radio("Select Search Mode:", ["Text Search", "Image Search"], horizontal=True)

query_features = None

if search_mode == "Text Search":
    text_query = st.text_input("Describe the image you want to find (e.g. 'a red car'):")
    if text_query:
        with st.spinner("Embedding text query..."):
            inputs = processor(text=text_query, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                features = model.get_text_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)
                query_features = features.cpu().numpy().astype('float32')

elif search_mode == "Image Search":
    uploaded_file = st.file_uploader("Upload an Image to Search...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.subheader("Uploaded Query Image")
        query_img = Image.open(uploaded_file).convert('RGB')
        st.image(query_img, width=200)
        
        with st.spinner("Extracting features..."):
            inputs = processor(images=query_img, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                features = model.get_image_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)
                query_features = features.cpu().numpy().astype('float32')

if query_features is not None:
    faiss.normalize_L2(query_features)
        
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    # IVF Search
    with col1:
        st.subheader("IVF Index Results")
        start_time = time.time()
        ivf_dist, ivf_idx = ivf_index.search(query_features, k_neighbors)
        ivf_time = (time.time() - start_time) * 1000
        st.info(f"⏱️ Search Time: **{ivf_time:.4f} ms**")
        
        cols = st.columns(k_neighbors)
        for i in range(k_neighbors):
            img_id = int(ivf_idx[0][i])
            img_path = get_image_path_by_id(img_id)
            if img_path and os.path.exists(img_path):
                cols[i].image(img_path, caption=f"DB_ID: {img_id}\nDist: {ivf_dist[0][i]:.2f}", use_container_width=True)
            else:
                cols[i].write("Image not found")
                
    # HNSW Search
    with col2:
        st.subheader("HNSW Index Results")
        start_time = time.time()
        hnsw_dist, hnsw_idx = hnsw_index.search(query_features, k_neighbors)
        hnsw_time = (time.time() - start_time) * 1000
        st.info(f"⏱️ Search Time: **{hnsw_time:.4f} ms**")
        
        cols = st.columns(k_neighbors)
        for i in range(k_neighbors):
            img_id = int(hnsw_idx[0][i])
            img_path = get_image_path_by_id(img_id)
            if img_path and os.path.exists(img_path):
                cols[i].image(img_path, caption=f"DB_ID: {img_id}\nDist: {hnsw_dist[0][i]:.2f}", use_container_width=True)
            else:
                cols[i].write("Image not found")
