import cv2
import torch
import numpy as np
import faiss
import time
import os
import sys
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Configuration
VIDEO_PATH = "tik.mp4"
NUM_FRAMES = 100
BATCH_SIZE = 16
MODEL_ID = "openai/clip-vit-base-patch32"

def extract_frames(video_path, num_frames=100):
    """Extracts frames from video."""
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return None
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        print("Error: Could not read frames from video.")
        return None

    print(f"Video detected with {total_frames} total frames.")
    indices = np.linspace(0, total_frames - 1, min(num_frames, total_frames)).astype(int)
    frames = []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    
    cap.release()
    return frames

def get_embeddings(frames):
    """Converts images to CLIP vectors."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        model = CLIPModel.from_pretrained(MODEL_ID).to(device)
        processor = CLIPProcessor.from_pretrained(MODEL_ID)
    except Exception:
        # Try local if network fails
        model = CLIPModel.from_pretrained(MODEL_ID, local_files_only=True).to(device)
        processor = CLIPProcessor.from_pretrained(MODEL_ID, local_files_only=True)

    all_vectors = []
    print(f"Vectorizing {len(frames)} frames...")
    
    for i in range(0, len(frames), BATCH_SIZE):
        batch = frames[i:i+BATCH_SIZE]
        inputs = processor(images=batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            vision_outputs = model.vision_model(**inputs)
            pooled_output = vision_outputs[1]
            features = model.visual_projection(pooled_output)
            features = features / features.norm(dim=-1, keepdim=True)
            all_vectors.append(features.cpu().numpy())
    
    return np.vstack(all_vectors).astype('float32')

def benchmark_faiss(vectors):
    """Builds and benchmarks HNSW and IVF indexes."""
    dim = vectors.shape[1]
    num_vectors = vectors.shape[0]
    
    print("\n" + "="*50)
    print(f"FAISS BENCHMARK: HNSW vs IVF (Data size: {num_vectors} vectors)")
    print("="*50)

    # --- IVF SETUP ---
    nlist = int(np.sqrt(num_vectors)) if num_vectors > 10 else 5
    quantizer = faiss.IndexFlatL2(dim)
    ivf_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    
    # IVF Training & Adding
    start_time = time.time()
    ivf_index.train(vectors)
    ivf_index.add(vectors)
    ivf_build_time = (time.time() - start_time) * 1000

    # --- HNSW SETUP ---
    M = 32
    hnsw_index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
    
    # HNSW Adding (No training needed)
    start_time = time.time()
    hnsw_index.add(vectors)
    hnsw_build_time = (time.time() - start_time) * 1000

    # --- SEARCH BENCHMARK ---
    query = vectors[0:1] # Use first frame as search query
    k = 5
    
    # IVF Search
    ivf_index.nprobe = 10
    latencies = []
    for _ in range(10): # Warmup and average
        start = time.time()
        ivf_index.search(query, k)
        latencies.append((time.time() - start) * 1000)
    ivf_search_time = np.mean(latencies)

    # HNSW Search
    hnsw_index.hnsw.efSearch = 64
    latencies = []
    for _ in range(10):
        start = time.time()
        hnsw_index.search(query, k)
        latencies.append((time.time() - start) * 1000)
    hnsw_search_time = np.mean(latencies)

    # --- RESULTS TABLE ---
    print("\n{:<20} | {:<15} | {:<15}".format("Metric", "IVF Index", "HNSW Index"))
    print("-" * 55)
    print("{:<20} | {:<15.2f} | {:<15.2f}".format("Build Time (ms)", ivf_build_time, hnsw_build_time))
    print("{:<20} | {:<15.4f} | {:<15.4f}".format("Avg Search (ms)", ivf_search_time, hnsw_search_time))
    print("{:<20} | {:<15} | {:<15}".format("Training Needed", "Yes", "No"))
    print("{:<20} | {:<15} | {:<15}".format("Index Type", "Cell-based", "Graph-based"))
    
    print("\n" + "="*50)
    print("ANALYSIS:")
    if hnsw_search_time < ivf_search_time:
        print(f"-> HNSW was {(ivf_search_time/hnsw_search_time):.1f}x faster than IVF for retrieval.")
    else:
        print(f"-> IVF was {(hnsw_search_time/ivf_search_time):.1f}x faster than HNSW for retrieval.")
    print("="*50)

def main():
    # 1. Extract frames
    frames = extract_frames(VIDEO_PATH, NUM_FRAMES)
    if not frames:
        return

    # 2. Get vectors
    vectors = get_embeddings(frames)
    
    # 3. Benchmark
    benchmark_faiss(vectors)

if __name__ == "__main__":
    main()
