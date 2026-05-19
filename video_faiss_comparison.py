import cv2
import torch
import numpy as np
import faiss
import time
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os

def extract_video_frames(video_path, num_frames=50):
    """Extracts exactly num_frames from the video uniformly."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        print("Error: Could not open video or video is empty.")
        return []

    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    frames = []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR (OpenCV) to RGB (PIL)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    
    cap.release()
    return frames

def main():
    video_path = "well.mp4"
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found.")
        return

    print(f"Processing video: {video_path}...")
    frames = extract_video_frames(video_path, num_frames=100)
    print(f"Extracted {len(frames)} frames.")

    # 1. Load CLIP Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model_id = "openai/clip-vit-base-patch32"
    
    try:
        print("Loading CLIP Model...")
        model = CLIPModel.from_pretrained(model_id).to(device)
        processor = CLIPProcessor.from_pretrained(model_id)
    except Exception as e:
        print(f"Network error detected: {e}")
        print("Attempting to load from local cache...")
        try:
            model = CLIPModel.from_pretrained(model_id, local_files_only=True).to(device)
            processor = CLIPProcessor.from_pretrained(model_id, local_files_only=True)
            print("Successfully loaded from local cache.")
        except Exception as local_e:
            print(f"Critical Error: Could not load model even from local cache. {local_e}")
            return

    # 2. Extract Vectors
    print("Extracting vector embeddings for frames...")
    all_vectors = []
    batch_size = 10
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i+batch_size]
        inputs = processor(images=batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            # Using the vision model directly and then projecting
            vision_outputs = model.vision_model(**inputs)
            pooled_output = vision_outputs[1]  # pooled_output
            features = model.visual_projection(pooled_output)
            # Normalize
            features = features / features.norm(dim=-1, keepdim=True)
            all_vectors.append(features.cpu().numpy())
    
    vectors = np.vstack(all_vectors).astype('float32')
    dim = vectors.shape[1]
    print(f"Vector collection complete. Shape: {vectors.shape}")

    # 3. Setup FAISS Indexes
    # IVF
    nlist = 5 # Small nlist because we only have 100 vectors
    quantizer = faiss.IndexFlatL2(dim)
    ivf_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    
    # HNSW
    M = 32
    hnsw_index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)

    # 4. Train and Add
    print("Training and building indexes...")
    ivf_index.train(vectors)
    ivf_index.add(vectors)
    hnsw_index.add(vectors)

    # 5. Benchmark Search
    # We'll use one of the frames as a query
    query_vector = vectors[0:1] # Use first frame as query
    k = 5

    print("\n--- PERFORMANCE COMPARISON ---")
    
    # IVF Search
    ivf_index.nprobe = 2
    start = time.time()
    ivf_dist, ivf_idx = ivf_index.search(query_vector, k)
    ivf_time = (time.time() - start) * 1000
    print(f"IVF Search Time: {ivf_time:.4f} ms")
    print(f"IVF Top Indices: {ivf_idx[0]}")

    # HNSW Search
    hnsw_index.hnsw.efSearch = 32
    start = time.time()
    hnsw_dist, hnsw_idx = hnsw_index.search(query_vector, k)
    hnsw_time = (time.time() - start) * 1000
    print(f"HNSW Search Time: {hnsw_time:.4f} ms")
    print(f"HNSW Top Indices: {hnsw_idx[0]}")

    # 6. Summary of Differences
    print("\n--- Key Differences ---")
    print(f"1. Memory: HNSW usually takes more memory because of the graph structure.")
    print(f"2. Speed: For large datasets, HNSW is often faster for low-latency retrieval.")
    print(f"3. Flexibility: IVF requires training (clustering), HNSW does not.")
    print(f"4. Recall: Both are approximate, but HNSW often provides better recall-speed trade-offs.")

if __name__ == "__main__":
    main()
