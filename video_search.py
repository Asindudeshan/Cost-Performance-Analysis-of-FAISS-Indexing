import cv2
import torch
import numpy as np
import faiss
import os
import sys
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Configuration
VIDEO_PATH = "tik.mp4"
VECTORS_CACHE = "tik_vectors.npy"
MODEL_ID = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_resources():
    try:
        model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE)
        processor = CLIPProcessor.from_pretrained(MODEL_ID)
    except:
        model = CLIPModel.from_pretrained(MODEL_ID, local_files_only=True).to(DEVICE)
        processor = CLIPProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    return model, processor

def get_video_vectors(model, processor):
    if os.path.exists(VECTORS_CACHE):
        print("Loading cached video vectors...")
        return np.load(VECTORS_CACHE)
    
    print("Extracting vectors from video (first time)...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        if len(frames) % 10 == 0: # Sample every 10th frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        if len(frames) >= 100: break
    cap.release()

    all_vectors = []
    for i in range(0, len(frames), 16):
        batch = frames[i:i+16]
        inputs = processor(images=batch, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            # Handle potential different output types
            if isinstance(outputs, torch.Tensor):
                features = outputs
            else:
                features = outputs.image_embeds
            
            features = features / features.norm(dim=-1, keepdim=True)
            all_vectors.append(features.cpu().numpy())
    
    vectors = np.vstack(all_vectors).astype('float32')
    np.save(VECTORS_CACHE, vectors)
    return vectors

def search_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found!")
        return

    model, processor = load_resources()
    vectors = get_video_vectors(model, processor)
    
    # Build HNSW Index
    index = faiss.IndexHNSWFlat(vectors.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    index.add(vectors)

    # Process Query Image
    try:
        query_img = Image.open(image_path).convert("RGB")
    except:
        print(f"Error: Could not open '{image_path}' as an image.")
        return

    inputs = processor(images=[query_img], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
        if isinstance(outputs, torch.Tensor):
            query_feat = outputs
        else:
            query_feat = outputs.image_embeds
            
        query_feat = query_feat / query_feat.norm(dim=-1, keepdim=True)
        query_vec = query_feat.cpu().numpy().astype('float32')

    # Search
    distances, indices = index.search(query_vec, 3)

    print("\n" + "="*30)
    print("SEARCH RESULTS")
    print("="*30)
    for i in range(len(indices[0])):
        print(f"Rank {i+1}: Matching Frame Index {indices[0][i]} (Score: {distances[0][i]:.4f})")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python video_search.py <image_filename>")
    else:
        search_image(sys.argv[1])
