import os
import time
import faiss
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from db_manager import init_db, insert_images_batch, get_image_path_by_id

# --- Configuration ---
IMAGE_DIR = "images"
NUM_TEST_IMAGES = 2000
DIMENSION = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_resources():
    print(f"Using device: {DEVICE}")
    model_id = "openai/clip-vit-base-patch32"
    try:
        print("Loading CLIP Model...")
        model = CLIPModel.from_pretrained(model_id).to(DEVICE)
        processor = CLIPProcessor.from_pretrained(model_id)
    except Exception as e:
        print(f"Network error: {e}. Attempting local load...")
        model = CLIPModel.from_pretrained(model_id, local_files_only=True).to(DEVICE)
        processor = CLIPProcessor.from_pretrained(model_id, local_files_only=True)
    return model, processor

def get_vectors(model, processor):
    # Check if we have images
    all_files = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    all_files = sorted(all_files)[:NUM_TEST_IMAGES]
    
    if not all_files:
        print("No images found in images/ directory!")
        return None, None

    print(f"Processing {len(all_files)} images...")
    vectors = []
    batch_size = 32
    for i in range(0, len(all_files), batch_size):
        batch_paths = all_files[i:i+batch_size]
        batch_imgs = []
        for p in batch_paths:
            try:
                batch_imgs.append(Image.open(p).convert("RGB"))
            except:
                continue
        
        inputs = processor(images=batch_imgs, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            # Extremely robust tensor extraction
            if isinstance(outputs, torch.Tensor):
                features = outputs
            elif hasattr(outputs, "image_embeds"):
                features = outputs.image_embeds
            elif hasattr(outputs, "pooler_output"):
                features = outputs.pooler_output
            else:
                features = outputs[0] if hasattr(outputs, "__getitem__") else outputs
                
            # Normalize for cosine similarity
            features = features / features.norm(dim=-1, keepdim=True)
            vectors.append(features.cpu().numpy())
    
    return np.vstack(vectors).astype('float32'), all_files

def run_research_benchmark(vectors):
    print("\n--- Starting Research Benchmark ---")
    
    # Ground Truth (Exact Search)
    print("Computing Ground Truth (Exact Search)...")
    exact_index = faiss.IndexFlatIP(DIMENSION)
    exact_index.add(vectors)
    
    num_queries = 50
    query_vectors = vectors[:num_queries]
    k = 10
    
    start = time.time()
    gt_dist, gt_idx = exact_index.search(query_vectors, k)
    exact_time = (time.time() - start) * 1000 / num_queries
    
    # --- IVF Benchmark ---
    ivf_results = []
    nlist = 50
    quantizer = faiss.IndexFlatIP(DIMENSION)
    ivf_index = faiss.IndexIVFFlat(quantizer, DIMENSION, nlist, faiss.METRIC_INNER_PRODUCT)
    ivf_index.train(vectors)
    ivf_index.add(vectors)
    
    for nprobe in [1, 2, 5, 10, 20]:
        ivf_index.nprobe = nprobe
        start = time.time()
        dist, idx = ivf_index.search(query_vectors, k)
        latency = (time.time() - start) * 1000 / num_queries
        
        # Calculate Recall
        recall = np.mean([len(np.intersect1d(idx[i], gt_idx[i])) / k for i in range(num_queries)])
        ivf_results.append((latency, recall, nprobe))
        print(f"IVF (nprobe={nprobe}): Latency={latency:.4f}ms, Recall={recall:.2f}")

    # --- HNSW Benchmark ---
    hnsw_results = []
    M = 32
    hnsw_index = faiss.IndexHNSWFlat(DIMENSION, M, faiss.METRIC_INNER_PRODUCT)
    hnsw_index.add(vectors)
    
    for ef in [16, 32, 64, 128]:
        hnsw_index.hnsw.efSearch = ef
        start = time.time()
        dist, idx = hnsw_index.search(query_vectors, k)
        latency = (time.time() - start) * 1000 / num_queries
        
        recall = np.mean([len(np.intersect1d(idx[i], gt_idx[i])) / k for i in range(num_queries)])
        hnsw_results.append((latency, recall, ef))
        print(f"HNSW (efSearch={ef}): Latency={latency:.4f}ms, Recall={recall:.2f}")

    # --- Plotting Results ---
    plt.figure(figsize=(10, 6))
    
    ivf_lat, ivf_rec, ivf_labs = zip(*ivf_results)
    hnsw_lat, hnsw_rec, hnsw_labs = zip(*hnsw_results)
    
    plt.plot(ivf_rec, ivf_lat, 'ro-', label='IVF (Inverted File)')
    plt.plot(hnsw_rec, hnsw_lat, 'bs-', label='HNSW (Graph)')
    
    # Label points
    for i, txt in enumerate(ivf_labs):
        plt.annotate(f"nprobe={txt}", (ivf_rec[i], ivf_lat[i]), textcoords="offset points", xytext=(0,10), ha='center')
    for i, txt in enumerate(hnsw_labs):
        plt.annotate(f"ef={txt}", (hnsw_rec[i], hnsw_lat[i]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.xlabel('Recall (Accuracy)')
    plt.ylabel('Latency (ms per query)')
    plt.title('Research Comparison: Speed vs Accuracy (IVF vs HNSW)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plot_file = "research_benchmark_plot.png"
    plt.savefig(plot_file)
    print(f"\nBenchmark chart saved as: {plot_file}")
    plt.close()

def text_search_example(model, processor, vectors, image_paths):
    print("\n--- Multimodal Search Example (Text-to-Image) ---")
    text_query = "a photo of a person" # Example
    print(f"Querying for: '{text_query}'")
    
    inputs = processor(text=[text_query], return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
        # Robust tensor extraction for text
        if isinstance(outputs, torch.Tensor):
            text_features = outputs
        elif hasattr(outputs, "text_embeds"):
            text_features = outputs.text_embeds
        elif hasattr(outputs, "pooler_output"):
            text_features = outputs.pooler_output
        else:
            text_features = outputs[0] if hasattr(outputs, "__getitem__") else outputs
            
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        query_vec = text_features.cpu().numpy().astype('float32')

    # Search using HNSW (Best accuracy)
    hnsw_index = faiss.IndexHNSWFlat(DIMENSION, 32, faiss.METRIC_INNER_PRODUCT)
    hnsw_index.add(vectors)
    hnsw_index.hnsw.efSearch = 64
    
    dist, idx = hnsw_index.search(query_vec, 3)
    
    print("Top 3 Matching Images:")
    for i in range(3):
        print(f"{i+1}. Index: {idx[0][i]}, Similarity: {dist[0][i]:.4f}, Path: {image_paths[idx[0][i]]}")

def main():
    model, processor = load_resources()
    vectors, image_paths = get_vectors(model, processor)
    
    if vectors is not None:
        run_research_benchmark(vectors)
        text_search_example(model, processor, vectors, image_paths)

if __name__ == "__main__":
    main()
