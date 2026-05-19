import os
import time
import faiss
import numpy as np

def get_file_size_mb(filepath):
    if os.path.exists(filepath):
        size_bytes = os.path.getsize(filepath)
        return size_bytes / (1024 * 1024)
    return 0

def calculate_recall(ground_truth, predictions):
    # Calculates how many predicted indices match the ground truth indices
    # ground_truth and predictions should be of shape (num_queries, k)
    correct = 0
    total = ground_truth.shape[0] * ground_truth.shape[1]
    
    for i in range(ground_truth.shape[0]):
        gt_set = set(ground_truth[i])
        pred_set = set(predictions[i])
        correct += len(gt_set.intersection(pred_set))
        
    return (correct / total) * 100

def run_benchmark():
    print("="*50)
    print("FAISS BENCHMARK: HNSW vs IVF")
    print("="*50)

    # 1. Load Indexes
    ivf_path = "ivf_index.faiss"
    hnsw_path = "hnsw_index.faiss"

    try:
        ivf_index = faiss.read_index(ivf_path)
        hnsw_index = faiss.read_index(hnsw_path)
        print("Indexes loaded successfully.")
    except Exception as e:
        print(f"Error loading indexes: {e}")
        return

    dim = ivf_index.d
    num_vectors = ivf_index.ntotal
    print(f"Total indexed vectors: {num_vectors}")
    print(f"Vector dimension: {dim}\n")

    # 2. Memory Usage
    print("--- Memory Usage (File Size) ---")
    ivf_size = get_file_size_mb(ivf_path)
    hnsw_size = get_file_size_mb(hnsw_path)
    print(f"IVF Index Size : {ivf_size:.2f} MB")
    print(f"HNSW Index Size: {hnsw_size:.2f} MB\n")

    # 3. Generate Query Vectors
    num_queries = 100
    k = 10 # Number of nearest neighbors to retrieve
    np.random.seed(42)
    # Generate dummy normalized queries for benchmark purposes
    queries = np.random.random((num_queries, dim)).astype('float32')
    faiss.normalize_L2(queries)

    print(f"--- Running Benchmark (Queries: {num_queries}, K: {k}) ---\n")

    # We use HNSW with high efSearch as the "Ground Truth" for Recall calculation
    hnsw_actual = faiss.downcast_index(hnsw_index.index) if hasattr(hnsw_index, 'index') else hnsw_index
    if hasattr(hnsw_actual, 'hnsw'):
        hnsw_actual.hnsw.efSearch = 64
    
    start_time = time.time()
    _, gt_indices = hnsw_index.search(queries, k)
    gt_time = time.time() - start_time

    # 4. HNSW Parameter Tuning
    print("HNSW Performance (Trade-off: efSearch)")
    print(f"{'efSearch':<10} | {'Latency (ms/query)':<20} | {'Recall (%)':<15}")
    print("-" * 50)
    
    # Correctly access the underlying index from IndexIDMap
    hnsw_actual = faiss.downcast_index(hnsw_index.index) if hasattr(hnsw_index, 'index') else hnsw_index
    
    for ef in [16, 32, 64]:
        if hasattr(hnsw_actual, 'hnsw'):
            hnsw_actual.hnsw.efSearch = ef
        
        start_time = time.time()
        _, indices = hnsw_index.search(queries, k)
        elapsed_time = time.time() - start_time
        
        latency_ms = (elapsed_time / num_queries) * 1000
        recall = calculate_recall(gt_indices, indices)
        print(f"{ef:<10} | {latency_ms:<20.4f} | {recall:<15.2f}")
    
    print("\n")

    # 5. IVF Parameter Tuning
    print("IVF Performance (Trade-off: nprobe)")
    print(f"{'nprobe':<10} | {'Latency (ms/query)':<20} | {'Recall (%)':<15}")
    print("-" * 50)
    
    ivf_actual = faiss.downcast_index(ivf_index.index) if hasattr(ivf_index, 'index') else ivf_index
    
    for nprobe in [1, 5, 10, 20]:
        try:
            if hasattr(ivf_actual, 'nprobe'):
                ivf_actual.nprobe = nprobe
            
            start_time = time.time()
            _, indices = ivf_index.search(queries, k)
            elapsed_time = time.time() - start_time
            
            latency_ms = (elapsed_time / num_queries) * 1000
            recall = calculate_recall(gt_indices, indices)
            print(f"{nprobe:<10} | {latency_ms:<20.4f} | {recall:<15.2f}")
        except Exception:
            pass
            
    print("\nBenchmark Complete.")

if __name__ == "__main__":
    run_benchmark()
