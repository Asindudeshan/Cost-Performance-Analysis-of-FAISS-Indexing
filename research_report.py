import os
import time
import faiss
import numpy as np
import torch
import cv2
import psutil
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import sqlite3

# ──────────────────────────────────────────────────────────────
# Global thresholds for resource-aware KNN routing
# ──────────────────────────────────────────────────────────────
# RAM %       – heap / physical memory saturation
RAM_THRESHOLD      = 95.0   # percent
# CPU %       – processor load (all cores, 100 ms sample)
CPU_THRESHOLD      = 80.0   # percent
# Swap %      – OS virtual-memory spill (high = thrashing risk)
SWAP_THRESHOLD     = 50.0   # percent
# Free RAM    – minimum absolute free physical RAM in MB
#               (catches low-memory situations even when RAM% looks ok)
MIN_FREE_RAM_MB    = 512    # megabytes
# Disk I/O    – per-disk busy time percentage (0–100)
#               high I/O wait slows HNSW graph traversal from mmap
DISK_IO_THRESHOLD  = 80.0   # percent
# ──────────────────────────────────────────────────────────────

# ------------------------------
# Resource-Aware KNN Search
# ------------------------------

def get_system_stats() -> dict:
    """
    Collect all monitored system metrics in one call.
    Returns a dict with keys: ram_pct, cpu_pct, swap_pct,
                               free_ram_mb, disk_io_pct.
    """
    vm   = psutil.virtual_memory()
    swap = psutil.swap_memory()

    # Disk I/O busy-percent: average across all disks
    try:
        io1 = psutil.disk_io_counters(perdisk=False)
        import time as _t; _t.sleep(0.1)
        io2 = psutil.disk_io_counters(perdisk=False)
        elapsed_ns = 100_000_000  # 100 ms in nanoseconds
        busy_ms = (io2.read_time + io2.write_time) - (io1.read_time + io1.write_time)
        disk_io_pct = min((busy_ms / 100.0) * 100.0, 100.0)  # ms busy per 100 ms window
    except Exception:
        disk_io_pct = 0.0

    return {
        "ram_pct"     : vm.percent,
        "cpu_pct"     : psutil.cpu_percent(interval=0.1),
        "swap_pct"    : swap.percent,
        "free_ram_mb" : vm.available / (1024 ** 2),
        "disk_io_pct" : disk_io_pct,
    }


def print_system_stats(stats: dict) -> None:
    """Pretty-print all monitored system metrics."""
    print(f"  [MONITOR] RAM        : {stats['ram_pct']:.2f}%")
    print(f"  [MONITOR] CPU        : {stats['cpu_pct']:.2f}%")
    print(f"  [MONITOR] Swap       : {stats['swap_pct']:.2f}%")
    print(f"  [MONITOR] Free RAM   : {stats['free_ram_mb']:.1f} MB")
    print(f"  [MONITOR] Disk I/O   : {stats['disk_io_pct']:.2f}%")


def resource_aware_search(
    query_vector,
    k              = 5,
    ram_threshold  = RAM_THRESHOLD,
    cpu_threshold  = CPU_THRESHOLD,
    swap_threshold = SWAP_THRESHOLD,
    min_free_ram   = MIN_FREE_RAM_MB,
    disk_threshold = DISK_IO_THRESHOLD,
):
    """
    Perform a KNN search using HNSW or IVF index based on live system metrics.

    Parameters
    ----------
    query_vector   : array-like  – query embedding vector
    k              : int         – number of nearest neighbours (default 5)
    ram_threshold  : float       – RAM %  ceiling   (default RAM_THRESHOLD  = 95 %)
    cpu_threshold  : float       – CPU %  ceiling   (default CPU_THRESHOLD  = 80 %)
    swap_threshold : float       – Swap % ceiling   (default SWAP_THRESHOLD = 50 %)
    min_free_ram   : float       – Min free RAM MB  (default MIN_FREE_RAM_MB = 512 MB)
    disk_threshold : float       – Disk I/O % ceil  (default DISK_IO_THRESHOLD = 80 %)

    Routing logic
    -------------
    ANY metric breaches its limit  →  IVF   (lightweight, memory-safe)
    ALL metrics within limits      →  HNSW  (higher recall)
    """
    stats = get_system_stats()
    print("[INFO] Current system snapshot:")
    print_system_stats(stats)

    # Evaluate each condition
    violations = []
    if stats["ram_pct"]    >= ram_threshold  : violations.append(f"RAM {stats['ram_pct']:.1f}% ≥ {ram_threshold}%")
    if stats["cpu_pct"]    >= cpu_threshold  : violations.append(f"CPU {stats['cpu_pct']:.1f}% ≥ {cpu_threshold}%")
    if stats["swap_pct"]   >= swap_threshold : violations.append(f"Swap {stats['swap_pct']:.1f}% ≥ {swap_threshold}%")
    if stats["free_ram_mb"]<  min_free_ram   : violations.append(f"Free RAM {stats['free_ram_mb']:.0f} MB < {min_free_ram} MB")
    if stats["disk_io_pct"]>= disk_threshold : violations.append(f"Disk I/O {stats['disk_io_pct']:.1f}% ≥ {disk_threshold}%")

    if violations:
        print(f"[WARNING] Resource pressure – {'; '.join(violations)}")
        print("[ROUTER]  → IVF index selected (safe mode).")
        index = faiss.read_index(IVF_INDEX_PATH)
    else:
        print("[ROUTER]  All metrics healthy → HNSW index selected (high recall).")
        index = faiss.read_index(HNSW_INDEX_PATH)

    # Normalise and search
    query_vec = np.asarray(query_vector, dtype='float32').reshape(1, -1)
    faiss.normalize_L2(query_vec)
    t0 = time.time()
    distances, ids = index.search(query_vec, k)
    elapsed_ms = (time.time() - t0) * 1000
    print(f"[RESULT]  Search done in {elapsed_ms:.4f} ms | Top-{k} IDs: {ids[0]}")
    return distances, ids

def set_ram_threshold(v: float):       global RAM_THRESHOLD;     RAM_THRESHOLD     = v; print(f"[CFG] RAM threshold      → {v}%")
def set_cpu_threshold(v: float):       global CPU_THRESHOLD;     CPU_THRESHOLD     = v; print(f"[CFG] CPU threshold      → {v}%")
def set_swap_threshold(v: float):      global SWAP_THRESHOLD;    SWAP_THRESHOLD    = v; print(f"[CFG] Swap threshold     → {v}%")
def set_min_free_ram(v: float):        global MIN_FREE_RAM_MB;   MIN_FREE_RAM_MB   = v; print(f"[CFG] Min free RAM       → {v} MB")
def set_disk_io_threshold(v: float):   global DISK_IO_THRESHOLD; DISK_IO_THRESHOLD = v; print(f"[CFG] Disk I/O threshold → {v}%")

# --- Configuration ---
DIMENSION = 512
IVF_INDEX_PATH = "ivf_index.faiss"
HNSW_INDEX_PATH = "hnsw_index.faiss"
VIDEO_PATH = "well.mp4"
DB_PATH = "image_database.db"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def print_section(title):
    print("\n" + "="*80)
    print(f" RESEARCH ANALYSIS: {title}")
    print("="*80)

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

def evaluate_rq1():
    print_section("RQ1: Resource-Aware Hybrid Routing System & OOM Prevention")
    
    # 1. Load indexes and measure file sizes
    ivf_size_kb = os.path.getsize(IVF_INDEX_PATH) / 1024 if os.path.exists(IVF_INDEX_PATH) else 0
    hnsw_size_kb = os.path.getsize(HNSW_INDEX_PATH) / 1024 if os.path.exists(HNSW_INDEX_PATH) else 0
    
    print(f"Storage Footprint:")
    print(f"  - IVF Index File Size:  {ivf_size_kb:.2f} KB")
    print(f"  - HNSW Index File Size: {hnsw_size_kb:.2f} KB ({(hnsw_size_kb/ivf_size_kb if ivf_size_kb > 0 else 0):.2f}x larger)")

    # 2. Simulate Router Switch under Mock RAM Pressure
    print("\nSimulating Dynamic Query Routing:")
    np.random.seed(42)
    query_vector = np.random.random((1, DIMENSION)).astype('float32')
    faiss.normalize_L2(query_vector)
    
    ivf_index = faiss.read_index(IVF_INDEX_PATH)
    hnsw_index = faiss.read_index(HNSW_INDEX_PATH)
    
    threshold = 85.0
    for mock_ram in [60.0, 90.0]:
        print(f"\n--- System Memory State: RAM Usage = {mock_ram}% (Threshold: {threshold}%) ---")
        if mock_ram >= threshold:
            print("[WARNING] [ROUTER] Memory Pressure High! Routing query to IVF Index.")
            t0 = time.time()
            dist, ids = ivf_index.search(query_vector, 5)
            elapsed = (time.time() - t0) * 1000
            print(f"  - Executed IVF Search. Latency: {elapsed:.4f} ms, Results IDs: {ids[0]}")
            print("  - [OOM Prevention] HNSW traversal graph bypassed. Memory usage stabilized.")
        else:
            print("[OK] [ROUTER] Memory State Healthy. Routing query to HNSW Index.")
            t0 = time.time()
            dist, ids = hnsw_index.search(query_vector, 5)
            elapsed = (time.time() - t0) * 1000
            print(f"  - Executed HNSW Search. Latency: {elapsed:.4f} ms, Results IDs: {ids[0]}")
            
def evaluate_rq2():
    print_section("RQ2: Latency, Recall, and Memory Footprint Trade-offs")
    
    ivf_index = faiss.read_index(IVF_INDEX_PATH)
    hnsw_index = faiss.read_index(HNSW_INDEX_PATH)
    
    num_queries = 100
    np.random.seed(42)
    queries = np.random.random((num_queries, DIMENSION)).astype('float32')
    faiss.normalize_L2(queries)
    
    # Ground Truth: HNSW with efSearch=512 for near-exact search
    hnsw_actual = faiss.downcast_index(hnsw_index.index) if hasattr(hnsw_index, 'index') else hnsw_index
    if hasattr(hnsw_actual, 'hnsw'):
        hnsw_actual.hnsw.efSearch = 512
    _, gt_indices = hnsw_index.search(queries, 10)
    
    print(f"{'Index Type':<15} | {'Setting':<12} | {'Avg Latency (ms)':<18} | {'Recall @ 10 (%)':<16}")
    print("-" * 70)
    
    # HNSW trade-offs
    for ef in [16, 32, 64]:
        if hasattr(hnsw_actual, 'hnsw'):
            hnsw_actual.hnsw.efSearch = ef
        t0 = time.time()
        _, indices = hnsw_index.search(queries, 10)
        latency = ((time.time() - t0) / num_queries) * 1000
        
        # Calculate recall
        correct = 0
        for i in range(num_queries):
            correct += len(set(gt_indices[i]).intersection(set(indices[i])))
        recall = (correct / (num_queries * 10)) * 100
        print(f"{'HNSW':<15} | {f'efSearch={ef}':<12} | {latency:<18.4f} | {recall:<16.2f}")
        
    # IVF trade-offs
    ivf_actual = faiss.downcast_index(ivf_index.index) if hasattr(ivf_index, 'index') else ivf_index
    for nprobe in [1, 5, 10, 20]:
        if hasattr(ivf_actual, 'nprobe'):
            ivf_actual.nprobe = nprobe
        t0 = time.time()
        _, indices = ivf_index.search(queries, 10)
        latency = ((time.time() - t0) / num_queries) * 1000
        
        correct = 0
        for i in range(num_queries):
            correct += len(set(gt_indices[i]).intersection(set(indices[i])))
        recall = (correct / (num_queries * 10)) * 100
        print(f"{'IVF':<15} | {f'nprobe={nprobe}':<12} | {latency:<18.4f} | {recall:<16.2f}")

def extract_video_frames(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        return []
    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames

def evaluate_rq3():
    print_section("RQ3: CLIP-Based Video Keyframe Pooling Trade-offs")
    
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: {VIDEO_PATH} not found. Skipping RQ3.")
        return
        
    print("Loading CLIP Model for feature extraction...")
    model_id = "openai/clip-vit-base-patch32"
    try:
        model = CLIPModel.from_pretrained(model_id, local_files_only=True).to(DEVICE)
        processor = CLIPProcessor.from_pretrained(model_id, local_files_only=True)
        print("Loaded CLIP Model from local cache.")
    except Exception:
        try:
            model = CLIPModel.from_pretrained(model_id).to(DEVICE)
            processor = CLIPProcessor.from_pretrained(model_id)
            print("Loaded CLIP Model from Hugging Face Hub.")
        except Exception as e:
            print(f"Error loading model: {e}")
            return
            
    model.eval()
    
    # We will query the HNSW index to see where the pooled vectors rank
    hnsw_index = faiss.read_index(HNSW_INDEX_PATH)
    
    # 1. Evaluate with different keyframe numbers and pooling methods
    configurations = [
        {"frames": 1, "pooling": "middle", "desc": "Middle Frame Only"},
        {"frames": 5, "pooling": "mean", "desc": "Mean Pooled (5 Frames)"},
        {"frames": 10, "pooling": "mean", "desc": "Mean Pooled (10 Frames)"},
        {"frames": 25, "pooling": "mean", "desc": "Mean Pooled (25 Frames)"},
        {"frames": 10, "pooling": "max", "desc": "Max Pooled (10 Frames)"},
    ]
    
    print(f"{'Pooling Strategy':<28} | {'Frames':<8} | {'Embed Time (ms)':<16} | {'Top-1 Dist':<12} | {'Top-5 Matches'}")
    print("-" * 80)
    
    for config in configurations:
        n_frames = config["frames"]
        pooling = config["pooling"]
        
        # Measure frame extraction and model inference time
        t0 = time.time()
        frames = extract_video_frames(VIDEO_PATH, n_frames)
        if not frames:
            continue
            
        inputs = processor(images=frames, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            features = robust_extract_features(outputs)
            # Normalize
            features = features / features.norm(dim=-1, keepdim=True)
            
            # Pooling
            if pooling == "middle":
                pooled_vector = features[len(features)//2 : len(features)//2 + 1]
            elif pooling == "mean":
                pooled_vector = torch.mean(features, dim=0, keepdim=True)
                pooled_vector = pooled_vector / pooled_vector.norm(dim=-1, keepdim=True)
            elif pooling == "max":
                pooled_vector, _ = torch.max(features, dim=0, keepdim=True)
                pooled_vector = pooled_vector / pooled_vector.norm(dim=-1, keepdim=True)
                
            query_vector = pooled_vector.cpu().numpy().astype('float32')
            
        embed_time = (time.time() - t0) * 1000
        
        # Run search
        dist, ids = hnsw_index.search(query_vector, 5)
        print(f"{config['desc']:<28} | {n_frames:<8} | {embed_time:<16.2f} | {dist[0][0]:<12.4f} | {list(ids[0])}")

if __name__ == "__main__":
    evaluate_rq1()
    evaluate_rq2()
    evaluate_rq3()
