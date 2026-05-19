import faiss

try:
    ivf_index = faiss.read_index("ivf_index.faiss")
    print("IVF Index loaded successfully!")
    print(f"Total vectors in IVF index: {ivf_index.ntotal}")
except Exception as e:
    print(f"Error loading IVF index: {e}")

try:
    hnsw_index = faiss.read_index("hnsw_index.faiss")
    print("HNSW Index loaded successfully!")
    print(f"Total vectors in HNSW index: {hnsw_index.ntotal}")
except Exception as e:
    print(f"Error loading HNSW index: {e}")
