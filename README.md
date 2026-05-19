# Cost-Performance Analysis of FAISS Indexing (HNSW vs IVF) & Resource-Aware Hybrid Search

An academic research framework and benchmark dashboard designed to evaluate the trade-off between search latency and memory footprint in FAISS vector indexes (HNSW vs. IVF) for multi-modal queries, featuring a novel **Resource-Aware Hybrid Routing Algorithm**.

## 📖 Research Overview
In large-scale multi-modal vector search, approximate nearest neighbor (ANN) indexes present distinct cost-performance characteristics:
- **HNSW (Hierarchical Navigable Small World):** Offers ultra-low query latency and near-perfect recall, but suffers from an extremely high in-memory (RAM) footprint.
- **IVF (Inverted File Index):** Provides an exceptionally low memory footprint, but exhibits a higher query latency and slight recall degradation depending on clustering configurations.

### 🧠 The Solution: Resource-Aware Hybrid Search
This project introduces a dynamic **Resource-Aware Hybrid Search Algorithm**. It continuously monitors system memory utilization (`psutil`) and routes queries dynamically:
- **RAM < 85% (Healthy Memory State):** Routes queries to **HNSW** for maximum accuracy and speed.
- **RAM ≥ 85% (Critical Memory State):** Dynamically switches query routing to **IVF** to prevent OS-level swapping (thrashing) and Out-Of-Memory (OOM) crashes, ensuring high system availability and stability under heavy resource constraints.

---

## 🛠️ Architecture & System Flow

```
                      +-----------------------------+
                      |     Multi-Modal Query       |
                      |  (Image, Video, Audio/Text) |
                      +--------------+--------------+
                                     |
                                     v
                       +-------------+-------------+
                       |   CLIP Feature Extraction |
                       +-------------+-------------+
                                     |
                                     v
                  +------------------+------------------+
                  |  Resource-Aware Dynamic Router      |
                  |  Live RAM Check (Threshold: 85%)    |
                  +--------+-------------------+--------+
                           |                   |
               (RAM < 85%) |                   | (RAM >= 85%)
                           v                   v
                     +-----+-----+       +-----+-----+
                     | HNSW Search |       | IVF Search|
                     +-----+-----+       +-----+-----+
                           |                   |
                           +---------+---------+
                                     | (Vector ID matched)
                                     v
                        +------------+------------+
                        |   SQLite Metadata DB    |
                        |   (Fetch Image Path)    |
                        +------------+------------+
                                     |
                                     v
                        +------------+------------+
                        |  Streamlit UI Rendering |
                        +-------------------------+
```

---

## 🚀 Getting Started & Installation

### 1. Prerequisites
Ensure you have Python 3.9+ installed.

### 2. Clone and Setup
```bash
# Clone the repository
git clone https://github.com/Asindudeshan/Cost-Performance-Analysis-of-FAISS-Indexing.git
cd Cost-Performance-Analysis-of-FAISS-Indexing

# Install dependencies
pip install -r requirements.txt
```

### 3. Build Vector Indexes
To process the local image dataset, extract CLIP feature embeddings, populate the SQLite database, and generate the HNSW/IVF indexes:
```bash
python build_indexes.py
```

### 4. Launch the Research Dashboard
Run the Streamlit application to benchmark the indexes and test the hybrid routing algorithm:
```bash
python -m streamlit run multimodal_benchmark.py
```

---

## 📊 Benchmark Metrics
The dashboard provides a real-time side-by-side comparison of:
- **Search Latency (ms):** Exact duration of vector similarity queries.
- **Storage Cost (KB):** In-memory size of `.faiss` index files.
- **Visual Accuracy:** Retrieval precision across diverse datasets.
