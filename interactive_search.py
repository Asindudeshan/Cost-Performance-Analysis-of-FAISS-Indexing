import os
import torch
import faiss
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# --- Configuration ---
IMAGE_DIR = "images"
DIMENSION = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_resources():
    print(f"Loading AI Models on {DEVICE}... Please wait.")
    model_id = "openai/clip-vit-base-patch32"
    try:
        model = CLIPModel.from_pretrained(model_id).to(DEVICE)
        processor = CLIPProcessor.from_pretrained(model_id)
    except:
        model = CLIPModel.from_pretrained(model_id, local_files_only=True).to(DEVICE)
        processor = CLIPProcessor.from_pretrained(model_id, local_files_only=True)
    return model, processor

def get_all_vectors(model, processor):
    all_files = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    all_files = sorted(all_files)
    
    # In a real app, we would load pre-computed vectors. 
    # For this demo, let's limit to 1000 for speed if not cached.
    print(f"Indexing {len(all_files)} images...")
    vectors = []
    batch_size = 32
    for i in range(0, min(1000, len(all_files)), batch_size):
        batch_paths = all_files[i:i+batch_size]
        batch_imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=batch_imgs, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            # Robust tensor extraction
            if isinstance(outputs, torch.Tensor):
                features = outputs
            elif hasattr(outputs, "image_embeds"):
                features = outputs.image_embeds
            elif hasattr(outputs, "pooler_output"):
                features = outputs.pooler_output
            else:
                features = outputs[0] if hasattr(outputs, "__getitem__") else outputs
                
            features = features / features.norm(dim=-1, keepdim=True)
            vectors.append(features.cpu().numpy())
    
    return np.vstack(vectors).astype('float32'), all_files

def main():
    model, processor = load_resources()
    vectors, image_paths = get_all_vectors(model, processor)
    
    # Build HNSW Index
    index = faiss.IndexHNSWFlat(DIMENSION, 32, faiss.METRIC_INNER_PRODUCT)
    index.add(vectors)
    index.hnsw.efSearch = 64
    
    print("\n" + "="*40)
    print("INTERACTIVE MULTIMODAL SEARCH READY")
    print("="*40)
    print("Type 'exit' to stop.")

    while True:
        query = input("\nEnter search description (e.g., 'a blue car'): ")
        if query.lower() == 'exit':
            break
        
        if not query:
            continue

        # Convert text to vector
        inputs = processor(text=[query], return_tensors="pt", padding=True).to(DEVICE)
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

        # Search
        distances, indices = index.search(query_vec, 5)
        
        print(f"\nTop 5 Results for '{query}':")
        for i in range(5):
            idx = indices[0][i]
            dist = distances[0][i]
            print(f"{i+1}. Score: {dist:.4f} | Path: {image_paths[idx]}")

if __name__ == "__main__":
    main()
