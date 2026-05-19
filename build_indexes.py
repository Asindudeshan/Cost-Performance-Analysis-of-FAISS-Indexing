import os
import faiss
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from db_manager import init_db, clear_db, insert_images_batch

# --- Configuration ---
NUM_IMAGES = 2000
DIMENSION = 512
IMAGE_DIR = "images"

# --- SQLite Setup ---
clear_db()

# --- PyTorch & CLIP Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print("Loading CLIP Model...")

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

class ImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            return img, img_path, True
        except Exception as e:
            # Return dummy on failure
            dummy = Image.new('RGB', (224, 224))
            return dummy, img_path, False

def build_indexes():
    print("Reading real images from folder...")
    all_files = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    all_files = sorted(all_files)[:NUM_IMAGES]
    print(f"Found {len(all_files)} images to process.")

    dataset = ImageDataset(all_files)
    # Using a custom collate_fn to just return lists since they are PIL images
    def collate_fn(batch):
        images, paths, flags = zip(*batch)
        return list(images), list(paths), list(flags)
        
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_fn)

    all_vectors = []
    all_db_ids = []

    print("Extracting vectors and saving to SQLite...")
    with torch.no_grad():
        for images, paths, valid_flags in dataloader:
            inputs = processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            features = model.get_image_features(**inputs)
            if hasattr(features, "pooler_output"):
                features = features.pooler_output
            elif hasattr(features, "image_embeds"):
                features = features.image_embeds
            elif not isinstance(features, torch.Tensor):
                features = features[0]
            features = features / features.norm(dim=-1, keepdim=True)
            features = features.cpu().numpy()

            batch_data = []
            for i in range(len(paths)):
                if valid_flags[i]:
                    batch_data.append((os.path.basename(paths[i]), paths[i]))
                    all_vectors.append(features[i])
            
            if batch_data:
                db_ids = insert_images_batch(batch_data)
                all_db_ids.extend(db_ids)

    print("Processing vectors...")
    all_vectors = np.array(all_vectors).astype('float32')
    all_db_ids = np.array(all_db_ids).astype('int64')

    if all_vectors.shape[1] > DIMENSION:
        all_vectors = np.ascontiguousarray(all_vectors[:, :DIMENSION])
    
    faiss.normalize_L2(all_vectors)

    print("Creating FAISS Indexes...")
    # IVF Index with IndexIDMap
    nlist = 100
    quantizer = faiss.IndexFlatL2(DIMENSION)
    ivf_base_index = faiss.IndexIVFFlat(quantizer, DIMENSION, nlist)
    ivf_index = faiss.IndexIDMap(ivf_base_index)
    
    print("Training IVF...")
    ivf_base_index.train(all_vectors)
    print("Adding vectors to IVF with DB IDs...")
    ivf_index.add_with_ids(all_vectors, all_db_ids)

    # HNSW Index with IndexIDMap
    M = 32
    hnsw_base_index = faiss.IndexHNSWFlat(DIMENSION, M)
    hnsw_index = faiss.IndexIDMap(hnsw_base_index)
    
    print("Adding vectors to HNSW with DB IDs...")
    hnsw_index.add_with_ids(all_vectors, all_db_ids)

    print("Saving indexes...")
    faiss.write_index(ivf_index, "ivf_index.faiss")
    faiss.write_index(hnsw_index, "hnsw_index.faiss")
    print("Indexes saved successfully! Database is ready.")

if __name__ == "__main__":
    build_indexes()
