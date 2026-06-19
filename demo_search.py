import os
import faiss
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import sqlite3

DB_PATH = 'image_database.db'
IMAGE_DIR = 'images'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print('Loading CLIP model...')
model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device)
processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
model.eval()

# choose a query image
files = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png','.jpg','.jpeg'))]
if not files:
    raise SystemExit('No images found in images/')
query_path = files[0]
print('Query image:', query_path)

# extract feature
img = Image.open(query_path).convert('RGB')
inputs = processor(images=img, return_tensors='pt')
inputs = {k: v.to(device) for k, v in inputs.items()}
with torch.no_grad():
    feat = model.get_image_features(**inputs)
    if hasattr(feat, 'pooler_output'):
        feat = feat.pooler_output
    elif hasattr(feat, 'image_embeds'):
        feat = feat.image_embeds
    feat = feat / feat.norm(dim=-1, keepdim=True)
    vec = feat.cpu().numpy().astype('float32')

faiss.normalize_L2(vec)

# load IVF index
if not os.path.exists('ivf_index.faiss'):
    raise SystemExit('ivf_index.faiss not found')
ivf = faiss.read_index('ivf_index.faiss')

k = 5
D, I = ivf.search(vec, k)
print('Distances:', D)
print('IDs:', I)

# map ids to file paths
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
for idx in I[0]:
    if int(idx) == -1:
        print('No match for -1')
    else:
        cursor.execute('SELECT file_path FROM images WHERE id = ?', (int(idx),))
        r = cursor.fetchone()
        print(int(idx), r[0] if r else 'UNKNOWN')
conn.close()
