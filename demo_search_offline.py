import os
import faiss
import numpy as np
import sqlite3

IVF_INDEX_PATH = 'ivf_index.faiss'
HNSW_INDEX_PATH = 'hnsw_index.faiss'
DB_PATH = 'image_database.db'

if not os.path.exists(IVF_INDEX_PATH):
    raise SystemExit('ivf_index.faiss not found')

print('Loading IVF index...')
ivf = faiss.read_index(IVF_INDEX_PATH)
print('Index loaded. ntotal =', ivf.ntotal)

if os.path.exists(HNSW_INDEX_PATH):
    print('Loading HNSW index (fallback for reconstruction)...')
    hnsw = faiss.read_index(HNSW_INDEX_PATH)
    print('HNSW loaded. ntotal =', hnsw.ntotal)
else:
    hnsw = None

# get a sample id from DB (first row)
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute('SELECT id, file_path FROM images ORDER BY id LIMIT 1')
row = cur.fetchone()
if not row:
    raise SystemExit('No images found in DB')
query_id, query_path = row
conn.close()
print('Using DB id', query_id, 'path:', query_path)

# reconstruct vector for that id using IndexIDMap if present
try:
    vec = ivf.reconstruct(query_id)
except Exception:
    # try downcasting if wrapped in IndexIDMap
    try:
        base = ivf.index if hasattr(ivf, 'index') else ivf
        vec = ivf.reconstruct(query_id)
    except Exception as e:
        # try HNSW reconstruct as fallback
        if hnsw is not None:
            try:
                vec = hnsw.reconstruct(query_id)
            except Exception as e2:
                raise SystemExit('Could not reconstruct vector from any index: ' + str(e2))
        else:
            raise SystemExit('Could not reconstruct vector from index: ' + str(e))

vec = np.array(vec).astype('float32').reshape(1, -1)

k = 5
print('Searching top', k)
D, I = ivf.search(vec, k)
print('Distances:', D)
print('IDs:', I)

# map ids to file paths
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
for idx in I[0]:
    if int(idx) == -1:
        print('No match for -1')
    else:
        cur.execute('SELECT file_path FROM images WHERE id = ?', (int(idx),))
        r = cur.fetchone()
        print(int(idx), r[0] if r else 'UNKNOWN')
conn.close()
