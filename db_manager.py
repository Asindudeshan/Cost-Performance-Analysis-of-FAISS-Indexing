import sqlite3
import os

DB_PATH = 'image_database.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            file_path TEXT
        )
    ''')
    conn.commit()
    conn.close()

def clear_db():
    if os.path.exists(DB_PATH):
        try:
            os.remove(DB_PATH)
        except OSError:
            pass
    init_db()

def insert_images_batch(image_data_list):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.executemany('INSERT INTO images (filename, file_path) VALUES (?, ?)', image_data_list)
    conn.commit()
    # Since we use AUTOINCREMENT and insert in order, we can fetch the IDs
    cursor.execute('SELECT id FROM images ORDER BY id DESC LIMIT ?', (len(image_data_list),))
    ids = [row[0] for row in cursor.fetchall()]
    ids.reverse() # return them in the order they were inserted
    conn.close()
    return ids

def insert_image(filename, file_path):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO images (filename, file_path) VALUES (?, ?)', (filename, file_path))
    conn.commit()
    img_id = cursor.lastrowid
    conn.close()
    return img_id

def get_image_path_by_id(img_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT file_path FROM images WHERE id = ?', (img_id,))
    result = cursor.fetchone()
    conn.close()
    if result:
        return result[0]
    return None
