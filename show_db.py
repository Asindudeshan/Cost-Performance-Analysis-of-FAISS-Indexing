import sqlite3
DB_PATH = 'image_database.db'
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM images')
total = cursor.fetchone()[0]
print(f'Total rows: {total}')
print('-' * 60)
for row in cursor.execute('SELECT id, filename, file_path FROM images ORDER BY id LIMIT 50'):
    print(row)
conn.close()
