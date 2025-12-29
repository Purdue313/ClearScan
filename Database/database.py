import sqlite3
from Source.image_model import ImageRecord

DB_NAME = "clearscan.db"

class Database:
    def __init__(self):
        self.conn = sqlite3.connect(DB_NAME)
        self.create_tables()

    def create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                uploaded_at TEXT NOT NULL,
                user TEXT NOT NULL
            );
        """)
        self.conn.commit()

    def insert_image(self, image: ImageRecord):
        cursor = self.conn.execute(
            "INSERT INTO images (file_path, uploaded_at, user) VALUES (?, ?, ?)",
            (image.file_path, image.uploaded_at, image.user)
        )
        self.conn.commit()
        image.id = cursor.lastrowid
        return image

    def fetch_all_images(self):
        rows = self.conn.execute(
            "SELECT id, file_path, uploaded_at, user FROM images"
        ).fetchall()

        return [
            ImageRecord(id=r[0], file_path=r[1], uploaded_at=r[2], user=r[3])
            for r in rows
        ]
