import sqlite3
from Source.Models.image_model import ImageRecord

DB_NAME = "clearscanImage.db"

class ImageDatabase:
    """
    Database access layer for ClearScan Medical Imaging.
    
    Stores full file paths to images in their original locations
    rather than copying files to a storage directory.
    """
    
    def __init__(self):
        self.conn = sqlite3.connect(DB_NAME)
        self.create_tables()

    def create_tables(self):
        """
        Creates the images table if it doesn't exist.
        
        The file_path column stores the complete absolute path
        to the image file in its original location.
        """
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
        """
        Inserts a new image record with its full file path.
        
        Args:
            image: ImageRecord object containing the full path and metadata
            
        Returns:
            ImageRecord: The same object with its database ID populated
        """
        cursor = self.conn.execute(
            "INSERT INTO images (file_path, uploaded_at, user) VALUES (?, ?, ?)",
            (image.file_path, image.uploaded_at, image.user)
        )
        self.conn.commit()
        image.id = cursor.lastrowid
        return image

    def fetch_all_images(self):
        """
        Retrieves all image records from the database.
        
        Returns:
            list[ImageRecord]: List of all stored image records with full paths
        """
        rows = self.conn.execute(
            "SELECT id, file_path, uploaded_at, user FROM images"
        ).fetchall()

        return [
            ImageRecord(id=r[0], file_path=r[1], uploaded_at=r[2], user=r[3])
            for r in rows
        ]