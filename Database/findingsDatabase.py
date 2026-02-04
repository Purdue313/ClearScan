import sqlite3

DB_NAME = "clearscanFindings.db"

class FindingsDatabase:
    """
    Stores machine learning findings separately from image metadata.
    """

    def __init__(self):
        self.conn = sqlite3.connect(DB_NAME)
        self.create_tables()

    # ============================================================
    # TABLE CREATION
    # ============================================================
    def create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS findings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER NOT NULL,
                label TEXT NOT NULL,
                probability REAL NOT NULL,
                model_version TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self.conn.commit()

    # ============================================================
    # INSERT FINDINGS
    # ============================================================
    def insert_findings(self, findings):
        """
        Inserts ML findings for an image.

        Args:
            findings: list of dicts with keys:
                      image_id, label, probability, model_version
        """
        query = """
            INSERT INTO findings (image_id, label, probability, model_version)
            VALUES (?, ?, ?, ?)
        """

        self.conn.executemany(
            query,
            [
                (
                    f["image_id"],
                    f["label"],
                    f["probability"],
                    f["model_version"]
                )
                for f in findings
            ]
        )

        self.conn.commit()

    # ============================================================
    # FETCH FINDINGS
    # ============================================================
    def fetch_findings_for_image(self, image_id):
        """
        Fetches findings for an image, ordered by confidence.
        """
        rows = self.conn.execute(
            """
            SELECT label, probability, model_version, created_at
            FROM findings
            WHERE image_id = ?
            ORDER BY probability DESC
            """,
            (image_id,)
        ).fetchall()

        return rows
