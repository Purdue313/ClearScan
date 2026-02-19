import sqlite3

DB_NAME = "clearscanFindings.db"

class FindingsDatabase:
    """
    Stores machine learning findings separately from image metadata.
    Extended to support diagnosis confirmation/rejection and ML feedback.
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

        # Diagnosis table: one row per image, stores the doctor's final call
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS diagnoses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER NOT NULL UNIQUE,
                confirmed_label TEXT NOT NULL,
                rejected_labels TEXT,          -- JSON list of labels the doctor rejected
                doctor_notes TEXT,
                feedback_submitted INTEGER DEFAULT 0,
                diagnosed_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Feedback table: each row is one label verdict from the doctor
        # used as training signal
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ml_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER NOT NULL,
                label TEXT NOT NULL,
                correct INTEGER NOT NULL,      -- 1 = correct, 0 = incorrect
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
        Returns: list of (label, probability, model_version, created_at)
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

    # ============================================================
    # SAVE DIAGNOSIS
    # ============================================================
    def save_diagnosis(self, image_id, confirmed_label, rejected_labels, doctor_notes):
        """
        Saves the doctor's final diagnosis for an image.
        Uses INSERT OR REPLACE so re-diagnosing overwrites the old record.

        Args:
            image_id        (int)
            confirmed_label (str)  : the label the doctor confirmed as correct
            rejected_labels (list) : labels the doctor explicitly marked wrong
            doctor_notes    (str)  : free-text notes
        """
        import json
        self.conn.execute(
            """
            INSERT INTO diagnoses (image_id, confirmed_label, rejected_labels, doctor_notes)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(image_id) DO UPDATE SET
                confirmed_label  = excluded.confirmed_label,
                rejected_labels  = excluded.rejected_labels,
                doctor_notes     = excluded.doctor_notes,
                feedback_submitted = 0,
                diagnosed_at     = CURRENT_TIMESTAMP
            """,
            (image_id, confirmed_label, json.dumps(rejected_labels), doctor_notes)
        )
        self.conn.commit()

    # ============================================================
    # FETCH DIAGNOSIS
    # ============================================================
    def fetch_diagnosis(self, image_id):
        """
        Returns the saved diagnosis for an image, or None if not yet diagnosed.
        Returns: dict or None
        """
        import json
        row = self.conn.execute(
            """
            SELECT confirmed_label, rejected_labels, doctor_notes,
                   feedback_submitted, diagnosed_at
            FROM diagnoses
            WHERE image_id = ?
            """,
            (image_id,)
        ).fetchone()

        if not row:
            return None

        return {
            "confirmed_label":   row[0],
            "rejected_labels":   json.loads(row[1]) if row[1] else [],
            "doctor_notes":      row[2],
            "feedback_submitted": bool(row[3]),
            "diagnosed_at":      row[4],
        }

    # ============================================================
    # MARK IMAGE AS DIAGNOSED
    # ============================================================
    def is_diagnosed(self, image_id):
        """Returns True if a diagnosis record exists for this image."""
        row = self.conn.execute(
            "SELECT 1 FROM diagnoses WHERE image_id = ?", (image_id,)
        ).fetchone()
        return row is not None

    # ============================================================
    # SUBMIT ML FEEDBACK
    # ============================================================
    def submit_feedback(self, image_id, confirmed_label, all_labels, model_version):
        """
        Writes one feedback row per label (correct / incorrect).
        Marks the diagnosis record as feedback_submitted = 1.

        Args:
            image_id        (int)
            confirmed_label (str)  : the label the doctor confirmed
            all_labels      (list) : every label the model produced for this image
            model_version   (str)
        """
        rows = [
            (image_id, label, 1 if label == confirmed_label else 0, model_version)
            for label in all_labels
        ]
        self.conn.executemany(
            """
            INSERT INTO ml_feedback (image_id, label, correct, model_version)
            VALUES (?, ?, ?, ?)
            """,
            rows
        )
        self.conn.execute(
            "UPDATE diagnoses SET feedback_submitted = 1 WHERE image_id = ?",
            (image_id,)
        )
        self.conn.commit()

    # ============================================================
    # FETCH FEEDBACK FOR TRAINING
    # ============================================================
    def fetch_unprocessed_feedback(self):
        """
        Returns all feedback rows not yet used for training.
        Caller is responsible for marking them processed after use.
        Returns: list of (image_id, label, correct, model_version, created_at)
        """
        return self.conn.execute(
            """
            SELECT id, image_id, label, correct, model_version, created_at
            FROM ml_feedback
            ORDER BY created_at ASC
            """
        ).fetchall()

    def fetch_feedback_stats(self):
        """
        Returns per-label accuracy stats across all submitted feedback.
        Useful for evaluating model drift.
        Returns: list of (label, total, correct_count, accuracy)
        """
        return self.conn.execute(
            """
            SELECT
                label,
                COUNT(*)                        AS total,
                SUM(correct)                    AS correct_count,
                ROUND(AVG(correct) * 100, 1)    AS accuracy_pct
            FROM ml_feedback
            GROUP BY label
            ORDER BY accuracy_pct ASC
            """
        ).fetchall()