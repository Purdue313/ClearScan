import sqlite3
from Source.Models.patient_model import PatientRecord

DB_NAME = "clearscanPatients.db"


class PatientDatabase:
    """
    Stores patient records and links them to image scans.

    Tables:
        patients  - core demographics
        image_patients - join table linking images to patients
    """

    def __init__(self):
        self.conn = sqlite3.connect(DB_NAME)
        self.conn.row_factory = sqlite3.Row
        self.create_tables()

    # ============================================================
    # SCHEMA
    # ============================================================
    def create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                first_name TEXT    NOT NULL,
                last_name  TEXT    NOT NULL,
                dob        TEXT    NOT NULL,
                mrn        TEXT    DEFAULT '',
                notes      TEXT    DEFAULT '',
                created_at TEXT    DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # One image → one patient.  ON CONFLICT REPLACE lets us reassign.
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS image_patients (
                image_id   INTEGER PRIMARY KEY,
                patient_id INTEGER NOT NULL,
                linked_at  TEXT    DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES patients(id)
            );
        """)

        self.conn.commit()

    # ============================================================
    # INSERT PATIENT
    # ============================================================
    def insert_patient(self, patient: PatientRecord) -> PatientRecord:
        cursor = self.conn.execute(
            """
            INSERT INTO patients (first_name, last_name, dob, mrn, notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (patient.first_name, patient.last_name, patient.dob,
             patient.mrn, patient.notes, patient.created_at)
        )
        self.conn.commit()
        patient.id = cursor.lastrowid
        return patient

    # ============================================================
    # FETCH ALL PATIENTS
    # ============================================================
    def fetch_all_patients(self) -> list[PatientRecord]:
        rows = self.conn.execute(
            "SELECT id, first_name, last_name, dob, mrn, notes, created_at "
            "FROM patients ORDER BY last_name, first_name"
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    # ============================================================
    # SEARCH PATIENTS
    # ============================================================
    def search_patients(self, query: str) -> list[PatientRecord]:
        """
        Case-insensitive search across last name, first name, and MRN.
        """
        q = f"%{query.strip()}%"
        rows = self.conn.execute(
            """
            SELECT id, first_name, last_name, dob, mrn, notes, created_at
            FROM patients
            WHERE last_name  LIKE ? COLLATE NOCASE
               OR first_name LIKE ? COLLATE NOCASE
               OR mrn        LIKE ? COLLATE NOCASE
            ORDER BY last_name, first_name
            """,
            (q, q, q)
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    # ============================================================
    # FETCH SINGLE PATIENT
    # ============================================================
    def fetch_patient(self, patient_id: int) -> PatientRecord | None:
        row = self.conn.execute(
            "SELECT id, first_name, last_name, dob, mrn, notes, created_at "
            "FROM patients WHERE id = ?",
            (patient_id,)
        ).fetchone()
        return self._row_to_record(row) if row else None

    # ============================================================
    # UPDATE PATIENT
    # ============================================================
    def update_patient(self, patient: PatientRecord):
        self.conn.execute(
            """
            UPDATE patients
            SET first_name=?, last_name=?, dob=?, mrn=?, notes=?
            WHERE id=?
            """,
            (patient.first_name, patient.last_name, patient.dob,
             patient.mrn, patient.notes, patient.id)
        )
        self.conn.commit()

    # ============================================================
    # LINK IMAGE → PATIENT
    # ============================================================
    def link_image_to_patient(self, image_id: int, patient_id: int):
        """
        Associates an image with a patient.
        If the image already has a patient, this replaces it.
        """
        self.conn.execute(
            """
            INSERT INTO image_patients (image_id, patient_id)
            VALUES (?, ?)
            ON CONFLICT(image_id) DO UPDATE SET
                patient_id = excluded.patient_id,
                linked_at  = CURRENT_TIMESTAMP
            """,
            (image_id, patient_id)
        )
        self.conn.commit()

    # ============================================================
    # GET PATIENT FOR IMAGE
    # ============================================================
    def fetch_patient_for_image(self, image_id: int) -> PatientRecord | None:
        row = self.conn.execute(
            """
            SELECT p.id, p.first_name, p.last_name, p.dob,
                   p.mrn, p.notes, p.created_at
            FROM patients p
            JOIN image_patients ip ON ip.patient_id = p.id
            WHERE ip.image_id = ?
            """,
            (image_id,)
        ).fetchone()
        return self._row_to_record(row) if row else None

    # ============================================================
    # GET ALL IMAGES FOR PATIENT
    # ============================================================
    def fetch_image_ids_for_patient(self, patient_id: int) -> list[int]:
        rows = self.conn.execute(
            "SELECT image_id FROM image_patients WHERE patient_id = ? "
            "ORDER BY linked_at DESC",
            (patient_id,)
        ).fetchall()
        return [r["image_id"] for r in rows]

    # ============================================================
    # HELPER
    # ============================================================
    @staticmethod
    def _row_to_record(row) -> PatientRecord:
        return PatientRecord(
            id         = row["id"],
            first_name = row["first_name"],
            last_name  = row["last_name"],
            dob        = row["dob"],
            mrn        = row["mrn"] or "",
            notes      = row["notes"] or "",
            created_at = row["created_at"] or "",
        )