import sqlite3
import calendar as cal_mod
from datetime import datetime, date, time, timedelta
from Source.Models.appointment_model import (
    AppointmentRecord, ALL_STATUSES,
    CLINIC_OPEN, CLINIC_CLOSE, SLOT_INTERVAL, DEFAULT_DURATION
)

DB_NAME = "clearscanSchedule.db"


class ScheduleDatabase:
    """
    Appointments storage with per-tech double-booking prevention.

    A time slot is only blocked for a specific tech.
    Multiple techs can have simultaneous appointments at the same time.
    A slot is free for a given tech if that tech has no overlapping booking.
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
            CREATE TABLE IF NOT EXISTS appointments (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id   INTEGER NOT NULL,
                appt_date    TEXT    NOT NULL,
                appt_time    TEXT    NOT NULL,
                appt_type    TEXT    NOT NULL,
                tech_user_id INTEGER,
                duration_min INTEGER NOT NULL DEFAULT 30,
                status       TEXT    NOT NULL DEFAULT 'Scheduled',
                notes        TEXT    DEFAULT '',
                created_by   TEXT    DEFAULT '',
                created_at   TEXT    DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_appt_date
            ON appointments(appt_date);
        """)
        self.conn.commit()
        self._migrate()

    def _migrate(self):
        """Add any missing columns to existing databases."""
        existing = {
            row[1] for row in
            self.conn.execute("PRAGMA table_info(appointments)").fetchall()
        }
        migrations = [
            ("tech_user_id", "INTEGER"),
        ]
        for col, col_type in migrations:
            if col not in existing:
                self.conn.execute(
                    f"ALTER TABLE appointments ADD COLUMN {col} {col_type}"
                )
                print(f"[ScheduleDB] Migrated: added column '{col}'")
        self.conn.commit()

    # ============================================================
    # INSERT
    # ============================================================
    def insert_appointment(self, appt: AppointmentRecord) -> AppointmentRecord:
        """
        Insert appointment. Raises ValueError only if the assigned tech
        already has an overlapping booking at that time.
        If no tech is assigned, no conflict check is performed.
        """
        if appt.tech_user_id:
            conflict = self._find_tech_conflict(
                appt.appt_date, appt.appt_time,
                appt.duration_min, appt.tech_user_id, exclude_id=None
            )
            if conflict:
                raise ValueError(
                    f"This tech already has an appointment at "
                    f"{conflict['appt_time']} ({conflict['appt_type']})."
                )

        cursor = self.conn.execute(
            """
            INSERT INTO appointments
                (patient_id, appt_date, appt_time, appt_type,
                 tech_user_id, duration_min, status, notes, created_by, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (appt.patient_id, appt.appt_date, appt.appt_time,
             appt.appt_type, appt.tech_user_id, appt.duration_min,
             appt.status, appt.notes, appt.created_by, appt.created_at)
        )
        self.conn.commit()
        appt.id = cursor.lastrowid
        return appt

    # ============================================================
    # FETCH
    # ============================================================
    def fetch_for_date(self, date_str: str) -> list:
        rows = self.conn.execute(
            "SELECT * FROM appointments WHERE appt_date = ? ORDER BY appt_time",
            (date_str,)
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def fetch_for_patient(self, patient_id: int) -> list:
        rows = self.conn.execute(
            "SELECT * FROM appointments WHERE patient_id = ? ORDER BY appt_date, appt_time",
            (patient_id,)
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def fetch_range(self, start_date: str, end_date: str) -> list:
        rows = self.conn.execute(
            "SELECT * FROM appointments "
            "WHERE appt_date >= ? AND appt_date <= ? "
            "ORDER BY appt_date, appt_time",
            (start_date, end_date)
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def fetch_by_id(self, appt_id: int):
        row = self.conn.execute(
            "SELECT * FROM appointments WHERE id = ?", (appt_id,)
        ).fetchone()
        return self._row_to_record(row) if row else None

    def fetch_all(self) -> list:
        rows = self.conn.execute(
            "SELECT * FROM appointments ORDER BY appt_date DESC, appt_time"
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    # ============================================================
    # AVAILABLE SLOTS (per-tech)
    # ============================================================
    def get_available_slots(self, date_str: str, duration_min: int,
                            tech_user_id: int = None) -> list:
        """
        Returns list of HH:MM strings free on the given date.

        If tech_user_id is given:  only that tech's bookings block slots.
        If tech_user_id is None:   all slots within clinic hours are returned
                                   (no tech assigned, so no conflicts).
        """
        open_min  = CLINIC_OPEN.hour  * 60 + CLINIC_OPEN.minute
        close_min = CLINIC_CLOSE.hour * 60 + CLINIC_CLOSE.minute

        booked = []
        if tech_user_id:
            existing = self.conn.execute(
                "SELECT appt_time, duration_min FROM appointments "
                "WHERE appt_date = ? AND tech_user_id = ? AND status != 'Cancelled'",
                (date_str, tech_user_id)
            ).fetchall()
            for row in existing:
                h, m  = map(int, row["appt_time"].split(":"))
                start = h * 60 + m
                booked.append((start, start + row["duration_min"]))

        slots = []
        t = open_min
        while t + duration_min <= close_min:
            end     = t + duration_min
            overlap = any(not (end <= bs or t >= be) for bs, be in booked)
            if not overlap:
                slots.append(f"{t // 60:02d}:{t % 60:02d}")
            t += SLOT_INTERVAL
        return slots

    # ============================================================
    # CONFLICT CHECK (per-tech)
    # ============================================================
    def is_slot_available(self, date_str, time_str, duration_min,
                          tech_user_id=None, exclude_id=None) -> bool:
        if not tech_user_id:
            return True
        return self._find_tech_conflict(
            date_str, time_str, duration_min, tech_user_id, exclude_id
        ) is None

    def _find_tech_conflict(self, date_str, time_str, duration_min,
                            tech_user_id, exclude_id):
        h, m      = map(int, time_str.split(":"))
        new_start = h * 60 + m
        new_end   = new_start + duration_min

        rows = self.conn.execute(
            "SELECT * FROM appointments "
            "WHERE appt_date = ? AND tech_user_id = ? AND status != 'Cancelled'",
            (date_str, tech_user_id)
        ).fetchall()

        for row in rows:
            if exclude_id and row["id"] == exclude_id:
                continue
            eh, em   = map(int, row["appt_time"].split(":"))
            ex_start = eh * 60 + em
            ex_end   = ex_start + row["duration_min"]
            if not (new_end <= ex_start or new_start >= ex_end):
                return dict(row)
        return None

    # ============================================================
    # UPDATE
    # ============================================================
    def update_status(self, appt_id: int, status: str):
        self.conn.execute(
            "UPDATE appointments SET status = ? WHERE id = ?",
            (status, appt_id)
        )
        self.conn.commit()

    def update_appointment(self, appt: AppointmentRecord):
        if appt.tech_user_id:
            conflict = self._find_tech_conflict(
                appt.appt_date, appt.appt_time,
                appt.duration_min, appt.tech_user_id, exclude_id=appt.id
            )
            if conflict:
                raise ValueError(
                    f"This tech already has an appointment at "
                    f"{conflict['appt_time']} ({conflict['appt_type']})."
                )
        self.conn.execute(
            """
            UPDATE appointments
            SET patient_id=?, appt_date=?, appt_time=?, appt_type=?,
                tech_user_id=?, duration_min=?, status=?, notes=?
            WHERE id=?
            """,
            (appt.patient_id, appt.appt_date, appt.appt_time,
             appt.appt_type, appt.tech_user_id, appt.duration_min,
             appt.status, appt.notes, appt.id)
        )
        self.conn.commit()

    def delete_appointment(self, appt_id: int):
        self.conn.execute("DELETE FROM appointments WHERE id = ?", (appt_id,))
        self.conn.commit()

    # ============================================================
    # HELPER
    # ============================================================
    @staticmethod
    def _row_to_record(row) -> AppointmentRecord:
        return AppointmentRecord(
            id           = row["id"],
            patient_id   = row["patient_id"],
            appt_date    = row["appt_date"],
            appt_time    = row["appt_time"],
            appt_type    = row["appt_type"],
            tech_user_id = row["tech_user_id"],
            duration_min = row["duration_min"],
            status       = row["status"],
            notes        = row["notes"] or "",
            created_by   = row["created_by"] or "",
            created_at   = row["created_at"] or "",
        )