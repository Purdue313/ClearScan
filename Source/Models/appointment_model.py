from dataclasses import dataclass
from datetime import datetime, time

# Only X-Ray available at this time
TYPE_XRAY = "X-Ray"
ALL_TYPES = [TYPE_XRAY]

# Status constants
STATUS_SCHEDULED = "Scheduled"
STATUS_CONFIRMED = "Confirmed"
STATUS_CANCELLED = "Cancelled"
STATUS_COMPLETED = "Completed"
STATUS_NO_SHOW   = "No Show"

ALL_STATUSES = [STATUS_SCHEDULED, STATUS_CONFIRMED,
                STATUS_CANCELLED, STATUS_COMPLETED, STATUS_NO_SHOW]

STATUS_COLORS = {
    STATUS_SCHEDULED: "#1d4ed8",
    STATUS_CONFIRMED: "#16a34a",
    STATUS_CANCELLED: "#dc2626",
    STATUS_COMPLETED: "#0d9488",
    STATUS_NO_SHOW:   "#d97706",
}

DEFAULT_DURATION = {TYPE_XRAY: 30}

# Clinic hours
CLINIC_OPEN  = time(8, 0)
CLINIC_CLOSE = time(17, 0)
SLOT_INTERVAL = 15   # minutes between bookable slots


@dataclass
class AppointmentRecord:
    patient_id:    int
    appt_date:     str          # "YYYY-MM-DD"
    appt_time:     str          # "HH:MM" 24-hour
    appt_type:     str
    duration_min:  int
    tech_user_id:  int   = None  # assigned X-Ray tech (users.id)
    status:        str   = STATUS_SCHEDULED
    notes:         str   = ""
    id:            int   = None
    created_by:    str   = ""
    created_at:    str   = ""
    patient_name:  str   = ""   # populated on fetch
    tech_name:     str   = ""   # populated on fetch

    @staticmethod
    def create(patient_id, appt_date, appt_time, appt_type,
               tech_user_id=None, duration_min=None,
               notes="", created_by=""):
        dur = duration_min or DEFAULT_DURATION.get(appt_type, 30)
        return AppointmentRecord(
            patient_id   = patient_id,
            appt_date    = appt_date,
            appt_time    = appt_time,
            appt_type    = appt_type,
            tech_user_id = tech_user_id,
            duration_min = dur,
            notes        = notes.strip(),
            created_by   = created_by,
            created_at   = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

    @property
    def display_time(self):
        try:
            h, m   = map(int, self.appt_time.split(":"))
            suffix = "AM" if h < 12 else "PM"
            h12    = h % 12 or 12
            return f"{h12}:{m:02d} {suffix}"
        except Exception:
            return self.appt_time

    @property
    def status_color(self):
        return STATUS_COLORS.get(self.status, "#64748b")