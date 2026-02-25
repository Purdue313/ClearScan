from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class PatientRecord:
    first_name: str
    last_name: str
    dob: str                  # "YYYY-MM-DD"
    mrn: str = ""             # Medical Record Number (optional, can be blank)
    notes: str = ""
    id: int = None
    created_at: str = ""

    @staticmethod
    def create(first_name, last_name, dob, mrn="", notes=""):
        return PatientRecord(
            first_name = first_name.strip(),
            last_name  = last_name.strip(),
            dob        = dob,
            mrn        = mrn.strip(),
            notes      = notes.strip(),
            created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

    @property
    def full_name(self):
        return f"{self.last_name}, {self.first_name}"

    @property
    def display(self):
        mrn_part = f"  MRN: {self.mrn}" if self.mrn else ""
        return f"{self.full_name}  |  DOB: {self.dob}{mrn_part}"