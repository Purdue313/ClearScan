from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QTextEdit, QSpinBox, QMessageBox,
    QListWidget, QListWidgetItem, QFrame, QLineEdit,
    QScrollArea, QWidget, QDateEdit, QSizePolicy
)
from PySide6.QtCore import Qt, QDate, Signal
from PySide6.QtGui import QColor

from Database.scheduleDatabase import ScheduleDatabase
from Database.patientDatabase import PatientDatabase
from Database.userDatabase import UserDatabase
from Source.Models.patient_model import PatientRecord
from Source.Models.appointment_model import (
    AppointmentRecord, ALL_TYPES, ALL_STATUSES,
    DEFAULT_DURATION, STATUS_SCHEDULED
)
from Source.Models.user_model import ROLE_XRAY_TECH

NAVY   = "#0a1628"
STEEL  = "#1c3150"
STEEL2 = "#243d5e"
TEAL   = "#0d9488"
TEAL_L = "#14b8a8"
WHITE  = "#ffffff"
GREY   = "#64748b"
LGREY  = "#94a3b8"
RED    = "#dc2626"
AMBER  = "#d97706"

FIELD_STYLE = f"""
    background:{STEEL}; color:{WHITE};
    border:1px solid #2d4a6e; border-radius:6px;
    padding:5px 10px; font-size:13px;
"""
SECTION_STYLE = (
    f"color:{LGREY}; font-size:10px; font-weight:700; "
    f"letter-spacing:2px; background:transparent;"
)


def section_label(text):
    l = QLabel(text)
    l.setStyleSheet(SECTION_STYLE)
    return l


# ================================================================
# New Patient Dialog  (separate modal, not inline)
# ================================================================
class NewPatientDialog(QDialog):
    def __init__(self, patient_db: PatientDatabase, parent=None):
        super().__init__(parent)
        self.patient_db   = patient_db
        self.new_patient  = None
        self.setWindowTitle("Create New Patient")
        self.setFixedWidth(400)
        self.setModal(True)
        self.setStyleSheet(f"""
            QDialog {{ background:{NAVY}; }}
            QLabel  {{ color:{WHITE}; background:transparent; font-size:13px; }}
            QLineEdit, QDateEdit {{
                {FIELD_STYLE}
            }}
            QDateEdit::drop-down {{ border:none; width:20px; }}
        """)
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(12)

        title = QLabel("New Patient")
        title.setStyleSheet(
            f"color:{TEAL_L}; font-size:15px; font-weight:700; background:transparent;"
        )
        root.addWidget(title)

        def lbl(t):
            l = QLabel(t)
            l.setStyleSheet(f"color:{LGREY}; font-size:11px; background:transparent;")
            return l

        def field(ph=""):
            f = QLineEdit()
            f.setPlaceholderText(ph)
            f.setFixedHeight(36)
            f.setStyleSheet(FIELD_STYLE)
            return f

        self.fn_edit  = field("First name")
        self.ln_edit  = field("Last name")
        self.mrn_edit = field("Optional")

        self.dob_edit = QDateEdit()
        self.dob_edit.setCalendarPopup(True)
        self.dob_edit.setDisplayFormat("yyyy-MM-dd")
        self.dob_edit.setDate(QDate(1990, 1, 1))
        self.dob_edit.setFixedHeight(36)
        self.dob_edit.setStyleSheet(FIELD_STYLE)

        root.addWidget(lbl("FIRST NAME *"))
        root.addWidget(self.fn_edit)
        root.addWidget(lbl("LAST NAME *"))
        root.addWidget(self.ln_edit)
        root.addWidget(lbl("DATE OF BIRTH *"))
        root.addWidget(self.dob_edit)
        root.addWidget(lbl("MRN"))
        root.addWidget(self.mrn_edit)

        self.err_lbl = QLabel("")
        self.err_lbl.setStyleSheet(f"color:{RED}; font-size:12px; background:transparent;")
        self.err_lbl.setVisible(False)
        root.addWidget(self.err_lbl)

        btn_row = QHBoxLayout()
        cancel = QPushButton("Cancel")
        cancel.setFixedHeight(36)
        cancel.setStyleSheet(f"""
            QPushButton {{
                background:transparent; color:{GREY};
                border:1px solid #2d4a6e; border-radius:6px;
                font-size:13px; padding:0 16px;
            }}
            QPushButton:hover {{ color:{WHITE}; }}
        """)
        cancel.clicked.connect(self.reject)

        create = QPushButton("Create Patient")
        create.setFixedHeight(36)
        create.setStyleSheet(f"""
            QPushButton {{
                background:{TEAL}; color:{WHITE}; border:none;
                border-radius:6px; font-size:13px; font-weight:600; padding:0 18px;
            }}
            QPushButton:hover {{ background:{TEAL_L}; }}
        """)
        create.clicked.connect(self._create)

        btn_row.addWidget(cancel)
        btn_row.addStretch()
        btn_row.addWidget(create)
        root.addLayout(btn_row)

    def _create(self):
        first = self.fn_edit.text().strip()
        last  = self.ln_edit.text().strip()
        dob   = self.dob_edit.date().toString("yyyy-MM-dd")
        mrn   = self.mrn_edit.text().strip()
        if not first or not last:
            self.err_lbl.setText("First and last name are required.")
            self.err_lbl.setVisible(True)
            return
        try:
            self.new_patient = self.patient_db.insert_patient(
                PatientRecord.create(
                    first_name=first, last_name=last, dob=dob, mrn=mrn
                )
            )
            self.accept()
        except Exception as e:
            self.err_lbl.setText(str(e))
            self.err_lbl.setVisible(True)


# ================================================================
# Booking Dialog
# ================================================================
class BookingDialog(QDialog):
    """
    Book or edit an appointment.
    Scrollable form - new patient opens as a separate modal dialog.
    Double-booking prevention is per-tech.
    """

    def __init__(self, schedule_db: ScheduleDatabase,
                 patient_db: PatientDatabase,
                 user_db: UserDatabase,
                 existing: AppointmentRecord = None,
                 preselect_date: str = None,
                 created_by: str = "",
                 parent=None):
        super().__init__(parent)
        self.schedule_db = schedule_db
        self.patient_db  = patient_db
        self.user_db     = user_db
        self.existing    = existing
        self.created_by  = created_by
        self.result_appt = None

        self.setWindowTitle("Edit Appointment" if existing else "Book Appointment")
        self.setFixedWidth(520)
        self.setMinimumHeight(500)
        self.setMaximumHeight(820)
        self.setModal(True)
        self.setStyleSheet(f"""
            QDialog   {{ background:{NAVY}; }}
            QLabel    {{ color:{WHITE}; background:transparent; font-size:13px; }}
            QComboBox, QSpinBox, QTextEdit, QDateEdit, QLineEdit {{
                background:{STEEL}; color:{WHITE};
                border:1px solid #2d4a6e; border-radius:6px;
                padding:5px 10px; font-size:13px;
            }}
            QComboBox QAbstractItemView {{
                background:{STEEL}; color:{WHITE};
                selection-background-color:{TEAL};
            }}
            QSpinBox::up-button, QSpinBox::down-button {{ background:{STEEL}; border:none; }}
            QDateEdit::drop-down {{ border:none; width:20px; }}
            QScrollArea {{ border:none; background:transparent; }}
            QScrollBar:vertical {{
                background:{NAVY}; width:6px; border:none;
            }}
            QScrollBar::handle:vertical {{
                background:#2d4a6e; border-radius:3px; min-height:20px;
            }}
        """)

        self._preselect_date = preselect_date
        self._build_ui()

        if existing:
            self._populate(existing)
        elif preselect_date:
            self.date_edit.setDate(QDate.fromString(preselect_date, "yyyy-MM-dd"))
            self._refresh_slots()

    # ============================================================
    # UI
    # ============================================================
    def _build_ui(self):
        # Outer layout: scrollable form + fixed bottom buttons
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # --- Scrollable form area ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        form_widget = QWidget()
        form_widget.setStyleSheet(f"background:{NAVY};")
        form = QVBoxLayout(form_widget)
        form.setContentsMargins(24, 20, 24, 12)
        form.setSpacing(10)

        # PATIENT
        form.addWidget(section_label("PATIENT"))
        pat_row = QHBoxLayout()
        self.patient_combo = QComboBox()
        self.patient_combo.setFixedHeight(36)
        self._load_patients()

        add_pt_btn = QPushButton("+ New Patient")
        add_pt_btn.setFixedHeight(36)
        add_pt_btn.setFixedWidth(120)
        add_pt_btn.setStyleSheet(f"""
            QPushButton {{
                background:{STEEL}; color:{TEAL_L};
                border:1px solid {TEAL}; border-radius:6px;
                font-size:12px; font-weight:600; padding:0 8px;
            }}
            QPushButton:hover {{ background:{TEAL}; color:{WHITE}; }}
        """)
        add_pt_btn.clicked.connect(self._open_new_patient_dialog)
        pat_row.addWidget(self.patient_combo, 1)
        pat_row.addWidget(add_pt_btn)
        form.addLayout(pat_row)

        # APPOINTMENT TYPE
        form.addSpacing(6)
        form.addWidget(section_label("APPOINTMENT TYPE"))
        self.type_combo = QComboBox()
        self.type_combo.setFixedHeight(36)
        for t in ALL_TYPES:
            self.type_combo.addItem(t, t)
        self.type_combo.setEnabled(len(ALL_TYPES) > 1)
        form.addWidget(self.type_combo)

        # DURATION
        form.addSpacing(6)
        form.addWidget(section_label("DURATION"))
        self.duration_spin = QSpinBox()
        self.duration_spin.setFixedHeight(36)
        self.duration_spin.setRange(10, 120)
        self.duration_spin.setSingleStep(5)
        self.duration_spin.setSuffix(" min")
        self.duration_spin.setValue(DEFAULT_DURATION.get(ALL_TYPES[0], 30))
        self.duration_spin.valueChanged.connect(self._refresh_slots)
        form.addWidget(self.duration_spin)

        # DATE
        form.addSpacing(6)
        form.addWidget(section_label("DATE"))
        self.date_edit = QDateEdit()
        self.date_edit.setFixedHeight(36)
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDate(QDate.currentDate())
        self.date_edit.setDisplayFormat("yyyy-MM-dd")
        self.date_edit.setMinimumDate(QDate.currentDate())
        self.date_edit.dateChanged.connect(self._refresh_slots)
        form.addWidget(self.date_edit)

        # ASSIGNED TECH
        form.addSpacing(6)
        form.addWidget(section_label("ASSIGNED X-RAY TECH"))
        self.tech_combo = QComboBox()
        self.tech_combo.setFixedHeight(36)
        self._load_techs()
        self.tech_combo.currentIndexChanged.connect(self._refresh_slots)
        form.addWidget(self.tech_combo)

        self.tech_note = QLabel("")
        self.tech_note.setStyleSheet(
            f"color:{AMBER}; font-size:11px; background:transparent;"
        )
        self.tech_note.setVisible(False)
        form.addWidget(self.tech_note)

        # AVAILABLE TIME SLOTS
        form.addSpacing(6)
        form.addWidget(section_label("AVAILABLE TIME SLOTS"))
        self.slot_list = QListWidget()
        self.slot_list.setFixedHeight(160)
        self.slot_list.setStyleSheet(f"""
            QListWidget {{
                background:{STEEL}; color:{WHITE};
                border:1px solid #2d4a6e; border-radius:6px;
                font-size:13px; outline:none;
            }}
            QListWidget::item {{ padding:7px 14px; border-bottom:1px solid {NAVY}; }}
            QListWidget::item:selected {{ background:{TEAL}; color:{WHITE}; }}
            QListWidget::item:hover    {{ background:{STEEL2}; }}
        """)
        form.addWidget(self.slot_list)

        # STATUS (edit only)
        if self.existing:
            form.addSpacing(6)
            form.addWidget(section_label("STATUS"))
            self.status_combo = QComboBox()
            self.status_combo.setFixedHeight(36)
            for s in ALL_STATUSES:
                self.status_combo.addItem(s, s)
            form.addWidget(self.status_combo)

        # NOTES
        form.addSpacing(6)
        form.addWidget(section_label("NOTES"))
        self.notes_edit = QTextEdit()
        self.notes_edit.setPlaceholderText("Optional notes or special instructions...")
        self.notes_edit.setFixedHeight(72)
        form.addWidget(self.notes_edit)

        form.addStretch()
        scroll.setWidget(form_widget)
        outer.addWidget(scroll, 1)

        # --- Fixed bottom bar (error + buttons) ---
        bottom = QFrame()
        bottom.setStyleSheet(
            f"background:{STEEL}; border-top:1px solid #2d4a6e;"
        )
        bot_lay = QVBoxLayout(bottom)
        bot_lay.setContentsMargins(24, 12, 24, 16)
        bot_lay.setSpacing(8)

        self.err_lbl = QLabel("")
        self.err_lbl.setStyleSheet(
            f"color:{RED}; font-size:12px; background:transparent;"
        )
        self.err_lbl.setVisible(False)
        bot_lay.addWidget(self.err_lbl)

        btn_row = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFixedHeight(40)
        cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background:transparent; color:{GREY};
                border:1px solid #2d4a6e; border-radius:7px;
                font-size:13px; padding:0 20px;
            }}
            QPushButton:hover {{ color:{WHITE}; border-color:{WHITE}; }}
        """)
        cancel_btn.clicked.connect(self.reject)

        self.save_btn = QPushButton(
            "Save Changes" if self.existing else "Book Appointment"
        )
        self.save_btn.setFixedHeight(40)
        self.save_btn.setStyleSheet(f"""
            QPushButton {{
                background:{TEAL}; color:{WHITE}; border:none;
                border-radius:7px; font-size:13px; font-weight:700;
                padding:0 24px;
            }}
            QPushButton:hover {{ background:{TEAL_L}; }}
        """)
        self.save_btn.clicked.connect(self._save)

        btn_row.addWidget(cancel_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.save_btn)
        bot_lay.addLayout(btn_row)

        outer.addWidget(bottom)

        self._refresh_slots()

    # ============================================================
    # DATA
    # ============================================================
    def _load_patients(self):
        self.patient_combo.clear()
        self._patients = self.patient_db.fetch_all_patients()
        self.patient_combo.addItem("-- Select patient --", None)
        for p in self._patients:
            label = f"{p.full_name}  (DOB: {p.dob})"
            if p.mrn:
                label += f"  |  MRN: {p.mrn}"
            self.patient_combo.addItem(label, p.id)

    def _load_techs(self):
        self.tech_combo.clear()
        self._techs = [
            u for u in self.user_db.fetch_all_users()
            if u.role == ROLE_XRAY_TECH and u.is_active
        ]
        self.tech_combo.addItem("-- Unassigned --", None)
        for t in self._techs:
            self.tech_combo.addItem(t.full_name, t.id)

    # ============================================================
    # SLOT REFRESH
    # ============================================================
    def _refresh_slots(self):
        date_str     = self.date_edit.date().toString("yyyy-MM-dd")
        duration     = self.duration_spin.value()
        tech_user_id = self.tech_combo.currentData()

        slots = self.schedule_db.get_available_slots(
            date_str, duration, tech_user_id
        )

        if self.existing and self.existing.appt_time not in slots:
            if self.existing.appt_date == date_str:
                slots.insert(0, self.existing.appt_time)

        self.slot_list.clear()
        for slot in slots:
            h, m   = map(int, slot.split(":"))
            suffix = "AM" if h < 12 else "PM"
            h12    = h % 12 or 12
            item   = QListWidgetItem(f"{h12}:{m:02d} {suffix}")
            item.setData(Qt.UserRole, slot)
            self.slot_list.addItem(item)

        if not slots:
            msg = (
                "No available slots -- tech fully booked on this date"
                if tech_user_id else
                "No slots available for this date/duration"
            )
            item = QListWidgetItem(msg)
            item.setForeground(QColor(GREY))
            item.setFlags(Qt.NoItemFlags)
            self.slot_list.addItem(item)

        if tech_user_id:
            appts  = self.schedule_db.fetch_for_date(date_str)
            busy   = sum(
                1 for a in appts
                if a.tech_user_id == tech_user_id and a.status != "Cancelled"
            )
            if busy:
                self.tech_note.setText(
                    f"This tech has {busy} booking(s) on this date."
                )
                self.tech_note.setVisible(True)
            else:
                self.tech_note.setVisible(False)
        else:
            self.tech_note.setVisible(False)

    # ============================================================
    # NEW PATIENT (separate modal)
    # ============================================================
    def _open_new_patient_dialog(self):
        dlg = NewPatientDialog(self.patient_db, parent=self)
        if dlg.exec() and dlg.new_patient:
            self._load_patients()
            for i in range(self.patient_combo.count()):
                if self.patient_combo.itemData(i) == dlg.new_patient.id:
                    self.patient_combo.setCurrentIndex(i)
                    break

    # ============================================================
    # POPULATE (edit mode)
    # ============================================================
    def _populate(self, appt: AppointmentRecord):
        for i in range(self.patient_combo.count()):
            if self.patient_combo.itemData(i) == appt.patient_id:
                self.patient_combo.setCurrentIndex(i)
                break
        idx = self.type_combo.findData(appt.appt_type)
        if idx >= 0:
            self.type_combo.setCurrentIndex(idx)
        self.duration_spin.setValue(appt.duration_min)
        self.date_edit.setDate(QDate.fromString(appt.appt_date, "yyyy-MM-dd"))
        for i in range(self.tech_combo.count()):
            if self.tech_combo.itemData(i) == appt.tech_user_id:
                self.tech_combo.setCurrentIndex(i)
                break
        self._refresh_slots()
        for i in range(self.slot_list.count()):
            item = self.slot_list.item(i)
            if item and item.data(Qt.UserRole) == appt.appt_time:
                self.slot_list.setCurrentItem(item)
                break
        if hasattr(self, "status_combo"):
            idx = self.status_combo.findData(appt.status)
            if idx >= 0:
                self.status_combo.setCurrentIndex(idx)
        self.notes_edit.setPlainText(appt.notes)

    # ============================================================
    # SAVE
    # ============================================================
    def _save(self):
        self.err_lbl.setVisible(False)

        patient_id = self.patient_combo.currentData()
        if not patient_id:
            self._show_err("Please select a patient.")
            return

        selected = self.slot_list.currentItem()
        if not selected or not selected.data(Qt.UserRole):
            self._show_err("Please select a time slot.")
            return

        time_str     = selected.data(Qt.UserRole)
        date_str     = self.date_edit.date().toString("yyyy-MM-dd")
        appt_type    = self.type_combo.currentData()
        duration     = self.duration_spin.value()
        tech_user_id = self.tech_combo.currentData()
        notes        = self.notes_edit.toPlainText().strip()
        status       = (
            self.status_combo.currentData()
            if hasattr(self, "status_combo") else STATUS_SCHEDULED
        )

        try:
            if self.existing:
                self.existing.patient_id   = patient_id
                self.existing.appt_date    = date_str
                self.existing.appt_time    = time_str
                self.existing.appt_type    = appt_type
                self.existing.tech_user_id = tech_user_id
                self.existing.duration_min = duration
                self.existing.notes        = notes
                self.existing.status       = status
                self.schedule_db.update_appointment(self.existing)
                self.result_appt = self.existing
            else:
                appt = AppointmentRecord.create(
                    patient_id   = patient_id,
                    appt_date    = date_str,
                    appt_time    = time_str,
                    appt_type    = appt_type,
                    tech_user_id = tech_user_id,
                    duration_min = duration,
                    notes        = notes,
                    created_by   = self.created_by,
                )
                self.result_appt = self.schedule_db.insert_appointment(appt)
        except ValueError as e:
            self._show_err(str(e))
            return

        self.accept()

    def _show_err(self, msg: str):
        self.err_lbl.setText(msg)
        self.err_lbl.setVisible(True)