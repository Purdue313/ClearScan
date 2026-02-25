from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QDateEdit,
    QStackedWidget, QWidget, QFormLayout, QTextEdit,
    QMessageBox, QFrame, QSizePolicy
)
from PySide6.QtCore import Qt, QDate, Signal
from PySide6.QtGui import QFont, QColor

from Database.patientDatabase import PatientDatabase
from Source.Models.patient_model import PatientRecord


# ── Colours (match the dark clinical theme) ─────────────────────
NAVY  = "#0a1628"
STEEL = "#1c3150"
TEAL  = "#0d9488"
TEAL_L= "#14b8a8"
WHITE = "#ffffff"
GREY  = "#64748b"
GREEN = "#16a34a"
RED   = "#dc2626"


class PatientSelectorDialog(QDialog):
    """
    Modal dialog shown before uploading a scan.

    Pages:
        0 – Search existing patients
        1 – Create new patient

    Returns the selected/created PatientRecord via .selected_patient
    after exec() returns QDialog.Accepted.
    """

    def __init__(self, patient_db: PatientDatabase, parent=None):
        super().__init__(parent)
        self.patient_db       = patient_db
        self.selected_patient = None
        self._all_patients    = []

        self.setWindowTitle("Select Patient")
        self.setMinimumSize(560, 480)
        self.setModal(True)
        self.setStyleSheet(f"""
            QDialog {{ background: {NAVY}; }}
            QLabel  {{ color: {WHITE}; font-size: 13px; background: transparent; }}
            QLineEdit, QDateEdit, QTextEdit {{
                background: {STEEL};
                color: {WHITE};
                border: 1px solid #2d4a6e;
                border-radius: 6px;
                padding: 6px 10px;
                font-size: 13px;
            }}
            QLineEdit:focus, QDateEdit:focus, QTextEdit:focus {{
                border-color: {TEAL};
            }}
        """)

        self._build_ui()
        self._load_patients()

    # ============================================================
    # UI
    # ============================================================
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(12)

        # ── Tab row ──────────────────────────────────────────────
        tab_row = QHBoxLayout()
        self.search_tab_btn = self._tab_btn("Search Patients", active=True)
        self.new_tab_btn    = self._tab_btn("+ New Patient",   active=False)
        self.search_tab_btn.clicked.connect(lambda: self._switch_page(0))
        self.new_tab_btn.clicked.connect(lambda: self._switch_page(1))
        tab_row.addWidget(self.search_tab_btn)
        tab_row.addWidget(self.new_tab_btn)
        tab_row.addStretch()

        # ── Stacked pages ────────────────────────────────────────
        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_search_page())
        self.stack.addWidget(self._build_new_page())

        # ── Bottom buttons ───────────────────────────────────────
        btn_row = QHBoxLayout()

        self.skip_btn = QPushButton("Skip — No Patient")
        self.skip_btn.setFixedHeight(38)
        self.skip_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {GREY};
                border: 1px solid #2d4a6e; border-radius: 6px;
                font-size: 13px; padding: 0 16px;
            }}
            QPushButton:hover {{ color: {WHITE}; border-color: {TEAL}; }}
        """)
        self.skip_btn.clicked.connect(self.reject)

        self.confirm_btn = QPushButton("Confirm Selection")
        self.confirm_btn.setFixedHeight(38)
        self.confirm_btn.setEnabled(False)
        self.confirm_btn.setStyleSheet(f"""
            QPushButton {{
                background: {TEAL}; color: {WHITE}; border: none;
                border-radius: 6px; font-size: 13px; font-weight: 600;
                padding: 0 20px;
            }}
            QPushButton:hover    {{ background: {TEAL_L}; }}
            QPushButton:disabled {{ background: #2d4a6e; color: {GREY}; }}
        """)
        self.confirm_btn.clicked.connect(self._confirm)

        btn_row.addWidget(self.skip_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.confirm_btn)

        root.addLayout(tab_row)
        root.addWidget(self._divider())
        root.addWidget(self.stack, 1)
        root.addWidget(self._divider())
        root.addLayout(btn_row)

    # ── Search page ──────────────────────────────────────────────
    def _build_search_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(10)

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("🔍  Search by name or MRN…")
        self.search_box.setFixedHeight(38)
        self.search_box.textChanged.connect(self._filter_patients)

        self.patient_list = QListWidget()
        self.patient_list.setStyleSheet(f"""
            QListWidget {{
                background: {STEEL}; color: {WHITE};
                border: 1px solid #2d4a6e; border-radius: 8px;
            }}
            QListWidget::item {{
                padding: 10px 14px;
                border-bottom: 1px solid {NAVY};
            }}
            QListWidget::item:selected {{
                background: #1a3a5c; color: {WHITE};
            }}
            QListWidget::item:hover {{ background: #243d5e; }}
        """)
        self.patient_list.itemSelectionChanged.connect(self._on_patient_selected)
        self.patient_list.itemDoubleClicked.connect(lambda: self._confirm())

        layout.addWidget(self.search_box)
        layout.addWidget(self.patient_list, 1)
        return page

    # ── New patient page ─────────────────────────────────────────
    def _build_new_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(0)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        form.setSpacing(12)
        form.setContentsMargins(0, 0, 0, 12)

        self.fn_edit  = QLineEdit(); self.fn_edit.setFixedHeight(36)
        self.ln_edit  = QLineEdit(); self.ln_edit.setFixedHeight(36)
        self.mrn_edit = QLineEdit(); self.mrn_edit.setFixedHeight(36)
        self.mrn_edit.setPlaceholderText("Optional")

        self.dob_edit = QDateEdit()
        self.dob_edit.setCalendarPopup(True)
        self.dob_edit.setDate(QDate(1980, 1, 1))
        self.dob_edit.setDisplayFormat("yyyy-MM-dd")
        self.dob_edit.setFixedHeight(36)
        self.dob_edit.setStyleSheet(f"""
            QDateEdit {{
                background: {STEEL}; color: {WHITE};
                border: 1px solid #2d4a6e; border-radius: 6px;
                padding: 0 8px; font-size: 13px;
            }}
            QDateEdit:focus {{ border-color: {TEAL}; }}
            QDateEdit::drop-down {{ border: none; width: 20px; }}
        """)

        self.notes_edit = QTextEdit()
        self.notes_edit.setPlaceholderText("Clinical notes (optional)")
        self.notes_edit.setFixedHeight(70)

        form.addRow("First name *", self.fn_edit)
        form.addRow("Last name *",  self.ln_edit)
        form.addRow("Date of birth *", self.dob_edit)
        form.addRow("MRN",          self.mrn_edit)
        form.addRow("Notes",        self.notes_edit)

        self.create_btn = QPushButton("Create Patient & Select")
        self.create_btn.setFixedHeight(38)
        self.create_btn.setStyleSheet(f"""
            QPushButton {{
                background: {GREEN}; color: {WHITE}; border: none;
                border-radius: 6px; font-size: 13px; font-weight: 600;
            }}
            QPushButton:hover {{ background: #15803d; }}
        """)
        self.create_btn.clicked.connect(self._create_patient)

        layout.addLayout(form)
        layout.addWidget(self.create_btn)
        layout.addStretch()
        return page

    # ============================================================
    # HELPERS
    # ============================================================
    def _tab_btn(self, text, active):
        btn = QPushButton(text)
        btn.setFixedHeight(34)
        btn.setCheckable(True)
        btn.setChecked(active)
        self._style_tab(btn, active)
        return btn

    def _style_tab(self, btn, active):
        if active:
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: {TEAL}; color: {WHITE}; border: none;
                    border-radius: 6px; font-size: 13px; font-weight: 600;
                    padding: 0 18px;
                }}
            """)
        else:
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: transparent; color: {GREY};
                    border: 1px solid #2d4a6e; border-radius: 6px;
                    font-size: 13px; padding: 0 18px;
                }}
                QPushButton:hover {{ color: {WHITE}; border-color: {TEAL}; }}
            """)

    def _divider(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"color: #2d4a6e;")
        return line

    def _switch_page(self, index):
        self.stack.setCurrentIndex(index)
        self._style_tab(self.search_tab_btn, index == 0)
        self._style_tab(self.new_tab_btn,    index == 1)
        # Clear any selection when switching pages
        if index == 0:
            self.confirm_btn.setEnabled(
                len(self.patient_list.selectedItems()) > 0
            )
        else:
            self.confirm_btn.setEnabled(False)

    # ============================================================
    # LOAD / FILTER PATIENTS
    # ============================================================
    def _load_patients(self):
        self._all_patients = self.patient_db.fetch_all_patients()
        self._populate_list(self._all_patients)

    def _filter_patients(self, text):
        if text.strip():
            results = self.patient_db.search_patients(text)
        else:
            results = self._all_patients
        self._populate_list(results)

    def _populate_list(self, patients):
        self.patient_list.clear()
        for p in patients:
            item = QListWidgetItem(p.display)
            item.setData(Qt.UserRole, p)
            self.patient_list.addItem(item)

        if not patients:
            placeholder = QListWidgetItem("No patients found")
            placeholder.setForeground(QColor(GREY))
            placeholder.setFlags(Qt.NoItemFlags)
            self.patient_list.addItem(placeholder)

    def _on_patient_selected(self):
        items = self.patient_list.selectedItems()
        has_valid = bool(items) and bool(items[0].data(Qt.UserRole))
        self.confirm_btn.setEnabled(has_valid)

    # ============================================================
    # CONFIRM SELECTION (search page)
    # ============================================================
    def _confirm(self):
        if self.stack.currentIndex() == 0:
            items = self.patient_list.selectedItems()
            if not items or not items[0].data(Qt.UserRole):
                return
            self.selected_patient = items[0].data(Qt.UserRole)
            self.accept()

    # ============================================================
    # CREATE NEW PATIENT
    # ============================================================
    def _create_patient(self):
        first = self.fn_edit.text().strip()
        last  = self.ln_edit.text().strip()
        dob   = self.dob_edit.date().toString("yyyy-MM-dd")

        if not first or not last:
            QMessageBox.warning(self, "Missing Fields",
                                "First name and last name are required.")
            return

        patient = PatientRecord.create(
            first_name = first,
            last_name  = last,
            dob        = dob,
            mrn        = self.mrn_edit.text().strip(),
            notes      = self.notes_edit.toPlainText().strip(),
        )
        patient = self.patient_db.insert_patient(patient)

        # Refresh the all-patients cache
        self._all_patients = self.patient_db.fetch_all_patients()

        self.selected_patient = patient
        self.accept()