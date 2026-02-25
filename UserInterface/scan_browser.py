from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QLineEdit, QComboBox, QDateEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QFrame, QSizePolicy, QAbstractItemView, QCheckBox,
    QScrollArea, QMessageBox
)
from PySide6.QtCore import Qt, QDate, Signal, QSortFilterProxyModel
from PySide6.QtGui import QFont, QColor, QPalette

from Database.imageDatabase import ImageDatabase
from Database.findingsDatabase import FindingsDatabase
from Database.patientDatabase import PatientDatabase
from machineLearningModel.prediction import LABELS


# ── Colour palette ─────────────────────────────────────────────────────────
NAVY   = "#0a1628"
STEEL  = "#1c3150"
TEAL   = "#0d9488"
TEAL_L = "#14b8a8"
MIST   = "#e8f4f8"
WHITE  = "#ffffff"
GREY   = "#64748b"
LGREY  = "#f1f5f9"
RED    = "#dc2626"
GREEN  = "#16a34a"
AMBER  = "#d97706"


class FilterBar(QFrame):
    """
    The filter control strip at the top of the browser.
    Emits filter_changed whenever any control is updated.
    """
    filter_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("FilterBar")
        self.setStyleSheet(f"""
            QFrame#FilterBar {{
                background: {STEEL};
                border-radius: 10px;
                padding: 4px;
            }}
        """)

        root = QHBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(16)

        # ── Search box ──────────────────────────────────────────
        self.search = QLineEdit()
        self.search.setPlaceholderText("🔍  Search by ID or filename…")
        self.search.setFixedHeight(36)
        self.search.setStyleSheet(f"""
            QLineEdit {{
                background: {NAVY};
                color: {WHITE};
                border: 1px solid #2d4a6e;
                border-radius: 6px;
                padding: 0 12px;
                font-size: 13px;
                font-family: 'Consolas', monospace;
            }}
            QLineEdit:focus {{ border-color: {TEAL}; }}
        """)
        self.search.textChanged.connect(self.filter_changed)

        # ── Date from ───────────────────────────────────────────
        date_label = self._label("Date range:")
        self.date_from = self._date_edit()
        self.date_from.setDate(QDate.currentDate().addYears(-1))
        dash = self._label("→")
        self.date_to = self._date_edit()
        self.date_to.setDate(QDate.currentDate())
        self.date_from.dateChanged.connect(self.filter_changed)
        self.date_to.dateChanged.connect(self.filter_changed)

        # ── Diagnosis status ────────────────────────────────────
        status_label = self._label("Status:")
        self.status_combo = QComboBox()
        self.status_combo.addItems([
            "All scans",
            "Diagnosed",
            "Not diagnosed",
        ])
        self._style_combo(self.status_combo)
        self.status_combo.currentIndexChanged.connect(self.filter_changed)

        # ── Diagnosed as ────────────────────────────────────────
        dx_label = self._label("Diagnosed as:")
        self.dx_combo = QComboBox()
        self.dx_combo.addItem("Any condition")
        for lbl in sorted(LABELS):
            self.dx_combo.addItem(lbl)
        self._style_combo(self.dx_combo, width=200)
        self.dx_combo.currentIndexChanged.connect(self.filter_changed)

        # ── Clear button ─────────────────────────────────────────
        self.clear_btn = QPushButton("✕ Clear")
        self.clear_btn.setFixedHeight(36)
        self.clear_btn.setFixedWidth(80)
        self.clear_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {GREY};
                border: 1px solid #2d4a6e;
                border-radius: 6px;
                font-size: 12px;
            }}
            QPushButton:hover {{ color: {WHITE}; border-color: {TEAL}; }}
        """)
        self.clear_btn.clicked.connect(self._clear_filters)

        root.addWidget(self.search, 2)
        root.addWidget(date_label)
        root.addWidget(self.date_from)
        root.addWidget(dash)
        root.addWidget(self.date_to)
        root.addWidget(status_label)
        root.addWidget(self.status_combo)
        root.addWidget(dx_label)
        root.addWidget(self.dx_combo)
        root.addWidget(self.clear_btn)

    # ── helpers ──────────────────────────────────────────────────
    def _label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color:{GREY}; font-size:12px; background:transparent;")
        return lbl

    def _date_edit(self):
        d = QDateEdit()
        d.setCalendarPopup(True)
        d.setFixedHeight(36)
        d.setDisplayFormat("yyyy-MM-dd")
        d.setStyleSheet(f"""
            QDateEdit {{
                background: {NAVY};
                color: {WHITE};
                border: 1px solid #2d4a6e;
                border-radius: 6px;
                padding: 0 8px;
                font-size: 13px;
            }}
            QDateEdit:focus {{ border-color: {TEAL}; }}
            QDateEdit::drop-down {{ border: none; width: 20px; }}
        """)
        return d

    def _style_combo(self, combo, width=160):
        combo.setFixedHeight(36)
        combo.setFixedWidth(width)
        combo.setStyleSheet(f"""
            QComboBox {{
                background: {NAVY};
                color: {WHITE};
                border: 1px solid #2d4a6e;
                border-radius: 6px;
                padding: 0 10px;
                font-size: 13px;
            }}
            QComboBox:focus {{ border-color: {TEAL}; }}
            QComboBox::drop-down {{ border: none; width: 24px; }}
            QComboBox QAbstractItemView {{
                background: {STEEL};
                color: {WHITE};
                selection-background-color: {TEAL};
                border: 1px solid #2d4a6e;
            }}
        """)

    def _clear_filters(self):
        self.search.clear()
        self.date_from.setDate(QDate.currentDate().addYears(-1))
        self.date_to.setDate(QDate.currentDate())
        self.status_combo.setCurrentIndex(0)
        self.dx_combo.setCurrentIndex(0)

    # ── public getters ───────────────────────────────────────────
    def get_filters(self):
        return {
            "search":    self.search.text().strip().lower(),
            "date_from": self.date_from.date().toString("yyyy-MM-dd"),
            "date_to":   self.date_to.date().toString("yyyy-MM-dd"),
            "status":    self.status_combo.currentText(),
            "dx_as":     self.dx_combo.currentText(),
        }


class ScanBrowserWindow(QWidget):
    """
    Full-screen scan browser with search + filter.

    Emits open_image(image_id) when the user double-clicks a row
    so MainWindow can switch to that image in the viewer.
    """

    open_image = Signal(int)   # image_id

    def __init__(self, image_db: ImageDatabase, findings_db: FindingsDatabase,
                 patient_db: PatientDatabase, parent=None):
        super().__init__(parent)
        self.image_db    = image_db
        self.findings_db = findings_db
        self.patient_db  = patient_db
        self._all_rows   = []   # cached full dataset

        self.setWindowTitle("ClearScan — Scan Browser")
        self.setMinimumSize(1100, 700)
        self._build_ui()
        self.refresh()

    # ============================================================
    # UI
    # ============================================================
    def _build_ui(self):
        self.setStyleSheet(f"""
            QWidget {{
                background: {NAVY};
                font-family: 'Segoe UI', sans-serif;
            }}
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(16)

        # ── Header ───────────────────────────────────────────────
        header = QHBoxLayout()

        title = QLabel("Scan Browser")
        title.setStyleSheet(f"""
            color: {WHITE};
            font-size: 22px;
            font-weight: 700;
            font-family: 'Segoe UI Semibold', 'Segoe UI', sans-serif;
            letter-spacing: 0.5px;
        """)

        self.count_badge = QLabel("")
        self.count_badge.setStyleSheet(f"""
            color: {TEAL_L};
            font-size: 13px;
            background: transparent;
            padding-left: 12px;
        """)

        self.refresh_btn = QPushButton("↺  Refresh")
        self.refresh_btn.setFixedHeight(36)
        self.refresh_btn.setFixedWidth(100)
        self.refresh_btn.setStyleSheet(f"""
            QPushButton {{
                background: {TEAL};
                color: {WHITE};
                border: none;
                border-radius: 6px;
                font-size: 13px;
                font-weight: 600;
            }}
            QPushButton:hover {{ background: {TEAL_L}; }}
        """)
        self.refresh_btn.clicked.connect(self.refresh)

        header.addWidget(title)
        header.addWidget(self.count_badge)
        header.addStretch()
        header.addWidget(self.refresh_btn)

        # ── Filter bar ───────────────────────────────────────────
        self.filter_bar = FilterBar()
        self.filter_bar.filter_changed.connect(self._apply_filters)

        # ── Table ────────────────────────────────────────────────
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "ID", "Patient", "Filename", "Uploaded", "Status", "Diagnosed As", "Top AI Finding"
        ])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(False)
        self.table.setSortingEnabled(True)
        self.table.doubleClicked.connect(self._on_row_double_clicked)

        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.Stretch)
        hh.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(6, QHeaderView.ResizeToContents)

        self.table.setStyleSheet(f"""
            QTableWidget {{
                background: {STEEL};
                color: {WHITE};
                border: none;
                border-radius: 10px;
                gridline-color: transparent;
                font-size: 13px;
                outline: none;
            }}
            QTableWidget::item {{
                padding: 10px 14px;
                border-bottom: 1px solid {NAVY};
            }}
            QTableWidget::item:selected {{
                background: #1a3a5c;
                color: {WHITE};
            }}
            QHeaderView::section {{
                background: {NAVY};
                color: {GREY};
                font-size: 11px;
                font-weight: 600;
                letter-spacing: 1px;
                text-transform: uppercase;
                padding: 10px 14px;
                border: none;
                border-bottom: 2px solid {TEAL};
            }}
            QScrollBar:vertical {{
                background: {NAVY};
                width: 8px;
                border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{
                background: #2d4a6e;
                border-radius: 4px;
            }}
        """)

        # ── Hint label ───────────────────────────────────────────
        hint = QLabel("Double-click a row to open the scan in the viewer")
        hint.setStyleSheet(f"color:{GREY}; font-size:11px; background:transparent;")
        hint.setAlignment(Qt.AlignRight)

        root.addLayout(header)
        root.addWidget(self.filter_bar)
        root.addWidget(self.table, 1)
        root.addWidget(hint)

    # ============================================================
    # DATA LOADING
    # ============================================================
    def refresh(self):
        """Reload everything from DB and re-apply current filters."""
        images   = self.image_db.fetch_all_images()
        self._all_rows = []

        for img in images:
            diagnosis = self.findings_db.fetch_diagnosis(img.id)
            findings  = self.findings_db.fetch_findings_for_image(img.id)
            top_ai    = findings[0][0] if findings else "—"
            patient   = self.patient_db.fetch_patient_for_image(img.id)
            date_str  = str(img.uploaded_at)[:10]

            self._all_rows.append({
                "id":        img.id,
                "path":      img.file_path,
                "filename":  img.file_path.replace("\\", "/").split("/")[-1],
                "date":      date_str,
                "diagnosed": diagnosis is not None,
                "dx_label":  diagnosis["confirmed_label"] if diagnosis else "",
                "top_ai":    top_ai,
                "patient":   patient.full_name if patient else "—",
            })

        self._apply_filters()

    # ============================================================
    # FILTERING
    # ============================================================
    def _apply_filters(self):
        f = self.filter_bar.get_filters()
        filtered = []

        for row in self._all_rows:
            # Search
            if f["search"]:
                haystack = f"{row['id']} {row['filename']}".lower()
                if f["search"] not in haystack:
                    continue

            # Date range
            if row["date"] < f["date_from"] or row["date"] > f["date_to"]:
                continue

            # Status
            if f["status"] == "Diagnosed" and not row["diagnosed"]:
                continue
            if f["status"] == "Not diagnosed" and row["diagnosed"]:
                continue

            # Diagnosed as
            if f["dx_as"] != "Any condition":
                if row["dx_label"] != f["dx_as"]:
                    continue

            filtered.append(row)

        self._populate_table(filtered)

    # ============================================================
    # TABLE POPULATION
    # ============================================================
    def _populate_table(self, rows):
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)

        for row in rows:
            r = self.table.rowCount()
            self.table.insertRow(r)
            self.table.setRowHeight(r, 46)

            # ID
            id_item = QTableWidgetItem(str(row["id"]))
            id_item.setTextAlignment(Qt.AlignCenter)
            id_item.setForeground(QColor(TEAL_L))
            id_item.setData(Qt.UserRole, row["id"])
            self.table.setItem(r, 0, id_item)

            # Patient
            pt_item = QTableWidgetItem(row["patient"])
            pt_item.setForeground(QColor(WHITE if row["patient"] != "—" else GREY))
            self.table.setItem(r, 1, pt_item)

            # Filename
            fn_item = QTableWidgetItem(row["filename"])
            fn_item.setToolTip(row["path"])
            self.table.setItem(r, 2, fn_item)

            # Date
            date_item = QTableWidgetItem(row["date"])
            date_item.setTextAlignment(Qt.AlignCenter)
            date_item.setForeground(QColor(GREY))
            self.table.setItem(r, 3, date_item)

            # Status badge
            if row["diagnosed"]:
                status_text  = "✔  Diagnosed"
                status_color = GREEN
            else:
                status_text  = "○  Pending"
                status_color = AMBER

            status_item = QTableWidgetItem(status_text)
            status_item.setTextAlignment(Qt.AlignCenter)
            status_item.setForeground(QColor(status_color))
            status_item.setFont(QFont("Segoe UI", 11, QFont.Bold))
            self.table.setItem(r, 4, status_item)

            # Diagnosed as
            dx_item = QTableWidgetItem(row["dx_label"] or "—")
            dx_item.setTextAlignment(Qt.AlignCenter)
            dx_item.setForeground(QColor(WHITE if row["dx_label"] else GREY))
            self.table.setItem(r, 5, dx_item)

            # Top AI finding
            ai_item = QTableWidgetItem(row["top_ai"])
            ai_item.setTextAlignment(Qt.AlignCenter)
            ai_item.setForeground(QColor(GREY))
            self.table.setItem(r, 6, ai_item)

        self.table.setSortingEnabled(True)
        total   = len(self._all_rows)
        showing = len(rows)
        if showing == total:
            self.count_badge.setText(f"{total} scan{'s' if total != 1 else ''}")
        else:
            self.count_badge.setText(f"{showing} of {total} scans")

    # ============================================================
    # OPEN IMAGE
    # ============================================================
    def _on_row_double_clicked(self, index):
        id_item = self.table.item(index.row(), 0)
        if id_item:
            image_id = id_item.data(Qt.UserRole)
            self.open_image.emit(image_id)