from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFrame, QScrollArea, QSizePolicy, QMessageBox, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView
)
from PySide6.QtCore import Qt, QDate, Signal, QSize
from PySide6.QtGui import QColor, QFont, QCursor

import calendar
from datetime import date, timedelta

from Database.scheduleDatabase import ScheduleDatabase
from Database.patientDatabase import PatientDatabase
from Database.userDatabase import UserDatabase
from Source.Models.appointment_model import (
    AppointmentRecord, ALL_STATUSES, STATUS_COLORS,
    STATUS_SCHEDULED, STATUS_CONFIRMED, STATUS_CANCELLED,
    STATUS_COMPLETED, STATUS_NO_SHOW
)
from UserInterface.booking_dialog import BookingDialog

NAVY   = "#0a1628"
STEEL  = "#1c3150"
STEEL2 = "#243d5e"
TEAL   = "#0d9488"
TEAL_L = "#14b8a8"
WHITE  = "#ffffff"
GREY   = "#64748b"
LGREY  = "#94a3b8"
GREEN  = "#16a34a"
AMBER  = "#d97706"
RED    = "#dc2626"
TODAY_BG = "#0d2035"

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ================================================================
# Day Cell - one square in the monthly calendar grid
# ================================================================
class DayCell(QFrame):
    clicked = Signal(str)   # emits date string "YYYY-MM-DD"

    def __init__(self, date_str: str, day_num: int,
                 is_today: bool, is_other_month: bool,
                 appointments: list, parent=None):
        super().__init__(parent)
        self.date_str = date_str
        self.setFixedHeight(100)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setCursor(QCursor(Qt.PointingHandCursor))

        bg = TODAY_BG if is_today else STEEL
        border = f"2px solid {TEAL}" if is_today else "1px solid #1a3050"
        self.setStyleSheet(f"""
            QFrame {{
                background:{bg};
                border:{border};
                border-radius:6px;
            }}
            QFrame:hover {{ background:{STEEL2}; }}
        """)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(2)

        # Day number
        day_lbl = QLabel(str(day_num))
        day_lbl.setStyleSheet(
            f"color:{'#2d4a6e' if is_other_month else (TEAL_L if is_today else WHITE)};"
            f"font-size:13px; font-weight:{'700' if is_today else '400'};"
            f"background:transparent;"
        )
        lay.addWidget(day_lbl)

        # Appointment pills (up to 3)
        for appt in appointments[:3]:
            pill = QLabel(f"  {appt.display_time}  {appt.appt_type}")
            pill.setFixedHeight(18)
            pill.setStyleSheet(f"""
                background:{appt.status_color};
                color:white;
                border-radius:3px;
                font-size:10px;
                padding:0 3px;
            """)
            pill.setToolTip(
                f"{appt.display_time} - {appt.appt_type}\n"
                f"Status: {appt.status}"
            )
            lay.addWidget(pill)

        if len(appointments) > 3:
            more = QLabel(f"  +{len(appointments)-3} more")
            more.setStyleSheet(f"color:{GREY}; font-size:10px; background:transparent;")
            lay.addWidget(more)

        lay.addStretch()

    def mousePressEvent(self, event):
        self.clicked.emit(self.date_str)


# ================================================================
# Calendar Grid Widget
# ================================================================
class CalendarGrid(QWidget):
    date_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{NAVY};")
        self._year  = date.today().year
        self._month = date.today().month
        self._appointments = {}   # date_str -> list[AppointmentRecord]
        self._build_layout()

    def _build_layout(self):
        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(0, 0, 0, 0)
        self._root.setSpacing(8)

        # Month navigation
        nav = QHBoxLayout()
        self.prev_btn = QPushButton("< Prev")
        self.prev_btn.setFixedSize(70, 30)
        self.prev_btn.setStyleSheet(f"""
            QPushButton {{
                background:transparent; color:{LGREY};
                border:1px solid #2d4a6e; border-radius:5px; font-size:12px;
            }}
            QPushButton:hover {{ color:{WHITE}; }}
        """)
        self.prev_btn.clicked.connect(self._prev_month)

        self.month_lbl = QLabel()
        self.month_lbl.setAlignment(Qt.AlignCenter)
        self.month_lbl.setStyleSheet(
            f"color:{WHITE}; font-size:16px; font-weight:700; background:transparent;"
        )

        self.next_btn = QPushButton("Next >")
        self.next_btn.setFixedSize(70, 30)
        self.next_btn.setStyleSheet(f"""
            QPushButton {{
                background:transparent; color:{LGREY};
                border:1px solid #2d4a6e; border-radius:5px; font-size:12px;
            }}
            QPushButton:hover {{ color:{WHITE}; }}
        """)
        self.next_btn.clicked.connect(self._next_month)

        today_btn = QPushButton("Today")
        today_btn.setFixedSize(60, 30)
        today_btn.setStyleSheet(f"""
            QPushButton {{
                background:{TEAL}; color:{WHITE}; border:none;
                border-radius:5px; font-size:12px;
            }}
            QPushButton:hover {{ background:{TEAL_L}; }}
        """)
        today_btn.clicked.connect(self._go_today)

        nav.addWidget(self.prev_btn)
        nav.addStretch()
        nav.addWidget(self.month_lbl)
        nav.addStretch()
        nav.addWidget(today_btn)
        nav.addWidget(self.next_btn)
        self._root.addLayout(nav)

        # Day name headers
        header_row = QHBoxLayout()
        header_row.setSpacing(4)
        for name in DAY_NAMES:
            lbl = QLabel(name)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setFixedHeight(24)
            lbl.setStyleSheet(
                f"color:{GREY}; font-size:11px; font-weight:700; "
                f"letter-spacing:1px; background:transparent;"
            )
            header_row.addWidget(lbl)
        self._root.addLayout(header_row)

        # Grid placeholder
        self._grid_widget = QWidget()
        self._grid_widget.setStyleSheet("background:transparent;")
        self._grid_layout = QVBoxLayout(self._grid_widget)
        self._grid_layout.setContentsMargins(0, 0, 0, 0)
        self._grid_layout.setSpacing(4)
        self._root.addWidget(self._grid_widget)

    def set_appointments(self, appointments: list):
        self._appointments = {}
        for a in appointments:
            self._appointments.setdefault(a.appt_date, []).append(a)
        self._render_grid()

    def _render_grid(self):
        # Clear existing grid rows
        while self._grid_layout.count():
            item = self._grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        today      = date.today()
        cal        = calendar.Calendar(firstweekday=0)
        month_days = cal.monthdatescalendar(self._year, self._month)

        self.month_lbl.setText(
            f"{calendar.month_name[self._month]} {self._year}"
        )

        for week in month_days:
            row = QHBoxLayout()
            row.setSpacing(4)
            for d in week:
                date_str     = d.strftime("%Y-%m-%d")
                appts        = self._appointments.get(date_str, [])
                is_today     = (d == today)
                is_other_mon = (d.month != self._month)

                cell = DayCell(date_str, d.day, is_today, is_other_mon, appts)
                cell.clicked.connect(self.date_selected)
                row.addWidget(cell)

            row_widget = QWidget()
            row_widget.setStyleSheet("background:transparent;")
            row_widget.setLayout(row)
            self._grid_layout.addWidget(row_widget)

    def _prev_month(self):
        if self._month == 1:
            self._month = 12
            self._year -= 1
        else:
            self._month -= 1
        self._render_grid()

    def _next_month(self):
        if self._month == 12:
            self._month = 1
            self._year += 1
        else:
            self._month += 1
        self._render_grid()

    def _go_today(self):
        today        = date.today()
        self._year   = today.year
        self._month  = today.month
        self._render_grid()

    @property
    def current_year(self):
        return self._year

    @property
    def current_month(self):
        return self._month


# ================================================================
# Day View - list of appointments for selected date
# ================================================================
class DayView(QFrame):
    edit_requested   = Signal(int)    # appt id
    cancel_requested = Signal(int)    # appt id

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{STEEL}; border-radius:10px;")
        self._appts = []
        self._date_str = ""

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(10)

        self.title_lbl = QLabel("Select a date")
        self.title_lbl.setStyleSheet(
            f"color:{WHITE}; font-size:15px; font-weight:700; background:transparent;"
        )
        lay.addWidget(self.title_lbl)

        # Appointment table
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Time", "Patient", "Tech", "Type", "Status"])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.setStyleSheet(f"""
            QTableWidget {{
                background:{STEEL}; color:{WHITE};
                border:none; font-size:12px; outline:none;
            }}
            QTableWidget::item {{
                padding:8px 10px;
                border-bottom:1px solid {NAVY};
            }}
            QTableWidget::item:selected {{ background:#1a3a5c; }}
            QHeaderView::section {{
                background:{NAVY}; color:{GREY};
                font-size:10px; font-weight:700; letter-spacing:1px;
                padding:8px 10px; border:none;
                border-bottom:1px solid {TEAL};
            }}
        """)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        hh.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(4, QHeaderView.ResizeToContents)

        self.empty_lbl = QLabel("No appointments on this date.")
        self.empty_lbl.setAlignment(Qt.AlignCenter)
        self.empty_lbl.setStyleSheet(f"color:{GREY}; font-size:13px; background:transparent;")
        self.empty_lbl.setVisible(False)

        btn_row = QHBoxLayout()
        self.edit_btn = QPushButton("Edit")
        self.edit_btn.setFixedHeight(30)
        self.edit_btn.setEnabled(False)
        self.edit_btn.setStyleSheet(f"""
            QPushButton {{
                background:#1d4ed8; color:{WHITE}; border:none;
                border-radius:5px; font-size:12px; padding:0 14px;
            }}
            QPushButton:hover    {{ background:#1e40af; }}
            QPushButton:disabled {{ background:#2d4a6e; color:{GREY}; }}
        """)
        self.edit_btn.clicked.connect(self._on_edit)

        self.cancel_btn = QPushButton("Cancel Appt.")
        self.cancel_btn.setFixedHeight(30)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background:transparent; color:{RED};
                border:1px solid {RED}; border-radius:5px;
                font-size:12px; padding:0 14px;
            }}
            QPushButton:hover    {{ background:{RED}; color:{WHITE}; }}
            QPushButton:disabled {{ border-color:#2d4a6e; color:{GREY}; }}
        """)
        self.cancel_btn.clicked.connect(self._on_cancel)

        self.table.itemSelectionChanged.connect(self._on_selection_changed)

        btn_row.addWidget(self.edit_btn)
        btn_row.addWidget(self.cancel_btn)
        btn_row.addStretch()

        lay.addWidget(self.table, 1)
        lay.addWidget(self.empty_lbl)
        lay.addLayout(btn_row)

    def load_date(self, date_str: str, appointments: list):
        self._date_str = date_str
        self._appts    = appointments

        # Friendly date title
        try:
            from datetime import datetime
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            self.title_lbl.setText(dt.strftime("%A, %B %-d %Y"))
        except Exception:
            self.title_lbl.setText(date_str)

        self.table.setRowCount(0)
        for appt in sorted(appointments, key=lambda a: a.appt_time):
            r = self.table.rowCount()
            self.table.insertRow(r)
            self.table.setRowHeight(r, 40)

            time_item = QTableWidgetItem(appt.display_time)
            time_item.setData(Qt.UserRole, appt.id)

            patient_item = QTableWidgetItem(appt.patient_name or f"Patient #{appt.patient_id}")
            tech_item    = QTableWidgetItem(appt.tech_name or "Unassigned")
            type_item    = QTableWidgetItem(appt.appt_type)
            status_item  = QTableWidgetItem(appt.status)
            status_item.setForeground(QColor(appt.status_color))
            if not appt.tech_name:
                tech_item.setForeground(QColor(GREY))

            for item in [time_item, patient_item, tech_item, type_item, status_item]:
                item.setTextAlignment(Qt.AlignVCenter | Qt.AlignLeft)

            self.table.setItem(r, 0, time_item)
            self.table.setItem(r, 1, patient_item)
            self.table.setItem(r, 2, tech_item)
            self.table.setItem(r, 3, type_item)
            self.table.setItem(r, 4, status_item)

        has = len(appointments) > 0
        self.table.setVisible(has)
        self.empty_lbl.setVisible(not has)
        self._update_buttons()

    def _on_selection_changed(self):
        self._update_buttons()

    def _update_buttons(self):
        has_sel = bool(self.table.selectedItems())
        self.edit_btn.setEnabled(has_sel)
        self.cancel_btn.setEnabled(has_sel)

    def _selected_id(self):
        row = self.table.currentRow()
        if row < 0:
            return None
        item = self.table.item(row, 0)
        return item.data(Qt.UserRole) if item else None

    def _on_edit(self):
        appt_id = self._selected_id()
        if appt_id:
            self.edit_requested.emit(appt_id)

    def _on_cancel(self):
        appt_id = self._selected_id()
        if appt_id:
            self.cancel_requested.emit(appt_id)


# ================================================================
# Main Schedule Window
# ================================================================
class ScheduleWindow(QWidget):
    """
    Full scheduling interface:
      Left  - monthly calendar grid
      Right - day view (appointments for selected date)
    """

    def __init__(self, schedule_db: ScheduleDatabase,
                 patient_db: PatientDatabase,
                 user_db: UserDatabase,
                 current_user=None,
                 parent=None):
        super().__init__(parent)
        self.schedule_db  = schedule_db
        self.patient_db   = patient_db
        self.user_db      = user_db
        self.current_user = current_user
        self._selected_date = date.today().strftime("%Y-%m-%d")
        self._all_appts     = []

        self.setWindowTitle("ClearScan - Schedule")
        self.setMinimumSize(1100, 700)
        self.setStyleSheet(f"background:{NAVY}; font-family:'Segoe UI',sans-serif;")

        self._build_ui()
        self.refresh()

    # ============================================================
    # UI
    # ============================================================
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Nav bar
        root.addWidget(self._build_nav())

        # Body: calendar left, day view right
        body = QWidget()
        body.setStyleSheet(f"background:{NAVY};")
        body_lay = QHBoxLayout(body)
        body_lay.setContentsMargins(20, 16, 20, 16)
        body_lay.setSpacing(16)

        # Left: calendar in a scroll area
        left = QScrollArea()
        left.setWidgetResizable(True)
        left.setStyleSheet("QScrollArea { border:none; background:transparent; }")
        self.calendar = CalendarGrid()
        self.calendar.date_selected.connect(self._on_date_selected)
        left.setWidget(self.calendar)

        # Right: day view
        self.day_view = DayView()
        self.day_view.setMinimumWidth(340)
        self.day_view.edit_requested.connect(self._edit_appointment)
        self.day_view.cancel_requested.connect(self._cancel_appointment)

        body_lay.addWidget(left, 3)
        body_lay.addWidget(self.day_view, 2)

        root.addWidget(body, 1)

    def _build_nav(self):
        nav = QFrame()
        nav.setFixedHeight(54)
        nav.setStyleSheet(f"background:{STEEL}; border-bottom:2px solid {TEAL};")
        lay = QHBoxLayout(nav)
        lay.setContentsMargins(20, 0, 20, 0)

        title = QLabel("Appointment Schedule")
        title.setStyleSheet(
            f"color:{WHITE}; font-size:17px; font-weight:700; background:transparent;"
        )
        lay.addWidget(title)
        lay.addStretch()

        can_book = (self.current_user is None or
                    self.current_user.has_permission("can_manage_schedule") or
                    self.current_user.has_permission("can_upload"))

        if can_book:
            book_btn = QPushButton("+ Book Appointment")
            book_btn.setFixedHeight(34)
            book_btn.setStyleSheet(f"""
                QPushButton {{
                    background:{TEAL}; color:{WHITE}; border:none;
                    border-radius:6px; font-size:13px; font-weight:600;
                    padding:0 18px;
                }}
                QPushButton:hover {{ background:{TEAL_L}; }}
            """)
            book_btn.clicked.connect(self._book_appointment)
            lay.addWidget(book_btn)
            lay.addSpacing(10)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setFixedHeight(34)
        refresh_btn.setStyleSheet(f"""
            QPushButton {{
                background:transparent; color:{GREY};
                border:1px solid #2d4a6e; border-radius:6px;
                font-size:12px; padding:0 14px;
            }}
            QPushButton:hover {{ color:{WHITE}; }}
        """)
        refresh_btn.clicked.connect(self.refresh)
        lay.addWidget(refresh_btn)
        return nav

    # ============================================================
    # REFRESH
    # ============================================================
    def refresh(self):
        # Fetch all appointments for the visible month (+/- 1 month buffer)
        y = self.calendar.current_year
        m = self.calendar.current_month
        start = f"{y}-{m:02d}-01"
        # Last day of month
        import calendar as cal
        last_day = cal.monthrange(y, m)[1]
        end   = f"{y}-{m:02d}-{last_day:02d}"

        self._all_appts = self.schedule_db.fetch_range(start, end)

        # Attach patient and tech names
        patient_map = {p.id: p.full_name for p in self.patient_db.fetch_all_patients()}
        tech_map    = {u.id: u.full_name for u in self.user_db.fetch_all_users()}
        for appt in self._all_appts:
            appt.patient_name = patient_map.get(appt.patient_id, f"Patient #{appt.patient_id}")
            appt.tech_name    = tech_map.get(appt.tech_user_id, "") if appt.tech_user_id else ""

        self.calendar.set_appointments(self._all_appts)
        self._refresh_day_view()

    def _refresh_day_view(self):
        day_appts = [a for a in self._all_appts if a.appt_date == self._selected_date]
        # Also fetch from DB in case selected date is outside current month view
        if not day_appts:
            day_appts = self.schedule_db.fetch_for_date(self._selected_date)
            patient_map = {p.id: p.full_name for p in self.patient_db.fetch_all_patients()}
            tech_map    = {u.id: u.full_name for u in self.user_db.fetch_all_users()}
            for appt in day_appts:
                appt.patient_name = patient_map.get(appt.patient_id, f"Patient #{appt.patient_id}")
                appt.tech_name    = tech_map.get(appt.tech_user_id, "") if appt.tech_user_id else ""

        self.day_view.load_date(self._selected_date, day_appts)

    # ============================================================
    # SLOT HANDLERS
    # ============================================================
    def _on_date_selected(self, date_str: str):
        self._selected_date = date_str
        self._refresh_day_view()

    def _book_appointment(self):
        dlg = BookingDialog(
            self.schedule_db, self.patient_db, self.user_db,
            preselect_date = self._selected_date,
            created_by     = self.current_user.username if self.current_user else "",
            parent         = self
        )
        if dlg.exec():
            self.refresh()

    def _edit_appointment(self, appt_id: int):
        appt = self.schedule_db.fetch_by_id(appt_id)
        if not appt:
            return
        dlg = BookingDialog(
            self.schedule_db, self.patient_db, self.user_db,
            existing   = appt,
            created_by = self.current_user.username if self.current_user else "",
            parent     = self
        )
        if dlg.exec():
            self.refresh()

    def _cancel_appointment(self, appt_id: int):
        reply = QMessageBox.question(
            self, "Cancel Appointment",
            "Mark this appointment as Cancelled?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.schedule_db.update_status(appt_id, STATUS_CANCELLED)
            self.refresh()