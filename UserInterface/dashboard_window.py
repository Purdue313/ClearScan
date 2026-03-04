from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFrame, QListWidget, QListWidgetItem, QScrollArea,
    QFileDialog, QMessageBox, QSizePolicy, QAbstractItemView
)
from PySide6.QtCore import Qt, Signal, QMimeData
from PySide6.QtGui import QColor, QFont, QDragEnterEvent, QDropEvent

from Database.imageDatabase import ImageDatabase
from Database.findingsDatabase import FindingsDatabase
from Database.patientDatabase import PatientDatabase
from Database.userDatabase import UserDatabase
from Database.scheduleDatabase import ScheduleDatabase
from Source.Models.user_model import UserRecord, ROLE_SCHEDULER

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
PURPLE = "#6d28d9"
BLUE   = "#1d4ed8"
PINK   = "#be185d"


# ================================================================
# Stat Card
# ================================================================
class StatCard(QFrame):
    def __init__(self, title, value, accent=TEAL, parent=None):
        super().__init__(parent)
        self.setFixedHeight(90)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setStyleSheet(f"""
            QFrame {{
                background: {STEEL};
                border-radius: 10px;
                border-left: 4px solid {accent};
            }}
        """)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(2)

        self.val_lbl = QLabel(str(value))
        self.val_lbl.setStyleSheet(
            f"color:{WHITE}; font-size:26px; font-weight:700; background:transparent;"
        )
        ttl_lbl = QLabel(title)
        ttl_lbl.setStyleSheet(f"color:{GREY}; font-size:11px; background:transparent;")

        lay.addWidget(self.val_lbl)
        lay.addWidget(ttl_lbl)

    def set_value(self, v):
        self.val_lbl.setText(str(v))


# ================================================================
# Upload Drop Zone
# ================================================================
class UploadZone(QFrame):
    """Drag-and-drop zone that also has a click-to-browse button."""

    files_dropped = Signal(list)   # list of file paths

    def __init__(self, can_upload=True, parent=None):
        super().__init__(parent)
        self.can_upload = can_upload
        self.setAcceptDrops(can_upload)
        self.setFixedHeight(110)
        self._set_idle_style()

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(6)
        lay.setAlignment(Qt.AlignCenter)

        if can_upload:
            icon_lbl = QLabel("Drop X-ray images here")
            icon_lbl.setAlignment(Qt.AlignCenter)
            icon_lbl.setStyleSheet(
                f"color:{LGREY}; font-size:13px; background:transparent;"
            )

            self.browse_btn = QPushButton("Browse Files")
            self.browse_btn.setFixedSize(120, 30)
            self.browse_btn.setStyleSheet(f"""
                QPushButton {{
                    background:{TEAL}; color:{WHITE}; border:none;
                    border-radius:6px; font-size:12px; font-weight:600;
                }}
                QPushButton:hover {{ background:{TEAL_L}; }}
            """)
            self.browse_btn.clicked.connect(self._browse)

            lay.addWidget(icon_lbl)
            lay.addWidget(self.browse_btn, alignment=Qt.AlignCenter)
        else:
            no_lbl = QLabel("Upload not available for your role")
            no_lbl.setAlignment(Qt.AlignCenter)
            no_lbl.setStyleSheet(f"color:{GREY}; font-size:13px; background:transparent;")
            lay.addWidget(no_lbl)

    def _browse(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Images", "",
            "Images (*.png *.jpg *.jpeg)"
        )
        if paths:
            self.files_dropped.emit(paths)

    def _set_idle_style(self):
        self.setStyleSheet(f"""
            QFrame {{
                background: {NAVY};
                border: 2px dashed #2d4a6e;
                border-radius: 10px;
            }}
        """)

    def _set_hover_style(self):
        self.setStyleSheet(f"""
            QFrame {{
                background: #0d1f3a;
                border: 2px dashed {TEAL};
                border-radius: 10px;
            }}
        """)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            exts = (".png", ".jpg", ".jpeg")
            if any(u.toLocalFile().lower().endswith(exts) for u in event.mimeData().urls()):
                self._set_hover_style()
                event.acceptProposedAction()
                return
        event.ignore()

    def dragLeaveEvent(self, event):
        self._set_idle_style()

    def dropEvent(self, event: QDropEvent):
        self._set_idle_style()
        paths = [
            u.toLocalFile() for u in event.mimeData().urls()
            if u.toLocalFile().lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if paths:
            self.files_dropped.emit(paths)


# ================================================================
# Dashboard Window
# ================================================================
class DashboardWindow(QWidget):
    """
    Full dashboard with two-column layout:
      Left  - user info, stat cards, quick actions, upload zone
      Right - recent scans list (full height)

    Signals:
        open_scanner()         go to scanner view
        open_browser()         open scan browser filter window
        open_account_mgmt()    open account management (sysadmin)
        sign_out()             return to login
        open_scan_id(int)      open a specific scan in the scanner
        upload_files(list)     list of paths dropped/selected on dashboard
    """

    open_scanner      = Signal()
    open_browser      = Signal()
    open_account_mgmt = Signal()
    open_schedule     = Signal()
    sign_out          = Signal()
    open_scan_id      = Signal(int)
    upload_files      = Signal(list)

    def __init__(self, user: UserRecord,
                 image_db: ImageDatabase,
                 findings_db: FindingsDatabase,
                 patient_db: PatientDatabase,
                 user_db: UserDatabase,
                 schedule_db: ScheduleDatabase = None,
                 parent=None):
        super().__init__(parent)
        self.user        = user
        self.image_db    = image_db
        self.findings_db = findings_db
        self.patient_db  = patient_db
        self.user_db     = user_db
        self.schedule_db = schedule_db
        self._is_scheduler = (user.role == ROLE_SCHEDULER)

        self.setStyleSheet(f"background:{NAVY}; font-family:'Segoe UI',sans-serif;")
        self._build_ui()
        self.refresh()

    # ============================================================
    # BUILD UI
    # ============================================================
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_nav())

        # Two-column body
        body_frame = QWidget()
        body_frame.setStyleSheet(f"background:{NAVY};")
        body_h = QHBoxLayout(body_frame)
        body_h.setContentsMargins(24, 20, 24, 20)
        body_h.setSpacing(20)

        body_h.addWidget(self._build_left_panel(), 2)
        body_h.addWidget(self._build_right_panel(), 3)

        root.addWidget(body_frame, 1)

    # ============================================================
    # NAV BAR
    # ============================================================
    def _build_nav(self):
        nav = QFrame()
        nav.setFixedHeight(58)
        nav.setStyleSheet(f"background:{STEEL}; border-bottom:2px solid {TEAL};")
        lay = QHBoxLayout(nav)
        lay.setContentsMargins(24, 0, 24, 0)
        lay.setSpacing(0)

        brand = QLabel("ClearScan")
        brand.setStyleSheet(
            f"color:{TEAL_L}; font-size:20px; font-weight:700; "
            f"letter-spacing:1px; background:transparent;"
        )
        lay.addWidget(brand)
        lay.addStretch()

        # Role badge
        role_badge = QLabel(self.user.role_display)
        role_badge.setStyleSheet(f"""
            background:{TEAL}22; color:{TEAL_L};
            border:1px solid {TEAL}; border-radius:10px;
            padding:2px 12px; font-size:11px; font-weight:600;
        """)
        lay.addWidget(role_badge)
        lay.addSpacing(16)

        # Manage accounts (sysadmin)
        if self.user.has_permission("can_manage_accounts"):
            acct_btn = QPushButton("Manage Accounts")
            acct_btn.setFixedHeight(32)
            acct_btn.setStyleSheet(f"""
                QPushButton {{
                    background:{PURPLE}; color:{WHITE}; border:none;
                    border-radius:6px; font-size:12px; font-weight:600;
                    padding:0 16px;
                }}
                QPushButton:hover {{ background:#7c3aed; }}
            """)
            acct_btn.clicked.connect(self.open_account_mgmt)
            lay.addWidget(acct_btn)
            lay.addSpacing(10)

        # Account circle (initials)
        initials = (self.user.first_name[:1] + self.user.last_name[:1]).upper()
        acct_circle = QPushButton(initials)
        acct_circle.setFixedSize(36, 36)
        acct_circle.setToolTip(f"{self.user.full_name}\n{self.user.role_display}")
        acct_circle.setStyleSheet(f"""
            QPushButton {{
                background:{TEAL}; color:{WHITE}; border:none;
                border-radius:18px; font-size:13px; font-weight:700;
            }}
            QPushButton:hover {{ background:{TEAL_L}; }}
        """)
        lay.addWidget(acct_circle)
        lay.addSpacing(12)

        signout_btn = QPushButton("Sign Out")
        signout_btn.setFixedHeight(32)
        signout_btn.setStyleSheet(f"""
            QPushButton {{
                background:transparent; color:{GREY};
                border:1px solid #2d4a6e; border-radius:6px;
                font-size:12px; padding:0 14px;
            }}
            QPushButton:hover {{ color:{WHITE}; border-color:{RED}; }}
        """)
        signout_btn.clicked.connect(self.sign_out)
        lay.addWidget(signout_btn)

        return nav

    # ============================================================
    # LEFT PANEL
    # ============================================================
    def _build_left_panel(self):
        panel = QWidget()
        panel.setStyleSheet("background:transparent;")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(20)

        # -- User info card --
        lay.addWidget(self._build_user_card())

        # -- Stat cards --
        lay.addWidget(self._section_label("OVERVIEW"))
        cards_grid = QVBoxLayout()
        cards_grid.setSpacing(10)

        row1 = QHBoxLayout(); row1.setSpacing(10)
        self.card_total   = StatCard("Total Scans",       "--", TEAL)
        self.card_pending = StatCard("Pending Diagnosis",  "--", AMBER)
        row1.addWidget(self.card_total)
        row1.addWidget(self.card_pending)
        cards_grid.addLayout(row1)

        row2 = QHBoxLayout(); row2.setSpacing(10)
        self.card_patients = StatCard("Patients", "--", PURPLE)
        self.card_users    = StatCard("System Users", "--", PINK)
        row2.addWidget(self.card_patients)
        if self.user.has_permission("can_manage_accounts"):
            row2.addWidget(self.card_users)
        cards_grid.addLayout(row2)

        lay.addLayout(cards_grid)

        # -- Quick actions --
        lay.addWidget(self._section_label("QUICK ACTIONS"))
        lay.addLayout(self._build_actions())

        # -- Upload zone --
        lay.addWidget(self._section_label("UPLOAD"))
        can_upload = self.user.has_permission("can_upload")
        self.upload_zone = UploadZone(can_upload=can_upload)
        self.upload_zone.files_dropped.connect(self.upload_files)
        lay.addWidget(self.upload_zone)

        lay.addStretch()
        return panel

    def _build_user_card(self):
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background:{STEEL};
                border-radius:12px;
                border-left:4px solid {TEAL};
            }}
        """)
        lay = QHBoxLayout(card)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(14)

        # Avatar circle
        initials = (self.user.first_name[:1] + self.user.last_name[:1]).upper()
        avatar = QLabel(initials)
        avatar.setFixedSize(48, 48)
        avatar.setAlignment(Qt.AlignCenter)
        avatar.setStyleSheet(f"""
            background:{TEAL}; color:{WHITE}; border-radius:24px;
            font-size:18px; font-weight:700;
        """)

        info = QVBoxLayout()
        info.setSpacing(2)

        name_lbl = QLabel(self.user.full_name)
        name_lbl.setStyleSheet(
            f"color:{WHITE}; font-size:15px; font-weight:600; background:transparent;"
        )

        role_lbl = QLabel(self.user.role_display)
        role_lbl.setStyleSheet(f"color:{TEAL_L}; font-size:12px; background:transparent;")

        if self.user.email:
            email_lbl = QLabel(self.user.email)
            email_lbl.setStyleSheet(f"color:{GREY}; font-size:11px; background:transparent;")
            info.addWidget(email_lbl)

        last_lbl = QLabel(
            f"Last login: {self.user.last_login or 'First session'}"
        )
        last_lbl.setStyleSheet(f"color:{GREY}; font-size:11px; background:transparent;")

        info.addWidget(name_lbl)
        info.addWidget(role_lbl)
        info.addWidget(last_lbl)

        lay.addWidget(avatar)
        lay.addLayout(info, 1)
        return card

    def _build_actions(self):
        lay = QVBoxLayout()
        lay.setSpacing(8)

        if self.user.has_permission("can_upload"):
            b = self._action_btn("Upload New Scan", TEAL)
            b.clicked.connect(self.open_scanner)
            lay.addWidget(b)

        if self.user.has_permission("can_view_scans"):
            b2 = self._action_btn("Browse & Filter Scans", BLUE)
            b2.clicked.connect(self.open_browser)
            lay.addWidget(b2)

        b_sched = self._action_btn("View Schedule", "#0f766e")
        b_sched.clicked.connect(self.open_schedule)
        lay.addWidget(b_sched)

        if self.user.has_permission("can_manage_accounts"):
            b3 = self._action_btn("Manage User Accounts", PURPLE)
            b3.clicked.connect(self.open_account_mgmt)
            lay.addWidget(b3)

        return lay

    # ============================================================
    # RIGHT PANEL - recent scans list
    # ============================================================
    def _build_right_panel(self):
        if self._is_scheduler:
            return self._build_calendar_panel()
        return self._build_scans_panel()

    def _build_calendar_panel(self):
        """Right panel for schedulers: embedded monthly calendar."""
        from UserInterface.schedule_window import CalendarGrid
        panel = QWidget()
        panel.setStyleSheet("background:transparent;")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)

        hdr = QHBoxLayout()
        hdr.addWidget(self._section_label("APPOINTMENT CALENDAR"))
        hdr.addStretch()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setFixedHeight(28)
        refresh_btn.setStyleSheet(f"""
            QPushButton {{
                background:transparent; color:{GREY};
                border:1px solid #2d4a6e; border-radius:5px;
                font-size:11px; padding:0 12px;
            }}
            QPushButton:hover {{ color:{WHITE}; }}
        """)
        refresh_btn.clicked.connect(self.refresh)
        hdr.addWidget(refresh_btn)
        lay.addLayout(hdr)

        self.cal_grid = CalendarGrid()
        self.cal_grid.date_selected.connect(self._on_cal_date_selected)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border:none; background:transparent; }")
        scroll.setWidget(self.cal_grid)
        lay.addWidget(scroll, 1)

        # Selected day label
        self.cal_day_lbl = QLabel("Click a date to see appointments")
        self.cal_day_lbl.setStyleSheet(
            f"color:{GREY}; font-size:11px; background:transparent;"
        )
        lay.addWidget(self.cal_day_lbl)
        return panel

    def _on_cal_date_selected(self, date_str: str):
        if not self.schedule_db:
            return
        from datetime import datetime
        appts = self.schedule_db.fetch_for_date(date_str)
        patient_map = {p.id: p.full_name for p in self.patient_db.fetch_all_patients()}
        for a in appts:
            a.patient_name = patient_map.get(a.patient_id, f"Patient #{a.patient_id}")
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            label = dt.strftime("%A, %B %d").replace(" 0", " ")
        except Exception:
            label = date_str
        if appts:
            lines = [f"{a.display_time}  {a.patient_name}  ({a.status})" for a in appts]
            self.cal_day_lbl.setText(f"{label}: " + "  |  ".join(lines))
        else:
            self.cal_day_lbl.setText(f"{label}: No appointments scheduled")
        self.cal_day_lbl.setStyleSheet(
            f"color:{TEAL_L}; font-size:11px; background:transparent;"
        )

    def _build_scans_panel(self):
        """Right panel for clinical staff: recent scans list."""
        panel = QWidget()
        panel.setStyleSheet("background:transparent;")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)

        hdr = QHBoxLayout()
        hdr.addWidget(self._section_label("RECENT SCANS"))
        hdr.addStretch()

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setFixedHeight(28)
        refresh_btn.setStyleSheet(f"""
            QPushButton {{
                background:transparent; color:{GREY};
                border:1px solid #2d4a6e; border-radius:5px;
                font-size:11px; padding:0 12px;
            }}
            QPushButton:hover {{ color:{WHITE}; }}
        """)
        refresh_btn.clicked.connect(self.refresh)
        hdr.addWidget(refresh_btn)

        hint = QLabel("  Double-click to open")
        hint.setStyleSheet(f"color:#2d4a6e; font-size:11px; background:transparent;")
        hdr.addWidget(hint)

        lay.addLayout(hdr)

        self.recent_list = QListWidget()
        self.recent_list.setStyleSheet(f"""
            QListWidget {{
                background:{STEEL}; color:{WHITE};
                border:none; border-radius:10px;
                font-size:13px; outline:none;
            }}
            QListWidget::item {{
                padding:0px;
                border-bottom:1px solid {NAVY};
            }}
            QListWidget::item:selected {{ background:#1a3a5c; }}
            QListWidget::item:hover    {{ background:{STEEL2}; }}
        """)
        self.recent_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.recent_list.itemDoubleClicked.connect(self._on_double_click)

        lay.addWidget(self.recent_list, 1)
        return panel

    # ============================================================
    # HELPERS
    # ============================================================
    def _section_label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet(f"""
            color:{GREY}; font-size:10px; font-weight:700;
            letter-spacing:2px; background:transparent;
        """)
        return lbl

    def _action_btn(self, text, color):
        b = QPushButton(text)
        b.setFixedHeight(38)
        b.setStyleSheet(f"""
            QPushButton {{
                background:{color}; color:{WHITE}; border:none;
                border-radius:8px; font-size:13px; font-weight:600;
                padding:0 16px; text-align:left; padding-left:16px;
            }}
            QPushButton:hover {{ background:{color}dd; }}
        """)
        return b

    # ============================================================
    # REFRESH DATA
    # ============================================================
    def refresh(self):
        images   = self.image_db.fetch_all_images()
        patients = self.patient_db.fetch_all_patients()
        users    = self.user_db.fetch_all_users()
        pending  = sum(
            1 for img in images if not self.findings_db.is_diagnosed(img.id)
        )

        self.card_total.set_value(len(images))
        self.card_pending.set_value(pending)
        self.card_patients.set_value(len(patients))
        self.card_users.set_value(len(users))

        # Scheduler sees calendar, not scan list
        if self._is_scheduler:
            if self.schedule_db and hasattr(self, 'cal_grid'):
                import calendar as _cal
                y = self.cal_grid.current_year
                m = self.cal_grid.current_month
                last = _cal.monthrange(y, m)[1]
                appts = self.schedule_db.fetch_range(
                    f"{y}-{m:02d}-01", f"{y}-{m:02d}-{last:02d}"
                )
                self.cal_grid.set_appointments(appts)
            return

        self.recent_list.clear()
        for img in list(reversed(images))[:30]:
            patient   = self.patient_db.fetch_patient_for_image(img.id)
            diagnosed = self.findings_db.is_diagnosed(img.id)
            findings  = self.findings_db.fetch_findings_for_image(img.id)
            top       = findings[0][0] if findings else "No analysis"
            pt_name   = patient.full_name if patient else "No patient linked"
            filename  = img.file_path.replace("\\", "/").split("/")[-1]
            date_str  = str(img.uploaded_at)[:10]

            row_widget = self._make_scan_row(
                img.id, filename, pt_name, date_str, top, diagnosed
            )
            item = QListWidgetItem()
            item.setData(Qt.UserRole, img.id)
            item.setSizeHint(row_widget.sizeHint())
            self.recent_list.addItem(item)
            self.recent_list.setItemWidget(item, row_widget)

    def _make_scan_row(self, image_id, filename, patient, date, top_finding, diagnosed):
        row = QWidget()
        row.setStyleSheet("background:transparent;")
        lay = QHBoxLayout(row)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(12)

        # Status dot
        dot = QLabel()
        dot.setFixedSize(10, 10)
        dot.setStyleSheet(f"""
            background:{'#16a34a' if diagnosed else '#d97706'};
            border-radius:5px;
        """)

        # Main info
        info = QVBoxLayout()
        info.setSpacing(2)

        fn_lbl = QLabel(filename)
        fn_lbl.setStyleSheet(f"color:{WHITE}; font-size:13px; font-weight:600; background:transparent;")

        sub_lbl = QLabel(f"{patient}  |  {top_finding}")
        sub_lbl.setStyleSheet(f"color:{GREY}; font-size:11px; background:transparent;")

        info.addWidget(fn_lbl)
        info.addWidget(sub_lbl)

        # Right side
        right = QVBoxLayout()
        right.setSpacing(2)
        right.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        date_lbl = QLabel(date)
        date_lbl.setAlignment(Qt.AlignRight)
        date_lbl.setStyleSheet(f"color:{GREY}; font-size:11px; background:transparent;")

        status_lbl = QLabel("Diagnosed" if diagnosed else "Pending")
        status_lbl.setAlignment(Qt.AlignRight)
        status_lbl.setStyleSheet(
            f"color:{'#16a34a' if diagnosed else '#d97706'}; "
            f"font-size:11px; font-weight:600; background:transparent;"
        )

        right.addWidget(date_lbl)
        right.addWidget(status_lbl)

        lay.addWidget(dot, alignment=Qt.AlignVCenter)
        lay.addLayout(info, 1)
        lay.addLayout(right)
        return row

    # ============================================================
    # DOUBLE-CLICK
    # ============================================================
    def _on_double_click(self, item: QListWidgetItem):
        image_id = item.data(Qt.UserRole)
        if image_id is not None:
            self.open_scan_id.emit(image_id)