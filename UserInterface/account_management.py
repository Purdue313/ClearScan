from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QLineEdit, QComboBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox, QDialog, QFormLayout,
    QCheckBox, QFrame, QAbstractItemView
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont

from Database.userDatabase import UserDatabase
from Source.Models.user_model import UserRecord, ALL_ROLES, ROLE_DISPLAY

NAVY   = "#0a1628"
STEEL  = "#1c3150"
TEAL   = "#0d9488"
TEAL_L = "#14b8a8"
WHITE  = "#ffffff"
GREY   = "#64748b"
GREEN  = "#16a34a"
RED    = "#dc2626"
AMBER  = "#d97706"


class UserFormDialog(QDialog):
    """Create or edit a user account."""

    def __init__(self, user_db: UserDatabase, user: UserRecord = None, parent=None):
        super().__init__(parent)
        self.user_db  = user_db
        self.existing = user
        self.setWindowTitle("Edit User" if user else "Create User")
        self.setMinimumWidth(420)
        self.setModal(True)
        self.setStyleSheet(f"""
            QDialog   {{ background:{NAVY}; }}
            QLabel    {{ color:{WHITE}; background:transparent; font-size:13px; }}
            QLineEdit, QComboBox {{
                background:{STEEL}; color:{WHITE};
                border:1px solid #2d4a6e; border-radius:6px;
                padding:6px 10px; font-size:13px;
            }}
            QLineEdit:focus, QComboBox:focus {{ border-color:{TEAL}; }}
            QComboBox QAbstractItemView {{
                background:{STEEL}; color:{WHITE};
                selection-background-color:{TEAL};
            }}
        """)
        self._build_ui()
        if user:
            self._populate(user)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(16)

        form = QFormLayout()
        form.setSpacing(12)
        form.setLabelAlignment(Qt.AlignRight)

        self.fn_edit   = QLineEdit()
        self.ln_edit   = QLineEdit()
        self.user_edit = QLineEdit()
        self.email_edit= QLineEdit()
        self.pw_edit   = QLineEdit()
        self.pw_edit.setEchoMode(QLineEdit.Password)
        self.pw_edit.setPlaceholderText("Leave blank to keep existing" if self.existing else "Required")

        self.role_combo = QComboBox()
        for role in ALL_ROLES:
            self.role_combo.addItem(ROLE_DISPLAY[role], role)

        self.active_check = QCheckBox("Account active")
        self.active_check.setChecked(True)
        self.active_check.setStyleSheet(f"color:{WHITE}; background:transparent;")

        form.addRow("First name *", self.fn_edit)
        form.addRow("Last name *",  self.ln_edit)
        form.addRow("Username *",   self.user_edit)
        form.addRow("Email",        self.email_edit)
        form.addRow("Password",     self.pw_edit)
        form.addRow("Role *",       self.role_combo)
        form.addRow("",             self.active_check)

        btn_row = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFixedHeight(36)
        cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background:transparent; color:{GREY};
                border:1px solid #2d4a6e; border-radius:6px;
                font-size:13px; padding:0 16px;
            }}
            QPushButton:hover {{ color:{WHITE}; }}
        """)
        cancel_btn.clicked.connect(self.reject)

        save_btn = QPushButton("Save" if self.existing else "Create")
        save_btn.setFixedHeight(36)
        save_btn.setStyleSheet(f"""
            QPushButton {{
                background:{TEAL}; color:{WHITE}; border:none;
                border-radius:6px; font-size:13px; font-weight:600; padding:0 20px;
            }}
            QPushButton:hover {{ background:{TEAL_L}; }}
        """)
        save_btn.clicked.connect(self._save)

        btn_row.addWidget(cancel_btn)
        btn_row.addStretch()
        btn_row.addWidget(save_btn)

        root.addLayout(form)
        root.addLayout(btn_row)

    def _populate(self, user: UserRecord):
        self.fn_edit.setText(user.first_name)
        self.ln_edit.setText(user.last_name)
        self.user_edit.setText(user.username)
        self.email_edit.setText(user.email)
        idx = self.role_combo.findData(user.role)
        if idx >= 0:
            self.role_combo.setCurrentIndex(idx)
        self.active_check.setChecked(user.is_active)

    def _save(self):
        first    = self.fn_edit.text().strip()
        last     = self.ln_edit.text().strip()
        username = self.user_edit.text().strip()
        email    = self.email_edit.text().strip()
        password = self.pw_edit.text()
        role     = self.role_combo.currentData()
        active   = self.active_check.isChecked()

        if not first or not last or not username:
            QMessageBox.warning(self, "Missing Fields",
                                "First name, last name, and username are required.")
            return

        if self.existing:
            # Check username change doesn't clash with another account
            if username != self.existing.username and self.user_db.username_exists(username):
                QMessageBox.warning(self, "Username Taken",
                                    f"'{username}' is already in use by another account.")
                return
            # Update existing
            self.existing.username   = username
            self.existing.first_name = first
            self.existing.last_name  = last
            self.existing.email      = email
            self.existing.role       = role
            self.existing.is_active  = active
            self.user_db.update_user(self.existing)
            if password:
                self.user_db.change_password(self.existing.id, password)
        else:
            # Create new
            if not password:
                QMessageBox.warning(self, "Missing Password",
                                    "A password is required for new accounts.")
                return
            if self.user_db.username_exists(username):
                QMessageBox.warning(self, "Username Taken",
                                    f"'{username}' is already in use.")
                return
            self.user_db.create_user(
                username=username, password=password, role=role,
                first_name=first, last_name=last, email=email
            )

        self.accept()


class AccountManagementWindow(QWidget):
    """
    Sysadmin-only screen for managing user accounts.
    Lists all users, allows create / edit / deactivate.
    """

    def __init__(self, user_db: UserDatabase, current_user: UserRecord, parent=None):
        super().__init__(parent)
        self.user_db      = user_db
        self.current_user = current_user
        self.setWindowTitle("ClearScan - Account Management")
        self.setMinimumSize(900, 600)
        self.setStyleSheet(f"background:{NAVY}; font-family:'Segoe UI',sans-serif;")
        self._build_ui()
        self.refresh()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(16)

        # Header row
        hdr = QHBoxLayout()
        title = QLabel("Account Management")
        title.setStyleSheet(f"color:{WHITE}; font-size:20px; font-weight:700; background:transparent;")

        self.create_btn = QPushButton("+ New User")
        self.create_btn.setFixedHeight(36)
        self.create_btn.setStyleSheet(f"""
            QPushButton {{
                background:{TEAL}; color:{WHITE}; border:none;
                border-radius:6px; font-size:13px; font-weight:600; padding:0 16px;
            }}
            QPushButton:hover {{ background:{TEAL_L}; }}
        """)
        self.create_btn.clicked.connect(self._create_user)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setFixedHeight(36)
        self.refresh_btn.setStyleSheet(f"""
            QPushButton {{
                background:transparent; color:{GREY};
                border:1px solid #2d4a6e; border-radius:6px;
                font-size:13px; padding:0 14px;
            }}
            QPushButton:hover {{ color:{WHITE}; }}
        """)
        self.refresh_btn.clicked.connect(self.refresh)

        hdr.addWidget(title)
        hdr.addStretch()
        hdr.addWidget(self.refresh_btn)
        hdr.addWidget(self.create_btn)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "ID", "Username", "Full Name", "Role", "Email", "Last Login", "Status"
        ])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.setSortingEnabled(True)
        self.table.setStyleSheet(f"""
            QTableWidget {{
                background:{STEEL}; color:{WHITE};
                border:none; border-radius:10px;
                font-size:13px; outline:none;
            }}
            QTableWidget::item {{
                padding:10px 14px;
                border-bottom:1px solid {NAVY};
            }}
            QTableWidget::item:selected {{ background:#1a3a5c; }}
            QHeaderView::section {{
                background:{NAVY}; color:{GREY};
                font-size:11px; font-weight:600;
                letter-spacing:1px; padding:10px 14px;
                border:none; border-bottom:2px solid {TEAL};
            }}
        """)

        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.Stretch)
        hh.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(4, QHeaderView.Stretch)
        hh.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(6, QHeaderView.ResizeToContents)

        self.table.doubleClicked.connect(self._edit_user)

        # Action buttons
        act_row = QHBoxLayout()
        self.edit_btn = QPushButton("Edit Selected")
        self.edit_btn.setFixedHeight(34)
        self.edit_btn.setStyleSheet(f"""
            QPushButton {{
                background:#1d4ed8; color:{WHITE}; border:none;
                border-radius:6px; font-size:12px; padding:0 14px;
            }}
            QPushButton:hover {{ background:#1e40af; }}
        """)
        self.edit_btn.clicked.connect(self._edit_user)

        self.deact_btn = QPushButton("Deactivate Selected")
        self.deact_btn.setFixedHeight(34)
        self.deact_btn.setStyleSheet(f"""
            QPushButton {{
                background:transparent; color:{RED};
                border:1px solid {RED}; border-radius:6px;
                font-size:12px; padding:0 14px;
            }}
            QPushButton:hover {{ background:{RED}; color:{WHITE}; }}
        """)
        self.deact_btn.clicked.connect(self._deactivate_user)

        hint = QLabel("Double-click a row to edit")
        hint.setStyleSheet(f"color:{GREY}; font-size:11px; background:transparent;")

        act_row.addWidget(self.edit_btn)
        act_row.addWidget(self.deact_btn)
        act_row.addStretch()
        act_row.addWidget(hint)

        root.addLayout(hdr)
        root.addWidget(self.table, 1)
        root.addLayout(act_row)

    def refresh(self):
        users = self.user_db.fetch_all_users()
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        for user in users:
            r = self.table.rowCount()
            self.table.insertRow(r)
            self.table.setRowHeight(r, 44)

            def cell(text, align=Qt.AlignLeft):
                item = QTableWidgetItem(str(text))
                item.setTextAlignment(align | Qt.AlignVCenter)
                return item

            id_item = cell(user.id, Qt.AlignCenter)
            id_item.setData(Qt.UserRole, user.id)
            id_item.setForeground(QColor(TEAL_L))

            status_text  = "Active"   if user.is_active else "Inactive"
            status_color = GREEN      if user.is_active else RED

            status_item = cell(status_text, Qt.AlignCenter)
            status_item.setForeground(QColor(status_color))

            self.table.setItem(r, 0, id_item)
            self.table.setItem(r, 1, cell(user.username))
            self.table.setItem(r, 2, cell(user.full_name))
            self.table.setItem(r, 3, cell(ROLE_DISPLAY.get(user.role, user.role)))
            self.table.setItem(r, 4, cell(user.email))
            self.table.setItem(r, 5, cell(user.last_login or "Never", Qt.AlignCenter))
            self.table.setItem(r, 6, status_item)

        self.table.setSortingEnabled(True)

    def _get_selected_user_id(self):
        row = self.table.currentRow()
        if row < 0:
            return None
        item = self.table.item(row, 0)
        return item.data(Qt.UserRole) if item else None

    def _create_user(self):
        dlg = UserFormDialog(self.user_db, parent=self)
        if dlg.exec():
            self.refresh()

    def _edit_user(self):
        user_id = self._get_selected_user_id()
        if user_id is None:
            QMessageBox.information(self, "No Selection", "Select a user first.")
            return
        user = self.user_db.fetch_by_id(user_id)
        if user is None:
            return
        dlg = UserFormDialog(self.user_db, user=user, parent=self)
        if dlg.exec():
            self.refresh()

    def _deactivate_user(self):
        user_id = self._get_selected_user_id()
        if user_id is None:
            QMessageBox.information(self, "No Selection", "Select a user first.")
            return
        if user_id == self.current_user.id:
            QMessageBox.warning(self, "Cannot Deactivate",
                                "You cannot deactivate your own account.")
            return
        reply = QMessageBox.question(
            self, "Confirm",
            "Deactivate this account? The user will no longer be able to log in.",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.user_db.deactivate_user(user_id)
            self.refresh()