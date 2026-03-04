from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QLineEdit, QFrame
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

from Database.userDatabase import UserDatabase

NAVY   = "#0a1628"
STEEL  = "#1c3150"
TEAL   = "#0d9488"
TEAL_L = "#14b8a8"
WHITE  = "#ffffff"
GREY   = "#64748b"
RED    = "#dc2626"


class LoginWindow(QWidget):
    """
    Login screen shown on launch.
    Emits login_successful(UserRecord) when credentials are valid.
    """

    login_successful = Signal(object)

    def __init__(self, user_db: UserDatabase, parent=None):
        super().__init__(parent)
        self.user_db = user_db
        self.setWindowTitle("ClearScan - Login")
        self.setMinimumSize(480, 540)
        self._build_ui()

    def _build_ui(self):
        self.setStyleSheet(f"background: {NAVY};")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addStretch(2)

        # Card
        card = QFrame()
        card.setFixedWidth(400)
        card.setStyleSheet(f"""
            QFrame {{
                background: {STEEL};
                border-radius: 16px;
            }}
        """)

        c = QVBoxLayout(card)
        c.setContentsMargins(40, 40, 40, 40)
        c.setSpacing(14)

        title = QLabel("ClearScan")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"""
            color: {TEAL_L};
            font-size: 28px;
            font-weight: 700;
            letter-spacing: 1px;
            background: transparent;
        """)

        sub = QLabel("Medical Imaging Platform")
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet(f"color: {GREY}; font-size: 13px; background: transparent;")

        div = QFrame()
        div.setFrameShape(QFrame.HLine)
        div.setStyleSheet("background: #2d4a6e; max-height:1px;")

        user_lbl = QLabel("Username")
        user_lbl.setStyleSheet(f"color:{GREY}; font-size:12px; background:transparent;")

        self.username_field = QLineEdit()
        self.username_field.setPlaceholderText("Enter username")
        self.username_field.setFixedHeight(42)
        self.username_field.setStyleSheet(self._fld())

        pass_lbl = QLabel("Password")
        pass_lbl.setStyleSheet(f"color:{GREY}; font-size:12px; background:transparent;")

        self.password_field = QLineEdit()
        self.password_field.setPlaceholderText("Enter password")
        self.password_field.setEchoMode(QLineEdit.Password)
        self.password_field.setFixedHeight(42)
        self.password_field.setStyleSheet(self._fld())
        self.password_field.returnPressed.connect(self._attempt_login)

        self.error_lbl = QLabel("")
        self.error_lbl.setAlignment(Qt.AlignCenter)
        self.error_lbl.setStyleSheet(f"color:{RED}; font-size:12px; background:transparent;")
        self.error_lbl.setVisible(False)

        login_btn = QPushButton("Sign In")
        login_btn.setFixedHeight(44)
        login_btn.setStyleSheet(f"""
            QPushButton {{
                background: {TEAL};
                color: {WHITE};
                border: none;
                border-radius: 8px;
                font-size: 15px;
                font-weight: 700;
            }}
            QPushButton:hover {{ background: {TEAL_L}; }}
        """)
        login_btn.clicked.connect(self._attempt_login)

        hint = QLabel("Default: admin / admin123")
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet(f"color: #3a5070; font-size: 11px; background: transparent;")

        c.addWidget(title)
        c.addWidget(sub)
        c.addSpacing(4)
        c.addWidget(div)
        c.addSpacing(4)
        c.addWidget(user_lbl)
        c.addWidget(self.username_field)
        c.addWidget(pass_lbl)
        c.addWidget(self.password_field)
        c.addWidget(self.error_lbl)
        c.addSpacing(4)
        c.addWidget(login_btn)
        c.addWidget(hint)

        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(card)
        row.addStretch()

        root.addLayout(row)
        root.addStretch(3)

    def _attempt_login(self):
        username = self.username_field.text().strip()
        password = self.password_field.text()

        if not username or not password:
            self._show_error("Please enter your username and password.")
            return

        user = self.user_db.authenticate(username, password)
        if user is None:
            self._show_error("Invalid username or password.")
            self.password_field.clear()
            self.password_field.setFocus()
            return

        self.error_lbl.setVisible(False)
        self.login_successful.emit(user)

    def _show_error(self, msg):
        self.error_lbl.setText(msg)
        self.error_lbl.setVisible(True)

    def _fld(self):
        return f"""
            QLineEdit {{
                background: {NAVY};
                color: {WHITE};
                border: 1px solid #2d4a6e;
                border-radius: 8px;
                padding: 0 14px;
                font-size: 14px;
            }}
            QLineEdit:focus {{ border-color: {TEAL}; }}
        """