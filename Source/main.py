import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from Database.userDatabase import UserDatabase
from Database.imageDatabase import ImageDatabase
from Database.findingsDatabase import FindingsDatabase
from Database.patientDatabase import PatientDatabase
from Database.scheduleDatabase import ScheduleDatabase

from UserInterface.login_window import LoginWindow
from UserInterface.dashboard_window import DashboardWindow
from UserInterface.ui import MainWindow
from UserInterface.account_management import AccountManagementWindow
from UserInterface.schedule_window import ScheduleWindow


class AppController:
    """
    Controls which screen is visible.
    Flow:  Login -> Dashboard -> Scanner (or Account Mgmt)
    """

    def __init__(self):
        # Shared databases
        self.user_db     = UserDatabase()
        self.image_db    = ImageDatabase()
        self.findings_db = FindingsDatabase()
        self.patient_db   = PatientDatabase()
        self.schedule_db  = ScheduleDatabase()

        self.current_user      = None
        self.login_win         = None
        self.dashboard_win     = None
        self.scanner_win       = None
        self.account_mgmt_win  = None
        self.schedule_win      = None

        self._show_login()

    # ----------------------------------------------------------
    def _show_login(self):
        if self.dashboard_win:
            self.dashboard_win.hide()
        if self.scanner_win:
            self.scanner_win.hide()

        self.login_win = LoginWindow(self.user_db)
        self.login_win.login_successful.connect(self._on_login)
        self.login_win.showMaximized()

    # ----------------------------------------------------------
    def _on_login(self, user):
        self.current_user = user
        self.login_win.hide()
        self._show_dashboard()

    # ----------------------------------------------------------
    def _show_dashboard(self):
        self.dashboard_win = DashboardWindow(
            user        = self.current_user,
            image_db    = self.image_db,
            findings_db = self.findings_db,
            patient_db  = self.patient_db,
            user_db     = self.user_db,
            schedule_db = self.schedule_db,
        )
        self.dashboard_win.open_scanner.connect(self._show_scanner)
        self.dashboard_win.open_browser.connect(self._open_browser)
        self.dashboard_win.open_account_mgmt.connect(self._show_account_mgmt)
        self.dashboard_win.sign_out.connect(self._sign_out)
        self.dashboard_win.open_schedule.connect(self._show_schedule)
        self.dashboard_win.open_scan_id.connect(self._open_scan_from_dashboard)
        self.dashboard_win.upload_files.connect(self._upload_from_dashboard)
        self.dashboard_win.showMaximized()

    # ----------------------------------------------------------
    def _show_scanner(self):
        if self.scanner_win is None:
            self.scanner_win = MainWindow(current_user=self.current_user)
            self.scanner_win.go_to_dashboard.connect(self._back_to_dashboard)
        self.dashboard_win.hide()
        self.scanner_win.showMaximized()

    # ----------------------------------------------------------
    def _back_to_dashboard(self):
        if self.scanner_win:
            self.scanner_win.hide()
        if self.dashboard_win:
            self.dashboard_win.refresh()
            self.dashboard_win.showMaximized()
        else:
            self._show_dashboard()

    # ----------------------------------------------------------
    def _open_browser(self):
        """Open scan browser from dashboard (reuses scanner's browser if open)."""
        if self.scanner_win is None:
            self.scanner_win = MainWindow(current_user=self.current_user)
        self.dashboard_win.hide()
        self.scanner_win.showMaximized()
        self.scanner_win._open_browser()

    # ----------------------------------------------------------
    def _open_scan_from_dashboard(self, image_id: int):
        """Double-clicked a recent scan on the dashboard."""
        self._show_scanner()
        self.scanner_win.load_images()
        self.scanner_win._open_image_from_browser(image_id)

    # ----------------------------------------------------------
    def _upload_from_dashboard(self, paths: list):
        """Files dropped/selected on the dashboard upload zone."""
        self._show_scanner()
        for path in paths:
            self.scanner_win.register_image(path)

    # ----------------------------------------------------------
    def _show_account_mgmt(self):
        self.account_mgmt_win = AccountManagementWindow(
            self.user_db, self.current_user
        )
        self.account_mgmt_win.show()
        self.account_mgmt_win.raise_()

    # ----------------------------------------------------------
    def _show_schedule(self):
        self.schedule_win = ScheduleWindow(
            self.schedule_db, self.patient_db, self.user_db,
            current_user=self.current_user
        )
        self.schedule_win.show()
        self.schedule_win.raise_()

    # ----------------------------------------------------------
    def _sign_out(self):
        self.current_user = None
        if self.dashboard_win:
            self.dashboard_win.close()
            self.dashboard_win = None
        if self.scanner_win:
            self.scanner_win.close()
            self.scanner_win = None
        if self.account_mgmt_win:
            self.account_mgmt_win.close()
            self.account_mgmt_win = None
        if self.schedule_win:
            self.schedule_win.close()
            self.schedule_win = None
        self._show_login()


def main():
    print("Creating QApplication...")
    app = QApplication(sys.argv)
    app.setApplicationName("ClearScan")

    print("Starting AppController...")
    controller = AppController()

    print("Starting event loop...")
    sys.exit(app.exec())