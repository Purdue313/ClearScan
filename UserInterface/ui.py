from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QListWidget, QMessageBox, QScrollArea,
    QCheckBox, QTabWidget
)
from PySide6.QtGui import QPixmap, QImageReader, QDragEnterEvent, QDropEvent, QImage
from PySide6.QtCore import Qt
from PIL import Image
import io

from Database.imageDatabase import ImageDatabase
from Database.findingsDatabase import FindingsDatabase
from Source.Models.image_model import ImageRecord
from machineLearningModel.prediction import load_model, predict_xray, predict_with_heatmap, CHECKPOINT_PATH

from UserInterface.diagnosis_window import DiagnosisWindow
from UserInterface.scan_browser import ScanBrowserWindow
from UserInterface.patient_selector import PatientSelectorDialog
from Database.patientDatabase import PatientDatabase
from machineLearningModel.ml_feedback_manager import MLFeedbackManager


class MainWindow(QWidget):
    def __init__(self):
        """
        Main application window for ClearScan.

        Responsibilities:
        - Provide UI for uploading and viewing medical images
        - Store image metadata in the database
        - Display images with correct orientation and scaling
        - Generate and display Grad-CAM heatmaps via toggle
        - Support drag-and-drop image uploads
        - Host the DiagnosisWindow for FR 3.6.1
        """
        print("Initializing MainWindow...")
        super().__init__()

        self.setWindowTitle("ClearScan Medical Imaging - AI Analysis with Heatmaps")
        self.setGeometry(100, 100, 1600, 900)
        self.setAcceptDrops(True)

        # ============================================================
        # DATABASE
        # ============================================================
        print("Initializing databases...")
        self.db          = ImageDatabase()
        self.findings_db = FindingsDatabase()
        self.patient_db  = PatientDatabase()

        self.images             = []
        self.current_pixmap     = None
        self.current_heatmap    = None
        self.current_image_path = None

        # ============================================================
        # ML MODEL + FEEDBACK MANAGER
        # ============================================================
        print("Loading ML model...")
        self.model           = load_model(CHECKPOINT_PATH)
        self.feedback_manager = MLFeedbackManager(self.findings_db, self.db)
        print("Model loaded successfully!")

        # ============================================================
        # MAIN LAYOUT  (left panel | right tabs)
        # ============================================================
        main_layout = QHBoxLayout(self)

        # ── LEFT PANEL ──────────────────────────────────────────
        left_panel = QVBoxLayout()

        self.upload_btn = QPushButton("Upload Image")
        self.upload_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50; color: white; border: none;
                padding: 10px; font-size: 14px; font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #45a049; }
        """)
        self.upload_btn.clicked.connect(self.upload_image)

        self.refresh_btn = QPushButton("Load Previous Scans")
        self.refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3; color: white; border: none;
                padding: 10px; font-size: 14px; font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #0b7dda; }
        """)
        self.refresh_btn.clicked.connect(self.load_images)

        self.browse_btn = QPushButton("🔍  Browse & Filter Scans")
        self.browse_btn.setStyleSheet("""
            QPushButton {
                background-color: #0d9488; color: white; border: none;
                padding: 10px; font-size: 14px; font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #0f766e; }
        """)
        self.browse_btn.clicked.connect(self._open_browser)

        # Heatmap toggle
        toggle_container = QHBoxLayout()
        self.heatmap_toggle = QCheckBox("Show Heatmap Overlay")
        self.heatmap_toggle.setStyleSheet("""
            QCheckBox { font-size:14px; font-weight:bold; padding:8px; spacing:10px; }
            QCheckBox::indicator { width:50px; height:26px; border-radius:13px; }
            QCheckBox::indicator:unchecked { background-color:#ccc; border:2px solid #999; }
            QCheckBox::indicator:checked   { background-color:#4CAF50; border:2px solid #45a049; }
        """)
        self.heatmap_toggle.stateChanged.connect(self.toggle_heatmap)
        self.heatmap_toggle.setEnabled(False)
        toggle_container.addWidget(self.heatmap_toggle)
        toggle_container.addStretch()

        # Image list
        self.image_list = QListWidget()
        self.image_list.setStyleSheet("""
            QListWidget { border:1px solid #ccc; border-radius:5px; padding:5px; }
            QListWidget::item { padding:8px; border-bottom:1px solid #eee; }
            QListWidget::item:selected { background-color:#2196F3; color:white; }
        """)
        self.image_list.itemSelectionChanged.connect(self.display_image)

        # AI results list
        self.results_label = QLabel("AI Analysis Results:")
        self.results_label.setStyleSheet("font-weight:bold; font-size:14px;")
        self.results_list = QListWidget()
        self.results_list.setStyleSheet("""
            QListWidget { border:1px solid #ccc; border-radius:5px; padding:5px; }
            QListWidget::item { padding:5px; }
        """)

        left_panel.addWidget(self.upload_btn)
        left_panel.addWidget(self.refresh_btn)
        left_panel.addWidget(self.browse_btn)
        left_panel.addLayout(toggle_container)
        left_panel.addWidget(QLabel("Stored Scans:"))
        left_panel.addWidget(self.image_list)
        left_panel.addWidget(self.results_label)
        left_panel.addWidget(self.results_list)

        # ── RIGHT PANEL: tabbed (Image Viewer | Diagnosis) ──────
        self.right_tabs = QTabWidget()
        self.right_tabs.setStyleSheet("""
            QTabWidget::pane  { border: 1px solid #ccc; border-radius:5px; }
            QTabBar::tab      { padding: 8px 20px; font-size:13px; }
            QTabBar::tab:selected { background:#2196F3; color:white;
                                    border-radius:4px 4px 0 0; }
        """)

        # Tab 1 — image viewer
        viewer_widget = QWidget()
        viewer_layout = QVBoxLayout(viewer_widget)
        self.image_label = QLabel("No image selected")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel { background-color:#f5f5f5; border:2px dashed #ccc; border-radius:5px; }
        """)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.image_label)
        viewer_layout.addWidget(self.scroll_area)

        # Tab 2 — diagnosis
        self.diagnosis_window = DiagnosisWindow(self.findings_db, self.db, model=self.model)
        self.diagnosis_window.diagnosis_saved.connect(self._on_diagnosis_saved)

        # Scan browser (separate window, created once, shown on demand)
        self.scan_browser = ScanBrowserWindow(self.db, self.findings_db, self.patient_db)
        self.scan_browser.open_image.connect(self._open_image_from_browser)

        self.right_tabs.addTab(viewer_widget,         "🖼  Image Viewer")
        self.right_tabs.addTab(self.diagnosis_window, "🩺  Diagnosis")

        # ── Assemble main layout ─────────────────────────────────
        main_layout.addLayout(left_panel, 1)
        main_layout.addWidget(self.right_tabs, 3)

        print("MainWindow initialized successfully!")

    # ============================================================
    # IMAGE UPLOAD
    # ============================================================
    def upload_image(self):
        # Step 1: select patient first
        dialog = PatientSelectorDialog(self.patient_db, self)
        accepted = dialog.exec()
        patient = dialog.selected_patient  # None if skipped

        # Step 2: pick file
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg)"
        )
        if file_path:
            self.register_image(file_path, patient)

    # ============================================================
    # DRAG & DROP
    # ============================================================
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.toLocalFile().lower().endswith((".png", ".jpg", ".jpeg")):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent):
        paths = [
            url.toLocalFile() for url in event.mimeData().urls()
            if url.toLocalFile().lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if not paths:
            return
        # Show patient selector once for all dropped files
        dialog = PatientSelectorDialog(self.patient_db, self)
        dialog.exec()
        patient = dialog.selected_patient
        for file_path in paths:
            self.register_image(file_path, patient)

    # ============================================================
    # IMAGE REGISTRATION
    # ============================================================
    def register_image(self, file_path, patient=None):
        """Register image, run ML inference, store findings, link to patient."""
        try:
            image = ImageRecord.create(file_path=file_path, user="test_user")
            image = self.db.insert_image(image)

            # Link to patient if one was selected
            if patient is not None:
                self.patient_db.link_image_to_patient(image.id, patient.id)

            results = predict_xray(self.model, file_path)

            findings_to_save = [
                {
                    "image_id":      image.id,
                    "label":         result["label"],
                    "probability":   result["probability"],
                    "model_version": "chexpert_resnet50_v1",
                }
                for result in results
            ]
            self.findings_db.insert_findings(findings_to_save)

            self.load_images()
            self.display_results(results)

            self.heatmap_toggle.setEnabled(True)
            self.current_image_path = file_path

            # Load the new image into the diagnosis panel
            self.diagnosis_window.load_image(image.id)

        except Exception as e:
            QMessageBox.critical(self, "Processing Error", str(e))

    # ============================================================
    # LOAD IMAGES FROM DATABASE
    # ============================================================
    def load_images(self):
        self.image_list.clear()
        self.images = self.db.fetch_all_images()
        for img in self.images:
            diagnosed = self.findings_db.is_diagnosed(img.id)
            badge     = " ✔" if diagnosed else ""
            self.image_list.addItem(
                f"ID {img.id} | {img.uploaded_at} | {img.user}{badge}"
            )

    # ============================================================
    # DISPLAY IMAGE
    # ============================================================
    def load_pixmap_with_orientation(self, file_path):
        reader = QImageReader(file_path)
        reader.setAutoTransform(True)
        image = reader.read()
        if image.isNull():
            return None
        return QPixmap.fromImage(image)

    def display_image(self):
        index = self.image_list.currentRow()
        if index < 0:
            return

        image = self.images[index]
        self.current_image_path = image.file_path

        pixmap = self.load_pixmap_with_orientation(image.file_path)
        if pixmap is None:
            QMessageBox.warning(self, "Error", f"Unable to load image:\n{image.file_path}")
            return

        self.current_pixmap = pixmap
        self.heatmap_toggle.setEnabled(True)

        results = self.load_results_for_image(image.id)
        if results:
            self.display_results(results)
        else:
            self.results_list.clear()
            self.results_list.addItem("No analysis available.")

        # Sync diagnosis panel with the selected image
        self.diagnosis_window.load_image(image.id)

        if self.heatmap_toggle.isChecked():
            self.generate_and_show_heatmap()
        else:
            self.current_heatmap = None
            self.update_image_display()

    # ============================================================
    # HEATMAP
    # ============================================================
    def toggle_heatmap(self, state):
        # stateChanged emits an int (2 = checked, 0 = unchecked)
        if int(state) == 2:
            # If no image path yet, try to get it from whatever is selected
            if not self.current_image_path:
                index = self.image_list.currentRow()
                if index >= 0:
                    self.current_image_path = self.images[index].file_path
                else:
                    self.heatmap_toggle.setChecked(False)
                    QMessageBox.warning(self, "No Image", "Please select an image first.")
                    return

            # If pixmap isn't loaded yet (image was selected before toggle turned on)
            # load it now so update_image_display has something to fall back to
            if self.current_pixmap is None:
                self.current_pixmap = self.load_pixmap_with_orientation(self.current_image_path)

            self.generate_and_show_heatmap()
        else:
            self.current_heatmap = None
            self.update_image_display()

    def generate_and_show_heatmap(self):
        if not self.current_image_path:
            return
        try:
            self.image_label.setText("Generating heatmap…\nPlease wait")
            self.image_label.repaint()
            result      = predict_with_heatmap(self.model, self.current_image_path, target_label=None)
            heatmap_pil = result["heatmap_image"]
            buf         = io.BytesIO()
            heatmap_pil.save(buf, format="PNG")
            buf.seek(0)
            qimg              = QImage()
            qimg.loadFromData(buf.read())
            self.current_heatmap = QPixmap.fromImage(qimg)
            self.update_image_display()
        except Exception as e:
            self.heatmap_toggle.setChecked(False)
            QMessageBox.critical(self, "Heatmap Error", f"Failed to generate heatmap:\n{str(e)}")

    def resizeEvent(self, event):
        self.update_image_display()
        super().resizeEvent(event)

    def update_image_display(self):
        if self.heatmap_toggle.isChecked() and self.current_heatmap is not None:
            scaled = self.current_heatmap.scaled(
                self.scroll_area.viewport().size(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled)
        elif self.current_pixmap is not None:
            scaled = self.current_pixmap.scaled(
                self.scroll_area.viewport().size(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled)

    # ============================================================
    # RESULTS
    # ============================================================
    def display_results(self, results):
        self.results_list.clear()
        if not results or not isinstance(results, list):
            self.results_list.addItem("No results available.")
            return
        for i, r in enumerate(results):
            label      = r.get("label", "Unknown")
            percentage = r.get("percentage")
            if percentage is None:
                percentage = f"{float(r.get('probability', 0.0)) * 100:.2f}%"
            self.results_list.addItem(f"{i+1}. {label} — {percentage}")

    def load_results_for_image(self, image_id):
        rows = self.findings_db.fetch_findings_for_image(image_id)
        return [
            {
                "label":       label,
                "probability": probability,
                "percentage":  f"{probability * 100:.2f}%",
            }
            for label, probability, model_version, created_at in rows
        ]

    # ============================================================
    # DIAGNOSIS SAVED CALLBACK
    # ============================================================
    def _on_diagnosis_saved(self, image_id: int):
        """
        Called when the DiagnosisWindow emits diagnosis_saved.
        Refreshes the scan list so the ✔ badge appears immediately.
        """
        self.load_images()

    # ============================================================
    # SCAN BROWSER
    # ============================================================
    def _open_browser(self):
        """Open the scan browser window and refresh its data."""
        self.scan_browser.refresh()
        self.scan_browser.show()
        self.scan_browser.raise_()
        self.scan_browser.activateWindow()

    def _open_image_from_browser(self, image_id: int):
        """
        Called when the user double-clicks a row in the scan browser.
        Selects that image in the main list and brings the main window forward.
        """
        # Find the row in self.images that matches image_id
        for i, img in enumerate(self.images):
            if img.id == image_id:
                self.image_list.setCurrentRow(i)
                break
        else:
            # Image not in list yet — reload then try again
            self.load_images()
            for i, img in enumerate(self.images):
                if img.id == image_id:
                    self.image_list.setCurrentRow(i)
                    break

        self.raise_()
        self.activateWindow()