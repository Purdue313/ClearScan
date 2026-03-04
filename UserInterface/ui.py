from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QListWidget, QMessageBox, QScrollArea,
    QCheckBox, QTabWidget
)
from PySide6.QtGui import QPixmap, QImageReader, QDragEnterEvent, QDropEvent, QImage
from PySide6.QtCore import Qt, Signal, QTimer
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
    go_to_dashboard = Signal()

    def __init__(self, current_user=None):
        print("Initializing MainWindow...")
        super().__init__()

        self.current_user = current_user

        self.setWindowTitle("ClearScan Medical Imaging")
        self.setGeometry(100, 100, 1600, 900)
        self.setAcceptDrops(True)

        # -- Databases --------------------------------------------
        print("Initializing databases...")
        self.db          = ImageDatabase()
        self.findings_db = FindingsDatabase()
        self.patient_db  = PatientDatabase()

        self.images             = []
        self.current_pixmap     = None
        self.current_heatmap    = None
        self.current_image_path = None

        # -- ML model (lazy-loaded after window shows) ------------
        self.model            = None
        self.feedback_manager = None
        self._model_loaded    = False
        QTimer.singleShot(100, self._load_model_async)

        # -- Layout -----------------------------------------------
        # Outer vertical layout: nav bar on top, content below
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        # Nav bar
        nav_bar = QWidget()
        nav_bar.setFixedHeight(46)
        nav_bar.setStyleSheet("background: #1c3150; border-bottom: 2px solid #0d9488;")
        nav_bar_layout = QHBoxLayout(nav_bar)
        nav_bar_layout.setContentsMargins(12, 0, 12, 0)

        brand_lbl = QLabel("ClearScan")
        brand_lbl.setStyleSheet("color:#14b8a8; font-size:16px; font-weight:700; background:transparent;")

        self.account_btn = QPushButton()
        self.account_btn.setFixedSize(34, 34)
        initials = ""
        if current_user:
            initials = (current_user.first_name[:1] + current_user.last_name[:1]).upper()
        self.account_btn.setText(initials if initials else "?")
        self.account_btn.setToolTip("Account / Back to Dashboard")
        self.account_btn.setStyleSheet("""
            QPushButton {
                background-color: #0d9488;
                color: white;
                border-radius: 17px;
                font-size: 13px;
                font-weight: 700;
                border: none;
            }
            QPushButton:hover { background-color: #14b8a8; }
        """)
        self.account_btn.clicked.connect(self.go_to_dashboard)

        nav_bar_layout.addWidget(brand_lbl)
        nav_bar_layout.addStretch()
        nav_bar_layout.addWidget(self.account_btn)

        outer_layout.addWidget(nav_bar)

        # Content row below nav bar
        content_widget = QWidget()
        main_layout = QHBoxLayout(content_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        outer_layout.addWidget(content_widget, 1)

        # Left panel
        left_panel = QVBoxLayout()

        # Only show upload if user has permission
        can_upload = (current_user is None or
                      current_user.has_permission("can_upload"))

        self.upload_btn = QPushButton("Upload Image")
        self.upload_btn.setEnabled(can_upload)
        self.upload_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50; color: white; border: none;
                padding: 10px; font-size: 14px; font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover   { background-color: #45a049; }
            QPushButton:disabled { background-color: #aaa; }
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

        self.browse_btn = QPushButton("Browse and Filter Scans")
        self.browse_btn.setStyleSheet("""
            QPushButton {
                background-color: #0d9488; color: white; border: none;
                padding: 10px; font-size: 14px; font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #0f766e; }
        """)
        self.browse_btn.clicked.connect(self._open_browser)

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

        self.image_list = QListWidget()
        self.image_list.setStyleSheet("""
            QListWidget { border:1px solid #ccc; border-radius:5px; padding:5px; }
            QListWidget::item { padding:8px; border-bottom:1px solid #eee; }
            QListWidget::item:selected { background-color:#2196F3; color:white; }
        """)
        self.image_list.itemSelectionChanged.connect(self.display_image)

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

        # Right tabs
        self.right_tabs = QTabWidget()
        self.right_tabs.setStyleSheet("""
            QTabWidget::pane  { border: 1px solid #ccc; border-radius:5px; }
            QTabBar::tab      { padding: 8px 20px; font-size:13px; }
            QTabBar::tab:selected { background:#2196F3; color:white;
                                    border-radius:4px 4px 0 0; }
        """)

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

        # Diagnosis tab -- only radiologists and above can diagnose
        can_diagnose = (current_user is None or
                        current_user.has_permission("can_diagnose"))
        self.diagnosis_window = DiagnosisWindow(
            self.findings_db, self.db, model=self.model
        )
        if not can_diagnose:
            self.diagnosis_window.setEnabled(False)
        self.diagnosis_window.diagnosis_saved.connect(self._on_diagnosis_saved)

        self.scan_browser = ScanBrowserWindow(self.db, self.findings_db, self.patient_db)
        self.scan_browser.open_image.connect(self._open_image_from_browser)

        self.right_tabs.addTab(viewer_widget,         "Image Viewer")
        self.right_tabs.addTab(self.diagnosis_window, "Diagnosis")

        main_layout.addLayout(left_panel, 1)
        main_layout.addWidget(self.right_tabs, 3)

        print("MainWindow initialized successfully!")

    # ============================================================
    # LAZY MODEL LOAD
    # ============================================================
    def _load_model_async(self):
        try:
            print("Loading ML model...")
            self.model            = load_model(CHECKPOINT_PATH)
            self.feedback_manager = MLFeedbackManager(self.findings_db, self.db)
            self._model_loaded    = True
            # Re-init diagnosis window with loaded model
            self.diagnosis_window.model = self.model
            self.heatmap_toggle.setEnabled(len(self.images) > 0)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Model load failed: {e}")

    # ============================================================
    # UPLOAD
    # ============================================================
    def upload_image(self):
        dialog  = PatientSelectorDialog(self.patient_db, self)
        dialog.exec()
        patient = dialog.selected_patient

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg)"
        )
        if file_path:
            self.register_image(file_path, patient)

    # ============================================================
    # DRAG AND DROP
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
        dialog  = PatientSelectorDialog(self.patient_db, self)
        dialog.exec()
        patient = dialog.selected_patient
        for file_path in paths:
            self.register_image(file_path, patient)

    # ============================================================
    # REGISTER IMAGE
    # ============================================================
    def register_image(self, file_path, patient=None):
        try:
            user_tag = self.current_user.username if self.current_user else "test_user"
            image    = ImageRecord.create(file_path=file_path, user=user_tag)
            image    = self.db.insert_image(image)

            if patient is not None:
                self.patient_db.link_image_to_patient(image.id, patient.id)

            if self.model is None:
                QMessageBox.warning(self, "Model Loading",
                    "The AI model is still loading. Please wait a moment and try again.")
                return
            results = predict_xray(self.model, file_path)

            self.findings_db.insert_findings([
                {
                    "image_id":      image.id,
                    "label":         r["label"],
                    "probability":   r["probability"],
                    "model_version": "chexpert_resnet50_v1",
                }
                for r in results
            ])

            self.load_images()
            self.display_results(results)
            self.heatmap_toggle.setEnabled(True)
            self.current_image_path = file_path
            self.diagnosis_window.load_image(image.id)

        except Exception as e:
            QMessageBox.critical(self, "Processing Error", str(e))

    # ============================================================
    # LOAD IMAGES
    # ============================================================
    def load_images(self):
        self.image_list.clear()
        self.images = self.db.fetch_all_images()
        for img in self.images:
            diagnosed = self.findings_db.is_diagnosed(img.id)
            badge     = " (v)" if diagnosed else ""
            self.image_list.addItem(
                f"ID {img.id} | {img.uploaded_at} | {img.user}{badge}"
            )

    # ============================================================
    # DISPLAY IMAGE
    # ============================================================
    def load_pixmap_with_orientation(self, file_path):
        reader = QImageReader(file_path)
        reader.setAutoTransform(True)
        image  = reader.read()
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

        self.diagnosis_window.load_image(image.id)

        if self.heatmap_toggle.isChecked():
            self.generate_and_show_heatmap()
        else:
            self.current_heatmap = None
            self.update_image_display()

    # ============================================================
    # HEATMAP TOGGLE
    # ============================================================
    def toggle_heatmap(self, state):
        if int(state) == 2:
            if not self.current_image_path:
                index = self.image_list.currentRow()
                if index >= 0:
                    self.current_image_path = self.images[index].file_path
                else:
                    self.heatmap_toggle.setChecked(False)
                    QMessageBox.warning(self, "No Image", "Please select an image first.")
                    return
            if self.current_pixmap is None:
                self.current_pixmap = self.load_pixmap_with_orientation(self.current_image_path)
            self.generate_and_show_heatmap()
        else:
            self.current_heatmap = None
            self.image_label.setStyleSheet("""
                QLabel {
                    background-color: #f5f5f5;
                    border: 2px dashed #ccc;
                    border-radius: 5px;
                }
            """)
            self.update_image_display()

    # ============================================================
    # GENERATE HEATMAP  (separate method -- called by toggle and display_image)
    # ============================================================
    def generate_and_show_heatmap(self):
        if not self.current_image_path:
            return
        try:
            # Check top prediction -- skip heatmap for No Finding
            if self.model is None:
                self.image_label.setText("AI model is still loading...\nPlease wait.")
                return
            top_results = predict_xray(self.model, self.current_image_path)
            if top_results and top_results[0]["label"] == "No Finding":
                self.current_heatmap = None
                self.image_label.setText(
                    "No Finding\n\n"
                    "The model did not detect any abnormalities.\n"
                    "No heatmap is available."
                )
                self.image_label.setStyleSheet("""
                    QLabel {
                        background-color: #f0faf4;
                        border: 2px solid #16a34a;
                        border-radius: 5px;
                        color: #15803d;
                        font-size: 16px;
                        font-weight: bold;
                    }
                """)
                return

            self.image_label.setText("Generating heatmap... Please wait")
            self.image_label.setStyleSheet("""
                QLabel {
                    background-color: #f5f5f5;
                    border: 2px dashed #ccc;
                    border-radius: 5px;
                }
            """)
            self.image_label.repaint()

            result      = predict_with_heatmap(self.model, self.current_image_path, target_label=None)
            heatmap_pil = result["heatmap_image"]
            buf         = io.BytesIO()
            heatmap_pil.save(buf, format="PNG")
            buf.seek(0)
            qimg = QImage()
            qimg.loadFromData(buf.read())
            self.current_heatmap = QPixmap.fromImage(qimg)
            self.update_image_display()

        except Exception as e:
            self.heatmap_toggle.setChecked(False)
            QMessageBox.critical(self, "Heatmap Error", f"Failed to generate heatmap:\n{str(e)}")

    # ============================================================
    # DISPLAY UPDATE
    # ============================================================
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
            self.results_list.addItem(f"{i+1}. {label} -- {percentage}")

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
    # CALLBACKS
    # ============================================================
    def _on_diagnosis_saved(self, image_id: int):
        self.load_images()

    def _open_browser(self):
        self.scan_browser.refresh()
        self.scan_browser.show()
        self.scan_browser.raise_()
        self.scan_browser.activateWindow()

    def _open_image_from_browser(self, image_id: int):
        for i, img in enumerate(self.images):
            if img.id == image_id:
                self.image_list.setCurrentRow(i)
                break
        else:
            self.load_images()
            for i, img in enumerate(self.images):
                if img.id == image_id:
                    self.image_list.setCurrentRow(i)
                    break
        self.raise_()
        self.activateWindow()