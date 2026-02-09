from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QListWidget,
    QMessageBox,
    QScrollArea,
    QSplitter,
    QCheckBox
)
from PySide6.QtGui import QPixmap, QImageReader, QDragEnterEvent, QDropEvent, QImage
from PySide6.QtCore import Qt
from PIL import Image
import io

from Database.imageDatabase import ImageDatabase
from Database.findingsDatabase import FindingsDatabase

from Source.Models.image_model import ImageRecord

from machineLearningModel.prediction import load_model, predict_xray, predict_with_heatmap, CHECKPOINT_PATH


class MainWindow(QWidget):
    def __init__(self):
        """
        Main application window for ClearScan with toggle heatmap support.

        Responsibilities:
        - Provide UI for uploading and viewing medical images
        - Store image metadata in the database
        - Display images with correct orientation and scaling
        - Generate and display Grad-CAM heatmaps via toggle
        - Support drag-and-drop image uploads
        """
        print("Initializing MainWindow...")
        super().__init__()

        self.setWindowTitle("ClearScan Medical Imaging - AI Analysis with Heatmaps")
        self.setGeometry(100, 100, 1400, 800)

        # Enable drag-and-drop on the entire window
        self.setAcceptDrops(True)

        # ============================================================
        # DATABASE
        # ============================================================
        print("Initializing databases...")
        self.db = ImageDatabase()
        self.findings_db = FindingsDatabase()

        self.images = []
        self.current_pixmap = None
        self.current_heatmap = None
        self.current_image_path = None

        # ============================================================
        # ML MODEL (LOADED ONCE)
        # ============================================================
        print("Loading ML model...")
        self.model = load_model(CHECKPOINT_PATH)
        print("Model loaded successfully!")

        # ============================================================
        # MAIN LAYOUT
        # ============================================================
        main_layout = QHBoxLayout(self)

        # ============================================================
        # LEFT PANEL: CONTROLS + IMAGE LIST
        # ============================================================
        left_panel = QVBoxLayout()

        # Upload button
        self.upload_btn = QPushButton("Upload Image")
        self.upload_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.upload_btn.clicked.connect(self.upload_image)

        # Refresh button
        self.refresh_btn = QPushButton("Load Previous Scans")
        self.refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
        """)
        self.refresh_btn.clicked.connect(self.load_images)

        # ============================================================
        # HEATMAP TOGGLE SWITCH
        # ============================================================
        toggle_container = QHBoxLayout()
        
        self.heatmap_toggle = QCheckBox("Show Heatmap Overlay")
        self.heatmap_toggle.setStyleSheet("""
            QCheckBox {
                font-size: 14px;
                font-weight: bold;
                padding: 8px;
                spacing: 10px;
            }
            QCheckBox::indicator {
                width: 50px;
                height: 26px;
                border-radius: 13px;
            }
            QCheckBox::indicator:unchecked {
                background-color: #ccc;
                border: 2px solid #999;
            }
            QCheckBox::indicator:unchecked:hover {
                background-color: #bbb;
            }
            QCheckBox::indicator:checked {
                background-color: #4CAF50;
                border: 2px solid #45a049;
            }
            QCheckBox::indicator:checked:hover {
                background-color: #45a049;
            }
        """)
        self.heatmap_toggle.stateChanged.connect(self.toggle_heatmap)
        self.heatmap_toggle.setEnabled(False)
        
        toggle_container.addWidget(self.heatmap_toggle)
        toggle_container.addStretch()

        # List of stored scans
        self.image_list = QListWidget()
        self.image_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #ccc;
                border-radius: 5px;
                padding: 5px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #eee;
            }
            QListWidget::item:selected {
                background-color: #2196F3;
                color: white;
            }
        """)
        self.image_list.itemSelectionChanged.connect(self.display_image)

        left_panel.addWidget(self.upload_btn)
        left_panel.addWidget(self.refresh_btn)
        left_panel.addLayout(toggle_container)
        left_panel.addWidget(QLabel("Stored Scans:"))
        left_panel.addWidget(self.image_list)
        
        # Ranked prediction results
        self.results_label = QLabel("AI Analysis Results:")
        self.results_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        self.results_list = QListWidget()
        self.results_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #ccc;
                border-radius: 5px;
                padding: 5px;
            }
            QListWidget::item {
                padding: 5px;
            }
        """)

        left_panel.addWidget(self.results_label)
        left_panel.addWidget(self.results_list)

        # ============================================================
        # RIGHT PANEL: DUAL IMAGE VIEWER
        # ============================================================
        right_panel = QHBoxLayout()
        
        # Original image viewer
        self.image_label = QLabel("No image selected")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: #f5f5f5;
                border: 2px dashed #ccc;
                border-radius: 5px;
            }
        """)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.image_label)
        
        # Heatmap viewer
        self.heatmap_label = QLabel("Toggle 'Show Heatmap Overlay' to visualize AI analysis")
        self.heatmap_label.setAlignment(Qt.AlignCenter)
        self.heatmap_label.setStyleSheet("""
            QLabel {
                background-color: #f5f5f5;
                border: 2px dashed #ccc;
                border-radius: 5px;
            }
        """)
        
        self.heatmap_scroll = QScrollArea()
        self.heatmap_scroll.setWidgetResizable(True)
        self.heatmap_scroll.setWidget(self.heatmap_label)
        
        # Use splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.scroll_area)
        splitter.addWidget(self.heatmap_scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        
        right_panel.addWidget(splitter)

        # ============================================================
        # ADD PANELS TO MAIN LAYOUT
        # ============================================================
        main_layout.addLayout(left_panel, 1)
        main_layout.addLayout(right_panel, 3)
        
        print("MainWindow initialized successfully!")

    # ============================================================
    # IMAGE UPLOAD
    # ============================================================
    def upload_image(self):
        """Opens a file dialog to select an image."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.png *.jpg *.jpeg)"
        )

        if not file_path:
            return

        self.register_image(file_path)

    # ============================================================
    # DRAG & DROP SUPPORT
    # ============================================================
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Accept drag events with image files"""
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.toLocalFile().lower().endswith((".png", ".jpg", ".jpeg")):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent):
        """Handle dropped image files"""
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith((".png", ".jpg", ".jpeg")):
                self.register_image(file_path)

    # ============================================================
    # IMAGE REGISTRATION
    # ============================================================
    def register_image(self, file_path):
        """Register image, run ML inference, and store ALL findings"""
        try:
            # Store image
            image = ImageRecord.create(
                file_path=file_path,
                user="test_user"
            )
            image = self.db.insert_image(image)

            # Run ML inference
            results = predict_xray(self.model, file_path)

            # Store ALL findings to database (all 14 conditions)
            findings_to_save = [
                {
                    "image_id": image.id,
                    "label": result["label"],
                    "probability": result["probability"],
                    "model_version": "chexpert_resnet50_v1"
                }
                for result in results
            ]
            self.findings_db.insert_findings(findings_to_save)

            # Update UI
            self.load_images()
            self.display_results(results)
            
            # Enable heatmap toggle
            self.heatmap_toggle.setEnabled(True)
            self.current_image_path = file_path

        except Exception as e:
            QMessageBox.critical(
                self,
                "Processing Error",
                str(e)
            )

    # ============================================================
    # LOAD IMAGES FROM DATABASE
    # ============================================================
    def load_images(self):
        """Fetch all stored images and populate list"""
        self.image_list.clear()
        self.images = self.db.fetch_all_images()

        for img in self.images:
            self.image_list.addItem(
                f"ID {img.id} | {img.uploaded_at} | {img.user}"
            )

    # ============================================================
    # IMAGE LOADING WITH ORIENTATION
    # ============================================================
    def load_pixmap_with_orientation(self, file_path):
        """Load image with EXIF orientation applied"""
        reader = QImageReader(file_path)
        reader.setAutoTransform(True)
        image = reader.read()

        if image.isNull():
            return None

        return QPixmap.fromImage(image)

    # ============================================================
    # DISPLAY IMAGE
    # ============================================================
    def display_image(self):
        """Display selected image and its findings"""
        index = self.image_list.currentRow()

        if index < 0:
            return

        image = self.images[index]
        self.current_image_path = image.file_path

        # Load and display image
        pixmap = self.load_pixmap_with_orientation(image.file_path)

        if pixmap is None:
            QMessageBox.warning(
                self,
                "Error",
                f"Unable to load image:\n{image.file_path}"
            )
            return

        self.current_pixmap = pixmap
        self.update_image_display()

        # Enable heatmap toggle
        self.heatmap_toggle.setEnabled(True)

        # Load and display results
        results = self.load_results_for_image(image.id)

        if results:
            self.display_results(results)
        else:
            self.results_list.clear()
            self.results_list.addItem("No analysis available.")

        # If toggle is on, regenerate heatmap for new image
        if self.heatmap_toggle.isChecked():
            self.generate_and_show_heatmap()
        else:
            # Clear previous heatmap
            self.heatmap_label.setText("Toggle 'Show Heatmap Overlay' to visualize AI analysis")
            self.current_heatmap = None

    # ============================================================
    # TOGGLE HEATMAP
    # ============================================================
    def toggle_heatmap(self, state):
        """Toggle heatmap display on/off"""
        if not self.current_image_path:
            self.heatmap_toggle.setChecked(False)
            QMessageBox.warning(self, "No Image", "Please select an image first.")
            return

        if state == Qt.Checked:
            # Generate and show heatmap
            self.generate_and_show_heatmap()
        else:
            # Hide heatmap, show placeholder
            self.heatmap_label.setText("Heatmap overlay disabled\n\nToggle switch to re-enable")
            self.current_heatmap = None

    def generate_and_show_heatmap(self):
        """Generate Grad-CAM heatmap for current image"""
        if not self.current_image_path:
            return

        try:
            # Show loading message
            self.heatmap_label.setText("Generating heatmap...\nPlease wait")
            self.heatmap_label.repaint()  # Force immediate update
            
            print(f"Generating heatmap for: {self.current_image_path}")
            
            # Generate heatmap
            result = predict_with_heatmap(
                self.model,
                self.current_image_path,
                target_label=None  # Use top prediction
            )
            
            # Convert PIL Image to QPixmap
            heatmap_pil = result['heatmap_image']
            
            # Convert PIL to QPixmap
            img_byte_arr = io.BytesIO()
            heatmap_pil.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            qimg = QImage()
            qimg.loadFromData(img_byte_arr.read())
            self.current_heatmap = QPixmap.fromImage(qimg)
            
            # Display heatmap
            self.update_heatmap_display()
            
            print(f"Heatmap generated for: {result['target_condition']}")
            print(f"  Probability: {result['probability']*100:.1f}%")
            
        except Exception as e:
            self.heatmap_toggle.setChecked(False)
            QMessageBox.critical(
                self,
                "Heatmap Error",
                f"Failed to generate heatmap:\n{str(e)}"
            )
            print(f"Error: {e}")

    # ============================================================
    # HANDLE WINDOW RESIZE
    # ============================================================
    def resizeEvent(self, event):
        """Rescale images when window is resized"""
        self.update_image_display()
        self.update_heatmap_display()
        super().resizeEvent(event)

    # ============================================================
    # UPDATE DISPLAYS
    # ============================================================
    def update_image_display(self):
        """Scale and display original image"""
        if self.current_pixmap is None:
            return

        scaled = self.current_pixmap.scaled(
            self.scroll_area.viewport().size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.image_label.setPixmap(scaled)

    def update_heatmap_display(self):
        """Scale and display heatmap"""
        if self.current_heatmap is None:
            return

        scaled = self.current_heatmap.scaled(
            self.heatmap_scroll.viewport().size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.heatmap_label.setPixmap(scaled)

    # ============================================================
    # DISPLAY RESULTS
    # ============================================================
    def display_results(self, results):
        """Display ALL ranked AI findings"""
        self.results_list.clear()

        if not results or not isinstance(results, list):
            self.results_list.addItem("No results available.")
            return

        # Display ALL results (all 14 conditions)
        for i, r in enumerate(results):
            label = r.get("label", "Unknown")
            percentage = r.get("percentage")

            if percentage is None:
                prob = float(r.get("probability", 0.0))
                percentage = f"{prob * 100:.2f}%"

            # Add ranking indicators
            prefix = f"{i+1}. "

            self.results_list.addItem(f"{prefix}{label} — {percentage}")

    def load_results_for_image(self, image_id):
        """Load stored ML findings from database"""
        rows = self.findings_db.fetch_findings_for_image(image_id)

        results = []
        for label, probability, model_version, created_at in rows:
            results.append({
                "label": label,
                "probability": probability,
                "percentage": f"{probability * 100:.2f}%"
            })

        return results