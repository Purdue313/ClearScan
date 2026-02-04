from email.mime import image
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QListWidget,
    QMessageBox,
    QScrollArea
)
from PySide6.QtGui import QPixmap, QImageReader, QDragEnterEvent, QDropEvent
from PySide6.QtCore import Qt, QMimeData

from Database.imageDatabase import ImageDatabase
from Database.findingsDatabase import FindingsDatabase

from Source.Models.image_model import ImageRecord

from machineLearningModel.prediction import load_model, predict_xray, CHECKPOINT_PATH


class MainWindow(QWidget):
    def __init__(self):
        """
        Main application window for ClearScan.

        Responsibilities:
        - Provide UI for uploading and viewing medical images
        - Store image metadata in the database
        - Display images with correct orientation and scaling
        - Support drag-and-drop image uploads
        """
        print("Initializing MainWindow...")
        super().__init__()

        self.setWindowTitle("ClearScan Medical Imaging")

        # Enable drag-and-drop on the entire window
        self.setAcceptDrops(True)

        # ============================================================
        # DATABASE
        # ============================================================
        # Handles all SQLite interactions (insert / fetch)
        print("Initializing databases...")
        self.db = ImageDatabase()
        self.findings_db = FindingsDatabase()  # ML findings

        # Cache of ImageRecord objects currently displayed
        self.images = []

        # Stores the original pixmap so it can be re-scaled on resize
        self.current_pixmap = None

        # ============================================================
        # ML MODEL (LOADED ONCE)
        # ============================================================
        print("Loading ML model...")
        self.model = load_model(CHECKPOINT_PATH)
        print("Model loaded successfully!")

        # ============================================================
        # MAIN LAYOUT (LEFT: CONTROLS | RIGHT: IMAGE VIEWER)
        # ============================================================
        main_layout = QHBoxLayout(self)

        # ============================================================
        # LEFT PANEL: CONTROLS + IMAGE LIST
        # ============================================================
        left_panel = QVBoxLayout()

        # Upload button (manual file dialog)
        self.upload_btn = QPushButton("Upload Image")
        self.upload_btn.clicked.connect(self.upload_image)

        # Refresh / load images from database
        self.refresh_btn = QPushButton("Load Previous Scans")
        self.refresh_btn.clicked.connect(self.load_images)

        # List of stored scans
        self.image_list = QListWidget()
        self.image_list.itemSelectionChanged.connect(self.display_image)

        left_panel.addWidget(self.upload_btn)
        left_panel.addWidget(self.refresh_btn)
        left_panel.addWidget(QLabel("Stored Scans:"))
        left_panel.addWidget(self.image_list)
        
        
        # Ranked prediction results
        self.results_label = QLabel("AI Analysis Results:")
        self.results_list = QListWidget()

        left_panel.addWidget(self.results_label)
        left_panel.addWidget(self.results_list)

        # ============================================================
        # RIGHT PANEL: IMAGE VIEWER
        # ============================================================
        # QLabel is used to display the image
        self.image_label = QLabel("No image selected")

        # Center text / image inside the label
        self.image_label.setAlignment(Qt.AlignCenter)

        # Scroll area allows large images without distortion
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.image_label)
        
        # # Heatmap display
        # self.heatmap_label = QLabel("Heatmap (Top Finding)")
        # self.heatmap_label.setAlignment(Qt.AlignCenter)
        # self.heatmap_label.setFixedHeight(300)

        # right_panel = QVBoxLayout()
        # right_panel.addWidget(self.image_label)
        # right_panel.addWidget(self.heatmap_label)

        # self.scroll_area.setWidget(QWidget())
        # self.scroll_area.widget().setLayout(right_panel)


        # ============================================================
        # ADD PANELS TO MAIN LAYOUT
        # ============================================================
        main_layout.addLayout(left_panel, 1)
        main_layout.addWidget(self.scroll_area, 3)
        
        print("✓ MainWindow initialized successfully!")

    # ============================================================
    # IMAGE UPLOAD (FILE DIALOG)
    # ============================================================
    def upload_image(self):
        """
        Opens a file dialog to select an image.

        The file itself is NOT copied.
        Only the full path is stored in the database.
        """
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
        """
        Triggered when a dragged object enters the window.

        We accept the event only if:
        - It contains URLs (files)
        - At least one file is an image
        """
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.toLocalFile().lower().endswith((".png", ".jpg", ".jpeg")):
                    event.acceptProposedAction()
                    return

        event.ignore()

    def dropEvent(self, event: QDropEvent):
        """
        Triggered when files are dropped onto the window.

        Each valid image file:
        - Is registered in the database
        - Triggers a UI refresh
        """
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()

            if file_path.lower().endswith((".png", ".jpg", ".jpeg")):
                self.register_image(file_path)

    # ============================================================
    # SHARED IMAGE REGISTRATION LOGIC
    # ============================================================
    def register_image(self, file_path):
        """
        Registers an image, runs ML inference,
        stores findings in a separate database,
        and updates the UI.
        """
        try:
            # ============================================================
            # 1. STORE IMAGE
            # ============================================================
            image = ImageRecord.create(
                file_path=file_path,
                user="test_user"
            )
            image = self.db.insert_image(image)

            # ============================================================
            # 2. RUN ML INFERENCE
            # ============================================================
            results = predict_xray(self.model, file_path)

            # ============================================================
            # 3. STORE FINDINGS (TOP RESULT ONLY)
            # ============================================================
            top = results["top_prediction"]

            self.findings_db.insert_findings([{
                "image_id": image.id,
                "label": top["label"],
                "probability": top["probability"],
                "model_version": "chexpert_resnet50_v1"
            }])


            # ============================================================
            # 4. UPDATE UI
            # ============================================================
            self.load_images()
            self.display_results(results)

            # QMessageBox.information(
            #     self,
            #     "Upload Complete",
            #     "Image uploaded and analyzed successfully."
            # )

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
        """
        Fetches all stored image metadata and populates the list widget.
        """
        self.image_list.clear()
        self.images = self.db.fetch_all_images()

        for img in self.images:
            self.image_list.addItem(
                f"ID {img.id} | {img.uploaded_at} | {img.user}"
            )

    # ============================================================
    # IMAGE LOADING WITH EXIF ORIENTATION
    # ============================================================
    def load_pixmap_with_orientation(self, file_path):
        """
        Loads an image from disk and applies EXIF orientation automatically.

        This is CRITICAL for medical images and phone photos where
        orientation may be stored in metadata instead of pixel layout.

        Returns:
            QPixmap or None if loading fails
        """
        reader = QImageReader(file_path)

        # This tells Qt to apply EXIF rotation and mirroring
        reader.setAutoTransform(True)

        image = reader.read()

        if image.isNull():
            return None

        return QPixmap.fromImage(image)

    # ============================================================
    # DISPLAY IMAGE FROM LIST SELECTION
    # ============================================================
    def display_image(self):
        """
        Displays the selected image and its associated ML findings.
        """
        index = self.image_list.currentRow()

        if index < 0:
            return

        image = self.images[index]

        # ==============================
        # LOAD AND DISPLAY IMAGE
        # ==============================
        pixmap = self.load_pixmap_with_orientation(image.file_path)

        if pixmap is None:
            QMessageBox.warning(
                self,
                "Error",
                f"Unable to load image:\n{image.file_path}\n\n"
                "The file may have been moved or deleted."
            )
            return

        self.current_pixmap = pixmap
        self.update_image_display()

        # ==============================
        # LOAD STORED FINDINGS
        # ==============================
        results = self.load_results_for_image(image.id)

        if results:
            self.display_results(results)
        else:
            self.results_list.clear()
            self.results_list.addItem("No analysis available for this image.")


    # ============================================================
    # HANDLE WINDOW RESIZE
    # ============================================================
    def resizeEvent(self, event):
        """
        Automatically called by Qt when the window is resized.

        We rescale the image so that:
        - Aspect ratio is preserved
        - Portrait and landscape images behave correctly
        """
        self.update_image_display()
        super().resizeEvent(event)

    # ============================================================
    # SCALE AND DISPLAY IMAGE
    # ============================================================
    def update_image_display(self):
        """
        Scales the current image to fit the visible area.

        Important details:
        - Uses the scroll area's *viewport* size
        - Preserves aspect ratio
        - Smooth transformation prevents pixelation
        """
        if self.current_pixmap is None:
            return

        scaled = self.current_pixmap.scaled(
            self.scroll_area.viewport().size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.image_label.setPixmap(scaled)
        
    # ============================================================
    # DISPLAY AI PREDICTION RESULTS
    # ============================================================
    def display_results(self, results):
        """
        Displays ranked AI findings for an image.

        Expected format:
            results = [
                {
                    "label": str,
                    "probability": float,
                    "percentage": str
                },
                ...
            ]
        """
        self.results_list.clear()

        if not results or not isinstance(results, list):
            self.results_list.addItem("No results available.")
            return

        for r in results:
            label = r.get("label", "Unknown")
            percentage = r.get("percentage")

            # Fallback if percentage string is missing
            if percentage is None:
                prob = float(r.get("probability", 0.0))
                percentage = f"{prob * 100:.2f}%"

            self.results_list.addItem(f"{label} — {percentage}")
            
    # ============================================================
    # DISPLAY AI PREDICTION RESULTS FROM DATABASE
    # ============================================================
    def load_results_for_image(self, image_id):
        """
        Loads stored ML findings for an image from the findings database
        and converts them into the display_results() format.
        """
        rows = self.findings_db.fetch_findings_for_image(image_id)

        results = []
        for label, probability, model_version, created_at in rows:
            results.append({
                "label": label,
                "probability": probability,
                "percentage": f"{probability * 100:.2f}%"
            })

        return results

