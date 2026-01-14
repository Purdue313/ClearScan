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

from Database.database import Database
from Source.Models.image_model import ImageRecord


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
        super().__init__()

        self.setWindowTitle("ClearScan Medical Imaging")

        # Enable drag-and-drop on the entire window
        self.setAcceptDrops(True)

        # ============================================================
        # DATABASE
        # ============================================================
        # Handles all SQLite interactions (insert / fetch)
        self.db = Database()

        # Cache of ImageRecord objects currently displayed
        self.images = []

        # Stores the original pixmap so it can be re-scaled on resize
        self.current_pixmap = None

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

        # ============================================================
        # ADD PANELS TO MAIN LAYOUT
        # ============================================================
        main_layout.addLayout(left_panel, 1)
        main_layout.addWidget(self.scroll_area, 3)

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
        Centralized image registration logic.

        Used by:
        - File dialog uploads
        - Drag-and-drop uploads
        """
        image = ImageRecord.create(
            file_path=file_path,
            user="test_user"
        )

        self.db.insert_image(image)

        QMessageBox.information(
            self,
            "Upload Complete",
            "Image registered successfully."
        )

        self.load_images()

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
        Displays the selected image from the list.

        Steps:
        - Load image using stored file path
        - Apply EXIF orientation
        - Cache the original pixmap
        - Scale it to fit the viewport
        """
        index = self.image_list.currentRow()

        if index < 0:
            return

        image = self.images[index]
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
