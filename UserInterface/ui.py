"""
ui.py

This module defines the graphical user interface for the ClearScan
Medical Imaging prototype using PySide6.

Design goals:
- Correct handling of both portrait and landscape images
- Automatic EXIF orientation correction
- Aspect-ratio-preserving image display
- Store full file paths without copying files
- Clean separation between controls and image viewer
- Clear, maintainable code suitable for a senior project
"""

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
from PySide6.QtGui import QPixmap, QImageReader
from PySide6.QtCore import Qt

from Database.database import Database
from Source.image_model import ImageRecord


class MainWindow(QWidget):
    """
    MainWindow defines the primary application window.

    Responsibilities:
    - Provide UI controls for uploading and selecting images
    - Display stored medical images accurately with correct orientation
    - Coordinate UI actions with database operations
    """

    def __init__(self):
        """
        Initialize the UI, database connection, and layout.
        """
        super().__init__()

        self.setWindowTitle("ClearScan Medical Imaging")
        self.resize(1100, 650)

        # Initialize database access layer
        self.db = Database()

        # Cache of ImageRecord objects currently displayed
        self.images = []

        # ============================================================
        # MAIN LAYOUT (HORIZONTAL SPLIT)
        # ============================================================
        # Left side: controls + image list
        # Right side: image display area
        main_layout = QHBoxLayout(self)

        # ============================================================
        # LEFT PANEL: CONTROLS AND IMAGE LIST
        # ============================================================
        left_panel = QVBoxLayout()

        # Button for uploading a new image
        self.upload_btn = QPushButton("Upload Image")
        self.upload_btn.clicked.connect(self.upload_image)

        # Button for loading previously uploaded images
        self.refresh_btn = QPushButton("Load Previous Scans")
        self.refresh_btn.clicked.connect(self.load_images)

        # List widget showing stored images
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
        self.image_label.setAlignment(Qt.AlignCenter)

        # Scroll area allows large images to be viewed without distortion
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.image_label)

        # ============================================================
        # ADD PANELS TO MAIN LAYOUT
        # ============================================================
        # Stretch factors control how much space each panel gets
        main_layout.addLayout(left_panel, 1)
        main_layout.addWidget(self.scroll_area, 3)

        # Track currently displayed pixmap for proper resizing
        self.current_pixmap = None

    # ============================================================
    # IMAGE UPLOAD WORKFLOW (UC-200)
    # ============================================================
    def upload_image(self):
        """
        Opens a file dialog allowing the user to select an image.
        The full path to the image is stored in the database without
        copying the file.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.png *.jpg *.jpeg)"
        )

        # User cancelled the dialog
        if not file_path:
            return

        # Create metadata object with full path
        image = ImageRecord.create(
            file_path=file_path,  # Store the full original path
            user="test_user"
        )

        # Persist metadata to database
        self.db.insert_image(image)

        QMessageBox.information(
            self,
            "Upload Complete",
            "Image path registered successfully."
        )

        # Refresh list of stored images
        self.load_images()

    # ============================================================
    # LOAD PREVIOUS SCANS (UC-500)
    # ============================================================
    def load_images(self):
        """
        Retrieves all stored image metadata from the database and
        displays it in the list widget.
        """
        self.image_list.clear()
        self.images = self.db.fetch_all_images()

        for img in self.images:
            self.image_list.addItem(
                f"ID {img.id} | {img.uploaded_at} | {img.user}"
            )

    # ============================================================
    # IMAGE DISPLAY LOGIC WITH EXIF ORIENTATION
    # ============================================================
    def load_pixmap_with_orientation(self, file_path):
        """
        Loads an image and applies the correct orientation based on
        EXIF metadata.

        EXIF orientation values:
        1: Normal (no rotation needed)
        2: Flip horizontal
        3: Rotate 180°
        4: Flip vertical
        5: Transpose (flip horizontal + rotate 90° CCW)
        6: Rotate 90° CW
        7: Transverse (flip horizontal + rotate 90° CW)
        8: Rotate 90° CCW

        Args:
            file_path: Full path to the image file

        Returns:
            QPixmap: Properly oriented image, or None if loading fails
        """
        # Use QImageReader to access EXIF data
        reader = QImageReader(file_path)
        reader.setAutoTransform(True)  # Automatically apply EXIF orientation
        
        image = reader.read()
        
        if image.isNull():
            return None
        
        return QPixmap.fromImage(image)

    def display_image(self):
        """
        Displays the currently selected image in the viewer.

        This method:
        - Loads the image from its original location using the stored path
        - Applies correct EXIF orientation
        - Stores the original QPixmap
        - Scales it to fit the viewer while preserving aspect ratio
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
                f"Unable to load image from:\n{image.file_path}\n\n"
                "The file may have been moved or deleted."
            )
            return

        # Store the original pixmap for future rescaling
        self.current_pixmap = pixmap

        self.update_image_display()

    def resizeEvent(self, event):
        """
        Qt automatically calls this method whenever the window
        is resized.

        We override it so that the image is rescaled dynamically
        while maintaining correct orientation and aspect ratio.
        """
        self.update_image_display()
        super().resizeEvent(event)

    def update_image_display(self):
        """
        Scales and displays the currently loaded image.

        Key points:
        - EXIF orientation has already been applied
        - Aspect ratio is preserved
        - Both portrait and landscape images display correctly
        """
        if self.current_pixmap is None:
            return

        scaled = self.current_pixmap.scaled(
            self.scroll_area.viewport().size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.image_label.setPixmap(scaled)