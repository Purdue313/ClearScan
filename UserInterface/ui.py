import os
import shutil
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QListWidget, QMessageBox
)
from PySide6.QtGui import QPixmap
from Database.database import Database
from Source.image_model import ImageRecord

STORAGE_DIR = "storage"

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ClearScan Medical Imaging")

        os.makedirs(STORAGE_DIR, exist_ok=True)
        self.db = Database()

        self.layout = QVBoxLayout()

        self.upload_btn = QPushButton("Upload Image")
        self.upload_btn.clicked.connect(self.upload_image)

        self.refresh_btn = QPushButton("Load Previous Scans")
        self.refresh_btn.clicked.connect(self.load_images)

        self.image_list = QListWidget()
        self.image_list.itemClicked.connect(self.display_image)

        self.image_label = QLabel("No image selected")
        self.image_label.setFixedHeight(300)

        self.layout.addWidget(self.upload_btn)
        self.layout.addWidget(self.refresh_btn)
        self.layout.addWidget(self.image_list)
        self.layout.addWidget(self.image_label)

        self.setLayout(self.layout)

        self.images = []

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg)"
        )

        if not file_path:
            return

        filename = os.path.basename(file_path)
        dest_path = os.path.join(STORAGE_DIR, filename)

        shutil.copy(file_path, dest_path)

        image = ImageRecord.create(dest_path, user="test_user")
        self.db.insert_image(image)

        QMessageBox.information(self, "Upload Complete", "Image uploaded successfully.")
        self.load_images()

    def load_images(self):
        self.image_list.clear()
        self.images = self.db.fetch_all_images()

        for img in self.images:
            self.image_list.addItem(
                f"ID {img.id} | {img.uploaded_at} | {img.user}"
            )

    def display_image(self):
        index = self.image_list.currentRow()
        image = self.images[index]

        pixmap = QPixmap(image.file_path)
        if pixmap.isNull():
            QMessageBox.warning(self, "Error", "Unable to load image.")
            return

        self.image_label.setPixmap(
            pixmap.scaled(
                self.image_label.width(),
                self.image_label.height()
            )
        )
