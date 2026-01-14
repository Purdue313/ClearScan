from dataclasses import dataclass
from datetime import datetime

@dataclass
class ImageRecord:
    id: int | None
    file_path: str
    uploaded_at: str
    user: str

    @staticmethod
    def create(file_path: str, user: str):
        return ImageRecord(
            id=None,
            file_path=file_path,
            uploaded_at=datetime.now().isoformat(timespec="seconds"),
            user=user
        )
