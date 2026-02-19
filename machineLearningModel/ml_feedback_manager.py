"""
ml_feedback_manager.py

Handles the feedback loop between doctor diagnoses and ML model improvement.

Responsibilities:
- Export doctor-verified feedback as a training CSV
- Compute per-label accuracy stats from accumulated feedback
- Trigger fine-tuning of the ResNet model on verified data
- Report on accuracy improvements before/after training
"""

import os
import csv
import json
from datetime import datetime
from pathlib import Path

from Database.findingsDatabase import FindingsDatabase
from Database.imageDatabase import ImageDatabase


# ============================================================
# PATHS
# ============================================================
FEEDBACK_EXPORT_DIR = Path("feedback_exports")
RETRAIN_LOG_PATH    = Path("feedback_exports/retrain_log.json")


# ============================================================
# FEEDBACK MANAGER
# ============================================================
class MLFeedbackManager:
    """
    Manages the doctor feedback → ML training pipeline.

    Usage (from MainWindow or a background thread):

        manager = MLFeedbackManager(findings_db, image_db)
        stats   = manager.get_accuracy_stats()
        report  = manager.export_feedback_csv()
        result  = manager.run_training_update(model)
    """

    def __init__(self, findings_db: FindingsDatabase, image_db: ImageDatabase):
        self.findings_db = findings_db
        self.image_db    = image_db
        FEEDBACK_EXPORT_DIR.mkdir(exist_ok=True)

    # ============================================================
    # ACCURACY STATS
    # ============================================================
    def get_accuracy_stats(self) -> list[dict]:
        """
        Returns per-label accuracy stats from all submitted feedback.

        Returns:
            list of dicts: label, total, correct, accuracy_pct
        """
        rows = self.findings_db.fetch_feedback_stats()
        return [
            {
                "label":        r[0],
                "total":        r[1],
                "correct":      r[2],
                "accuracy_pct": r[3],
            }
            for r in rows
        ]

    def get_overall_accuracy(self) -> float | None:
        """
        Returns overall accuracy across all labels and feedback submissions.
        Returns None if no feedback exists yet.
        """
        stats = self.get_accuracy_stats()
        if not stats:
            return None
        total   = sum(s["total"]   for s in stats)
        correct = sum(s["correct"] for s in stats)
        return round((correct / total) * 100, 1) if total > 0 else None

    # ============================================================
    # EXPORT FEEDBACK CSV
    # ============================================================
    def export_feedback_csv(self) -> Path:
        """
        Exports all unprocessed doctor feedback as a CSV that can be used
        to fine-tune or re-evaluate the model.

        CSV columns:
            image_id, image_path, label, correct, model_version, created_at

        Returns:
            Path to the written CSV file.
        """
        feedback_rows = self.findings_db.fetch_unprocessed_feedback()
        if not feedback_rows:
            return None

        # Build image_id -> file_path lookup
        all_images  = self.image_db.fetch_all_images()
        path_lookup = {img.id: img.file_path for img in all_images}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path  = FEEDBACK_EXPORT_DIR / f"feedback_{timestamp}.csv"

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["feedback_id", "image_id", "image_path",
                 "label", "correct", "model_version", "created_at"]
            )
            for row in feedback_rows:
                fb_id, image_id, label, correct, model_version, created_at = row
                image_path = path_lookup.get(image_id, "FILE_NOT_FOUND")
                writer.writerow([
                    fb_id, image_id, image_path,
                    label, correct, model_version, created_at
                ])

        return csv_path

    # ============================================================
    # RUN TRAINING UPDATE
    # ============================================================
    def run_training_update(self, model) -> dict:
        """
        Runs a lightweight fine-tuning pass on the ResNet model using
        doctor-verified feedback data.

        Steps:
        1.  Export the feedback CSV
        2.  Build a feedback dataset (image_path, correct_label pairs)
        3.  Fine-tune the final classification head for a small number of epochs
        4.  Evaluate accuracy on the feedback set (before vs after)
        5.  Save an updated checkpoint if accuracy improved
        6.  Log the result to retrain_log.json

        Args:
            model: the loaded PyTorch model (from load_model())

        Returns:
            dict with keys: success, accuracy_before, accuracy_after,
                            checkpoint_saved, csv_path, message
        """
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import Dataset, DataLoader
            from torchvision import transforms
            from PIL import Image as PILImage

            from machineLearningModel.prediction import CHECKPOINT_PATH, LABELS as CONDITIONS

            # ── 1. Gather feedback data ──────────────────────────
            feedback_rows = self.findings_db.fetch_unprocessed_feedback()
            if not feedback_rows:
                return {
                    "success": False,
                    "message": "No feedback available for training.",
                }

            all_images  = self.image_db.fetch_all_images()
            path_lookup = {img.id: img.file_path for img in all_images}

            # Build (image_path, label_index, correct) triples
            # We only train on rows where the doctor confirmed a label (correct=1)
            train_samples = []
            for fb_id, image_id, label, correct, model_version, created_at in feedback_rows:
                if correct == 1 and label in CONDITIONS:
                    img_path = path_lookup.get(image_id)
                    if img_path and os.path.exists(img_path):
                        label_idx = CONDITIONS.index(label)
                        train_samples.append((img_path, label_idx))

            if not train_samples:
                return {
                    "success": False,
                    "message": "No confirmed-positive feedback samples found.",
                }

            # ── 2. Dataset ───────────────────────────────────────
            # Build multi-hot label vectors to match BCEWithLogitsLoss
            # used in the original trainer.py
            num_labels = len(CONDITIONS)

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])

            class FeedbackDataset(Dataset):
                def __init__(self, samples, transform, num_labels):
                    self.samples    = samples
                    self.transform  = transform
                    self.num_labels = num_labels

                def __len__(self):
                    return len(self.samples)

                def __getitem__(self, idx):
                    path, label_idx = self.samples[idx]
                    img = PILImage.open(path).convert("RGB")
                    # Multi-hot vector: 1.0 at the confirmed label, 0.0 elsewhere
                    target = torch.zeros(self.num_labels, dtype=torch.float32)
                    target[label_idx] = 1.0
                    return self.transform(img), target

            dataset    = FeedbackDataset(train_samples, transform, num_labels)
            dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            # ── 3. Evaluate BEFORE ───────────────────────────────
            acc_before = self._evaluate(model, dataloader, device)

            # ── 4. Fine-tune classification head only ─────────────
            # Freeze all layers except the final FC layer
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True

            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.fc.parameters(), lr=1e-4)

            EPOCHS = 3
            model.train()
            for epoch in range(EPOCHS):
                for images, labels in dataloader:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss    = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

            # Re-enable all gradients for future inference
            for param in model.parameters():
                param.requires_grad = True

            # ── 5. Evaluate AFTER ────────────────────────────────
            model.eval()
            acc_after = self._evaluate(model, dataloader, device)

            # ── 6. Save checkpoint if improved ───────────────────
            checkpoint_saved = False
            if acc_after >= acc_before:
                timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_ckpt    = Path(CHECKPOINT_PATH).parent / f"checkpoint_ft_{timestamp}.pth"
                torch.save(model.state_dict(), new_ckpt)
                checkpoint_saved = True

            # ── 7. Write retrain log ─────────────────────────────
            log_entry = {
                "timestamp":        datetime.now().isoformat(),
                "samples_used":     len(train_samples),
                "epochs":           EPOCHS,
                "accuracy_before":  acc_before,
                "accuracy_after":   acc_after,
                "checkpoint_saved": checkpoint_saved,
            }
            self._append_log(log_entry)

            return {
                "success":          True,
                "accuracy_before":  acc_before,
                "accuracy_after":   acc_after,
                "checkpoint_saved": checkpoint_saved,
                "samples_used":     len(train_samples),
                "message": (
                    f"Training complete. Accuracy: "
                    f"{acc_before:.1f}% → {acc_after:.1f}%"
                ),
            }

        except Exception as e:
            return {"success": False, "message": str(e)}

    # ============================================================
    # HELPERS
    # ============================================================
    @staticmethod
    def _evaluate(model, dataloader, device) -> float:
        """
        Returns accuracy % on the feedback set.
        Uses sigmoid + threshold since the model is multi-label (BCEWithLogitsLoss).
        A prediction is 'correct' if the confirmed label's sigmoid output >= 0.5.
        """
        import torch
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, targets in dataloader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                probs   = torch.sigmoid(outputs)          # [B, 14]
                preds   = (probs >= 0.5).float()          # [B, 14]
                # Count a sample correct if every label matches
                correct += (preds == targets).all(dim=1).sum().item()
                total   += targets.size(0)
        return round((correct / total * 100), 1) if total > 0 else 0.0

    def _append_log(self, entry: dict):
        """Appends a training run entry to the JSON log file."""
        log = []
        if RETRAIN_LOG_PATH.exists():
            try:
                with open(RETRAIN_LOG_PATH) as f:
                    log = json.load(f)
            except (json.JSONDecodeError, IOError):
                log = []
        log.append(entry)
        with open(RETRAIN_LOG_PATH, "w") as f:
            json.dump(log, f, indent=2)

    def write_stats_log(self):
        """
        Writes current per-label accuracy stats to the retrain log JSON.
        Called immediately after each feedback submission so the file
        exists and stays current even without a full retrain.
        """
        stats   = self.get_accuracy_stats()
        overall = self.get_overall_accuracy()
        entry   = {
            "timestamp":        datetime.now().isoformat(),
            "type":             "feedback_stats",
            "overall_accuracy": overall,
            "per_label":        stats,
        }
        self._append_log(entry)

    def get_training_history(self) -> list[dict]:
        """Returns the full history of retraining runs."""
        if not RETRAIN_LOG_PATH.exists():
            return []
        try:
            with open(RETRAIN_LOG_PATH) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []