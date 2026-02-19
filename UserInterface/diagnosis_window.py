from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QListWidget, QListWidgetItem, QTextEdit, QMessageBox,
    QScrollArea, QFrame, QSizePolicy
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QColor

from Database.findingsDatabase import FindingsDatabase
from Database.imageDatabase import ImageDatabase
from machineLearningModel.ml_feedback_manager import MLFeedbackManager


class FindingRow(QFrame):
    """
    A single expandable row for one ML finding.
    Shows label + probability bar, expands to show rationale
    and confirm / reject buttons.
    """

    confirmed = Signal(str)   # emits the confirmed label
    rejected  = Signal(str)   # emits the rejected  label

    # Rationale copy per condition (expand this dict as the model grows)
    RATIONALE = {
        "Pneumonia":           "Increased opacity in the lower lobes suggests bacterial or viral pneumonia.",
        "Cardiomegaly":        "Cardiac silhouette exceeds 50% of the thoracic width on PA view.",
        "Pleural Effusion":    "Blunting of the costophrenic angle is consistent with fluid accumulation.",
        "Atelectasis":         "Linear or plate-like opacities indicate partial lung collapse.",
        "Consolidation":       "Homogeneous airspace opacity with air bronchograms detected.",
        "Edema":               "Perihilar haziness and Kerley B lines suggest pulmonary oedema.",
        "Pneumothorax":        "Absence of lung markings at the periphery with visible pleural line.",
        "Fracture":            "Cortical discontinuity detected in the visualised osseous structures.",
        "Nodule":              "Focal rounded opacity < 3 cm. Follow-up CT recommended.",
        "Mass":                "Focal rounded opacity ≥ 3 cm. Malignancy must be excluded.",
        "Infiltrate":          "Ill-defined airspace opacity may represent infection or inflammation.",
        "Emphysema":           "Hyperinflation with flattened diaphragm and increased retrosternal space.",
        "Fibrosis":            "Reticular opacities and volume loss consistent with fibrotic change.",
        "Pleural Thickening":  "Irregular soft-tissue density along the pleural surface.",
        "No Finding":          "No significant acute cardiothoracic abnormality identified.",
    }

    def __init__(self, rank, label, probability, already_confirmed=False, already_rejected=False):
        super().__init__()
        self.label       = label
        self.probability = probability
        self._expanded   = False

        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                border: 1px solid #ddd;
                border-radius: 6px;
                background: #fff;
                margin: 2px 0;
            }
        """)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 8, 10, 8)
        outer.setSpacing(4)

        # ── Header row ──────────────────────────────────────────
        header = QHBoxLayout()

        rank_lbl = QLabel(f"{rank}.")
        rank_lbl.setFixedWidth(24)
        rank_lbl.setStyleSheet("color:#888; font-size:12px;")

        self.name_lbl = QLabel(label)
        self.name_lbl.setFont(QFont("Arial", 11, QFont.Bold))

        pct = probability * 100
        self.pct_lbl = QLabel(f"{pct:.1f}%")
        self.pct_lbl.setStyleSheet("color:#555; font-size:11px; min-width:46px;")
        self.pct_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.expand_btn = QPushButton("▸ Details")
        self.expand_btn.setFixedWidth(80)
        self.expand_btn.setStyleSheet("""
            QPushButton {
                background: transparent; border: 1px solid #aaa;
                border-radius: 4px; font-size: 11px; padding: 2px 6px;
            }
            QPushButton:hover { background: #f0f0f0; }
        """)
        self.expand_btn.clicked.connect(self._toggle_expand)

        header.addWidget(rank_lbl)
        header.addWidget(self.name_lbl, 1)
        header.addWidget(self.pct_lbl)
        header.addWidget(self.expand_btn)

        # ── Probability bar ──────────────────────────────────────
        bar_bg = QFrame()
        bar_bg.setFixedHeight(6)
        bar_bg.setStyleSheet("background:#eee; border-radius:3px; border:none;")
        bar_layout = QHBoxLayout(bar_bg)
        bar_layout.setContentsMargins(0, 0, 0, 0)
        bar_layout.setSpacing(0)

        fill_w = max(4, int(pct))
        self.bar_fill = QFrame()
        self.bar_fill.setFixedHeight(6)
        color = "#e53935" if pct >= 60 else "#fb8c00" if pct >= 30 else "#43a047"
        self.bar_fill.setStyleSheet(
            f"background:{color}; border-radius:3px; border:none;"
        )
        bar_layout.addWidget(self.bar_fill, fill_w)
        bar_layout.addStretch(100 - fill_w)

        # ── Detail panel (hidden by default) ────────────────────
        self.detail_panel = QWidget()
        self.detail_panel.setVisible(False)
        detail_layout = QVBoxLayout(self.detail_panel)
        detail_layout.setContentsMargins(24, 4, 0, 4)
        detail_layout.setSpacing(6)

        rationale_text = self.RATIONALE.get(label, "No additional rationale available.")
        rationale_lbl = QLabel(rationale_text)
        rationale_lbl.setWordWrap(True)
        rationale_lbl.setStyleSheet("color:#444; font-size:11px;")

        btn_row = QHBoxLayout()
        self.confirm_btn = QPushButton("✔ Confirm Diagnosis")
        self.confirm_btn.setStyleSheet("""
            QPushButton {
                background:#2e7d32; color:white; border:none;
                border-radius:4px; padding:5px 14px; font-size:12px;
            }
            QPushButton:hover { background:#1b5e20; }
            QPushButton:disabled { background:#aaa; }
        """)
        self.confirm_btn.clicked.connect(lambda: self.confirmed.emit(self.label))

        self.reject_btn = QPushButton("✘ Mark Incorrect")
        self.reject_btn.setStyleSheet("""
            QPushButton {
                background:#c62828; color:white; border:none;
                border-radius:4px; padding:5px 14px; font-size:12px;
            }
            QPushButton:hover { background:#b71c1c; }
            QPushButton:disabled { background:#aaa; }
        """)
        self.reject_btn.clicked.connect(lambda: self.rejected.emit(self.label))

        btn_row.addWidget(self.confirm_btn)
        btn_row.addWidget(self.reject_btn)
        btn_row.addStretch()

        detail_layout.addWidget(rationale_lbl)
        detail_layout.addLayout(btn_row)

        outer.addLayout(header)
        outer.addWidget(bar_bg)
        outer.addWidget(self.detail_panel)

        # Apply existing state if re-opening a previously diagnosed image
        if already_confirmed:
            self._mark_confirmed()
        elif already_rejected:
            self._mark_rejected()

    def _toggle_expand(self):
        self._expanded = not self._expanded
        self.detail_panel.setVisible(self._expanded)
        self.expand_btn.setText("▾ Details" if self._expanded else "▸ Details")

    def _mark_confirmed(self):
        self.name_lbl.setStyleSheet("color:#2e7d32; font-weight:bold;")
        self.confirm_btn.setEnabled(False)
        self.reject_btn.setEnabled(False)
        self.confirm_btn.setText("✔ Confirmed")

    def _mark_rejected(self):
        self.name_lbl.setStyleSheet("color:#c62828; text-decoration:line-through;")
        self.confirm_btn.setEnabled(False)
        self.reject_btn.setEnabled(False)
        self.reject_btn.setText("✘ Rejected")


class DiagnosisWindow(QWidget):
    """
    Doctor-facing panel for FR 3.6.1.

    Shows:
    - All ML findings as expandable rows with rationale
    - Confirm / Reject buttons per finding
    - Doctor notes text box
    - Submit Feedback button (saves to DB + triggers ML feedback)
    - Diagnosed badge if image has already been reviewed
    """

    diagnosis_saved = Signal(int)   # emits image_id when saved

    def __init__(self, findings_db: FindingsDatabase, image_db: ImageDatabase, model=None, parent=None):
        super().__init__(parent)
        self.findings_db      = findings_db
        self.image_db         = image_db
        self.model            = model
        self.feedback_manager = MLFeedbackManager(findings_db, image_db)
        self.image_id         = None
        self.findings         = []
        self.confirmed_label  = None
        self.rejected_labels  = set()
        self._rows: list[FindingRow] = []

        self._build_ui()

    # ============================================================
    # UI CONSTRUCTION
    # ============================================================
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # ── Title bar ───────────────────────────────────────────
        title_row = QHBoxLayout()
        title = QLabel("Diagnosis Review")
        title.setFont(QFont("Arial", 14, QFont.Bold))

        self.status_badge = QLabel()
        self.status_badge.setFixedHeight(24)
        self.status_badge.setStyleSheet(
            "padding:2px 10px; border-radius:10px; font-size:11px; font-weight:bold;"
        )
        self.status_badge.setVisible(False)

        title_row.addWidget(title)
        title_row.addStretch()
        title_row.addWidget(self.status_badge)
        root.addLayout(title_row)

        # ── Placeholder when no image selected ──────────────────
        self.placeholder = QLabel("Select an image to begin diagnosis review.")
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setStyleSheet("color:#999; font-size:13px;")
        root.addWidget(self.placeholder)

        # ── Scroll area for finding rows ─────────────────────────
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setVisible(False)
        self.scroll.setStyleSheet("QScrollArea { border: none; }")

        self.findings_container = QWidget()
        self.findings_layout    = QVBoxLayout(self.findings_container)
        self.findings_layout.setContentsMargins(0, 0, 0, 0)
        self.findings_layout.setSpacing(4)
        self.findings_layout.addStretch()

        self.scroll.setWidget(self.findings_container)
        root.addWidget(self.scroll, 1)

        # ── Doctor notes ─────────────────────────────────────────
        self.notes_label = QLabel("Doctor Notes:")
        self.notes_label.setStyleSheet("font-weight:bold; font-size:12px;")
        self.notes_label.setVisible(False)

        self.notes_box = QTextEdit()
        self.notes_box.setPlaceholderText(
            "Add any clinical observations, caveats, or follow-up instructions…"
        )
        self.notes_box.setFixedHeight(90)
        self.notes_box.setVisible(False)

        root.addWidget(self.notes_label)
        root.addWidget(self.notes_box)

        # ── Action buttons ───────────────────────────────────────
        btn_row = QHBoxLayout()

        self.save_btn = QPushButton("💾  Save Diagnosis")
        self.save_btn.setFixedHeight(38)
        self.save_btn.setStyleSheet("""
            QPushButton {
                background:#1565c0; color:white; border:none;
                border-radius:5px; font-size:13px; font-weight:bold;
            }
            QPushButton:hover   { background:#0d47a1; }
            QPushButton:disabled { background:#aaa; }
        """)
        self.save_btn.setEnabled(False)
        self.save_btn.setVisible(False)
        self.save_btn.clicked.connect(self._save_diagnosis)

        self.feedback_btn = QPushButton("📤  Submit ML Feedback")
        self.feedback_btn.setFixedHeight(38)
        self.feedback_btn.setStyleSheet("""
            QPushButton {
                background:#6a1b9a; color:white; border:none;
                border-radius:5px; font-size:13px; font-weight:bold;
            }
            QPushButton:hover    { background:#4a148c; }
            QPushButton:disabled { background:#aaa; }
        """)
        self.feedback_btn.setEnabled(False)
        self.feedback_btn.setVisible(False)
        self.feedback_btn.clicked.connect(self._submit_feedback)

        self.retrain_btn = QPushButton("🧠  Retrain Model")
        self.retrain_btn.setFixedHeight(38)
        self.retrain_btn.setStyleSheet("""
            QPushButton {
                background:#e65100; color:white; border:none;
                border-radius:5px; font-size:13px; font-weight:bold;
            }
            QPushButton:hover    { background:#bf360c; }
            QPushButton:disabled { background:#aaa; }
        """)
        self.retrain_btn.setEnabled(False)
        self.retrain_btn.setVisible(False)
        self.retrain_btn.clicked.connect(self._run_retrain)

        btn_row.addWidget(self.save_btn)
        btn_row.addWidget(self.feedback_btn)
        btn_row.addWidget(self.retrain_btn)
        root.addLayout(btn_row)

    # ============================================================
    # PUBLIC: load findings for a selected image
    # ============================================================
    def load_image(self, image_id: int):
        """Call this whenever the user selects an image."""
        self.image_id        = image_id
        self.confirmed_label = None
        self.rejected_labels = set()

        # Fetch ML findings
        rows = self.findings_db.fetch_findings_for_image(image_id)
        self.findings = [
            {"label": r[0], "probability": r[1], "model_version": r[2]}
            for r in rows
        ]

        # Fetch any existing diagnosis
        existing = self.findings_db.fetch_diagnosis(image_id)

        self._rebuild_rows(existing)
        self._show_content(True)

        if existing:
            self.confirmed_label = existing["confirmed_label"]
            self.rejected_labels = set(existing["rejected_labels"])
            self.notes_box.setPlainText(existing["doctor_notes"] or "")
            self._set_badge("✔ Diagnosed", "#2e7d32")
            self.save_btn.setEnabled(True)
            if not existing["feedback_submitted"]:
                self.feedback_btn.setEnabled(True)
        else:
            self.notes_box.clear()
            self._set_badge(None, None)
            self.save_btn.setEnabled(False)
            self.feedback_btn.setEnabled(False)

    # ============================================================
    # PRIVATE: rebuild finding rows
    # ============================================================
    def _rebuild_rows(self, existing_diagnosis):
        # Clear old rows
        for row in self._rows:
            row.setParent(None)
        self._rows.clear()

        confirmed = existing_diagnosis["confirmed_label"] if existing_diagnosis else None
        rejected  = set(existing_diagnosis["rejected_labels"]) if existing_diagnosis else set()

        for i, finding in enumerate(self.findings):
            label = finding["label"]
            prob  = finding["probability"]

            row = FindingRow(
                rank=i + 1,
                label=label,
                probability=prob,
                already_confirmed=(label == confirmed),
                already_rejected=(label in rejected),
            )
            row.confirmed.connect(self._on_confirm)
            row.rejected.connect(self._on_reject)

            # Insert before the trailing stretch
            self.findings_layout.insertWidget(
                self.findings_layout.count() - 1, row
            )
            self._rows.append(row)

    def _show_content(self, visible: bool):
        self.placeholder.setVisible(not visible)
        self.scroll.setVisible(visible)
        self.notes_label.setVisible(visible)
        self.notes_box.setVisible(visible)
        self.save_btn.setVisible(visible)
        self.feedback_btn.setVisible(visible)
        self.retrain_btn.setVisible(visible)

    def _set_badge(self, text, color):
        if text is None:
            self.status_badge.setVisible(False)
            return
        self.status_badge.setText(text)
        self.status_badge.setStyleSheet(
            f"padding:2px 10px; border-radius:10px; font-size:11px; "
            f"font-weight:bold; background:{color}; color:white;"
        )
        self.status_badge.setVisible(True)

    # ============================================================
    # SLOTS: confirm / reject signals from rows
    # ============================================================
    def _on_confirm(self, label: str):
        """A doctor has clicked Confirm on a finding row."""
        # Only one confirmation allowed — update previous row if needed
        if self.confirmed_label and self.confirmed_label != label:
            for row in self._rows:
                if row.label == self.confirmed_label:
                    row.name_lbl.setStyleSheet("")
                    row.confirm_btn.setEnabled(True)
                    row.confirm_btn.setText("✔ Confirm Diagnosis")
                    break

        self.confirmed_label = label
        # Visually mark the row
        for row in self._rows:
            if row.label == label:
                row._mark_confirmed()
                break

        self.save_btn.setEnabled(True)

    def _on_reject(self, label: str):
        """A doctor has clicked Reject on a finding row."""
        self.rejected_labels.add(label)
        for row in self._rows:
            if row.label == label:
                row._mark_rejected()
                break

    # ============================================================
    # SAVE DIAGNOSIS
    # ============================================================
    def _save_diagnosis(self):
        if not self.confirmed_label:
            QMessageBox.warning(
                self, "No Confirmation",
                "Please confirm at least one diagnosis before saving."
            )
            return

        notes = self.notes_box.toPlainText().strip()
        self.findings_db.save_diagnosis(
            image_id        = self.image_id,
            confirmed_label = self.confirmed_label,
            rejected_labels = list(self.rejected_labels),
            doctor_notes    = notes,
        )

        self._set_badge("✔ Diagnosed", "#2e7d32")
        self.feedback_btn.setEnabled(True)
        self.diagnosis_saved.emit(self.image_id)

        QMessageBox.information(
            self, "Saved",
            f"Diagnosis saved.\n\nConfirmed: {self.confirmed_label}\n"
            f"Rejected: {', '.join(self.rejected_labels) or 'None'}"
        )

    # ============================================================
    # SUBMIT ML FEEDBACK
    # ============================================================
    def _submit_feedback(self):
        if not self.confirmed_label:
            QMessageBox.warning(self, "No Diagnosis", "Save a diagnosis first.")
            return

        all_labels    = [f["label"] for f in self.findings]
        model_version = self.findings[0]["model_version"] if self.findings else "unknown"

        self.findings_db.submit_feedback(
            image_id        = self.image_id,
            confirmed_label = self.confirmed_label,
            all_labels      = all_labels,
            model_version   = model_version,
        )

        # Write accuracy stats to JSON immediately so the file always exists
        self.feedback_manager.write_stats_log()

        self.feedback_btn.setEnabled(False)
        self.retrain_btn.setEnabled(True)
        self._set_badge("✔ Feedback Submitted", "#4a148c")

        QMessageBox.information(
            self, "Feedback Submitted",
            "Diagnosis recorded as training feedback.\n"
            "Click 'Retrain Model' to apply it now, or submit more feedback first."
        )

    # ============================================================
    # RETRAIN MODEL
    # ============================================================
    def _run_retrain(self):
        if self.model is None:
            QMessageBox.warning(
                self, "No Model",
                "Model reference not passed to DiagnosisWindow.\n"
                "Pass model= when constructing DiagnosisWindow."
            )
            return

        self.retrain_btn.setEnabled(False)
        self.retrain_btn.setText("🧠  Retraining…")
        self.repaint()

        result = self.feedback_manager.run_training_update(self.model)

        self.retrain_btn.setText("🧠  Retrain Model")

        if result["success"]:
            saved = "New checkpoint saved." if result["checkpoint_saved"] else "No checkpoint saved (accuracy did not improve)."
            QMessageBox.information(
                self, "Retraining Complete",
                f"{result['message']}\n"
                f"Samples used: {result['samples_used']}\n"
                f"{saved}"
            )
            self._set_badge("✔ Model Updated", "#e65100")
        else:
            self.retrain_btn.setEnabled(True)
            QMessageBox.warning(self, "Retraining Failed", result["message"])