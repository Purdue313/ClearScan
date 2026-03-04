from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QListWidget, QListWidgetItem, QTextEdit, QMessageBox,
    QScrollArea, QFrame, QSizePolicy
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QColor, QCursor

from Database.findingsDatabase import FindingsDatabase
from Database.imageDatabase import ImageDatabase
from machineLearningModel.ml_feedback_manager import MLFeedbackManager


# ================================================================
# FindingRow - clickable anywhere, multi-confirm/reject toggle
# ================================================================
class FindingRow(QFrame):
    """
    Expandable row for one ML finding.
    - Click anywhere on the row header to expand/collapse.
    - Confirm and Reject are independent toggles (multiple allowed).
    - Clicking Confirm again un-confirms; same for Reject.
    """

    confirmed_changed = Signal(str, bool)   # label, is_confirmed
    rejected_changed  = Signal(str, bool)   # label, is_rejected

    RATIONALE = {
        "Pneumonia":          "Increased opacity in the lower lobes suggests bacterial or viral pneumonia.",
        "Cardiomegaly":       "Cardiac silhouette exceeds 50% of the thoracic width on PA view.",
        "Pleural Effusion":   "Blunting of the costophrenic angle is consistent with fluid accumulation.",
        "Atelectasis":        "Linear or plate-like opacities indicate partial lung collapse.",
        "Consolidation":      "Homogeneous airspace opacity with air bronchograms detected.",
        "Edema":              "Perihilar haziness and Kerley B lines suggest pulmonary oedema.",
        "Pneumothorax":       "Absence of lung markings at the periphery with visible pleural line.",
        "Fracture":           "Cortical discontinuity detected in the visualised osseous structures.",
        "Nodule":             "Focal rounded opacity < 3 cm. Follow-up CT recommended.",
        "Mass":               "Focal rounded opacity >= 3 cm. Malignancy must be excluded.",
        "Infiltrate":         "Ill-defined airspace opacity may represent infection or inflammation.",
        "Emphysema":          "Hyperinflation with flattened diaphragm and increased retrosternal space.",
        "Fibrosis":           "Reticular opacities and volume loss consistent with fibrotic change.",
        "Pleural Thickening": "Irregular soft-tissue density along the pleural surface.",
        "No Finding":         "No significant acute cardiothoracic abnormality identified.",
    }

    def __init__(self, rank, label, probability,
                 already_confirmed=False, already_rejected=False):
        super().__init__()
        self.label        = label
        self.probability  = probability
        self._expanded    = False
        self._confirmed   = False
        self._rejected    = False

        self.setFrameShape(QFrame.StyledPanel)
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self._set_base_style()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 8, 10, 8)
        outer.setSpacing(4)

        # -- Header row (entire area is click target) -------------
        self.header_widget = QWidget()
        self.header_widget.setCursor(QCursor(Qt.PointingHandCursor))
        self.header_widget.setStyleSheet("background:transparent;")
        header = QHBoxLayout(self.header_widget)
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)

        rank_lbl = QLabel(f"{rank}.")
        rank_lbl.setFixedWidth(24)
        rank_lbl.setStyleSheet("color:#888; font-size:12px; background:transparent;")

        self.name_lbl = QLabel(label)
        self.name_lbl.setFont(QFont("Arial", 11, QFont.Bold))
        self.name_lbl.setStyleSheet("background:transparent;")

        pct = probability * 100
        self.pct_lbl = QLabel(f"{pct:.1f}%")
        self.pct_lbl.setStyleSheet("color:#555; font-size:11px; min-width:46px; background:transparent;")
        self.pct_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.arrow_lbl = QLabel("  >")
        self.arrow_lbl.setStyleSheet("color:#aaa; font-size:13px; background:transparent;")
        self.arrow_lbl.setFixedWidth(20)

        header.addWidget(rank_lbl)
        header.addWidget(self.name_lbl, 1)
        header.addWidget(self.pct_lbl)
        header.addWidget(self.arrow_lbl)

        # -- Probability bar --------------------------------------
        bar_bg = QFrame()
        bar_bg.setFixedHeight(6)
        bar_bg.setStyleSheet("background:#eee; border-radius:3px; border:none;")
        bar_layout = QHBoxLayout(bar_bg)
        bar_layout.setContentsMargins(0, 0, 0, 0)
        bar_layout.setSpacing(0)

        fill_w = max(4, int(pct))
        color  = "#e53935" if pct >= 60 else "#fb8c00" if pct >= 30 else "#43a047"
        bar_fill = QFrame()
        bar_fill.setFixedHeight(6)
        bar_fill.setStyleSheet(f"background:{color}; border-radius:3px; border:none;")
        bar_layout.addWidget(bar_fill, fill_w)
        bar_layout.addStretch(100 - fill_w)

        # -- Detail panel (hidden by default) ---------------------
        self.detail_panel = QWidget()
        self.detail_panel.setVisible(False)
        self.detail_panel.setStyleSheet("background:transparent;")
        detail_layout = QVBoxLayout(self.detail_panel)
        detail_layout.setContentsMargins(24, 4, 0, 4)
        detail_layout.setSpacing(8)

        rationale_text = self.RATIONALE.get(label, "No additional rationale available.")
        rationale_lbl  = QLabel(rationale_text)
        rationale_lbl.setWordWrap(True)
        rationale_lbl.setStyleSheet("color:#444; font-size:11px; background:transparent;")

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self.confirm_btn = QPushButton("Confirm Correct")
        self.confirm_btn.setFixedHeight(30)
        self.confirm_btn.setStyleSheet(self._confirm_style(False))
        self.confirm_btn.clicked.connect(self._toggle_confirm)

        self.reject_btn = QPushButton("Mark Incorrect")
        self.reject_btn.setFixedHeight(30)
        self.reject_btn.setStyleSheet(self._reject_style(False))
        self.reject_btn.clicked.connect(self._toggle_reject)

        btn_row.addWidget(self.confirm_btn)
        btn_row.addWidget(self.reject_btn)
        btn_row.addStretch()

        detail_layout.addWidget(rationale_lbl)
        detail_layout.addLayout(btn_row)

        outer.addWidget(self.header_widget)
        outer.addWidget(bar_bg)
        outer.addWidget(self.detail_panel)

        # Make header widget act as click target for expanding
        self.header_widget.mousePressEvent = lambda e: self._toggle_expand()

        # Apply pre-existing state
        if already_confirmed:
            self._set_confirmed(True)
        elif already_rejected:
            self._set_rejected(True)

    # -- Styles ---------------------------------------------------
    def _set_base_style(self):
        self.setStyleSheet("""
            QFrame {
                border: 1px solid #ddd;
                border-radius: 6px;
                background: #fff;
                margin: 2px 0;
            }
            QFrame:hover {
                border-color: #aaa;
                background: #fafafa;
            }
        """)

    def _confirmed_frame_style(self):
        return """
            QFrame {
                border: 2px solid #2e7d32;
                border-radius: 6px;
                background: #f1f8f1;
                margin: 2px 0;
            }
        """

    def _rejected_frame_style(self):
        return """
            QFrame {
                border: 2px solid #c62828;
                border-radius: 6px;
                background: #fff5f5;
                margin: 2px 0;
            }
        """

    def _confirm_style(self, active):
        if active:
            return """QPushButton {
                background:#2e7d32; color:white; border:none;
                border-radius:4px; padding:4px 14px; font-size:12px; font-weight:bold;
            } QPushButton:hover { background:#1b5e20; }"""
        return """QPushButton {
            background:white; color:#2e7d32; border:2px solid #2e7d32;
            border-radius:4px; padding:4px 14px; font-size:12px;
        } QPushButton:hover { background:#f1f8f1; }"""

    def _reject_style(self, active):
        if active:
            return """QPushButton {
                background:#c62828; color:white; border:none;
                border-radius:4px; padding:4px 14px; font-size:12px; font-weight:bold;
            } QPushButton:hover { background:#b71c1c; }"""
        return """QPushButton {
            background:white; color:#c62828; border:2px solid #c62828;
            border-radius:4px; padding:4px 14px; font-size:12px;
        } QPushButton:hover { background:#fff5f5; }"""

    # -- Toggle expand --------------------------------------------
    def _toggle_expand(self):
        self._expanded = not self._expanded
        self.detail_panel.setVisible(self._expanded)
        self.arrow_lbl.setText("  v" if self._expanded else "  >")

    # -- Toggle confirm -------------------------------------------
    def _toggle_confirm(self):
        self._set_confirmed(not self._confirmed)
        self.confirmed_changed.emit(self.label, self._confirmed)

    def _set_confirmed(self, state: bool):
        self._confirmed = state
        self.confirm_btn.setStyleSheet(self._confirm_style(state))
        self.confirm_btn.setText("Confirmed (click to undo)" if state else "Confirm Correct")
        if state:
            self.name_lbl.setStyleSheet("color:#2e7d32; font-weight:bold; background:transparent;")
            self.setStyleSheet(self._confirmed_frame_style())
            # Clear reject if setting confirm
            if self._rejected:
                self._set_rejected(False)
                self.rejected_changed.emit(self.label, False)
        else:
            self.name_lbl.setStyleSheet("background:transparent;")
            self._set_base_style()

    # -- Toggle reject --------------------------------------------
    def _toggle_reject(self):
        self._set_rejected(not self._rejected)
        self.rejected_changed.emit(self.label, self._rejected)

    def _set_rejected(self, state: bool):
        self._rejected = state
        self.reject_btn.setStyleSheet(self._reject_style(state))
        self.reject_btn.setText("Marked Incorrect (click to undo)" if state else "Mark Incorrect")
        if state:
            self.name_lbl.setStyleSheet(
                "color:#c62828; text-decoration:line-through; background:transparent;"
            )
            self.setStyleSheet(self._rejected_frame_style())
            # Clear confirm if setting reject
            if self._confirmed:
                self._set_confirmed(False)
                self.confirmed_changed.emit(self.label, False)
        else:
            self.name_lbl.setStyleSheet("background:transparent;")
            self._set_base_style()


# ================================================================
# DiagnosisWindow
# ================================================================
class DiagnosisWindow(QWidget):
    """
    Doctor-facing panel for FR 3.6.1.
    Supports multiple confirmed and multiple rejected labels.
    """

    diagnosis_saved = Signal(int)

    def __init__(self, findings_db: FindingsDatabase,
                 image_db: ImageDatabase, model=None, parent=None):
        super().__init__(parent)
        self.findings_db      = findings_db
        self.image_db         = image_db
        self.model            = model
        self.feedback_manager = MLFeedbackManager(findings_db, image_db)
        self.image_id         = None
        self.findings         = []
        self.confirmed_labels = set()   # now a SET - multiple allowed
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

        # Title bar
        title_row = QHBoxLayout()
        title = QLabel("Diagnosis Review")
        title.setFont(QFont("Arial", 14, QFont.Bold))

        self.status_badge = QLabel()
        self.status_badge.setFixedHeight(24)
        self.status_badge.setVisible(False)

        title_row.addWidget(title)
        title_row.addStretch()
        title_row.addWidget(self.status_badge)
        root.addLayout(title_row)

        # Hint label
        self.hint_lbl = QLabel(
            "Click any finding to expand.  Confirm or mark incorrect - multiple selections allowed."
        )
        self.hint_lbl.setStyleSheet("color:#888; font-size:11px;")
        self.hint_lbl.setVisible(False)
        root.addWidget(self.hint_lbl)

        # Placeholder
        self.placeholder = QLabel("Select an image to begin diagnosis review.")
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setStyleSheet("color:#999; font-size:13px;")
        root.addWidget(self.placeholder)

        # Scroll area
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

        # Doctor notes
        self.notes_label = QLabel("Doctor Notes:")
        self.notes_label.setStyleSheet("font-weight:bold; font-size:12px;")
        self.notes_label.setVisible(False)

        self.notes_box = QTextEdit()
        self.notes_box.setPlaceholderText(
            "Add any clinical observations, caveats, or follow-up instructions..."
        )
        self.notes_box.setFixedHeight(80)
        self.notes_box.setVisible(False)

        root.addWidget(self.notes_label)
        root.addWidget(self.notes_box)

        # Action buttons
        btn_row = QHBoxLayout()

        self.save_btn = QPushButton("Save Diagnosis")
        self.save_btn.setFixedHeight(38)
        self.save_btn.setStyleSheet("""
            QPushButton {
                background:#1565c0; color:white; border:none;
                border-radius:5px; font-size:13px; font-weight:bold;
            }
            QPushButton:hover    { background:#0d47a1; }
            QPushButton:disabled { background:#aaa; }
        """)
        self.save_btn.setEnabled(False)
        self.save_btn.setVisible(False)
        self.save_btn.clicked.connect(self._save_diagnosis)

        self.feedback_btn = QPushButton("Submit ML Feedback")
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

        self.retrain_btn = QPushButton("Retrain Model")
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
    # PUBLIC: load image
    # ============================================================
    def load_image(self, image_id: int):
        self.image_id         = image_id
        self.confirmed_labels = set()
        self.rejected_labels  = set()

        rows = self.findings_db.fetch_findings_for_image(image_id)
        self.findings = [
            {"label": r[0], "probability": r[1], "model_version": r[2]}
            for r in rows
        ]

        existing = self.findings_db.fetch_diagnosis(image_id)
        self._rebuild_rows(existing)
        self._show_content(True)

        if existing:
            # Support both old single-label and new multi-label format
            cl = existing["confirmed_label"]
            if cl:
                if isinstance(cl, list):
                    self.confirmed_labels = set(cl)
                else:
                    self.confirmed_labels = {cl}
            self.rejected_labels = set(existing["rejected_labels"])
            self.notes_box.setPlainText(existing["doctor_notes"] or "")
            self._set_badge("Diagnosed", "#2e7d32")
            self.save_btn.setEnabled(True)
            if not existing["feedback_submitted"]:
                self.feedback_btn.setEnabled(True)
        else:
            self.notes_box.clear()
            self._set_badge(None, None)
            self.save_btn.setEnabled(False)
            self.feedback_btn.setEnabled(False)

    # ============================================================
    # REBUILD ROWS
    # ============================================================
    def _rebuild_rows(self, existing_diagnosis):
        for row in self._rows:
            row.setParent(None)
        self._rows.clear()

        confirmed_set = set()
        rejected_set  = set()
        if existing_diagnosis:
            cl = existing_diagnosis["confirmed_label"]
            if cl:
                confirmed_set = {cl} if isinstance(cl, str) else set(cl)
            rejected_set = set(existing_diagnosis["rejected_labels"])

        for i, finding in enumerate(self.findings):
            label = finding["label"]
            prob  = finding["probability"]

            row = FindingRow(
                rank             = i + 1,
                label            = label,
                probability      = prob,
                already_confirmed = (label in confirmed_set),
                already_rejected  = (label in rejected_set),
            )
            row.confirmed_changed.connect(self._on_confirmed_changed)
            row.rejected_changed.connect(self._on_rejected_changed)

            self.findings_layout.insertWidget(
                self.findings_layout.count() - 1, row
            )
            self._rows.append(row)

    def _show_content(self, visible: bool):
        self.placeholder.setVisible(not visible)
        self.hint_lbl.setVisible(visible)
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
    # SLOTS
    # ============================================================
    def _on_confirmed_changed(self, label: str, is_confirmed: bool):
        if is_confirmed:
            self.confirmed_labels.add(label)
            self.rejected_labels.discard(label)
        else:
            self.confirmed_labels.discard(label)
        self.save_btn.setEnabled(
            len(self.confirmed_labels) > 0 or len(self.rejected_labels) > 0
        )

    def _on_rejected_changed(self, label: str, is_rejected: bool):
        if is_rejected:
            self.rejected_labels.add(label)
            self.confirmed_labels.discard(label)
        else:
            self.rejected_labels.discard(label)
        self.save_btn.setEnabled(
            len(self.confirmed_labels) > 0 or len(self.rejected_labels) > 0
        )

    # ============================================================
    # SAVE DIAGNOSIS
    # ============================================================
    def _save_diagnosis(self):
        if not self.confirmed_labels and not self.rejected_labels:
            QMessageBox.warning(
                self, "Nothing Selected",
                "Please confirm or mark incorrect at least one finding before saving."
            )
            return

        notes = self.notes_box.toPlainText().strip()

        # Store confirmed as a list (first item is primary)
        confirmed_list = sorted(self.confirmed_labels)
        primary        = confirmed_list[0] if confirmed_list else ""

        self.findings_db.save_diagnosis(
            image_id        = self.image_id,
            confirmed_label = primary,            # DB primary field
            rejected_labels = list(self.rejected_labels),
            doctor_notes    = notes,
        )

        # Also persist the full confirmed set in notes if multiple
        if len(confirmed_list) > 1:
            extra = "Confirmed findings: " + ", ".join(confirmed_list)
            full_notes = (notes + "\n" + extra).strip() if notes else extra
            self.findings_db.save_diagnosis(
                image_id        = self.image_id,
                confirmed_label = primary,
                rejected_labels = list(self.rejected_labels),
                doctor_notes    = full_notes,
            )
            self.notes_box.setPlainText(full_notes)

        confirmed_str = ", ".join(confirmed_list) if confirmed_list else "None"
        rejected_str  = ", ".join(sorted(self.rejected_labels)) if self.rejected_labels else "None"

        self._set_badge("Diagnosed", "#2e7d32")
        self.feedback_btn.setEnabled(True)
        self.diagnosis_saved.emit(self.image_id)

        QMessageBox.information(
            self, "Saved",
            f"Diagnosis saved.\n\nConfirmed: {confirmed_str}\nIncorrect: {rejected_str}"
        )

    # ============================================================
    # SUBMIT FEEDBACK
    # ============================================================
    def _submit_feedback(self):
        if not self.confirmed_labels:
            QMessageBox.warning(self, "No Confirmed Diagnosis",
                                "Confirm at least one finding before submitting feedback.")
            return

        all_labels    = [f["label"] for f in self.findings]
        model_version = self.findings[0]["model_version"] if self.findings else "unknown"
        primary       = sorted(self.confirmed_labels)[0]

        self.findings_db.submit_feedback(
            image_id        = self.image_id,
            confirmed_label = primary,
            all_labels      = all_labels,
            model_version   = model_version,
        )
        self.feedback_manager.write_stats_log()

        self.feedback_btn.setEnabled(False)
        self.retrain_btn.setEnabled(True)
        self._set_badge("Feedback Submitted", "#4a148c")

        QMessageBox.information(
            self, "Feedback Submitted",
            "Diagnosis recorded as training feedback.\n"
            "Click 'Retrain Model' to apply it now."
        )

    # ============================================================
    # RETRAIN
    # ============================================================
    def _run_retrain(self):
        if self.model is None:
            QMessageBox.warning(self, "No Model",
                "Model reference not available. Please try again after the model finishes loading.")
            return

        self.retrain_btn.setEnabled(False)
        self.retrain_btn.setText("Retraining...")
        self.repaint()

        result = self.feedback_manager.run_training_update(self.model)
        self.retrain_btn.setText("Retrain Model")

        if result["success"]:
            saved = "New checkpoint saved." if result["checkpoint_saved"] else \
                    "No checkpoint saved (accuracy did not improve)."
            QMessageBox.information(
                self, "Retraining Complete",
                f"{result['message']}\nSamples used: {result['samples_used']}\n{saved}"
            )
            self._set_badge("Model Updated", "#e65100")
        else:
            self.retrain_btn.setEnabled(True)
            QMessageBox.warning(self, "Retraining Failed", result["message"])