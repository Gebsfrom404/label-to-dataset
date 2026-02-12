"""Reusable ComfyUI workflow selector: dropdown of saved JSON files + Custom."""
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QPlainTextEdit,
                               QPushButton, QVBoxLayout, QWidget)

from ltd.settings import get_settings

WORKFLOWS_DIR = Path('./Comfy-workflows')


class WorkflowSelector(QWidget):
    """Dropdown listing JSON files from Comfy-workflows/ plus a Custom option.

    When a file is selected the text area is hidden and the file is read on
    demand via ``get_workflow_text()``.  When *Custom* is chosen the text area
    is shown so the user can paste arbitrary JSON.

    Args:
        settings_key: unique key for persisting the selected workflow across
                      app restarts (e.g. 'detection', 'modification', 'caption').
    """

    CUSTOM = '-- Custom --'

    workflow_changed = Signal()

    def __init__(self, settings_key: str = '', parent=None):
        super().__init__(parent)
        self._settings_key = settings_key

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Top row: combo + refresh
        top = QHBoxLayout()
        self.combo = QComboBox()
        self.combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents)
        top.addWidget(self.combo, stretch=1)

        self.refresh_btn = QPushButton('↻')
        self.refresh_btn.setFixedWidth(30)
        self.refresh_btn.setToolTip('Refresh workflow list')
        top.addWidget(self.refresh_btn)
        layout.addLayout(top)

        # Text area (visible only for Custom)
        self.text_edit = QPlainTextEdit()
        self.text_edit.setPlaceholderText(
            'Paste ComfyUI API workflow JSON here...')
        self.text_edit.setMinimumHeight(100)
        self.text_edit.setVisible(False)
        layout.addWidget(self.text_edit)

        self._populate()

        self.combo.currentIndexChanged.connect(self._on_combo_changed)
        self.refresh_btn.clicked.connect(self._populate)

    # ------------------------------------------------------------------

    def _settings_prefix(self) -> str:
        if self._settings_key:
            return f'workflow_selector/{self._settings_key}'
        return 'workflow_selector'

    def _populate(self):
        """Scan Comfy-workflows/ and rebuild the combo box."""
        prev = self.combo.currentText()
        self.combo.blockSignals(True)
        self.combo.clear()

        WORKFLOWS_DIR.mkdir(parents=True, exist_ok=True)
        json_files = sorted(WORKFLOWS_DIR.glob('*.json'),
                            key=lambda p: p.name.lower())

        for f in json_files:
            self.combo.addItem(f.stem, str(f))  # display stem, data = path

        self.combo.addItem(self.CUSTOM)

        # Restore: try prev text first (within-session), then saved setting
        idx = self.combo.findText(prev)
        if idx < 0 and self._settings_key:
            saved = get_settings().value(
                f'{self._settings_prefix()}/selected', '', type=str)
            if saved:
                idx = self.combo.findText(saved)
        if idx >= 0:
            self.combo.setCurrentIndex(idx)
        elif self.combo.count() > 1:
            # Default to first workflow file if available
            self.combo.setCurrentIndex(0)
        else:
            self.combo.setCurrentIndex(self.combo.count() - 1)

        self.combo.blockSignals(False)
        self._on_combo_changed()

    def _on_combo_changed(self):
        is_custom = self.combo.currentText() == self.CUSTOM
        self.text_edit.setVisible(is_custom)
        # Persist selection
        if self._settings_key:
            get_settings().setValue(
                f'{self._settings_prefix()}/selected',
                self.combo.currentText())
        self.workflow_changed.emit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_workflow_text(self) -> str:
        """Return workflow JSON string (from file or custom text area)."""
        if self.combo.currentText() == self.CUSTOM:
            return self.text_edit.toPlainText().strip()

        path_str = self.combo.currentData()
        if path_str:
            path = Path(path_str)
            if path.exists():
                return path.read_text(encoding='utf-8').strip()
        return ''

    def has_workflow(self) -> bool:
        return bool(self.get_workflow_text())
