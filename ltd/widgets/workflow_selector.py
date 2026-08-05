"""Reusable ComfyUI workflow selector: dropdown of saved JSON files + Custom."""
import json
import re
from pathlib import Path

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QPlainTextEdit,
                               QPushButton, QVBoxLayout, QWidget)

from ltd.comfyui.workflow import (get_input_prompt, is_api_format,
                                  set_input_prompt)
from ltd.settings import get_settings

WORKFLOWS_DIR = Path('./Comfy-workflows')


class WorkflowSelector(QWidget):
    """Dropdown listing JSON files from Comfy-workflows/ plus a Custom option.

    When a file is selected the text area is hidden and the file is read on
    demand via ``get_workflow_text()``.  When *Custom* is chosen the text area
    is shown so the user can paste arbitrary JSON.

    If the selected workflow contains an ``LTD_Input_Prompt`` node, a prompt
    box appears pre-filled with the prompt stored in the workflow.  Edits are
    injected into the JSON returned by ``get_workflow_text()`` at run time —
    the workflow file on disk is never modified.

    Args:
        settings_key: unique key for persisting the selected workflow across
                      app restarts (e.g. 'detection', 'modification', 'caption').
    """

    CUSTOM = '-- Custom --'

    workflow_changed = Signal()

    def __init__(self, settings_key: str = '', parent=None):
        super().__init__(parent)
        self._settings_key = settings_key
        # Prompt stored in the currently selected workflow (None = no node)
        self._prompt_default = None
        # Workflow the prompt box was last filled from, so typing in the
        # Custom JSON area doesn't wipe the prompt on every keystroke.
        self._prompt_source = None

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

        # Prompt area (visible only when the workflow has LTD_Input_Prompt)
        prompt_header = QHBoxLayout()
        self.prompt_label = QLabel('Prompt (LTD_Input_Prompt):')
        prompt_header.addWidget(self.prompt_label, stretch=1)
        self.prompt_reset_btn = QPushButton('↺')
        self.prompt_reset_btn.setFixedWidth(30)
        self.prompt_reset_btn.setToolTip(
            'Reset to the prompt saved in the workflow')
        prompt_header.addWidget(self.prompt_reset_btn)
        layout.addLayout(prompt_header)

        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setPlaceholderText(
            'Instruction sent to the workflow (not saved into the workflow '
            'file)...')
        self.prompt_edit.setMinimumHeight(60)
        self.prompt_edit.setMaximumHeight(120)
        layout.addWidget(self.prompt_edit)
        self._set_prompt_visible(False)

        self._populate()

        self.combo.currentIndexChanged.connect(self._on_combo_changed)
        self.refresh_btn.clicked.connect(self._populate)
        self.prompt_reset_btn.clicked.connect(self._reset_prompt)
        self.prompt_edit.textChanged.connect(self._save_prompt)

        # Custom JSON is re-scanned for a prompt node after typing settles
        self._custom_timer = QTimer(self)
        self._custom_timer.setSingleShot(True)
        self._custom_timer.setInterval(500)
        self._custom_timer.timeout.connect(self._refresh_prompt_field)
        self.text_edit.textChanged.connect(self._custom_timer.start)

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
        self._refresh_prompt_field()
        self.workflow_changed.emit()

    # ------------------------------------------------------------------
    # Prompt handling
    # ------------------------------------------------------------------

    def _set_prompt_visible(self, visible: bool):
        self.prompt_label.setVisible(visible)
        self.prompt_reset_btn.setVisible(visible)
        self.prompt_edit.setVisible(visible)

    def _raw_workflow_text(self) -> str:
        """Workflow JSON as stored (file contents or Custom text area)."""
        if self.combo.currentText() == self.CUSTOM:
            return self.text_edit.toPlainText().strip()

        path_str = self.combo.currentData()
        if path_str:
            path = Path(path_str)
            if path.exists():
                try:
                    return path.read_text(encoding='utf-8').strip()
                except OSError:
                    return ''
        return ''

    @staticmethod
    def _parse(text: str) -> dict | None:
        """Parse API-format workflow JSON, or None if it isn't one.

        Deliberately avoids ``load_workflow`` — this runs on every selection
        change and must not hit the network for UI→API conversion.
        """
        if not text:
            return None
        try:
            workflow = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(workflow, dict) or not is_api_format(workflow):
            return None
        return workflow

    def _prompt_settings_key(self) -> str:
        name = re.sub(r'[^A-Za-z0-9_-]', '_', self.combo.currentText())
        return f'{self._settings_prefix()}/prompt/{name}'

    def _refresh_prompt_field(self):
        """Show/hide the prompt box for the current workflow and fill it."""
        workflow = self._parse(self._raw_workflow_text())
        default = get_input_prompt(workflow) if workflow else None

        self._prompt_default = default
        self._set_prompt_visible(default is not None)
        if default is None:
            self._prompt_source = None
            return

        source = self.combo.currentText()
        if source == self._prompt_source:
            return  # same workflow — keep whatever the user typed
        self._prompt_source = source

        override = ''
        if self._settings_key:
            override = get_settings().value(
                self._prompt_settings_key(), '', type=str)

        self.prompt_edit.blockSignals(True)
        self.prompt_edit.setPlainText(override or default)
        self.prompt_edit.blockSignals(False)

    def _save_prompt(self):
        if self._settings_key and self._prompt_default is not None:
            get_settings().setValue(self._prompt_settings_key(),
                                    self.prompt_edit.toPlainText())

    def _reset_prompt(self):
        """Restore the prompt stored in the workflow, dropping the override."""
        if self._prompt_default is None:
            return
        if self._settings_key:
            get_settings().remove(self._prompt_settings_key())
        self.prompt_edit.blockSignals(True)
        self.prompt_edit.setPlainText(self._prompt_default)
        self.prompt_edit.blockSignals(False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_workflow_text(self) -> str:
        """Return workflow JSON string with the user's prompt injected.

        The prompt is written into a copy of the JSON only — the workflow
        file on disk is left untouched.
        """
        text = self._raw_workflow_text()
        if self._prompt_default is None:
            return text

        prompt = self.prompt_edit.toPlainText()
        if prompt == self._prompt_default:
            return text

        workflow = self._parse(text)
        if workflow is None:
            return text
        set_input_prompt(workflow, prompt)
        return json.dumps(workflow)

    def get_prompt(self) -> str | None:
        """Current prompt text, or None when the workflow has no prompt node."""
        if self._prompt_default is None:
            return None
        return self.prompt_edit.toPlainText()

    def has_workflow(self) -> bool:
        return bool(self._raw_workflow_text())
