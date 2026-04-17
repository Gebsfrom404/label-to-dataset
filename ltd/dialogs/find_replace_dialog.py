"""Find and replace dialog for tags (adapted from taggui)."""
import re

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog,
                               QDialogButtonBox, QFormLayout, QLabel,
                               QLineEdit, QVBoxLayout)


class FindReplaceDialog(QDialog):
    def __init__(self, image_count: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Find and Replace')
        self.setMinimumWidth(450)
        self._image_count = image_count

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.find_edit = QLineEdit()
        self.find_edit.setClearButtonEnabled(True)
        form.addRow('Find:', self.find_edit)

        self.replace_edit = QLineEdit()
        self.replace_edit.setClearButtonEnabled(True)
        form.addRow('Replace with:', self.replace_edit)

        self.scope_combo = QComboBox()
        self.scope_combo.addItems(['All images', 'Selected images'])
        self.scope_combo.setCurrentIndex(1)
        form.addRow('Scope:', self.scope_combo)

        self.match_whole = QCheckBox('Whole tags only')
        self.match_whole.setChecked(True)
        form.addRow(self.match_whole)

        self.use_regex = QCheckBox('Use regex')
        form.addRow(self.use_regex)

        layout.addLayout(form)

        self.result_label = QLabel('')
        layout.addWidget(self.result_label)

        buttons = QDialogButtonBox()
        self.replace_btn = buttons.addButton(
            'Replace All', QDialogButtonBox.ButtonRole.ActionRole)
        self.replace_btn.setEnabled(False)
        buttons.addButton(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.find_edit.textChanged.connect(self._update_button)
        self.use_regex.toggled.connect(self._update_button)

    def _update_button(self):
        text = self.find_edit.text()
        if not text:
            self.replace_btn.setEnabled(False)
            self.replace_btn.setText('Replace All')
            self.result_label.clear()
            return
        if self.use_regex.isChecked():
            try:
                re.compile(text)
            except re.error as e:
                self.replace_btn.setEnabled(False)
                self.result_label.setText(f'Invalid regex: {e}')
                return
        self.replace_btn.setEnabled(True)
        self.result_label.clear()

    @property
    def find_text(self) -> str:
        return self.find_edit.text()

    @property
    def replace_text(self) -> str:
        return self.replace_edit.text()

    @property
    def whole_match(self) -> bool:
        return self.match_whole.isChecked()

    @property
    def is_regex(self) -> bool:
        return self.use_regex.isChecked()

    @property
    def scope_all(self) -> bool:
        return self.scope_combo.currentIndex() == 0
