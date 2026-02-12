from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QComboBox, QLabel, QStackedWidget, QVBoxLayout,
                               QWidget)

from ltd.settings import get_settings


class ModuleSelector(QWidget):
    """Dropdown that switches between module settings panels."""
    module_changed = Signal(int)

    def __init__(self, label_text: str, modules: list,
                 settings_key: str = '', parent=None):
        super().__init__(parent)
        self.modules = modules
        self._settings_key = settings_key
        self._setup_ui(label_text)

    def _setup_ui(self, label_text: str):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel(label_text))

        self.combo = QComboBox()
        for module in self.modules:
            self.combo.addItem(module.name)
        layout.addWidget(self.combo)

        self.settings_stack = QStackedWidget()
        for module in self.modules:
            widget = module.get_settings_widget()
            self.settings_stack.addWidget(widget)
        layout.addWidget(self.settings_stack)

        self.combo.currentIndexChanged.connect(self.settings_stack.setCurrentIndex)
        self.combo.currentIndexChanged.connect(self.module_changed.emit)
        self.combo.currentIndexChanged.connect(self._save_selection)

        # Restore saved selection
        self._restore_selection()

    def current_module(self):
        idx = self.combo.currentIndex()
        if 0 <= idx < len(self.modules):
            return self.modules[idx]
        return None

    def current_index(self) -> int:
        return self.combo.currentIndex()

    def _save_selection(self):
        if self._settings_key:
            get_settings().setValue(
                f'module_selector/{self._settings_key}/selected',
                self.combo.currentText())

    def _restore_selection(self):
        if not self._settings_key:
            return
        saved = get_settings().value(
            f'module_selector/{self._settings_key}/selected', '', type=str)
        if saved:
            idx = self.combo.findText(saved)
            if idx >= 0:
                self.combo.setCurrentIndex(idx)
