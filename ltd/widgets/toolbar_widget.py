from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QToolBar, QWidget)

from ltd.settings import DEFAULT_SETTINGS, get_settings
from ltd.widgets.settings_widgets import (SettingsComboBox, SettingsLineEdit,
                                           SettingsSpinBox)


class ToolbarWidget(QToolBar):
    """Top toolbar with ComfyUI URL, theme toggle, and font size."""
    theme_changed = Signal(str)
    font_size_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__('Toolbar', parent)
        self.setMovable(False)
        self.settings = get_settings()
        self._setup_ui()

    def _setup_ui(self):
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(4, 2, 4, 2)

        # ComfyUI URL
        layout.addWidget(QLabel('ComfyUI:'))
        self.comfyui_url = SettingsLineEdit(
            'comfyui_url', DEFAULT_SETTINGS['comfyui_url'])
        self.comfyui_url.setMinimumWidth(200)
        self.comfyui_url.setPlaceholderText('http://127.0.0.1:8188')
        layout.addWidget(self.comfyui_url)

        layout.addStretch()

        # Theme
        layout.addWidget(QLabel('Theme:'))
        self.theme_combo = SettingsComboBox('theme', DEFAULT_SETTINGS['theme'])
        self.theme_combo.addItems(['dark', 'light'])
        self.theme_combo.currentTextChanged.connect(self.theme_changed.emit)
        layout.addWidget(self.theme_combo)

        # Font size
        layout.addWidget(QLabel('Font:'))
        self.font_size_spin = SettingsSpinBox(
            'font_size', DEFAULT_SETTINGS['font_size'], 8, 24)
        self.font_size_spin.setSuffix('pt')
        self.font_size_spin.valueChanged.connect(self.font_size_changed.emit)
        layout.addWidget(self.font_size_spin)

        self.addWidget(container)
