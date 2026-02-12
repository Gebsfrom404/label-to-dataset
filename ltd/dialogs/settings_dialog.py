"""Application settings dialog."""
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QFormLayout,
                               QVBoxLayout)

from ltd.settings import DEFAULT_SETTINGS
from ltd.widgets.settings_widgets import (SettingsComboBox, SettingsLineEdit,
                                           SettingsSpinBox)


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Settings')
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        # ComfyUI URL
        self.comfyui_url = SettingsLineEdit(
            'comfyui_url', DEFAULT_SETTINGS['comfyui_url'])
        form.addRow('ComfyUI URL:', self.comfyui_url)

        # Theme
        self.theme = SettingsComboBox('theme', DEFAULT_SETTINGS['theme'])
        self.theme.addItems(['dark', 'light'])
        form.addRow('Theme:', self.theme)

        # Font size
        self.font_size = SettingsSpinBox(
            'font_size', DEFAULT_SETTINGS['font_size'], 8, 24)
        form.addRow('Font size:', self.font_size)

        # Image list width
        self.image_width = SettingsSpinBox(
            'image_list_image_width',
            DEFAULT_SETTINGS['image_list_image_width'], 80, 400)
        form.addRow('Thumbnail width:', self.image_width)

        layout.addLayout(form)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
