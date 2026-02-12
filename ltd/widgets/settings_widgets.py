from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit,
                               QPlainTextEdit, QSpinBox)

from ltd.settings import get_settings


class SettingsCheckBox(QCheckBox):
    def __init__(self, key: str, default: bool, text: str | None = None):
        super().__init__(text)
        self.settings = get_settings()
        self.setChecked(self.settings.value(key, default, type=bool))
        self.stateChanged.connect(
            lambda state: self.settings.setValue(
                key, state == Qt.CheckState.Checked.value))


class SettingsComboBox(QComboBox):
    def __init__(self, key: str, default: str | None = None):
        super().__init__()
        self.key = key
        self.default = default
        self.settings = get_settings()

    def addItems(self, texts: list[str]):
        setting = self.settings.value(self.key, self.default, type=str)
        super().addItems(texts)
        self.currentTextChanged.connect(
            lambda text: self.settings.setValue(self.key, text))
        if setting and setting in texts:
            self.setCurrentText(setting)


class SettingsSpinBox(QSpinBox):
    def __init__(self, key: str, default: int, minimum: int, maximum: int):
        super().__init__()
        self.setRange(minimum, maximum)
        self.settings = get_settings()
        self.setValue(self.settings.value(key, default, type=int))
        self.valueChanged.connect(lambda value: self.settings.setValue(key, value))


class SettingsDoubleSpinBox(QDoubleSpinBox):
    def __init__(self, key: str, default: float, minimum: float,
                 maximum: float):
        super().__init__()
        self.setRange(minimum, maximum)
        self.settings = get_settings()
        self.setValue(self.settings.value(key, default, type=float))
        self.valueChanged.connect(lambda value: self.settings.setValue(key, value))


class SettingsLineEdit(QLineEdit):
    def __init__(self, key: str, default: str = ''):
        super().__init__()
        self.settings = get_settings()
        self.setText(self.settings.value(key, default, type=str))
        self.textChanged.connect(lambda text: self.settings.setValue(key, text))


class SettingsPlainTextEdit(QPlainTextEdit):
    def __init__(self, key: str, default: str = ''):
        super().__init__()
        self.settings = get_settings()
        self.setPlainText(self.settings.value(key, default, type=str))
        self.textChanged.connect(
            lambda: self.settings.setValue(key, self.toPlainText()))
