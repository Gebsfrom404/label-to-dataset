"""Modal loading dialog shown while blocking work runs.

Used by every tab that does mass loading. Supports both indeterminate
mode (default) and determinate mode (call ``set_progress`` to switch).
"""
from contextlib import contextmanager

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QApplication, QDialog, QLabel, QProgressBar,
                               QVBoxLayout)


class LoadingDialog(QDialog):
    """Frameless modal dialog with a message and progress bar."""

    def __init__(self, message: str = 'Loading...', parent=None):
        super().__init__(parent)
        self.setWindowTitle('Loading')
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.CustomizeWindowHint
            | Qt.WindowType.WindowTitleHint)
        self.setModal(True)
        self.setFixedSize(320, 90)

        layout = QVBoxLayout(self)
        self._label = QLabel(message)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._label)

        self._bar = QProgressBar()
        self._bar.setRange(0, 0)  # indeterminate by default
        self._bar.setTextVisible(False)
        layout.addWidget(self._bar)

    def set_message(self, text: str):
        self._label.setText(text)

    def set_progress(self, current: int, total: int):
        """Switch to determinate mode and update the bar."""
        if total <= 0:
            self._bar.setRange(0, 0)
            self._bar.setTextVisible(False)
            return
        if self._bar.maximum() != total:
            self._bar.setRange(0, total)
            self._bar.setTextVisible(True)
            self._bar.setFormat('%v / %m')
        self._bar.setValue(current)

    def set_indeterminate(self):
        self._bar.setRange(0, 0)
        self._bar.setTextVisible(False)


@contextmanager
def loading_dialog(message: str = 'Loading...', parent=None):
    """Context manager that shows a loading dialog and processes events.

    Usage::

        with loading_dialog('Loading folder...', self) as dlg:
            self.model.load_directory(directory)
            dlg.set_message('Reading metadata...')
            dlg.set_progress(50, 100)
    """
    dlg = LoadingDialog(message, parent)
    dlg.show()
    QApplication.processEvents()
    try:
        yield dlg
    finally:
        dlg.close()
