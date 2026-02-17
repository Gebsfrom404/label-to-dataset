"""Simple modal loading dialog shown while blocking work runs."""
from contextlib import contextmanager

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QApplication, QDialog, QLabel, QProgressBar,
                               QVBoxLayout)


class LoadingDialog(QDialog):
    """Frameless modal dialog with a message and indeterminate progress bar."""

    def __init__(self, message: str = 'Loading...', parent=None):
        super().__init__(parent)
        self.setWindowTitle('Loading')
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.CustomizeWindowHint
            | Qt.WindowType.WindowTitleHint)
        self.setModal(True)
        self.setFixedSize(280, 80)

        layout = QVBoxLayout(self)
        self._label = QLabel(message)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._label)

        bar = QProgressBar()
        bar.setRange(0, 0)  # indeterminate
        layout.addWidget(bar)

    def set_message(self, text: str):
        self._label.setText(text)


@contextmanager
def loading_dialog(message: str = 'Loading...', parent=None):
    """Context manager that shows a loading dialog and processes events.

    Usage::

        with loading_dialog('Loading folder...', self):
            self.model.load_directory(directory)
    """
    dlg = LoadingDialog(message, parent)
    dlg.show()
    QApplication.processEvents()
    try:
        yield dlg
    finally:
        dlg.close()
