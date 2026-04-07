from PySide6.QtCore import Qt
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QLabel


class ElidedLabel(QLabel):
    """QLabel that elides text with '...' when it doesn't fit."""

    def __init__(self, text='', parent=None):
        super().__init__(text, parent)
        self.setSizePolicy(self.sizePolicy().horizontalPolicy(),
                           self.sizePolicy().verticalPolicy())
        self._full_text = text

    def setText(self, text: str):
        self._full_text = text
        self._update_elided()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_elided()

    def _update_elided(self):
        fm = QFontMetrics(self.font())
        elided = fm.elidedText(self._full_text, Qt.TextElideMode.ElideMiddle,
                               self.width())
        super().setText(elided)
        self.setToolTip(self._full_text if elided != self._full_text else '')

    def minimumSizeHint(self):
        hint = super().minimumSizeHint()
        hint.setWidth(0)
        return hint
