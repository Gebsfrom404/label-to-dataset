"""Small "ⓘ" button that shows a rich-text tooltip on hover or click.

Two flavours:
  - ``InfoButton(content_html)``  — static content
  - ``DynamicInfoButton(provider)`` — calls a ``() -> str`` provider each time
    so the tooltip can change based on focus / state
"""
from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QToolButton, QToolTip, QWidget


def focus_in(widget: QWidget | None) -> bool:
    """True when ``widget`` exists and contains the current focus widget."""
    if widget is None:
        return False
    from PySide6.QtWidgets import QApplication
    focus = QApplication.focusWidget()
    while focus is not None:
        if focus is widget:
            return True
        focus = focus.parentWidget()
    return False


_INFO_STYLE = """
QToolButton {
    border: 1px solid palette(mid);
    border-radius: 9px;
    padding: 0 0 1px 0;
    background: transparent;
    min-width: 18px;  max-width: 18px;
    min-height: 18px; max-height: 18px;
}
QToolButton:hover  { background: palette(midlight); }
QToolButton:pressed{ background: palette(mid); }
"""


class InfoButton(QToolButton):
    """Round "i" button. Hover or click → rich-text tooltip."""

    def __init__(self, content: str = '', parent=None):
        super().__init__(parent)
        self.setText('i')
        self.setStyleSheet(_INFO_STYLE)
        self.setCursor(Qt.CursorShape.WhatsThisCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        font = self.font()
        font.setBold(True)
        font.setItalic(True)
        self.setFont(font)
        self._content = content

    def set_content(self, content: str):
        self._content = content

    def event(self, e):
        if e.type() == QEvent.Type.ToolTip:
            QToolTip.showText(QCursor.pos(), self._content, self)
            return True
        return super().event(e)

    def mousePressEvent(self, event):
        QToolTip.showText(self.mapToGlobal(self.rect().bottomLeft()),
                          self._content, self)
        super().mousePressEvent(event)


class DynamicInfoButton(InfoButton):
    """Info button whose content is recomputed on each show."""

    def __init__(self, content_provider: Callable[[], str], parent=None):
        super().__init__('', parent)
        self._provider = content_provider

    def event(self, e):
        if e.type() == QEvent.Type.ToolTip:
            self._content = self._provider()
        return super().event(e)

    def mousePressEvent(self, event):
        self._content = self._provider()
        super().mousePressEvent(event)
