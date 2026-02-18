from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QPushButton, QVBoxLayout, QWidget


class CollapsiblePanel(QFrame):
    """Collapsible panel with a clickable header and togglable content area."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._title = title
        self._expanded = False
        self._available = True

        self.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header button
        self._header = QPushButton(f'\u25b6  {title}')
        self._header.setCheckable(False)
        self._header.setCursor(Qt.CursorShape.PointingHandCursor)
        self._header.setStyleSheet(
            'QPushButton { text-align: left; padding: 8px 12px; '
            'font-weight: bold; border: none; border-radius: 0; }'
        )
        self._header.clicked.connect(self._toggle)
        layout.addWidget(self._header)

        # Content area
        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(12, 8, 12, 8)
        self._content.setVisible(False)
        layout.addWidget(self._content)

    @property
    def content_layout(self) -> QVBoxLayout:
        return self._content_layout

    def _toggle(self):
        if not self._available:
            return
        self._expanded = not self._expanded
        self._content.setVisible(self._expanded)
        arrow = '\u25bc' if self._expanded else '\u25b6'
        self._header.setText(f'{arrow}  {self._title}')

    def set_available(self, available: bool):
        """Mark panel as available/unavailable. Unavailable panels are red and non-expandable."""
        self._available = available
        if not available:
            self._expanded = False
            self._content.setVisible(False)
            self._header.setText(f'\u25b6  {self._title}')
            self._header.setStyleSheet(
                'QPushButton { text-align: left; padding: 8px 12px; '
                'font-weight: bold; border: none; border-radius: 0; '
                'background-color: #5c1a1a; color: #ff9999; }'
            )
        else:
            self._header.setStyleSheet(
                'QPushButton { text-align: left; padding: 8px 12px; '
                'font-weight: bold; border: none; border-radius: 0; }'
            )

    def set_status_text(self, text: str):
        """Show status text next to title (e.g. unavailability reason)."""
        if text:
            arrow = '\u25bc' if self._expanded else '\u25b6'
            self._header.setText(f'{arrow}  {self._title}  \u2014  {text}')
        else:
            arrow = '\u25bc' if self._expanded else '\u25b6'
            self._header.setText(f'{arrow}  {self._title}')
