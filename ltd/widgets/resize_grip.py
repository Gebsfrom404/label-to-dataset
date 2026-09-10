"""Drag handle that resizes a widget's height, like a textarea's resize corner."""
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QMouseEvent, QPainter, QPaintEvent, QPalette
from PySide6.QtWidgets import QWidget

from ltd.settings import get_settings


class HeightResizeGrip(QWidget):
    """Thin bar placed directly under ``target``; dragging it sets the target's height.

    The target gets a fixed height, clamped to ``minimum`` and — when a
    ``container`` is given — to the free space left in that container's layout
    at the start of the drag, so growing the target never pushes the widgets
    below it out of view. The height is persisted under ``settings_key``.
    """

    def __init__(self, target: QWidget, minimum: int = 60, default: int = 120,
                 settings_key: str | None = None,
                 container: QWidget | None = None, parent=None):
        super().__init__(parent)
        self._target = target
        self._minimum = minimum
        self._settings_key = settings_key
        self._container = container
        self._drag_start_y: float | None = None
        self._drag_start_height = 0
        self._drag_max = 0

        self.setFixedHeight(8)
        self.setCursor(Qt.CursorShape.SizeVerCursor)
        self.setToolTip('Drag to resize')

        height = default
        if settings_key:
            height = get_settings().value(settings_key, default, type=int)
        target.setFixedHeight(max(minimum, height))

    def _free_space(self) -> int:
        """Pixels the container's layout isn't using (0 without a container)."""
        container = self._container
        layout = container.layout() if container is not None else None
        if container is None or layout is None:
            return 0
        return max(0, container.height() - layout.sizeHint().height())

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        self._drag_start_y = event.globalPosition().y()
        self._drag_start_height = self._target.height()
        if self._container is None:
            self._drag_max = 16777215  # QWIDGETSIZE_MAX
        else:
            self._drag_max = self._drag_start_height + self._free_space()
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._drag_start_y is None:
            super().mouseMoveEvent(event)
            return
        delta = int(event.globalPosition().y() - self._drag_start_y)
        height = min(self._drag_start_height + delta, self._drag_max)
        self._target.setFixedHeight(max(self._minimum, height))
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._drag_start_y is None:
            super().mouseReleaseEvent(event)
            return
        self._drag_start_y = None
        if self._settings_key:
            get_settings().setValue(self._settings_key, self._target.height())
        event.accept()

    def paintEvent(self, event: QPaintEvent):
        # Three dots centred on the bar, in a muted text colour so the grip
        # reads in both the dark and light palettes.
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = self.palette().color(QPalette.ColorRole.WindowText)
        color.setAlpha(170 if self.underMouse() else 90)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        cx, cy = self.width() / 2, self.height() / 2
        for dx in (-8, 0, 8):
            painter.drawEllipse(QPointF(cx + dx, cy), 1.5, 1.5)

    def enterEvent(self, event):
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.update()
        super().leaveEvent(event)
