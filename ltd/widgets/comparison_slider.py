"""Before/after comparison slider widget with scrollbar."""
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QPainter, QPen, QPixmap, QWheelEvent
from PySide6.QtWidgets import (QGraphicsPixmapItem, QGraphicsScene,
                               QGraphicsView, QScrollBar,
                               QVBoxLayout, QWidget)


class _SliderView(QGraphicsView):
    """Internal view that handles zoom and draws the divider line."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene_ = QGraphicsScene(self)
        self.setScene(self.scene_)
        self.setRenderHints(QPainter.RenderHint.Antialiasing |
                            QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._before_pixmap: QPixmap | None = None
        self._after_pixmap: QPixmap | None = None
        self._before_item: QGraphicsPixmapItem | None = None
        self._after_item: QGraphicsPixmapItem | None = None
        self._slider_pos = 0.5
        self._image_width = 0
        self._image_height = 0

    @property
    def slider_pos(self):
        return self._slider_pos

    @slider_pos.setter
    def slider_pos(self, value):
        self._slider_pos = max(0.0, min(1.0, value))
        self._update_clip()
        self.viewport().update()

    def set_before(self, pixmap: QPixmap):
        self._before_pixmap = pixmap
        self._recalc_dimensions()
        self._rebuild()

    def set_after(self, pixmap: QPixmap):
        self._after_pixmap = pixmap
        self._recalc_dimensions()
        self._rebuild()

    def set_images(self, before: QPixmap, after: QPixmap):
        self._before_pixmap = before
        self._after_pixmap = after
        self._recalc_dimensions()
        self._rebuild()

    def _recalc_dimensions(self):
        """Set scene dimensions to the larger of before/after."""
        bw = self._before_pixmap.width() if self._before_pixmap else 0
        bh = self._before_pixmap.height() if self._before_pixmap else 0
        aw = self._after_pixmap.width() if self._after_pixmap else 0
        ah = self._after_pixmap.height() if self._after_pixmap else 0
        self._image_width = max(bw, aw)
        self._image_height = max(bh, ah)

    def _rebuild(self):
        self.scene_.clear()
        self._before_item = None
        self._after_item = None

        if self._after_pixmap:
            self._after_item = self.scene_.addPixmap(self._after_pixmap)
            ox = (self._image_width - self._after_pixmap.width()) / 2
            oy = (self._image_height - self._after_pixmap.height()) / 2
            self._after_item.setOffset(ox, oy)

        if self._before_pixmap:
            self._before_item = self.scene_.addPixmap(self._before_pixmap)
            self._before_item.setZValue(1)

        self._update_clip()
        if self._image_width > 0:
            self.setSceneRect(QRectF(0, 0, self._image_width, self._image_height))
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def _update_clip(self):
        if self._before_item and self._before_pixmap:
            clip_w = int(self._slider_pos * self._image_width)
            if clip_w <= 0:
                self._before_item.setPixmap(QPixmap())
                self._before_item.setOffset(0, 0)
            else:
                cropped = self._before_pixmap.copy(0, 0, clip_w, self._image_height)
                self._before_item.setPixmap(cropped)
                self._before_item.setOffset(0, 0)

    def drawForeground(self, painter: QPainter, rect: QRectF):
        super().drawForeground(painter, rect)
        if self._image_width > 0:
            x = self._slider_pos * self._image_width
            pen = QPen(Qt.GlobalColor.white, 2)
            painter.setPen(pen)
            painter.drawLine(QPointF(x, 0), QPointF(x, self._image_height))

    def wheelEvent(self, event: QWheelEvent):
        modifiers = event.modifiers()
        delta = event.angleDelta().y()
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            factor = 1.15 if delta > 0 else 1 / 1.15
            self.scale(factor, factor)
        else:
            bar = self.verticalScrollBar()
            bar.setValue(bar.value() - delta)


class ComparisonSlider(QWidget):
    """Horizontal curtain wipe with scrollbar control.

    Left = before (original), right = after (modified).
    Scrollbar at bottom controls slider position.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._view = _SliderView(self)
        layout.addWidget(self._view)

        self._scrollbar = QScrollBar(Qt.Orientation.Horizontal)
        self._scrollbar.setRange(0, 1000)
        self._scrollbar.setValue(500)
        self._scrollbar.setPageStep(100)
        layout.addWidget(self._scrollbar)

        self._scrollbar.valueChanged.connect(self._on_scrollbar_changed)

    def _on_scrollbar_changed(self, value: int):
        self._view.slider_pos = value / 1000.0

    # --- Proxy methods ---

    def viewport(self):
        return self._view.viewport()

    def set_before(self, pixmap: QPixmap):
        self._view.set_before(pixmap)

    def set_after(self, pixmap: QPixmap):
        self._view.set_after(pixmap)

    def set_images(self, before: QPixmap, after: QPixmap):
        self._view.set_images(before, after)
